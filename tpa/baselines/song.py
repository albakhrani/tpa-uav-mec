"""B_SONG: static-hover restriction of Song et al., "Evolutionary Multi-Objective
Reinforcement Learning Based Trajectory Control and Task Offloading in UAV-Assisted Mobile
Edge Computing", IEEE TMC 22(12), 2023.

Published method (EMORL): a multi-policy evolutionary MORL that evolves a population of
policies over preference weights (delay, UAV energy, collected tasks), producing a set of
non-dominated policies in one run; actions cover UAV trajectory and offloading.
Restriction used here:
  * trajectory removed (fixed hover); one preference (the shared scalarised reward with
    fixed weights w_base/4) instead of a Pareto set, so the multi-policy population is
    reduced to a single search distribution;
  * gradient-free antithetic evolution strategies (Salimans et al. 2017 form) over a
    LINEAR policy a = tanh(W x + b), x = [S_sys; mean-pooled tf; max-pooled tf]
    (49 + 17 + 17 = 83), a reshaped to (q_max, 2M); W, b start at 0 (neutral action:
    a_ij = 0, f~ = 0.5);
  * fitness of a perturbation = sum of per-slot rewards over ONE training episode
    (T_ep slots); centered-rank shaping over the 2 x pairs returns; plain gradient step
    theta += lr_es / (2 pairs sigma_es) sum_i F_i eps_i;
  * actions are mapped by the shared executor exactly as for B_HUANG.

Budget mapping (equal environment slots): the SAC arms train 1000 episodes of
T_ep slots. Each ES iteration evaluates 2 x pairs perturbed policies for one episode each,
so the 1000-episode budget gives  n_iter = floor(1000 / (2 x pairs))  full iterations
(config pairs = 6 -> 83 iterations = 996 episodes); iteration n uses episodes
[2 pairs n, 2 pairs (n+1)). When the episode budget is not divisible by
2 x pairs, the trailing partial iteration (1000 - 996 = 4 episodes at pairs = 6) still
runs with its perturbations and every episode's return is logged (info["es_episode"],
partial=True), but NO update is made from it, so theta after the budget equals theta
after the last full iteration. The environment-slot budget is therefore identical to the
SAC arms (1000 x T_ep slots), of which 2 pairs n_iter x T_ep drive updates.
The two members of an antithetic pair replay the same trace window
(trainer, `paired_episodes`).
Episode e inside an iteration evaluates theta + s sigma eps_p with p = e // 2 and
s = +1 for even e, -1 for odd e. The iteration's noise eps (pairs x dim) is drawn from
streams.policy at its first episode. Evaluation (train=False / eval_clone) uses the
mean theta.
"""
from __future__ import annotations

import copy

import numpy as np

from ..control import task_features
from ..learner.networks import s_sys_dim, task_feat_dim
from .executor import BaseController, Executor, argmax_pref_choose, ftilde_alloc, opt


def pooled_input(S_sys, tf) -> np.ndarray:
    """x = [S_sys; mean(tf); max(tf)] over the q_max window rows (zeros for empty rows,
    the valid-mask column makes the pooled mask = fill fraction)."""
    return np.concatenate([np.asarray(S_sys, np.float64), tf.mean(axis=0), tf.max(axis=0)])


class SongController(BaseController):
    fixed_weights = True
    LEARNED_STATE_KEYS = ("theta", "norm")    # eval_clone takes a copy of self.theta (ES)

    def __init__(self, cfg, arm_key, streams, r_ref, h_min, device="cpu", timer=None, _shared=None):
        super().__init__(cfg, arm_key, streams, r_ref, h_min, device, timer)
        self.ex = Executor(cfg, arm_key, h_min)
        M, q = int(cfg.scenario.n_uav), int(cfg.task.q_max)
        self.in_dim = s_sys_dim(cfg) + 2 * task_feat_dim(cfg)
        self.last_update = None
        self.out_shape = (q, 2 * M)
        self.out_dim = q * 2 * M
        self.n_par = self.out_dim * self.in_dim + self.out_dim
        self.pairs = int(opt(cfg, "baselines.es_pairs", 5))
        self.sigma_es = float(opt(cfg, "baselines.es_sigma", 0.02))
        self.lr_es = float(opt(cfg, "baselines.es_lr", 0.01))
        self.rng = streams.policy
        self.theta = np.zeros(self.n_par) if _shared is None else _shared.copy()
        self.frozen = _shared is not None
        self.eps = None              # (pairs, n_par) for the current iteration
        self.ep_in_iter = -1         # episode index within the iteration (-1: none started)
        self.returns = np.zeros(2 * self.pairs)
        self.ep_return = 0.0
        self.n_iter = 0
        self.episodes_done = 0
        self.ep_active = False
        # The two members of an antithetic pair (episodes 2p, 2p+1) must see
        # the same trace and load; the trainer replays the even episode's environment for
        # the odd one when this flag is set (common random numbers within a pair).
        self.paired_episodes = not self.frozen

    # ------------------------------------------------------------ ES bookkeeping
    @property
    def n_iterations_budget(self) -> int:
        """Full iterations within the SAC arms' episode budget: floor(episodes / (2 pairs))."""
        return int(self.cfg.run.episodes) // (2 * self.pairs)

    @property
    def n_partial_episodes(self) -> int:
        """Trailing episodes of the budget that form an incomplete iteration."""
        return int(self.cfg.run.episodes) % (2 * self.pairs)

    def _close_episode(self) -> dict | None:
        """Close the running episode. Returns a log dict; it contains the ES update info
        when this episode completed a full iteration within the budget."""
        if not self.ep_active:
            return None
        self.returns[self.ep_in_iter] = self.ep_return
        self.ep_active = False
        partial = self.n_iter >= self.n_iterations_budget
        log = dict(es_episode=dict(iteration=self.n_iter, member=self.ep_in_iter,
                                   ret=float(self.ep_return), partial=bool(partial)))
        self.episodes_done += 1
        if self.ep_in_iter == 2 * self.pairs - 1:
            if partial:                         # beyond the budget: never update
                self.eps = None
                self.ep_in_iter = -1
            else:
                log.update(self._update())
        return log

    def _update(self) -> dict:
        r = self.returns
        ranks = np.empty(len(r))
        ranks[np.argsort(r, kind="stable")] = np.arange(len(r))
        F = ranks / (len(r) - 1) - 0.5                        # centered ranks in [-0.5, 0.5]
        g = np.zeros(self.n_par)
        for p in range(self.pairs):
            g += (F[2 * p] - F[2 * p + 1]) * self.eps[p]
        g /= (2 * self.pairs * self.sigma_es)
        self.theta = self.theta + self.lr_es * g
        self.n_iter += 1
        self.eps = None
        self.ep_in_iter = -1
        # The trainer's episode row reads the SAC keys of every update
        # (trainer._ep_row); ES has no critic, actor loss or temperature: reported as NaN
        return dict(es_iter=self.n_iter, ret_mean=float(r.mean()), ret_max=float(r.max()),
                    alpha=float("nan"), critic_loss=float("nan"), actor_loss=float("nan"))

    def _open_episode(self) -> None:
        if self.frozen:
            return
        if self.eps is None:
            self.eps = self.rng.standard_normal((self.pairs, self.n_par))
            self.ep_in_iter = -1
        self.ep_in_iter += 1
        self.ep_return = 0.0
        self.ep_active = True

    def _params(self, train: bool) -> np.ndarray:
        if not train or self.frozen or not self.ep_active:
            return self.theta
        p, s = divmod(self.ep_in_iter, 2)
        return self.theta + (1.0 if s == 0 else -1.0) * self.sigma_es * self.eps[p]

    # ------------------------------------------------------------ protocol
    def reset_segment(self, t0: int) -> None:
        self.seg_id += 1

    def start_episode(self) -> None:
        self.last_update = self._close_episode()
        self._open_episode()

    def policy(self, x, theta) -> np.ndarray:
        k = self.out_dim * self.in_dim
        W = theta[:k].reshape(self.out_dim, self.in_dim)
        return np.tanh(W @ x + theta[k:]).reshape(self.out_shape)

    def act(self, obs, label, train):
        if train and not self.frozen and not self.ep_active:
            self._open_episode()                              # first episode without start_episode()
        with self.timer("drm"):
            win = self.ex.window(obs)
            tf = task_features(obs, win, self.cfg, self.r_ref)
        with self.timer("actor"):
            a = self.policy(pooled_input(obs.S_sys, tf), self._params(train))
        with self.timer("drm"):
            dec = self.ex.run(obs, win, argmax_pref_choose(a, self.ex.M),
                              ftilde_alloc(a, self.ex.M, np.asarray(obs.cap_bound, np.float64)))
        self.last = dict(n_window=len(win))
        return dec, 0.0

    def observe(self, obs, dec, outcome, train):
        J, rew = self._reward(dec, outcome, train)
        if train:
            self.slots_seen += 1
            if self.ep_active:
                self.ep_return += rew
        info = dict(J=J, reward=rew)
        if self.last_update is not None:                  # episode closed at start_episode()
            log, self.last_update = self.last_update, None
            info["es_episode"] = log.pop("es_episode")
            if "es_iter" in log:                          # an ES step was taken
                info["update"] = log
        return info

    def eval_clone(self, streams):
        c = SongController(self.cfg, self.arm_key, streams, self.r_ref, self.h_min,
                           self.device, None, _shared=self.theta)
        c.norm.set_state(self.norm.get_state())
        return c

    def get_state(self) -> dict:
        st = self._base_state()
        st.update(theta=self.theta.copy(), eps=None if self.eps is None else self.eps.copy(),
                  ep_in_iter=self.ep_in_iter, returns=self.returns.copy(),
                  ep_return=self.ep_return, n_iter=self.n_iter, ep_active=self.ep_active,
                  episodes_done=self.episodes_done, rng=copy.deepcopy(self.rng.bit_generator.state))
        return st

    def set_state(self, st: dict) -> None:
        self._set_base_state(st)
        self.theta = np.asarray(st["theta"], np.float64).copy()
        self.eps = None if st["eps"] is None else np.asarray(st["eps"], np.float64).copy()
        self.ep_in_iter = int(st["ep_in_iter"])
        self.returns = np.asarray(st["returns"], np.float64).copy()
        self.ep_return = float(st["ep_return"])
        self.n_iter = int(st["n_iter"])
        self.ep_active = bool(st["ep_active"])
        self.episodes_done = int(st.get("episodes_done", 0))
        self.rng.bit_generator.state = copy.deepcopy(st["rng"])
