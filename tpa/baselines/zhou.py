"""B_ZHOU: static-hover restriction of Zhou et al., "Computation Rate Maximization in
UAV-Enabled Wireless-Powered Mobile-Edge Computing Systems", IEEE JSAC 36(9), 2018.

Published method: maximise the users' computation rate under partial/binary offloading,
energy-harvesting causality and UAV speed constraints, by alternating optimisation of
CPU frequencies, offloading time / transmit power (closed forms) and the UAV trajectory.
Restriction used here: the trajectory is fixed at the hover deployment, wireless
power transfer is disabled (no harvesting, as in the harvesting-disabled evaluation),
binary full offloading to one UAV is the only mode the simulator offers, and the per-slot
optimisation reduces to:
  * assignment: rate-greedy -- each task (FIFO window) to the feasible UAV with the largest
    predicted rate R_ij (computation-rate maximisation with the server CPU shared);
  * allocation: equal split of UAV j's slot-start residual capacity among the tasks
    assigned to j this slot (two-pass), then f = clip(max(f_eq, f_req), 0, F_res).
No learning; deterministic given the trace.
"""
from __future__ import annotations

from ..types import Decision
from .executor import BaseController, Executor


class _RuleController(BaseController):
    """Non-learning baseline: executor + a fixed (choose, alloc) rule."""

    fixed_weights = True             # reward is logged only; fixed w_base/4 (no Eq. 59)

    def __init__(self, cfg, arm_key, streams, r_ref, h_min, device="cpu", timer=None):
        super().__init__(cfg, arm_key, streams, r_ref, h_min, device, timer)
        self.ex = Executor(cfg, arm_key, h_min)

    # rules: override
    def choose(self, obs, i, k, cands):
        raise NotImplementedError

    def alloc(self, obs, j, items, F0):
        raise NotImplementedError

    def reset_segment(self, t0: int) -> None:
        self.seg_id += 1

    def start_episode(self) -> None:
        pass

    def act(self, obs, label: int, train: bool):
        with self.timer("drm"):
            win = self.ex.window(obs)
            dec: Decision = self.ex.run(obs, win,
                                        lambda i, k, c: self.choose(obs, i, k, c),
                                        lambda j, it, F0: self.alloc(obs, j, it, F0))
        self.last = dict(n_window=len(win))
        return dec, 0.0

    def observe(self, obs, dec, outcome, train: bool):
        J, rew = self._reward(dec, outcome, train)
        if train:
            self.slots_seen += 1
        return dict(J=J, reward=rew)

    def eval_clone(self, streams):
        c = type(self)(self.cfg, self.arm_key, streams, self.r_ref, self.h_min, self.device)
        c.norm.set_state(self.norm.get_state())
        return c

    def get_state(self) -> dict:
        return self._base_state()

    def set_state(self, st: dict) -> None:
        self._set_base_state(st)


class ZhouController(_RuleController):
    def choose(self, obs, i, k, cands):
        """Rate-greedy: argmax predicted R_ij over feasible pairs (ties: lowest j)."""
        return max(cands, key=lambda c: (c[3], -c[0]))[0]

    def alloc(self, obs, j, items, F0):
        """Equal split of the slot-start residual of UAV j among its tasks this slot."""
        n = len(items)
        return [F0 / n] * n
