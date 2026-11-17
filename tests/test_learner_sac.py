"""SAC (Eq. 51, Alg. 2): true s' in the TD target, gradients into a
fake memory module from both losses, auto-alpha, polyak, state round-trip."""
import copy

import numpy as np
import pytest
import torch

from tpa.config import load_config
from tpa.learner.networks import Actor, TwinCritic, action_dim, s_sys_dim, task_feat_dim
from tpa.learner.sac import SAC

CFG = load_config()
Q = CFG.task.q_max
A = action_dim(CFG)
D = CFG.memory.d


class FakeMem(torch.nn.Module):
    """Stand-in for MemoryParams: r(t) = tanh(W feat[t])."""

    def __init__(self, T=200, k=8):
        super().__init__()
        self.lin = torch.nn.Linear(k, D)
        g = torch.Generator().manual_seed(7)
        self.register_buffer("feat", torch.randn(T, k, generator=g))

    def recon(self, refs, grad):
        idx = torch.tensor([max(t, 0) for _, t in refs], device=self.feat.device)
        with torch.set_grad_enabled(grad):
            return torch.tanh(self.lin(self.feat[idx]))


def mk_batch(B=8, n_valid=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    tf = torch.randn(B, Q, task_feat_dim(CFG), generator=g)
    tf[..., -1] = 0
    tf[:, :n_valid, -1] = 1
    tf2 = torch.randn(B, Q, task_feat_dim(CFG), generator=g)
    tf2[..., -1] = 0
    tf2[:, :n_valid, -1] = 1
    ts = torch.randint(1, 150, (B,), generator=g).tolist()
    return {"S_sys": torch.randn(B, s_sys_dim(CFG), generator=g), "tf": tf,
            "a": torch.rand(B, Q, A, generator=g) * 2 - 1,
            "reward": torch.randn(B, generator=g),
            "S_sys2": torch.randn(B, s_sys_dim(CFG), generator=g), "tf2": tf2,
            "sigma": torch.rand(B, 1, generator=g), "sigma2": torch.rand(B, 1, generator=g),
            "refs": [(0, t) for t in ts], "refs2": [(0, t + 1) for t in ts]}


def mk_sac(mem=True, cfg=CFG, seed=0):
    torch.manual_seed(seed)
    fm = FakeMem() if mem else None
    sac = SAC(cfg, Actor(cfg, mem), TwinCritic(cfg, mem), fm, "cpu",
              rng=np.random.Generator(np.random.PCG64(seed)))
    return sac, fm


def test_td_target_uses_true_next_state():
    sac, fm = mk_sac(mem=False)
    b = mk_batch()
    st = sac.rng.bit_generator.state
    y = sac.td_target(b, None)
    # manual: same noise, target critics on s'
    sac.rng.bit_generator.state = st
    noise = torch.as_tensor(sac.rng.standard_normal((8, Q, A)), dtype=torch.float32)
    with torch.no_grad():
        a2, lp2 = sac.actor.sample(b["S_sys2"], b["tf2"], noise=noise)
        q1, q2 = sac.critics_targ(b["S_sys2"], b["tf2"], a2)
        manual = b["reward"] + CFG.learner.gamma * (torch.min(q1, q2) - sac.alpha * lp2)
    assert torch.allclose(y, manual, atol=1e-6)
    # changing s (not s') leaves the target unchanged; changing s' changes it
    b_s = dict(b, S_sys=b["S_sys"] + 3.0)
    sac.rng.bit_generator.state = st
    assert torch.allclose(sac.td_target(b_s, None), y)
    b_s2 = dict(b, S_sys2=b["S_sys2"] + 3.0)
    sac.rng.bit_generator.state = st
    assert not torch.allclose(sac.td_target(b_s2, None), y)
    # using s as the next state (s2 = s) would give a different target
    b_bug = dict(b, S_sys2=b["S_sys"], tf2=b["tf"])
    sac.rng.bit_generator.state = st
    assert not torch.allclose(sac.td_target(b_bug, None), y)


def test_td_target_uses_r_next_from_refs2():
    sac, fm = mk_sac(mem=True)
    b = mk_batch()
    calls = []

    def recon(refs, grad):
        calls.append((list(refs), grad))
        return fm.recon(refs, grad)

    sac.losses(b, recon)
    kinds = {(tuple(r), g) for r, g in calls}
    assert (tuple(b["refs"]), True) in kinds
    assert (tuple(b["refs2"]), False) in kinds
    assert (tuple((s, t - 1) for s, t in b["refs"]), False) in kinds


def test_memory_gradient_from_both_losses():
    cfg0 = CFG.with_overrides({"learner.lambda_mem": 0.0})
    sac, fm = mk_sac(mem=True, cfg=cfg0)
    b = mk_batch()
    cl, al, _, _ = sac.losses(b, fm.recon)
    g_c = torch.autograd.grad(cl, fm.lin.weight, retain_graph=True)[0]
    g_a = torch.autograd.grad(al, fm.lin.weight, retain_graph=True)[0]
    assert g_c.abs().sum() > 0, "critic loss must reach memory params"
    assert g_a.abs().sum() > 0, "actor loss (lambda_mem = 0) must reach memory params"
    # actor loss must not reach critic parameters; critic loss must not reach the actor
    crit_p = list(sac.critics.parameters())
    ga = torch.autograd.grad(al, crit_p, retain_graph=True, allow_unused=True)
    assert all(g is None for g in ga)
    gc = torch.autograd.grad(cl, list(sac.actor.parameters()), retain_graph=True, allow_unused=True)
    assert all(g is None for g in gc)
    # L_mem alone reaches memory params
    sac1, fm1 = mk_sac(mem=True)
    r_t = fm1.recon(b["refs"], True)
    r_p = fm1.recon([(s, t - 1) for s, t in b["refs"]], False)
    lm, _ = sac1.l_mem(r_t, r_p)
    assert torch.autograd.grad(lm, fm1.lin.weight)[0].abs().sum() > 0


def test_l_mem_formula():
    sac, _ = mk_sac(mem=True, cfg=CFG.with_overrides({"memory.delta_max": 1.0}))
    r_t = torch.tensor([[3.0, 4.0] + [0.0] * (D - 2)])
    lm, viol = sac.l_mem(r_t, torch.zeros_like(r_t))
    assert float(lm) == pytest.approx(25.0 + (5.0 - 1.0) ** 2)
    assert float(viol) == 1.0


def test_update_steps_everything_once():
    sac, fm = mk_sac(mem=True)
    b = mk_batch()
    w_mem = fm.lin.weight.detach().clone()
    w_targ = [p.detach().clone() for p in sac.critics_targ.parameters()]
    w_crit = [p.detach().clone() for p in sac.critics.parameters()]
    la = sac.alpha
    info = sac.update(b, fm.recon)
    assert not torch.equal(w_mem, fm.lin.weight)
    assert sac.opt_mem.state[fm.lin.weight]["step"] == 1
    assert sac.alpha != la
    tau = CFG.learner.tau
    for t0, c0, t1, c1 in zip(w_targ, w_crit, sac.critics_targ.parameters(), sac.critics.parameters()):
        assert torch.allclose(t1, (1 - tau) * t0 + tau * c1, atol=1e-7)
    assert np.isfinite(info["critic_loss"]) and np.isfinite(info["actor_loss"])


def test_target_entropy():
    sac, _ = mk_sac(mem=False)
    b = mk_batch(n_valid=3)
    assert torch.allclose(sac.target_entropy(b["tf"]), torch.full((8,), -0.5 * 3 * A))


def test_memoryless_update_runs():
    sac, _ = mk_sac(mem=False)
    info = sac.update(mk_batch(), None)
    assert "l_mem" not in info and np.isfinite(info["critic_loss"])


def test_state_roundtrip():
    sac, fm = mk_sac(mem=True, seed=1)
    b = mk_batch()
    sac.update(b, fm.recon)
    st = sac.get_state()
    sac2, fm2 = mk_sac(mem=True, seed=5)
    sac2.set_state(st)
    sac.update(b, fm.recon)
    sac2.update(b, fm2.recon)
    for p, q in zip(sac.actor.parameters(), sac2.actor.parameters()):
        assert torch.equal(p, q)
    for p, q in zip(sac.critics_targ.parameters(), sac2.critics_targ.parameters()):
        assert torch.equal(p, q)
    assert torch.equal(fm.lin.weight, fm2.lin.weight)
    assert sac.alpha == sac2.alpha


def test_act_and_random_action():
    sac, _ = mk_sac(mem=True)
    b = mk_batch(B=1)
    a = sac.act(b["S_sys"][0].numpy(), b["tf"][0].numpy(), np.zeros(D), 0.1)
    assert a.shape == (Q, A) and np.all(np.abs(a) <= 1)
    ad = sac.act(b["S_sys"][0].numpy(), b["tf"][0].numpy(), np.zeros(D), 0.1, deterministic=True)
    assert ad.shape == (Q, A)
    ra = sac.random_action()
    assert ra.shape == (Q, A) and np.all(np.abs(ra) <= 1)
