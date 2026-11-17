"""Actor / twin critics (Eqs. 48-50)."""
import numpy as np
import torch

from tpa.config import load_config
from tpa.learner.networks import (Actor, TwinCritic, action_dim, param_count, s_sys_dim,
                                  state_dim, task_feat_dim)

CFG = load_config()
Q = CFG.task.q_max
A = action_dim(CFG)


def batch(B=4, n_valid=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    S = torch.randn(B, s_sys_dim(CFG), generator=g)
    tf = torch.randn(B, Q, task_feat_dim(CFG), generator=g)
    tf[..., -1] = 0.0
    tf[:, :n_valid, -1] = 1.0
    tf[:, n_valid:, :] = 0.0
    r = torch.randn(B, CFG.memory.d, generator=g)
    sg = torch.rand(B, 1, generator=g)
    return S, tf, r, sg


def test_dims():
    assert s_sys_dim(CFG) == 49 and task_feat_dim(CFG) == 17
    assert state_dim(CFG, True) == 658 and state_dim(CFG, False) == 593
    assert A == 10


def test_shapes_and_range():
    S, tf, r, sg = batch()
    for mem in (True, False):
        act = Actor(CFG, mem)
        rr, ss = (r, sg) if mem else (None, None)
        a, logp = act.sample(S, tf, rr, ss, generator=torch.Generator().manual_seed(1))
        assert a.shape == (4, Q, A) and logp.shape == (4,)
        assert torch.all(a.abs() <= 1.0)
        assert act.deterministic(S, tf, rr, ss).shape == (4, Q, A)
        q1, q2 = TwinCritic(CFG, mem)(S, tf, a, rr, ss)
        assert q1.shape == (4,) and q2.shape == (4,)
        assert not torch.allclose(q1, q2)                     # independent critics


def test_masked_logp_ignores_empty_slots():
    torch.manual_seed(0)
    act = Actor(CFG, True)
    S, tf, r, sg = batch(n_valid=3)
    noise = torch.randn(4, Q, A, generator=torch.Generator().manual_seed(2))
    _, lp1 = act.sample(S, tf, r, sg, noise=noise)
    noise2 = noise.clone()
    noise2[:, 3:, :] = torch.randn(4, Q - 3, A, generator=torch.Generator().manual_seed(3)) * 5
    _, lp2 = act.sample(S, tf, r, sg, noise=noise2)
    assert torch.allclose(lp1, lp2)
    # equals the per-row sum over the valid rows only
    mean, log_std = act(S, tf, r, sg)
    u = mean + log_std.exp() * noise
    dist = torch.distributions.Normal(mean, log_std.exp())
    lp_rows = (dist.log_prob(u) - torch.log(1 - torch.tanh(u) ** 2)).sum(-1)
    assert torch.allclose(lp1, lp_rows[:, :3].sum(-1), atol=1e-3)
    # empty window -> logp 0
    tf0 = tf.clone()
    tf0[..., -1] = 0
    _, lp0 = act.sample(S, tf0, r, sg, noise=noise)
    assert torch.all(lp0 == 0)


def test_critic_sees_task_features_and_masks_actions():
    torch.manual_seed(0)
    c = TwinCritic(CFG, True)
    S, tf, r, sg = batch(n_valid=3)
    a = torch.rand(4, Q, A) * 2 - 1
    q, _ = c(S, tf, a, r, sg)
    tf2 = tf.clone()
    tf2[:, 0, 0] += 1.0                                     # change a task feature
    assert not torch.allclose(q, c(S, tf2, a, r, sg)[0])
    a2 = a.clone()
    a2[:, 3:, :] = 0.7                                       # change actions of empty slots
    assert torch.allclose(q, c(S, tf, a2, r, sg)[0])


def test_param_count_helper():
    pa, pc = param_count(Actor(CFG, True)), TwinCritic(CFG, True).param_count()
    pa0, pc0 = param_count(Actor(CFG, False)), TwinCritic(CFG, False).param_count()
    print(f"\nparams with memory: actor={pa} twin critics={pc} total={pa + pc}")
    print(f"params without memory: actor={pa0} twin critics={pc0} total={pa0 + pc0}")
    assert pa > pa0 and pc > pc0
    lin = torch.nn.Linear(3, 2)
    assert param_count(lin) == 8
