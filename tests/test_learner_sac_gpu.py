"""CUDA learner with a CPU rollout mirror of the actor; device-independent state."""
import numpy as np
import pytest
import torch

from test_learner_sac import CFG, D, FakeMem, mk_batch, mk_sac
from tpa.learner.networks import Actor, TwinCritic
from tpa.learner.sac import SAC

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")


def to_dev(b, dev):
    return {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}


def mk_cuda_from(sac_cpu, fm_cpu, seed=0):
    fm = FakeMem().to("cuda")
    fm.load_state_dict(fm_cpu.state_dict())
    s = SAC(CFG, Actor(CFG, True), TwinCritic(CFG, True), fm, "cuda",
            rng=np.random.Generator(np.random.PCG64(seed)))
    s.set_state(sac_cpu.get_state())
    return s, fm


def test_cpu_mirror_is_actor_on_cpu():
    sac, _ = mk_sac(mem=True)
    assert sac.actor_cpu is sac.actor


def test_get_state_is_cpu_only():
    sac, fm = mk_sac(mem=True)
    sac.update(mk_batch(), fm.recon)
    st = sac.get_state()

    def walk(o):
        if isinstance(o, torch.Tensor):
            assert o.device.type == "cpu"
        elif isinstance(o, dict):
            [walk(v) for v in o.values()]
        elif isinstance(o, (list, tuple)):
            [walk(v) for v in o]
    walk(st)


@cuda
def test_cuda_update_matches_cpu_and_act_on_cpu():
    sac_c, fm_c = mk_sac(mem=True, seed=3)
    sac_g, fm_g = mk_cuda_from(sac_c, fm_c, seed=3)
    assert next(sac_g.actor.parameters()).device.type == "cuda"
    assert next(sac_g.critics_targ.parameters()).device.type == "cuda"
    assert sac_g.log_alpha.device.type == "cuda"
    assert next(sac_g.actor_cpu.parameters()).device.type == "cpu"
    b = mk_batch(B=64)
    sac_c.update(b, fm_c.recon)
    sac_g.update(to_dev(b, "cuda"), fm_g.recon)
    for mc, mg in ((sac_c.actor, sac_g.actor), (sac_c.critics, sac_g.critics),
                   (sac_c.critics_targ, sac_g.critics_targ), (fm_c, fm_g)):
        for p, q in zip(mc.parameters(), mg.parameters()):
            assert torch.allclose(p, q.cpu(), atol=1e-4)
    assert sac_c.alpha == pytest.approx(sac_g.alpha, abs=1e-6)
    # mirror synced after the update
    for p, q in zip(sac_g.actor.parameters(), sac_g.actor_cpu.parameters()):
        assert torch.equal(p.cpu(), q)
    b1 = mk_batch(B=1)
    a = sac_g.act(b1["S_sys"][0].numpy(), b1["tf"][0].numpy(), np.zeros(D), 0.1)
    assert a.shape == (CFG.task.q_max, 10)
    ad_g = sac_g.act(b1["S_sys"][0].numpy(), b1["tf"][0].numpy(), np.zeros(D), 0.1, deterministic=True)
    ad_c = sac_c.act(b1["S_sys"][0].numpy(), b1["tf"][0].numpy(), np.zeros(D), 0.1, deterministic=True)
    assert np.allclose(ad_g, ad_c, atol=1e-4)


@cuda
def test_state_roundtrip_across_devices():
    sac_c, fm_c = mk_sac(mem=True, seed=4)
    sac_g, fm_g = mk_cuda_from(sac_c, fm_c, seed=4)
    b = mk_batch()
    sac_g.update(to_dev(b, "cuda"), fm_g.recon)
    st = sac_g.get_state()                                  # cuda -> cpu checkpoint
    sac_c2, fm_c2 = mk_sac(mem=True, seed=9)
    sac_c2.set_state(st)
    for p, q in zip(sac_g.actor.parameters(), sac_c2.actor.parameters()):
        assert torch.equal(p.cpu(), q)
    for p, q in zip(fm_g.parameters(), fm_c2.parameters()):
        assert torch.equal(p.cpu(), q)
    assert sac_c2.alpha == sac_g.alpha
    sac_g2, fm_g2 = mk_cuda_from(sac_c2, fm_c2, seed=9)    # cpu -> cuda
    for p, q in zip(sac_g.critics_targ.parameters(), sac_g2.critics_targ.parameters()):
        assert torch.equal(p, q)
    # optimiser state lands on the parameter device and the next updates agree
    sac_g.update(to_dev(b, "cuda"), fm_g.recon)
    sac_g2.rng.bit_generator.state = st["rng"]
    sac_g2.update(to_dev(b, "cuda"), fm_g2.recon)
    for p, q in zip(sac_g.actor.parameters(), sac_g2.actor.parameters()):
        assert torch.allclose(p, q, atol=1e-6)
