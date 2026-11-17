"""Synthetic slot driver for the memory tests (same per-slot order as tpa/control.py:
encode -> write_short -> MEPR -> candidate / key -> consolidate -> mu update -> retrieve),
plus a smoke test. Inputs are periodic signals + noise from an explicit test Generator;
the per-UAV occupancy o_t of the zone-resolved candidate is a periodic signal in [0, 1]
from a SEPARATE generator (seed + 2000), so that the s_sys / tau_bar / zone-arrival draws
are the same as in the original implementation (tests/test_memory_d4.py compares
key_centering=false + candidate_source="pattern" against the stored reference).

zone_mean_mode = "ema" (default; the 0.99 EMA the reference arrays depend on) |
"cumulative" (cumulative mean of za since t = 0, the long-run zone mean of the simulator,
used by the age-matched phase-share check); min_age (default None) is passed to retrieve
so that info carries top_old_slot / top_old_age.
rng_a15 = default_rng(seed + 3000) is passed to the memory under A15 (key_shuffled) and
carried in get_state / set_state."""
from __future__ import annotations

import copy

import numpy as np
import torch

from tpa.config import load_config
from tpa.memory import (MEPR, MemoryFlags, MemoryParams, SegmentStore, TwoLevelMemory,
                        flags_for_arm, s_sys_dim, task_feat_dim)


def make_cfg(**ov):
    """Config with dotted overrides (memory__k_long=8 -> memory.k_long). Keys that are not
    in default.toml are inserted directly."""
    from tpa.config import Config
    ov = {k.replace("__", "."): v for k, v in ov.items()}
    base = load_config()
    known = {k: v for k, v in ov.items() if k.split(".")[-1] in base[k.split(".")[0]]}
    cfg = load_config(overrides=known) if known else base
    extra = {k: v for k, v in ov.items() if k not in known}
    if extra:
        data = cfg.to_dict()
        for k, v in extra.items():
            sec, leaf = k.split(".")
            data[sec][leaf] = v
        cfg = Config(data)
    return cfg


class Driver:
    def __init__(self, cfg, flags: MemoryFlags, seed: int = 0, period: int = 40, params=None,
                 zone_mean_mode: str = "ema", min_age: int | None = None):
        self.cfg, self.flags, self.period = cfg, flags, period
        if zone_mean_mode not in ("ema", "cumulative"):
            raise ValueError(f"unknown zone_mean_mode {zone_mean_mode!r}")
        self.zone_mean_mode, self.min_age = zone_mean_mode, min_age
        self.params = params if params is not None else MemoryParams(cfg, seed=seed)
        self.rng = np.random.default_rng(seed)
        self.rng_a14 = np.random.default_rng(seed + 1000)
        self.rng_uo = np.random.default_rng(seed + 2000)       # per-UAV occupancy noise
        self.rng_a15 = np.random.default_rng(seed + 3000)      # A15 key-shuffle draws
        self.M = cfg.scenario.n_uav
        self.mem = TwoLevelMemory(cfg, flags, self.params,
                                  rng_a14=self.rng_a14 if flags.displaced else None,
                                  rng_a15=self.rng_a15 if flags.key_shuffled else None)
        self.mepr = MEPR(cfg, flags)
        self.store = SegmentStore(cfg, flags)
        self.mem.attach_store(self.store)
        self.Z = cfg.scenario.n_zones
        self.sd, self.tf = s_sys_dim(cfg), task_feat_dim(cfg)
        self.zone_mean = np.ones(self.Z)
        self.za_sum = np.zeros(self.Z)                          # "cumulative" accumulator
        self.t = 0
        self.update_lru = True
        self.r, self.info, self.sigma, self.cons = {}, {}, {}, {}

    def inputs(self, t):
        ph = 2 * np.pi * t / self.period
        s_sys = 0.5 * np.sin(ph + np.arange(self.sd)) + 0.1 * self.rng.normal(size=self.sd)
        tb = 0.5 * np.cos(ph + 0.3 * np.arange(2 * self.tf)) + 0.1 * self.rng.normal(size=2 * self.tf)
        lam = 3.0 * (1.0 + 0.8 * np.sin(ph + 2 * np.pi * np.arange(self.Z) / self.Z))
        za = self.rng.poisson(lam).astype(float)
        return s_sys, tb, za

    def occupancy(self, t):
        """Per-UAV occupancy o_t (M,) in [0, 1]: periodic + small noise from the separate generator."""
        ph = 2 * np.pi * t / self.period
        uo = 0.5 + 0.4 * np.sin(ph + 2 * np.pi * np.arange(self.M) / self.M) \
            + 0.05 * self.rng_uo.normal(size=self.M)
        return np.clip(uo, 0.0, 1.0)

    def step(self):
        t = self.t
        s_sys, tb, za = self.inputs(t)
        uo = self.occupancy(t)
        qin = np.concatenate([s_sys, tb])
        x = np.concatenate([qin, self.mem.mbar_short()])
        if self.zone_mean_mode == "cumulative":                 # includes the current slot (simulator)
            self.za_sum = self.za_sum + za
            self.zone_mean = self.za_sum / (t + 1)
        else:
            self.zone_mean = 0.99 * self.zone_mean + 0.01 * za
        self.store.record(t, x, qin, za, self.zone_mean, uav_occ=uo)
        label = int((t % self.period) * 24 // self.period)
        with torch.no_grad():
            e = self.params.encode(x)
        self.mem.write_short(e, label, t)
        P, sig, cons = self.mepr.step_buffered(e, za, self.zone_mean, uav_occ=uo)
        # ordering: z_t -> key (mu_{t-1}, inside consolidate) -> write -> mu update
        z = self.mem.candidate_for(self.mepr)
        if cons:
            self.mem.consolidate(z, label, t, p_ref=self.mepr.P_ref_used,
                                 v_ref=self.mepr.v_ref_used)
        if z is not None:
            self.mem.observe_candidate(z)
        r, info = self.mem.retrieve(qin, t, update_lru=self.update_lru, min_age=self.min_age)
        self.r[t], self.info[t], self.sigma[t], self.cons[t] = r, info, sig, cons
        self.t += 1
        return r, info

    def run(self, T):
        for _ in range(T):
            self.step()
        return self

    def get_state(self):
        return dict(mem=self.mem.get_state(), mepr=self.mepr.get_state(),
                    store=self.store.get_state(), rng=copy.deepcopy(self.rng.bit_generator.state),
                    rng_a14=copy.deepcopy(self.rng_a14.bit_generator.state),
                    rng_uo=copy.deepcopy(self.rng_uo.bit_generator.state),
                    rng_a15=copy.deepcopy(self.rng_a15.bit_generator.state),
                    zone_mean=self.zone_mean.copy(), za_sum=self.za_sum.copy(), t=self.t)

    def set_state(self, s):
        self.mem.set_state(s["mem"])
        self.mepr.set_state(s["mepr"])
        self.store.set_state(s["store"])
        self.rng.bit_generator.state = s["rng"]
        self.rng_a14.bit_generator.state = s["rng_a14"]
        self.rng_uo.bit_generator.state = s["rng_uo"]
        if "rng_a15" in s:
            self.rng_a15.bit_generator.state = s["rng_a15"]
        self.zone_mean = s["zone_mean"].copy()
        self.za_sum = s["za_sum"].copy()
        self.t = s["t"]


def test_driver_smoke():
    cfg = make_cfg()
    d = Driver(cfg, flags_for_arm("A0")).run(150)
    assert d.store.n == 150
    assert all(d.sigma[t] == 0.0 for t in range(cfg.memory.window_w))
    assert np.isfinite(d.r[149]).all()
