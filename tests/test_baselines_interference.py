"""Executor incremental interference (interference_set = "slot_assigned"); slot admission versus the
prefix-only check, in refuse and raise modes (constructed and real-simulator slots); the serial-radio
wait (same-device uploads, obs.device_busy_s); Executor.admit agrees with tpa.drm.slot_admissible."""
from __future__ import annotations

import numpy as np
import pytest

from tpa.baselines import QinController, ZhouController
from tpa.config import load_config
from tpa.rng import Streams
from tpa.types import Decision, SlotObs, Task

from test_baselines_controllers import make, run, small_cfg

CFG = load_config()
M, Z = CFG.scenario.n_uav, CFG.scenario.n_zones
BASE_R = np.array([1e8, 5e7, 1e7, 1e7, 1e7])


def mk_task(tid):
    return Task(tid=tid, device=tid, zone=0, t_arr=0, size_bits=8e6, cycles=1e8, deadline_s=5.0,
                priority=0.5, complexity=1.0, loss_u=np.zeros(CFG.task.max_tx_attempts), truth_bin=-1)


def mk_obs(n=2, rate_fn=None, h_min=None):
    cap = np.full(M, 4e9)
    return SlotObs(t=0, S_sys=np.zeros(49), queue=[mk_task(k) for k in range(n)],
                   rates_pred=np.tile(BASE_R, (n, 1)), h_now=np.ones((n, M)),
                   zone_arrivals=np.zeros(Z), zone_mean=np.ones(Z), residual_f=cap.copy(),
                   cap_bound=cap, energy=np.full(M, 1e6), e_min=np.zeros(M), burst=np.zeros(M),
                   lam_eff=np.zeros(M), mu=np.full(M, 10.0), f_max=np.full(M, 4e9),
                   eta=np.ones(M), h_min=h_min, rate_fn=rate_fn)


def same_uav_penalty(n):
    """rate_fn: every uplink already assigned to UAV j this slot divides all rates to j by 10
    (co-channel interference of a second same-UAV assignment)."""
    calls = []

    def fn(pairs):
        calls.append(list(pairs))
        R = np.tile(BASE_R, (n, 1)).astype(float)
        for _, j in pairs:
            R[:, j] /= 10.0
        return R
    return fn, calls


def identity_rates(n):
    return lambda pairs: np.tile(BASE_R, (n, 1)).astype(float)


def controller(cls):
    return cls(CFG, "B_ZHOU" if cls is ZhouController else "B_QIN", Streams(1), 1e8, CFG.channel.h_min)


def decide(cls, obs):
    c = controller(cls)
    dec, _ = c.act(obs, 0, False)
    return [(a.task_index, a.uav) for a in dec.assignments], dec


@pytest.mark.parametrize("cls", [ZhouController, QinController])
def test_second_same_uav_assignment_changes_choice(cls):
    base, _ = decide(cls, mk_obs())
    assert base == [(0, 0), (1, 0)]                      # slot-start rates: both to UAV 0
    fn, calls = same_uav_penalty(2)
    inc, dec = decide(cls, mk_obs(rate_fn=fn))
    assert inc == [(0, 0), (1, 1)]                       # UAV 0 degraded after task 0
    assert calls and calls[0] == [(0, 0)]                # recomputed after the first assignment
    # task 1's prediction uses its rate WITH itself interfering (UAV 1: 5e7 / 10)
    t1 = dec.assignments[1]
    assert t1.t_total_pred == pytest.approx(8e6 * (1 + CFG.task.tx_overhead) / 5e6 + 1e8 / t1.f)


def crowding(n):
    """rate_fn: a single upload to UAV j keeps BASE_R[j]; a second same-UAV upload cuts every
    rate to j by 10 (co-channel interference)."""
    def fn(pairs):
        R = np.tile(BASE_R, (n, 1)).astype(float)
        cnt = np.bincount([j for _, j in pairs], minlength=M)
        for j in range(M):
            if cnt[j] >= 2:
                R[:, j] /= 10.0
        return R
    return fn


def two_tasks_tight_first():
    obs = mk_obs(n=2, rate_fn=crowding(2))
    obs.queue[0].deadline_s = 0.5        # alone on UAV 0: 0.088 s upload, fine; crowded: 0.88 s > 0.5
    return obs


@pytest.mark.parametrize("cls", [ZhouController, QinController])
def test_second_upload_breaking_first_is_refused(cls):
    """With the prefix-only check task 1 joins task 0 on UAV 0 and breaks task 0's
    deadline; with slot admission it is moved to another UAV."""
    c = controller(cls)
    c.ex.slot_admission = False
    dec, _ = c.act(two_tasks_tight_first(), 0, False)
    assert [(a.task_index, a.uav) for a in dec.assignments] == [(0, 0), (1, 0)]   # prefix-only check: breaks task 0
    got, dec = decide(cls, two_tasks_tight_first())
    assert got == [(0, 0), (1, 1)]                                                 # slot admission: task 1 moved
    obs = two_tasks_tight_first()
    R = obs.rate_fn([(a.task_index, a.uav) for a in dec.assignments])
    a0 = dec.assignments[0]
    rem0 = obs.queue[0].deadline_s
    assert 8e6 * (1 + CFG.task.tx_overhead) / R[0, 0] + 1e8 / a0.f <= rem0 + 1e-12


def _admitted_infeasible(arm_key, admission: bool = True, T=300, load=0.85, **ex_flags):
    """Real simulator, seed 9001 val block: count admitted tasks that miss their remaining
    deadline at obs.rate_fn(all slot pairs) with their allocated f -- including the
    serial-radio wait (device_busy_s at slot start + the device's earlier uploads this slot)
    -- and check that the simulator's fixed per-task rate equals that rate. Also counts
    slots with >= 2 different devices admitted. Returns a dict."""
    return _sim_stats(arm_key, admission, T, load, **ex_flags)


def _sim_stats(arm_key, admission, T, load, **ex_flags):
    from tpa.config import load_config as lc
    from tpa.env.queueing import f_required, mm1_wait, t_trans
    from tpa.env.simulator import Simulator
    from tpa.env.world import build_world
    from tpa.factory import make_controller
    cfg = lc()
    assert cfg.channel.interference_set == "slot_assigned"
    st = Streams(9001, "val")
    sim = Simulator(cfg, build_world(cfg, st.world), st, arm_key=arm_key, load=load)
    ctrl = make_controller(cfg, arm_key, st, sim.r_ref, sim.h_min)
    ctrl.ex.slot_admission = admission
    for k, v in ex_flags.items():
        setattr(ctrl.ex, k, v)
    ctrl.reset_segment(0)
    ctrl.start_episode()
    bad = n_adm = n_cmp = multi_dev = slots_with = 0
    r_min = cfg.channel.r_min_mbps * 1e6
    for t in range(T):
        obs = sim.begin_slot(t)
        dec, me = ctrl.act(obs, 0, False)
        pairs = [(a.task_index, a.uav) for a in dec.assignments]
        R = obs.rate_fn(pairs) if pairs else None
        busy = dict(getattr(obs, "device_busy_s", None) or {})
        serial: dict = {}
        want = {}
        devs = set()
        for a in dec.assignments:                  # simulator upload order = assignment order
            tk, j = obs.queue[a.task_index], a.uav
            devs.add(tk.device)
            Rf = float(R[a.task_index, j])
            want[tk.tid] = Rf
            rem = tk.deadline_s - (obs.t - tk.t_arr) * cfg.scenario.slot_s
            tt = t_trans(tk.size_bits, tk.complexity, Rf, cfg)
            wait = busy.get(tk.device, 0.0) + serial.get(tk.device, 0.0)
            serial[tk.device] = serial.get(tk.device, 0.0) + tt
            other = (wait + tt + mm1_wait(float(obs.lam_eff[j]), float(obs.mu[j]), float(obs.eta[j]))
                     + ctrl.ex.t_mem)
            n_adm += 1
            if Rf < r_min or a.f < f_required(tk.cycles, tk.complexity, rem - other) * (1 - 1e-9):
                bad += 1
        slots_with += bool(dec.assignments)
        multi_dev += len(devs) >= 2
        sim.end_slot(dec, me)
        for tid, Rf in want.items():
            if tid in sim.fixed_rate:              # still uploading at slot end
                assert sim.fixed_rate[tid] == pytest.approx(Rf, rel=1e-12)
                n_cmp += 1
    return dict(bad=bad, n_adm=n_adm, n_cmp=n_cmp, multi_dev=multi_dev, slots_with=slots_with,
                per_slot=n_adm / T, busy_given=getattr(obs, "device_busy_s", None) is not None)


@pytest.mark.parametrize("arm_key", ["B_QIN", "B_ZHOU"])
def test_real_sim_no_admitted_task_infeasible_at_final_rates(arm_key):
    before = _admitted_infeasible(arm_key, admission=False)
    after = _admitted_infeasible(arm_key, admission=True)
    print(f"\n{arm_key} seed 9001 val 300 slots load 0.85: infeasible/admitted prefix-only "
          f"{before['bad']}/{before['n_adm']}, with admission {after['bad']}/{after['n_adm']} "
          f"(fixed_rate compared on {after['n_cmp']} uploads)")
    assert after["bad"] == 0 and after["n_adm"] > 0 and after["n_cmp"] > 0


@pytest.mark.parametrize("arm_key", ["B_QIN", "B_ZHOU"])
def test_real_sim_coadmits_different_devices(arm_key):
    """With raise-admission, slots with >= 2 different devices admitted occur
    (seed 9001 val, load 0.55), and no admitted task is infeasible at the final rates
    including the serial-radio wait."""
    old = _admitted_infeasible(arm_key, load=0.55, allow_raise=False)
    new = _admitted_infeasible(arm_key, load=0.55)
    print(f"\n{arm_key} seed 9001 val 300 slots load 0.55 | refuse mode (serial radio on): "
          f"multi-device slots {old['multi_dev']}, assignments/slot {old['per_slot']:.3f}, "
          f"infeasible {old['bad']}/{old['n_adm']} | raise mode + serial radio: multi-device slots "
          f"{new['multi_dev']}, assignments/slot {new['per_slot']:.3f}, infeasible "
          f"{new['bad']}/{new['n_adm']} | obs.device_busy_s provided: {new['busy_given']}")
    assert new["multi_dev"] > 0 and new["bad"] == 0


# ---------------------------------------------------------------- raise-admission / serial radio, constructed
def mild(n):
    """rate_fn: every extra distinct device in the slot lowers ALL rates by 20% (Eq. 9's
    cross-UAV phi term): a second device needs only a small raise of the first one's f."""
    def fn(pairs):
        nd = len({int(i) for i, _ in pairs})       # tasks k have device k
        return np.tile(BASE_R, (n, 1)).astype(float) * 0.8 ** max(nd - 1, 0)
    return fn


@pytest.mark.parametrize("cls", [ZhouController, QinController])
def test_constructed_two_devices_coadmitted_with_raise(cls):
    c = controller(cls)
    c.ex.allow_raise = False
    dec, _ = c.act(mk_obs(n=2, rate_fn=mild(2)), 0, False)
    assert len(dec.assignments) == 1                       # refuse mode: second device refused
    got, dec = decide(cls, mk_obs(n=2, rate_fn=mild(2)))
    assert len(got) == 2 and {obs_t for obs_t, _ in got} == {0, 1}
    obs = mk_obs(n=2, rate_fn=mild(2))
    R = obs.rate_fn([(a.task_index, a.uav) for a in dec.assignments])
    for a in dec.assignments:                              # both meet deadlines at final rates
        tk = obs.queue[a.task_index]
        assert 8e6 * (1 + CFG.task.tx_overhead) / R[a.task_index, a.uav] + tk.cycles / a.f \
            <= tk.deadline_s + 1e-12


def test_executor_raise_floor_kept_in_pass2():
    """The raised admission f is a floor: with an all-zero method allocation the first task
    gets exactly its raised f_req (> its alone-f_req)."""
    from tpa.baselines import Executor
    ex = Executor(CFG, "B_QIN", CFG.channel.h_min)
    obs = mk_obs(n=2, rate_fn=mild(2))
    first = lambda i, k, c: c[0][0]
    dec = ex.run(obs, ex.window(obs), first, lambda j, it, F0: [0.0] * len(it))
    assert len(dec.assignments) == 2
    a0 = dec.assignments[0]
    tt_final = 8e6 * (1 + CFG.task.tx_overhead) / (BASE_R[a0.uav] * 0.8)
    assert a0.f == pytest.approx(1e8 / (5.0 - tt_final), rel=1e-9)


def test_serial_radio_same_device():
    """A device's second upload waits for its first (one radio)."""
    obs = mk_obs(n=2, rate_fn=identity_rates(2))
    for tk in obs.queue:
        tk.device = 7                                       # same device
    obs.queue[1].deadline_s, obs.queue[1].cycles = 0.15, 1e6   # alone: 0.088 s upload, fine
    got, _ = decide(ZhouController, obs)
    assert got == [(0, 0)]                                  # waits 0.088 s behind task 0 -> refused
    obs.queue[1].deadline_s = 0.5                           # enough room for the wait
    got, dec = decide(ZhouController, obs)
    assert len(got) == 2
    a1 = dec.assignments[1]                                 # t_total_pred includes the wait
    tt = 8e6 * (1 + CFG.task.tx_overhead) / BASE_R[0]
    assert a1.uav == 0 and a1.t_total_pred == pytest.approx(tt + tt + 1e6 / a1.f, rel=1e-9)


def test_device_busy_at_slot_start():
    obs = mk_obs(n=1, rate_fn=identity_rates(1))
    obs.queue[0].deadline_s = 0.55
    obs.device_busy_s = {0: 0.5}                            # 0.5 s of in-flight upload first
    assert decide(ZhouController, obs)[0] == []
    obs.device_busy_s = {0: 0.1}
    assert decide(ZhouController, obs)[0] == [(0, 0)]


def test_executor_admit_is_slot_admissible():
    """The executor delegates admission to tpa.drm.slot_admissible (one rule for DRM and
    the baselines): same ok / f_req / raises / delay on real-simulator slots."""
    from tpa.baselines import Executor
    from tpa.drm.drm import slot_admissible
    sim, _, cfg = make("B_ZHOU", seed=9001)
    ex = Executor(cfg, "B_ZHOU", sim.h_min)
    n_cmp = 0
    for t in range(30):
        obs = sim.begin_slot(t)
        if obs.h_min is not None:
            ex.drm.h_min = float(obs.h_min)
        F, E = obs.residual_f.copy(), obs.energy.copy()
        win = [int(i) for i in ex.window(obs)][:3]
        admitted = []
        for i in win:
            for j in range(ex.M):
                mine = ex.admit(obs, admitted, (i, j), F, E, obs.eta)
                ref = slot_admissible(obs, admitted, (i, j), cfg, t_mem_s=ex.t_mem, F_res=F,
                                      E_res=E, eta=obs.eta, h_min=ex.drm.h_min, allow_raise=True)
                assert (mine is not None) == ref.ok
                n_cmp += 1
                if ref.ok:
                    assert mine["f_req"] == ref.f_req and mine["wait"] == ref.delay
                    assert mine["raises"] == dict(ref.raises)
            ok = [j for j in range(ex.M) if ex.admit(obs, admitted, (i, j), F, E, obs.eta)]
            if ok:
                adm = ex.admit(obs, admitted, (i, ok[0]), F, E, obs.eta)
                for p_, fn in adm["raises"].items():
                    F[admitted[p_][1]] -= fn - admitted[p_][2]
                    admitted[p_] = (admitted[p_][0], admitted[p_][1], fn)
                F[ok[0]] -= adm["f_req"]
                admitted.append((i, ok[0], adm["f_req"]))
        sim.end_slot(Decision(), 0.0)
    assert n_cmp > 0
