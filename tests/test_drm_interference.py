"""DRM under interference_set = "slot_assigned": incremental interference,
residual-rate-budget admission (`slot_admissible`) and obs.h_min."""
import numpy as np
import pytest

from tpa.config import load_config
from tpa.drm import DRM, device_busy, finalize_predictions, slot_admissible
from tpa.env.queueing import f_required, predict_total_time, t_trans
from test_drm_drm import CFG, M, act, mk_drm, mk_obs, mk_task


def crowding_rate_fn(n_tasks=2, base=1e8, drop=1e4):
    """Like channel.rate: task r's rate to UAV j is base / drop^(n_j), n_j = number of
    OTHER tasks in `pairs` on UAV j (a task never interferes with itself); no cross-UAV
    coupling."""
    calls = []

    def fn(pairs):
        calls.append(list(pairs))
        out = np.empty((n_tasks, M))
        for r in range(n_tasks):
            n = np.zeros(M)
            for idx, j in pairs:
                if int(idx) != r:
                    n[int(j)] += 1
            out[r] = base / drop ** n
        return out
    return fn, calls


def with_fn(obs, fn):
    obs.rate_fn = fn
    obs.eta = np.full(M, 0.9)
    return obs


def test_rate_fn_changes_decisions():
    A, B = mk_task(0, prio=1.0), mk_task(1, prio=0.0)
    a = act(2, ftilde=0.5)                                    # f* = 2e9 > f_req
    a[:2, :M] = [1, 0, 0, 0, 0]                               # both prefer UAV 0
    # no rate_fn: both on UAV 0
    dec = mk_drm().run(mk_obs([A, B]), np.array([0, 1]), a, None, 0.0)
    assert [(x.task_index, x.uav) for x in dec.assignments] == [(0, 0), (1, 0)]
    # rate_fn: with A on UAV 0, B's rate to UAV 0 is 1e4 < R_min -> UAV 1
    fn, calls = crowding_rate_fn()
    dec2 = mk_drm().run(with_fn(mk_obs([A, B]), fn), np.array([0, 1]), a, None, 0.0)
    assert [(x.task_index, x.uav) for x in dec2.assignments] == [(0, 0), (1, 1)]
    assert [(0, 0)] in calls and [(0, 0), (1, 1)] in calls
    assert dec2.assignments[1].t_total_pred == pytest.approx(dec.assignments[1].t_total_pred)
    # milder crowding (rate 1e7): A (f = 2e9) still meets its deadline -> B stays on UAV 0,
    # and B's T_total uses the lowered rate
    fn3, _ = crowding_rate_fn(drop=10.0)
    dec3 = mk_drm().run(with_fn(mk_obs([A, B]), fn3), np.array([0, 1]), a, None, 0.0)
    assert [(x.task_index, x.uav) for x in dec3.assignments] == [(0, 0), (1, 0)]
    t_comp = 1e9 / 2e9
    assert dec.assignments[1].t_total_pred == pytest.approx(1e6 * 1.1 / 1e8 + t_comp)
    assert dec3.assignments[1].t_total_pred == pytest.approx(1e6 * 1.1 / 1e7 + t_comp)
    assert dec3.util_terms["rate_frac"][0] == pytest.approx(1.0 + 0.1)
    assert dec3.util_terms["admit_rate"] == pytest.approx([1e8, 1e7])


def test_second_upload_refused_if_it_breaks_the_first():
    """Slot admission in refuse mode (slot_raise_f=False override): A is sized exactly (f = f_req at
    1e8 b/s). Adding B on UAV 0 would cut A's rate to 1e7 and make A miss its deadline ->
    B is refused on UAV 0 (goes to UAV 1)."""
    A, B = mk_task(0, prio=1.0), mk_task(1, prio=0.0)
    a = act(2, ftilde=0.0)                                    # f = f_req
    a[:2, :M] = [1, 0, 0, 0, 0]
    fn, _ = crowding_rate_fn(drop=10.0)
    refuse = {"slot_raise_f": False}
    dec = mk_drm(refuse).run(with_fn(mk_obs([A, B]), fn), np.array([0, 1]), a, None, 0.0)
    assert [(x.task_index, x.uav) for x in dec.assignments] == [(0, 0), (1, 1)]
    # final rates (all slot pairs) leave every admitted task satisfiable
    final = fn([(x.task_index, x.uav) for x in dec.assignments])
    for x in dec.assignments:
        t = [A, B][x.task_index]
        T = predict_total_time(t.size_bits, t.cycles, t.complexity, final[x.task_index, x.uav],
                               x.f, 0.0, 10.0, 0.9, 0.0, CFG)
        assert T <= t.deadline_s * (1 + 1e-9)
    # without slot admission, the prefix-only check puts B on UAV 0 and breaks A
    old = mk_drm({"slot_admission": False}).run(with_fn(mk_obs([A, B]), fn), np.array([0, 1]),
                                                a, None, 0.0)
    assert [(x.task_index, x.uav) for x in old.assignments] == [(0, 0), (1, 0)]
    finalo = fn([(0, 0), (1, 0)])
    T_A = predict_total_time(1e6, 1e9, 1.0, finalo[0, 0], old.assignments[0].f, 0.0, 10.0, 0.9, 0.0, CFG)
    assert T_A > A.deadline_s
    # all other UAVs blocked -> B is re-queued instead
    obs = with_fn(mk_obs([A, B]), fn)
    obs.h_now[1, 1:] = 0.0
    dec_b = mk_drm(refuse).run(obs, np.array([0, 1]), a, None, 0.0)
    assert [(x.task_index, x.uav) for x in dec_b.assignments] == [(0, 0)]


def test_raise_option_keeps_first_satisfiable():
    A, B = mk_task(0, prio=1.0), mk_task(1, prio=0.0)
    a = act(2, ftilde=0.0)
    a[:2, :M] = [1, 0, 0, 0, 0]
    fn, _ = crowding_rate_fn(drop=10.0)
    obs = with_fn(mk_obs([A, B]), fn)
    assert CFG.drm.slot_raise_f is True and mk_drm().raise_earlier is True   # raise-admission is the default
    dec = mk_drm().run(obs, np.array([0, 1]), a, None, 0.0)
    assert [(x.task_index, x.uav) for x in dec.assignments] == [(0, 0), (1, 0)]
    fA_new = f_required(1e9, 1.0, 2.0 - 1e6 * 1.1 / 1e7)
    assert dec.assignments[0].f == pytest.approx(fA_new)
    # finalized at the final rate 1e7 with the RAISED f: exactly the deadline
    assert dec.assignments[0].t_total_pred == pytest.approx(2.0)
    assert dec.util_terms["t_total_admit"][0] == pytest.approx(2.0)  # admission-time value kept
    assert dec.util_terms["cpu_frac"][0] == pytest.approx((fA_new + dec.assignments[1].f) / 4e9)
    assert sum(x.f for x in dec.assignments) <= 4e9
    assert mk_drm({"slot_raise_f": False}).raise_earlier is False      # flag override


def test_slot_admissible_direct():
    A, B = mk_task(0), mk_task(1)
    fn, _ = crowding_rate_fn(drop=10.0)
    obs = with_fn(mk_obs([A, B]), fn)
    adm = slot_admissible(obs, [], (0, 0), CFG)
    assert adm.ok and adm.f_req == pytest.approx(f_required(1e9, 1.0, 2.0 - 1.1e6 / 1e8))
    f_A = adm.f_req
    adm2 = slot_admissible(obs, [(0, 0, f_A)], (1, 0), CFG)
    assert not adm2.ok                                        # would break A
    assert adm2.rates[0, 0] == pytest.approx(1e7)
    adm3 = slot_admissible(obs, [(0, 0, 2e9)], (1, 0), CFG)
    assert adm3.ok and adm3.f_req == pytest.approx(f_required(1e9, 1.0, 2.0 - 1.1e6 / 1e7))
    adm4 = slot_admissible(obs, [(0, 0, f_A)], (1, 0), CFG, F_res=np.full(M, 4e9), allow_raise=True)
    assert adm4.ok and adm4.raises[0] == pytest.approx(adm3.f_req)
    assert not slot_admissible(obs, [(0, 0, f_A)], (1, 0), CFG, F_res=np.full(M, 5.4e8),
                               allow_raise=True).ok          # 5.4e8 < raise (2.6e7) + f_req_B (5.29e8)
    assert not slot_admissible(obs, [], (0, 0), CFG, f_cand=0.5 * f_A).ok


def test_rate_fn_none_is_identical():
    tasks = [mk_task(k, cycles=2e8 + 1e7 * k) for k in range(10)]
    a = act(10, ftilde=0.3)
    d1 = mk_drm().run(mk_obs(tasks), np.arange(10), a, None, 0.0)
    obs = mk_obs(tasks)
    obs.rate_fn = None
    d2 = mk_drm().run(obs, np.arange(10), a, None, 0.0)
    assert [(x.task_index, x.uav, x.f, x.t_total_pred) for x in d1.assignments] == \
        [(x.task_index, x.uav, x.f, x.t_total_pred) for x in d2.assignments]
    assert d1.util_terms["n_admission_checks"] == 0


def test_h_min_from_obs():
    obs = mk_obs([mk_task(0)])
    obs.h_now[:] = CFG.channel.h_min * 0.1                    # below cfg h_min
    assert mk_drm().run(obs, np.array([0]), act(1), None, 0.0).assignments == []
    obs.h_min = CFG.channel.h_min * 0.01                      # obs threshold in force
    assert len(mk_drm().run(obs, np.array([0]), act(1), None, 0.0).assignments) == 1
    obs2 = mk_obs([mk_task(0)])
    obs2.h_min = 2.0                                          # h_now = 1 < 2
    assert mk_drm().run(obs2, np.array([0]), act(1), None, 0.0).assignments == []


def test_burst_budget_field():
    dec = mk_drm().run(mk_obs([mk_task(0)]), np.array([0]), act(1), None, 0.0)
    assert dec.burst_budget is not None and dec.burst_budget.shape == (M,)
    assert np.array_equal(dec.burst_budget, dec.util_terms["burst_budget_frac"])


def run_real(T=300, flags=None, seed=9001, rng_seed=0):
    """Real simulator (val block, slot_assigned), random actor, reactive DRM (A9)."""
    from _env_helpers import make_sim
    cfg = load_config()
    assert cfg.channel.interference_set == "slot_assigned"
    sim, cfg, world = make_sim(seed, block="val", cfg=cfg, arm_key="A9")
    drm = DRM(cfg, dict({"pattern_terms": False}, **(flags or {})), "A9")
    rng = np.random.Generator(np.random.PCG64(rng_seed))
    rec = {"ratio": [], "infeasible_final": 0, "n_assign": 0, "multi_slots": 0,
           "realized_eq_final": 0, "realized_checked": 0, "drm_ms": [], "uploads": [], "final": []}
    import time
    r_min = cfg.channel.r_min_mbps * 1e6
    for t in range(T):
        obs = sim.begin_slot(t)
        pi = drm.priority(obs, None, 0.0)
        win = drm.select_window(pi, obs)
        a = rng.uniform(-1, 1, size=(cfg.task.q_max, 2 * M))
        t0 = time.perf_counter_ns()
        dec = drm.run(obs, win, a, None, 0.0)
        rec["drm_ms"].append((time.perf_counter_ns() - t0) / 1e6)
        pairs = [(x.task_index, x.uav) for x in dec.assignments]
        final = obs.rate_fn(pairs) if pairs else None
        rec["n_assign"] += len(pairs)
        rec["multi_slots"] += len(pairs) >= 2
        acc = {}                                              # device chain at FINAL rates
        for x, r_adm in zip(dec.assignments, dec.util_terms["admit_rate"]):
            tk = obs.queue[x.task_index]
            Rf = final[x.task_index, x.uav]
            rec["ratio"].append(Rf / r_adm)
            rem = tk.deadline_s - (obs.t - tk.t_arr) * cfg.scenario.slot_s
            d = acc.get(tk.device, device_busy(obs, tk.device))
            tt = t_trans(tk.size_bits, tk.complexity, Rf, cfg)
            acc[tk.device] = d + tt
            rec["uploads"].append((tk, obs.t * cfg.scenario.slot_s + d + tt, d))
            from tpa.env.queueing import mm1_wait
            q = mm1_wait(float(obs.lam_eff[x.uav]), float(obs.mu[x.uav]), float(obs.eta[x.uav]))
            rec["final"].append((tk, obs.t, x.t_total_pred, q))
            T_f = d + predict_total_time(tk.size_bits, tk.cycles, tk.complexity, Rf, x.f,
                                         float(obs.lam_eff[x.uav]), float(obs.mu[x.uav]),
                                         float(obs.eta[x.uav]), drm.t_mem, cfg)
            if T_f > rem * (1 + 1e-9) or Rf < r_min:
                rec["infeasible_final"] += 1
        tids = [(obs.queue[x.task_index].tid, x) for x in dec.assignments]
        sim.end_slot(dec, 0.0)
        for tid, x in tids:
            if tid in sim.fixed_rate:
                rec["realized_checked"] += 1
                if sim.fixed_rate[tid] == pytest.approx(final[x.task_index, x.uav], rel=1e-12):
                    rec["realized_eq_final"] += 1
    return rec


def test_real_simulator_admitted_tasks_satisfiable_at_realized_rates():
    rec = run_real()
    assert rec["n_assign"] > 0 and rec["multi_slots"] > 0
    assert rec["infeasible_final"] == 0
    assert rec["realized_checked"] > 0 and rec["realized_eq_final"] == rec["realized_checked"]
    # the LAST admission check of each slot already uses the realized (end-of-slot) rate
    old = run_real(flags={"slot_admission": False})
    assert old["infeasible_final"] > 0                        # infeasible without slot admission


def upload_match(rec, slot_s=1.0, tol=1e-6):
    """Compare realized upload finish (Task.t_upload_done, slots) with the predicted
    finish t + device wait + T_trans(final rate). Returns counts."""
    out = {"n": 0, "exact": 0, "early": 0, "late": 0, "n_wait": 0, "exact_wait": 0}
    for tk, pred, d in rec["uploads"]:
        if tk.t_upload_done is None or tk.t_upload_done < 0:
            continue
        real = tk.t_upload_done * slot_s
        out["n"] += 1
        out["n_wait"] += d > 0
        if abs(real - pred) <= tol * max(1.0, pred):
            out["exact"] += 1
            out["exact_wait"] += d > 0
        elif real < pred:
            out["early"] += 1
        else:
            out["late"] += 1
    return out


def test_real_simulator_serial_radio_finish_matches_prediction():
    """With the device-busy wait in T_trans, realized upload finishes of A9
    assignments match the prediction (slot model, T_mem = 0, no packet loss). A
    realized finish may only be EARLIER, when an in-flight predecessor of the same
    device is aborted at its deadline and frees the radio."""
    rec = run_real(T=300)
    m = upload_match(rec)
    assert m["n"] > 0 and m["n_wait"] > 0
    assert m["late"] == 0
    assert m["exact"] >= 0.95 * m["n"]


# ---------------------------------------------------------------- serial radio
def same_device(*tasks, dev=7):
    for t in tasks:
        t.device = dev
    return tasks


def test_device_wait_constructed():
    A, B = same_device(mk_task(0, prio=1.0), mk_task(1, prio=0.0))
    a = act(2, ftilde=0.5)                                    # f = 2e9 -> T_comp 0.5
    a[0, :M] = [1, 0, 0, 0, 0]
    a[1, :M] = [0, 1, 0, 0, 0]                                # different UAVs, same radio
    tt = 1e6 * 1.1 / 1e8
    for fn in (None, crowding_rate_fn(drop=1.0)[0]):          # both DRM paths
        obs = mk_obs([A, B])
        if fn is not None:
            with_fn(obs, fn)
        dec = mk_drm().run(obs, np.array([0, 1]), a, None, 0.0)
        assert [x.uav for x in dec.assignments] == [0, 1]
        assert dec.assignments[0].t_total_pred == pytest.approx(tt + 0.5)
        assert dec.assignments[1].t_total_pred == pytest.approx(tt + tt + 0.5)   # waits for A
        assert dec.util_terms["admit_delay"] == pytest.approx([0.0, tt])
        # plus an upload already in flight on the device at slot start
        obs = mk_obs([A, B])
        obs.device_busy_s = {7: 0.3}
        if fn is not None:
            with_fn(obs, fn)
        dec = mk_drm().run(obs, np.array([0, 1]), a, None, 0.0)
        assert dec.assignments[0].t_total_pred == pytest.approx(0.3 + tt + 0.5)
        assert dec.assignments[1].t_total_pred == pytest.approx(0.3 + 2 * tt + 0.5)
    # other devices are not delayed
    C = mk_task(2)
    obs = mk_obs([C])
    obs.device_busy_s = {7: 0.3}
    d = mk_drm().run(obs, np.array([0]), act(1, ftilde=0.5), None, 0.0)
    assert d.assignments[0].t_total_pred == pytest.approx(tt + 0.5)
    # None = idle
    obs.device_busy_s = None
    assert mk_drm().run(obs, np.array([0]), act(1, ftilde=0.5), None, 0.0).assignments[0].t_total_pred \
        == pytest.approx(tt + 0.5)


def test_device_wait_makes_task_infeasible():
    A, = same_device(mk_task(0, deadline=2.0))
    obs = mk_obs([A])
    obs.device_busy_s = {7: 1.995}                           # 1.995 + 0.011 > 2 s
    assert mk_drm().run(obs, np.array([0]), act(1, ftilde=1.0), None, 0.0).assignments == []
    obs2 = with_fn(mk_obs([A]), crowding_rate_fn(1)[0])
    obs2.device_busy_s = {7: 1.995}
    assert not slot_admissible(obs2, [], (0, 0), CFG).ok


def test_slot_admissible_device_chain_rechecks_earlier():
    """A and B share a radio, C is on another device. Adding C on UAV 0 slows A (drop 10),
    which also delays B (it waits for A's longer upload): B must be re-checked with the
    longer wait."""
    A, B = same_device(mk_task(0), mk_task(1, deadline=0.62))
    C = mk_task(2)
    fn, _ = crowding_rate_fn(n_tasks=3, drop=10.0)
    obs = with_fn(mk_obs([A, B, C]), fn)
    tt = 1.1e6 / 1e8
    # B on UAV 1 behind A: wait tt, own tt, f = 2e9 -> 0.5 s: 0.522 <= 0.62
    adm_b = slot_admissible(obs, [(0, 0, 2e9)], (1, 1), CFG)
    assert adm_b.ok and adm_b.delay == pytest.approx(tt)
    # C on UAV 0 cuts A's rate to 1e7: B's wait becomes 0.11 -> 0.11 + 0.011 + 0.5 > 0.62
    adm_c = slot_admissible(obs, [(0, 0, 2e9), (1, 1, 2e9)], (2, 0), CFG)
    assert not adm_c.ok
    assert adm_c.rates[0, 0] == pytest.approx(1e7)


# ---------------------------------------------------------------- finalized J1 predictions
def test_finalize_predictions_two_task_slot():
    """B's admission on UAV 0 cuts A's rate 1e8 -> 1e7: A's finalized t_total_pred is Eq. 13
    at the final rate; the admission-time values are kept in util_terms["t_total_admit"]."""
    A, B = mk_task(0, prio=1.0), mk_task(1, prio=0.0)
    a = act(2, ftilde=0.5)                                    # f = 2e9
    a[:2, :M] = [1, 0, 0, 0, 0]
    fn, _ = crowding_rate_fn(drop=10.0)
    dec = mk_drm().run(with_fn(mk_obs([A, B]), fn), np.array([0, 1]), a, None, 0.0)
    assert [(x.task_index, x.uav) for x in dec.assignments] == [(0, 0), (1, 0)]
    assert dec.util_terms["t_total_admit"] == pytest.approx([1.1e6 / 1e8 + 0.5, 1.1e6 / 1e7 + 0.5])
    hand_A = predict_total_time(1e6, 1e9, 1.0, 1e7, 2e9, 0.0, 10.0, 0.9, 0.0, CFG)
    assert hand_A == pytest.approx(1.1e6 / 1e7 + 0.5)
    assert dec.assignments[0].t_total_pred == pytest.approx(hand_A)
    assert dec.assignments[1].t_total_pred == pytest.approx(1.1e6 / 1e7 + 0.5)
    # same device: B also waits for A's (slower) upload
    A2, B2 = same_device(mk_task(0, prio=1.0), mk_task(1, prio=0.0))
    dec2 = mk_drm().run(with_fn(mk_obs([A2, B2]), fn), np.array([0, 1]), a, None, 0.0)
    assert dec2.assignments[1].t_total_pred == pytest.approx(2 * 1.1e6 / 1e7 + 0.5)
    # without rate_fn nothing changes
    d3 = mk_drm().run(mk_obs([A, B]), np.array([0, 1]), a, None, 0.0)
    assert [x.t_total_pred for x in d3.assignments] == pytest.approx(d3.util_terms["t_total_admit"])
    # direct call is idempotent
    before = [x.t_total_pred for x in dec.assignments]
    finalize_predictions(with_fn(mk_obs([A, B]), fn), dec, CFG, 0.0)
    assert [x.t_total_pred for x in dec.assignments] == pytest.approx(before)


def test_real_simulator_finalized_prediction_equals_realized():
    """A9 (T_mem = 0), no packet loss: for every task that finishes, the finalized
    predicted T_total minus the M/M/1 predictor term (Eq. 16 is a predictor only;
    the reservation model has no UAV queue) equals realized t_done - t_assign."""
    rec = run_real(T=300)
    n = exact = 0
    for tk, t_as, pred, q in rec["final"]:
        if tk.t_done is None or tk.t_done < 0:
            continue
        n += 1
        real = (tk.t_done - t_as) * 1.0
        if abs(real - (pred - q)) <= 1e-6 * max(1.0, real):
            exact += 1
    assert n > 0
    assert exact == n, f"{n - exact} of {n} finished tasks differ from the finalized prediction"


# ---------------------------------------------------------------- one admission rule for DRM and executor
def test_drm_and_executor_same_admission():
    """DRM.run and the baseline Executor are ONE admission rule (slot_admissible with
    allow_raise=True). Constructed 3-task slot: A (dev 0) -> UAV 0, B (dev 1) -> UAV 1,
    C (dev 2) -> UAV 0; C's admission cuts A's rate 1e8 -> 1e7 and requires raising A's f.
    Each task has a single candidate UAV (others blocked by h_now = 0) so the two paths'
    choice rules coincide; both methods allocate zero (f = raised floor)."""
    from tpa.baselines.executor import Executor
    A, B, C = mk_task(0), mk_task(1), mk_task(2)
    A.device, B.device, C.device = 0, 1, 2
    fn, _ = crowding_rate_fn(n_tasks=3, drop=10.0)

    def obs_():
        o = with_fn(mk_obs([A, B, C]), fn)
        o.f_max = np.full(M, 4e9)
        o.h_now[:] = 0.0
        o.h_now[0, 0] = o.h_now[1, 1] = o.h_now[2, 0] = 1.0
        return o

    a = act(3, ftilde=0.0)                                    # f* = 0 -> f = f_req (raised)
    drm = mk_drm()
    d_drm = drm.run(obs_(), np.array([0, 1, 2]), a, None, 0.0)
    ex = Executor(CFG, "B_ZHOU", CFG.channel.h_min)
    o = obs_()
    d_ex = ex.run(o, ex.window(o), lambda i, k, c: c[0][0], lambda j, it, F0: [0.0] * len(it))
    key = lambda d: [(x.task_index, x.uav, x.f, x.t_total_pred) for x in d.assignments]
    assert [(x.task_index, x.uav) for x in d_drm.assignments] == [(0, 0), (1, 1), (2, 0)]
    assert key(d_ex) == pytest.approx(key(d_drm))
    # the third admission raised A: f_A = f_req at 1e7 (> f_req at 1e8)
    fA_raised = f_required(1e9, 1.0, 2.0 - 1.1e6 / 1e7)
    assert d_drm.assignments[0].f == pytest.approx(fA_raised)
    assert fA_raised > f_required(1e9, 1.0, 2.0 - 1.1e6 / 1e8)
    assert d_drm.assignments[0].t_total_pred == pytest.approx(2.0)
    assert d_drm.util_terms["cpu_frac"] == pytest.approx(d_ex.util_terms["cpu_frac"])
    # (util_terms["t_total_admit"] is a diagnostic only and differs by construction: the
    # executor computes it in its pass 2 with the already-raised f, DRM at admission time.)
    # in refuse mode (diagnostic override) DRM would not admit C: the raise rule matters in this slot
    d_ref = mk_drm({"slot_raise_f": False}).run(obs_(), np.array([0, 1, 2]), a, None, 0.0)
    assert [(x.task_index, x.uav) for x in d_ref.assignments] == [(0, 0), (1, 1)]
