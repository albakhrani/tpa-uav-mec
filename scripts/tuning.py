"""Equal-budget hyperparameter tuning on validation seeds.

Every tunable arm gets 12 configurations drawn from ITS OWN declared search space with a
fixed per-arm sampler seed (SeedSequence([7, sum(map(ord, arm))]); independent of any run); the first
N_STAGE1 = 6 of them are tuned, the same budget for every arm. Two stages:
  stage 1: 6 configs x validation seed 9004 x 300 episodes
  stage 2: top 2 configs (by stage-1 score) x validation seed 9005 x 1000 episodes
(Validation seeds 9001-9003 served earlier diagnostics; tuning does not use them.)
Score = validation-block completion at medium load of the selected checkpoint (the trainer's
`best_val`), averaged over the stage's seeds. The winner's overrides are written to
runs/tune/selected.json and applied to that arm's main-campaign rows. Test seeds are never
used; the stage-2 seeds differ from the stage-1 seeds so selection does not reuse traces.

  python scripts/tuning.py sample      -> runs/tune/configs.json (deterministic)
  python scripts/tuning.py enqueue1    -> stage-1 rows (phase "tune")
  python scripts/tuning.py score1      -> runs/tune/stage1_scores.json, enqueue stage-2 rows
  python scripts/tuning.py select      -> runs/tune/selected.json + a Markdown result table (repository root)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tpa.arms import ARMS as _ALL_ARMS        # noqa: E402
from tpa.config import load_config            # noqa: E402
from tpa.run.logio import read_all            # noqa: E402
from tpa.run.manifest import code_hash        # noqa: E402
from tpa.run.queue import RunQueue            # noqa: E402

TUNE = ROOT / "runs" / "tune"
MECHANISM_ONLY = {"A15"}                 # mechanism-check arms: never tuned, refused by scripts/enqueue.py
ARMS = {a: f for a, f in _ALL_ARMS.items() if a not in MECHANISM_ONLY}
ADOPTED_FILE = ROOT / "runs" / "d3" / "adopted.json"   # adopted scenario (25 UAVs, sizes, deadlines, loads) for every row
N_CONFIGS = 12                          # sampled per arm (deterministic; runs/tune/configs.json holds all 12)
N_STAGE1 = 6                            # first 6 of the 12 sampled configurations enter stage 1 (the tuning budget)
STAGE1_SEEDS, STAGE1_EPISODES = (9004,), 300          # one validation seed in stage 1
STAGE2_SEEDS, STAGE2_EPISODES, STAGE2_TOP = (9005,), 1000, 2   # top 2 on seed 9005
LR = (3e-4, 3e-3)                       # log-uniform
U = (2, 4, 8)
ENT = (0.25, 0.5, 1.0)                  # target-entropy scale, every SAC-based arm
THETA_B = (0.6, 0.75, 0.9)
BETA_C = (0.1, 0.25, 0.5)
LSTM_H = (48, 64, 80)                   # A10 width candidates; checked against the +-5% rule
ES_SIGMA = (0.005, 0.05)                # log-uniform (so the 12 configs are distinct)
ES_LR = (0.002, 0.05)                   # log-uniform
LOOKUP_EWM = (0.05, 0.1, 0.2)
LOOKUP_F = (0.75, 1.0, 1.5)
DELAY = (100, 300, 900)                 # A16 consolidation delay Delta = w, 3w, P/4 (slots)

SAC_ARMS = [a for a, f in ARMS.items() if f.controller in ("tpa", "lstm", "framestack")]
GATED = [a for a in SAC_ARMS if ARMS[a].memory and ARMS[a].gate_on and ARMS[a].use_ml]
BURST = [a for a, f in ARMS.items() if f.burst_on]


def space_of(arm: str) -> dict:
    """Declared search space per arm (dims differ; the budget of N_STAGE1 = 6 tuned configurations does not)."""
    f = ARMS[arm]
    sp: dict = {}
    if f.controller in ("tpa", "lstm", "framestack"):
        sp["learner.lr"] = ("log", LR)
        sp["learner.update_every"] = ("choice", U)
        sp["learner.target_entropy_scale"] = ("choice", ENT)
    elif f.controller == "huang":
        sp["learner.lr"] = ("log", LR)
        sp["learner.update_every"] = ("choice", U)
    elif f.controller == "song":
        sp["baselines.es_sigma"] = ("log", ES_SIGMA)
        sp["baselines.es_lr"] = ("log", ES_LR)
    elif f.controller == "lookup":
        sp["baselines.lookup_ewm"] = ("choice", LOOKUP_EWM)
        sp["baselines.lookup_f_scale"] = ("choice", LOOKUP_F)
    else:                                   # zhou, qin: nothing learned
        return {}
    if f.burst_on:
        sp["burst.theta_b"] = ("choice", THETA_B)
    # theta_p is NOT tuned: a pre-registered, outcome-blind criterion on the gate fixes it (no tuning grid)
    if f.controller == "lstm":
        sp["baselines.lstm_hidden"] = ("choice", LSTM_H)
    if arm == "A13":
        sp["drm.beta_c"] = ("choice", BETA_C)
    if getattr(f, "delayed", False):                      # A16: Delta is the 5th tunable (A0's four plus Delta)
        sp["memory.delay_slots"] = ("choice", DELAY)
    return sp


def sample(arm: str) -> list[dict]:
    # fixed sampler seed per arm (reproducible; independent of any run)
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([7, sum(map(ord, arm))])))
    sp = space_of(arm)
    if not sp:
        return [{}]
    out, seen = [], set()
    while len(out) < N_CONFIGS:
        cfg = {}
        for k, (kind, v) in sp.items():
            if kind == "log":
                cfg[k] = float(np.exp(rng.uniform(np.log(v[0]), np.log(v[1]))))
                cfg[k] = float(f"{cfg[k]:.3g}")
            else:
                cfg[k] = v[int(rng.integers(len(v)))]
        key = json.dumps(cfg, sort_keys=True)
        if key in seen and len(seen) < _space_size(sp):
            continue
        seen.add(key)
        out.append(cfg)
    if "memory.delay_slots" in sp:                        # A16: every Delta value twice among the first 6
        for i, c in enumerate(out):
            c["memory.delay_slots"] = DELAY[i % len(DELAY)]
    return out


def _space_size(sp: dict) -> float:
    n = 1.0
    for kind, v in sp.values():
        n *= math.inf if kind == "log" else len(v)
    return n


def cmd_sample():
    TUNE.mkdir(parents=True, exist_ok=True)
    only = set(sys.argv[sys.argv.index("--arms") + 1].split(",")) if "--arms" in sys.argv else None
    configs = {arm: sample(arm) for arm in ARMS}
    if only is not None:                                   # --arms: add arms to an existing configs.json
        old = json.loads((TUNE / "configs.json").read_text())
        for arm in old:
            if arm not in only and configs.get(arm) != old[arm]:
                raise SystemExit(f"{arm}: the sampler no longer reproduces configs.json; nothing written")
        configs = {**old, **{a: configs[a] for a in only}}
    if "A10" in configs:                                   # enforce the +-5% rule for A10 widths
        from tpa.baselines.lstm_ac import param_counts
        ok = []
        for c in configs["A10"]:
            pc = param_counts(load_config(overrides=c))
            if abs(pc["rel_diff"]) <= 0.05:
                ok.append(c)
            else:
                c2 = dict(c, **{"baselines.lstm_hidden": 64})
                ok.append(c2)
        configs["A10"] = ok
    (TUNE / "configs.json").write_text(json.dumps(configs, indent=1))
    for arm, cs in configs.items():
        print(f"{arm}: {len(cs)} configs, space {sorted(space_of(arm))}")


def _rid(stage, arm, i, seed, workload: str = "sinusoidal"):
    """Tuning rows on another workload carry its name as a suffix, so that
    the sinusoidal rows of the same arm / configuration keep their run ids."""
    sfx = "" if workload == "sinusoidal" else f"-{workload}"
    return f"tune{stage}-{arm}-c{i:02d}-s{seed}{sfx}"


def _opt(name: str, default: str) -> str:
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def adopted_overrides() -> dict:
    """The adopted scenario (ADOPTED_FILE) is not in config/default.toml (the defaults
    keep the 5-UAV test geometry); every tuning and campaign row carries it explicitly."""
    return dict(json.loads(ADOPTED_FILE.read_text())["overrides"])


def row_overrides(cfg_sample: dict, episodes: int) -> dict:
    """Adopted scenario + the sampled per-arm hyperparameters + the stage's episode count.
    The memory defaults (theta_p 0.35, rho_merge 0.95, arrivals key, layernorm, zone) come
    from config/default.toml (adopted from outcome-blind mechanism checks) and are never sampled."""
    ov = adopted_overrides()
    ov.update(cfg_sample)
    ov["run.episodes"] = int(episodes)
    return ov


def cmd_enqueue1():
    # `--arms A,B`: only these arms; `--workload pareto`: rows on that workload
    # (run ids get the suffix "-pareto"); `--config 4`: only this configuration index (e.g. A0's
    # selected configuration on the Pareto workload as the reference run of the forecasting mechanism check).
    only = set(sys.argv[sys.argv.index("--arms") + 1].split(",")) if "--arms" in sys.argv else None
    workload = _opt("--workload", "sinusoidal")
    cfg_only = int(_opt("--config", "-1"))
    configs = json.loads((TUNE / "configs.json").read_text())
    q = RunQueue(ROOT / "runs" / "runs.db")
    cc = code_hash()
    n = 0
    for arm, cs in configs.items():
        if not space_of(arm):
            continue                                       # zhou, qin: nothing to tune
        if only is not None and arm not in only:
            continue
        for i, c in enumerate(cs[:N_STAGE1]):
            if cfg_only >= 0 and i != cfg_only:
                continue
            ov = row_overrides(c, STAGE1_EPISODES)
            ch = load_config(overrides=ov).hash()
            for seed in STAGE1_SEEDS:
                n += q.enqueue(_rid(1, arm, i, seed, workload), arm, seed, workload, "tune", ch, cc,
                               overrides=ov, priority=1 if ARMS[arm].memory else 0)
    print(f"enqueued {n} stage-1 rows (workload {workload})")
    print(q.status_table())
    q.close()


def _score(rid: str) -> float:
    d = ROOT / "runs" / "tune" / rid
    r = d / "RESULT.json"
    if not r.exists():
        return math.nan
    return float(json.loads(r.read_text()).get("best_val", math.nan))


def cmd_score1():
    # optional `--arms A0,A1,...`: enqueue stage 2 only for these arms (an arm enters stage 2 only
    # after its stage-1 checks have passed); every arm is scored
    only = set(sys.argv[sys.argv.index("--arms") + 1].split(",")) if "--arms" in sys.argv else None
    workload = _opt("--workload", "sinusoidal")           # e.g. pareto: the Pareto tuning rows of A16'
    configs = json.loads((TUNE / "configs.json").read_text())
    scores = {}
    for arm, cs in configs.items():
        if not space_of(arm):
            continue
        rows = []
        for i, c in enumerate(cs[:N_STAGE1]):
            vals = [_score(_rid(1, arm, i, s, workload)) for s in STAGE1_SEEDS]
            rows.append(dict(i=i, config=c, seeds=vals,
                             score=float(np.mean(vals)) if all(np.isfinite(vals)) else math.nan))
        scores[arm] = rows
    sfx = "" if workload == "sinusoidal" else f"_{workload}"
    (TUNE / f"stage1_scores{sfx}.json").write_text(json.dumps(scores, indent=1))
    q = RunQueue(ROOT / "runs" / "runs.db")
    cc = code_hash()
    n = 0
    for arm, rows in scores.items():
        if only is not None and arm not in only:
            print(f"{arm}: not in --arms; not enqueuing stage 2")
            continue
        done = [r for r in rows if np.isfinite(r["score"])]
        if len(done) < len(rows):
            print(f"{arm}: {len(done)}/{len(rows)} stage-1 configs scored; not enqueuing stage 2")
            continue
        top = sorted(done, key=lambda r: -r["score"])[:STAGE2_TOP]
        print(f"{arm}: stage-1 top {[ (r['i'], round(r['score'],4)) for r in top ]}")
        for r in top:
            ov = row_overrides(r["config"], STAGE2_EPISODES)
            ch = load_config(overrides=ov).hash()
            for seed in STAGE2_SEEDS:
                n += q.enqueue(_rid(2, arm, r["i"], seed, workload), arm, seed, workload, "tune", ch, cc,
                               overrides=ov, priority=2 if ARMS[arm].memory else 1)
    print(f"enqueued {n} stage-2 rows (workload {workload})")
    q.close()


def cmd_select():
    configs = json.loads((TUNE / "configs.json").read_text())
    s1 = json.loads((TUNE / "stage1_scores.json").read_text())
    selected, lines = {}, ["# Tuning result (equal budget, validation seeds only)\n",
                           f"Stage 1: {N_STAGE1} of {N_CONFIGS} sampled configs x seeds {STAGE1_SEEDS} x {STAGE1_EPISODES} episodes. Stage 2: top {STAGE2_TOP} x seeds {STAGE2_SEEDS} x {STAGE2_EPISODES} episodes. "
                           "Score = medium-load validation completion of the selected checkpoint. Test seeds untouched.\n",
                           "| Arm | Stage-1 top-2 (idx: score) | Stage-2 scores | Selected | Overrides |", "|---|---|---|---|---|"]
    for arm, rows in s1.items():
        top = sorted([r for r in rows if np.isfinite(r["score"])], key=lambda r: -r["score"])[:STAGE2_TOP]
        s2 = {}
        for r in top:
            vals = [_score(_rid(2, arm, r["i"], s)) for s in STAGE2_SEEDS]
            s2[r["i"]] = float(np.mean(vals)) if all(np.isfinite(vals)) else math.nan
        fin = {i: v for i, v in s2.items() if np.isfinite(v)}
        if not fin:
            lines.append(f"| {arm} | {[(r['i'], round(r['score'], 4)) for r in top]} | pending | — | — |")
            continue
        best_i = max(fin, key=fin.get)
        best_cfg = configs[arm][best_i]
        selected[arm] = best_cfg
        lines.append(f"| {arm} | {[(r['i'], round(r['score'], 4)) for r in top]} | "
                     f"{ {i: round(v, 4) for i, v in s2.items()} } | c{best_i:02d} | `{json.dumps(best_cfg)}` |")
    for arm in ARMS:
        if not space_of(arm):
            selected[arm] = {}
            lines.append(f"| {arm} | n/a (no tunable parameters) | — | default | `{{}}` |")
    (TUNE / "selected.json").write_text(json.dumps(selected, indent=1))
    (ROOT / "TUNING_RESULT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    {"sample": cmd_sample, "enqueue1": cmd_enqueue1, "score1": cmd_score1,
     "select": cmd_select}[sys.argv[1]]()
