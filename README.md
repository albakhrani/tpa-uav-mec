# TPA-UAV-MEC

Code and results of the paper

> A. A. AL-Bakhrani, Z. Xu, M. S. Obaidat, M. Li, Q. Xia, Z. Li, and Z. Feng, "TPA-UAV-MEC: Time-Scale
> Separation in Memory-Enhanced Deep Reinforcement Learning for UAV-Assisted Mobile Edge Computing."

The analysis plan was registered before any test result was read. Its summary is deposited at
[doi:10.5281/zenodo.24123945](https://doi.org/10.5281/zenodo.24123945), and `docs/registration_summary.pdf` is a
copy of that summary. Four files were registered by their SHA-256 on 2026-11-10 at 23:03 UTC (2026-11-11, 02:03
UTC+3):

| Registered file | SHA-256 |
|---|---|
| the analysis plan (`ANALYSIS_PLAN.md`) | `00c9bec2f228032abeaabcd1b3b4f4a23fa3db7415f0256a36e915b95805bbf2` |
| the specification (`SPEC.md`) | `d8ea2270d3cecca14e7c1e0d9fcecbd00ce843bd94b47989822e24f658a92120` |
| `config/default.toml` | `dabb88425d2bdbd325e3cbae20b6c8c852259a76152657f6f68077b8676b44a3` |
| `scripts/analyze_campaign.py` | `47ed523f7f5a21b2ba7b59143cf25f477739a72461fa5bfb6d51c11e67c3eb85` |

The deposited summary says that the full plan and its dated amendments will be released with the code. They are
published here as `docs/analysis_plan.md`.

It is an edited copy. Internal role names, document names, decision and review codes, machine names and paths
were removed or replaced, and some labels and test file names were renamed. The copy opens with a note that lists
the edits and states what did not change. The registered files themselves are available from the corresponding
author on request.

`config/default.toml` and `scripts/analyze_campaign.py` are in this repository with edited comments and docstrings.
Their values and code statements are as registered, but their SHA-256 differs from the registered values.

## What the paper asks

Demand on UAV edge servers recurs over periods far longer than a controller's observation window. The paper asks
when a long-term memory can add information about future demand that the controller's recent observation window
lacks. It bounds the gain of any long-term memory over a window-based forecast by the *predictive gap*, the share of
intensity variance that only the past beyond the window explains. It then tests the bound with TPA-UAV-MEC (TPA), a
soft actor-critic controller for UAVs at fixed hover with a short-term store and an auditable long-term store that is
written only when a significance gate opens.

Its hypotheses, seeds, tests and the 1 pp equivalence margin were fixed in a dated analysis plan before any test
result was read. The evaluation compares TPA with:

- ablations;
- a capacity-matched single store (A3);
- memoryless and recurrent controllers;
- a phase table that knows the true period (A12);
- four published methods, restricted to fixed hover.

The workloads are sinusoidal demand (the main campaign), Markov-modulated (MMPP) arrivals, Gauss-Markov device mobility
and heavy-tailed on-off arrivals.

## What the code does

| Path | Content |
|---|---|
| `tpa/env/` | Slot-level simulator: UAVs at fixed hover over a square area, IoT devices in demand zones, FDMA uplink with fading and interference, task queues, energy, and the workloads (sinusoidal, MMPP, Pareto on-off, Gauss-Markov mobility). |
| `tpa/memory/` | TPA's memory: short-term store, pattern extraction and its significance, consolidation into the long-term store under arrival keys, retrieval, and the reconstruction of retrieved content for replayed transitions. |
| `tpa/learner/` | Soft actor-critic: networks, objectives, replay buffer. |
| `tpa/drm/` | Resource manager: task priority, choice of UAV by utility, admission, CPU allocation, burst mode. |
| `tpa/baselines/` | The published methods B_ZHOU and B_QIN (rules), B_HUANG (DDPG) and B_SONG (evolution strategies), restricted to fixed hover, and the recurrent (A10), frame-stacking (A11) and phase-table (A12) controllers. |
| `tpa/arms.py` | Every controller of the paper as a set of flags: A0 (TPA), A1 to A16, A16P (A16′) and the four baselines. |
| `tpa/control.py`, `tpa/factory.py` | TPA's per-slot control loop and the construction of any controller. |
| `tpa/run/` | Trainer, evaluator, checkpoints, Parquet logs, run manifests (with code and configuration hashes) and the SQLite run queue. |
| `tpa/stats/` | Paired and Welch t-tests, Holm correction, effect sizes (Cohen's d and d_z) and the per-seed trial table. The equivalence test (two one-sided tests at ±1 pp) is in `scripts/analyze_campaign.py`. |
| `config/default.toml` | Every parameter of the model, the controllers and training, with its default value, except the override-only `memory.delay_slots` of A16 and A16′ (default 300 in `tpa/memory/stores.py`). The reported runs override the scenario (`runs/d3/adopted.json`), the tuned values (`runs/tune/selected.json`) and the episode count, and the rows with a memory delay of 25 ms also `t_mem_ms.A0`. |
| `runs/d3/adopted.json` | The scenario of every reported run: 25 UAVs, the ranges of task sizes and deadlines, and the light, medium and heavy loads 0.17, 0.374 and 0.578. |
| `runs/tune/configs.json`, `runs/tune/selected.json` | The twelve hyperparameter configurations sampled for each tunable controller, of which the first six entered tuning, and the configuration selected on validation seeds that every main and robustness row used (A15 and the rows with a memory delay of 25 ms used that of A0). |
| `scripts/tuning.py` | The search space of each controller, hyperparameter sampling, the two tuning stages on validation seeds, and the selection. |
| `scripts/enqueue.py`, `scripts/enqueue_main.py`, `scripts/enqueue_main_extra.py`, `scripts/enqueue_robust.py` | Row definitions: rows of given controllers and seeds on one workload with one set of overrides (A15 excluded); the 260 main rows; the 56 further main rows; the 72 rows on the other workloads. |
| `scripts/run_queue.py`, `scripts/cpu_topology.py` | The launcher of the worker processes, which claim rows from the queue and train and test them, each pinned to one logical CPU; and the CPU map it reads, built from a Windows API. |
| `scripts/b1_eval.py` | The evaluation-only sensitivity rows (file prefix `b1`): the selected A0 policy of each test seed, evaluated again with one setting changed at a time. |
| `scripts/analyze_campaign.py`, `scripts/a16_partA.py` | The registered analysis, which wrote `results/`. The second file holds the statistics of its per-cycle analysis and of P10; its name does not refer to the variant A16. |
| `figures/src/` | Scripts that draw the result figures of the paper from `results/`. |
| `results/` | Outputs of the registered analysis, described below. |
| `tests/` | Test suite. |
| `docs/` | A copy of the deposited summary of the registered plan, and an edited copy of the registered analysis plan (with its dated amendments). |

## Install

Python 3.12. The campaign, its analysis and the evaluation-only sensitivity rows ran on Python 3.12.16 under Ubuntu
22.04. The robustness rows, their analysis, the figure rebuild and the check described at the end of this file ran on
Python 3.12.10 under Windows 11.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

`requirements-lock.txt` pins the versions of the Windows environment; its torch entry is the CUDA 12.6 build of torch
2.14.1. The campaign machine used the CPU build of the same version (2.14.1+cpu, as its run manifests record) with the
same NumPy 2.5.4 and SciPy 1.18.1. Every reported run trained and was tested on the CPU, so either build serves.

Run the tests from the repository root:

```bash
python -m pytest -q
```

On a Windows 11 laptop with Python 3.12.10 and a CUDA GPU, the suite gave 657 passed and 199 skipped tests, with none
failed. The skipped tests are:

- parameter combinations of the memory tests that other cells already cover (197);
- a test that needs the queue file `runs/runs.db` with the campaign's tuning rows;
- a test of the path in which the trainer is missing.

The tests marked for CUDA run only where a GPU is available. Tests that start worker processes need at least 40 GB
of free disk space where pytest puts its temporary files (choose the place with `--basetemp`).

## Results

`results/main/` holds the registered analysis of the main campaign: 316 rows on the sinusoidal workload, with every
controller on 30, 8 or 6 test seeds. `results/robust/` holds the analysis of the other three workloads: 72 rows, A0,
A3, A9 and A10 on six test seeds per workload.

Each analysis ran once, after the last run, on the run records:

- `scripts/analyze_campaign.py --phase main --out DIR`;
- `scripts/analyze_campaign.py --phase robust --out DIR`, with `--db` pointing at the queue file of the robustness rows
  and `--runs-root` at the folder that holds their run directories.

The run records themselves are not part of this repository: training logs with one row per episode, test tables with
one row per test run and one per minute, and checkpoints.

Fractions are stored as fractions, except the values whose names end in `_pp`, which are in percentage points
(`mde_pp` in `contrasts.csv`, `mean_pp` of the interaction in `summary.json`). The paper prints completion, shares and
their differences in percent and percentage points (pp). The loads 0.17, 0.374 and 0.578 are the light, medium and
heavy loads.

| File | Content |
|---|---|
| `trials.csv` | One row per controller, test seed, workload and load: every metric, averaged over the three test runs of that seed. These are the per-seed tables behind every contrast. |
| `contrasts.csv` | The tested paired contrasts. In `main/`: H1 and the families S1 to S4 (29 rows); the A2 − A1 contrast, the S4 interaction (in `summary.json`) and the comparisons of `tierc.csv` are not in it. In `robust/`: A0 − A3, A0 − A9 and A0 − A10 per workload. Columns: family, controllers, workload, load, n, mean difference, 95 % interval, t, p, Holm-adjusted p, effect sizes, the equivalence test at ±1 pp (registered for H1, computed for every row), and normality and Wilcoxon checks. |
| `descriptive.csv` | Mean and 95 % interval half-width over seeds, per controller, workload and load, for 19 metrics: `completion`, `resource_eff`, `resource_eff_cb`, `cpu_util`, `energy_eff`, `hit_rate`, `wait_s`, `trans_s`, `comp_s`, `pred_bias`, `e_uplink_j`, `e_downlink_j`, `e_hover_j`, `e_comp_j`, `e_mem_j`, `lt_occupancy`, `cons_rate`, `mem_util`, `sigma_mean`. Counts and wall-clock columns are only in `trials.csv`. |
| `tierc.csv` | A0 against A13, A15 and A0 with a memory delay of 25 ms (`A0-tmem25`) at every load: mean differences, 95 % intervals and effect sizes, and the paired t-test of A0 − A0-tmem25 at heavy load. Empty in `robust/`. |
| `exploratory.csv` | Descriptive comparisons outside the registered families: A8 and A9 against each baseline at heavy load. Empty in `robust/`. |
| `per_cycle_runs.csv`, `per_cycle_summary.csv`, `per_cycle_contrasts.csv` (main only) | Per demand cycle on the sinusoid, for the 13 memory variants: completion, gate-open share and long-term occupancy per run, load, test run and cycle; their means per controller, load and cycle; and A0 − A3 per cycle, over cycles 2-3 and over the whole run. |
| `p6_p9.json` (main only) | Predictions P6 (long-term occupancy) and P9 (gate-open fraction), and the phase imbalance of each seed. |
| `b1_sensitivity.csv`, `b1_runs.csv`, `b1_reproduction.csv` (main only) | Sensitivity of the selected A0 policies: per setting, per row, and the check that the default rows reproduce the first heavy-load test run of the main campaign. |
| `p10.json` (robust only) | Prediction P10 on MMPP: why it could not be evaluated from the records (no per-slot gate decision and no MMPP state are written), and the gate-open fraction by load of A0 and A3 (descriptive). |
| `summary.json` | Row counts, loads, workloads, seeds per controller (`n_by_arm`), the code hash of the runs, the contrasts with the S4 interaction, the tier-C and exploratory comparisons, the per-cycle contrasts with P6 and P9, P10, and the sensitivity table with its reproduction summary. It also names the analysis figures, which are not part of this repository. |
| `TABLES.md` | A readable summary of the files above: H1, the contrasts, tier C, the exploratory comparisons, eight of the descriptive metrics, the per-cycle tables with P6 and P9, P10 and the sensitivity table. For publication, a few headings and notes were reworded, and references to the dates of plan decisions and to sections of the analysis plan were removed (the generation time in the first lines is kept), here and in the note `p10_statement` of `robust/p10.json` and `robust/summary.json`. Every result value is as the analysis wrote it. |

### Tables of the paper

| Paper | Caption | Source |
|---|---|---|
| Table I | Positioning of TPA against the forms of memory in DRL | Literature; no data. |
| Table II | Derived constants against measured diagnostics | Closed-form values and diagnostics on validation seeds recorded before the campaign; not in `results/`. |
| Table III | Predictive gap and window share at w = 100 | Computed from the workload models; not in `results/`. |
| Table IV | Controllers | Design. The n column is `n_by_arm` of `main/summary.json`, except for A16 and A16′, which have no rows in `results/`. |
| Table V | Cost and energy | Per-module times: timing runs on one pinned core, not in `results/`. Energies per completed task at heavy load: `main/trials.csv` (`e_comp_j`, `e_mem_j`, `e_uplink_j`, `e_downlink_j` over `completed`, mean over seeds). Energy efficiency: `energy_eff` of `main/descriptive.csv`. |
| Table S1 | Constants derived in Section S-I | Closed form. |
| Table S2 | Registered predictions | Verdicts: P1 `contrasts.csv` (H1) and `per_cycle_contrasts.csv`; P2 and P5 `contrasts.csv`; P3 `trials.csv` (A2 − A1); P4 `interaction` of `summary.json` with its interval from `trials.csv`; P6 and P9 `p6_p9.json`; P7 `robust/trials.csv` (`lt_occupancy`); P8 `descriptive.csv` (`hit_rate`); P10 `robust/p10.json`; P11 to P14 `trials.csv` (`t_*_ms`, `n_expired`, `failed`, `assigned`, `slots`, `burst_share`). Predictions: section 4 of `docs/registration_summary.pdf`, word for word. In `docs/analysis_plan.md`, P1 to P14 are listed under the heading "Pre-registered predictions", and the file's opening note says where P15 and P16 are. |
| Table S3 | Dated design changes | Registration record; no data. The No. column gives the amendment number in `docs/analysis_plan.md`. |
| Table S4 | Tuning spaces and selected configurations | Spaces: `space_of` in `scripts/tuning.py`. Selected configurations: `runs/tune/selected.json`. |
| Table S5 | Validation completion of the finalists | Tuning records on validation seeds; not in this repository. |
| Table S6 | Every registered contrast at every load | `contrasts.csv`, `tierc.csv`, `per_cycle_contrasts.csv` (A0 − A3 at light and medium load, over the whole run) and `interaction` of `summary.json` (n, mean, t, p). The other light- and medium-load cells, the A2 − A1 row, the interval and d_z of the interaction and the equivalence test of the 25 ms row are computed from `trials.csv`. |
| Table S7 | Descriptive contrasts, not registered | `exploratory.csv` at heavy load; light and medium load from `trials.csv`. |
| Table S8 | Other workloads at heavy load | `robust/contrasts.csv`. |
| Table S9 | Memory diagnostics by cycle | `per_cycle_summary.csv` (cycle 1), `p6_p9.json` (cycles 2-3), `descriptive.csv` (`lt_occupancy`, `cons_rate`, `hit_rate`), and `robust/descriptive.csv` for the other workloads. |
| Table S10 | Failure classes at heavy load | `trials.csv`: `n_expired`, `n_overflow` and `late` over `failed`. |
| Table S11 | Energy, time shares, and timing of TPA | `trials.csv` (energies per completed task, `t_*_ms`, `burst_share`, `e_burst_j`) and `descriptive.csv` (`wait_s`, `trans_s`, `comp_s`, `pred_bias`). |
| Table S12 | Sensitivity of the trained A0 policies at heavy load | `b1_sensitivity.csv`. |
| Tables S13, S14 | Matched settings of all comparisons; fidelity of the baselines | Protocol and implementation; no data. |
| Table S15 | Link budget on the calibration worlds | Channel model on validation seeds 9001 to 9003; not in `results/`. |
| Table S16 | Parameters, values, and provenance | `config/default.toml` and `runs/d3/adopted.json`; the episode count is the override `run.episodes = 500` of every row. |

### Figures of the paper

The figure scripts read `results/` and write PDF files. Their file names, and the figure titles they write into the
PDF files, keep the numbering of an earlier draft (`figS_seed.py` carries no number). The comment after each command
gives the figure of the paper.

```bash
mkdir -p figures/out
python figures/src/fig7_paired.py results/main figures/out/fig4.pdf                                  # Fig. 4
python figures/src/fig4_forest.py results/main figures/out/fig5.pdf --final                          # Fig. 5
python figures/src/fig5_load.py results/main figures/out/fig6.pdf --final                            # Fig. 6
python figures/src/fig8_fate.py results/main figures/out/fig7.pdf                                    # Fig. 7
python figures/src/fig9_mechanism_on_test_seeds.py results/main figures/out/fig8.pdf                 # Fig. 8
python figures/src/fig6_workloads.py results/main figures/out/figS1.pdf --robust results/robust --final   # Fig. S1
python figures/src/figS_seed.py results/main figures/out/figS3.pdf                                   # Fig. S3
python figures/src/fig6_workloads_diff.py results/main results/robust figures/out/figS4.pdf          # Fig. S4
```

On Windows, set `PYTHONIOENCODING=utf-8` first, because the forest script prints a minus sign.

Rebuilt this way, all eight figures have the same text and numbers as the figures of the paper. Figs. 6, S1 and S4
are also identical pixel for pixel; the other five show the same marks and text, placed a few pixels differently by
the layout.

The other figures do not come from these scripts:

- Figs. 1 and 2 (scenario and architecture) are drawings.
- Fig. 3 shows closed-form bounds and diagnostics on validation seeds recorded before the campaign.
- Fig. 9 shows post hoc controls run after the campaign, outside the dated analysis plan; its tables are not part
  of this repository.
- Fig. S2 is the per-minute plot that `scripts/analyze_campaign.py` draws (`fig_minute_series_heavy.png`, shown with
  its title strip cropped) from the per-minute test tables, which are not included.

## Rerun one row

A row is one controller trained on one seed and then tested. The following lines rerun the campaign row
`main-A0-s1-sinusoidal` (TPA, test seed 1) on any operating system:

```python
import json, sys
sys.path[:0] = [".", "scripts"]
import tuning
from tpa.config import load_config
from tpa.run import trainer

selected = json.load(open("runs/tune/selected.json"))
overrides = tuning.row_overrides(selected["A0"], 500)   # scenario + A0's selected hyperparameters + 500 episodes
cfg = load_config(overrides=overrides)
row = dict(run_id="main-A0-s1-sinusoidal", arm="A0", seed=1, workload="sinusoidal", phase="main",
           run_dir="runs/main/main-A0-s1-sinusoidal", device="cpu")
print(trainer.run(row, cfg))
```

First set `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS` and
`VECLIB_MAXIMUM_THREADS` to 1, as the campaign's worker processes did.

The run writes its manifest, the training and test tables (Parquet), the best and final checkpoints and `RESULT.json`
into `run_dir`. Its configuration hash, `cfg.hash()`, is 7444a215b32c, the hash that the campaign recorded for this
row. Phase `main` tests the checkpoint with the best validation score on the seed's test block; phase `tune`
evaluates it on the validation block instead.

Other rows differ in:

- `run_id` and `run_dir`;
- `arm` and `seed`;
- `workload`: `mmpp`, `pareto` and `sinusoidal|mobility=gauss_markov`, all with phase `robust`;
- the selected configuration of the arm. A15 uses A0's selected configuration, and the rows with a memory delay of
  25 ms add `t_mem_ms.A0 = 25.0` to it.

The campaign itself ran through the SQLite queue:

```bash
python scripts/enqueue.py --db runs/example.db --phase main --arms A0 --seeds 1 --workload sinusoidal \
    --overrides learner.lr=0.00239 learner.update_every=8 learner.target_entropy_scale=0.25 burst.theta_b=0.9 run.episodes=500
python scripts/run_queue.py --db runs/example.db --workers 1 --phase main --max-runs 1
```

The first command enqueues the same row (configuration hash 7444a215b32c) into the queue file `runs/example.db`. The
second starts one worker process that trains and tests it under `runs/main/`. The examples leave the default queue
file, `runs/runs.db`, alone, because `tests/test_tuning_arms.py` checks it for the campaign's tuning rows whenever it
exists.

Some details of the queue:

- `python scripts/enqueue_main.py --episodes 500 --dry-run` prints, for each of the 19 controllers of the 260 main
  rows, its seeds, row count, configuration hash and selected configuration.
  `python scripts/enqueue_robust.py --episodes 500 --selected runs/tune/selected.json --db runs/robust/runs_robust.db --dry-run`
  does the same for the 48 MMPP and Gauss-Markov rows, and with `--family pareto` for the 24 on-off rows.
- `scripts/enqueue_main_extra.py` defines the other 56 main rows: A5 and A7 on seeds 9 to 30, and A15 and A0 with a
  memory delay of 25 ms on seeds 1 to 6, the last two with A0's selected configuration.
  - It refuses to run, even with `--dry-run`, unless the code hash starts with its `EXPECT_CODE`, the campaign's
    2892281e5ba9 (see below). It also opens its queue file (`--db`, default `runs/runs.db`) read-only, also with
    `--dry-run`, and stops if that file does not exist. To use it with this repository, set `EXPECT_CODE` to this
    repository's code hash and pass `--db` with an existing queue file, such as `runs/example.db`.
  - `scripts/enqueue.py` cannot replace it: it refuses A15, and it would give the rows with a memory delay the run ids
    and run folders of the plain A0 rows.
  - These rows can also run directly, as shown above: with `arm="A15"`, or with the override `t_mem_ms.A0 = 25.0` and
    a run id and run folder of their own.
- A worker process marks a row as failed when its configuration hash or code hash differs from that of the worker's
  tree (`--allow-code-change` skips the code check).
- Before each claim, a worker process also stops in either of two cases:
  - the disk that holds `runs/` has less than 40 GB free (`run.disk_min_free_gb`);
  - the projected size of all rows of the queue file exceeds 100 GB (`run.disk_budget_gb`). The projection takes the
    measured sizes of finished rows, their mean for pending and running rows, and the rolling checkpoints of running
    rows.
- The launcher pins worker processes using `runs/cpu_topology.json`, which `scripts/cpu_topology.py` builds on Windows.
  On other systems, write that file yourself or run rows directly as shown above.
- The launcher of `scripts/b1_eval.py` was written for the campaign machine and runs only on Linux: it reads `/proc`
  and calls `os.statvfs`. It pins its worker processes to a fixed CPU set (`TPA_CPUS`), and it counts worker
  processes by the names of their code directories (`TPA_TREES`). On another Linux machine, adapt these two constants
  before using `--launch`.

## Expected run times

Measured on the machines of the campaign:

- **One main row (500 episodes).** Training took a median of 8.3 h on one core of the campaign machine, an Intel Xeon
  Gold 5218R virtual machine running 20 to 30 rows in parallel.
  - The median per controller ranged from 2.9 h (B_SONG) to 18.2 h (A4), and single rows took 1.9 h to 20.5 h.
  - These figures cover the 302 rows that trained in one attempt. For the 14 rows resumed from a checkpoint, the
    recorded training time (`train_wall_s`) covers only the part after the resume.
  - The test of the selected checkpoint comes on top.
  - The recorded training times of the 316 main rows add up to about 2,780 core-hours, without the parts before those
    resumes.
- **One robustness row (500 episodes, training and test).** A median of 4.9 h to 6.9 h per controller on a laptop
  with an Intel Core i9-13900HX running six rows in parallel.
- **The check** described under "Relation to the code of the reported runs" (A0, 20 episodes, heavy load only,
  training and test): 30 to 35 minutes in a single-threaded process on the same laptop.
- **The test suite:** 2 h 28 min in one process with single-threaded numerical libraries, while another heavy
  computation shared the laptop.

## Seeds

- **Test seeds.** 1 to 30 in the main campaign: A0, A3, A5, A7, A9, A10 and A12 run on 30 seeds; the other variants
  and the baselines on 8; A13, A15 and A0 with a memory delay of 25 ms on 6. Each other workload uses seeds 1 to 6.
- **Validation seeds.**
  - 9004: tuning stage 1 (six configurations per tunable controller, 300 episodes).
  - 9005: stage 2 (the best two, 1,000 episodes).
  - 9001 to 9003: the load calibration and the mechanism checks before the campaign; 9001 also served the timing runs.
- **Random streams.** Within a seed, each of the following comes from a separate random stream:
  - the world;
  - the arrivals together with the attributes of each task;
  - the fading;
  - the device mobility.

  The stream key is `SeedSequence([seed, stream, 1000 * block + run])`, where `run` numbers the evaluation runs of a
  block and the world stream uses 0 in every block (`tpa/rng.py`). The amount drawn from each stream per slot does not
  depend on the controller, so every controller sees the same world, arrivals, fading and mobility. The one exception
  is B_SONG, whose antithetic pairs of training episodes run twice on the same window, so it trains on half as many
  distinct windows. Training, validation and test use separate blocks of the same seed's world.
- **Hyperparameter sampling.** The sampler uses a fixed seed per controller (`scripts/tuning.py`).
- **Reproducibility.** Runs on the CPU are bitwise reproducible on a given machine and library build, also when
  resumed from a checkpoint.

## Relation to the code of the reported runs

Every row in `results/` recorded the code hash 2892281e5ba9 (`code_hashes` in both `summary.json` files). The code
hash is the SHA-256 over every `tpa/**/*.py` file and `config/default.toml`, taken in the order of their relative
paths, each entered as its relative path, its size in bytes and its raw bytes, line endings included (`code_hash` in
`tpa/run/manifest.py`). The code hash of this repository is 2966789c94d8. `.gitattributes`
turns off Git's line-ending conversion, so a clone keeps the bytes and the hash.

This repository was copied from a later state of the campaign code. It leaves out 31 scripts of that later state (29
in Python, 2 in PowerShell): diagnostics, the load calibration, mechanism checks, checks of the tuning results, a
restart of the tuning rows, profiling, timing and power measurements, and reports. All of them are outside the code
hash, and none of them runs a row of `results/`. Apart from these scripts, this repository differs from the campaign
code (2892281e5ba9) in two ways:

1. Comments and docstrings were edited for publication, to remove internal working notes, and a few files were
   renamed. For every Python file outside `tests/`, the abstract syntax tree with docstrings set aside is identical to
   that of the file it was copied from, and `config/default.toml` holds the same values. In `tests/` the same holds,
   except that eleven test functions were renamed and seven printed or skip messages were reworded.
2. Like the later state it was copied from, this repository also holds the forecasting variants A16 and A16′, which
   the campaign code lacks. No completion or test result of these two variants is reported: Section S-V of the
   supplement describes the mechanism check that A16 failed, and A16′ was stopped before its check was computed.
   - Their code sits in `tpa/arms.py`, `tpa/config.py`, `tpa/control.py`, `tpa/memory/` and `scripts/tuning.py`.
   - It changes what is computed only under the flags of these two arms.
   - For every other arm it adds an event field (`delay_src`) that stays −1 and enters no computation, and
     `tpa/config.py` accepts the override-only key `memory.delay_slots` in any row.
   - The registered analysis, `scripts/analyze_campaign.py`, ran from this later state (code hash 790391a95b5d).

**The check.** Both points were checked with one row: A0, test seed 1, 20 episodes, heavy load only
(`workload.loads = [0.578]`, configuration hash 0671da0ff68f). It ran in single-threaded processes on the laptop,
three times: with this repository, with the campaign code (2892281e5ba9), and with the later state (790391a95b5d).
Every stored value is identical bit for bit across the three runs, in:

- the last rolling checkpoint (networks, optimizer states, memory stores, replay buffer and every random stream),
  which the check copied before the trainer deletes it at the end of the run;
- the best and final checkpoints;
- the training table (20 rows; its 52 columns other than the 6 timing columns) and the test table (543 rows; its 51
  columns other than the 8 timing columns);
- `RESULT.json`.

Only wall-clock values were left out of the comparison: the timing columns of the two tables (`wall_s`, `t_*_ms`,
`reaction_ms_p50`, `reaction_ms_p90`, `recovery_ms`) and `train_wall_s` in `RESULT.json`. In the run manifest, only
the code hash, the time, the process id, the output folder and the comment text of `config/default.toml` differ.

**Numbering in comments.** Equation and algorithm numbers cited in the comments of `tpa/`, `scripts/` and `tests/`
follow the numbering of the first submitted version of the manuscript, not that of the paper. In the figure scripts,
table and proposition numbers (Table III, Proposition 2) are those of the paper, and figure numbers those of the
earlier draft named above, except "supplement Fig. S1" in `fig5_load.py`, `fig6_workloads.py` and
`build_results_figs.sh`, which is the paper's number.

## License and citation

MIT License (`LICENSE`). Please cite the paper; `CITATION.cff` holds the reference.
