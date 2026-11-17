# Analysis plan: TPA-UAV-MEC revision campaign

> **Edited copy of the registered plan.** This is the study's analysis plan as registered on 2026-11-10 at 23:03 UTC
> (2026-11-11, 02:03 UTC+3), edited for publication. The registered file is `ANALYSIS_PLAN.md`, SHA-256
> `00c9bec2f228032abeaabcd1b3b4f4a23fa3db7415f0256a36e915b95805bbf2`. The deposited summary is
> [doi:10.5281/zenodo.24123945](https://doi.org/10.5281/zenodo.24123945). The plan's last section, A16B1.7, carries
> the local date 2026-11-11 (01:55 UTC+3, that is 22:55 UTC on 2026-11-10).
>
> **What the edits removed or replaced.** Internal names, codes and working notes:
> - **Roles.** The project's internal role names. Where the record depends on who knew, decided or checked
>   something, the roles are named neutrally: "the implementer" (or "the memory implementer"), "the decision-maker",
>   "a second reviewer".
> - **Documents and codes.** The names of internal documents and folders, and review codes. The two registered
>   documents are called "this plan" and "the specification". A line reference into the registered specification now
>   gives its section and row.
> - **Decision codes.** Each internal decision code became one of these:
>   - the plan's own amendment or section number;
>   - the date of the decision;
>   - a short description, for example "the DRM speed-up decision".
>
>   A code was dropped where the heading already gives the section.
> - **Review steps.** The project's names for its review steps, now "the stage-1 readout", "the stage-2 readout", "the
>   early-campaign check" and "the registered analysis".
> - **Step and record labels.**
>   - labels of build, deployment and decision steps;
>   - the labels of the three diagnostics of A16's M4 failure, now the baseline, coverage and decoder checks;
>   - two labels of the timing report;
>   - a few labels of internal records.
> - **Machines.** Host names, home-folder and server paths, process ids, pid files and terminal-session names. "Lab
>   server" reads "compute server". A code tree is named by the amendment that built it or by its code hash.
> - **Working notes.** Remarks on how the record was kept, such as the name of a background application during a
>   timing probe, and one remark in §7 on the project's earlier records.
> - **Manuscript references.** Line references to the manuscript of the first submission, which now read "the first
>   submission".
> - **One table column.** In the table of A3.4, the column that named the team role behind each row. The independent
>   reviewer's row now opens with "Independent code review".
>
> **Renamed labels.** Some labels were renamed to keep them apart from the predictions P1–P16 and from the arms:
> - the pilot decisions P1–P5 read "pilot decisions 1–5";
> - the proofs P1–P4 of A9.3 read "proofs 1–4";
> - the items A1–A4 of Amendment 16, Part A read "items 1–4";
> - in the plan's own text, the rows added by Amendment 13 read "Amendment 13 rows". The quoted decisions keep "A13
>   rows".
>
> Test files, test functions and one script that were renamed for this repository are cited by their names here.
>
> **What did not change.** Every hypothesis, test, threshold, sample size, seed, prediction, rule, date, clock time,
> code hash and result reads as registered, with these exceptions:
> - Dates and code hashes were added only where they replace a decision code or a code-tree name.
> - Two time stamps that were part of removed log-file names (A3.4, A11.4) went with those names.
>
> The amendment numbers are those cited in the deposited summary. The summary's predictions P15 and P16, which the
> plan states without labels, are the A16 prediction of A14.4 and the A16′ contrast of A15.4.
>
> **Times.** A marked clock time carries one of these marks:
> - "UTC" (also "Z" or "+00:00");
> - "server clock": the compute server's clock, about UTC, as A15.8 says;
> - "local" or "UTC+3": local time is UTC+3 (A6.3: 04:33 local = 01:33 UTC). Where the registered file named this time
>   zone by a city, this copy writes "UTC+3".
>
> Times without a mark are as registered. In Amendments 1 to 5 they are local time. The evidence: A3.4 marks 14:23 as
> local, and only in local time do the runs of A4.5 and A5.4 come before the tuning launch of A6.3.
>
> **Verbatim sections.** Where a decision or a rule is quoted verbatim, editorial changes are in square brackets. The
> records that follow the quotes in A16A.8 and A16B1.7 are edited like the rest of the plan. In the table of A14.5,
> the vertical bars of |Δ| are escaped so that the table renders. In the registered file they split the cells.
>
> **Slips kept as registered.** These passages keep three slips of the registered text. Other words on their lines
> were edited.
> - Under the table of A14.5, the note on P2 says that tier B is n = 8 "since Amendment 7". Amendment 7 set tier B to
>   n = 12 (A7.1), and n = 8 dates from Amendment 10 (A10.4).
> - The introduction of Amendment 13, A13.1 and A13.4 count 68 added rows, and A13.4 splits them as 44 + 18 + 6. A15
>   runs on 6 seeds (A13.2), so 56 rows were added (44 + 6 + 6). That is the count `scripts/enqueue_main_extra.py`
>   enforces.
> - A14.4 says the S1 family would have "14 members instead of 10". A13 had left S1 (A7.1), so the counts were 13 and
>   9; A16A.5 gives the 9. The A16 contrasts never entered the analysis (A15.8).
>
> **Hashes.** Because of the edits, this file reproduces none of these hashes:
> - the registered SHA-256;
> - the hash of an earlier version of the plan recorded in A7.6;
> - the hash in the superseded hash file of 2026-11-10 19:07 UTC (A16B1.7).
>
> The registered file is available from the corresponding author on request.
>
> **Mechanism criteria.** M1–M4 name the mechanism criteria. They were computed without task outcomes:
> - M1 and M2 on validation seeds 9001–9003;
> - M3 on a synthetic periodic driver (driver seeds 0–2);
> - M4 of A14.2 on validation seed 9004 and on the driver. M4 as redefined in Amendment 15 was never computed for A16′
>   (A15.8).
>
> What each criterion measures:
> - M1: the gate-open share;
> - M2: the diversity of the long-term store (at least 100 distinct entries). It was re-specified on keys in Amendment
>   4, and as phase coverage and key structure in Amendment 5. A6.1 withdrew the coverage half;
> - M3: the retrieval criterion, redefined in A3.3 and re-specified in Amendments 4 and 5;
> - M4: the mechanism check of variant A16 (A14.2), redefined in Amendment 15.
>
> **File and code references.** Paths name files of the project's code tree. Files that are not in this repository,
> such as run records and diagnostic scripts, are not published here. Line references into code and configuration
> files (`file.py:N`, `config/default.toml` line N) refer to the files as they were when the passage was written.
> Later code changes, and the comment edits of this repository, shift some of these lines.

**Status: FROZEN 2026-10-27.** Pilot decisions 1–4 (§6 of the pilot report) were approved by the author on 2026-10-24; pilot decision 5 (exclusive timing) was completed on 2026-10-27 (§8). The plan is frozen before the first tuning or main-campaign run. After freezing, any change is an amendment with a date and a reason, and is reported in the paper.

## 1. Design

- **Configuration:** `config/default.toml` as frozen at campaign start. The config hash and code hash are recorded in every run manifest.
- **Channel and load calibration:** adopted per the adoption record:
  - h_min from R_min;
  - slot_assigned interference;
  - load on the binding resource;
  - per-UAV FDMA, H = 100 m.
- **Arms:**
  - A0–A9, A13, A14 (specification §6);
  - A10 (LSTM actor-critic, parameter-matched), A11 (frame-stacked memoryless), A12 (zone × phase lookup);
  - B_ZHOU, B_QIN, B_HUANG, B_SONG (static-hover restrictions).
- **Seeds:**
  - Test seeds 1…n. They are never used for tuning, piloting or checkpoint selection.
  - Validation seeds 9001–9006 (tuning) and pilot seeds 8001–8003 never enter the analysis.
- **Training:** 1000 episodes of 900 slots; memory segments of 3P with constant load, cycling light → medium → heavy. The checkpoint is selected on the same seed's validation block (specification §7).
- **Evaluation:**
  - The best checkpoint is frozen, with no gradient updates.
  - 3 test runs of 10,800 slots per load, at loads {0.25, 0.55, 0.85} (light, medium, heavy); stores start empty.
  - The trial unit is the mean of the 3 test runs.
- **Pairing:** common random numbers. Within a seed every arm sees identical arrivals, channels and world (tested in `tests/test_system.py`). All contrasts are therefore paired by seed.

## 2. Primary hypothesis (confirmatory, one test)

- **H1:** A0 (two-level memory) and A3 (capacity-matched single flat store) differ in heavy-load (0.85) task completion on the test block.
- **Test:** two-sided paired t-test on the n per-seed differences, α = 0.05.
- **Report:** mean difference with 95% t-CI, Cohen's d on the pooled SD (as in the manuscript), and d_z.
- **n = 30 (APPROVED, pilot decision 1)**, chosen from the pilot runtime: about 3 weeks for the whole campaign at n = 30 against about 4.5 weeks at n = 50. n is fixed by the compute budget, **not** by any pilot effect estimate.
  - **Indicative MDE:** pilot paired SD of A0 − A3 at heavy load = 0.0030 (n = 3, very uncertain), which gives MDE ≈ 0.16 pp at n = 30 and power 0.8.
- **Equivalence (secondary; APPROVED, pilot decision 4).** SESOI = 1.0 pp completion. A TOST equivalence test at α = 0.05 (two one-sided paired t-tests against ±1.0 pp) is reported for H1 and for every S1/S2 contrast. This lets "no meaningful difference" be concluded, as well as "difference".

## 3. Secondary contrasts (Holm-corrected within each family; never pooled across families)

| Family | Contrasts | Metric |
|---|---|---|
| S1 ablation | A0 vs A1, A2, A4, A5, A6, A7, A8, A9, A13, A14 (10) | heavy-load completion |
| S2 implicit vs explicit memory | A10 vs A6 (like-for-like, reactive DRM); A10 vs A0; A11 vs A9; A12 vs A0 (4) | heavy-load completion |
| S3 baselines | A0 vs B_ZHOU, B_QIN, B_HUANG, B_SONG × 3 loads (12) | completion |
| S4 load dependence | A0 vs A9 at each load (3), plus the interaction test: one-sample t of (diff at heavy − diff at light) | completion |

- **Paired tests everywhere.** Per-trial records now exist for every arm, including the baselines, so the manuscript's Welch fallback is not needed.
- **Effect sizes:** each contrast reports the mean difference, 95% CI, d (pooled SD), d_z, raw p and Holm p.

## 4. Descriptive outcomes (no inferential claims beyond CIs)

- **Rates and efficiencies:** resource efficiency (both composites, specification §6), CPU utilisation, energy efficiency (Gcycles/kJ), memory hit rate (defined arms only).
- **Times:** latency components (wait, uplink, compute), predictor bias.
- **Energy:** device uplink energy, UAV downlink energy, hover term.
- **Memory diagnostics:** long-term occupancy and consolidation rate.
- **Wall-clock (exclusive timing runs only, specification §8.6):** per-module times, reaction time, recovery latency.
- **Phase-resolved per-minute series** for the warm-up analysis.

## 5. Robustness workloads (exploratory; separate families, Holm within each)

- **Workloads:** MMPP, Pareto on-off, Gauss–Markov mobility, patrol circles, packet loss 0.05 and 0.10, and the shifted and burst families.
- **Arms:** A0, A3, A9, A10, A12 and the best baseline from S3 (chosen by S3's medium-load mean).
- **Contrasts:** A0 vs each, heavy load.
- **Seeds:** n_r = 10 (about 7 days in the pilot projection).

## 6. Assumptions, missing data, deviations

- **Normality:** Shapiro–Wilk on each paired difference vector is reported. If p < 0.01, the Wilcoxon signed-rank test is reported as a **sensitivity** analysis; the primary test is unchanged.
- **Crashes:** a crashed or interrupted run resumes from its checkpoint, which is bitwise reproducible on CPU and tested on GPU. A run that cannot complete after 3 attempts is reported as missing. Its contrasts use the complete pairs, and the reduced n is stated.
- **No exclusions by outcome.** No run is excluded or re-run because of its result.
- **No interim analysis** of test-block results. The analysis script runs once, after all main-campaign runs are done.
- **Device:** memory arms train with the CUDA learner and CPU rollout; the other learned arms on CPU. The device per arm is recorded. Results are reported as produced.
- **Tuning:** equal budget per arm on validation seeds (specification §7, two-stage); selected configurations are frozen before the campaign. APPROVED pilot-driven additions (pilot decisions 2, 3), at the same 12-configuration budget for every arm:
  - θ_p grid {0.15, 0.25, 0.40, 0.60} (pilot decision 2);
  - target-entropy scale {0.25, 0.5, 1.0} (pilot decision 3).

## 7. Disclosures (to appear in the revised manuscript)

- **This campaign replaces all numbers in the submitted manuscript.**
- **Channel and load:** the calibration choices of §1 and the pre-registered diagnostic, including its failure on C3. DRM has a UAV choice for about 5% of task-slots.
- **Model and specification deviations:** per-UAV FDMA (a model change); the specification §10–§11 deviations (SAC with twin critics, U = 4, h(t) dropped, fixed-key long-term addressing, and the others listed there).
- **Baselines:** they run without burst mode or Eq. 59, they admit by raising f while DRM refuses (found in code review), and B_SONG's paired-trace ES budget.
- **Hardware and GPU:** the GPU cuBLASLt workaround (`DISABLE_ADDMM_CUDA_LT=1`), the skipped GPU-DPC core (CPU 2), and the hardware row (Table VI).
- **Memory activity:** pilot memory diagnostics, i.e. gate-open rate, long-term occupancy and hit rate, are reported for every memory arm whatever the outcome.

## 8. Timing on the reference machine (exclusive runs, 2026-10-27)

Measured with `scripts/pilot_timing.py` on an otherwise idle machine (no other Python process; Windows Defender, WMI and a desktop application at 25–40% of one core each during the probe), one job at a time, pinned to one physical P-core (logical CPU 6), 1 thread, 2 episodes with updates active. Files: `runs/pilot/timing_exclusive.{log,json}`, `runs/pilot/timing_sweep.{log,json}`.

| Arm | CPU (ms/slot) | GPU learner (ms/slot) | Learner update (ms), CPU → GPU |
|---|---|---|---|
| A0 | 28.1 | 22.3 | 93.5 → 69.5 |
| A3 | 31.1 | 20.1 | 107.4 → 65.1 |
| A9 | 10.2 | 6.7 | 34.0 → 20.0 |
| A12 | 1.1 | n/a | n/a |

Decision path (actor + DRM + MEPR + retrieval): about 3.3 ms per slot for memory arms, 0.8 ms for A9.

**Worker sweep** (A0 on CPU, N concurrent workers each pinned to one physical core; GPU = N workers sharing the single GPU):

| N | CPU mean ms/slot | aggregate slots/s | | N | GPU mean ms/slot | aggregate slots/s |
|---|---|---|---|---|---|---|
| 4 | 34.6 | 116 | | 1 | 14.9 | 67 |
| 8 | 46.7 | 171 | | 2 | 19.6 | 102 |
| 12 | 69.1 | 174 | | 3 | 21.2 | 141 |
| 16 | 73.5 | 218 | | | | |

- CPU throughput saturates at about 8 workers; 12 adds nothing and 16 only slightly more (E-cores, memory bandwidth). **Campaign configuration: 8 CPU workers + 3 GPU workers**, which is about 170 + 140 = 310 slots/s aggregate.
- The GPU learner is 1.3–1.5× faster per slot in isolation and scales to 3 sharing workers; the memory arms (A0–A8, A13, A14) go to the GPU workers, the memoryless learned arms (A9, A10, A11, B_HUANG) to the CPU workers, and the non-learning arms (A12, B_ZHOU, B_QIN, B_SONG) to the CPU workers.
- **Projection at n = 30** (main campaign only; tuning and robustness scale similarly): memory arms 330 runs × 900k slots ÷ 141 slots/s ≈ 24 days on 3 GPU workers; this is the bottleneck. Options at campaign launch time: (a) accept ≈ 3.5 weeks main + ≈ 1.5 weeks tuning + ≈ 1 week robustness; (b) also run memory arms on the 8 CPU workers in parallel (≈ 46 ms/slot each → +170 slots/s, bringing the memory-arm block to ≈ 11 days). Option (b) is the default; results are identical by construction only on CPU (the GPU path is deterministic per test but not bitwise-identical to the CPU path), so the device per run is recorded and reported.

**All wall-clock figures in the paper** (reaction time, recovery latency, per-module overhead) come from exclusive timing runs only, as in specification §8.6, not from campaign runs.

FROZEN: 2026-10-27, code hash 9a0b331cc10d, config hash 0c9cb7897e41 (config/default.toml as of this date; the tuned per-arm overrides selected in the tuning stage are recorded in the queue rows and run manifests and do not alter this freeze).

## Amendment 1 (2026-10-27, during tuning stage 1; no effect on the analysis)

**Reason.** An unclean reboot at 04:18 killed the first tuning launch after 1.3 h: Windows event 41 reports BugcheckCode = 0 and LongPowerButtonPressDetected = true, i.e. a manual hard power-off by holding the power button, not a crash or power cut. No run data were lost beyond the minutes since each run's last rolling checkpoint (≤ 10 episodes). On relaunch, per-core benchmarks showed that this laptop's 16 E-cores are unusable for the workload under the Balanced power plan: alone they run ≈ 15× slower than a P-core (≈ 40 vs ≈ 600 iterations in a 3-s numpy benchmark), and with the P-cores busy they fall to ≈ 2% of P-core speed (8 vs 400). Two stage-1 runs pinned to E-cores made no progress in 4 h. The worker sweep of §8 did not reveal this because its "16 workers" figure mixed 8 fast P-core workers with 8 near-idle E-core workers.

**Change.** Workers are pinned only to the 16 P-core hyperthreads (launcher mode `p-ht`, logical CPUs 0–15, CPU 2 skipped for the GPU driver): 15 workers = 3 GPU + 12 CPU. Measured aggregate throughput on the P-core threads: 16 HT workers give ≈ 1.24× the throughput of 8 one-per-core workers (≈ 4080 vs ≈ 3300 benchmark iterations/s), each HT worker running at ≈ 62% of exclusive speed.

**Unchanged.** Everything in §1–§7; the per-run computation, seeds, configs, scoring and selection rule are untouched. All runs interrupted by the reboot resumed from their rolling checkpoints (bitwise-reproducible on CPU; deterministic on GPU). The §8 campaign projection is to be re-estimated from the tuning stage's measured throughput before the main campaign is launched.

## Amendment log entry (2026-10-27, ~06:30): tuning PAUSED by decision

The reports were reviewed and the main campaign was **not** approved. Tuning stage 1 was paused after about 1.5 h of the relaunch: all 15 workers stopped, 0 of 408 runs complete, 15 partially run (their rolling checkpoints and logs are kept under `runs/tune/`; status snapshot in `runs/tune/PAUSED_2026-10-27_status.txt`). The queue rows are back to `pending`. The code changes that were ordered (see the diagnosis report and a dated pre-registration note) will change the code hash, so the workers' hash check will refuse these rows and **no stage-1 score from this launch will be used**; tuning restarts from scratch after Amendment 2.

## Amendment 2 (2026-10-27, written after the instruction of ~06:30; tuning not restarted, main campaign not approved)

### A2.1 Plain statement of the pilot result before any of the changes below

In the pilot (the pilot report, 3 seeds × 100 episodes, code 9a0b331cc10d / config 0c9cb7897e41) **A0 was not better than the memoryless A9 at any load**: completion A0 0.368 / 0.336 / 0.316 vs A9 0.378 / 0.351 / 0.327 at light / medium / heavy; paired A0 − A9 = −0.011 (the pilot report, lines 62–72, 81). A0 − A3 (the primary contrast) was −0.0015. The memory gate was open on 98–99% of slots and the long-term store used 1–2 distinct slots. These numbers are the reason for the pause; nothing in this amendment was chosen with the aim of changing them.

### A2.2 Why the code and configuration changed (the diagnosis report and a dated pre-registration note)

| Finding (file) | Change | Where |
|---|---|---|
| The diagnosis (item a): 87–98% of failed tasks are unservable even on an empty UAV (uplink time ≫ deadline); the A0-vs-B_QIN "gap" (0.37 vs 0.60) is a world/seed effect, on common seeds B_QIN = 0.354 vs A0 = 0.369 at light load | No code change; motivates the scenario recalibration in item (a) of the pre-registration note | The diagnosis report §(a), the diagnosis run records `failures_800{1,2,3}.json` |
| The diagnosis (item b): `slot_raise_f` adds ≈ +0.5 pp for DRM; the baselines' executor already raised | **One admission rule** for DRM and every baseline: `drm.slot_raise_f = true` | The pre-registration note §(c); `config/default.toml` line 178; `tpa/drm/drm.py` (`slot_admissible`, `raise_earlier`); `tpa/baselines/executor.py` |
| The diagnosis (item c): the long-term store collapses to 1–2 slots because the candidate representation saturates (cosine to existing keys 0.94–1.000 under `tanh`); the gate is open 100% because σ_p (median 0.36–0.50) sits far above θ_p = 0.15; the addressing mode is not the cause | New config keys `memory.candidate_repr` (tanh / layernorm / novelty) and `memory.lt_addressing` (fixed_key default; content; soft); `soft` is implemented but **excluded** from the adoption grid (found in code review: unbounded event log); θ_p removed from tuning and fixed by the outcome-blind gate criterion; store-size pass condition (≤ 256 MB per 3P segment) | The pre-registration note §(b); `config/default.toml` lines 107–111; `tpa/memory/stores.py`, `tpa/memory/mepr.py`, `tpa/memory/diag.py`; `scripts/tuning.py` line 78 |
| Code review of the above (the code-review record) | NaN at the first SAC update after the representation change: masked softmax over an all-masked row and a `safe_norm` clamp; integration test added | `tpa/memory/stores.py`; `tests/` |

**Hashes.** Code hash is now **e454246cdc33** (was 9a0b331cc10d at the freeze); config hash is now **a8104a68f14c** (was 0c9cb7897e41). The queue's hash check refuses every row enqueued before this amendment, so the 15 partially run stage-1 rows from the paused launch (`runs/tune/PAUSED_2026-10-27_status.txt`) are discarded; **tuning restarts from scratch** with freshly enqueued rows once approved. Test suite at this hash: see A2.8.

### A2.3 Scenario recalibration: stopped as pre-registered

Item (a) of the A2.2 pre-registration note declared five candidates within the manuscript's Table VI ranges and four criteria (C1 light completion 0.80–0.97, C2 heavy ≥ 10 pp below light, C3 ≥ 0.30 task-slots with ≥ 2 feasible UAVs, C4 ≤ 10% failures with 0 feasible UAVs). **No candidate passed** (the diagnosis run records `scenario.log`, `scenario.json`): C3 and C4 pass from candidate 3 on, but C1 fails for all five (light completion 0.606–0.787). The cause identified in the RESULT section of that note is the `workload.load_basis = "binding"` calibration, which keeps the fleet saturated at nominal "light" load; changing the load definition is outside the pre-registered grid and is a separate decision, not one for this file. Consequently the **mechanism checks of item (b) of that note have not been run** (they need an adopted scenario), and no θ_p / candidate-representation choice has been made. Nothing about the scenario is adopted by this amendment.

### A2.4 Power settings, verified from `runs/power_settings_20261027-0726.log`

Applied by the author from an elevated PowerShell with `scripts/apply_power_settings.ps1` at 07:26:49 (console output in the log):

| Setting | Value in the log | Verified |
|---|---|---|
| Power-mode overlay | `ActiveOverlayAcPowerScheme = ded574b5-…` = Best performance (on top of the Balanced base scheme, which is expected) | yes |
| AC sleep / hibernate timeout | `0x00000000` (off) for both | yes |
| Power throttling (EcoQoS) for the venv and system `python.exe` / `pythonw.exe` | all four listed as `Never On` | yes |
| Core parking minimum (CPMINCORES, CPMINCORES1 = 100) | the commands ran; `powercfg /setacvalueindex` prints nothing, so the log does not show the resulting value | not verifiable from the log |
| Alienware Command Center thermal mode | set to **Performance** by the author at ≈ 08:15 (author's report; not visible in the logs) | author's statement |

Author's readings from Alienware Command Center: before the fix ≈ 4.3 GHz / 95 °C at ≈ 8% load; after the fix ≈ 4.8 GHz / 82 °C; idle 1.6 GHz / 58 °C. **A reading under sustained all-core load has not been supplied yet** (requested for ≈ 08:30 and ≈ 08:36 during phase 3 below). No sensor driver is installed, so package temperature and package power are not readable by the benchmark; only `% Processor Performance` (clock relative to nominal) and the GPU's own sensor are logged. System settings were changed only by the author.

### A2.5 Core re-benchmark after the settings (same kernel as Amendment 1: 400×400 numpy matmul+tanh, iterations per 3-s window, one thread per logical CPU)

`scripts/corebench_full.py`, run 08:24–08:37, after the thermal-mode change; result `runs/benchmarks/corebench_20261027-0824.json`, console in `runs/benchmarks_after.log`. "Before" numbers are those of Amendment 1 (Balanced plan, EcoQoS active).

| Measurement | Before (Amendment 1) | After | Change |
|---|---|---|---|
| P-core alone (8 physical, 3 s) | ≈ 600 | 684–783, mean ≈ 735 | ≈ +20% |
| E-core alone (16, 3 s) | ≈ 40 | 30–53, mean ≈ 44 | ≈ unchanged |
| E-core with all P-cores busy | ≈ 8 (≈ 2% of a P-core) | 127 per core sustained (57% of a loaded P-thread) | **E-cores are no longer starved** |
| 8 P-core workers, aggregate (6 s) | ≈ 3,300 | 4,127 | +25% |
| 16 P-thread workers, aggregate (6 s) | ≈ 4,080 | 4,594 | +13% |
| 16 E-core workers alone, aggregate (6 s) | — | 1,448 (mean 90, min 67) | — |
| All 32 logical CPUs, aggregate (6 s) | — | 4,335 (min 4: start-up transient starves some threads) | — |
| All 32, **sustained 10 min** | — | P 224 per thread (3,584) + E 127 per core (2,032) = **5,616** | +22% over 16 P-threads alone |

Clock behaviour during the 10-min all-core phase: P-core clock 109% of nominal at t = 34 s, settling to 100–102% from t ≈ 500 s, minimum single-core sample 93%; E-cores 128% → 116%. **Corrected in A2.10: Windows' "% Processor Performance" is relative to the BASE clock, and the i9-13900HX P-core base is 2.2 GHz (max turbo 5.4 GHz). "100–102% of nominal" under all-core load therefore means the CPU is held near base clock at TjMax (≈ 100 °C) and ≈ 52 W: the package is thermally limited during sustained all-core load.** (The earlier reading of this line as "not hard throttling" was wrong.) GPU during the phase: 40–44 °C, 7–32 W (idle, with periodic driver activity). The author's Alienware reading (A2.10) confirms ≈ 100 °C during this phase.

### A2.6 Real workload, P-core vs E-core

`scripts/pilot_timing.py _job`, seed 8001, single thread, CUDA hidden, rest of the machine idle; `runs/benchmarks/real_workload_P_vs_E_0837.log`:

| Arm | P-core (logical 6) | E-core (logical 20) | E / P |
|---|---|---|---|
| A9 | 10.0 ms/slot (33 ms/update) | 21.8 ms/slot (76 ms/update) | 2.2× slower |
| A0 (CPU learner) | 23.1 ms/slot (77 ms/update) | 54.0 ms/slot (186 ms/update) | 2.3× slower |

These are *exclusive* E-core figures. Under full load the kernel benchmark puts an E-core at 57% of a loaded P-thread, so a memory-arm run on an E-core worker would take roughly 4–5× the exclusive P-core time (an A0 run of 900k slots ≈ 13.5 h exclusive on an E-core, likely more than 24 h with the machine loaded).

### A2.7 Proposed worker layout (proposal only; to be decided)

**L1 (recommended): keep Amendment 1's layout, 15 workers = 3 GPU + 12 CPU, pinned to the 16 P-core hyperthreads (`--cores p-ht`, CPU 2 skipped), E-cores unused.**
- The E-cores' +22% aggregate is a kernel-benchmark figure; on the real workload an E-core worker is 2.2–2.3× slower exclusive and ≈ 4–5× slower under load, so its runs become the campaign's stragglers (each run is a fixed 900k slots; the queue cannot split a run across cores).
- Adding 16 E-core processes also costs the P-threads ≈ 22% each (3,584 sustained vs 4,594 unloaded aggregate) and adds 16 × ≈ 1 GB of resident memory and the corresponding heat, which is the direction the instructions asked to avoid if the package is near 100 °C.
- Expected throughput is therefore Amendment 1's, improved by the +13–25% P-core gain from the power settings.

**L2 (alternative, only if the author's sustained-load reading is clearly below ≈ 95 °C): L1 plus up to 8 E-core workers restricted by `--cpu-arms` to the cheap non-learning arms (A12, B_ZHOU, B_QIN) and B_SONG**, whose runs are short enough (≈ 1–3 ms/slot on a P-core) that a 2–5× slowdown does not create stragglers. This needs one launcher change (an `e` core mode pinning to logical CPUs 16–31) and would be measured for one hour before being kept.

**Not proposed:** any layout that puts learned arms (A0–A11, A13, A14, B_HUANG) on E-cores.

Until a layout is chosen, nothing runs. The §8 projection is to be recomputed from the first hour of the restarted tuning stage, as Amendment 1 already required.

### A2.8 Test suite at the new hash

Full suite at code e454246cdc33 / config a8104a68f14c, run 08:46–08:51 (`runs/test_suite_amend2_0846.log`): **341 passed, 17 skipped, 0 failed** (skips: GPU-only and trace-file tests, as before). A code review (the code-review record) covers the A2.2 code changes: all seven modules accepted with minor findings; campaign still NO-GO pending approval.

### A2.9 Unchanged

§1–§7 (hypotheses, arms, seeds, metrics, statistics (pilot decisions 1–4), scoring and selection rule). Test seeds remain untouched; validation seeds 9001–9003 served the A2.2–A2.3 diagnostics with non-learning arms and short A0/A9 blocks only; tuning uses 9004–9006 first and 9003 only in stage 2 (§7 of this plan). The main campaign still requires explicit approval.

### A2.10 Thermal readings during the sustained all-core phase (author, Alienware Command Center) and correction to A2.5

| Time | CPU load | Clock | Temperature | Package power | Note |
|---|---|---|---|---|---|
| 08:39 | 100% (all-core phase) | 2.2 GHz | 100 °C | 52 W, 0.8 V | at base clock, TjMax |
| 08:40 | 12% (phase just ended) | 4.6 GHz | 99 °C | 48 W | |
| 08:44 | 8% (near idle) | 3.7 GHz | 78 °C | 34 W | fans CPU 28% / 58%, GPU 26% / 23% |

Alienware thermal mode stays on Performance; no overclock was applied (the Custom OC screen was cancelled without saving); no manual fan control; the laptop is now raised for airflow. The A2.5 interpretation is corrected in place: Windows' "% Processor Performance" is relative to the 2.2 GHz base clock, so 100–102% under all-core load means the package is held at base clock by the thermal limit (≈ 100 °C, ≈ 52 W). The benchmark numbers themselves are unchanged.

### A2.11 Worker layout: decision

**L1 approved**: 15 workers = 3 GPU + 12 CPU on the P-core hyperthreads (`--cores p-ht`, CPU 2 skipped), E-cores unused. **L2 rejected**: the sustained package temperature is ≈ 100 °C, not below 95 °C. No separate worker sweep; aggregate throughput is measured during the first hour of the restarted tuning, as Amendment 1 requires, and the §8 projection is recomputed from it.

### A2.12 Decision of 2026-10-27: load definition (pre-registered before any run)

Scenario candidates 3 (16 UAVs, task size 0.5–2.5 MB, deadline 3–10 s) and then 5 (25 UAVs, same task ranges) are re-tested with the offered load re-expressed as a fraction of a **measured sustainable capacity λ\***: a capacity probe sweeps the offered arrival rate over a pre-declared grid of ≥ 12 points (0.05× to 1.5× the current nominal heavy-load rate) with the two non-learning baselines B_QIN and B_ZHOU on diagnostic seeds 9001–9003 only; λ\* is the offered rate at which the throughput of tasks completed within deadline peaks, averaged over seeds and both baselines. New loads light / medium / heavy = 0.25 / 0.55 / 0.85 × λ\*; C1–C4 are re-checked with the same thresholds; the first candidate in order that passes all four is adopted. If none passes, stop and report with the failure breakdown at 0.25 λ\*. If one passes, the mechanism checks of item (b) of the A2.2 pre-registration note run on it. No memory or learned arm and no tuning seed (9004–9006) is used. Tuning and the main campaign remain stopped until approved.

**Load-definition result and confirmation (2026-10-27, RESULT section of the A2.12 pre-registration note).** Candidate 3 was undetermined (λ\* at the last grid point, not bracketed) and candidate 5 (25 UAVs, size 0.5–2.5 MB, deadline 3–10 s) is **adopted and confirmed**, with λ\* = load 0.68 and loads light / medium / heavy = 0.17 / 0.374 / 0.578 (`runs/d3/adopted.json`). Three disclosures:
1. **C1 passed on the seed mean (0.816), while seed 9001 alone was 0.727**, below the 0.80 floor; seeds 9002 and 9003 were 0.860 each. The criterion was declared on the mean and is applied as declared.
2. **Capacity finding, for the paper's workload section:** the measured sustainable throughput of tasks completed within deadline is **15–20% of λ_binding** (≈ 18 tasks/s at 16 UAVs and ≈ 24 tasks/s at 25 UAVs against λ_binding ≈ 100 and ≈ 135 tasks/s), so the analytic "binding-resource" estimate (`tpa/env/workload.py` lines 53–72) overstates deliverable capacity by ≈ 5–6×. Loads are therefore now defined **empirically** as 0.25 / 0.55 / 0.85 × the measured λ\*, not as fractions of the analytic rate.
3. **At the new light load, the remaining failures are feasible-but-never-admitted tasks** (99–100% `feasible_refused` for B_QIN; `infeasible_unloaded` 0.00), i.e. scheduler-driven losses, not deadline-infeasible tasks as the diagnosis (item a) found for the original scenario.


## Amendment 3 (2026-10-27, decision; tuning and the main campaign remain stopped)

### A3.1 Before any of this: the pilot contrast was null

The pilot (A2.1) found A0 not better than the memoryless A9 at any load (paired A0 − A9 = −0.011, A0 − A3 = −0.0015, code 9a0b331cc10d). Nothing in this amendment was chosen with the aim of changing that number; everything below is outcome-blind (no completion or reward read) and was pre-registered in a dated pre-registration note before the run.

### A3.2 Diagnosis: the long-term store collapses because every candidate shares a constant offset (the diagnosis report of Amendment 3)

Seed 9001, one 10,800-slot block, adopted scenario at medium load 0.374, A0 at seeded initialisation, trough/peak located from observed arrivals:

| Representation | peak-vs-trough cosine, raw | same-phase cosine, raw | peak-vs-trough, centred (3 cycles) | ‖mean‖ / ‖deviation‖ |
|---|---|---|---|---|
| fleet-mean encoder input x_t | 0.988 | 0.999 | −0.73, −0.88, −0.84 | 5.1 |
| current LT candidate z_t (tanh) | 0.886 | 0.993 | −0.80, −0.95, −0.95 | 4.0 |
| zone-resolved candidate v_t, raw | 0.834 | 0.992 | −0.81, −0.98, −0.97 | 3.7 |

By the declared rule case 1 held (x raw 0.988 > 0.95). The three facts that decided the fix: the raw cosine is set by a shared constant offset 3.7–5.1× longer than the typical deviation, not by a missing cycle signal (centred vectors are anti-correlated across phases, and PCA of x_t loads on the per-zone arrival ratios); the encoder does not flatten the signal (z separates phases at least as well as x); and the zone-resolved candidate carries the same offset (raw 0.834 > ρ_merge 0.80), so changing the input alone would not stop the merge. 74 of 327 input dims are constant in the block (eta, burst, 15 energy fractions, 8 τ̄ dims).

### A3.3 Decision of 2026-10-27 (pre-registered before any run)

- **Offset-free key (primary):** the long-term merge test and the long-term retrieval keys use k_t = z_t − μ_{t−1}, μ the normalised EMA of every defined candidate with horizon H_μ = 2P = 7,200 slots (one declared value); stored contents keep the full z_t; gate unchanged; M_S unchanged; applies identically to A0–A8, A13, A14 and to A3's flat store. `memory.key_centering`, `memory.key_mean_horizon_slots`. **Disclosure for the paper:** the manuscript's Eq. 45–46 (in the first submission) compare and retrieve on raw vectors; centring is an implementation choice forced by the measured offset.
- **Zone-resolved candidate (secondary):** z_t = repr(W_v v_t), v_t = [window mean of observed zone arrivals / long-run zone mean (Z); per-UAV in-flight tasks / q_max (M)], specification AMB-13; P(t) unchanged and still drives σ_p and the DRM priority; S_sys unchanged in Eq. (48). `memory.candidate_source = "zone"`. **Disclosures:** the manuscript never defines z(t), so this deviates from specification AMB-9 (z from vec(P)) and, under "zone", P(t) enters the long-term write only through the gate σ_p > θ_p (found in code review); under "zone" the long-term contents carry no gradient to the encoder W_S (only M_S does); the simulator has no per-UAV queue, so "queue occupancy" is the per-UAV in-flight count / q_max, unclipped and able to exceed 1 (environment implementer; found in code review).
- **M3 replaced:** the age-based M3 was unpassable (the A14 control satisfied it on 100% of reads). New M3 = phase consistency of retrieval on the synthetic periodic driver: share of slots in [w+P_d, w+3P_d) whose top-1 entry is a long-term entry written in an earlier cycle (age ≥ P_d/2) within ±P_d/8 of the current phase; chance 0.25; pass iff A0 − A14 ≥ 0.20 on each driver seed 0, 1, 2. M1, M2 and the 256 MB store bound unchanged. Pre-run amendment on a minor code-review finding: the A14 control's displacement U[P/4, 3P/4] is taken in units of the driver's period (`workload.period_slots` overridden to P_d for the driver runs only), so the control reads displaced memory on every evaluation slot; its conditional value and no-memory share are logged.
- **Rerun:** Stage A not rerun (θ_p = 0.35 stands). Stage B grid: repr (layernorm, novelty, tanh) × ρ_merge (0.80, 0.90, 0.95, 0.98), fixed_key, centring on, zone candidate; first pass in declared order adopted; none → stop. Information only, never adopted: each half of the amendment alone at the reference point.

### A3.4 Implementation, verification, test suite, hashes, code review

| Files | Verified independently |
|---|---|
| `tpa/types.py:64` (`SlotObs.uav_occupancy`), `tpa/env/simulator.py:276–279`, `tests/test_env_d4_zone_occupancy.py` (zone arrivals = generator's per-zone count; occupancy = in-flight count) | diff read; 37 env tests rerun, passed |
| `tpa/memory/stores.py` (μ, `key_of`, `candidate_for`, keyed `consolidate`, `_levels` with keys, A14 snapshot keys, A3 `F_key`), `retrieval.py` (keys ≠ values), `recon.py` (log `uav_occ`, `v_ref`, `cand_key`; keys of every filled slot from the event log), `mepr.py` (`compute_v_batch`, `v_ref`), `diag.py` (`phase_hit`, `phase_consistency`), `control.py:171–189` (ordering z → key → write → μ), `config/default.toml:110–112`, `tests/test_memory_d4.py` (17 tests), extended recon / units / driver tests; pre-Amendment 3 reference arrays `tests/data/d4_reference_*.npz` generated on the unmodified code (implementer's attestation: 14:23 local, md5 of `stores.py` then 191e2ff5c04b91434363cab19dc0df7d; that md5 was not independently verified, but the `key_centering=false` + `pattern` path reproduces the reference bit-for-bit, which only the unmodified code could have produced) | key code paths read against the pre-registration note; memory test subset rerun by the implementer (211 passed, 187 skipped); full suite run independently |
| `tests/test_learner_phase_leak.py` (11 tests): no cycle phase or slot index in the state of Eq. (48) for A0, A9–A11, B_ZHOU, B_QIN, B_HUANG, B_SONG; `label` is a diagnostic tag only; A12 the declared exception (AMB-22); report in the code-review record | label path re-read; tests rerun, passed |
| Independent code review (code-review record, lines 855–930): all modules approved with minors, 0 majors, 0 blockers; **"[Approval] for the [Amendment 3] mechanism rerun: YES"**, conditional on the minor finding on the A14 control being decided before the run (done, pre-run amendment of the pre-registration note) | — |

**Full test suite at the new hashes:** test-suite log: **436 passed, 188 skipped** (skips: reconstruction combinations pruned for runtime off the default (zone, centred) combination, and GPU tests). **Code hash 537875428e9b** (was e454246cdc33), **config hash aba006d108a3** (was a8104a68f14c). The queue's hash check again refuses every previously enqueued row; tuning restarts from scratch once approved. Minor code-review findings left open: defaults carry centring + zone before adoption (every queued row passes repr/ρ/θ_p explicitly); A2/A4 not in the off-default reconstruction combinations; hygiene (tests/data README, "content" mode now records keys, old M3 logging — the latter done in the pre-run amendment).

### A3.5 Stage B result (rerun of A3.3)

Run: the mechanism script with 15 worker processes, 15:21–16:30 (69 min), 36 validation blocks + 12 driver jobs, then the two information-only points; records: the result file, the Stage B log and the console log. Outcome-blind. θ_p = 0.35, fixed_key, centred keys, zone candidate, validation seeds 9001–9003, driver seeds 0–2 (P_d = 600, A14 displaced in driver-period units; no-memory share 0.00 on every read, so the conditional equals the unconditional value).

**Result: no grid point passed. Nothing adopted. Stopped as pre-registered (A3.3: none → stop and report).** M1 passed everywhere (gate 0.10–0.14, all 36 blocks); M2 failed everywhere (mean distinct 0–5 vs ≥ 100); M3 failed everywhere, with the A14 control **above** A0 at every point; store 33–34 MB ≤ 256 MB.

| Point (repr / ρ_merge) | gate 9001/9002/9003 | occupancy | distinct (M2) | M3 A0 vs A14, driver seeds 0 / 1 / 2 | old M3 A0 (age > P_d/2) s0/s1/s2 |
|---|---|---|---|---|---|
| layernorm / 0.80 | 0.13 / 0.10 / 0.11 | 2, 3, 17 | 2, 1, 0 | 0.00 vs 0.39; 0.09 vs 0.50; 0.03 vs 0.32 | 0.08 / 0.33 / 0.28 |
| layernorm / 0.90 | 0.13 / 0.10 / 0.10 | 4, 5, 114 | 2, 1, 0 | 0.03 vs 0.39; 0.14 vs 0.49; 0.06 vs 0.35 | 0.07 / 0.35 / 0.30 |
| layernorm / 0.95 | 0.13 / 0.10 / 0.11 | 11, 18, 491 | 0, 0, 0 | 0.03 vs 0.34; 0.29 vs 0.44; 0.10 vs 0.29 | 0.09 / 0.50 / 0.41 |
| layernorm / 0.98 | 0.13 / 0.10 / 0.11 | 170, 179, 500 | 0, 0, 0 | 0.00 vs 0.00; 0.00 vs 0.00; 0.00 vs 0.06 | 0.00 / 0.00 / 0.00 |
| novelty / 0.80 | 0.14 / 0.10 / 0.10 | 3, 3, 6 | 3, 3, 3 | 0.06 vs 0.27; 0.09 vs 0.26; 0.03 vs 0.18 | 0.44 / 0.51 / 0.35 |
| novelty / 0.90 | 0.13 / 0.10 / 0.11 | 6, 4, 31 | 4, 2, 5 | 0.01 vs 0.18; 0.11 vs 0.19; 0.01 vs 0.12 | 0.25 / 0.17 / 0.17 |
| novelty / 0.95 | 0.13 / 0.11 / 0.11 | 26, 14, 203 | 4, 2, 8 | 0.00 vs 0.00 (three seeds) | 0.00 (three seeds) |
| novelty / 0.98 | 0.13 / 0.10 / 0.11 | 248, 179, 500 | 3, 9, 3 | 0.00 vs 0.00 (three seeds) | 0.00 (three seeds) |
| tanh / 0.80 | 0.13 / 0.10 / 0.11 | 3, 3, 5 | 3, 1, 1 | 0.04 vs 0.28; 0.01 vs 0.33; 0.05 vs 0.31 | 0.05 / 0.45 / 0.38 |
| tanh / 0.90 | 0.13 / 0.10 / 0.11 | 4, 4, 19 | 2, 1, 0 | 0.02 vs 0.17; 0.05 vs 0.28; 0.01 vs 0.19 | 0.03 / 0.54 / 0.35 |
| tanh / 0.95 | 0.13 / 0.10 / 0.11 | 13, 13, 134 | 0, 0, 0 | 0.04 vs 0.15; 0.19 vs 0.40; 0.06 vs 0.18 | 0.17 / 0.60 / 0.45 |
| tanh / 0.98 | 0.13 / 0.10 / 0.10 | 144, 151, 500 | 0, 0, 0 | 0.00 vs 0.02; 0.00 vs 0.00; 0.00 vs 0.09 | 0.00 (three seeds) |
| *info only:* layernorm / 0.80, pattern candidate, centred keys (offset-free key alone) | 0.13 / 0.10 / 0.10 | 3, 3, 9 | 3, 1, 0 | 0.02 vs 0.10; 0.00 vs 0.23; 0.05 vs 0.19 | 0.59 / 0.00 / 0.70 |
| *info only:* layernorm / 0.80, zone candidate, raw keys (zone-resolved candidate alone) | 0.13 / 0.10 / 0.11 | 1, 1, 1 | 1, 1, 1 | 0.00 vs 0.38; 0.00 vs 0.32; 0.00 vs 0.25 | 0.04 / 0.07 / 0.16 |

What the numbers show, stated without changing anything:

1. **Centring does stop the total collapse; raw zone keys do not.** With raw keys the zone candidate collapses to a single long-term slot on every seed (zone-resolved candidate alone: occupancy 1, 1, 1), as fact 3 of A3.2 predicted. With centred keys the store fills at high ρ_merge (occupancy 170–500 at 0.98, 114–491 at 0.95 on seed 9003) and reaches 2–31 slots at 0.80–0.90. So the shared offset was the mechanism of the 1–3-slot collapse, as A3.2 said.
2. **M2 is computed on the stored contents, not on the keys** (the mechanism-check script of the A2.2 diagnosis, lines 58–68: a row is "distinct" iff its cosine to every other filled row of `M_L()` is below 0.99). Contents keep the full z_t by A3.3, so rows written a few slots apart are near-duplicates, and a store of 500 rows forming near-duplicate clusters counts 0 distinct. M2 was declared "unchanged" and is applied as declared, so it fails; whether "≥ 100 distinct" should be measured on keys, or as a cluster count rather than an isolated-row count, is a criterion question left for decision. Nothing was changed. Key-space diversity of the filled store was not measured in this run (it needs a new block; `tpa/memory/diag.py` `run_diagnostic` records `key_cos` for that purpose).
3. **The new M3 is confounded by its age condition, in the opposite direction from the old one** (M3 check log: layernorm / 0.80, driver seed 0, outcome-blind). On the driver the top-1 entry is a long-term entry on 100% of the 1,200 evaluation slots for both arms. For A0 the top-1 age has median 106 slots and only 7.7% of reads have age ≥ P_d/2, so 92% are non-hits by the "earlier cycle" condition alone. A14 reads a snapshot displaced by δ ~ U[150, 450], so its top-1 ages are A0's shifted by δ (min 183, median 655) and 93% satisfy the age condition. The phase-offset histograms of the top-1 entry (bins of P_d/8 from 0 to P_d/2) are nearly identical for the two arms, A0: 464 / 377 / 290 / 69 and A14: 464 / 408 / 223 / 105. **A0 and the displaced control retrieve entries at the same phase offsets; the 0.3–0.5 gap is the age filter, which the displacement satisfies mechanically.** The old M3 (share of top-1 age > P_d/2) is 0.00–0.60 for A0 and 0.53–0.97 for A14 at the same points: the same confound seen from the other side. A memory whose retrieval returns its most recent write cannot pass the new M3 against a displaced copy of itself, whatever its content.
4. **At ρ_merge ≥ 0.95 the driver's M3 is zero for both arms** (and old M3 is 0.00 for A0): once the store fills, the top-1 is a write younger than P_d/2 on every read. At layernorm / 0.80 the driver fills 12 long-term slots from 1,746 gated events.
5. **In one sentence:** under every setting tried, the long-term store either holds 1–31 slots (ρ ≤ 0.90) or fills with near-duplicate contents (ρ ≥ 0.95), and retrieval returns a recent write rather than an earlier cycle's. The method as implemented (with the disclosed deviations from Eq. 45–46) has not shown cycle recall in any mechanism check, before or after Amendment 3.

Nothing was adjusted after the run. Tuning and the main campaign remain stopped.



## Amendment 4 (2026-10-28, decision; tuning and the main campaign remain stopped)

### A4.1 Pre-registration

A dated pre-registration note, written before any run (2026-10-27): the phase-clustering probe (A4.2), the interpretation rule (≥ 0.60 same-phase share on all three driver seeds for key (i) or (ii) → adopt that key, re-specify M2 on keys and M3 age-matched, rerun Stage B once; else stop all mechanism work), the diagnosis summary (A4.4), and this record. Outcome-blind throughout; validation seeds 9001–9003 and driver seeds 0–2 only.

### A4.2 Phase-clustering probe (probe script and result file)

Driver period P_d = 600, run w + 3P_d, keys centred by the bias-corrected EMA with H = 2P_d; same-phase share = share of evaluation slots whose cosine nearest neighbour among keys written ≥ P_d/2 earlier lies within ±P_d/8 of the current phase (chance 0.25).

| Key | seed 0 | seed 1 | seed 2 | NN among all earlier: same-phase share / median age |
|---|---|---|---|---|
| (i) a_t centred (zone arrivals over w / long-run zone mean, no learned map) | **0.797** | **0.790** | **0.820** | 1.000 / 1 slot |
| (ii) [a_t ; o_t] centred | 0.891 | 0.878 | 0.867 | 1.000 / 4 |
| (iii) z_t centred (learned map at seeded init, layernorm; reference) | 1.000 | 0.999 | 0.997 | 1.000 / 7 |
| raw a_t (uncentred) | 1.000 | 1.000 | 1.000 | 1.000 / 1 |

**Interpretation rule (A4.1): key (i) passes on all three seeds → first branch.** A first run double-counted offset 0 in the first bin; corrected to the declared inclusive edges and rerun (both logs kept). **Disclosures:** the driver's arrivals and occupancy are clean synthetic sinusoids (Poisson noise on mean 3; sd 0.05 on amplitude 0.4), so "the noise level" is the driver's; the driver's own 0.99-EMA zone mean was not used for (i)/(ii) (the specification's long-run mean was); raw a_t already scores 1.000, so centring is not what makes the arrival key work; the driver's z_t also carries the phase, so on the driver the Amendment 3 keys were not the bottleneck of Stage B's M3.

### A4.3 Implementation: the arrival key (addendum to the pre-registration note; memory implementer; independent code review)

- **Key:** `memory.key_source = "arrivals"` (default stays `"candidate"`, the Amendment 3 behaviour; the runs of this amendment pass `"arrivals"` explicitly). k_t = a_t − μ_{t−1}, a_t ∈ R^Z the window mean of the recorded per-zone arrivals divided by the recorded long-run zone mean of slot t, μ the normalised EMA of A3.3 (α = 1/(2P)), first defined key uncentred, defined once w slots are recorded; no learned parameter on the key side (W_q, W_k, W_z receive no gradient); key dimension Z = 6. One cached sequence in the segment store serves the merge test, the M_S / M_L / A3 write keys, the retrieval query and reconstruction (`tpa/memory/recon.py` 131–178, `stores.py` 409–429).
- **Retrieval:** logits_j = cos(k_t, K_j) / τ_key, τ_key = 0.1 (`memory.key_temperature`, one declared value), broadcast over the heads; value path, W_V and the level weights unchanged (`retrieval.py` 93–97). A14 reads the snapshot's keys with the live query. `retrieve(min_age)` adds `top_old_slot/age` on the source actually read, never touching r(t) or the LRU clock (`stores.py` 709–719).
- **Disclosures for the paper:** Eq. 46's multi-head projections are dropped on the key side; keys are the AMB-13 arrival quantity, not projections of the encoded state; M_S rows written in the first w − 1 slots of a segment carry no key and are not attended until overwritten (found in code review; slots t0+99 … t0+198 only).
- **M2 on keys:** greedy leader count of the filled L_key rows in M_L slot-index order at cosine 0.95 (found in code review: index order, not write order, declared before the run), mean over seeds ≥ 100; isolated-row count logged. **M3 age-matched:** driver with cumulative zone mean, min_age = P_d/2, share of evaluation slots whose oldest-restricted top entry lies within ±P_d/8 of the current phase; A0 − A14 ≥ 0.20 on each driver seed. M1 and the store bound unchanged.
- **Tests:** `tests/test_memory_d5.py` 23 tests (hand values of a_t / μ / k_t, merge on the key, `"candidate"` reproduces the Amendment 3 reference rollouts bit-for-bit, replay = rollout for A0/A1/A2/A3/A4/A14 under `"arrivals"`, top_old on live and snapshot sources, M2/M3 hand values, driver cumulative mode, state round trips, gradients); `tests/test_memory_recon.py` arrivals sweep; `tests/test_memory_integration.py` arrivals cases. Implementer's subset 250 passed / 196 skipped. **Full suite** (test-suite log): **480 passed, 197 skipped** (10 min). **Code hash 915f563a5cbd** (was 537875428e9b), **config hash f3293b682e41** (was aba006d108a3): the queue's hash check again refuses every previously enqueued row.
- **Independent verification:** the arrival-key computation, the cosine logits, retrieve's top-old fields and the diag functions read line by line against the addendum; the smoke run of the mechanism script (1 point, 1 seed, 1,200 slots) completed; the full suite run independently.
- **Independent code review:** code-review record (lines 978–1037): every module OK, four MINOR findings (disclosure, wording, hygiene, probe floor), no MAJOR, no BLOCKER; the review's own probe: every event key equals the store's arrival key bitwise, retrieve equals hand cos/τ attention to 1.2e-7. **Approval for the rerun: YES**, with the two record conditions written into the pre-registration note before the run.

### A4.4 Diagnosis summary

Written: the four measured findings (offset ratio and raw collapse; centring stops the collapse but contents stay near-duplicate; retrieval returns recent writes, median top-1 age 106 of 600; A0's phase-offset histogram equals the shuffled control's), the A4.2 probe numbers and the pilot null, with file references and no interpretation.

### A4.5 Stage B rerun with the arrival key (interpretation rule of A4.1, once)

Run: the mechanism script with 15 worker processes, 01:50–02:08 (18 min), 12 validation blocks + 4 driver jobs; records: the result file, the Stage B log and the console log. Outcome-blind. θ_p = 0.35, fixed_key, `key_source = "arrivals"`, contents zone / layernorm, validation seeds 9001–9003, driver seeds 0–2 (P_d = 600, cumulative zone mean, min_age = 300; A14 displaced in driver-period units; every evaluation read had an old entry available in both arms: defined share 1.00, no-memory share 0.00).

**Result: no grid point passed. Nothing adopted. Stopped as pre-registered.** M1 passed on all 12 blocks (gate 0.10–0.13); M2-keys failed everywhere (mean greedy groups 2.7–7.3 vs ≥ 100); M3 age-matched failed everywhere (A14 ≥ A0 on every seed at every point); store 32–33 MB.

| ρ_merge | gate 9001/9002/9003 | LT occupancy | key groups (cos 0.95) | isolated keys | M3 age-matched, A0 vs A14, driver seeds 0 / 1 / 2 |
|---|---|---|---|---|---|
| 0.80 | 0.13 / 0.10 / 0.10 | 2, 3, 3 | 2, 3, 3 | 2, 3, 3 | 0.43 vs 0.90; 0.42 vs 0.91; 0.46 vs 0.92 |
| 0.90 | 0.13 / 0.10 / 0.10 | 3, 4, 7 | 3, 4, 7 | 3, 4, 7 | 0.66 vs 0.92; 0.68 vs 0.92; 0.76 vs 0.92 |
| 0.95 | 0.13 / 0.10 / 0.10 | 5, 7, 10 | 5, 7, 10 | 5, 7, 10 | 0.83 vs 0.89; 0.90 vs 0.87; 0.77 vs 0.88 |
| 0.98 | 0.13 / 0.10 / 0.10 | 10, 10, 19 | 5, 5, 9 | 1, 0, 0 | 0.84 vs 0.87; 0.82 vs 0.84; 0.82 vs 0.86 |

What the numbers show, stated without changing anything:

1. **The arrival key retrieves same-phase entries from earlier cycles.** With the oldest-restricted top entry, A0's same-phase share is 0.77–0.90 at ρ ≥ 0.95 (chance 0.25), against 0.00–0.29 under the Amendment 3 candidate keys at the same points (A3.5). On the driver, retrieval by the arrival key does recall the earlier cycle's phase.
2. **The shuffled control is not a null for a key that carries the phase.** A14 reads a snapshot of the same store, δ ~ U[150, 450] slots old, with the live query k_t. The snapshot's entries keep their own arrival keys, so the live query finds a same-phase entry in it as readily as in the live store, and the restriction to age ≥ P_d/2 (ages taken from the live t) leaves the control more qualifying entries, not fewer. Both arms sit at 0.84–0.92 at ρ ≥ 0.95; at ρ = 0.80 A0 drops to 0.42–0.46 because its 2–3 slots are refreshed by merges (L_time reset, `stores.py` line 577) and so fail the age condition on the live store, while the same slots in the older snapshot pass it. The A4.3 criterion "A0 − A14 ≥ 0.20" cannot be met by a memory whose displaced copy is as good a phase index as the live one, whatever its content; the control tests whether retrieval depends on *when* the store was read, which a phase-keyed store does not.
3. **The store still holds 2–19 long-term slots.** M2-keys asks for ≥ 100 distinct keys; with Z = 6-dimensional arrival keys and the merge test at cosine ≥ ρ, nearly every gated candidate (≈ 1,000 per block at a 10% gate) merges into an existing slot: a_t over the demand cycle traces a closed curve in R^6, and at cosine 0.95 that curve is covered by 5–10 slots (key groups = occupancy at ρ ≤ 0.95: every held slot is distinct, but there are few of them). At ρ = 0.98 the store grows to 10–19 slots of which 5–9 are distinct at 0.95. This is a property of the key's dimension and the merge rule, not of the offset (which centring removes) nor of the encoder (which is no longer on the key path).

Nothing was adjusted after the run. Tuning and the main campaign remain stopped.



## Amendment 5 (decision of 2026-10-28; tuning and the main campaign remain stopped)

### A5.1 Pre-registration

A dated pre-registration note, written before any code change or run: the key-shuffled control A15 (A5.2), M2 re-specified as phase coverage of the long-term store (A5.3), M3 re-specified as the age-matched same-phase share A0 vs A15 (A5.3), one rerun on the ρ grid with first-pass adoption (A5.4), and the implementation, code-review and record rule. The decision accepted the two observations of Amendment 4 as errors of the criteria (A14 is not a null for a phase-keyed store; a slot-count floor is wrong for a 6-dimensional key) and stated that the result of Amendment 4 stands: under the arrival key A0 recalls same-phase entries from earlier cycles at 0.77–0.90 (chance 0.25).

### A5.2 A15, the key-shuffled control (memory implementer; independent code review)

- **Definition as implemented:** A15 = A0 with the same store, gate, contents and retrieval; at every slot t with a defined arrival key, π(t) ~ U{t0 + w − 1 ≤ t′ < t} is drawn once from the dedicated stream `a15` (id 11; driver: seed + 3000), in slot order and independently of the gate (the draw sits in `write_short`, which every caller runs before `consolidate`; `tpa/memory/stores.py` 447–465); π(t) = t at the first defined slot. The write key k_{π(t)} is used for the long-term merge test, the stored key of a new / replaced long-term slot, the M_S row key and the flat-store row key; **the retrieval query stays the true k_t**; contents are untouched. π is logged per slot in the segment store, carried in the state, and reconstruction reads it from the log and raises rather than re-draw (`recon.py` 187–230). `ArmFlags.key_shuffled`, `flags_for_arm("A15")`; A15 is a mechanism-check arm only and appears in no campaign or queue arm list (`tests/test_arms.py`).
- **Verification:** implementer's subset 240 passed / 181 skipped; `tests/test_memory_d6.py` 18 tests (hand-value substitution, π once per slot regardless of gate, replay = rollout for A15 under fixed_key and soft, Amendment 3 references and Amendment 4 arrivals tests unchanged, phase_coverage hand values, state round trips, controller path). The independent code review's own probe: every event key equals k_{π(t)} bitwise, π uniform (KS-D 0.042), replay = rollout to 3e-7, A0 and A15 share identical gate decisions and contents. **Full suite**: **502 passed, 198 skipped**. **Code hash a6e66ab01db9** (was 915f563a5cbd); config hash f3293b682e41 unchanged (no default touched).
- **Independent code review (code-review record, lines 1039–1099):** all modules OK; MINORs (M2 compares first-write keys binned by latest-write phase, one inert, age summary added before the run, two on hygiene); one MAJOR **of record**: `scripts/tuning.py` builds its arm set from the whole `ARMS` registry (lines 52–54, 118, 144–152, 169–170, 221) and would sweep A15 into a future tuning sample; to be filtered before tuning restarts. **Rerun approved: YES**; the three record conditions were written into the pre-registration note before the run.

### A5.3 Criteria as run

- **M2 phase coverage:** on each validation block at block end, every filled long-term slot is binned by the phase of its latest write (8 bins of P/8, P = 3,600, origin t0); pass iff all 8 bins are occupied on each seed and the mean within-bin key cosine exceeds the mean between-bin key cosine on each seed (`tpa/memory/diag.py` `phase_coverage`).
- **M3 age-matched vs A15:** driver (cumulative zone mean, P_d = 600, seeds 0–2, min_age = P_d/2), share of evaluation slots whose oldest-restricted top entry lies within ±P_d/8 of the current phase; pass iff A0 − A15 ≥ 0.20 on each seed; A15's value is the empirical null; A0 vs A14 logged alongside (information only). M1 (gate 5–60%) and the 256 MB store bound unchanged.

### A5.4 Rerun result

Run: the mechanism script with `--workers 15`, 02:53–03:11 (18 min), 12 validation blocks + 4 driver jobs; records: the result file, the Stage B log and the console log. Outcome-blind. θ_p = 0.35, fixed_key, arrivals key, contents zone / layernorm, validation seeds 9001–9003, driver seeds 0–2 (P_d = 600, cumulative zone mean, min_age = 300; every evaluation read of every arm had a qualifying old entry: defined share 0.97–1.00).

**Result: no grid point passed all four criteria. Nothing adopted. Stopped as pre-registered.** M1 passed on all 12 blocks (gate 0.10–0.13). **M3 against the key-shuffled control passed at ρ = 0.95 and 0.98 on every driver seed** (A0 0.77–0.90 vs A15 0.15–0.37; smallest margin 0.42). The key-structure half of M2 passed at ρ ≥ 0.95 on every seed (within-bin cosine 0.76–0.97 vs between-bin −0.35 to 0.76). **The coverage half of M2 failed on every block:** the store never occupies more than 5 of the 8 phase bins. Store 32–33 MB.

| ρ_merge | gate 9001/9002/9003 | LT occupancy | phase bins occupied (of 8) | within > between | M3 A0 vs A15, seeds 0 / 1 / 2 (A14, info) |
|---|---|---|---|---|---|
| 0.80 | 0.13 / 0.10 / 0.10 | 2, 3, 3 | 2, 3, 2 | undefined, undefined, yes | 0.43 vs 0.36; 0.42 vs 0.49; 0.46 vs 0.26 (A14 0.90–0.92) |
| 0.90 | 0.13 / 0.10 / 0.10 | 3, 4, 7 | 3, 3, 4 | undefined, yes, yes | 0.66 vs 0.26; 0.68 vs 0.60; 0.76 vs 0.27 (A14 0.92) |
| 0.95 | 0.13 / 0.10 / 0.10 | 5, 7, 10 | 4, 4, 5 | yes, yes, yes | **0.83 vs 0.35; 0.90 vs 0.37; 0.77 vs 0.15** (A14 0.87–0.89) |
| 0.98 | 0.13 / 0.10 / 0.10 | 10, 10, 19 | 5, 4, 4 | yes, yes, yes | **0.84 vs 0.22; 0.82 vs 0.33; 0.82 vs 0.20** (A14 0.84–0.87) |

Bins at block end (latest-write phase, 8 bins of P/8, origin t0): ρ 0.98 → 9001 [1,3,0,0,4,1,0,1], 9002 [0,0,0,0,2,2,4,2], 9003 [1,0,7,9,0,0,0,2]; ρ 0.95 → [0,1,0,0,2,1,0,1], [0,0,0,0,1,1,3,2], [1,0,2,5,1,0,0,1]. Driver (per arm, end of run, ρ 0.98): A0 occupancy 53–66, median entry age 434–639 slots, share older than P_d/2 0.64–0.73; A15 occupancy 53–70, median age 41–57, share older than P_d/2 0.12–0.31 (the shuffled keys merge candidates into unrelated slots, so A15's entries are refreshed constantly); A14 = A0's store read late.

What the numbers show, stated without changing anything:

1. **Against a valid null, the arrival-keyed long-term memory recalls earlier cycles.** The age-matched same-phase share of A0 is 0.77–0.90 at ρ ≥ 0.95 while the key-shuffled A15, with the same contents, gate, write rate and (nearly) the same occupancy, sits at 0.15–0.37, near the 0.25 chance level. The gap is 0.42–0.62 on every seed, against a required 0.20. This is the first mechanism check in which A0 beats a control that it should beat; the time-displaced A14 (0.84–0.92) is on record as an invalid null for this key, as Amendment 4 found.
2. **The store does not span the cycle.** Across all 12 blocks the filled slots occupy 2–5 of the 8 phase bins. One outcome-blind diagnostic block (seed 9003, ρ 0.98, the same configuration; a gate-phase script and its result file, run after the grid, interpretation only) shows why: **the consolidation gate σ_p > θ_p opens at three phases of the cycle and almost nowhere else.** Of the 1,080 slots on which it opened, 329 / 343 / 364 fall in phase bins 2 / 3 / 7, 22 + 13 + 9 in bins 0 / 4 / 6, and none in bins 1 and 5 (mean σ_p per bin 0.22–0.34 against θ_p = 0.35). Every one of the 1,080 long-term write events therefore carries one of those phases (19 new slots, 1,061 merges), and the 19 filled slots at block end lie in bins 0 / 2 / 3 / 7 ([1,0,7,9,0,0,0,2]). Coverage of all 8 bins is impossible for a store that is only written when the gate opens, and the gate (Eq. 47, σ_p = ‖P − P_ref‖ / ‖P_ref‖) opens where the pattern changes fastest, not uniformly over the cycle. M2-coverage therefore measures the gate's phase selectivity, not the key or the store; it is applied as declared and fails.
3. **Key structure holds where the store has pairs:** within-bin cosine exceeds between-bin cosine on every seed-point that has a within-bin pair (8 of 8 at ρ ≥ 0.90, 1 of 1 at ρ 0.80); the comparison is undefined on the other three.

Nothing was adjusted after the run. Tuning and the main campaign remain stopped.



## Amendment 6 (decision of 2026-10-28, final for the mechanism phase; tuning APPROVED and restarted; main campaign still needs approval)

### A6.1 Decisions recorded

- **M2-coverage withdrawn.** The gate's phase selectivity is a designed property of Eq. 47 (significance = deviation from the reference): the long-term store records transitions, M_S records the stable phases. The diagnosis summary (finding 5) holds the measured histogram (gate open in bins 2 / 3 / 7 on 329 / 343 / 364 of 1,080 openings; none in bins 1 and 5); **paper disclosure**. The key-structure half of M2 (within-bin > between-bin key cosine) stands and passed at ρ ≥ 0.95 on every seed.
- **Adopted (first grid point passing M1, M2-key-structure, M3 vs A15 and the store bound):** ρ_merge = 0.95, θ_p = 0.35, `key_source = "arrivals"`, `candidate_source = "zone"`, `candidate_repr = "layernorm"`, `key_centering = true`, `lt_addressing = "fixed_key"`, τ_key = 0.1. Written into `config/default.toml` as the memory defaults for every memory arm (A0–A8, A13, A14, and A3's flat store). θ_p and ρ_merge are mechanism-selected, outcome-blind, and **not tuned**.

### A6.2 Before tuning

- **(a) A15 filtered out** of the tuning arm set (`scripts/tuning.py` `MECHANISM_ONLY = {"A15"}`; `ARMS` rebuilt without it, so SAC_ARMS / GATED / BURST / sample / enqueue1 / score1 / select never see it) and of the campaign (`scripts/enqueue.py` refuses it). `tests/test_tuning_arms.py` (6 tests) and `tests/test_arms.py` cover both. No search space contains any `memory.*` key.
- **Adopted scenario on every row (found in code review, closed):** the scenario adopted in A2.12 (25 UAVs, task size 0.5–2.5 MB, deadline 3–10 s, loads 0.17 / 0.374 / 0.578; `runs/d3/adopted.json`) is not in `config/default.toml` (whose geometry the unit tests pin); `tuning.py` `row_overrides` and `enqueue.py` merge it into every queued row, and the row's config hash is computed on the merged config, which is what the worker recomputes.
- **(b) Arrival key identical across arms:** `tests/test_memory_d7.py` runs every memory arm (A0–A8, A13, A14) on the same driver inputs and checks that the per-slot key sequence equals A0's and that every stored key (L_key, S_key, A3's F_key) is the arrival key of its write slot. A0 and A3 share arrivals, keys and gate decisions; besides the time-scale separation they differ only through x_t's short-term mean term m̄ (M_S mean vs flat-FIFO mean), which moves e_t and σ_p at float level (documented in the test).
- **(c) No new state input:** keys select contents; r(t) is unchanged in dimension; `tests/test_learner_phase_leak.py` rerun (11 passed).
- **(d) Full suite:** **522 passed, 198 skipped** (a first run at the new defaults had 56 failures, all tests that implicitly assumed the defaults before this amendment; they now pin the configuration they test explicitly, with no assertion weakened and no code change; code-review blocker closed). **Code hash c4ba4894f0b4 (was a6e66ab01db9), config hash 203b2058f1b5** (config was f3293b682e41 before this amendment). Independent code review (code-review record, line 1101 onward): defaults equal the adopted point and nothing else in the config changed (revert probe reproduces f3293b682e41); A15 unreachable from tuning and campaign; every row carries the adopted scenario with matching hashes; arrival key identical across arms; nothing new in the Eq. 48 state; equal budget and the declared stage score are the only outcome read. **Final configuration approved for tuning: YES**, conditions: (i) suite green at these hashes before enqueueing (met, above); (ii) worker device layout recorded in the tuning log (below); record items: test wording (two minors, fixed); A10 width rule checked on the default geometry (all widths pass on both geometries); device is outside the config hash (layout recorded); the archived 2026-10-27 launch is never scored.

### A6.3 Tuning restart

Frozen plan unchanged (§6, Amendment 2): equal budget of 12 configurations per tunable arm from its own declared space, stage 1 on validation seeds 9004 / 9005 × 300 episodes, stage 2 top-2 on seeds 9006 / 9003 × 1,000 episodes, score = medium-load validation completion of the selected checkpoint (`best_val`); test seeds never used. Housekeeping before enqueueing: the 408 pending rows of the paused 2026-10-27 launch (code 9a0b331cc10d) were deleted from `runs/runs.db` and their 24 partial run directories moved to `runs/tune_paused_20261027/`; `runs/tune/configs.json` regenerated (`tuning.py sample`, deterministic per-arm sampler; A15 absent). Worker layout L1 (15 workers = 3 GPU + 12 CPU on the P-core hyperthreads, CPU 2 skipped).

**Launched 2026-10-28 04:33 local (01:33 UTC), `runs/tune/LAUNCH_20261028.txt`, `runs/tune/launch_20261028.log`:** 408 stage-1 rows enqueued (17 tunable arms × 12 configurations × seeds 9004, 9005; B_ZHOU and B_QIN have nothing to tune), row config hash 33267ddd646d (adopted scenario + 300 episodes, before the per-arm sample), code c4ba4894f0b4. Workers: `run_queue.py --workers 15 --gpu-workers 3 --phase tune --cores p-ht --gpu-arms A0,…,A8,A13,A14` — 3 GPU workers (logical CPUs 0, 1, 3) restricted to the memory arms, 12 CPU workers (logical CPUs 4–15) unrestricted (§8 option (b): memory arms also run on CPU; device recorded per run manifest), CPU 2 skipped, keep-awake on. All 15 workers claimed rows within 10 s. **Stage-1 aggregate throughput is measured after the first hour (`scripts/tune_throughput.py`, `runs/tune/throughput_1h.log`) and the §8 projection recomputed from it; the stage-1 summary is reported before stage 2 is enqueued (`tuning.py score1`). The main campaign still requires approval.**

### A6.4 Stage-1 throughput after the first hour (`runs/tune/throughput_1h.log`, measured 02:33 UTC, one hour after launch)

- **Aggregate 95 slots/s** over the 15 workers (342,000 slots of A0 in 3,600 s; every worker on an A0 row because memory arms carry queue priority 1). The heartbeat counts episodes in steps of 10 (checkpoint interval), so the figure is a lower bound by up to one checkpoint per worker.
- **Per run (timing columns of the training log only; no outcome read):** CPU workers 83–85 ms/slot at update_every = 8 (112 learner updates per episode), 177 ms/slot on the run with update_every = 2 (450 updates per episode, on a GPU worker). Module costs per slot: DRM 14–44 ms, MEPR 3.7–4.4 ms, retrieval 2.8–3.0 ms, actor 2.4 ms, env 2.0 ms. In the pilot timing (§8, 5-UAV default geometry, exclusive runs) the decision path was 3.3 ms per slot and DRM 0.33 ms; the scenario adopted in A2.12 (25 UAVs, loads 0.17 / 0.374 / 0.578) multiplies the per-slot DRM assignment work by 40–130×, and the tuning configurations with update_every = 2 quadruple the learner updates.
- **Projection recomputed at 95 slots/s:** stage 1 (408 runs × 270,000 slots) ≈ 13.4 days; stage 2 (68 runs × 900,000 slots) ≈ 7.5 days; main campaign at n = 30 (19 arms × 30 seeds × 900,000 slots = 513 M slots) ≈ 62 days. The frozen §8 projection assumed 310 slots/s (≈ 3.5 weeks for the campaign); the measured rate is 3.3× lower, almost entirely from the adopted scenario's DRM cost, not from the memory. **No change made; to be decided** (n, episodes per run, or arms per stage). Tuning continues meanwhile; the stage-1 summary is reported before stage 2.


## Amendment 7 (decision of 2026-10-28: budget revision; declared before any test-seed result exists; main campaign still needs approval)

**Reason (words of the decision; editorial changes in square brackets):** the measured throughput (95 slots/s aggregate on the adopted [A2.12] scenario, A6.4) makes the frozen plan (§2 n = 30 for 19 arms, §5 robustness n_r = 10 over 7 workload families) infeasible on the available hardware. The evidence the reviewers asked for is kept; the number of runs is cut where no reviewer asked for them. No test-seed run exists; no stage-1 (validation-seed) result has been read (`runs/tune/stage1_scores.json` does not exist). Nothing is cut by outcome.

### A7.0 DRM speed-up decision not received — item 0 of the budget revision not executed

Item 0 of the decision orders "First execute [the decision recorded in Amendment 8] (DRM speed-up with bit-identical decisions) exactly as written." **No DRM speed-up order has been received** (after the decision of Amendment 6, the next decision received was that of this amendment; there is no text of the DRM speed-up decision). Nothing in the DRM code was changed; the code hash is unchanged at **c4ba4894f0b4** and the config hash at **203b2058f1b5**. The counts in A7.1–A7.4 are, per item 0, fixed regardless of the DRM speed-up; the projection in A7.5 is therefore at the 95 slots/s before the DRM speed-up and is to be recomputed once the DRM speed-up decision arrives and is executed. Until the DRM speed-up is executed and its throughput reported, this amendment covers items 1 to 5 of the decision (A7.1–A7.5) only.

### A7.1 Arm tiers for the main campaign (replaces §2 n = 30 for every arm)

Test seeds 1..n per tier, paired by seed; tier A's seed set ⊃ tier B's ⊃ tier C's, so every paired contrast uses the seeds common to both arms.

| Tier | n | Arms | Inference |
|---|---|---|---|
| A | 30 | A0, A3, A9, A10, A12 | Primary test A0 vs A3 unchanged (§2: paired t, α = 0.05, SESOI 1 pp with TOST). Secondary family S2 (§3) unchanged. |
| B | 12 | A1, A2, A4, A5, A6, A7, A8, A11, A14, B_ZHOU, B_QIN, B_HUANG, B_SONG | Paired mean difference, 95% CI, d_z and Holm p within family; families S1 / S3 / S4 as in §3 but on n = 12 for every contrast that involves a tier-B arm (S1 contrasts A0 vs A9 and S4 use n = 30, both arms tier A; S2's A11 vs A9 uses n = 12). |
| C | 6 | A13 | Effect size and CI only; no hypothesis test (A13 leaves the S1 test family). |

- **MDE statement for the paper:** for a paired t-test at α = 0.05 (two-sided), power 0.8, n = 12: MDE = (t₀.₉₇₅,₁₁ + t₀.₈₀,₁₁) · SD_d / √12 = (2.201 + 0.876) / 3.464 · SD_d ≈ 0.89 · SD_d. The decision's "≈ 3 pp" corresponds to SD_d ≈ 3.4 pp; the pilot paired SD (§2, 0.30 pp, n = 3, before the changes of A2.2) would give 0.27 pp. The paper reports the MDE computed from the campaign's own SD_d per family, together with this formula, and states that the ablation (tier B) is powered for effects of roughly 3 pp or larger. Computed by the analysis script (not yet written; §6: it runs once, after all main-campaign runs are done); not read before.
- Arm sets per tier are fixed now; nothing is excluded by outcome. The campaign enqueue (`scripts/enqueue.py --phase main`) is called per tier with the tier's seed range (A: 1–30, B: 1–12, C: 1–6); the row counts are checked against this table before launch (312 rows: 150 + 156 + 6).

### A7.2 Tuning, equal budget, reduced (replaces §6 "12-configuration budget" and A6.3)

- 6 configurations per tunable arm: the **first 6 of the 12 already sampled** (same sampler, same spaces; `runs/tune/configs.json` unchanged, 12 per arm, A15 absent). Stage 1 on validation seed 9004 only, 300 episodes. Stage 2: top 2 of stage 1 on seed 9005, 1,000 episodes. Score unchanged (`best_val`, medium-load validation completion). Arms without a learner (A12 is a lookup with its own 2-parameter space and *is* tuned; B_ZHOU and B_QIN have nothing to tune and are unchanged). Validation seeds 9001–9003 and 9006 are no longer used by tuning.
- **Code (`scripts/tuning.py`, outside the code hash):** `N_STAGE1 = 6`, `STAGE1_SEEDS = (9004,)`, `STAGE2_SEEDS = (9005,)`; `enqueue1`, `score1` and `select` iterate `configs[arm][:N_STAGE1]`. `tests/test_tuning_arms.py` gains `test_reduced_tuning_budget_constants` (constants; configs.json equals the deterministic sample; 17 × 6 × 1 = 102 rows) and `test_tuning_queue_holds_exactly_the_reduced_set` (the live queue is exactly c00–c05 × seed 9004 for the 17 tunable arms). 13 passed (`tests/test_tuning_arms.py` + `tests/test_arms.py`).
- **Queue revision as executed (queue-revision log, 04:07 UTC):** the launcher and its 15 workers of the 04:33-local launch were stopped (process trees killed; 2.57 h after launch; only A0 rows had started). Of the 408 stage-1 rows, 306 were deleted (every seed-9005 row and every c06–c11 row); 9 of them were running (A0 c00–c05 × 9005 at 30–50 episodes; A0 c06 × 9004 at 110 and c07 × 9004 at 100 episodes) and their partial run directories were moved to a separate folder (kept on disk, never scored). The 6 kept running rows (A0 c00–c05 × 9004, at 40–100 episodes) were returned to pending and resume from their last 10-episode checkpoint (§6 crash rule; resume is bitwise on CPU). Result: 102 pending rows, 6 per tunable arm, all seed 9004; the 12 pilot rows untouched. Stage-1 progress carried over: 378,000 slots on the kept rows.
- Relaunch: same worker layout L1 and the same command as A6.3 (`runs/tune/launch_20261028b.log`, `runs/tune/LAUNCH_20261028b.txt`), after the plan hash below was recorded. The stage-1 summary is reported before stage 2 (`tuning.py score1`).

### A7.3 Robustness block (replaces §5)

- Workloads: **MMPP (aperiodic)** and **Gauss–Markov user mobility** only. Pareto on-off, patrol circles, packet loss (0.05, 0.10) and the shifted and burst families are **dropped from this resubmission and listed as future work** in the paper.
- Arms: A0, A3, A9, A10 (the "best baseline from S3" is dropped). Contrasts A0 vs each, heavy load. n_r = 6. Exploratory; Holm within each workload family. 2 × 4 × 6 = 48 runs.

### A7.4 Exclusive timing and energy

Exclusive timing and energy runs are made during the campaign on one reserved P-core, as pre-registered (specification §8.6, §8 of this plan), 1 h total; the campaign workers leave that core free for that hour.

### A7.5 Projection for A7.1–A7.4 at the throughput of 95 slots/s before the DRM speed-up (A6.4)

| Block | Runs | Slots | Days at 95 slots/s |
|---|---|---|---|
| Tuning stage 1 (remaining after the 378,000 carried over) | 102 × 270,000 | 27.2 M | 3.3 |
| Tuning stage 2 | 34 × 900,000 | 30.6 M | 3.7 |
| Main campaign (A7.1) | 312 × 900,000 | 280.8 M | 34.2 |
| Robustness (A7.3) | 48 × 900,000 | 43.2 M | 5.3 |
| Exclusive timing and energy (A7.4) | — | — | 0.04 |
| **Total** | | **381.8 M** | **≈ 46.5** |

(Frozen plan at the same rate: 13.4 + 7.5 + 62 days for tuning and the main campaign alone, A6.4.) `scripts/tune_throughput.py` now prints this projection from the A7.1–A7.4 counts and the live aggregate rate; the recorded 95 slots/s is the first-hour measurement on the 15-worker layout and is to be remeasured after the DRM speed-up. **The main campaign still needs approval**.

### A7.6 Hash of the revised plan

Recorded in `runs/tune/PLAN_HASH_A7.txt` (SHA-256 of this plan ending with this line) before the relaunch; the launch record `LAUNCH_20261028b.txt` repeats it. Code c4ba4894f0b4, config 203b2058f1b5, unchanged.


## Amendment 8 (decision of 2026-10-28 and its follow-up: DRM speed-up with bit-identical decisions; compute, not science; tuning stopped at 04:18 UTC and NOT relaunched; main campaign still needs approval)

The DRM speed-up decision was issued before the budget revision but was received after Amendment 7 and the relaunch of A7.2 (04:09 UTC). Per this decision, tuning was stopped at 04:18 UTC (launcher and 15 workers killed; the 15 running rows released to pending; queue-revision log); the budget-revision counts (A7.1–A7.4) are unchanged. No decision, equation or hyper-parameter changes here: every value that reaches a decision is computed by the same float64 expression as before, which the equivalence test (A8.4) asserts bit for bit on every arm.

### A8.1 Profile of Algorithm 3 (a profiling script; its logs and result files of the reference implementation for A0, B_QIN and A9)

One frozen-policy block on the validation stream of seed 9001 (test_run 0; medium load 0.374 of the adopted scenario, 25 UAVs, 200 devices; untrained controller, so the queue stays at its cap of 128 with 32 tasks in the window: the worst case for Algorithm 3), per-function wall time by wrapping the decision-path functions with `perf_counter_ns` (inclusive; one process, 1 thread, idle machine). A0 over 10,800 slots, B_QIN and A9 over 3,600:

| Function (inclusive) | A0: ms/slot (calls/slot) | share of DRM | B_QIN | A9 |
|---|---|---|---|---|
| DRM total (controller timer) | **50.1** | 100 % | 68.0 | 61.0 |
| `slot_admissible` (admission check) | 41.2 (267) | 82 % | 55.7 (282) | 51.2 (308) |
| of which `obs.rate_fn` (slot-assigned rate prediction: one full 200 × 25 FDMA rate matrix per call, 126–146 µs) | 37.2 (279) | 74 % | 44.8 (307) | 45.9 (320) |
| `DRM._pair` (per-(task, UAV) satisfiability, 1,600 scalar calls) | 4.6 | 9 % | — | 5.2 |
| priority + sort, window, task features, load index, burst, finalize_predictions, allocation | 0.8 together | 1.6 % | | |

The 14–44 ms of A6.4 is the admission check, and inside it the rate prediction: every feasible (task, UAV) candidate re-predicts the full rate matrix with itself added as an interferer. Priority computation and sort, the utility scan, the allocation and the burst check are below 0.2 ms each. The per-slot numbers depend on the regime: with a saturated queue (untrained or early-training policy) 40–50 ms; in the stage-1 training logs (`runs/tune/tune1-A0-c0*-s9004/train`, 15 workers on hyper-threaded cores) 12.9–15.8 ms.

**Learner update (item 2 of the decision, reported; the search space is not changed).** Same training logs (medium of the last 20 episodes): CPU worker, update_every 8: 75.4 s per 900-slot episode with 112.5 updates; update_every 2: 181.1 s with 450 updates; marginal 0.31 s per update under the 15-worker layout (learner ≈ 50 % of the wall at update_every 8, ≈ 85 % at update_every 2; the reference DRM 17 % / 7 %). GPU worker (c06, update_every 2): 75.4 s per episode with 450 updates, ≈ 0.13 s per update. Exclusive, 1 thread, cProfile (the learner profiling script and its CPU logs at update_every 8 for A0 and A9): A0 110 ms per update early in a segment and 134 ms late (replay reconstruction 38 ms, backward 44 ms, forward passes 30 ms, Adam 16 ms); A9 53 ms. At update_every ≤ 4 the learner update, not the DRM, is the largest per-slot cost of the memory arms; it cannot be reordered without changing the gradients bit for bit, so it was not touched.

### A8.2 Optimisation (item 2 of the decision and its follow-up; `cfg.drm.impl = "fast"`, the original loops kept verbatim under `"reference"`)

Config switch `[drm] impl = "fast"` (`config/default.toml`); `tpa/drm/drm.py` `DRM.run` dispatches to `_run_fast` or `_run_reference` (the loop as it was before the DRM speed-up, unchanged apart from a diagnostic counter; verified in the independent code review against its text before the DRM speed-up), `tpa/baselines/executor.py` `Executor.run` likewise (`_run_fast` + `_pass2` vs `_run_reference`).

**First round and its attribution.** The first fast path kept one admission check per candidate but restricted the rate prediction to the rows read (`channel.rate_fdma_rows`) and, where the utility does not depend on the admission result (β_C = 0: every arm but A13), visited candidates in descending utility and took the first admitted one (the same argmax as the reference's strict-`>` loop; A13 and the Zhou / Qin rules check every candidate). The reading of 36 ms/slot after this round came from profiles contaminated by concurrent load (the full suite, the equivalence run and the code-review probes shared the machine) and from a wrapper list that missed the new admission function; the exclusive cProfile with builtins (the cProfile log of the fast path, 300 slots, A0) attributes the 18.3 ms/slot of that round completely: admission check 12.8 ms (147 calls × 87 µs, of which 11.0 ms in the row-restricted rate computation = 47,300 numpy-level calls per 300 slots), vectorised feasibility 1.8 ms, loop body 1.0 ms, burst / priority / window / finalize 0.7 ms. There was no 30 ms Python loop; the cost was numpy call overhead inside the per-candidate rate computation.

**What this machine's BLAS allows (the BLAS probe log, 300 random trials per check, numpy 2.5.4 with scipy-openblas 0.3.34).** The reference forms the (M × M) cross-cell matrix `full = p_kj.T @ w_kl` by one (25 × 200) @ (200 × 25) dgemm per interferer set. A dgemm column result here depends on the width of the right-hand matrix (columns copied into a 1 / 12 / 240 / 800-wide matrix differ from the 25-wide result in 299 / 32 / 90 / 90 of 300 trials), and a matrix–vector product differs from the dgemm column in 300 / 300. So the candidate rate matrices cannot be batched across candidates (one wide dgemm) nor replaced by matrix–vector products without changing bits: **one 25 × 200 × 25 dgemm per admission check (≈ 9 µs) is the floor**, and a Numba kernel could not reproduce OpenBLAS's accumulation order either. Reductions over the last axis of a 3-D array equal the 2-D row sums (0 / 300) and `np.log2` on an array equals its scalar / sub-array evaluation (0 / 300), so everything around the dgemm can be reorganised freely.

**Second round (final).**
1. `channel.FdmaScan`: one object per interferer prefix (x, n_l, inv_n, w_kl exactly as `rate_fdma`), rebuilt only after an accepted assignment (≈ 11 per slot). For a candidate (d, j) it forms the w_kl `rate_fdma` would form (column j replaced: x_j with x[d, j] = 1 times 1 / n_l[j], n_l[j] + 1), runs the one dgemm, and evaluates `cross`, `n_eff`, `zeta`, `bw`, `sinr`, `log2` at the entries the check reads only (elementwise, `rate_fdma`'s association order). `tests/test_d8_equivalence.py::test_fdma_scan_entries_bit_identical_to_rate_fdma`: bytes-equal to `rate_fdma` on the extended set at random entries.
2. `DRM._pair_batch`: the satisfiability check (`_pair`'s expressions) for all tasks still to visit × all UAVs in one array operation, recomputed only after an accepted assignment — the residual-budget update is the only sequential dependency, as the decision stated — plus one batch for the r_max pre-pass. `slot_admissible_scan`: `slot_admissible`'s arithmetic with constants hoisted, the per-UAV Eq. 16 wait precomputed per slot, rates from `FdmaScan.entries`.
3. Candidate order and the first-admitted rule as in the first round (utility-ordered when β_C = 0; every candidate for A13, Zhou and Qin; `prefix_order` for Huang and Song whose choice depends on a_ij only). The executor's pass 1 uses the same scan and batch; its pass 2 is the reference's.
4. `tpa/config.py` `Section`: config values bound as plain attributes (≈ 8,000 config reads per slot went through `__getattr__`); values, immutability, hashing and pickling unchanged (checked in the independent code review).

Only the diagnostic count `util_terms["n_admission_checks"]` differs between the implementations (fewer checks; the fast executor path now emits it too, after a minor code-review finding); it is not read by any decision, score or analysis.

### A8.3 Measured cost after the second round (exclusive, 1 thread; smoke-run logs of the fast path, 300 slots; per-arm logs of the fast and reference paths, 1,800 slots per arm)

Saturated-queue block (queue 128, window 32, seed 9001 medium load, untrained policy):

| Arm | DRM reference ms/slot | DRM fast ms/slot | factor | admission checks / slot (fast) | per check |
|---|---|---|---|---|---|
| A0 | 39.3 (300 slots) / 50.1 (10,800) | **10.0** | 3.9× | 147 | 44 µs (dgemm 9, w_kl copy and column 4, cross 3, elementwise tail 10, check body 10, overhead) |
| A9 | 61.0 (3,600) | 11.8 | 5.2× | 196 | 43 µs |
| A13 | — | 14.0 | | 194 (every candidate, β_C ≠ 0) | 49 µs |
| B_QIN | 68.0 (3,600) | 21.6 | 3.1× | 282 (every candidate; 24 assignments per slot) | 56 µs |

Per arm (1,800 slots each, same block; the per-arm logs): A0 41.4 → 11.1 ms/slot (3.7×); A1–A11 38.5–45.9 → 10.4–12.8 (3.3–4.0×); A12 68.7 → 18.1 (3.8×; 346 checks/slot); A13 32.6 → 13.3 (2.5×; every candidate checked); A14 36.8 → 11.1 (3.3×); B_ZHOU 54.5 → 24.2 (2.3×; 340 checks, 26 assignments/slot); B_QIN 44.2 → 19.8 (2.2×); B_HUANG 67.9 → 16.5 (4.1×); B_SONG 58.0 → 16.0 (3.6×). Admission checks per slot on the fast path 171–346; the code review's own table is in the code-review record.

Training regime (A0, the learner profiling script with `--no-cprofile`, 1,200 slots after warm-up, update_every 8, queue saturated at 128 under the untrained policy): DRM **12.4 ms/slot fast vs 44.6 reference** (3.6×); MEPR 1.5, retrieval 0.95, actor 0.9 unchanged. Remaining DRM cost per slot ≈ (admission checks) × 44 µs + 1.5 ms (batched feasibility, prefix rebuilds, window re-prediction, burst, priority, finalize); the number of checks is set by the physics (most window tasks exhaust every feasible UAV before being re-queued under a saturated queue), not by the implementation.

**Against the decision's target (≤ 5 ms/slot; stop and report if above ≈ 10):** the honest attempt lands at 10–12 ms/slot for the memory and memoryless DRM arms and 14–22 ms for A13 / the rule baselines on the saturated-queue block; the floor is one dgemm per admission check and ≈ 150–280 checks per slot. Reaching ≤ 5 ms bit-identically would require fewer checks, which the admission rule (residual rate budgets with the candidate as interferer) does not allow without computing the candidate's rates. Reported here and stopped, as ordered; the choice between the budget options and the server is to be decided. Tuning is NOT relaunched.

### A8.4 Equivalence test (A8.2, mandatory; `tests/test_d8_equivalence.py`; per-arm result files and the log of equivalence run 3)

For every arm (A0–A15 and B_ZHOU, B_QIN, B_HUANG, B_SONG), 3 validation blocks (seeds 9001 / 9002 / 9003, loads heavy / medium / heavy, adopted scenario) × 500 slots, plus — for every burst arm — one 200-slot block with `burst.theta_b` lowered to 0.05 (diagnostic override, never a campaign value) so that the burst on / off / cap / cooldown branch is exercised (found in code review: at θ_b = 0.75 burst never turns on in untrained validation blocks): the full simulator + controller run twice from identical streams and identical controller clones, once per implementation; compared per slot by exact equality: assignments X*(t), F*(t) (task_index, uav, f, t_total_pred), burst flags B(t+1) and budget, load index, reward, admission diagnostics (admit_rate, admit_delay, cpu_frac, rate_frac, r_max, n_feasible_pairs, burst_phi, eta_res, eta_res_target) and the simulator outcome (completed / failed / late tids, energy). Any divergence compounds through the state and fails at the first slot. **Result:** **36 passed, 0 failed** (20 arm blocks, 14 burst-forced blocks, 2 channel-level tests; 1,854 s). Branch coverage per arm: every arm shows feasible pairs (490k–761k over the three blocks), refused candidates (172k–485k on the DRM arms; the reference executor emits no count, the fast executor's counts are in the JSON), re-queued tasks (12k–31k), expired tasks (6–411), overflow (70k–89k) and raises (1.1k–232k); the burst on / off branch fires in the forced block of every burst arm (797–3,302 burst-on slots, 1,141–2,365 transitions per 200 slots) and, for A2, even at θ_b = 0.75 (1,477 / 2,760); `aborted` is a simulator outcome not reached in these blocks (minor code-review finding), not an Algorithm 3 branch.

### A8.5 Independent code review

Code-review record: **Approval of A8.2 (no decision path changed; equivalence test covers every branch): YES**, conditions (i) rows enqueued / claimed under code 044907211a93 and config bb9d6c8cbca1, any further `tpa/` edit lapses it; (ii) record numpy 2.5.4 / scipy-openblas 0.3.34 / CPU with the campaign and rerun the equivalence test before pooling results from another machine or BLAS build; (iii) the five minor findings are record items. The code review recovered the earlier `drm.py` / `executor.py` (before the DRM speed-up) from the project records and diffed them mechanically: `_run_reference` bodies equal the old `run` bodies except the rename, the docstring and the `n_raises` diagnostic line; no pre-existing function changed. Major finding (burst branch uncovered by the original blocks) closed by the burst-forced test. Code-review probes on the fallbacks (no scan, no rows, slot_admission off, access = "shared") all bit-identical.

### A8.6 Hashes and test suite

Code hash **044907211a93** (was c4ba4894f0b4), config hash **bb9d6c8cbca1** (was 203b2058f1b5; `drm.impl` is the only value change). Full suite at these hashes: **524 passed, 198 skipped, 0 failed** (608 s). The 102 stage-1 rows in `runs/runs.db` still carry code c4ba4894f0b4 (430 episodes of progress on 7 rows); the worker's hash check refuses them under the new code, so when the relaunch is ordered they are restarted, not resumed (a restart script with `--apply`: partial directories to a separate folder, rows re-enqueued with the new hashes). No relaunch until this amendment has been read.

### A8.7 Projection (A7.5) — not measurable as ordered (the 15-worker throughput needs the relaunch, which waits for a decision). Estimate from the training logs: at update_every 8 a CPU worker spent 12.9 s of a 75.4-s episode in the reference DRM; at 3.7× that becomes 3.5 s → 66 s per episode, i.e. ≈ 1.14× the A6.4 rate (≈ 108 slots/s instead of 95; the GPU workers and the update_every-2 rows gain less, the baselines slightly more). At 108 slots/s the counts of A7.5 take ≈ 41 days (stage 1 2.9 + stage 2 3.3 + main 30.1 + robustness 4.6) instead of 46.5. The learner update (A8.1) remains the dominant cost of the memory arms; a GPU-only or server option changes that term, the DRM term is now ≈ 15 % of a CPU worker's episode.


## Amendment 9 (2026-11-03/04, decision of 2026-11-03: checkpoint and replay memory; serialisation and storage layout only, no value that enters a decision, an update or a metric changes)

### A9.1 Motivation (the second incident of 2026-11-03, recorded in the code-review record)
Stage 1 started on the compute server with 63 workers; the per-worker memory planned from runs that never filled the replay buffer (about 1.5 GB) was wrong. At a full buffer a worker peaked at 14.5-18.5 GB (server VmHWM), almost all of it the checkpoint save: `torch.save` pickled the whole state into an in-memory buffer before writing (`tpa/run/checkpoint.py:48-55` at code 044907211a93). The OOM killer removed 5 runs, the VM froze and was rebooted; stage 1 was resumed with 14 workers. An investigation found three lossless reductions; C1 + C2 + C3 + the loader fix were approved, C1' and C1'' rejected.

### A9.2 The changes (a new code tree; only two files under `tpa/` change; `config/` unchanged)
- **C1 + C3, `tpa/run/checkpoint.py`.** `_dump` streams pickle protocol 5 through a zstd stream (pyarrow `CompressedOutputStream`) straight into the `.tmp` file; no in-memory copy of the state. Magic `TPAPKZ1\n`; without pyarrow it writes the existing plain-pickle format. `save` is unchanged (tmp + fsync + os.replace).
- **Loader fix, same file.** `load` reads the zstd pickle, the plain pickle and the pre-Amendment-9 `torch.save` zip, and applies `map_location` to every tensor in all three (the plain-pickle branch ignored it before). Limit: pickled tensors are rebuilt on the device they were saved from, so a checkpoint holding CUDA tensors needs CUDA to load; the compute server runs CPU only.
- **C2, `tpa/learner/replay.py`.** Row i stores S_sys, tf, a, reward, sigma, ref; its next state (S_sys2, tf2, sigma2, ref2) is row i+1's current state when the push that writes row i+1 brings bytes identical to row i's next state (`_nx[i]`), otherwise a side copy (`_side[i]`): the newest row and the last pushed transition of each episode (the final transition is dropped). `sample` draws the same indices with the same rng call and returns the same keys in the same order, byte for byte. `set_state` reads both formats; an old-format buffer is converted by the same bytewise rule. `live_segments` gives the same set. New `arrays()` expands to the ten pre-Amendment-9 arrays (diagnostics only).
- `tests/test_replay_index_and_checkpoint_formats.py` (outside the hash): the pre-Amendment-9 Replay copied verbatim as the reference; byte-identical samples, rng and live segments through wrap-arounds and episode ends; old-format conversion; the three checkpoint formats and map_location.

### A9.3 Proofs (laptop, Windows 11, numpy 2.5.4, torch 2.14.1 CPU; proof records, summary `SUMMARY.json`)
Each proof run mirrors a stage-1 row (`scripts/tuning.py:159-172`): adopted scenario (A2.12) + `runs/tune/configs.json[arm][0]` (c00) + 140 episodes, validation seed 9004, workload sinusoidal, phase tune with the end-of-run val_full evaluation (never a test seed). The overrides equal the server rows `tune1-<arm>-c00-s9004` except `run.episodes`; with 300 episodes the config hash reproduces the rows' (A0 9b30adfcf7c0, A2 3683fc4f3b12, A10 e1d040d6135e, A11 1c490be04e93). A10 (LSTM baseline, own sequence replay) is a stage-1 tuning arm and ran its own c00. Criterion: bit for bit, except wall-clock fields, which are listed: train `wall_s`, `t_*_ms`; val_full additionally `reaction_ms_p50`, `reaction_ms_p90`, `recovery_ms` (per-decision wall-clock of `act()`, `tpa/run/evaluator.py:50, 77-80`). Compared (`p_compare.py`): the full rolling checkpoint at episode 140 (trainer state, all 11 RNG streams, simulator, the whole controller state incl. networks, optimiser states, memory stores, normaliser, and the replay buffer expanded to its ten arrays), `ckpt_best.pt`, `ckpt_final.pt`, the train table (140 rows, 52 columns) and the val_full table (1629 rows, 50-51 columns), `RESULT.json` (except `train_wall_s`). Negative control: one ulp changed in a replay row and in one train cell is detected, and nothing else.
- **Proof 1** old (044907211a93) vs new (02e2abd853f8), 140 episodes: **A0, A2, A10, A11 all bit-identical.**
- **Proof 2** cross-format resume: old code to episode 120 (old-format rolling checkpoint), new code resumes to 140; equal to old uninterrupted and to new uninterrupted: **A0 and A10, all three bit-identical.**
- **Proof 3** at the final tree (code 02e2abd853f85cfe…): full suite **567 passed, 198 skipped, 0 failed** (incl. tests/test_replay_index_and_checkpoint_formats.py 3, the heartbeat tests 4, `test_checkpoint_resume_bitwise` A0/A3/A9 + novelty, `test_d8_equivalence.py` 36); `tests/test_d8_equivalence.py` run explicitly: **36 passed** (`p3_final_suite.log`; equivalence-suite log). Interim run before test_replay_index_and_checkpoint_formats.py existed: 564 passed, 198 skipped, 0 failed.
- **Proof 4** peak RSS (psutil, 20 ms; Windows working set; the old code's 15.7-19.0 GB matches the server's VmHWM 14.5-18.5 GB):

| arm | old: save peak | new: run to 140 incl. saves | new: resume from new-format ckpt (then train, save) | new: resume from old-format ckpt (once, at the switch) | ckpt at ep 140 old -> new | save s old -> new |
|---|---|---|---|---|---|---|
| A0 | 15.79 | 3.95 | 3.77 / 4.31 / 4.39 | 7.49 | 2.95 -> 1.15 GB | 31 -> 9 |
| A2 | 17.58 | 4.56 | 4.34 / 5.10 / 5.17 | 8.22 | 3.28 -> 1.37 GB | 36 -> 10 |
| A10 | 15.73 | 4.54 | 4.32 / 6.33 / 6.40 | 7.23 | 2.73 -> 1.02 GB | 29 -> 7 |
| A11 | 19.05 | 4.08 | 3.85 / 4.08 / 4.12 | 9.43 | 3.72 -> 0.97 GB | 30 -> 5 |

Training wall time new vs old: -2 % to +7 % with other runs sharing the laptop (within that noise).

### A9.4 Hashes
Code hash **02e2abd853f8** (02e2abd853f85cfe2efdad5b433797cf1116fa5ec8dc234984b08ff1da47d6e1; was 044907211a93). Config hash unchanged (no change under `config/`). Any further edit under `tpa/` voids these proofs.

### A9.5 Rows that span both codes
The worker refuses a row whose code hash differs from the tree (`scripts/run_queue.py:226-227`); `--allow-code-change` is never used. After the queue is stopped, a restamp script changes only `code_hash` (old -> new), only on pending rows carrying the old hash; done, failed and running rows are never touched. It writes `runs/tune/AMENDMENT9_RESTAMP.json` and, in every started run's directory, `AMENDMENT9_RESUME.json` (episodes_done at the switch, rolling-checkpoint size and sha256, old and new hash, utc, the pre-switch manifest's code_hash and created_utc and a copy of that manifest; the trainer rewrites manifest.json on resume, `tpa/run/trainer.py:114`). A run that spans the switch continues from its old-format rolling checkpoint under the new code; proof 2 shows this is bit-identical to running either code throughout, so it is analysed as one run, and its marker records the switch.


## Amendment 10 (2026-11-04, decision incl. its Revision 12:45 UTC+3: schedule; FINAL)

Goal: finish tuning, main campaign and robustness in about 10 days, with no change to the primary test (A0 vs A3, Tier A n = 30) and nothing decided after looking at outcomes. Written before stage 1 finished and before any stage-2 validation curve exists. No new amendment after this one unless a run fails or the server breaks.

### A10.1 R1 Placement (compute only)
Read-only NUMA check of the compute server's VM (note of 2026-11-04 on throughput, "Action now" 1). If the 20 workers sit on one node, a launcher option outside the code hash pins workers alternately across nodes; it is tested for 10 min with one benchmark process per node and applied at the next restart (stage 2 start). If the check shows no imbalance, R1 is dropped and the result is recorded. Time box 2 h. Pinning changes no computed value (same CPU model and instructions; acceptance and the equivalence suite (A8.4) passed on this machine). Result: A10.7.

### A10.2 R2 Robustness family on the laptop
MMPP + Gauss-Markov x A0/A3/A9/A10 x n = 6 (48 runs) run entirely on the laptop (64 GB, Amendment 9 code, 6 workers at 6.1 GB), started right after the stage-2 readout. The family is analysed on its own; nothing from the laptop is pooled with server runs. The machine is recorded in every run manifest. It has its own short checklist at its end.

### A10.3 R3 Training length rule (verbatim; editorial changes in square brackets; fixed now, before any stage-2 curve exists)
E_main = the smallest E in {500, 600, 700, 800, 900, 1000} such that, for EVERY tuned arm's selected configuration, the stage-2 validation score at episode E is within 1.0 pp (completion metric) of its maximum over episodes <= 1000, using the validation checkpoints every val_every_episodes. One E for all arms and baselines. If no E < 1000 satisfies it, E_main = 1000.

The computed E_main and the table that produced it are added here at [the stage-2 readout], before `tuning.py score2`/`select` results are used to enqueue the main campaign.

### A10.4 R4 Tier B n = 8
Tier B n = 8 (was 12, A7.1) for A1, A2, A4-A8, A11, A14 and the four baselines (B_ZHOU, B_QIN, B_HUANG, B_SONG). Tier A (A0, A3, A9, A10, A12; n = 30) and Tier C (A13, n = 6) unchanged. Test seeds 1..8 are the first 8 of the 12, never re-chosen. Robustness n = 6 unchanged. Main-campaign row count: 5 x 30 + 13 x 8 + 1 x 6 = 260 runs x loads as the plan defines (A7.1's 312 replaced). The MDE statement of A7.1 is recomputed for n = 8 by the analysis script.

### A10.5 Readouts (Revision; replaces the automatic starts of the decision's "Stage 2" and "Main campaign" sections)
Each readout is a fixed checklist; the review takes under one hour and never reopens the plan.
- **Stage-1 readout (end of stage 1 -> stage 2).** The tuning result with: per arm the 6 configuration scores, the chosen top 2 and their margin; failed/requeued runs; learning curves (train completion vs episode, one small figure per arm, png); the M3 mechanism check on A0 vs A15 from the stage-1 runs (outcome-blind). Stop conditions (stop and report): any failed run; any arm whose 6 scores are all within noise (no tuning signal); any curve still rising steeply at episode 300 for a Tier A arm; M3 failing. Otherwise "stage-1 readout ready" is recorded; stage 2 starts on "approved". Stage 2 = top 2 x seed 9005 x 1000 episodes (A7.2).
- **Stage-2 readout (end of stage 2 -> main campaign).** The tuning result extended with the stage-2 validation curves per arm, the selected configuration per arm, E_main by A10.3 with its table, and the planned row count (A10.4). Stop conditions: failed runs; an arm whose selected configuration is worse than its stage-1 score by more than 2 pp; E_main not attainable. The main campaign starts on "approved" (enqueue dry run must show 260 runs x loads). Robustness (A10.2) starts on the laptop right after the stage-2 readout.
- **Early-campaign check (first 10 % of the main campaign, about 26 runs).** Failure count, per-arm throughput, disk and memory; the projection is confirmed. No outcome is looked at.
- **Registered analysis (end of main campaign and robustness).** The analysis script runs once; the full result tables are reviewed before anything goes into the manuscript.
- Worker count stays at 20 (or the R1 placement at 20) unless MemAvailable < 15 GB (the trigger set in the decision of Amendment 9).

### A10.6 Operationalisation of the stage-1 readout checklist (declared before stage 1 finished and before any score was read for this purpose)
- **Scores.** Per configuration: `best_val` of its stage-1 run (`RESULT.json`; medium-load validation completion of the selected checkpoint, A7.2), computed read-only exactly as `scripts/tuning.py:_score` and `cmd_score1`, without enqueuing (`cmd_score1` enqueues stage 2, so it runs only after "approved"). Top 2 = the two highest; margin = score(1st) - score(2nd), and score(2nd) - score(3rd) is shown too.
- **Failed/requeued.** From `runs.db` (state failed; attempts > 1; `error` text) and the worker logs.
- **Noise and "no tuning signal".** Noise of arm a = pooled standard error of the medium-load completion over the 3 end-of-run `val_full` runs per configuration (`SE_c = sd/sqrt(3)`, noise_a = sqrt(mean_c SE_c^2)); the 6 scores are "all within noise" iff max - min <= 2 x sqrt(2) x noise_a (the two-sided 95 % range of a difference of two independent scores). If `val_full` medium-load rows are missing, the check is reported as not computable (stop and report).
- **[Superseded by A10.6 revision 1 below]** **"Still rising steeply at episode 300" (Tier A arms A0, A3, A9, A10, A12).** Per configuration, train completion per episode from the train parquet; because the load changes by segment, the rise is the mean over loads of [mean completion of the episodes 251-300 at that load - mean over episodes 201-250 at that load]. Steep iff this rise > 2.0 pp for any of the arm's 6 configurations. Reported for every arm; the stop applies to Tier A.
- **M3 (A5.3 statistic, outcome-blind).** For each of the 6 A0 stage-1 configurations: the A0 memory parameters of its selected checkpoint (`ckpt_best.pt`, `ctrl.params`) are loaded into the Amendment 5 synthetic driver (`tests/test_memory_driver.py` Driver, `zone_mean_mode="cumulative"`, P_d = 600, driver seeds 0, 1, 2, run length w + 3 P_d, min_age = P_d/2) for A0 and for the key-shuffled control A15 (same parameters, same driver seed); statistic `age_matched_phase_share`; pass iff A0 - A15 >= 0.20 on each driver seed, for every configuration. No completion or reward is computed. Script: the stage-1 report script (outside the code hash).

### A10.7 R1 result (2026-11-04, within the 2 h box)
Read-only check (09:18 server clock): the VM has 2 NUMA nodes (vCPUs 0-31 node 0, 32-63 node 1; 64 GB each). All 20 workers were pinned to vCPUs 1-20, i.e. all on node 0; node 0 had 0.84 GB free against 32.9 GB on node 1, and 3-32 % of each worker's pages already lay on node 1 (remote). Launcher option `scripts/run_queue.py --numa-alternate` (outside the code hash; order node0, node1, node0, ...; tests/test_infra_run_queue.py::test_numa_alternate_order; dry run on the server pinned 1, 32, 2, 33). Benchmark (7 episodes of a copy of tune1-A6-c05-s9004 at episode 160, new-format full-buffer checkpoint, writing only to a separate benchmark folder): vCPU 28 (node 0) 167-190 s per episode = 185-211 ms per slot, 59 % of its pages remote; vCPU 44 (node 1) 115-118 s = 128-131 ms per slot, 99 % local: node 1 about 1.5x faster. R1 applies: the tuning launch script now passes `--numa-alternate` by default (NUMA_ALTERNATE=1) from the next restart, the stage-2 start (10 workers per node at 20 workers). The running stage-1 workers are not restarted.

### A10.6 revision 1 (answer of 2026-11-04 13:30 UTC+3, option (b)): "rising steeply" replaced
The A10.6 bullet "Still rising steeply at episode 300" is replaced by the rule as stated in that answer, verbatim:
  rise = mean(load-matched train completion, episodes 251-300) - mean(episodes 201-250);
  SE from the 50-episode samples of both segments (pooled, independent); flag if rise > 2.0 pp
  AND rise / SE > 2. Applied identically to all 17 arms and all 6 configurations.
Reason: the 2.0 pp threshold alone is inside noise (SE 1.1-1.3 pp, sign swings on flat curves), so A10.6 as written was a noise detector, not a trend detector; choosing (b) before any computation on it keeps the stage-1 readout outcome-blind. A flag is a stop-and-report condition only; it never alters a run.
Computation (stage-1 report script): for each load l, d_l = mean over the episodes 251-300 at load l minus mean over the episodes 201-250 at load l; pooled variance s_l^2 = ((n_a - 1) s_a^2 + (n_b - 1) s_b^2) / (n_a + n_b - 2) of the two independent samples; SE_l = s_l sqrt(1/n_a + 1/n_b); rise = mean_l d_l; SE = sqrt(sum_l SE_l^2) / L (loads independent). Episodes are numbered from 1 here (code indices 200-249 and 250-299). A flag on any configuration of any arm is a stage-1 readout stop-and-report (the decision applies the rule to all 17 arms; the earlier Tier-A-only stop is superseded). [Stop status superseded by revision 2: a flag is a disclosure.]
Disclosure: while checking the noise of the original rule (before the question was asked), the implementer computed per-episode z = rise / SE with an unpooled independent SE for the finished runs of A0-A3 (code-review record, entry "Amendment 10: R1 done ..."; e.g. A0 c02 z = +3.4). That z is close to the adopted statistic, so its outcome on those runs was partly known to the implementer, though not shown to the decision-maker, when the rule was chosen. The question's statement that option (b) would not be computed on the real data referred to the segment-mean variant proposed there.

### A10.6 revision 2 (decision of 2026-11-04 14:50 UTC+3, appended to the answer of revision 1): a "still rising" flag is a disclosure, not a stop
Decision (pre-registered, applies to all arms alike): at the stage-1 readout a "still rising" flag (A10.6 revision 1) is a DISCLOSURE, not a stop. The report lists every flagged configuration with rise and z; the stage-1 readout is READY unless one of the other stop conditions holds (failed run, no tuning signal, M3 failing). A flagged configuration among an arm's top 2 is acceptable: stage 2 will show its plateau.
Reason: stage-1 runs are 300 episodes by design (a ranking step), while stage 2 trains the selected configurations to 1000 episodes. A significant rise in episodes 251-300 therefore means "still learning at 300", which is expected for some configurations and is exactly what stage 2 and the R3 plateau rule (A10.3) exist to handle; it is not a defect of the run or of the ranking. The disclosure of revision 1 is accepted and noted; it does not change the decision, which was made on the noise argument, not on any flagged value.
Revision 1's flag computation is unchanged. The stage-1 report script lists the flags (and whether a flagged configuration is in its arm's top 2) as disclosures and no longer counts them as a stop condition.

### A10.8 Operationalisation of the stage-2 readout checklist (declared 2026-11-06, while stage 2 was running: 18 of 34 rows done; before any stage-2 score or validation curve was read by the implementer, who had seen only row states and episode counts)
Script: the stage-2 report script (outside the code hash; read-only; it never enqueues and never runs `tuning.py select`). Outputs: the stage-2 tuning result (the stage-1 readout files are not overwritten), the stage-2 report (JSON) and the stage-2 figures (one png per arm).
- **Rows.** The 34 stage-2 rows (`tune2-<arm>-c<i>-s9005`, 17 tuned arms x top 2) from `runs.db`. Any failed stage-2 row: stop and report. The report is final only when all 34 are done.
- **Stage-2 score and selection.** Per configuration: `best_val` of its stage-2 run (`RESULT.json`), read exactly as `scripts/tuning.py:_score`. Selected configuration of an arm = the one of its two with the higher stage-2 score, exactly as `cmd_select` (`max` over the finite scores; on an exact tie the configuration ranked first in stage 1). `tuning.py select` itself (it writes `runs/tune/selected.json` and overwrites the tuning result) runs only after "approved", after the tuning result has been saved as the stage-1 tuning result.
- **Stop: selected configuration worse than its stage-1 score by more than 2 pp.** Stage-2 score (seed 9005, 1000 episodes) minus stage-1 score (seed 9004, 300 episodes; `runs/tune/stage1_scores.json`) of the selected configuration < -2.0 pp, for any arm.
- **[Superseded by A10.8 revision 1 below]** **R3, E_main (rule A10.3, read literally).** For each tuned arm's selected configuration, v(e) = the validation score logged at episode e in `RESULT.json` `val_curve` (medium-load validation completion, every `val_every_episodes` = 50 episodes and at the end), v_max = max of v(e) over e <= 1000. The arm satisfies E iff v_max - v(E) <= 0.010 (float64, with 1e-12 slack against rounding). E_main = the smallest E in {500, 600, 700, 800, 900, 1000} that every tuned arm satisfies; if none below 1000 does, E_main = 1000. The report gives the table v(500) ... v(1000), v_max and its episode, and the smallest satisfying E per arm. For information only (not used by the rule): the same computed on the best-so-far curve max over e' <= E of v(e'), which is what checkpoint selection would retain.
- **[Superseded by A10.8 revision 1 below]** **Stop: E_main not attainable.** Read as: no E in the set, including 1000, is satisfied by every arm (i.e. E_main = 1000 by the fallback while at least one arm's v(1000) is more than 1.0 pp below its own maximum). Then the fallback value is reported and the stage-2 readout stops for a decision.
- **Planned row count.** Tier A (A0, A3, A9, A10, A12) seeds 1-30; Tier B (A1, A2, A4, A5, A6, A7, A8, A11, A14, B_ZHOU, B_QIN, B_HUANG, B_SONG) seeds 1-8; Tier C (A13) seeds 1-6: 150 + 104 + 6 = 260 rows, workload sinusoidal, each run evaluated on the loads the plan defines. `scripts/enqueue.py` takes one override set per call and does not read `selected.json`, so the main campaign is enqueued per arm with that arm's selected overrides and `run.episodes = E_main` by `scripts/enqueue_main.py` (outside the code hash), whose `--dry-run` prints the rows without touching the queue; the stage-2 readout requires its total to be 260.
- **Curves.** One figure per arm: the stage-2 validation curves of its two configurations, the selected one marked, E_main and the 1.0 pp band shown.

### A10.8 revision 1 (answer of 2026-11-06 15:40 UTC+3, option (b)): R3 on the best-so-far curve
Written while stage 2 was at 18 of 34 rows done; decided before any stage-2 score or curve was read by the decision-maker or the implementer. The A10.8 bullets "R3, E_main (rule A10.3, read literally)" and "Stop: E_main not attainable" are replaced by the rule as stated in that answer, verbatim:
  best(E) = max over validation checkpoints at episodes <= E of the validation score.
  An arm satisfies E iff best(E) >= best(1000) - 1.0 pp.
  E_main = the smallest E in {500, 600, 700, 800, 900, 1000} satisfied by EVERY tuned arm's
  selected configuration (one E for all arms and baselines). E = 1000 is always satisfied, so
  "not attainable" cannot occur and that stop is removed.
Reason: best-so-far matches what a main-campaign run of E episodes actually evaluates, the best-validation checkpoint up to E; the literal reading would let a single dip at one checkpoint decide E_main, which is noise, not learning.
Disclosure in the stage-2 report (not used by the rule): for each arm the episode of best(1000), and the literal-reading result (A10.8 as first declared) next to the best-so-far one. The other stop conditions stay: a failed stage-2 run; a selected configuration more than 2.0 pp below its stage-1 score; a planned row count other than 260. Computation in the stage-2 report script: validation checkpoints from `RESULT.json` `val_curve`, float64, 1e-12 slack.


### A10.8 revision 2 (2026-11-07 19:45 UTC+3, section 3 of the decision of Amendment 12): condition 2 restated as a same-seed degradation check
**Rule.** "For each selected configuration, the validation score at the last validation checkpoint (episode 1000) is not more than 2.0 pp below its validation score at episode 300, on the same seed (9005)." Computed for all 17 tuned arms when their stage-2 data are final (stage-2 report script, field `rev2`; `RESULT.json` `val_curve`). It replaces the stop "selected configuration worse than its stage-1 score by more than 2 pp" of A10.5 / A10.8 for the stage-2 readout.

**Disclosure.** This rule was restated AFTER the stage-2 readout stop of 2026-11-07 and after the implementer had read the stage-2 scores. The first wording (A10.8: stage-2 score on seed 9005 minus stage-1 score on seed 9004) compared two validation seeds; it stopped the stage-2 readout for every arm (11.07-23.47 pp below stage 1). Test (c) (decision of 2026-11-07 on condition 2, PASS criterion fixed before the test ran) re-evaluated the stage-1 best checkpoints on the seed-9005 validation block: for all 28 configurations of the 14 arms whose checkpoints held their learned state, 85-111 % of the gap appears when only the evaluation seed changes (criterion: >= 75 %; records: the seed-check outputs and the code-review record's entry on test (c)). The seed changes the world: seed 9005 has path-loss exponent 2.89 vs 3.37, radio capacity 149 vs 94 tasks/s, and with `load_basis = "binding"` 1.59x more offered tasks for 1.07x more compute. Condition 2 exists to catch a selected configuration that degrades with longer training; the cross-seed wording could not separate that from the seed. The intermediate wording of the 2026-11-07 conditional approval (best-so-far up to 1000 vs the score at 300) could never fail, since the best-so-far includes episode 300 (found by the implementer); the rule above uses the score at the last checkpoint instead. The cross-seed difference stays in the stage-2 report as a disclosure.

### A10.9 Amendment 10 result: stage-2 readout and E_main (2026-11-08, written before any main-campaign row is enqueued)
Stage-2 report (the stage-2 tuning result / its JSON version, sha256 b54112227e381271...), generated 2026-11-08T02:59:44+00:00 (server clock) under code 2892281e5ba9: **stage-2 readout ready**. Conditions of the pre-approval of 2026-11-06 as amended by the decision of 2026-11-07 (Amendment 12): (1) no failed stage-2 row, all 17 tuned arms have a selected configuration; (2) A10.8 revision 2 holds for all 17 (table below; worst -1.09 pp, limit -2.0 pp), test (c) passes for all 17 arms and all 34 configurations (seed-only share of the gap 0.85-1.11, rule >= 0.75; seed-check outputs), and after Amendment 12 the stage-1 rows of A12, B_HUANG and B_SONG reproduced their previous results exactly, so their top 2 and their stage-2 runs stand; (3) **E_main = 500** by A10.8 revision 1 (best-so-far; every arm satisfies E = 500); literal reading (disclosure): 1000 by its fallback, not attained by every arm; (4) planned rows 260.

## 1. Stage-2 scores and the selected configuration (completion, %)

| Arm | 1st in stage 1 | stage 1 | stage 2 | 2nd in stage 1 | stage 1 | stage 2 | selected | ep 1000 - ep 300, seed 9005 (pp; rev 2) | stage 2 - stage 1, cross-seed (pp; disclosure) | tuned parameters |
|---|---|---|---|---|---|---|---|---|---|---|
| A0 | c02 | 36.85 | 20.65 | c04 | 33.45 | 20.66 | c04 | +0.52 | -12.79 | `{"learner.lr": 0.00239, "learner.update_every": 8, "learner.target_entropy_scale": 0.25, "burst.theta_b": 0.9}` |
| A1 | c01 | 33.86 | 20.58 | c00 | 32.78 | 20.76 | c00 | +0.81 | -12.02 | `{"learner.lr": 0.00133, "learner.update_every": 2, "learner.target_entropy_scale": 0.25, "burst.theta_b": 0.6}` |
| A2 | c03 | 33.69 | 20.49 | c04 | 33.24 | 20.50 | c04 | +0.21 | -12.74 | `{"learner.lr": 0.00234, "learner.update_every": 4, "learner.target_entropy_scale": 1.0, "burst.theta_b": 0.6}` |
| A3 | c03 | 33.56 | 20.99 | c02 | 33.16 | 20.86 | c03 | -1.02 | -12.57 | `{"learner.lr": 0.000734, "learner.update_every": 8, "learner.target_entropy_scale": 1.0, "burst.theta_b": 0.6}` |
| A4 | c02 | 34.42 | 20.73 | c05 | 33.63 | 20.72 | c02 | -0.76 | -13.70 | `{"learner.lr": 0.000474, "learner.update_every": 2, "learner.target_entropy_scale": 1.0, "burst.theta_b": 0.9}` |
| A5 | c01 | 35.38 | 21.30 | c02 | 34.14 | 20.34 | c01 | +1.36 | -14.08 | `{"learner.lr": 0.000589, "learner.update_every": 4, "learner.target_entropy_scale": 1.0, "burst.theta_b": 0.75}` |
| A6 | c03 | 38.67 | 20.90 | c02 | 36.92 | 21.30 | c02 | -0.67 | -15.62 | `{"learner.lr": 0.00111, "learner.update_every": 4, "learner.target_entropy_scale": 1.0, "burst.theta_b": 0.75}` |
| A7 | c01 | 33.70 | 20.80 | c02 | 32.69 | 21.35 | c02 | -0.64 | -11.35 | `{"learner.lr": 0.00175, "learner.update_every": 8, "learner.target_entropy_scale": 0.25, "burst.theta_b": 0.9}` |
| A8 | c02 | 35.12 | 20.97 | c05 | 34.87 | 21.06 | c05 | +0.54 | -13.81 | `{"learner.lr": 0.0028, "learner.update_every": 8, "learner.target_entropy_scale": 0.25}` |
| A9 | c05 | 35.58 | 21.34 | c02 | 33.96 | 22.02 | c02 | -1.09 | -11.94 | `{"learner.lr": 0.000354, "learner.update_every": 4, "learner.target_entropy_scale": 1.0, "burst.theta_b": 0.75}` |
| A10 | c00 | 37.40 | 20.73 | c02 | 33.80 | 21.07 | c02 | +0.00 | -12.73 | `{"learner.lr": 0.00141, "learner.update_every": 4, "learner.target_entropy_scale": 0.5, "burst.theta_b": 0.75, "baselines.lstm_hidden": 48}` |
| A11 | c03 | 35.37 | 21.37 | c00 | 34.48 | 21.45 | c00 | -0.04 | -13.03 | `{"learner.lr": 0.000581, "learner.update_every": 2, "learner.target_entropy_scale": 0.5, "burst.theta_b": 0.9}` |
| A12 | c02 | 40.52 | 21.25 | c00 | 40.34 | 21.44 | c00 | +0.00 | -18.90 | `{"baselines.lookup_ewm": 0.1, "baselines.lookup_f_scale": 1.0, "burst.theta_b": 0.9}` |
| A13 | c05 | 37.20 | 21.04 | c02 | 36.14 | 25.07 | c02 | +0.05 | -11.07 | `{"learner.lr": 0.00179, "learner.update_every": 8, "learner.target_entropy_scale": 1.0, "burst.theta_b": 0.75, "drm.beta_c": 0.1}` |
| A14 | c04 | 33.05 | 20.44 | c05 | 31.24 | 20.25 | c04 | -0.75 | -12.61 | `{"learner.lr": 0.00102, "learner.update_every": 2, "learner.target_entropy_scale": 1.0, "burst.theta_b": 0.75}` |
| B_HUANG | c03 | 46.47 | 21.19 | c02 | 44.83 | 21.36 | c02 | +0.00 | -23.47 | `{"learner.lr": 0.000549, "learner.update_every": 2}` |
| B_SONG | c04 | 43.01 | 22.82 | c03 | 42.99 | 23.59 | c03 | +0.16 | -19.40 | `{"baselines.es_sigma": 0.00534, "baselines.es_lr": 0.00599}` |

## 2. R3 table: best(E) = best validation score up to episode E of the selected configuration, in % (rule: best(E) >= best(1000) - 1.0 pp)

| Arm | selected | 500 | 600 | 700 | 800 | 900 | 1000 | best(1000) (its episode) | smallest E satisfied | literal smallest E (disclosure) |
|---|---|---|---|---|---|---|---|---|---|---|
| A0 | c04 | 20.66 | 20.66 | 20.66 | 20.66 | 20.66 | 20.66 | 20.66 (50) | 500 | 500 |
| A1 | c00 | 20.67 | 20.67 | 20.67 | 20.67 | 20.67 | 20.76 | 20.76 (950) | 500 | 700 |
| A2 | c04 | 20.50 | 20.50 | 20.50 | 20.50 | 20.50 | 20.50 | 20.50 (150) | 500 | none |
| A3 | c03 | 20.72 | 20.72 | 20.99 | 20.99 | 20.99 | 20.99 | 20.99 (700) | 500 | 700 |
| A4 | c02 | 20.73 | 20.73 | 20.73 | 20.73 | 20.73 | 20.73 | 20.73 (250) | 500 | none |
| A5 | c01 | 21.30 | 21.30 | 21.30 | 21.30 | 21.30 | 21.30 | 21.30 (150) | 500 | 500 |
| A6 | c02 | 21.30 | 21.30 | 21.30 | 21.30 | 21.30 | 21.30 | 21.30 (200) | 500 | 500 |
| A7 | c02 | 21.35 | 21.35 | 21.35 | 21.35 | 21.35 | 21.35 | 21.35 (450) | 500 | none |
| A8 | c05 | 21.06 | 21.06 | 21.06 | 21.06 | 21.06 | 21.06 | 21.06 (50) | 500 | 500 |
| A9 | c02 | 21.75 | 21.75 | 22.02 | 22.02 | 22.02 | 22.02 | 22.02 (650) | 500 | none |
| A10 | c02 | 21.07 | 21.07 | 21.07 | 21.07 | 21.07 | 21.07 | 21.07 (250) | 500 | 500 |
| A11 | c00 | 21.08 | 21.08 | 21.08 | 21.45 | 21.45 | 21.45 | 21.45 (800) | 500 | 600 |
| A12 | c00 | 21.44 | 21.44 | 21.44 | 21.44 | 21.44 | 21.44 | 21.44 (50) | 500 | 500 |
| A13 | c02 | 25.07 | 25.07 | 25.07 | 25.07 | 25.07 | 25.07 | 25.07 (250) | 500 | none |
| A14 | c04 | 20.31 | 20.31 | 20.31 | 20.31 | 20.44 | 20.44 | 20.44 (850) | 500 | 500 |
| B_HUANG | c02 | 21.36 | 21.36 | 21.36 | 21.36 | 21.36 | 21.36 | 21.36 (50) | 500 | 500 |
| B_SONG | c03 | 23.39 | 23.39 | 23.59 | 23.59 | 23.59 | 23.59 | 23.59 (700) | 500 | 500 |

`x` = best(E) more than 1.0 pp below best(1000). E_main is the smallest E with no `x` in its column. The last column is the literal reading (the score logged at E itself within 1.0 pp of the maximum; `none` = not satisfied at any E in the set), shown so the effect of the choice is visible.

A10.8 revision 2 (same seed 9005, selected configuration, validation score at the last checkpoint vs episode 300):

| Arm | selected | episode 300 | episode 1000 | difference (pp) |
|---|---|---|---|---|
| A0 | c04 | 19.84 | 20.36 (ep 1000) | +0.52 |
| A1 | c00 | 19.79 | 20.60 (ep 1000) | +0.81 |
| A2 | c04 | 19.00 | 19.20 (ep 1000) | +0.21 |
| A3 | c03 | 20.72 | 19.70 (ep 1000) | -1.02 |
| A4 | c02 | 19.04 | 18.28 (ep 1000) | -0.76 |
| A5 | c01 | 18.41 | 19.77 (ep 1000) | +1.36 |
| A6 | c02 | 20.84 | 20.18 (ep 1000) | -0.67 |
| A7 | c02 | 19.63 | 18.99 (ep 1000) | -0.64 |
| A8 | c05 | 18.70 | 19.24 (ep 1000) | +0.54 |
| A9 | c02 | 21.39 | 20.29 (ep 1000) | -1.09 |
| A10 | c02 | 20.90 | 20.90 (ep 1000) | +0.00 |
| A11 | c00 | 20.09 | 20.05 (ep 1000) | -0.04 |
| A12 | c00 | 21.44 | 21.44 (ep 1000) | +0.00 |
| A13 | c02 | 21.72 | 21.77 (ep 1000) | +0.05 |
| A14 | c04 | 20.03 | 19.28 (ep 1000) | -0.75 |
| B_HUANG | c02 | 21.17 | 21.17 (ep 1000) | +0.00 |
| B_SONG | c03 | 22.94 | 23.10 (ep 1000) | +0.16 |

## 3. Planned main-campaign rows (A10.4; `scripts/enqueue_main.py`, dry computation, nothing enqueued)

| Tier | Arms | seeds | rows |
|---|---|---|---|
| A | A0, A3, A9, A10, A12 | 1-30 | 150 |
| B | A1, A2, A4, A5, A6, A7, A8, A11, A14, B_ZHOU, B_QIN, B_HUANG, B_SONG | 1-8 | 104 |
| C | A13 | 1-6 | 6 |

Total: **260** rows (required 260), `run.episodes = 500`, workload sinusoidal, phase main.

## Amendment 11 (2026-11-05, decision incl. the three conditions of its review: bug fix only, B_HUANG and B_SONG on the trainer path)

### A11.1 What failed
All 6 stage-1 rows of B_HUANG and all 6 of B_SONG failed on the compute server on 2026-11-04 (20:29-20:37 server clock), each before its first rolling checkpoint (episodes_done 0, attempts 1); the failed run directories hold only `manifest.json`. No other arm had a failure. These two baselines had never produced a result, so no result is affected.

### A11.2 Diagnosis (the code of Amendment 9, code hash 02e2abd853f8; the diagnosis record and the answer to the first two of the review's conditions)
- **B_HUANG, `KeyError: 'alpha'`.** `tpa/run/trainer.py:91` (`_ep_row`) reads `u["alpha"]` from every update record; DDPG's `tpa/baselines/huang.py:89` returned only `critic_loss` and `actor_loss`. First update at slot 2000 (`learner.warmup_slots`), so the third episode's row raised.
- **B_SONG, `slots must be consecutive: expected 900, got 1800`.** The reviewed pairing (`trainer.py:148-166`) replays the even window for the odd episode and starts the next even episode at `ep * T_ep`; inside a segment the Simulator forbids that jump (`tpa/env/simulator.py:275-276`). The existing test passed only because its config has one pair per segment (`tests/_sys_helpers.py:6-13`).
- **B_SONG, latent.** The ES update record (`tpa/baselines/song.py:128`) had none of `alpha`, `critic_loss`, `actor_loss`; `_ep_row` would have raised at the first ES update (episode 13).
- **Not affected: every other arm.** Only B_SONG sets `paired_episodes` (`song.py:83`; measured for all 20 arm keys on both trees with a path-check script). B_ZHOU, B_QIN and A12 emit no update record, so `_ep_row` takes its NaN branch; their episodes start at consecutive windows; the slot guard raises for every arm, so no window can be skipped silently. B_ZHOU and B_QIN have no stage-1 rows (nothing to tune).

### A11.3 The fix (a new code tree; three files under `tpa/`, `config/` unchanged)
- `tpa/baselines/huang.py`: the update record also carries `alpha = NaN`.
- `tpa/baselines/song.py`: the ES update record also carries `alpha`, `critic_loss`, `actor_loss` = NaN.
- `tpa/run/trainer.py`, inside the existing `if paired:` block (lines 161-172; entered only when `ctrl.paired_episodes` is true, i.e. only for B_SONG): on an even episode, if the Simulator's expected slot lags `slot0`, it is moved to `slot0` before the pair snapshot is taken. This keeps the reviewed design: even episodes at `ep * T_ep`, the odd window skipped, 12 episodes = one 3P segment at constant load. This hunk is the one part of the diff outside the baseline modules; it was accepted on review of the fix report.
- **Review condition: the three NaN fields are absent by design.** DDPG (B_HUANG) has no entropy temperature; evolution strategies (B_SONG) have no critic, actor loss or temperature. `alpha` for B_HUANG, and `alpha`, `critic_loss`, `actor_loss` for B_SONG, are written as NaN only so that the episode row has the same columns for every arm. They are diagnostics of the train log; no analysis reads them as values for these baselines, and nothing computed from them enters a decision, an update or a metric.
- New test `tests/test_baselines_trainer_e2e.py` (outside the hash): B_ZHOU, B_QIN, B_HUANG, B_SONG, 6 episodes each through `trainer.run` on the stage-1 path (phase tune, validation, checkpoints, end-of-run val_full) with 12 episodes per segment; checks episode windows (B_SONG: 0, 0, 2T, 2T, 4T, 4T; the others consecutive), the logged updates and the NaN fields. On the Amendment 9 code the same test fails for exactly B_HUANG and B_SONG.

### A11.4 Proofs (laptop; proof records)
- **Real settings (stage-1 config c00, seed 9004), Amendment 11 code:** B_HUANG 4 episodes complete (updates in episodes 3-4: 176, 225); B_SONG 14 episodes complete (first ES update logged at episode 13; the load changes after 12 episodes, i.e. one segment per ES iteration).
- **Non-baseline arm untouched:** A0 c00, seed 9004, 20 episodes, Amendment 9 vs Amendment 11 code, `p_compare.py` (criterion of A9.3): **bit-identical** (rolling, best and final checkpoints; train 20 x 52; val_full 1629 x 51; RESULT.json).
- **Baselines that did not fail (review condition, run although only B_SONG is paired):** B_ZHOU and B_QIN, 14 episodes, Amendment 9 vs Amendment 11 code: **bit-identical** (train 14 x 52; val_full 1629 x 50; identical checkpoint sizes).
- **Per-arm path check:** for the 18 arms that did not fail, the Amendment 9 and Amendment 11 codes give identical episode starts, update counts and update keys (one path record per code).
- **Suites on the Amendment 11 code:** full suite **573 passed, 198 skipped, 0 failed** (771 tests incl. the 4 new end-to-end tests, tests/test_replay_index_and_checkpoint_formats.py, the heartbeat tests, test_checkpoint_resume_bitwise, the equivalence suite (A8.4)); `tests/test_d8_equivalence.py` run alone: **36 passed** (`p3_suite.log`; equivalence-suite log).
- **Server staging (existing venv):** tar sha256 12b7b5bad43c..., 144/144 file hashes, `code_hash()` 1f4eb52fa26b; test run on the server **530 passed, 205 skipped, 0 failed**; acceptance tests **36 passed**, accepted for code 1f4eb52fa26b at 2026-11-05T01:29:26Z.

### A11.5 Hashes
Code hash **1f4eb52fa26b** (1f4eb52fa26bc0a0a1db069afb197c355ea59da0789b4c76f9294d0d751268d7; was 02e2abd853f8). Config hash unchanged.

### A11.6 Deployment and rows that span both codes
As Amendment 9 (A9.5): staged code tree, acceptance tests, queue stopped, code swap (old code tree, hash 02e2abd853f8, kept), restamp script (only `code_hash`, only pending rows with the old hash; `--allow-code-change` never used; report `runs/tune/AMENDMENT11_RESTAMP.json`, markers `AMENDMENT11_RESUME.json`). Rows done under 02e2abd853f8 stay as they are and are listed "done before switch": the A0 and baseline bit-identity proofs show that the two codes give identical results for every arm except B_HUANG and B_SONG, which have no result under the old code. The 12 failed rows are returned to pending with `scripts/run_queue.py --requeue-failed` after their pre-requeue state (attempts, error) is saved to `runs/tune/AMENDMENT11_REQUEUE.json`; they restart from episode 0 under the new hash.

### A11.7 Schedule
Stage 2 may start for the 15 arms that completed stage 1, under the stage-1 readout checklist for those arms (A10.5, A10.6 rev. 1-2); the stage-1 report marks B_HUANG and B_SONG "pending Amendment 11" (stage-1 report script, option `--pending-arms`). The two baselines run their stage 1 and stage 2 afterwards. Only the main campaign waits for everything. A failed row is now reported at once (an alert on failed rows in the mirrored server records every 30 min, and a 2-min watch).

## Amendment 12 (2026-11-07, decision: best/final checkpoints hold every controller's learned state)

### A12.1 Defect
`tpa/run/trainer.py` `_net_state` (lines 244-254 under 1f4eb52fa26b) saved a fixed key list ("sac", "params", "norm") into `ckpt_best.pt` and `ckpt_final.pt`. B_HUANG keeps its learned DDPG under "agent" and B_SONG its ES parameters under "theta"; their checkpoints held only the normaliser. Consequences: (i) reloading their best checkpoint evaluated an untrained policy (test (c) could not be applied to them); (ii) the trainer's end-of-run evaluation loads the best checkpoint into the final controller, so for these two arms it scored the FINAL learned state with the best episode's normaliser, not the best checkpoint as this plan states. Validation scores during training (`best_val`, `val_curve`) were computed on the live controller and are unaffected. A12 is NOT affected (correction of the implementation report of 2026-11-07): it learns nothing by design (its table starts empty at every segment and evaluation run, `tpa/baselines/lookup.py` docstring) and its evaluation clone takes only the normaliser.

### A12.2 Fix (a new code tree; `config/` unchanged)
Every controller declares `learned_state_keys()`: the `get_state()` keys its `eval_clone` takes from the trained controller. TPAController: ("sac", "params", "norm") with memory, ("sac", "norm") without (`tpa/control.py`); BaseController default ("norm",) (`tpa/baselines/executor.py`); MemlessSAC/FrameStack and LSTM ("sac", "norm"); HuangController ("agent", "norm"); SongController ("theta", "norm"); LookupController ("norm",) with the reason. `_net_state` saves exactly the declared keys and raises if one is missing (`tpa/run/trainer.py`). New test file `tests/test_checkpoint_learned_state.py`.

### A12.3 Proofs (laptop; proof records)
- (a) all 20 arm keys (19 + A15): declared keys and `ckpt_best.pt` keys after a tiny trainer run are equal (proof record of the Amendment 12 code); against the Amendment 11 code only B_HUANG (+agent) and B_SONG (+theta) change (proof record of the Amendment 11 code).
- (b) A0, A10, A11, configuration c00, seed 9004, 20 episodes, Amendment 11 vs Amendment 12 code: bit-identical in every compared item (rolling, best and final checkpoints, train and val_full tables, RESULT) (one comparison record per arm).
- (c) real settings, seed 9004, validation every 2/3/13 episodes, no end-of-run evaluation: reloading `ckpt_best.pt` reproduces the logged best validation score exactly for A12 (0.40343426640545904, episode 2), B_HUANG (0.4455605664172923, episode 6, 1,701 DDPG updates) and B_SONG (0.42826827247985066, episode 26, after two ES iterations). On the Amendment 11 code the same runs give the identical logged best scores, but the reload gives the untrained policy's score (B_HUANG 0.42433701040009464, B_SONG 0.41002143129494983) (`runs/proofc_*`, `logs/proofc_*`).
- (d) full suite on the Amendment 12 code: 597 passed, 199 skipped, 0 failed; equivalence suite (A8.4) 36 passed (`p_d_suite.log`; equivalence-suite log). `tests/test_checkpoint_learned_state.py` on the Amendment 11 code: 5 failed (the declaration test, and the round-trip and trainer-reload tests for B_HUANG and B_SONG), 20 passed; A12 passes on both codes as stated in A12.1.

### A12.4 Hashes
Code hash 2892281e5ba9 (was 1f4eb52fa26b); config hash unchanged.

### A12.5 Deployment and the three arms (decision sections 1-2)
As Amendments 9 and 11: staged code tree, acceptance tests, code swap (old code tree, hash 1f4eb52fa26b, kept). The 18 stage-1 rows of A12, B_HUANG and B_SONG (seed 9004, 300 episodes) are re-run under the new code by a re-run script (old run directories moved unchanged to `runs/tune/_pre_amendment12/`, rows reset to pending under the new hash, record `runs/tune/AMENDMENT12_RERUN.json`); for A12 the re-run is a determinism check (its scores must reproduce). Then `tuning.py score1`, test (c) with the PASS rule of the decision of 2026-11-07 on condition 2, and the stage-2 rule of decision section 2 (unchanged top 2: the existing stage-2 runs stand; changed top 2: new stage-2 runs and R3 before any main-campaign row). All other stage-1 and stage-2 rows stay as they are: proof (b) and A12.1 show that their results are identical under both codes.
## Amendment 13 (decision of 2026-11-09, approved; statistical power and controls; declared before any main-campaign or robustness result was read by anyone)

Motivation: the pre-emptive adversarial review and the theory-and-contribution synthesis, section 3: Reviewer 3 of the first submission named A5 and A7, so n = 8 for them is indefensible; A15 is the only valid content null; the campaign runs with T_mem = 0 and the measured decision-path cost must be reported with a sensitivity run. Nothing in the running campaign changes; the rows below are ADDED (68 runs).

### A13.1 A5 and A7 move to tier A (n = 30)
Test seeds 1..30 with the same common random numbers as the other tier-A arms; the 22 extra runs per arm (seeds 9..30) are enqueued at E_main = 500 with each arm's selected configuration (`runs/tune/selected.json`) under the current code hash 2892281e5ba9, run ids `main-A5-s<seed>-sinusoidal`, `main-A7-s<seed>-sinusoidal`. Their S1 contrasts (A0 vs A5, A0 vs A7) then use n = 30 pairs; the S1 family and its Holm correction are otherwise unchanged (see A14.4 for the arms Amendment 14 adds to S1). `scripts/enqueue_main.py` (260 rows) is not changed; the additional rows are written by `scripts/enqueue_main_extra.py`, which refuses unless it adds exactly 68 rows.

### A13.2 A15 (key-shuffled control, specification §6, A5.2) enters the campaign at tier C (n = 6)
Seeds 1..6, all three loads as the plan defines for tier C (the trainer evaluates every load of the run; the exclusion of A15 from tuning and from the campaign in A6.2 (a) is lifted for the campaign only). A15 is A0 with the long-term write keys taken from a random earlier slot (`tpa/arms.py`, `key_shuffled`); it has A0's tunables and was never tuned (A6.2 (a)), so it runs with **A0's selected configuration** (implementation choice, disclosed here: the control differs from A0 only in the shuffled keys, and tuning it separately would confound the control with a tuning difference). Analysis: mechanism-level disclosure, A0 vs A15 paired by seed at every load, reported with mean difference, 95 % CI, d and d_z; NOT a member of any Holm family.

### A13.3 T_mem sensitivity
The campaign runs with T_mem = 0: `config/default.toml` `[t_mem_ms] A0 = 0.0`, read by `tpa/env/queueing.py` `t_mem_seconds` (Eq. 13; 0 for the memory-free arms). Justification reported in the paper: the measured decision-path latency of the exclusive timing runs (A7.4) in the timing report, MEPR 1.14 ms + retrieval 0.87 ms = 2.0 ms of memory modules, decision path 13.33 ms (A0) vs 11.82 ms (A9); the old "15-20 %" figure must not survive anywhere. Sensitivity run: A0 with the override `t_mem_ms.A0 = 25.0` (about 10x the measured value), n = 6 (test seeds 1..6), E_main = 500, A0's selected configuration, run ids `main-A0-tmem25-s<seed>-sinusoidal` (queue `arm` = A0; the config hash differs from A0's by design and is recorded in the rows and manifests). Analysis: A0 vs A0-tmem25 paired by seed at heavy load (descriptive at the other loads); the analysis script labels these runs `A0-tmem25` from their run id so they never pool with A0.

### A13.4 Execution
68 rows (44 + 18 + 6), phase main, workload sinusoidal, enqueued seed by seed at the main rows' priority, so workers claim them after the 190 pending main rows. A second launcher with 10 extra workers on vCPUs 11-20 and 42-51 (free: the other project's campaign has ended and those cores were idle on 2026-11-09; the sharing split of 2026-11-06 is amended by the decision of 2026-11-09, and the other project is to be told), NUMA-alternating, staggered 60 s, stopped only by PID (the stop script covers both launchers); the memory rule of the hourly watch is unchanged (TPA's own Pss + swap over all its workers > 75 GB, OOM kill of a TPA worker, MemAvailable < 8 GB). The 20 main-campaign workers are never stopped for this. The registered analysis (end of the main campaign) now includes these 68 rows.

## Amendment 14 (decision of 2026-11-09 with its addendum of 12:40 UTC+3, approved; theory-derived variant A16 "TPA-F" and a workload with deep-past predictability; declared before any main-campaign or robustness result was read by anyone)

Motivation (the theory analysis, section 2.7, Theorem 7; the theory-and-contribution synthesis, sections 1-2): the implemented long-term store writes, at slot s, a key and a content that are both functions of the recent window (Theorem 1), so on a stationary periodic workload it is a denoiser of the present, not a forecaster. Theorem 7.1: any controller whose decision depends on the last w slots (and own state of bounded age) is bounded by the optimal w'-Markov policy; if the workload has deep-past predictability xi_w(Delta) = I(X_{t+1:t+Delta}; X_{<= t-w} | X_{t-w+1:t}) > 0, a controller with history strictly improves on it. Theorem 7.2: a store with DELAYED CONSOLIDATION, which writes at s + Delta the content phi(X_{s+1:s+Delta}) under the key k_s computed at s, makes retrieval a Nadaraya-Watson forecast of what follows the current window; A0 is its Delta = 0 case. Amendment 14 adds that store as a new arm and a workload with xi_w > 0, and pre-registers the predictions.

### A14.1 New arm A16 ("TPA-F", delayed consolidation)
A16 = A0 with the long-term WRITE rule changed and nothing else: short-term store, significance gate (sigma_p at s), retrieval (query k_t), SAC and DRM identical to A0. Write rule, fixed exactly (addendum of 12:40 UTC+3; to be written into the specification with the new code): the gate decision and the key k_s are taken at slot s; the content is written at slot s + Delta; content = LN(W_v [ next-Delta per-zone normalised arrival profile over (s, s+Delta] ; occupancy trajectory summary over (s, s+Delta] ]). Delta in {100, 300, 900} slots (= w, 3w, P/4) is a stage-1 tunable; the other five tunables are A0's. Implementation: `tpa/memory/stores.py` write path plus a Delta-slot ring buffer of (k_s, pending content accumulators), under a NEW arm key `A16` (`tpa/arms.py`), in a new code tree built from the code of Amendment 12. Required before deployment: A0-A15 bit-identical before and after (p_compare on A0, A3, A10, 20 episodes); tests (i) A16 with Delta = 0 reproduces A0 bit for bit, (ii) the content stored at key k_s equals the hand-computed profile of (s, s+Delta], (iii) the replay/reconstruction path (`tpa/memory/recon.py`) handles the delayed write, or A16 is excluded from reconstruction with a documented reason. If A16 cannot be made bit-identical for all other arms: stop and report before deploying.

### A14.2 Mechanism check M4 (outcome-blind; validation seeds; driver + validation block)
The retrieved r_L(t) must predict the next-Delta per-zone arrivals better than the window-extrapolation baseline: MSE ratio (retrieval / window extrapolation) < 0.8 for A16, and NOT for A0 (ratio >= 1). Pass/fail is written into this plan before any completion metric of A16 is read; M4 is part of the checklist of A16's stage-1 readout.

**M4 result (written 2026-11-09 22:35 UTC, before any completion metric of A16 was read; `scripts/m4_check.py`, files `runs/tune/m4/*.json`, copies with the stage-1 readout records of A16): FAIL for every A16 configuration.** Validation block (seed 9004, medium load, the stage-1 best checkpoint of each row): A16 ratios c00 (Delta 100) 69.2, c01 (300) 122.7, c02 (900) 25.2, c03 (100) 12.2, c04 (300) 20.8, c05 (900) 15.3 (rule: < 0.8); A0 c04 (its selected configuration) at Delta 100 / 300 / 900: 34.3 / 27.6 / 10.7 (expected >= 1.0: holds). Synthetic driver (period 600, seeds 0-2, mean ratio): A16 22.1 / 17.5 / 29.6 / 3.06 / 3.35 / 11.6; A0 8.8 / 3.2 / 4.2. Diagnostic (`runs/tune/m4/diag_driver.jsonl`, driver, seed 0): no scale artefact (mean norm of the retrieved vector 5.9-7.5 vs 8.0 for the target; norm-matching leaves the ratios at 3.1-26); 26-32 long-term rows filled; the trained content space is nearly constant across slots (cosine of the window candidate to the realised future profile 0.98-0.995, cosine of the retrieved content 0.88-0.96), and for Delta 300/900 even the constant mean of the future profiles beats the window. Reading: the long-term content retrieved under the present key does not forecast the next-Delta profile better than the present window, for the delayed write rule as much as for A0's; M4 is a STOP condition of A16's stage-1 readout (A14.4), so A16 stage 2 is not started, pending a decision. No completion metric of A16 had been read when this paragraph was written; the stage-1 readout script of A16 was run afterwards.

### A14.3 Workload with deep-past predictability
Pareto on-off (`tpa/env/workload.py` kind "pareto", config defaults; Theorem 7 example (c): the hazard of leaving the current state decreases with its age, so xi_w > 0 for every w) joins the robustness family: Pareto x {A0, A3, A9, A10, A16} x n = 6 test seeds (30 runs, laptop, after the current robustness family). MMPP and Gauss-Markov add A16 x n = 6 (12 runs). Robustness contrasts: A0 vs each other arm at heavy load, Holm within each workload family (section 5), plus the A16 contrasts of A14.4.

### A14.4 Pipeline and pre-registered contrasts for A16
Same readouts as every arm: stage 1 (6 configurations x seed 9004 x 300 episodes; stage-1 readout checklist of A10.5/A10.6 plus M4), stage 2 (top 2 x seed 9005 x 1000 episodes; stage-2 readout checklist incl. A10.8 revision 2), then tier A, n = 30 test seeds at E_main = 500 in the main phase, loads light/medium/heavy as tier A, on the extra workers (the 20 main-campaign workers are never stopped). Contrasts added to the plan now: S1 family gains A16 vs A0 (the primary contrast of Amendment 14), A16 vs A3, A16 vs A9 and A16 vs A10 (the S1 Holm family therefore has 14 members instead of 10); robustness: A16 vs A0 on Pareto (prediction: A16 > A0 if Theorem 7 holds), on MMPP (prediction: A16 ~ A0: memoryless switching, xi_w ~ 0), on the sinusoid (prediction: A16 ~ A0: the window suffices, Theorem 1).

### A14.5 Pre-registered predictions (the theory analysis, section 2.9, copied verbatim on 2026-11-09 before any result was read; editorial changes in square brackets)

| ID | Quantity ([this plan] reference) | Prediction | Falsifier |
|---|---|---|---|
| P1 | A0 − A3, heavy, n = 30 ([this plan] §2) | within ±1 pp; TOST passes if SD_d ≲ 1.5 pp; no difference in cycle 1 of the 3P runs | CI excludes 0 with \|Δ\| ≥ 1 pp, present in cycle 1 |
| P2 | A0 vs A1, A4, A5, A14 (S1, n = 12) | within tier-B MDE (≈ 0.89 SD_d) | Holm p < 0.05 with \|Δ\| ≥ 3 pp |
| P3 | A2 vs A1 (S1) | A2 ≤ A1 (reverses [the first submission]) | A2 > A1 beyond MDE |
| P4 | A0 − A9 by load; S4 interaction | information mechanism ⇒ Δ_light ≥ Δ_heavy (interaction ≤ 0) | interaction > 0 with CI excluding 0 **and** P1–P2 null (then the gain is not informational) |
| P5 | A10 vs A6; A10 vs A0 (S2) | null | \|Δ\| ≥ MDE |
| P6 | LT occupancy, sinusoid, A0 | ≤ 19 always; ≈ 5–10 typical | > 19 after cycle 1 |
| P7 | LT occupancy, MMPP | ≫ 19; LRU replacement if sojourn ≲ w | ≤ 19 |
| P8 | memory hit rate ([specification] §6 definition) | ≈ 0 if top-1 is across levels; 0.75–0.90 (after cycle 1) if within M_L | — (diagnostic of the metric, not of the method) |
| P9 | gate-open fraction, sinusoid | 0.10–0.14, two intervals per cycle, load-insensitive | ≥ 3 disjoint intervals; strong load dependence |
| P10 | gate on MMPP | openings cluster after level changes; fraction lower at light load | uniform openings |
| P11 | exclusive timing (A7.4) | DRM > MEPR > retrieval ≈ actor; A0 − A9 overhead ≈ 2.5 ms, independent of load | memory modules > DRM; overhead grows with load |
| P12 | failure taxonomy per arm | admission-refusal share identical across memory arms within seed up to ordering effects | large between-arm differences in `infeasible` share |
| P13 | A6 vs A0 heavy admissions | A6 ≥ A0 or ≈ | A0 > A6 beyond MDE |
| P14 | A8 vs A0 per-slot admissions in burst-on slots | A8 ≤ A0 | — |

Notes on the table, dated 2026-11-09: P2 says "n = 12" as the theory analysis wrote it; tier B is n = 8 since Amendment 7, and A5 and A7 are n = 30 since A13.1. P9 is REFINED by the addendum of 12:40 UTC+3, before any main or robustness result was read: sigma_p(theta) ~ 0.279 sqrt(1 - kappa_phi cos(2 theta + phi0)) was derived with the per-seed phase imbalance kappa_phi = |sum_z e^{2 i phi_z}| / Z; refined P9 (replaces "load-insensitive"): the gate-open fraction correlates with kappa_phi across seeds and weakly DECREASES with load (for kappa_phi < 0.57 the gate opens only through Poisson noise). Original wording kept above for the record. T_mem numbers everywhere from the exclusive timing runs (A7.4) in the timing report (MEPR 1.14 + retrieval 0.87 = 2.0 ms; decision path 13.33 vs 11.82 ms).

### A14.6 Order of work and reporting
Amendment 13 enqueue first (no code change); then A16 in a new code tree with the proofs of A14.1, staged deployment as Amendments 9/11/12 (staging tree, acceptance tests, swap, re-stamp of pending rows); A16 stage 1 on the extra workers. Reports: after the Amendment 13 enqueue, after the A16 proofs, at A16's stage-1 readout, stage-2 readout and start. No result of any main or robustness row is read until the registered analysis; the A16 readouts read validation data only.

### A14.7 Implementation, proofs and deployment of A16 (2026-11-09; implementation record)
Code tree built from the code of Amendment 12; code hash 5253c84764d7 (was 2892281e5ba9); `config/default.toml` unchanged (config hash of every other arm's rows unchanged: Delta lives in the override-only key `memory.delay_slots`, `tpa/config.py` OVERRIDE_ONLY_KEYS). Code: `tpa/arms.py` (flag `delayed`, arm A16), `tpa/memory/__init__.py` (MemoryFlags.delayed), `tpa/memory/stores.py` (TwoLevelMemory: pending list; `consolidate` pushes (s, k_s, label) and flushes due writes; `_flush_due`; `delayed_content(s)` = `candidate_zone(compute_v(za[s+1..s+Delta], zone_mean[s+Delta], mean uav_occ[s+1..s+Delta]))`; `_consolidate_now` = the former consolidate with the event field `delay_src`; Delta = 0 writes z_s at s), `tpa/memory/recon.py` (SegmentStore logs `delay_src`; `_gather`/`reconstruct` recompute the delayed content from the logged (s, Delta) window, so A16 stays on the specification §5 training path). The pending list and `delay_src` enter checkpoints and store states only for A16, so every other arm's checkpoint bytes are unchanged. Specification: A16 row with the exact write rule. Tuning: `scripts/tuning.py` space_of adds `memory.delay_slots` in {100, 300, 900} for `delayed` arms and assigns the three values cyclically, so the first 6 configurations hold each Delta twice; `sample --arms A16` adds A16 to configs.json and refuses if any other arm's entry would change (verified unchanged); `enqueue1 --arms`.
Proofs (laptop; proof records of Amendment 14): `tests/test_delayed_consolidation_a16.py` (7 tests): (i) A16 with Delta = 0 reproduces A0 bit for bit through `tpa.run.trainer.run` (RESULT, train and val_full tables, final checkpoint); (ii) every write is Delta slots after its key slot, its `cand_key` is k_s, its content equals the hand-computed LN(W_v [profile ; occupancy]) of (s, s+Delta] and the fresh-slot gated row; pending entries survive checkpoint/resume (identical r(t) afterwards) and are dropped at a segment reset; (iii) reconstruction equals the rollout r(t) (atol 1e-5) for Delta = 5 and Delta = 0 and the gradient reaches W_v through the delayed write (A16 is NOT excluded from reconstruction); refusals (candidate keys, novelty repr, flat, Delta < 0). Bit-identity of the other arms: A0, A3, A10, configuration c00, seed 9004, 20 episodes, code of Amendment 12 vs the new code: ALL_EQUAL (rolling, best and final checkpoints, train and val_full tables, RESULT) (one comparison record per arm); a first build of the new code had added the new state keys to every memory arm's rolling checkpoint (`pending`, `ev_src`), found by this proof and corrected before any deployment. Staging copy on the compute server: 155 file hashes verified; acceptance tests passed (38 passed; accepted 2026-11-09T10:20:23Z for code 5253c84764d7). Test suites: see the code-review record, entry "Amendment 14 implemented", for the counts.
Deployment (deviation from "as Amendments 9/11/12", flagged): the live tree is NOT swapped. The 20 main-campaign workers are never stopped (decision rule), and `scripts/run_queue.py` claims a pending row before checking its code hash (lines 236-247: a mismatch marks the row failed), so a swapped live tree or mixed hashes under the launchers running the code of Amendment 12 would fail rows. Instead the new code runs as a SECOND code tree whose `runs/` is a symlink to the live run directory (one queue, one run-directory tree, shared `configs.json`), with workers that claim only A16 rows (a separate launcher with `--arms A16`, stopped by PID); the launchers running the code of Amendment 12 claim only `--phase main` rows, so A16's stage-1 and stage-2 (phase tune) rows are safe from them; A16's main rows are enqueued only when no main row on the code of Amendment 12 is pending or running. Every A16 run records 5253c84764d7 in its manifest. Memory: 36 workers would reach the 75 GB rule, so the extra launcher is reduced to 4 workers (vCPUs 11, 42, 12, 43) and A16 stage 1 takes 6 workers on 13-15 / 44-46: 30 workers in total, as now. M4 (A14.2) is computed by `scripts/m4_check.py` (definitions in its docstring: y_fut, y_hat_L = key-attention-weighted mean of the stored contents, z_win = A0's candidate; MSE ratio over the validation block of seed 9004 at medium load; also the synthetic driver) and read by the stage-1 readout script of A16 with the A10.5/A10.6 checklist; M3 does not apply to A16.

Deployed 2026-11-09 11:33 UTC exactly as described: second code tree, shared queue, A16 added to the shared configs.json with the 19 other arms' entries unchanged, 6 stage-1 rows `tune1-A16-c00..c05-s9004` (Delta 100, 300, 900, 100, 300, 900) running on vCPUs 13-15 / 44-46 from the A16 launcher; extra launcher at 4 workers (11, 12, 42, 43); the 20 main workers untouched. Details and test counts: the code-review record, entry "Amendment 14 deployed as a second tree".

## Amendment 15 (decision of 2026-11-10, 03:30 UTC+3: A16' = A16 with a frozen content map, M4 redefined with a held-out decoder, tested on the Pareto on-off workload; one bounded retry, then stop; declared before any A16' run and before any completion metric of A16 was read)

### A15.0 Reading of the diagnosis table of A16's M4 failure (baseline, coverage and decoder checks; verbatim; editorial changes in square brackets)
1. Coverage is not the cause (covered ratios still >> 0.8). Accepted.
2. The decisive numbers are [those of the decoder check]: the SAME ridge decoder recovers the next-Delta profile from the current
   WINDOW content with held-out R^2 0.94-0.97, but from A16's RETRIEVED content with 0.03-0.65. Two
   structural reasons, both consistent with the theory:
   (a) On the sinusoid the window already determines the future (Theorem 1); M4-as-posed therefore asks
       the store to beat a baseline that is near-optimal on this workload. The store cannot win here by
       construction; the right workload for this test is one with xi_w > 0 (Pareto on-off).
   (b) A16's content is LN(W_v [.]) with W_v TRAINED by the SAC gradient. On the sinusoid the gradient has
       no incentive to preserve forecast information (the window suffices), so the learned projection
       discards it: untrained W_v gave ratio 0.67, trained gives 12-183. A Nadaraya-Watson forecaster
       must store the target in a FIXED representation.
3. Conclusion: the write rule was tested in the one setting where theory says it cannot help, with a
   content map that training is free to destroy. Stopping A16 now would record a negative result that
   the theory itself predicts; it would not test Theorem 7.

### A15.1-A15.5 Decision text (verbatim; editorial changes in square brackets)
A15.1 A16' = A16 with W_v for the long-term content FROZEN at its seeded initialisation (no gradient
      through the long-term content path; retrieval value projection W_V and everything else train as in
      A0). Gate unchanged ([the coverage check] showed coverage is not binding). Delta in {100, 300, 900} as before.
      Proofs: Delta = 0 with W_v trainable reproduces A0; other arms untouched (A0/A3/A10 bit-identical);
      unit test that W_v receives no gradient for A16'.
A15.2 M4 redefined (written before any run, replacing the raw-distance statistic): ridge decoder from the
      retrieved long-term vector to the raw next-Delta per-zone profile, fitted on the first half of the
      validation block, scored on the second; same decoder from the window candidate. PASS iff
      R^2(retrieved) >= R^2(window) + 0.10 on the PARETO ON-OFF validation block (where the window is not
      sufficient). On the sinusoid the statistic is recorded as a disclosure only (prediction: no gain).
A15.3 Runs: A16' stage 1 = 6 configurations x seed 9004 x 300 episodes on the PARETO workload (config
      override workload.kind = "pareto", defaults), on the 6 idle A16 vCPUs (~7-8 h). M4 (A15.2) on the
      Pareto validation block for A16' and for A0's selected configuration.
A15.4 Decision rule, fixed now:
      - M4 PASS on Pareto: A16' proceeds ONLY within the Pareto family: stage 2 (top 2, seed 9005, 1000 ep,
        Pareto), then Pareto x {A0, A3, A9, A10, A16'} x n = 12 (laptop, test seeds 1..12). Pre-registered
        contrast: A16' vs A0 on Pareto heavy-load completion, paired, SESOI 1 pp. No A16' rows in the
        sinusoidal main campaign (theory predicts equivalence there; not worth 30 runs).
      - M4 FAIL on Pareto: A16 and A16' are STOPPED. The paper reports the delayed-consolidation variant as
        a theory-motivated mechanism check that failed, with the [M4-failure] diagnosis, and Theorem 7 is kept as
        the scope statement (what a forecasting store would need), not as a demonstrated result.
      Either way, no further A16 variants after this.
A15.5 Records: Amendment 15 in [this plan] (motivation = [the M4-failure diagnosis]), the M4 redefinition with its reason,
      [the code-review record], […]. The sinusoidal A16 stage-1 rows stay in the record as the disclosed failed check.
Main campaign, A13 rows and the laptop robustness family: unchanged.

### A15.6 M4 as redefined (operational definition, written 2026-11-10 before any A16' run; `scripts/m4_decoder.py` in the A16' tree, outside the code hash)
For a frozen stage-1 best checkpoint (`ckpt_best.pt`) evaluated on the validation block (seed 9004, block "val", test run 0, medium load) of the workload named: at every slot t with a defined arrival key, at least one filled long-term row and t + Delta recorded, y_hat_L(t) = softmax_j(cos(k_t, L_key_j) / tau) M_L[j] over the filled rows (the long-term read, as A14.2), z_win(t) = A0's candidate LN(W_v compute_v(window ending at t)) (the window baseline), and the target v_fut(t) = compute_v(za[t+1..t+Delta], zone_mean[t+Delta], mean uav_occ[t+1..t+Delta]): the RAW next-Delta profile (per-zone mean arrivals / zone mean, then the UAV occupancy; Z + M = 31 dimensions). Ridge decoder with intercept, lambda = 0.01 x mean diagonal of X'X (intercept unpenalised), from X = y_hat_L to v_fut, fitted on the first half of the valid slots in time order and scored on the second half; the same decoder from X = z_win. R^2 = 1 - SSE / SST on the held-out half (SST around the held-out mean). gain = R^2(retrieved) - R^2(window). **PASS iff gain >= 0.10 on the Pareto on-off validation block** (A15.2). Delta of a run = its `memory.delay_slots`. Aggregation over the six A16' stage-1 configurations (the decision fixes the criterion, not the aggregation; implementation-level pre-registration, flagged for a decision): the verdict is PASS iff the criterion holds for at least four of the six configurations (a majority; the configurations differ in SAC hyper-parameters, which can fail for reasons unrelated to the store); all six values are disclosed, as are the values of A0's selected configuration c04 (its Pareto-trained run `tune1-A0-c04-s9004-pareto`, and its sinusoidal checkpoint `tune1-A0-c04-s9004` evaluated on the Pareto block) and, on the sinusoid as a disclosure (prediction: no gain), the decoder-check values already computed for A16 c00-c05 and A0 c04 (R^2 retrieved 0.62 / 0.48 / 0.03 / 0.65 / 0.49 / -3.32 vs window 0.95 / 0.97 / 0.95 / 0.94 / 0.97 / 0.97; A0 0.64 / 0.69 / 0.65 vs 0.95 / 0.97 / 0.97: no gain, as predicted). No completion metric, score or learning curve of any A16' run is read before the M4 verdict is written here.

### A15.7 Implementation and deployment record (updated as the work proceeds)
Code tree built from the code of Amendment 14. Arm key `A16P` (printed A16'): `tpa/arms.py` (flag `frozen_content`; `"A16P": replace(_A0, delayed=True, frozen_content=True)`), `tpa/memory/__init__.py` (MemoryFlags.frozen_content; `_ARM_FLAGS["A16P"]`), `tpa/memory/stores.py` (`TwoLevelMemory.__init__`: under `frozen_content`, requires delayed and Delta >= 1, then `requires_grad = False` on `params.W_v.weight` and `.bias`; W_v feeds only the long-term content path, so W_S, W_g, W_Q, W_K, W_V, W_omega and the SAC networks train as in A0 / A16), `tpa/control.py` (the learner's parameter copy gets the same flags when it is a separate object, i.e. on a GPU). No new state: every other arm's code path and checkpoint bytes are unchanged (proof: A0 / A3 / A10 20-episode runs, code of Amendment 14 vs the new code). `scripts/tuning.py`: `_rid(..., workload)` adds the suffix `-pareto` to tuning run ids on another workload; `enqueue1 --arms A16P --workload pareto` (queue column `workload = "pareto"`; the trainer trains on it and runs its end-of-run val_full evaluation on it (`tpa/run/trainer.py:105`, `:226-231`) [corrected 2026-11-10, A15.8 (1): checkpoint selection, `ckpt_best.pt` and `best_val`, runs on the SINUSOIDAL medium-load validation block for every row, `tpa/run/evaluator.py:154-161`; the earlier wording "trains and validates on it" was wrong]), `enqueue1 --arms A0 --workload pareto --config 4` (A0's selected configuration on Pareto, the M4 comparison run, enqueued after the six A16' rows so that it is claimed last), `score1 --workload pareto` (stage 2 on PASS only, on "approved"). The decision's "config override workload.kind = pareto" is realised through the queue's workload column, which is how every workload is selected in this code base; Pareto parameters are the config defaults (`pareto_shape 1.5`, `pareto_mean_s 300`, `pareto_on_mult 2.0`). Tests `tests/test_frozen_content_map_a16p.py` (a-d), `tests/test_arms.py`, `tests/test_checkpoint_learned_state.py`, `tests/test_tuning_arms.py` adjusted for the new arm. Specification: A16P row. Deployment as a THIRD code tree (runs symlink to the live runs; workers `--arms A16P`; own start and stop scripts) on the six idle A16 vCPUs 13-15 / 44-46; the 20 + 4 main-campaign workers, the Amendment 13 rows and the laptop family are untouched. Proofs (laptop; proof records of Amendment 15): `tests/test_frozen_content_map_a16p.py` 4 passed with `test_delayed_consolidation_a16.py`, `test_arms.py`, `test_checkpoint_learned_state.py`, `test_tuning_arms.py` (49 passed, 1 skipped; the one data-file failure, configs.json without A16P, gone after `tuning.py sample --arms A16P`: 7 passed, 1 skipped); bit-identity, code of Amendment 14 vs the new code, for A0, A3, A10 (configuration c00, seed 9004, 20 episodes): ALL_EQUAL on rolling, best and final checkpoints, train and val_full tables and RESULT (one comparison record per arm; the proof runs' checkpoints were deleted afterwards to protect the laptop disk, which is at the 40 GB floor of the robustness workers' disk guard); full suite 571 passed, 199 skipped, 3 failed, all three `tests/test_infra_run_queue.py` dry runs refused by the disk guard (`free disk 36.6 GB < disk_min_free_gb 40.0 GB` on the laptop's temp drive; environmental, the same three pass whenever 40 GB are free); the DRM speed-up equivalence suite (Amendment 8) 40 passed. Staging copy on the compute server (159 files, every sha256 verified, code 790391a95b5d): test suite 567 passed, 206 skipped, 0 failed; acceptance tests 40 passed (accepted 2026-11-10T01:13:21Z for code 790391a95b5d). Deployed 2026-11-10 01:14 UTC as the third code tree (runs/ symlink; staging and acceptance records kept with the run records). `tuning.py sample --arms A16P` added A16P to the shared configs.json (12 configurations; Delta 100, 300, 900, 100, 300, 900 for c00-c05; the 20 other arms' entries unchanged); `enqueue1 --arms A16P --workload pareto`: 6 rows `tune1-A16P-c00..c05-s9004-pareto`; `enqueue1 --arms A0 --workload pareto --config 4`: `tune1-A0-c04-s9004-pareto` (created last, so claimed last). Start script dry run, then start 2026-11-10 01:15:05 UTC: 6 workers on vCPUs 13, 44, 14, 45, 15, 46 (staggered 60 s); the first manifests record 790391a95b5d and workload pareto. The 20 + 4 workers running the code of Amendment 12, the Amendment 13 rows and the laptop family were not touched. Expected finish of the six rows: about 8 h; the A0 Pareto row follows on the first freed worker (about 8 h more). Disclosure runs of the redefined M4 for the sinusoidal A0 c04 checkpoint on the Pareto block started at the same time (`runs/tune/m4/dec_tune1-A0-c04-s9004_on_pareto_d*.json`).

### A15.8 Pre-verdict review of M4 and of the deployment (2026-11-10, 06:20-12:00 UTC; written BEFORE any A16' M4 value was computed or read)
Full report: the review written before the verdict (tables, file:line references). A read-only adversarial review in four areas (every finding re-checked by an independent second reviewer who tried to refute it) and two outcome-blind feasibility checks (two feasibility scripts in the records of Amendment 15: untrained A16P controller, NO checkpoint, seed-9004 Pareto validation traces). No score, curve or completion metric of any A16' or A0-Pareto run has been read.
(1) Checkpoint selection runs on the sinusoid for every row: `tpa/run/evaluator.py:154-161` calls `evaluate(..., block="val")` with the default `workloads=("sinusoidal",)` (`:124-125`); `tpa/run/trainer.py:204-209` saves `ckpt_best.pt` / `best_val` on it; only the end-of-run evaluation (`trainer.py:226-231`) uses the row's workload. Affects the A16' and A0 Pareto rows (A15.7 corrected) and, since the start, the laptop robustness family (MMPP and Gauss-Markov rows select on the plain sinusoid without the mobility option; all arms alike; disclosure in the registered analysis). A fix needs an amendment (pass the row's workload and env options into `validate()`; sinusoidal rows stay bit-identical). The main campaign is unaffected.
(2) The A15.2 / A15.6 statistic cannot discriminate on the Pareto block. On the M4 block a NO-INFORMATION input (the decoder then returns the fit-half mean) passes the decided rule at Delta 300 (gain +0.395) and 900 (+0.536), i.e. for c01, c02, c04, c05 = 4 of 6. Through the decoder, the ideal causal Nadaraya-Watson store (arrival keys, unlimited capacity, no gate: the Theorem-7 forecaster) has skill against the no-information predictor of -0.06 / -0.25 / -0.07 at Delta 100 / 300 / 900 on the M4 block and at most +0.03 on the three val_full traces, while the non-causal oracle passes everywhere (held-out R^2 0.71-0.95). Decoder-free (arrivals part, Z = 6), the ideal causal store beats window persistence everywhere (+0.09 to +0.62) but not the causal running mean of past profiles: -0.12 / -0.11 / -0.13 on the M4 block, -0.13 to +0.07 across the four traces. Cause (found in the review): heavy-tailed on/off sojourns and the cumulative zone-mean normaliser shift the level between the two halves of one trace; the gain then ranks inputs by low prediction variance, not by skill.
(3) Other confirmed points: ridge lambda from the uncentred Gram (A15.6 wording) makes R^2 depend on a constant input offset (untrained c02: gain +0.331 vs +0.007 with the centred Gram); the 0.10 threshold vs the achievable headroom; the 4-of-6 aggregation (implementation-level) treats dependent configurations as independent tests; A14.2's A0 negative control is not part of A15.2; W_g still trains on the content path, as A15.1 says ("everything else ... as in A0").
(4) Corrections made now, outside the code hash, nothing of A16' read: `scripts/m4_decoder.py` z_win uses the MEPR window of w + 1 slots = A0's actual candidate, as declared (was w slots; found in the review; +0.002 to +0.004 on untrained controllers); refusal of a --delta other than the run's delay and of unfinished rows; `--ckpt best|final`; disclosed controls in every result file that never change `passes` (no-information R^2, R^2_OS, gain vs no-information, skill-floor rule, centred lambda, lambda grid, arrivals-only R^2, purged split, time-shift null, filled-row counts). Deploy scripts: the stop scripts' final `run_queue.py --requeue-stale --stale-min 0` requeued every running row of the shared queue (tpa/run/queue.py `requeue_stale` ignores phase and arms) and is replaced by a script that requeues the rows of dead workers (identity check by pid and worker id; scratch-copy test on the server PASS); the main start script no longer refuses because of other trees' tune rows; exact launcher-name matching; unknown arguments refused; start scripts return only after their launcher spawned w00 (no shared worker-id stamp); the stop script checks its change of directory; the extra launcher's pin check covers all trees. The A0 Pareto row (no launcher claimed `--arms A0`) runs since 06:09 UTC on vCPU 13 (A16' start script restricted to arm A0).
(5) Options for the decision: (A) keep A15.2 / A15.6 and report the verdict with the controls and an "uninformative PASS" flag; (B) replace the criterion, e.g. by the skill-floor rule, which no causal store can pass on these traces, i.e. A15.4's FAIL branch; (C) rule M4 undecidable on the Pareto workload with this test and record A16 and A16' as stopped under A15.4, reason: the diagnosis table of A16's M4 failure and the two outcome-blind feasibility tables, Theorem 7 kept as the scope statement; (D) a decoder-free test, a new definition, useful only if the ideal store showed skill, which the decoder-free table does not. Recommendation: (C); the declared statistic and its controls can still be computed after the decision as a disclosure. Until the decision, `scripts/m4_decoder.py` is not run on any A16P checkpoint.

### A15.8 decision (2026-11-10 16:30 UTC+3): option C, A16 and A16' stopped
Decision: C. Stop A16 and A16'. Do not compute M4 for A16', do not redefine M4 again, do not enqueue the Pareto family with A16'.
Reason. Amendment 15 allowed ONE bounded retry after A16 failed M4. It was found, before any A16' M4 value or
completion metric was read, that the decided M4 rule cannot separate a forecasting store from an empty one on the
Pareto block, and that checkpoint selection runs on the sinusoid for every row. A second redefinition would be a
second retry, which the amendment excludes. The question A16' was meant to illustrate (a long-term store can add
information only under beyond-window predictability) is answered in the paper by the analysis: the predictive gap
G_w is about 0.2-0.5 % of the intensity variance on the sinusoid, below 1e-10 under MMPP, and 4-7 % on Pareto on-off
(10-34 % of the predictable part).
Applied on 2026-11-10 (server clock, about UTC):
- 12:54 UTC: the two running A16P rows were stopped by their own PIDs: the A16' stop script sent SIGINT to the verified A16P launcher, which terminated its workers (row `tune1-A16P-c03-s9004-pareto`, episode 300; row `tune1-A16P-c05-s9004-pareto`, episode 300) and released their rows. Records and checkpoints are kept as they are. The queue allows only pending / running / done / failed (`tpa/run/queue.py` CHECK constraint), so both rows stay `pending` with the error note "stopped by A15.8 [decision] ([…] 2026-11-10, option C): not to be resumed; records and checkpoints kept as is", not failed; the A16' and A16 start scripts refuse A16 / A16P from now on; the analysis guards report such rows as stopped (missing), not as open. The other four A16P rows (c00, c01, c02, c04) and `tune1-A0-c04-s9004-pareto` had finished before the decision and stay `done`. No M4 value of A16' was computed and no A16 or A16' result was read.
- 12:56 UTC: the freed vCPUs went to the main campaign: the extra launcher was stopped by its verified PID (its 4 rows released) and restarted with its cap of 10 workers on vCPUs 11-15 / 42-46 (dry run first; NUMA-alternating; memory rule 75 GB unchanged). Verified 13:07 UTC: 30 TPA workers (20 + 10), 30 main rows running, 0 failed, TPA_MEM 59.8 GB.
- Projection (queue wall times only, projection script: per-arm mean wall time of main rows finished since 2026-11-09 12:00, list-scheduled in claim order on 30 workers): last main row about 2026-11-13 02:45 UTC, about 13 h earlier than with 24 workers (2026-11-13 15:54 UTC).
- The Pareto robustness family with A0, A3, A9, A10 (A14.3) is unchanged (24 runs, laptop, after the current family). Note A15.8 (1): its checkpoints, like every row's, would be selected on the sinusoidal validation block unless an amendment changes `validate()`.
- Server watch reduced from hourly to every 3 hours (decision of 2026-11-10).

## Amendment 16, Part A (2026-11-10; decisions of that date on Amendment 16, Part A, and on P10 and the paper facts; analysis code and reporting only, no change to tpa/ or config/; written before any main-campaign or robustness result was read; Part B waits for the corresponding author)

### A16A.0 Decision text, Part A (verbatim; editorial changes in square brackets)
[Item 1.] P10 statistics as specified in [the decision on P10 and the paper facts] (open fraction by load; clustering ratio within
    w = 100 slots after a level change, mean over seeds with 95 % t-interval), if the records allow; otherwise report
    that P10 cannot be computed.
[Item 2.] Per-cycle reporting on the sinusoid for every memory variant: long-term occupancy at the end of each cycle,
    gate-open share and open phase bins (8 bins of P/8) per cycle, and the A0 - A3 completion difference per cycle
    (paired, 95 % interval). P6 and P9 are judged on cycles 2-3; whole-run values are reported beside them.
    The primary test (A0 - A3, heavy, whole run, TOST +/-1 pp) is unchanged.
[Item 3.] Exploratory, labelled as not registered: descriptive paired differences A8 - B_x and A9 - B_x for the four
    baselines (heavy load, n = 8), to separate burst mode and visiting order from learning.
[Item 4.] Disclosure recorded: the pilot (seeds 8001-8003, earlier build, validation traces) informed the power analysis
    and the equivalence margin.
Implement [items 1-3] in scripts/analyze_campaign.py with unit tests on synthetic inputs; record in [this plan] as
"Amendment 16, Part A" with today's date and in [the code-review record]. Report the test counts.

### A16A.1 Records check for P10 (code and manifests only, no metric read)
The test evaluation writes, per (load, test run), one run row and 180 minute rows of 60 slots (`tpa/run/evaluator.py:64-72, 131-150`; `tpa/run/metrics.py:124-158`). Per minute: `cons_rate` (share of slots with the gate open), `arrivals`, `completed`, `mem_util`; per run: `lt_occupancy` after the last slot (`evaluator.py:67-68`). Nothing per slot is written, and the MMPP state is not written; it is policy-independent and could be regenerated from (seed, block, test run) (`tpa/env/workload.py:96-97, 120-126, 175-178`), but it is not in the records. Hence: **P10 cannot be computed from the records; the paper states this in S-II (no new runs).** Two further facts, from the code and the config only: (i) even with per-slot records the frozen clustering statistic could not discriminate: with six independent zones and p_stay = (0.98, 0.95) a level change of some zone occurs about every 6 slots, so the stationary share of slots within w = 100 slots after a change is 1 - 8.0e-7 and the ratio is ~1 whatever the gate does; (ii) on a finite run it is about N / (N - c1 - 1) ~ 1.0007 > 1 for any gate (the slots before the first change lie in the 100-slot warm-up, where the gate cannot open), so with a t-interval over seeds it would have confirmed P10 for uniform openings too. The open fraction by load (part (a) of the request) is shown only as a descriptive table, not as a P10 test.

### A16A.2 Item 2 as implemented (`scripts/a16_partA.py`, called by `scripts/analyze_campaign.py`; phase main and the rehearsal)
- Cycle c of a 10,800-slot test run = minute rows 60(c-1)..60c-1 (run-start origin, P = 3,600). Trial = mean of the 3 test runs of a seed; 95 % t-intervals over seeds.
- Completion per cycle = sum completed / sum arrivals over the cycle's minute rows (the run-level definition restricted to the cycle). A0 - A3 per cycle, paired by seed, at every load, for cycles 1, 2, 3, cycles 2-3 pooled, and the whole run from the run rows (= H1 at heavy load). The primary test H1 (A0 - A3, heavy, whole run, TOST +-1 pp) is unchanged.
- Gate-open share per cycle = open slots / 3,600 (open slots per minute = 60 x cons_rate, exact); cycle 1 also over its post-warm-up slots (no opening is possible in the first w = 100 slots).
- Open phase bins: 8 bins of 450 slots from the run start (the plan's origin t0). Minutes 7, 22, 37, 52 of each cycle straddle a bin edge 30 | 30 and are split half-half for the point value, with sharp bounds (each half holds at most 30 openings). The bins are not phase-aligned across seeds; per-run counts are in `per_cycle_runs.csv`.
- Open intervals per cycle = circular runs of open bins under rule (i) bins certainly open (lower bound > 0) and rule (ii) bins holding >= 5 % of the cycle's openings; every (run, cycle) is classified as none (no opening) / all8 (all 8 bins open, not resolvable at P/8) / 1 / 2 / >=3, and the full distribution is reported.
- Long-term occupancy at the end of each cycle: exact for cycle 3 (the run row's lt_occupancy); bounds for cycles 1-2 from the minute means L = 600 x mem_util - S (S = minute mean of min(i + 1, K_S); snapped to the 1/60 grid), since L never decreases within a run; n/a for arms without a long-term store (A3; A1 is 0).
- Memory variants: every arm with memory (A0-A8, A13, A14, A15) and A0-tmem25.
- **P6 judged on cycles 2-3 (declaration, to be confirmed or changed before the registered analysis):** primary = the registered falsifier: any A0 run with end-of-run LT occupancy > 19; since the occupancy never decreases, this is also its maximum over cycles 2-3 and over the whole run. Secondary diagnostic: rows opened in cycles 2-3 (end of cycle 3 minus end of cycle 1, with bounds).
- **P9 judged on cycles 2-3 (declaration, likewise):** the A0 open share over cycles 2-3 (trial) against 0.10-0.14; the interval distribution of A0 (run, cycle) records in cycles 2-3 under rule (ii) (primary) and rule (i) (sensitivity), >=3 being the falsifier side; the refined P9: Pearson r with a Fisher-z 95 % CI and Spearman rho between kappa_phi (registered unweighted, |sum_z exp(2 i phi_z)| / Z of each seed's world) and the cycles-2-3 open share across seeds at each load, and the paired load differences light - heavy and medium - heavy. Whole-run values are reported beside each.

### A16A.3 Item 3, exploratory, NOT registered
A8 - B_x and A9 - B_x for B_ZHOU, B_QIN, B_HUANG, B_SONG at heavy load (n = 8): paired mean difference with 95 % t-CI, d and d_z; no p-values, not in any Holm family; `exploratory.csv` and TABLES section 2b, labelled "EXPLORATORY, NOT REGISTERED".

### A16A.4 Item 4, disclosure
The pilot (the pilot report: 12 runs of A0, A3, A9, A12 on seeds 8001-8003, run on 2026-10-23 under code 9a0b331cc10d, an earlier build than the campaign's 2892281e5ba9, evaluated on validation traces (val_full) only; decisions approved 2026-10-24, timing section added 2026-10-27) informed the power analysis and the equivalence margin.

### A16A.5 Corrections of the analysis script to the registered plan (made with Part A, before any result was read)
A13 left S1 (A7.1: tier C, effect size and CI only), so S1 has 9 members; tier C is reported apart, effect size and 95 % CI at every load for A0 - A13 (A7.1), A0 - A15 (A13.2) and A0 - A0-tmem25 (A13.3), outside every Holm family, with the only p-value being the A13.3 paired t of A0 vs A0-tmem25 at heavy load (A13.3 says "paired by seed at heavy load (descriptive at the other loads)"; flagged for a decision); the T_mem rows are labelled A0-tmem25 from their run id (A13.3; pooled, they had made H1 raise a length mismatch); the robustness arms are A3, A9, A10 (A7.3; A12 was a leftover of the superseded section 5); the family key is the queue workload, so the Gauss-Markov family is no longer merged into 'sinusoidal'; `--phase robust` refuses until every robust row is done, failed or stopped by a decision; the rehearsal reads only sinusoidal tuning rows; Markdown cells are escaped; every rehearsal output carries the REHEARSAL tag.

### A16A.6 Tests, review and rehearsal
`tests/test_analyze_campaign.py` (16 tests, synthetic inputs, known answers including kappa_phi(9003) = 0.4026, the Fig. 3(b) value) plus `tests/test_infra_stats.py`: 23 passed on the laptop and on the server. Read-only adversarial review in three areas (each finding re-checked by a second reviewer who tried to refute it; no run data read): the record semantics hold (mem_util, S_fill timing, monotone L, exact open counts, bounds); fixed: minute means snapped to the 1/60 grid (float noise could have counted exactly 19 opened rows as > 19), A3 occupancy n/a instead of a spurious exact 0, sharp straddler bounds, Markdown escaping of 'sinusoidal|mobility=...', an early-return crash path, the REHEARSAL tag, the interval distribution (the earlier "<= 2 intervals" share pooled 'none' and 'all8' with 1-2 intervals, biased toward the prediction), and P10 reported as not computable. Rehearsal on tuning data (validation block, on the server): every new section produced, no per-cycle error; its numbers are validation data and are not interpreted. Files: `scripts/a16_partA.py`, `scripts/analyze_campaign.py`, `tests/test_analyze_campaign.py` in each of the three code trees of the campaign (2892281e5ba9, 5253c84764d7, 790391a95b5d) and in the server's live tree (outside the code hash; codes unchanged).

### A16A.7 Paper facts confirmed from records (P10 decision, items 5-6)
Laptop robustness families: the 48 MMPP and Gauss-Markov rows run on the laptop (i9-13900HX), CPU only for every controller (row device cpu, learner.learn_device cpu, CUDA hidden), code 2892281e5ba9; the Pareto family has not started. Main campaign: every main manifest so far (135) records the compute server as host (VMware VM, Xeon Gold 5218R, 64 vCPUs), CPU only (CUDA unavailable to the workers). Pilot: see A16A.4.

### A16A.8 P6 / P9 decision (2026-11-10 19:15 UTC+3; verbatim; editorial changes in square brackets)
1. P6 ACCEPTED as declared (A16A.2/A16A.5): any A0 run whose long-term store ends with more than 19 entries
   falsifies P6. (Occupancy never decreases while K_L is not reached, so the end-of-run count equals the registered
   "after cycle 1" reading.) Also report, descriptively, per load: maximum, median and the number of runs above 19.
2. P9 ACCEPTED as declared: A0 open share over cycles 2-3; the distribution of open intervals per cycle under the
   5 % rule; the correlation of the per-world open share with kappa_phi. Compute kappa_phi per test world exactly as
   the paper does: |(1/Z) sum_z exp(2 i phi_z)| with the zone phases of build_world (world stream
   SeedSequence([seed, 1, 0])). Report the whole-run values beside the cycle 2-3 values.
3. A13.3 paired t (A0 vs A0 with T_mem = 25 ms, heavy load, n = 6) ACCEPTED; the paper now reports it as a paired
   t-test, not as an untested effect size.
4. Checkpoint selection on the sinusoid for every row: KEEP the registered rule for all families, including the
   Pareto family. No amendment. The rule is uniform across controllers; changing it for one family now would make
   families differ and needs new code before [the registered analysis]. The paper discloses that rows trained and tested on other
   workloads keep the checkpoint selected on the sinusoidal validation block.
5. P10: disclosed as not evaluable. Reply with one line naming the record that is missing (for the wording).
6. Continue: main campaign to completion (projected 13 Nov 02:45 UTC), the laptop robustness families and the
   Pareto family (A0, A3, A9, A10, n = 6). [In the registered analysis] run analyze_campaign.py once and [report] its
   outputs; nothing is read before. Keep the 3-hour watch and do no other work unless something fails […].

Applied on 2026-11-10, before any main or robustness result was read:
- (1) P6 as declared; the P6 table and `p6_p9.json` now give per load the maximum, the median and the number of runs above 19 of the end-of-run occupancy (`scripts/analyze_campaign.py` partA_summaries / partA_tables).
- (2) P9 as declared; kappa_phi is computed per test world as |(1/Z) sum_z exp(2 i phi_z)| from the zone phases of build_world with the world stream SeedSequence([seed, 1, 0]): that is the stream `tpa/rng.py:41-43` gives every block of a seed (`a16_partA.kappa_phi_of_seed`), and a unit test asserts the equality for seeds 1, 17 and 9003 (`tests/test_analyze_campaign.py`); whole-run values stand beside the cycle 2-3 values.
- (3) The A13.3 paired t (heavy load, n = 6) stays as implemented (tier-C table, outside every Holm family).
- (4) The registered checkpoint-selection rule stays for every family, the Pareto family included; no amendment; the paper discloses it.
- (5) P10 is disclosed as not evaluable. The line sent for the wording: "Missing records: the per-slot gate decision (the test logs keep only cons_rate, the share of gate-open slots per 60-slot minute row) and the per-zone MMPP state or its change times (not logged at all)."
- (6) The Pareto family (A0, A3, A9, A10 x test seeds 1-6 = 24 rows, Pareto on-off with the config defaults, run.episodes = 500, each arm's selected configuration from the stage-2 selection file, identical to the server's `runs/tune/selected.json`; config hashes identical to the running family's) is enqueued by `scripts/enqueue_robust.py --family pareto` (dry run: 24 rows) into the laptop queue `runs/robust/runs_robust.db` and started with 6 workers by a one-shot waiter script, started 2026-11-10 16:07 UTC, as soon as the MMPP / Gauss-Markov family has finished; it does nothing if a robust row has failed (log in the Amendment 16 records). In the registered analysis, `scripts/analyze_campaign.py` runs once per phase and its outputs are reported; nothing is read before.

## Amendment 16, Part B1 (2026-11-10; approved by the corresponding author, 19:20 UTC+3; evaluation-only sensitivity of the frozen A0 policies; no training, no tuning, no change to tpa/ or config/; registered before any B1 row existed and before any main-campaign or robustness test result was read; Part B2 (Milan trace) NOT approved)

### A16B1.0 Decision text (verbatim; editorial changes in square brackets)
Decisions of the corresponding author: B1 yes; B2 (Milan trace) no; title stays T1; biographies stay as they are;
public SHA-256 deposit of the plan on Zenodo yes.

B1 design (register in [this plan] as "Amendment 16 Part B1" BEFORE any test result is read)
Evaluation-only sensitivity of the frozen A0 policies. No training, no tuning, no change to tpa/.
- Checkpoints: the selected (tested) checkpoint of every main A0 row, heavy load, test seeds 1-30.
- Trace: the FIRST of the three registered test runs of each seed (same block and streams as the main test).
- One setting changed at a time, everything else as in the main campaign:
  theta_p in {0.30, 0.40}; rho_merge in {0.90, 0.98}; burst threshold theta_b at x0.75 and x1.25;
  each DRM priority weight (a_T, a_M, a_D) at x0.75 and x1.25; the load-balance weight omega_L at x0.75 and
  min(1, x1.25). Take the exact names and default values from [the specification and] default.toml and list them in the amendment.
  (About 14 settings x 30 seeds = about 420 runs of 3P slots; queue them behind every main row.)
- Statistic: paired difference in heavy-load completion to the default A0 on the same run, mean with 95 % t-interval,
  per setting; also the gate-open share and the long-term occupancy per setting. Descriptive only: no test,
  no Holm family. Reported in the paper as a sensitivity table (supplement) with one sentence in the main text.
- Nothing from these rows is read before [the registered analysis]; they are read with the rest.

Public deposit (after B1 is registered)
Write [a hash file] in the project root (plain text, no internal process words, no names of tools or
people other than the authors):
- title line: "SHA-256 commitments of the registered analysis plan, specification and analysis code";
- date and time (UTC) of computation;
- SHA-256 of [the registered plan, the registered specification], config/default.toml and scripts/analyze_campaign.py, as they are after B1 is
  registered ([the code tree of Amendment 15] used for the paper);
- the three code-tree hashes of the campaign (2892281e5ba9, 5253c84764d7, 790391a95b5d) with one line each saying
  which rows used them (main campaign; A13 rows; A16/A16' rows, stopped);
- one sentence: "The documents are available to the editor on request and will be released with the code."
The corresponding author uploads this file to Zenodo and publishes it before [the registered analysis].

### A16B1.1 Settings (names and defaults from the specification and config/default.toml; one setting changed at a time)
Every B1 row uses the configuration of the main A0 rows, i.e. the queue overrides of `main-A0-s<seed>-sinusoidal`
(burst.theta_b = 0.9, learner.lr = 0.00239, learner.target_entropy_scale = 0.25, learner.update_every = 8,
run.episodes = 500, scenario.n_uav = 25, task.deadline_s = [3.0, 10.0], task.size_mb = [0.5, 2.5],
workload.loads = [0.17, 0.374, 0.578]; the enqueue step takes them from the 30 queue rows and refuses unless they are
identical), plus run.test_runs = 1 (A16B1.2), plus at most one of the settings below. A multiplier applies to the value
in force in the main campaign: for burst.theta_b that is A0's selected 0.9 (the stage-2 selection file; default.toml holds
the tuning start 0.75 [T]), for every other key the default.toml value. The values are the decimal literals below,
never computed products.

| specification symbol | config key | specification table | config/default.toml (line) | main A0 value | B1 values | run-id tags |
|---|---|---|---|---|---|---|
| θ_p, significance gate | memory.theta_p | 0.15 T (specification §3a, row θ_p, superseded by Amendment 6) | 0.35 (117, adopted in Amendment 6, not tuned) | 0.35 | 0.30, 0.40 | theta_p-0.30, theta_p-0.40 |
| ρ_merge, long-term merge threshold | memory.rho_merge | 0.8 (specification §3a, row ρ_merge, superseded by Amendment 6) | 0.95 (106, adopted in Amendment 6) | 0.95 | 0.90, 0.98 | rho_merge-0.90, rho_merge-0.98 |
| θ_b, burst threshold (Eq. 57) | burst.theta_b | 0.75 T (specification §3a, row θ_b) | 0.75 [T] (187) | 0.9 (tuned) | 0.675, 1.125 | theta_b-x0.75, theta_b-x1.25 |
| α_T, priority weight (Eq. 52) | drm.alpha_t | 0.4 (specification §3a, row α_T, α_M, α_D) | 0.4 (171) | 0.4 | 0.3, 0.5 | alpha_t-x0.75, alpha_t-x1.25 |
| α_M, priority weight (Eq. 52) | drm.alpha_m | 0.3 (specification §3a, row α_T, α_M, α_D) | 0.3 (172) | 0.3 | 0.225, 0.375 | alpha_m-x0.75, alpha_m-x1.25 |
| α_D, priority weight (Eq. 52) | drm.alpha_d | 0.3 (specification §3a, row α_T, α_M, α_D) | 0.3 (173) | 0.3 | 0.225, 0.375 | alpha_d-x0.75, alpha_d-x1.25 |
| ω_L, load-balance weight (Eq. 56) | drm.omega_l | 0.6 (specification §3a, row ω_L) | 0.6 (180) | 0.6 | 0.45, 0.75 (= min(1, 0.75)) | omega_l-x0.75, omega_l-x1.25 |

14 settings. Each acts inside the controller only; the simulator, the world and the random streams read none of these
keys (`tpa/env/`, `tpa/rng.py`): θ_p is the MEPR gate threshold (`tpa/memory/mepr.py:128`), ρ_merge the long-term
addressing threshold (`tpa/memory/stores.py:293, 590`), θ_b the Eq. 57 threshold (`tpa/drm/burst.py:51, 89`), α_T,
α_M, α_D the Eq. 52 priority weights (`tpa/drm/drm.py:363, 447`; π is not renormalised, so one weight changes alone),
ω_L the Eq. 56 load-index weight (`tpa/drm/drm.py:368, 481`). A0's arm flags override none of them, and the config
loader applies no bounds (`tpa/config.py`). The load pressure Φ_j of Eq. 58 is not bounded by 1 (`tpa/drm/burst.py:32-43`),
so θ_b = 1.125 is a working threshold.

### A16B1.2 Rows, evaluation and queue
- Rows: 15 per test seed 1-30, the 14 settings and one default re-evaluation with no setting changed (tag `default`):
  450 rows. The approval's "about 420" counts the settings; the 30 default rows give every paired difference one code
  path (A16B1.3).
- Source checkpoint: the selected checkpoint (`ckpt_best.pt`, chosen on the validation block by the registered rule)
  of `main-A0-s<seed>-sinusoidal`. That row must be done; its queue overrides plus run.test_runs = 1 plus the setting
  must equal the B1 row's overrides, and its code hash must equal the B1 row's (2892281e5ba9). The SHA-256 of the
  checkpoint file goes into the B1 row's manifest and RESULT.json.
- Trace: the main test's run k = 0 at heavy load 0.578: Streams(seed, "test", 0), t0 = run.test_seed_offset_slots =
  10,000,000, 10,800 slots = 3P (`tpa/run/evaluator.py:136-146`). With run.test_runs = 1, evaluate() runs only k = 0;
  loads = [0.578].
- Evaluation: `scripts/b1_eval.py` eval_only() builds the controller as `tpa/run/trainer.py` run() does before
  training (lines 117-124: train streams, build_world, the segment-0 simulator that supplies r_ref and h_min,
  make_controller), loads the checkpoint's learned state as run() does before its test evaluation (lines 227-228,
  _merge_net_state; the checkpoint holds the learned_state_keys, i.e. networks, memory parameters and the frozen
  normaliser, `tpa/control.py:298-302`), and calls the unchanged `tpa.run.evaluator.evaluate(block="test")`. The
  evaluation clone takes only that learned state from the controller (`tpa/control.py:289-296`), and deterministic
  actions draw no random numbers (`tpa/learner/sac.py:104-116`; on the CPU the actor mirror is the actor and
  set_state re-syncs it, lines 43-50 and 223-235). So a default row should reproduce the main test's first heavy run
  exactly; A16B1.6 checks this on validation data, and the registered analysis reports it for the test data (A16B1.3).
- Queue: an own queue file `runs/b1/runs_b1.db` (schema of `tpa/run/queue.py`; phase main; queue arm A0; run ids
  `b1-A0-s<seed>-<tag>`; seed-major claim order, the default row first), never the main queue. "Behind every main row":
  a B1 row is claimable only when no main row is pending in `runs/runs.db` (every main row has been claimed) and its
  source row is done with `ckpt_best.pt` and `RESULT.json` present. The B1 launcher (`scripts/b1_eval.py --launch`,
  on the server) adds B1 workers only on the campaign's vCPUs (1-20, 32-51) that no
  live campaign worker uses, never beyond 30 campaign workers in total (main and B1, the current allocation), and only while
  MemAvailable and disk stay above floors; it never signals or changes any other process. Stop:
  a stop script (by PID only).

### A16B1.3 Statistic (descriptive only: no test, no p-value, no Holm family)
For setting s and seed i: d_si = completion(s, i) - completion(default, i) at heavy load, from the two B1 rows of seed
i (same checkpoint, same trace, same code path). Per setting: n, the mean of d_s and its 95 % t-interval
(t_0.975,n-1 x sd / sqrt(n)) over the seeds with both rows done. Also per setting the mean and 95 % t-interval of the
gate-open share (the run row's cons_rate, the share of gate-open slots of the 10,800-slot run) and of the long-term
occupancy at the end of the run (the run row's lt_occupancy), with the default rows' means beside them and the paired
differences of both to the default. Completion is the run row's completion, the metric of the primary analysis.
Reproduction check (reported, not a test): per seed, the default row's run row against the main test's first heavy run
of `main-A0-s<seed>-sinusoidal` (row_kind run, load 0.578, test_run 0), in every column except the wall-clock ones
(wall_s, reaction_ms_p50, reaction_ms_p90, recovery_ms and the t_*_ms timers): the number of seeds identical in all of
them, and the largest absolute difference of completion, cons_rate and lt_occupancy. If a seed differs, the table
still uses the B1 default rows and the difference is reported. A failed B1 row is reported as missing; its seed drops
out of that setting's pairs. The paper reports the table in the supplement and one sentence in the main text.

### A16B1.4 Reading
Nothing from the B1 rows is read before the registered analysis. `scripts/analyze_campaign.py --phase main` reads them with the main
campaign; it refuses unless every main row and every B1 row is done or failed, and writes `b1_sensitivity.csv`,
`b1_reproduction.csv` and section 7 of `TABLES.md`.

### A16B1.5 Public deposit
The hash file in the project root: the SHA-256 of the registered plan, the registered specification, config/default.toml and
scripts/analyze_campaign.py of the code tree of Amendment 15 after this amendment and its implementation, the campaign's
code-tree hashes with the rows that used them (taken from the queue), and the approval's closing sentence. The
corresponding author uploads it to Zenodo before the registered analysis.

### A16B1.6 Implementation and tests (implementation record)
Files (outside the code hash; no change to tpa/ or config/): `scripts/b1_eval.py` (row plan, gate, eval_only,
worker, launcher, --status, --release-dead), `tests/test_sensitivity_eval.py`, the B1 part of `scripts/analyze_campaign.py`
(b1_rows guard, b1_analysis, b1_tables; outputs b1_sensitivity.csv, b1_runs.csv, b1_reproduction.csv, TABLES.md
section 7) with its tests at the end of `tests/test_analyze_campaign.py`, and the B1 deployment scripts (start, stop
and finish projection); the 3-hour watch counts B1 workers and prints the B1 queue.
- Validation-data check (validation seed 9001, validation block only; last test of `tests/test_sensitivity_eval.py`): a short
  A0 training run (tpa.run.trainer.run, phase tune, 2 episodes, 600-slot evaluation runs) evaluated itself on val_full
  (3 loads x 3 runs, heavy run k = 0 after light and medium). eval_only() of its ckpt_best.pt on val_full run k = 0 at
  heavy load gave the same run row and minute rows, bit for bit in every column but the wall-clock ones, on the laptop
  and on the server. The same evaluation without the checkpoint's learned state differs (completion, wait_s, cpu_util,
  sigma_mean). alpha_t x1.25 and omega_l x0.75 change the run; theta_b x0.75 leaves that barely trained policy's run
  unchanged (its load pressure never lies between 0.675 and 0.9). Separate tests show that each of the 14 settings
  reaches the evaluation clone (MEPR theta, memory rho, Burst theta_b, DRM a_T, a_M, a_D, omega_l) and that nothing else
  changes there, and that the world and the streams do not depend on any setting.
- Tests: laptop, final code: 64 passed, 1 skipped (a missing-trainer test that skips by design); 43 of them are in tests/test_sensitivity_eval.py (20) and tests/test_analyze_campaign.py (23). Server: 57 passed, 1 skipped on the code before the second review's fixes; the server run on the final code was cut off when the server became unreachable and is repeated before the enqueue.
- Review (two read-only adversarial rounds, each finding re-checked by a second reviewer who tried to refute it): round 1 on four areas (evaluation fidelity, server safety, analysis, plan text), 16 findings: 2 confirmed (the launcher did not return a dead worker's row; its time-limit and error exits left workers unreleased) and fixed; 9 more adopted as hardening (a B1 worker yields to an overbooked vCPU share of the campaign; queue-file and --dry-run guards; the stop script reports a failed release; rows 'stopped by' a decision are never claimed; the analysis fails closed without the B1 queue; per-seed checkpoint check; stronger tests; the plan-wording corrections below). The fidelity review found no deviation, and an independent in-memory check of the default-row reproduction gave 0 differences. Round 2 on the fixes, 7 findings: 4 confirmed and fixed (the drain error counter; release of workers that exit in a failing pass; --launch and --release-dead no longer create a queue file, and --launch needs all 450 rows; a pandas NaN in the hash check), 3 more adopted (the analysis checks the registered row set, reports for how many seeds the main checkpoint was compared, and --b1-root follows --b1-db). Launcher simulation tests were added, and a mutation check showed that undoing either launcher fix fails its test.
- Corrections to the wording of A16B1.1 and A16B1.2 (made before any B1 row ran): (i) besides the controller, the
  unchanged evaluator also reads memory.theta_p, to count detections (`tpa/run/evaluator.py:36, 59`); that test is the
  controller's own gate test on the same sigma_p (`tpa/memory/mepr.py:128`), so it only replays the gate, and no B1
  statistic uses the detection columns. (ii) The evaluation clone is built from the controller's cfg (the B1 row's),
  r_ref and h_min as well as from its learned state (`tpa/control.py:292-295`); r_ref and h_min are world-level values,
  which the segment-0 simulator reproduces. (iii) The actor mirror is re-synced at `tpa/learner/sac.py:237`; on the CPU
  the mirror is the actor itself (lines 44-48).
- Guard details: a B1 row stopped by a decision, or one whose source row failed (it can never run), is reported as
  missing and does not block the analysis; a main row stopped by a decision does not hold B1 back. In --phase main the
  analysis also refuses when the B1 queue file is missing, unless --no-b1 is given (recorded in TABLES.md section 7 and
  summary.json). The B1 rows of a seed count only if all of them carry one checkpoint hash and it equals the SHA-256 of
  that seed's main ckpt_best.pt; otherwise the seed's B1 rows are left out and reported. Restart order while B1
  workers run: stop B1 first, then stop the main-side workers, start the main and the extra launchers, and start B1 last; B1 workers also
  yield by themselves when more than 30 campaign workers run or another campaign worker shares their vCPU.
- Deposit (A16B1.5): the queue, not the approval's parenthesis, gives the rows per code-tree hash. 2892281e5ba9
  carries every main-campaign row (all arms, the A13, A15 and T_mem rows included), the robustness rows, the B1 rows
  and the tuning rows of A12 and two baselines; 5253c84764d7 carries only the six A16 rows (stopped); 790391a95b5d
  carries the six A16' rows (stopped) and one A0 reference row on the Pareto workload. The tuning trees behind the
  selected configurations (044907211a93, 02e2abd853f8, 1f4eb52fa26b) are outside the approval's three hashes and are
  not listed.
- Queue: the 450 rows are written once by `scripts/b1_eval.py --enqueue` on the server (a dry run there on
  2026-11-10 planned 450 rows with 15 config hashes under code 2892281e5ba9 from the 30 main A0 rows, whose overrides
  are identical), and the launcher is started once with the B1 start script; the times are recorded in the code-review record,
  and the B1 projection script projects the finish from queue timings only.

### A16B1.7 Confirmations (2026-11-11, 01:55 UTC+3; verbatim; editorial changes in square brackets; before any B1 or main row was read)
1. theta_b: CONFIRMED as registered: base = the selected A0 value 0.9, settings x0.75 = 0.675 and x1.25 = 1.125.
   The trigger is Phi_j > theta_b (tpa/drm/burst.py, Eq. 57/58); Phi_j can exceed 1 when priorities exceed 1, so 1.125
   is a valid probe, outside the tuned grid {0.6, 0.75, 0.9}. For every B1 setting also report the share of
   UAV-slots with burst on, so the reader sees how much 1.125 suppresses burst mode.
2. Paired reference: CONFIRMED: the 30 default B1 rows (one per seed, same checkpoint, first test run) are each
   seed's paired reference. [In the registered analysis] also check that each default B1 row equals the main A0 first test run of the
   same seed (completion bit for bit); report any difference in one line; the B1 differences use the default B1 rows
   either way.
3. Memory: TPA_MEM 69.5 GB against the 75 GB trigger. Keep the existing rule (no new worker above the trigger);
   B1 rows replace finished main rows and do not fill replay buffers, so no change is needed now.

Applied on 2026-11-11, before any B1 or main row was read:
- (1) theta_b as registered (0.675 and 1.125 around A0's 0.9). Per setting the analysis now also reports the share of
  UAV-slots with burst on, both levels (setting and default) and the paired difference, with 95 % t-intervals. It is
  exact from the run row: the simulator charges delta_e_w x slot_s of burst energy for every UAV-slot with B_j = 1
  (`tpa/env/simulator.py:486-487`), and the run row sums it as e_burst_j (`tpa/run/metrics.py:95, 148`). So
  share = e_burst_j / (burst.delta_e_w x scenario.slot_s x scenario.n_uav x slots), with the row's own configuration
  (`scripts/analyze_campaign.py` b1_burst_on_share). The run row's burst_share is a different quantity: the mean
  excess utilisation above nominal capacity, `tpa/run/metrics.py:86`.
- (2) The paired reference is the default B1 row, as implemented. The reproduction check already compares every
  non-wall-clock column. Section 7 now adds one line naming the seeds whose default-row completion differs bit for bit
  from the main test's first heavy run (or saying that none does); the B1 differences use the default B1 rows either
  way.
- (3) Memory: no change.
- The hash file is computed again after this record: the plan and scripts/analyze_campaign.py changed,
  and the file of 2026-11-10 19:07 UTC is superseded.
