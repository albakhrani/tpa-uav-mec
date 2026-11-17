# Campaign analysis: phase robust, block test

Generated 2026-11-13T09:45:20+00:00 by `scripts/analyze_campaign.py`. Rows of phase `robust` in the queue: 72; done 72; failed (reported missing): none; stopped (reported missing): none; run directories without a `test` table: none. Code hashes in the data: ['2892281e5ba9']. Workloads: ['mmpp', 'pareto', 'sinusoidal|mobility=gauss_markov']. Loads (light, medium, heavy): 0.17, 0.374, 0.578. Seeds per arm at heavy load: {'A0': 6, 'A10': 6, 'A3': 6, 'A9': 6}. alpha = 0.05; SESOI = 1.0 pp.

## 2. Contrasts (Holm within each family; completion, pp)

| family | contrast | n | mean diff | 95 % CI | d | d_z | p | Holm p | TOST p | MDE (pp) | Shapiro p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R:mmpp | A0 - A3, heavy, mmpp | 6 | -0.21 | [-0.91, 0.49] | -0.0313 | -0.316 | 0.474 | 0.474 | 0.0172 | 0.95 | 0.0721 |
| R:mmpp | A0 - A9, heavy, mmpp | 6 | -0.75 | [-1.47, -0.04] | -0.113 | -1.11 | 0.0421 | 0.0841 | 0.209 | 0.97 | 0.328 |
| R:mmpp | A0 - A10, heavy, mmpp | 6 | -1.76 | [-2.95, -0.56] | -0.247 | -1.54 | 0.013 | 0.0391 | 0.917 | 1.63 | 0.8 |
| R:pareto | A0 - A3, heavy, pareto | 6 | 0.38 | [-0.08, 0.84] | 0.083 | 0.875 | 0.0849 | 0.0849 | 0.00892 | 0.62 | 0.274 |
| R:pareto | A0 - A9, heavy, pareto | 6 | -1.36 | [-2.17, -0.55] | -0.277 | -1.76 | 0.00756 | 0.0227 | 0.849 | 1.10 | 0.563 |
| R:pareto | A0 - A10, heavy, pareto | 6 | -2.39 | [-4.14, -0.64] | -0.45 | -1.44 | 0.017 | 0.034 | 0.952 | 2.38 | 0.344 |
| R:sinusoidal\|mobility=gauss_markov | A0 - A3, heavy, sinusoidal\|mobility=gauss_markov | 6 | -0.55 | [-1.28, 0.19] | -0.063 | -0.78 | 0.114 | 0.114 | 0.0878 | 1.00 | 0.441 |
| R:sinusoidal\|mobility=gauss_markov | A0 - A9, heavy, sinusoidal\|mobility=gauss_markov | 6 | -1.55 | [-2.65, -0.46] | -0.168 | -1.49 | 0.0147 | 0.0293 | 0.876 | 1.48 | 0.281 |
| R:sinusoidal\|mobility=gauss_markov | A0 - A10, heavy, sinusoidal\|mobility=gauss_markov | 6 | -1.39 | [-2.12, -0.67] | -0.154 | -2.01 | 0.00437 | 0.0131 | 0.889 | 0.99 | 0.161 |

## 3. Descriptive outcomes (mean over seeds, +- 95 % t-CI half-width; wall-clock columns excluded)

| arm | workload | load | n | completion | resource_eff | resource_eff_cb | cpu_util | energy_eff | hit_rate | wait_s | trans_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 | mmpp | 0.170 | 6 | 0.5508 +- 0.15 | 0.1654 +- 0.0098 | 0.07989 +- 0.01 | 0.4598 +- 0.1 | 3443 +- 5.6e+02 | nan | 1.345 +- 0.46 | 2.559 +- 0.16 |
| A10 | mmpp | 0.170 | 6 | 0.5782 +- 0.17 | 0.06595 +- 0.0086 | 0.08376 +- 0.011 | 0.4285 +- 0.065 | 4081 +- 1.2e+03 | nan | 1.272 +- 0.54 | 2.581 +- 0.19 |
| A3 | mmpp | 0.170 | 6 | 0.5538 +- 0.15 | 0.259 +- 0.0098 | 0.08108 +- 0.013 | 0.4256 +- 0.079 | 3541 +- 5.1e+02 | nan | 1.342 +- 0.46 | 2.594 +- 0.21 |
| A9 | mmpp | 0.170 | 6 | 0.5583 +- 0.15 | 0.06429 +- 0.0089 | 0.08166 +- 0.011 | 0.4644 +- 0.085 | 3644 +- 4.4e+02 | nan | 1.32 +- 0.45 | 2.572 +- 0.18 |
| A0 | mmpp | 0.374 | 6 | 0.2842 +- 0.098 | 0.1726 +- 0.0084 | 0.09078 +- 0.009 | 0.5164 +- 0.11 | 3873 +- 8.3e+02 | nan | 1.069 +- 0.22 | 2.723 +- 0.23 |
| A10 | mmpp | 0.374 | 6 | 0.3071 +- 0.11 | 0.07672 +- 0.0084 | 0.09752 +- 0.011 | 0.4872 +- 0.077 | 4590 +- 1.1e+03 | nan | 1.019 +- 0.16 | 2.702 +- 0.19 |
| A3 | mmpp | 0.374 | 6 | 0.287 +- 0.098 | 0.268 +- 0.0086 | 0.0925 +- 0.011 | 0.4732 +- 0.084 | 4023 +- 7.5e+02 | nan | 1.066 +- 0.22 | 2.77 +- 0.29 |
| A9 | mmpp | 0.374 | 6 | 0.2933 +- 0.097 | 0.07424 +- 0.0077 | 0.09439 +- 0.01 | 0.5245 +- 0.087 | 4118 +- 6e+02 | nan | 1.049 +- 0.2 | 2.705 +- 0.22 |
| A0 | mmpp | 0.578 | 6 | 0.193 +- 0.07 | 0.1771 +- 0.0089 | 0.09798 +- 0.0092 | 0.5573 +- 0.12 | 4219 +- 9.8e+02 | nan | 0.6674 +- 0.31 | 2.961 +- 0.32 |
| A10 | mmpp | 0.578 | 6 | 0.2105 +- 0.079 | 0.08334 +- 0.009 | 0.1061 +- 0.012 | 0.5297 +- 0.091 | 4944 +- 1.2e+03 | nan | 0.6396 +- 0.27 | 2.909 +- 0.27 |
| A3 | mmpp | 0.578 | 6 | 0.1951 +- 0.072 | 0.2736 +- 0.0084 | 0.09986 +- 0.011 | 0.5076 +- 0.091 | 4448 +- 9.6e+02 | nan | 0.6651 +- 0.31 | 3.019 +- 0.38 |
| A9 | mmpp | 0.578 | 6 | 0.2005 +- 0.07 | 0.08055 +- 0.0079 | 0.1026 +- 0.01 | 0.5699 +- 0.095 | 4442 +- 6.2e+02 | nan | 0.6563 +- 0.3 | 2.916 +- 0.3 |
| A0 | pareto | 0.170 | 6 | 0.5016 +- 0.14 | 0.1578 +- 0.0095 | 0.07435 +- 0.01 | 0.4335 +- 0.079 | 3526 +- 6.4e+02 | nan | 1.301 +- 0.31 | 2.76 +- 0.3 |
| A10 | pareto | 0.170 | 6 | 0.5318 +- 0.15 | 0.06152 +- 0.0074 | 0.07817 +- 0.0096 | 0.4033 +- 0.063 | 4682 +- 1.1e+03 | nan | 1.272 +- 0.37 | 2.704 +- 0.24 |
| A3 | pareto | 0.170 | 6 | 0.4875 +- 0.13 | 0.2532 +- 0.0088 | 0.07247 +- 0.011 | 0.4071 +- 0.083 | 3517 +- 5.1e+02 | nan | 1.33 +- 0.27 | 2.738 +- 0.21 |
| A9 | pareto | 0.170 | 6 | 0.5119 +- 0.14 | 0.05906 +- 0.0073 | 0.07502 +- 0.0094 | 0.42 +- 0.069 | 4034 +- 7.2e+02 | nan | 1.327 +- 0.34 | 2.659 +- 0.2 |
| A0 | pareto | 0.374 | 6 | 0.2417 +- 0.07 | 0.162 +- 0.011 | 0.0802 +- 0.012 | 0.474 +- 0.1 | 3777 +- 6.3e+02 | nan | 0.9784 +- 0.29 | 2.981 +- 0.37 |
| A10 | pareto | 0.374 | 6 | 0.2713 +- 0.085 | 0.07012 +- 0.008 | 0.08919 +- 0.01 | 0.4502 +- 0.082 | 5181 +- 1.3e+03 | nan | 0.9462 +- 0.22 | 2.875 +- 0.26 |
| A3 | pareto | 0.374 | 6 | 0.237 +- 0.067 | 0.2581 +- 0.0093 | 0.0788 +- 0.012 | 0.4416 +- 0.1 | 3978 +- 7.2e+02 | nan | 0.9852 +- 0.29 | 2.982 +- 0.32 |
| A9 | pareto | 0.374 | 6 | 0.2584 +- 0.077 | 0.06686 +- 0.0083 | 0.08504 +- 0.011 | 0.4756 +- 0.083 | 4403 +- 7e+02 | nan | 0.9783 +- 0.25 | 2.831 +- 0.26 |
| A0 | pareto | 0.578 | 6 | 0.1618 +- 0.049 | 0.1657 +- 0.011 | 0.08514 +- 0.012 | 0.5089 +- 0.11 | 4055 +- 6.7e+02 | nan | 0.6004 +- 0.28 | 3.257 +- 0.42 |
| A10 | pareto | 0.578 | 6 | 0.1858 +- 0.062 | 0.07604 +- 0.0074 | 0.09688 +- 0.0096 | 0.4892 +- 0.089 | 5685 +- 1.5e+03 | nan | 0.5739 +- 0.23 | 3.121 +- 0.3 |
| A3 | pareto | 0.578 | 6 | 0.158 +- 0.048 | 0.2616 +- 0.0088 | 0.08333 +- 0.011 | 0.4722 +- 0.11 | 4339 +- 8.1e+02 | nan | 0.6068 +- 0.28 | 3.264 +- 0.35 |
| A9 | pareto | 0.578 | 6 | 0.1755 +- 0.054 | 0.07217 +- 0.0085 | 0.09194 +- 0.011 | 0.5184 +- 0.086 | 4737 +- 7.3e+02 | nan | 0.5978 +- 0.26 | 3.074 +- 0.31 |
| A0 | sinusoidal\|mobility=gauss_markov | 0.170 | 6 | 0.6239 +- 0.21 | 0.1067 +- 0.013 | 0.08685 +- 0.015 | 0.4986 +- 0.1 | 3070 +- 4.9e+02 | 0.0581 +- 0.068 | 1.356 +- 0.79 | 2.395 +- 0.22 |
| A10 | sinusoidal\|mobility=gauss_markov | 0.170 | 6 | 0.6531 +- 0.22 | 0.07247 +- 0.013 | 0.09198 +- 0.016 | 0.5266 +- 0.093 | 3768 +- 8.4e+02 | nan | 1.233 +- 0.78 | 2.447 +- 0.17 |
| A3 | sinusoidal\|mobility=gauss_markov | 0.170 | 6 | 0.6402 +- 0.22 | 0.2657 +- 0.014 | 0.09039 +- 0.017 | 0.483 +- 0.081 | 3259 +- 7.1e+02 | 0 +- 0 | 1.298 +- 0.8 | 2.459 +- 0.21 |
| A9 | sinusoidal\|mobility=gauss_markov | 0.170 | 6 | 0.6521 +- 0.22 | 0.07182 +- 0.012 | 0.09113 +- 0.015 | 0.5158 +- 0.094 | 3742 +- 5.7e+02 | nan | 1.233 +- 0.82 | 2.428 +- 0.19 |
| A0 | sinusoidal\|mobility=gauss_markov | 0.374 | 6 | 0.3182 +- 0.12 | 0.1143 +- 0.0096 | 0.09846 +- 0.011 | 0.547 +- 0.087 | 3472 +- 4.8e+02 | 0.05478 +- 0.065 | 1.007 +- 0.38 | 2.566 +- 0.35 |
| A10 | sinusoidal\|mobility=gauss_markov | 0.374 | 6 | 0.3359 +- 0.13 | 0.08174 +- 0.0094 | 0.1038 +- 0.012 | 0.5928 +- 0.1 | 4026 +- 8.5e+02 | nan | 0.9821 +- 0.36 | 2.553 +- 0.33 |
| A3 | sinusoidal\|mobility=gauss_markov | 0.374 | 6 | 0.3252 +- 0.12 | 0.2746 +- 0.011 | 0.1018 +- 0.014 | 0.5256 +- 0.074 | 3806 +- 9.4e+02 | 0 +- 0 | 0.9986 +- 0.38 | 2.614 +- 0.39 |
| A9 | sinusoidal\|mobility=gauss_markov | 0.374 | 6 | 0.3395 +- 0.14 | 0.08187 +- 0.0078 | 0.104 +- 0.01 | 0.579 +- 0.088 | 4180 +- 4.8e+02 | nan | 0.9756 +- 0.34 | 2.541 +- 0.31 |
| A0 | sinusoidal\|mobility=gauss_markov | 0.578 | 6 | 0.2187 +- 0.092 | 0.121 +- 0.0088 | 0.1078 +- 0.011 | 0.5953 +- 0.082 | 3811 +- 5e+02 | 0.05195 +- 0.061 | 0.5325 +- 0.31 | 2.831 +- 0.4 |
| A10 | sinusoidal\|mobility=gauss_markov | 0.578 | 6 | 0.2327 +- 0.098 | 0.0897 +- 0.0094 | 0.1141 +- 0.012 | 0.6453 +- 0.1 | 4387 +- 8.5e+02 | nan | 0.5207 +- 0.3 | 2.787 +- 0.36 |
| A3 | sinusoidal\|mobility=gauss_markov | 0.578 | 6 | 0.2242 +- 0.091 | 0.2822 +- 0.0097 | 0.1117 +- 0.013 | 0.5697 +- 0.08 | 4231 +- 1e+03 | 0 +- 0 | 0.5244 +- 0.31 | 2.878 +- 0.43 |
| A9 | sinusoidal\|mobility=gauss_markov | 0.578 | 6 | 0.2343 +- 0.1 | 0.08969 +- 0.0084 | 0.1141 +- 0.011 | 0.6316 +- 0.084 | 4579 +- 5.7e+02 | nan | 0.52 +- 0.3 | 2.775 +- 0.35 |

Full descriptive table (all 19 metrics): `descriptive.csv`; trial table: `trials.csv`; contrasts: `contrasts.csv`.

## 4. Figures

The analysis also writes these figures, which are not part of this repository: fig_forest_R_mmpp.png, fig_forest_R_pareto.png, fig_forest_R_sinusoidal_mobility-gauss_markov.png, fig_completion_vs_load.png and fig_minute_series_heavy.png.

## 6. P10 on MMPP (registered: openings cluster after level changes, fraction lower at light load; falsifier: uniform openings)

**P10 cannot be computed from the records (no per-slot gate decision and no MMPP state are written); it is not evaluated. The open fraction by load below is descriptive only.** Clustering ratio: the test records hold no per-slot gate decision (only the per-minute share cons_rate, tpa/run/metrics.py) and no MMPP state (tpa/run/evaluator.py writes run and minute rows only); the MMPP state is policy-independent and could be regenerated from (seed, block, test run), but it is not in the records. Base share of the frozen statistic: 0.9999992 (stationary share of slots within w = 100 slots after a level change of any zone for mmpp_p_stay = [0.98, 0.95], Z = 6: the frozen ratio is ~1 whatever the gate does, so it could not discriminate even with per-slot records; on a finite run it is about N / (N - c1 - 1) ~ 1.0007 > 1 for ANY gate (the slots before the first level change lie in the warm-up, where the gate cannot open), i.e. biased toward confirming P10).

Descriptive only, NOT a P10 test: open fraction per run (= per trial: the mean of the run's 3 test runs; seeds as units; share of slots with the gate open; mean +- 95 % t-CI):

| arm | load | n | all slots | post-warm-up slots |
|---|---|---|---|---|
| A0 | light | 6 | 0.6635 +- 0.0229 | 0.6697 +- 0.0231 |
| A0 | medium | 6 | 0.6567 +- 0.0179 | 0.6628 +- 0.0181 |
| A0 | heavy | 6 | 0.6524 +- 0.0212 | 0.6585 +- 0.0214 |
| A3 | light | 6 | 0.7062 +- 0.0509 | 0.7128 +- 0.0514 |
| A3 | medium | 6 | 0.6984 +- 0.0529 | 0.7049 +- 0.0534 |
| A3 | heavy | 6 | 0.6948 +- 0.0545 | 0.7013 +- 0.0551 |
