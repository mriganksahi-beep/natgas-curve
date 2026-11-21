# H0 / H0b: Model 3 on the holdout

Full results: `reports/model3_results_holdout.csv` (all methods and horizons) and `reports/holdout/model3_H0b.csv`. The protocol's own cost convention is "k ticks round trip per leg", so 4 ticks = $80 per spread. The synthesis convention is k ticks per contract per side, so 4 ticks = $160 per spread. Both are shown below.

| h | Method | OOS R² | DM p | Net $/trade @4 ticks per leg (protocol) | Net @4 ticks per side |
| --- | --- | --- | --- | --- | --- |
| 1 | linear | 0.03 | 0.42 | −53 | −133 |
| 1 | convex | 0.06 | 0.35 | +35 | −45 |
| 1 | spline | 0.05 | 0.38 | −92 | −172 |
| 1 | gbm | 0.09 | 0.24 | −25 | −105 |
| 1 | mlp | 0.10 | 0.27 | −86 | −166 |
| 4 | linear | 0.19 | 0.31 | +409 | +329 |
| 4 | convex | 0.45 | 0.17 | +556 | +476 |
| 4 | spline | 0.29 | 0.26 | +396 | +316 |
| 4 | gbm | 0.43 | 0.15 | +363 | +283 |
| 4 | mlp | 0.46 | 0.15 | +527 | +447 |

**Verdict:** fail. No method meets criterion 1 (DM p < 0.05).

**H0b** (\|gap\| > $0.30, decided before the holdout): the nonlinear methods traded only in Nov 2022, +$6,725 net per spread at 4 weeks. Linear traded in Nov 2022, Dec 2023 and Jan 2024, +$2,197. Positive, but resting on one or two episodes.
