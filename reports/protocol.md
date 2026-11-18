# Evaluation protocol

*Written 17 November 2026, before any futures-price data has been loaded. Changes after this date are logged at the bottom with the reason.*

## Question
How far up the chain **weather → weekly storage → end-of-season storage → futures spread** can free public data forecast out of sample, and which methods work at each step?

## Data
| Item | Source | Use |
| --- | --- | --- |
| Weekly working gas in storage, Lower 48 | EIA (series NW2_EPG0_SWO_R48_BCF), 2010– | Targets for Models 1 and 2 |
| Daily mean temperature, 17 US cities | Open-Meteo archive | Gas-weighted heating and cooling degree days |
| Henry Hub futures, contracts 1–4 | EIA, 1994 – 5 Apr 2024 | Model 3 cross-check |
| Named Henry Hub contracts (e.g. March, April) | Databento (free credit), 2010– | Model 3 targets |

## Timing rules (no look-ahead)
- The EIA storage week ends on Friday and is published the following Thursday at 10:30 ET.
- **Models 1–2:** a forecast for week *t* may use weather through Friday *t*, but storage only through week *t−1*.
- **Model 3:** the forecast is made at Thursday's settlement price, after that day's storage report. It may use only information public by then.
- Every model is refit walk-forward, using only rows dated before the forecast.
- Automated tests in `tests/` check these rules. They must pass before any result is reported.

## Samples
| Period | Use |
| --- | --- |
| 2015–2017 | Training only (2010–2014 is needed to build the 5-year norms) |
| 2018–2021 | Model and setting selection |
| 2022 onward | Holdout. Scored once per model version, never used for choices |

Two windows are reported both with and without them: Winter Storm Uri (Feb 2021) and the Freeport LNG outage (June 2022 onward).

## Models and success criteria

**Model 1: next week's storage change**
- Benchmarks: 5-year average, last week's change, and the public HughFOD nowcast (13.1 Bcf MAE, 2022–2026).
- Success:
  - Holdout MAE at least 40% below the 5-year average.
  - 80% ranges cover 75–85% of outcomes.

**Model 2: storage at end-March and end-October**
- Method: simulate the rest of the season through Model 1, once with each historical year's weather.
- Benchmark: today's level plus the 5-year-average path.
- Success:
  - Lower error than the benchmark at 4-, 8- and 12-week horizons.
  - Calibrated 80% ranges.

**Model 3: futures spread**
- Targets:
  - The change over 1 and 4 weeks in the March–April spread (held from August to February).
  - The contract 1 minus contract 4 spread, measured only over windows containing no roll.
- Inputs: the Model 2 end-of-season forecast, and the gap between the market spread and a fair value fitted on storage.
- Methods: random walk (the benchmark), linear regression, a convex storage-theory curve, a spline, and LightGBM.
- Success requires both of the following:
  1. Out-of-sample R² > 0 against the random walk, with a one-sided Diebold–Mariano p-value below 0.05.
  2. A sign-following spread strategy with positive holdout P&L at 2 ticks round-trip cost per leg. A cost grid from 2 to 10 ticks is also reported.

A null result on Model 3 gets reported the same way, as a finding.

## Statistics
- Confidence intervals come from a block bootstrap with 8-week blocks.
- Serial-correlation-robust (HAC) errors are used for overlapping horizons.
- Each model version and its holdout score is logged, and no version is dropped from the log.

## Change log
| Date | Change | Why |
| --- | --- | --- |
| 2026-11-17 | Model 1 v2 adds a holiday-week flag (Thanksgiving, Christmas, New Year) | Holiday weeks were the largest misses in the 2022+ holdout. **This was noticed in the holdout first.** It was then checked on 2018–2021, where 6 of 9 holiday weeks were also missed in the same direction (actual draw smaller than forecast). Both v1 and v2 holdout scores are reported. |

## Results log
| Date | Version | Holdout 2022+ MAE (Bcf) | vs 5-yr avg (32.3) | 80% range coverage | Validation 2018–21 MAE |
| --- | --- | --- | --- | --- | --- |
| 2026-11-17 | Model 1 v1, ridge | 14.7 | −54% | 81.3% | 16.95 |
| 2026-11-17 | Model 1 v1, gradient boosting | 14.9 | −54% | 87.3% | 19.21 |
| 2026-11-17 | Model 1 v2 (+holiday), ridge | 13.9 | −57% | 81.7% | 16.30 |
| 2026-11-17 | Model 1 v2 (+holiday), gradient boosting | 14.9 | −54% | 88.4% | 19.21 |
| 2026-11-18 | Model 1 v2, ridge, **own data** (EIA bulk file + own Open-Meteo pull), 253 weeks | 14.0 | −57% | 83.8% | 16.48 |
| 2026-11-18 | Model 1 v2, gradient boosting, own data | 14.9 | −54% | 86.2% | 19.40 |

Rows dated 2026-11-17 used development inputs cached in the HughFOD/gas-storage-nowcast repo. Rows dated 2026-11-18 use our own pulls: storage from the EIA bulk file (identical to the cached series in all 880 weeks), and degree days from Open-Meteo via `src/fetch.py` (correlation with the cached series 0.9997 for HDD and 0.998 for CDD over 1,142 weeks).
