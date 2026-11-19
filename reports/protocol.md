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
| 2026-11-18 | Model 2 v2: analog weather re-centred on the latest 10-year normal; Model 1 misses de-meaned before resampling | v1 was biased toward too much withdrawal: actual end-of-season storage came in above the median by +34 to +358 Bcf, depending on horizon. Diagnostics showed two causes. First, the analog years (2005 on) are colder than recent winters, by +42 HDD per season in 2018–21 and +80 in 2022+. Second, the pool of misses included 2016–17 forecasts from models trained on only 1–2 years, which ran about −12 Bcf/week. Both causes are visible in validation alone, and both fixes are standard practice rather than tuned values. **v1 holdout results had already been seen**, so v2's holdout score is not a clean first look. |
| 2026-11-18 | Model 2: ablation, run on 2018–21 only, that stops simulated misses feeding the next week's lag features | No consistent gain: ranges were narrower and under-covered. Kept the feedback (v2). |
| 2026-11-18 | Model 1 v3: the 5-year average now matches the same calendar date in each prior year | v1/v2 matched t − 52k weeks, which drifts about a day per year, so the "same week" five years back was up to 6 days off. In transition months this mattered: for the week ending 13 Nov 2026 the drifted average was −25 Bcf against +27 calendar-aligned. Found while checking the first live forecast. A bug fix, not a tuning choice, so no settings were re-chosen on the holdout. It affects both the feature and the benchmark. |
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
| 2026-11-18 | **Model 1 v3 (calendar-aligned 5-yr average), ridge, own data** | **13.8** | −56% (vs 31.3) | 84.6% | 16.52 |
| 2026-11-18 | Model 1 v3, gradient boosting, own data | 14.6 | −53% | 85.8% | — |

Rows dated 2026-11-17 used development inputs cached in the HughFOD/gas-storage-nowcast repo. Rows dated 2026-11-18 use our own pulls: storage from the EIA bulk file (identical to the cached series in all 880 weeks), and degree days from Open-Meteo via `src/fetch.py` (correlation with the cached series 0.9997 for HDD and 0.998 for CDD over 1,142 weeks).

### Model 1 v2 (own data): block-bootstrap confidence intervals, holdout 2022+ (253 weeks)
The loss differential is d_t = |benchmark miss| − |model miss|. Intervals use a moving block bootstrap with 8-week blocks and 10,000 resamples. Blocks of 1, 4, 13 and 26 weeks are in `reports/archive/model1_bootstrap_v2_own.csv`, and the conclusions are unchanged across all of them.

| Comparison | Advantage (Bcf/week) | 95% CI | MAE ratio (95% CI) | Newey–West t |
| --- | --- | --- | --- | --- |
| Ridge vs 5-year average | 18.2 | 11.8 to 24.7 | 0.44 (0.37 to 0.54) | 5.2 |
| Gradient boosting vs 5-year average | 17.3 | 11.5 to 23.5 | 0.46 (0.39 to 0.55) | 5.4 |
| Ridge vs gradient boosting | 0.9 | −1.0 to 2.6 | 0.94 (0.84 to 1.07) | 0.9 |

Conclusion: both models beat the 5-year average by a wide, well-determined margin. Ridge and gradient boosting are statistically tied, so "ridge wins" is a point estimate, not an established result.

### Model 1 v3 (own data): block-bootstrap, 8-week blocks, holdout 2022+ (253 weeks)
| Comparison | Advantage (Bcf/week) | 95% CI | MAE ratio (95% CI) | Newey–West t |
| --- | --- | --- | --- | --- |
| Ridge vs 5-year average | 17.5 | 11.0 to 24.3 | 0.44 (0.36 to 0.55) | 5.0 |
| Gradient boosting vs 5-year average | 16.6 | 10.6 to 23.0 | 0.47 (0.40 to 0.57) | 5.1 |
| Ridge vs gradient boosting | 0.9 | −0.9 to 2.8 | 0.94 (0.82 to 1.07) | 0.9 |

The conclusions are unchanged from v2.

### Live record
- Run by GitHub Actions every Wednesday. The log is in `reports/live_forecasts.csv`.
- The first live forecast (week ending 2026-11-13, made 2026-11-18) used v2 with the drifted 5-year average. It is kept as logged. From the next week, the live job uses v3.

### Model 2 v2: end-of-season storage (walk-forward; every week of each season is an origin)
| Period | Weeks to go | Origins | Model 2 MAE | 5-yr path MAE | Inside 80% range | 80% width | Bias |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Validation 2018–21 | 1–4 | 32 | 64.8 | 59.5 | 81% | 198 | +7 |
| | 5–8 | 32 | 126.7 | 119.2 | 72% | 490 | +6 |
| | 9–12 | 33 | 156.5 | 145.2 | 88% | 771 | +46 |
| | 13–16 | 32 | 242.3 | 192.8 | 94% | 987 | +165 |
| | 17+ | 72 | 282.0 | 195.6 | 83% | 1,022 | +146 |
| Holdout 2022+ | 1–4 | 40 | 40.4 | 46.2 | 90% | 186 | +27 |
| | 5–8 | 40 | 77.4 | 101.0 | 90% | 455 | +40 |
| | 9–12 | 39 | 180.2 | 171.7 | 85% | 667 | +8 |
| | 13–16 | 36 | 238.8 | 195.9 | 89% | 833 | +32 |
| | 17+ | 87 | 234.7 | 210.2 | 84% | 845 | −69 |

Against the protocol's criteria:
- **Calibration: met.** Coverage is 72–94%, slightly wide on average.
- **Beating the 5-year path at 4, 8 and 12 weeks: partly met.**
  - Holdout: beats it at 1–4 and 5–8 weeks, but not at 9–12.
  - Validation: does not beat it at any horizon.
  - Beyond 8 weeks the median adds nothing over the 5-year path. Its value is the calibrated range, not a better centre.

Figure: `reports/figures/model2_fans.png`, generated by `src/plots.py`.

Known weakness: ranges made at the start of winter (19 or more weeks to go) are too wide. All 21 such forecasts in 2018–2025 contained the actual, against 80% expected.

### Live end-of-season record
- From 2026-11-19, the weekly job also logs Model 2's forecast of end-of-season storage to `reports/live_end_of_season.csv`. Each row gives the median, the 80% range, the 5-year path, and the probability of finishing below that path.
- For weeks not yet in the data, the calendar inputs are rebuilt with the same rule as Model 1. They were checked to match Model 1 exactly on three historical weeks.
- First entry, from storage through 2026-11-06: end-March 2027 median 1,968 Bcf, 80% range 1,253–2,688 Bcf, 5-year path 1,922 Bcf.

### Model 2 simulation noise (2026-11-19)
- The median is a Monte Carlo estimate. With 40 miss paths, rerunning with a different random seed moved the median by about 25 Bcf on average, and by up to 132 Bcf.
- The default is now 200 paths, which cuts the seed-to-seed change to 11 Bcf on average (95th percentile 32).
- Backtest scores are unchanged to within about 2 Bcf. Results are in `data/processed/model2_backtest_v2b_from2016.csv`, which now starts in November 2016 so that Model 3 has more winters.
- The first live end-of-season row (2026-11-19) used 40 paths and stays as logged.

---

## Model 3: storage view vs the futures curve (pre-registered 2026-11-19, before any contract-level price data was loaded)

### Question
Does the futures market's end-of-winter spread move toward what the public storage view implies? Put concretely: when the March–April spread is rich or cheap relative to a storage-based fair value, does it then move back?

### Data and timing
- **Prices:** daily CME settlement prices of named Henry Hub contracts from Databento's `statistics` schema (`src/prices.py`).
  - These are checked against EIA's contract-1 series, where the two overlap, before any use.
  - The futures roll does not affect the primary spread, because both legs are fixed named contracts.
- **Primary target:** S = March minus April settlement, same delivery year (H–J), in $/MMBtu.
- **Observation dates:** each EIA storage Thursday, using that day's settlement.
  - The storage report comes out at 10:30 ET, before the 14:30 ET settlement, so the report is public when S_t is observed.
  - The first observation in each winter is the first Thursday on or after 1 November. The last is the last Thursday at least 25 calendar days before the March contract expires, so that the 4-week horizon ends before expiry.
- **Horizons:** h = 1 and 4 weeks. Target: ΔS = S(t+h) − S(t).

### Inputs at date t (all known by that Thursday's settlement)
1. **Model 2 v2b on storage through the week just reported:**
   - the end-March median, minus the 5-year average of end-March storage (the expected surplus)
   - the 10th percentile, minus the same 5-year average (downside tightness)
2. The April settlement price. Spreads in $ scale with the price level.
3. Weeks to the March contract's expiry.
4. The spread itself, S_t. It enters only through the gap below.

### Fair value and the gap
- For each method, fit S on inputs 1–3 using only earlier winters (walk-forward by winter, refit each November), giving fair value F̂_t. The gap is G_t = S_t − F̂_t.
- **Prediction:** ΔS forecast = −b·G_t, with b fit on earlier winters.
- **Methods compared:**
  - (a) Random walk, ΔS = 0. This is the benchmark.
  - (b) Linear fair value.
  - (c) Convex fair value: S = c0 + c1·exp(−c2 · expected surplus), plus a price-level term. This is the theory-of-storage shape.
  - (d) Spline (GAM) in expected surplus.
  - (e) Gradient boosting.
  - (f) A small PyTorch MLP: 2 hidden layers of 16 units, weight decay, early stopping on the last training winter.
- Settings for (d)–(f) are chosen on validation winters only.

### Samples
| Period | Use |
| --- | --- |
| Winters 2016–17 to 2020–21 | Training and validation. Origins before 1 Jan 2022 |
| Origins on or after 1 Jan 2022 | Holdout (part of winter 2021–22, then 2022–23 to 2025–26). Scored once per method version |

There are about 15 observations per winter, roughly 150 in total. Observations overlap at h = 4, so every interval uses whole winters as bootstrap blocks and HAC errors.

### Success criteria (each method vs the random walk, holdout)
1. Out-of-sample R² for ΔS is above 0, with a one-sided Diebold–Mariano p-value below 0.05 (HAC errors, lags = h).
2. Strategy:
   - The position is −sign(G_t) in 1 spread (short March and long April when rich), held h weeks.
   - Costs are 2, 4, 6 and 10 ticks round trip per leg (1 tick = $0.001/MMBtu = $10 per contract).
   - The test is mean P&L per trade > 0 at 4 ticks, with a winter-block bootstrap 95% CI reported.
3. Both are reported for h = 1 and h = 4, with and without winter 2021–22.

A null result is reported the same way. With about 70 holdout observations, only a large effect could clear these bars, so the confidence intervals are the main output.

### Secondary target (reported separately, same procedure)
- The October–January spread (V–F), in delivery-year pairs.
- Inputs: Model 2's end-October forecast from summer origins (April–September).
