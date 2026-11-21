# Q2: Market-implied end-of-March storage vs Model 2 (agent `implied_storage`)

*All data used is dated on or before 31 Dec 2021. Winter 2021-22 enters only through Nov–Dec 2021 and has no realised outcome here.*

## 1. Plain-English findings
- **The curve only says something about storage when storage is short.** When the expected end-March surplus is above about −250 to +130 Bcf (depending on which winters the curve is fitted on), the March−April spread barely moves with storage. In that range it cannot be turned back into a storage level. Of 70 Model 2-era weeks, 43 can be inverted. Nearly all of 2019-20 and 2020-21 cannot, and the realised-outcome version of the mapping cannot invert any week at all.
- **When it can be read, the market sits a little below Model 2's median, and well inside its range.** The average percentile is **0.44** (95% CI 0.37–0.50), and the average gap is **−105 Bcf** (−230 to −15). The implied level never left Model 2's 10–90% band. It did not leave it even in Nov 2018, when the spread was at its sample extreme. Model 2's band is about 1,600 Bcf wide, so percentiles are a blunt yardstick.
- **Against the plain 5-year path, the market is clearly more pessimistic: −243 Bcf (−353 to −165).** Model 2's own low bias (about 140 Bcf below the 5-year path on these weeks) hides half of this. The implied level was also too low against what actually happened (bias −265 Bcf). On the same weeks, the 5-year path was the most accurate of the three (MAE 183 Bcf, against 292 for Model 2 and 339 for the market).
- **"Departures predict the spread" is mostly the spread mean-reverting.** The gap predicts the 4-week spread change (out-of-sample R² 0.39 against no change). The spread level alone gets 0.31, and once the spread level is controlled for, the gap is insignificant (t = 1.1).
- **"Who moves" is not identified.** The implied level does move back toward Model 2, but measurement noise in the inversion alone would produce that pattern (expected slope −0.55, observed −0.28). Model 2's median drifts toward the market (+0.18 per week), but the confidence interval includes zero.

## 2. Method
1. **Model 2 distribution.** I reran `Simulator().simulate(t)` for all 95 Nov–Feb origin weeks from 4 Nov 2016 to 31 Dec 2021, about 1,200 scenarios each. Percentiles 1–99 are cached in `data/processed/research_implied_storage/model2_percentiles.csv`.
2. **Weekly panel, winters 2011-12 to Dec 2021** (`data.py`). This follows the Model 3 timing:
   - The price is the settlement on the first trading day on or after the Friday after the storage week.
   - Rows stop 25 days before the March contract expires. Expiry is computed as 3 business days before 1 March, which matches the data in every year from 2011 to 2021.
   - The scarcity baseline is avg₅ = the mean end-March level of the prior years. Storage starts in Jan 2010, so this needs at least 2 prior years and the sample starts in 2011-12.
     - *x_real* = realised end-March level − avg₅ (ex-post).
     - *x_info* = storage at t + the average remaining draw from the same calendar date to end-March in the prior years − avg₅ (ex-ante, the "5-year path").
3. **Mapping**, fitted walk-forward by winter with nonlinear least squares on all earlier winters:
   y = a + b·exp(−c·x/1000), with a ∈ [−0.20, 0.10], b > 0, c > 0.
   - y is the spread, either in $/MMBtu or as a share of the April price.
   - Units were chosen on the pre-Model-2 test winters 2013-14 to 2015-16 (inversion error against the realised level). Percent of April won narrowly: 613 vs 620 Bcf.
4. **Inversion:** x̂ = −(1000/c)·ln((y−a)/b). Its delta-method standard error is s.e. = σ·1000 / (c·(y−a)).
   - A week is *informative* only if y > a and s.e. ≤ 250 Bcf. This threshold was fixed in advance, with 150 and 400 as sensitivity checks.
   - The implied level is L̂ = avg₅ + x̂, floored at 500 Bcf.
5. **Comparison with Model 2:**
   - Percentile u = F_M2(L̂), interpolated over p1–p99.
   - Gap G = L̂ − Model 2 median.
   - Who moves: over h = 1 and 4 weeks, the changes ΔL̂, ΔM (Model 2 median) and ΔG = ΔL̂ − ΔM are each regressed on G.
6. **Uncertainty:** whole-winter block bootstrap (2,000 resamples over 6 winters) and HAC t-statistics (lags = h).

## 3. Results
**Mapping and coverage.** The realised-outcome mapping is too flat. Regressing on the ex-post outcome attenuates the slope, because in November the market cannot know the outcome. With a free asymptote the fit even became almost linear (a → −2). It inverts 0 of 70 Model 2-era weeks, so the main specification is the **ex-ante x_info mapping**. Fig. `figures/implied_storage_mapping.png` shows the curve. It steepened sharply once 2018-19 entered the fit: c rose from 0.86 to 4.25 per 1,000 Bcf.

| Informative weeks (x_info) | 2016-17 | 2017-18 | 2018-19 | 2019-20 | 2020-21 | 2021-22 (to Dec) | Total |
|---|---|---|---|---|---|---|---|
| s.e. ≤ 150 Bcf | 0/12 | 0/12 | 6/13 | 3/13 | 0/12 | 3/8 | 12 |
| **s.e. ≤ 250 Bcf (main)** | 10/12 | 11/12 | 12/13 | 4/13 | 2/12 | 4/8 | **43** |
| s.e. ≤ 400 Bcf | 12 | 12 | 13 | 7 | 3 | 4 | 51 |

**(a) Level and stability of the departure.** Figure `figures/implied_storage_headline.png` shows each winter: the market-implied level, Model 2's 10–90% band, the 5-year path and the realised outcome.

| Spec | n (winters) | Mean u [95% CI] | Mean gap to M2 median, Bcf [CI] | Mean L̂ − 5-yr path, Bcf [CI] |
|---|---|---|---|---|
| s.e. ≤ 250 (main) | 43 (6) | 0.44 [0.37, 0.50] | −105 [−230, −15] | −243 [−353, −165] |
| s.e. ≤ 150 | 12 (3) | 0.33 [0.22, 0.40] | −313 [−496, −172] | — |
| s.e. ≤ 400 | 51 (6) | 0.47 [0.40, 0.52] | −63 [−179, 26] | — |
| Main, excluding the 6 floored Nov–Dec 2018 weeks | 37 (6) | 0.46 [0.37, 0.51] | −74 [−213, 9] | −211 [−332, −136] |

- **By winter**, the mean u was 0.53, 0.48, 0.41, 0.43, 0.42 and 0.25 for 2016-17 through Nov–Dec 2021. Nov–Dec 2021 is the most pessimistic stretch: gap −426 Bcf.
- **Stability:** the within-winter standard deviation of u is 0.09, and the lag-1 autocorrelation is 0.39.
- **Share of weeks outside Model 2's 10–90% band: 0 of 43.** Model 2's median 80% width on these weeks was 1,583 Bcf, while the implied level's own s.e. was about 191 Bcf.
- **Convexity alone already predicts some of this.** If the market priced E_M2[f(X)] instead of f(median), the implied level would sit at u ≈ 0.42. The departure beyond that is +0.03 [−0.03, 0.14].

**Accuracy against the realised outcome** (39 weeks, 2016-17 to 2020-21):

| Forecast | MAE (Bcf) | Bias (Bcf) |
|---|---|---|
| Market-implied | 339 | −265 |
| Model 2 median | 292 | −193 |
| 5-year path | 183 | — |

Model 2's error minus the market's error averages −47 Bcf [−134, 24], i.e. Model 2 slightly better, not significantly. The realised level sat at Model 2's 57th percentile on average.

**(b) Do departures predict the March−April spread?** In this table ΔS is in $/MMBtu and the gap G is per 100 Bcf (main spec, 43 rows).

| h | Regressor | Slope [winter CI] | HAC t | In-sample R² | Walk-forward OOS R² vs no change (33 rows, 5 winters) |
|---|---|---|---|---|---|
| 1 | Gap to Model 2 | +0.021 [0.007, 0.032] | 1.7 | 0.07 | 0.08 |
| 1 | Gap to 5-yr path | +0.018 [0.007, 0.028] | 1.9 | 0.05 | 0.07 |
| 1 | Spread level only | −0.13 | −1.4 | 0.05 | −1.36 |
| 4 | Gap to Model 2 | +0.067 [0.016, 0.092] | 3.4 | 0.28 | **0.39** |
| 4 | Gap to 5-yr path | +0.049 [0.010, 0.082] | 2.5 | 0.14 | 0.28 |
| 4 | Spread level only | −0.63 [−1.11, −0.56] | −6.3 | 0.53 | 0.31 |

- The sign makes sense: when the market looks tighter than Model 2, the spread falls.
- With both regressors at h = 4, the gap's slope drops to 0.014 (t 1.1), while the spread keeps −0.57 (t −5.3).
- No week had u outside 0.1–0.9, so the "large departure" bucket is empty.

**(c) Who moves toward whom** (main spec). Figure `figures/implied_storage_who_moves.png` shows the h = 1 case.

| h | n (winters) | ΔL̂ on G | ΔM on G | ΔG on G (HAC t) | Noise-only ΔL̂ slope |
|---|---|---|---|---|---|
| 1 | 35 (6) | −0.28 [−0.70, −0.04] | +0.18 [−0.13, 0.39] | −0.46 (−3.2) | ≈ −0.55 |
| 4 | 21 (4) | −0.77 [−1.10, 0.20] | +0.30 [−0.07, 1.44] | −1.07 (−6.6) | ≈ −0.87 |

The gap closes fast, but the market-side share is no larger than i.i.d. inversion noise (−mean(s.e.²)/var(G)) would produce. That noise benchmark is an upper bound, because spread errors are autocorrelated.

## 4. What this tells us
**(a) The futures curve.** The March−April spread is an option-like scarcity signal. It is flat and uninformative in comfortable winters and steep when storage is short. Its shape is pinned down mostly by one winter, 2018-19. A curve fitted before that winter was far too flat for it, and the implied level ran into the 500 Bcf floor. Most of the curve's "predictive" content for its own next move is ordinary spread mean reversion.

**(b) Storage expectations.** Measured against the 2011-16 mapping, the market priced end-March storage about 200–250 Bcf below the 5-year path on average in 2016-21, and was wrong on average by about that much. That gap is consistent with a scarcity, or insurance, premium. It is not separable from a shift in the curve, because any mapping fitted on past winters absorbs the average premium of those winters. Model 2's median is closer to the market than the 5-year path is, but only because Model 2 has its own low bias. There is no evidence that the market knows something Model 2 does not.

## 5. Caveats
- **Six winters, with the effective sample dominated by 2016-19.** One winter, 2018-19, shapes both the mapping and most of the large gaps. Bootstrap intervals over 6 winters understate the true uncertainty.
- **The premium level is not identified,** only its change relative to the fit window. The "pessimism" could equally be the spread-to-storage curve shifting over time.
- **Some choices were made after seeing results:**
  - I added the asymptote bound after the free fit degenerated.
  - I made x_info the main specification after x_real inverted no weeks.
  - Neither choice used ΔS or any outcome of the departure.
- **Model 2 limits the comparison.** Its median carries a known low bias and its early-winter band is very wide (it reaches below 0 Bcf in Nov 2018), so percentiles have little power.
- **Variants tried:** 2 targets × 2 units × 2 asymptote rules × 3 s.e. thresholds = 24 mapping cells, plus 4 regressors × 2 horizons in (b). The Jan−April and strip−April spreads were not tried.

## 6. Proposed holdout test (frozen)
- **Rows:** Nov–Feb storage weeks with origin ≥ 2022-01-01, using the panel rules in `data.py` and the s.e. ≤ 250 Bcf informative filter.
- **Implied level:** the x_info mapping in % of April, a ∈ [−0.2, 0.1], refit each winter on all earlier winters with a realised end-March level (from 2011-12). Then G = L̂ − Model 2 v2b median (`simulate.py` with seed 0), in units of 100 Bcf.
- **Two forecasts of ΔS over 4 weeks** ($/MMBtu), with coefficients frozen from the 2016–21 fit:
  - Gap model: ΔŜ = −0.110 + 0.0675·G
  - Spread-only model: ΔŜ = 0.013 − 0.633·S
- **Statistic:** the one-sided Diebold–Mariano test on squared errors, gap model vs spread-only model, with HAC lag 4.
- **Pass:** gap-model MSE is lower than spread-only MSE with p < 0.05, **and** gap-model OOS R² against no change is > 0.

Secondary result, reported but not pass/fail: the mean of L̂ − 5-year path on holdout rows, with its winter-bootstrap CI.

## 7. How to rerun
```
PYTHONPATH=. python -m src.research.implied_storage.simulate   # ~1.5 min, cached after the first run
PYTHONPATH=. python -m src.research.implied_storage.analysis   # ~1.5 min (bootstraps); writes results.txt
PYTHONPATH=. python -m src.research.implied_storage.figures    # ~10 s
```
Outputs go to `data/processed/research_implied_storage/` (`results.txt`, `implied_weekly_main.csv`, `who_moves_h*.csv`) and `reports/agents/figures/implied_storage_*.png`.
