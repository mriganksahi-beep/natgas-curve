# Physical tail risk for a short March−April spread (agent `scenario_risk`, study v3 idea 4)

*22 Nov 2026. Uses only data dated on or before 2021-12-31 (each loader asserts this). Model 2 rows are filtered on `origin`. The position is short 1 March−April spread (10,000 MMBtu per leg), so loss = rise of the spread × $10,000.*

## 1. Plain-English findings
- **Over 2016–21, the Model 2 scenario model (PHYS-M2) gives the best-calibrated tail of the models tested.** At the 1-week horizon it had 1 breach of its 99% VaR in 79 weeks, against 2 for EWMA and 4 for historical simulation and the other physical variant. Its ES backtest Z2 is 0.53, against 2.54 for EWMA (0 means correct). The cost is a larger ES, on average 1.2× EWMA's at 1 week and 2.2× at 4 weeks. The sample is 6 winters with 4 blow-up weeks, so none of these differences is significant.
- **It gave the only real advance warning in Nov 2018.** On Friday 9 Nov 2018, its 1-week ES was $3,255 and the spread then rose $6,780. That is 2.1× its ES, against 5.7× for EWMA and 2.9× for GARCH. Over 4 weeks its ES covered the $7,700 loss, which no volatility model did. The physical model's real test, Jan 2014, falls before Model 2 exists. The weather-climatology version (PHYS-WX) failed there, at 4.6× its ES.
- **Physical ES mostly re-expresses the spread level.** Its sensitivity is proportional to how far the spread sits above its floor. For ranking blow-up weeks, the spread level alone (AUC 0.75–0.80) does as well as PHYS-M2 (0.72) and better than any volatility model (0.52–0.65).
- **No ES-budget sizing beats simply holding 1 spread.** Sizing to a fixed ES budget makes the book largest in calm, low-premium weeks. Constant size earns $15,810 per average spread held over 2016–21, with a Sharpe ratio of 1.08. Physical sizing gives up 44% (PHYS-WX) to 66% (PHYS-M2) of that premium. In exchange its worst week per average spread is 31–51% smaller. Among ES-sized books, the two physical books are the best: they are the only ES-sized books positive in 2018-19, and they beat EWMA and GARCH on Sharpe.
- **None of the models is calibrated over 2013–21.** With Jan 2014 included, every model breaches its 99% VaR in 4–8% of weeks.

## 2. Method
Each Friday from 1 Nov to 15 Feb, every model forecasts the loss over the next h = 1 and 4 weeks. The forecast is a full quantile function on 1,000 levels. From it: VaR95 and VaR99 are quantiles, and ES97.5 is the mean beyond the 97.5% quantile. A forecast's window must end before the March expiry and by 2021-12-31. Prices are Friday settles. Storage and Model 2 enter at origin d−7, because the report for week d−7 comes out on Thursday d−1.

| Model | Specification (fixed before any evaluation) |
|---|---|
| HS-1y, HS-3y | Trailing 52 or 156 weekly changes (overlapping 4-week changes for h = 4) of the generic front-winter March−April spread |
| HS-win | Seasonal historical simulation: Nov–Feb weeks of the last 3 winters plus this winter to date (added as a stronger benchmark) |
| EWMA | RiskMetrics, λ = 0.94, normal, σ√h |
| GARCH-t | GARCH(1,1) with Student-t errors, refit each Friday on an expanding window; h = 4 by 4,000 simulated paths |
| **PHYS** | loss = (S − a)·(e^(−c·ΔE/1000) − 1) + η√h·z |
| HYB | EWMA on z = ΔS/(S_prev − a), then σ = (S − a)·σ_z·√h |

The terms in the PHYS formula:
- **(S − a):** the spread above its ample-storage floor, floored at $0.005.
- **a, b, c:** from the convex map S = a + b·e^(−c·x/1000), where x is expected end-March storage minus its 5-year average (the 5-year path, `x_info`). The map is fitted on earlier winters only, so the first fit is for 2013-14.
- **ΔE:** the h-week change in expected end-March storage, from one of two sources:
  - **PHYS-M2:** Model 2's scenarios (p1–p99, origin d−7), centred and scaled by √(h/weeks to go). This assumes the remaining weather uncertainty resolves evenly week by week.
  - **PHYS-WX:** ΔE = −β × (h-week HDD anomaly). The anomalies are sampled from the same calendar window (±2 weeks) in every earlier winter since 2004-05, and β is the Bcf-per-HDD slope fitted on earlier winters (1.44–1.53).
- **η:** non-weather noise, the robust s.d. (1.4826 × MAD) of weekly ΔS in earlier winters ($0.017–0.046).
- **The intuition:** a weather shock moves expected storage, and the convex map turns it into a spread move. The move is large only when the spread is already far above its floor.
- **PHYS-M2-pure:** PHYS-M2 with η = 0, as an ablation.

HYB is EWMA scaled by the same physical sensitivity, normal.

Evaluation:
- **Coverage:** breach rates with Kupiec and Christoffersen tests.
- **ES backtest:** Acerbi–Szekely Z2 = mean(L·1{L > VaR97.5} / (0.025·ES)) − 1.
- **Scoring:** CRPS (2 × mean pinball loss over the grid) and tail pinball (levels ≥ 0.90). Each model is compared with EWMA using HAC t-statistics and a 1,000-draw bootstrap that resamples whole winters.
- **Blow-up AUC:** blow-ups are defined in advance as a 1-week loss above $1,000 or a 4-week loss above $2,000. The forecast ES is the ranking score, benchmarked against the spread level.
- **Samples:** every model is compared on identical weeks within each sample:
  - **M2:** Nov 2016 – Dec 2021, all models; 79 weekly forecasts at h = 1 and 63 at h = 4, 6 winters.
  - **WX:** Nov 2013 – Dec 2021; 131 / 106 forecasts, 9 winters.
  - **LONG:** Nov 2012 – Dec 2021, (a)–(c) only.

## 3. Results
**Table 1. M2 sample, h = 1 (n = 79; blow-ups: 4 weeks in 4 winters).** CRPS Δ is relative to EWMA (negative is better), with its winter-bootstrap CI in $ per spread.

| Model | Breaches 95% | Breaches 99% (n) | Kupiec p (99%) | Z2 [CI] | Mean ES $ | CRPS Δ vs EWMA | Blow-up AUC [CI] |
|---|---|---|---|---|---|---|---|
| HS-1y | 8.9% | 5.1% (4) | 0.01 | 4.3 [0.0, 9.6] | 1,644 | +0.4% | 0.57 |
| HS-3y | 10.1% | 5.1% (4) | 0.01 | 4.5 | 2,123 | +1.4% | 0.44 |
| HS-win | 6.3% | 1.3% (1) | 0.82 | 2.8 | 3,265 | +1.8% | 0.40 |
| EWMA | 3.8% | 2.5% (2) | 0.25 | 2.5 [−1, 8.1] | 2,091 | — | 0.52 |
| GARCH-t | 6.3% | 2.5% (2) | 0.25 | 1.0 [−1, 3.7] | 2,985 | −0.4% | 0.64 |
| HYB | 8.9% | 5.1% (4) | 0.01 | 3.9 [0.6, 7.6] | 1,668 | **−4.7%** [−73, −2] | 0.75 |
| PHYS-WX | 7.6% | 5.1% (4) | 0.01 | 3.4 | 1,405 | +0.2% | 0.70 |
| **PHYS-M2** | **2.5%** | **1.3% (1)** | 0.82 | **0.5** [−1, 3.5] | 2,506 | −1.4% [−38, 11] | 0.72 [0.55, 0.87] |
| PHYS-M2-pure | 15.2% | 8.9% (7) | <0.001 | 10.1 | 2,043 | −1.6% | 0.72 |
| Spread level | | | | | | | **0.75** [0.59, 0.89] |

The Christoffersen independence test at 99% gives p > 0.5 for every model except PHYS-M2-pure (p = 0.009).

**h = 4 (M2, n = 63, overlapping windows, so the tests are indicative only):**
- **PHYS-M2:** 95% breaches 1.6%, no 99% breaches, Z2 −0.41 (conservative). Its mean ES is $8,599, against $3,997 for EWMA and $6,270 for GARCH-t (99% breaches 1.6%, Z2 0.01).
- **CRPS:** PHYS-M2 is +3.2% vs EWMA (not significant). PHYS-WX is +4.2% with CI [+18, +118] $, significantly worse.
- **Blow-up AUC:** rests on 2 events, so it is not informative.

**WX sample (2013–21, includes Jan 2014), h = 1:** 99% breaches are:

| Model | 99% breaches | Z2 |
|---|---|---|
| GARCH-t | 3.8% | 2.3 |
| EWMA | 6.1% | 4.9 |
| HYB | 7.6% | 5.8 |
| PHYS-WX | 7.6% | 7.1 |
| HS variants | 5.3–8.4% | — |

HYB has the best CRPS (−6.5% vs EWMA, CI excludes 0), and PHYS-WX is no better than EWMA (+1.4%). In the LONG sample, GARCH-t is the best volatility model (99% breaches 3.4%, Z2 1.9).

**Early warning** (`figures/scenario_risk_episodes.png`). The jump week is the largest 1-week rise in each episode month. The table gives the loss divided by the ES each model forecast the Friday before the jump.

| Episode (Friday) | Loss 1w / 4w | HS-3y | EWMA | GARCH-t | HYB | PHYS-WX | PHYS-M2 |
|---|---|---|---|---|---|---|---|
| Jan 2014 (17 Jan) | $3,990 / $4,590 | 4.2 / 2.2 | 3.4 / 1.9 | **2.3 / 1.2** | 3.0 / 1.7 | 4.6 / 2.0 | n/a |
| Nov 2018 (9 Nov) | $6,780 / $7,700 | 5.9 / 5.4 | 5.7 / 3.2 | 2.9 / 1.6 | 3.8 / 2.2 | 4.0 / 1.2 | **2.1 / 0.93** |
| Nov 2021 (12 Nov) | $1,970 / −$4,690 | 0.8 | 0.5 | 0.2 | 0.4 | 0.6 | 0.2 |

- **The alarm ratio** is the ES that Friday divided by the model's median earlier ES. In Nov 2018 it was 2.2 for PHYS-M2, 2.1 for GARCH, 1.3 for EWMA and 0.5 for HS-win.
- **In Jan 2014 PHYS-WX barely moved (1.2).** Its map, fitted on 2 loose winters, had the floor a pinned at −0.2, which flattens the convexity.
- **The Oct 2021 surge lies outside the Nov–Feb window.** By Nov 2021 every model was already alarmed (PHYS-M2 ES $10,800, 7×), and the spread then fell.

`figures/scenario_risk_es_paths.png` shows PHYS-M2's ES rising with the spread level, and GARCH reacting after the jump.

**Sizing** (`figures/scenario_risk_sizing.png`). Each Friday the book is short N = $5,000 / ES(1w) spreads, at 4 ticks per contract per side on |ΔN| (2 legs), with entry and exit each winter. M2 sample, values per average spread held:

| Book | P&L | Worst week | Sharpe | Winters positive |
|---|---|---|---|---|
| Constant 1 spread | **+15,810** | −6,940 | **1.08** | 6/6 |
| PHYS-WX | +8,788 | −4,777 | 0.92 | 6/6 |
| PHYS-M2 | +5,418 | −3,413 | 0.74 | 5/6 |
| HYB | +3,236 | −2,105 | 0.68 | 5/6 |
| HS-3y | +10,286 | −11,012 | 0.57 | 5/6 |
| GARCH-t | +3,755 | −4,771 | 0.43 | 4/6 |
| EWMA | +3,146 | −8,938 | 0.23 | 5/6 |

- **2018-19:** PHYS-WX made +$6,071, PHYS-M2 +$3,260 and constant size +$4,000. EWMA lost −$15,876 and GARCH −$5,894.
- **Costs:** at 0 ticks the ranking holds (Sharpe: constant 1.16, HYB 1.12, PHYS-WX 1.09, PHYS-M2 1.00). At 10 ticks, physical books keep 0.35–0.68 while EWMA and GARCH fall to about 0.1.
- **WX sample:** every ES-sized book loses. PHYS-WX alone lost −$60k in 2013-14 because its low ES in Dec 2013 made it big into the polar vortex. Constant size made +$13,890.
- **4-week non-overlapping holds:** the same ordering holds.

**Model 2 diagnostic** (ex post, not used in any forecast). The realised s.d. of weekly changes in Model 2's mean is 96 Bcf, against 131 implied by √(1/weeks to go). That is a ratio of 0.73 at 1 week and 0.75 at 4 weeks, so PHYS-M2 overstates weather risk by about a third.

## 4. What this tells us
**(a) The futures curve.**
- The March−April spread's risk is level-dependent: the further it sits above its floor, the larger its moves.
- This one fact explains most of what the physical model adds. HYB, which is just EWMA scaled by (S − a), wins on CRPS, and the raw spread level ranks blow-ups best.
- Volatility models fitted on the year-round generic series under-forecast early-winter jumps. They react only after the jump (GARCH's ES spikes the week after 9 Nov 2018).
- Because premium and risk both scale with the level, inverse-ES sizing removes the premium along with the risk.

**(b) Storage expectations.**
- Uncertainty about end-March storage, from either source, translates into spread risk through the convex map. Weather alone is not enough: with η = 0, the 99% breach rate is 9%. Non-weather repricing (risk premium, positioning, the curve's own anticipation of weather) is a material part of spread risk.
- Model 2's scenario spread is about a third too wide for week-to-week updating.

## 5. Caveats
- **Small sample.** The Model 2 era is 6 winters with 4 blow-up weeks at 1 week and 2 at 4 weeks. Almost all of the PHYS-M2 advantage comes from Nov 2018, and no difference in Table 1 is significant except HYB's CRPS gain.
- **4-week windows overlap,** so Kupiec at h = 4 is optimistic.
- **The map is fragile.** The floor a sits at its bound (−0.2) for 2014–17, and the curvature c jumps from about 1.6–1.9 to 3.7 from 2020.
- **Jan 2014 is untestable for PHYS-M2.**
- **Variants:**
  - 9 risk models: 8 planned plus 1 ablation.
  - 2 horizons and 3 samples.
  - Sizing at 3 cost levels and 2 holding schemes.
  - 1 blow-up threshold per horizon.
  - No parameter was tuned on any evaluation sample. The spread-level ranker was added after seeing the coverage results, before computing any AUC.

## 6. Proposed holdout test (frozen)
**H-SR**, run once on 2022+ (re-used holdout) and live in winter 2026-27.
- **Model:** PHYS-M2 exactly as in `src/research/scenario_risk/models.py`. The map and η are refit walk-forward on earlier winters, with no rescaling of Model 2. EWMA (λ = 0.94) is the benchmark.
- **Forecasts:** every Friday 1 Nov – 15 Feb, h = 1 week.
- **Target:** the 1-week rise in the front-winter March−April spread.
- **Pass requires both of the following:**
  1. PHYS-M2's 99% VaR breach rate has Kupiec p ≥ 0.05.
  2. Its Acerbi–Szekely Z2 is lower than EWMA's on the same weeks.
- **Secondary, reported but not part of pass/fail:** blow-up AUC (loss > $1,000) for PHYS-M2 against the spread level, and the Sharpe ratio of the PHYS-M2-sized book against constant size at 4 ticks per side.
- **Expectation:** (1) and (2) pass, the AUC ties, and constant size wins.

## 7. How to rerun
```
PYTHONPATH=. python -m src.research.scenario_risk.models     # forecasts, ~15 s (cached)
PYTHONPATH=. python -m src.research.scenario_risk.evaluate   # backtests + sizing, ~2 min
PYTHONPATH=. python -m src.research.scenario_risk.extra      # blow-up AUC, cost sensitivity, ~3 min
PYTHONPATH=. python -m src.research.scenario_risk.figures    # figures, ~5 s
```
Caches are in `data/processed/research_scenario_risk/`. Delete `weekly_panel.pkl` and `forecasts.pkl` to rebuild them. GARCH needs `pip install arch`.

## 8. New avenues
1. **Stop rule instead of ES budget.** Hold 1 spread, and cut to 0 only when PHYS-M2's ES alarm ratio exceeds 2× its median. *Expected value: high.* It keeps the constant-size premium and targets exactly the 2018-type week where physical ES led.
2. **Forecast-weather ΔE.** Replace climatology or Model 2 dispersion with the spread of 1–2-week weather forecast ensembles, which the market is already pricing (synthesis finding 1). Also freeze the 0.75 dispersion scale found here. *Expected value: medium;* it should sharpen ES without losing coverage.
3. **A physically constrained map before 2016.** Fix c from storage theory or pool it across winters, so that PHYS-WX is not crippled by the bound on a in 2014, and test it on 2010–15. *Expected value: medium;* Jan 2014 is the only other true blow-up.
4. **HYB centre with PHYS tails.** Blend the best-CRPS model (HYB) with PHYS-M2's tail quantiles above 0.95. *Expected value: medium-low;* it might win both scores, but n is tiny.
