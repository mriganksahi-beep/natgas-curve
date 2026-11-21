# Holdout results (run once, 21 Nov 2026)

The frozen tests come from `reports/synthesis_v2.md`. The holdout covers 1 Jan 2022 to mid-Nov 2026: four full winters plus Jan–Feb 2022, including the 2022 energy crisis. Every runner was first checked against its own validation numbers, run with an end date of 31 Dec 2021, and then run once on the holdout. No specification was changed after the holdout results were seen. The details for each test are in this folder.

| ID | Test | Result | Verdict |
| --- | --- | --- | --- |
| H0 | Model 3 v1 as pre-registered | No method beats "no change" significantly at either horizon. Best DM p: 0.15 (gbm, h = 4) and 0.24 (gbm, h = 1). At 4 weeks every method has a positive OOS R² (0.19–0.46) and positive net $/trade (+$283 to +$476 at 4 ticks per side), but the CIs include zero. At 1 week, net $/trade is negative | **Fail** (criterion 1) |
| H0b | Model 3, only when \|gap\| > $0.30 | Convex, spline, gbm and mlp each traded only in Nov 2022: 2 trades, +$6,725 net at 4 weeks. Linear made 7 trades over 2 winters, +$2,197 (CI +$386 to +$6,725) | **Pass on paper, effectively 1 episode** |
| H1 | Fade extreme curve kinks | −$54 per butterfly at 4 ticks per side (CI −$571 to +$508); 3 of 7 episodes positive. Only 2022 paid | **Fail** |
| H2 | Fade spikes after a stressed week | Slope −0.455, one-sided p = 0.036 (n = 12), but the winter-bootstrap CI includes zero. 2 trades, +$3,360 mean net. 10 of the 12 stressed weeks are Nov 2022 – Jan 2023 | **Pass on paper, effectively 1 episode** |
| H3 | Implied-vs-forecast storage gap predicts the spread | DM p = 0.92. The gap model loses to spread-only. Only 9 readable weeks. The test was already failing on validation with the frozen coefficients, which was missed when it was proposed | **Fail** |
| H4 | Kalman state beats March−April | Difference −0.017 (CI −0.23 to +0.13) | **Fail** (as expected) |
| H5a | Storage surprises explain weekly spread moves | R² 0.023 (≤ 0.05) | **Pass** (as expected) |
| H5b | Curve scarcity leads the storage report two weeks out | Correlation −0.21 (CI −0.37 to −0.02); about 69% of it explained by realised weather (71% in validation) | **Pass** (as expected) |

## What survives
1. **The descriptive findings replicate out of sample:**
   - Weekly storage surprises measured against norms explain almost none of the spread's moves.
   - The curve moves ahead of the storage reports, mostly on weather.
   - The winter curve behaves as a single factor, so a filtered state adds nothing.

   The market prices storage from weather forecasts before the data confirms it.
2. **No systematic relative-value trade survives:**
   - Curve kinks, the implied-storage gap and the Kalman signal all fail.
   - Model 3's full-sample forecasts point the right way at 4 weeks but are not significant.
3. **One narrow pattern recurs:** early-winter scarcity spikes in the March−April spread, which partly unwind over the following weeks.
   - It appeared in Nov–Dec 2018 and Nov 2021 (validation), and in Nov 2022 and late Jan 2026 (holdout).
   - Four episodes is evidence for a hypothesis, not proof.
   - Selling the Jan 2014 polar-vortex spike lost about $5,500 per spread, so this is a premium with tail risk, not an arbitrage.
   - It is now a declared hypothesis, to be followed live from winter 2026–27.
