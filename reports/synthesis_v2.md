# Study v2 synthesis: what the gas curve knows about storage

*21 Nov 2026. Based on four research analyses (`reports/agents/`) and an independent review (`reports/agents/review.md`). All evidence uses data up to 31 Dec 2021. The holdout (2022 onward) has not been touched. The tests below are frozen before it is.*

## Findings, strongest first
1. **The curve prices weather before storage confirms it.**
   - Weekly storage surprises, measured against the 5-year average, explain essentially none of the weekly moves in the March−April spread (R² 0.000, CI 0.000–0.018).
   - The weather of the next 1–2 weeks adds 8 points of R² (CI 2.6–14.3), and up to 27 when its effect scales with how wide the spread already is.
   - Moves in curve scarcity precede the storage reports for the following 1–2 weeks. About 70% of that lead disappears once the realised weather is controlled for, and in 2018–21 no lead remains beyond weather.
   - Reading: the market is forecasting storage from weather forecasts. A model built on past weather is late by construction.
   - Confirmed by two independent analyses and the review.
2. **The winter curve is one number.**
   - One factor explains 99% of the variance of the five winter spreads.
   - A Kalman-filtered "market scarcity" state is no better than the March−April spread alone. Its correlation with known storage differs by −0.08 (CI −0.24 to 0.04).
3. **The curve's month-to-month structure is internally consistent at tradable scale.**
   - **Kinks are slow to correct.** Small and medium kinks (a month out of line with its neighbours) close only 12–30% within a month.
   - **Below extreme sizes, kinks lose money after costs.**
   - **The summer carry bound mostly holds.**
   - **Autumn contango is really the rent on scarce storage space.** It tracks the storage surplus with a rank correlation of 0.97 across 9 years.
   - **The steepest scarcity pricing sits at March−April:** a median implied convenience yield of 27.5¢ per month.
4. **Public weather-based storage forecasts don't beat the simple 5-year path at long horizons.**
   - For end-March storage on the same 39 weeks, mean absolute error is 183 Bcf for the 5-year path, 292 for Model 2's median and 339 for the curve-implied level. That covers only 5 winters, with no significance test.
   - Where the market's pricing can be read as storage (43 of 70 weeks, the tight ones), it is more pessimistic than the 5-year path by about 243 Bcf, and always inside Model 2's 10–90% range.
   - The gap to Model 2 specifically (−105 Bcf) does not hold up under the review.
5. **The one recurring exception: extreme scarcity spikes partly fade, but this is a premium harvest, not an arbitrage.** Three analyses (Model 3, the extreme-kink fade, and fading spikes after an already-stressed week) all make their money in the same two episodes: Nov–Dec 2018 and Oct–Nov 2021. That is one effect seen three ways, not three confirmations.
   - **Extreme-kink fade** (2018–21, costs counted on both sides): +$2,006 per butterfly at 4 ticks. The 3 episodes outside the big two average +$164.
   - **Stressed-week spike fade:** slope −0.49 (t −3.3) pooled over 2010–21, but t −1.6 within the training winters alone. It was chosen as the best of about 32 tests, so the multiple-testing-adjusted p is about 0.2–0.4.
   - **Tail risk is real.** Selling the Jan 2014 polar-vortex spike lost about $5,500 per spread.

## What this means
- **For the futures curve:**
  - The curve looks efficient and forward-looking.
  - It holds a physically consistent shape.
  - It moves as one scarcity factor.
  - It prices weather news 1–2 weeks before storage data confirms it.
  - The only candidate mispricing is that extreme scarcity premiums overshoot after a period of stress, and that rests on two episodes.
- **For storage expectations:**
  - The market's implied end-of-winter storage is pessimistic relative to simple norms, consistent with an insurance premium. The size of the premium cannot be cleanly identified, because any mapping fitted on past winters absorbs their average premium.
  - The market's implied storage is no more accurate than the 5-year path. Neither is our weather-scenario model at long horizons.

## Frozen holdout tests (run once on 2022+ data)
The cost convention for every trading test: k ticks per contract per side, so a round trip costs 2k ticks per contract.

| ID | Test | Pass condition |
| --- | --- | --- |
| H0 | Model 3 v1 exactly as pre-registered (`protocol.md`), all five methods, h = 1 and 4 | As in the protocol |
| H0b | Model 3 v1, trading only when \|gap\| > $0.30 (declared now, after validation; never retuned) | Mean net $/trade > 0 at 4 ticks per side |
| H1 | Extreme-kink fade: `curve_consistency` spec (threshold 0.0424 log, 20-day hold) | Mean net > 0 at 4 ticks per side **and** ≥ 60% of episodes positive; inconclusive if < 3 episodes |
| H2 | Stressed-week spike fade: `regimes_weather` spec (frozen HMM, jump > $0.05, stressed the week before, 4-week hold) | Slope < 0 with one-sided HAC p < 0.05 **and** mean net > 0 at 4 ticks per side |
| H3 | Implied-storage gap vs spread-only: `implied_storage` spec | DM p < 0.05 and OOS R² > 0 |
| H4 | Kalman state vs March−April: correlation with known storage | Difference < 0 with CI upper end < 0 (expected to fail) |
| H5 | Replication: storage-surprise R² for weekly spread changes ≤ 0.05; and the correlation of weekly scarcity changes with the surprise two reports later ≤ −0.10 | Both hold |

Given the evidence above, the prior expectation is that H5 holds, H4 fails, and H0–H3 are inconclusive or weak. The holdout spans 4 winters and includes the 2022 energy crisis.
