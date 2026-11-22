# Study v3: results on the re-used holdout (run once, 22 Nov 2026)

The frozen tests are in `reports/synthesis_v3.md`. The 2022–2026 holdout was already used once in v2, and the v3 thesis was formed with it in view. **Only V1 is a clean test:** its design never saw 2022+ data. The other results are weak confirmation or descriptive. Every runner was first checked against its own development numbers. Details are in the `v3_*.md` files.

| ID | Test | Result | Verdict |
| --- | --- | --- | --- |
| **V1** | The curve as a 2-week weather forecaster | Skill gain over persistence +0.22 (one-sided DM p = 0.03), positive in all 4 full winters. Front month alone: skill 0.26, correlation 0.51 with realised weather. Persistence collapsed (−0.09), so the "complements persistence" part did not hold | **Pass** (primary) |
| V2 | March−April drift from 9 months out to expiry | It fell in 4 of 4 winters, mean net +$7,615 per spread. Without 2022–23: +$3,160, all positive. Worst open loss −$14,050 (Oct 2021), yet the trade still finished +$4,520 | Pass (weak confirmation) |
| V3 | Autumn contango drift (prices only) | Contango fell in 3 of 5 years (rule needs ≥ 4); mean +$1,043 per year; no tail year | Inconclusive (right direction) |
| V4 | Weather-insurance rule vs its price-only twin | The premium replicated (decay slope 1.04), but Model 2 trailed the price-only rule by $662 per trade (+$500 needed). Nov–Dec 2022 dominates | Fail (Model 2 adds nothing) |
| V5 | Physical scenario risk model | 99% VaR breaches 3 of 64 (Kupiec p = 0.031). ES better than EWMA (1.37 vs 2.36), but the CI includes zero. GARCH-t is as good. Sizing by it gives up the premium | Fail |
| V6 | Fear vs physics | Profitable but no better than the plain price rule. "The physics part is retained" did not replicate | Descriptive; no added value |
| V7 | Physical-trigger risk overlay | Kept + cut = 1.084, against 1.10 required; one week of re-entry timing decides it | Narrow fail |

## What we now know
1. **The March−April winter premium is real and persistent.** It fell into expiry in 10 of 11 development winters and 4 of 4 holdout winters.
   - Against physical storage outcomes, the expiry payout is worth about $0.
   - The risk is the path: open losses of $12,000–14,000 per spread in 2014, 2018 and Oct 2021, even in winters that ended profitable.
2. **Physical storage models do not improve on the spread itself.** That holds for trading, timing and risk alike. The market already prices storage; public weather-based models add nothing.
3. **The curve itself is a weather forecaster.** The front month's weekly move predicts the next two weeks' weather better than persistence, and it got better out of sample (skill 0.26). The curve tends to price the *end* of a cold snap before the weather turns.
