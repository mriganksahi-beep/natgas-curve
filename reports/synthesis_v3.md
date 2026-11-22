# Study v3 synthesis: the widowmaker as insurance

*22 Nov 2026. Based on seven analyses (`reports/agents/`) and an independent review (`reports/agents/review_v3.md`). All evidence uses data up to 31 Dec 2021. The re-used 2022+ holdout is untouched by v3 until the tests below have run.*

**One disclosure first.** The motivating fact in `study_v3.md` (that the November spread exceeded the expiry spread in 14 of 16 winters) was computed using winters up to 2026. The v3 thesis was therefore formed with the holdout in view. The holdout result for the premium is weak confirmation only. The clean tests are live in 2026–27 and on pre-2010 winters, if that data can be obtained.

## What survived review
1. **The March−April spread carries a large premium that decays into expiry.**
   - It sits on average **+25.9¢ above its expiry value 9 months out** (95% CI 16.0–34.6) and fell in 10 of 11 winters. 2014 is the exception.
   - Shorting it in early November and holding to expiry earned **+$1,155 to +$1,448 per spread per winter**, depending on the entry day, net of costs.
   - About 60% of the decay happens in December–January.
   - Elsewhere on the curve, premia are small and unstable. A diversified "harvest the curve" portfolio lost money out of sample.
2. **The risk is the path, not the expiry.** The worst expiry loss was −$3,140 (2014), but mark-to-market drawdowns reached **−$12,010 (2014) and −$13,270 (2018–19)**.
   - Measured against physical storage outcomes, the expiry payout is worth about $0. Its actuarial value is about −$0.02, essentially the average past expiry spread.
   - Breaking even on the November premium at expiry alone would require roughly a 49% chance of a 2014-size outcome.
   - **The market is pricing the winter's mark-to-market risk, not the expiry outcome.**
3. **No physical model beats the spread itself out of sample.** The following all fail to add predictive power beyond the spread level:
   - Model 2's tail probability
   - an analog weather-storage tail
   - physics-implied fair values
   - scenario risk models
4. **No risk overlay demonstrably beats trading a smaller size.** Stops, volatility triggers, physics triggers, staggered entry and hedges were all tried: 123 variants.
   - Two disaster types need opposite defences. 2014 was persistent scarcity (stops and hedges help). 2018 was a squeeze that reverted (only being flat beforehand helps).
   - Options (an H−J call spread) are the natural protection, but untested without options data.
5. **The curve contains a modest 2-week weather forecast.**
   - Front month plus persistence: skill 0.21 against 0.12 for persistence alone, a gain of +0.10 (CI +0.03 to +0.19). This held in both sub-periods and in 9 of 10 winters.
   - It does not help storage forecasting.

## What did not survive review
- **"The premium is priced on fear (recent volatility), not physics":** mostly mechanical. The volatility measure correlates 0.89 with the spread level. With level-neutral measures, "fear explains 80%" falls to 12–43%, and the fear/physics split reverses.
- **The "spread minus physics-implied ≥ $0.10" rule:** chosen after seeing 142 check cells. Outside 2018–19 and the cut-off 2021 it earned +$117 per trade. The adjusted p is about 0.25.
- **Scenario-risk calibration superiority:** too few weeks to tell the models apart. Its Nov 2018 warning equals a spread-level-only alarm.
- **The autumn storage-space premium:** driven by 2020 alone.

## The emerging thesis
The widowmaker is a **path-risk premium**. Physical scarcity is fairly priced on average, and the expiry payout is worth about zero. Yet buyers pay about 26¢ to hold winter exposure, and sellers earn it only by surviving drawdowns of $12–13k per spread. Neither the physical models (scarcity) nor the volatility models predict *which* winter delivers the drawdown. That is the open question, and a reason to bring an informed view to a desk like Statar's.

## Frozen tests

### Re-used holdout (2022+), run once
Results are reported as weak or descriptive, per the review.

| ID | Test | Role |
| --- | --- | --- |
| V1 | Weather skill of the curve (`curve_weather` spec: frozen coefficients, gain over persistence, one-sided DM) | **Primary.** The design never saw 2022+ |
| V2 | March−April drift from 9 months out to expiry (`premium_map` spec) | Weak confirmation |
| V3 | Autumn Nov−Oct contango drift, price only (`storage_space_option` spec) | Weak confirmation |
| V4 | Widowmaker W1 (premium > $0.15) **vs its price-only twin (spread > $0.17)**; Model 2 adds value only if W1 beats the twin by more than $500/trade | Descriptive |
| V5 | Scenario-risk H-SR, adding the spread-level and HYB comparators | Descriptive |
| V6 | Fear-vs-physics H-FvP | Descriptive (the minimum detectable effect exceeds the effect) |
| V7 | Overlay H-PRM (physical trigger) vs half-size scaling | Descriptive |

### Live, winter 2026–27 (rules frozen before 1 Dec 2026)
- **March−April 2027 drift:** the mark-to-market path, worst drawdown, and expiry value.
- **Weather-skill test:** the curve-implied 2-week HDD anomaly vs persistence, logged weekly.
- **Alarm log:** the PHYS-M2 alarm vs the spread-level alarm, logged weekly.
