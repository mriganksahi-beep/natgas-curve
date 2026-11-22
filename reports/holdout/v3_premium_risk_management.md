# Holdout V7: H-PRM, the physical-trigger overlay vs half-size scaling (agent `premium_risk_management`)

*Run once on the re-used holdout (winters 2022-23 to 2025-26) on 22 Nov 2026. This test is descriptive. The spec is frozen in `reports/agents/premium_risk_management.md` §6, and nothing was changed after seeing the results.*

## Verdict: **FAIL (narrowly)**
- **The overlay kept 0.564 of the premium and cut the worst drawdown by 0.520.** Kept + cut = **1.084**, below the 1.10 threshold. The kept ≥ 0.40 condition was met.
- **The test was not inconclusive.** The base's worst drawdown was −$7,650, which exceeds the $5,000 floor, and the base mean was +$3,490.
- **The overlay beat half-size scaling by 0.08.** That is about the margin it showed in development (1.07). It lowered the tail at a cost slightly better than 1:1, but not by the pre-registered 10-point margin.
- **The verdict is knife-edge.** One alarm of 2.09 on 23 Dec 2022 delayed re-entry by a week. At ≤ 2.0 the score would have been 1.23 (see Caveats).
- **The price-only $2k stop did badly: 0.42.** It sold near the bottom of the Nov 2022 drawdown and re-entered into the next leg.

## Spec (as frozen)
- **Base.** Short 1 H−J at the settle of the first trading day on or after the first Friday on or after 1 Nov. Held to the March expiry, marked to market daily, at $80 per spread per side (4 ticks per contract).
- **Overlay.** PHYS-M2, exactly as in `scenario_risk/models.py` (h = 1, ES97.5, walk-forward map and η, Model 2 origin d−7), every Friday from 1 Nov to 15 Feb.
  - Alarm = ES ÷ $1,530.
  - Alarm > 2.0: flat at the next settle. Re-entry at the next settle after a Friday with alarm ≤ 2.0.
  - It is executed with the development code (`overlays.phys_trigger`, m = 2, re-entry).
- **Statistics.** Kept = mean overlay P&L ÷ mean base P&L. Cut = 1 − worst overlay drawdown ÷ worst base drawdown, where the drawdown is peak-to-trough within each winter.
- **Pass** if kept + cut ≥ 1.10 and kept ≥ 0.40. **Inconclusive** if the base's worst drawdown is under $5,000 or the base mean is ≤ 0.
- **Model 2 holdout percentiles** come from `research_implied_storage/model2_percentiles_full_to_2026.csv`. Its 2016–21 rows are identical to the development cache.

## Development check (end date 2021-12-31, winters 2016-17 to 2021-22)
- **ES reproduction.** The PHYS-M2 ES was recomputed with scenario_risk's own functions (`fit_maps`, `m2_dE`, `phys`). It matches the cached forecasts: n = 79/79, correlation 0.9999, median |relative difference| 1.4% (Monte Carlo noise), maximum 6.1%. The median ES is $1,526, against the frozen $1,530.
- **Rule reproduction.** The report's numbers are reproduced exactly with both the frozen and the rolling-median denominator:

  | | Base mean | Base worst DD | Overlay mean | Overlay worst DD | Kept | Cut |
  |---|---|---|---|---|---|---|
  | Development | $3,162 | −$13,270 | $1,352 | −$4,740 | 0.428 | 0.643 |

- **The development run would itself have failed the threshold** (kept + cut = 1.07). The threshold was deliberately set above the development result.

## Holdout results
**Per winter, in $ per spread, net of costs** (DD = worst peak-to-trough drawdown within the winter):

| Winter | Base P&L | Base DD | PHYS-M2 P&L | PHYS-M2 DD | PHYS-M2 exposure | $2k stop P&L | $2k stop DD | Half-size P&L | Half-size DD |
|---|---|---|---|---|---|---|---|---|---|
| 2022-23 | +9,360 | **−7,650** | +3,270 | −1,040 | 0.56 | −750 | −6,730 | +4,680 | −3,825 |
| 2023-24 | +3,790 | −780 | +3,790 | −780 | 1.00 | +3,790 | −780 | +1,895 | −390 |
| 2024-25 | +720 | −1,410 | +720 | −1,410 | 1.00 | +720 | −1,410 | +360 | −705 |
| 2025-26 | +90 | −3,670 | +90 | −3,670 | 1.00 | +460 | −3,670 | +45 | −1,835 |

**Score over the 4 winters:**

| Rule | Mean | Worst DD | Worst winter | Kept | Cut | Kept + cut |
|---|---|---|---|---|---|---|
| Base | $3,490 | −$7,650 | +$90 | 1.00 | 0 | 1.00 |
| **PHYS-M2 overlay (V7)** | $1,968 | −$3,670 | +$90 | **0.564** | **0.520** | **1.084** |
| Half-size scaling | $1,745 | −$3,825 | +$45 | 0.50 | 0.50 | 1.00 |
| $2k stop, 4-week re-entry | $1,055 | −$6,730 | −$750 | 0.30 | 0.12 | 0.42 |

## What happened
- **The only tail event in the holdout was Nov 2022.** The base was down −$5,470 by 23 Nov, and its winter drawdown reached −$7,650. It still ended at +$9,360, the richest winter in the holdout.
- **PHYS-M2 alarmed on the first available Friday, 11 Nov 2022.** The alarm was 5.3, rose to 10.1 on 25 Nov, and fell to 1.54 by 30 Dec.
  - The overlay went flat on 14 Nov at about −$800, avoiding the drawdown.
  - It re-entered on 3 Jan 2023 and so missed most of the November–December decay. That gave up about $6,100 of premium in that winter.
  - This is the same pattern as Nov 2018, Nov 2019 and Nov 2021: the alarm fires when the premium is richest.
- **2025-26 shows the overlay's blind spot.** The base's −$3,670 drawdown, now the overlay's worst, came in late January 2026. The alarm peaked at only 1.57 on 30 Jan, so the overlay never acted.
- **The alarm stayed below 2 throughout 2023-24, 2024-25 and 2025-26.** In those winters the overlay was identical to the base.
- **The $2k stop sold the dip and re-bought the rally.** In 2022-23 it exited on 22 Nov, re-entered on 20 Dec into a second drawdown, and turned +$9,360 into −$750. This repeats the development finding that settle-based stops are path-unlucky in reverting squeezes.

## Reading
- **The overlay is consistent with development, but not better than scaling by a meaningful margin.** Kept + cut was 1.07 in development and 1.08 in the holdout. Each is driven by a single November squeeze in which it was flat (2018, 2022).
- **The PHYS-M2 alarm fires reliably in rich, volatile Novembers.** Firing there limits the drawdown but sacrifices most of that winter's premium. It does not fire for late-winter drawdowns.
- **Caveats.**
  - 4 winters, with one tail event, so the "cut" statistic is effectively n = 1.
  - The holdout is re-used, and the thesis was formed with these winters in view (`review_v3`).
  - **The verdict hinges on one Friday near the threshold.** On 23 Dec 2022 the alarm was 2.09, so the overlay stayed flat for one more week.
    - Had it been ≤ 2.0, re-entry would have come on 27 Dec. The 2022-23 overlay P&L would then have been +$5,270 instead of +$3,270, giving kept 0.71, kept + cut 1.23 and a PASS.
    - This is not Monte Carlo luck. Over 200 seeds that alarm was 2.105 ± 0.028 and never ≤ 2.0. It is, though, a measure of how knife-edge the statistic is: one week of re-entry timing moves it by 0.14.
    - (This is a sensitivity only. The frozen verdict stands.)

## Rerun
```
PYTHONPATH=. python -m src.holdout.premium_risk_management --end 2021-12-31   # development check, ~1 min
PYTHONPATH=. python -m src.holdout.premium_risk_management --holdout          # the one holdout run (done)
```
- Outputs are in `data/processed/research_premium_risk_management/holdout/{dev,holdout}/`: `output.txt`, `per_winter.csv`, `score.csv` and `phys_m2_es.csv`.
- scenario_risk's loaders are repointed in memory (cutoff, cache, Model 2 file). None of their files is modified.
