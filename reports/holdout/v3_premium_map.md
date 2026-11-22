# Holdout V2: March−April drift from 9 months out to expiry (agent `premium_map`)

*Run once on 22 Nov 2026, on the re-used 2022+ holdout. The spec is exactly as frozen in `reports/agents/premium_map.md` §6, with nothing changed after the results were seen. **Weak confirmation only:** this holdout was used in v2, and the v3 thesis was formed with 2022–26 in view (see `synthesis_v3.md`).*

## Spec (frozen)
- **Pairs.** March/April 2023, 2024, 2025 and 2026 are primary. March/April 2022 is reported separately, because its entry falls in 2021.
- **Entry.** The settle on the last trading day on or before the March expiry minus 9 calendar months.
- **Target.** D = (Mar − Apr at entry) − (Mar final settle − Apr settle on the March expiry day), in $/MMBtu, and D as a % of the average leg price at entry.
- **Trade.** Short 1 spread at entry and hold to March expiry. Net = D × 10,000 − $160 (4 ticks per contract per side).
- **Pass** requires mean net > 0 **and** D > 0 in at least 3 of 4 winters.
- **Also reported (descriptive, requested):** before costs, the worst mark-to-market P&L of the short relative to entry, and the worst peak-to-trough drawdown during the hold.

## Development check (end date 2021-12-31)
`python -m src.holdout.premium_map --end 2021-12-31` covers the March 2012–2021 winters. March 2011 is lost because its entry would fall before the data starts.
- **Drift.** Mean D is **+26.3¢ (+8.2%)**; D > 0 in 9 of 10 winters, with 2013-14 at −5.8¢; mean net is +$2,472.
- **Agreement with the map.** This matches the L9 cell of the premium map: +25.9¢ over 11 years, and +25.4¢ on the same 10 winters. Single-day entry differs from the cell's Friday-bucket mean by only a few cents per year. The exception is 2014, at −5.8¢ against −11.6¢.
- **Drawdowns.** They reproduce the review: 2013-14 had a worst peak-to-trough of −$12,010, and 2018-19 −$13,520, with a worst mark-to-market of −$12,070 against entry.

## Holdout result
| Winter | Entry | S at entry | S at expiry | D | D % | Net $ | Worst MtM vs entry | Peak-to-trough |
|---|---|---|---|---|---|---|---|---|
| 2021-22 (separate) | 2021-05-24 | 0.395 | −0.073 | +0.468 | +16.6% | +4,520 | −14,050 (2021-10-05) | −14,230 |
| 2022-23 | 2022-05-24 | 2.017 | −0.097 | +2.114 | +34.4% | +20,980 | −620 | −12,580 |
| 2023-24 | 2023-05-26 | 0.288 | −0.193 | +0.481 | +14.7% | +4,650 | −580 | −780 |
| 2024-25 | 2024-05-24 | 0.245 | −0.053 | +0.298 | +9.4% | +2,820 | −560 | −1,410 |
| 2025-26 | 2025-05-23 | 0.318 | +0.101 | +0.217 | +5.3% | +2,010 | −360 | −3,670 |

(S = the March−April spread level in $/MMBtu, derived from settles. No raw outright prices are shown.)

**Primary (2023–2026):**
- Mean D is +77.8¢ (+15.9%), and D > 0 in **4 of 4** winters.
- Mean net is **+$7,615** per spread per winter, for a total of +$30,460.
- **Verdict: PASS.**

## Reading
- **2022-23 dominates the mean.** The energy crisis put the spread at about $2 at entry, and it collapsed to −$0.10. Without that winter, the other three still pass on their own: mean net +$3,160, all positive, and D of +9.8% on average. That is close to the development average of +8.2%.
- **Path risk was mild in the holdout.** Measured against entry, the worst open loss in the four primary winters was only −$620.
  - **2022-23:** the short gained heavily early and then gave back $12,580 from its peak, but never went far below entry.
  - **2021-22:** the separately reported winter carried the largest open loss of any winter seen: **−$14,050 on 5 Oct 2021**, beyond 2014's −$12,010 peak-to-trough. It still ended +$4,520. That drawdown falls inside the development period, consistent with the review's finding that the risk is the path, not the expiry.
- **No disaster winter in the holdout.** No 2014-type winter occurred in 2022–26, so the holdout confirms the carry but says nothing new about the tail.
- **The premium was wider in 2022–26** (+9–34% of price) than in development, so it has not been arbitraged away.

## Rerun
`python -m src.holdout.premium_map --holdout` takes a few seconds. Results are written to `data/processed/research_premium_map/holdout_v2_results.csv`, and the dev check to `holdout_v2_devcheck.csv`.
