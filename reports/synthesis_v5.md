# Study v5 synthesis: what natural gas options know

> **Locked 2022–26 result (run once, 24 Nov 2026; `reports/holdout/SUMMARY_v5.md`):**
> - **P PASSED.** CSO vol richness was +0.24 [0.065, 0.42].
> - **S1 failed** (+4.7, lower bound below 0).
> - **S2 failed** as expected (+$176, lower bound below 0).
> - **S3 failed** the fairness band (+2.4, CI spans 0).
> - Tight-storage cheapness, which was logged but not frozen, **reversed** to rich.

*24 Nov 2026. Based on five agent reports (`reports/agents/v5_*.md`) and an independent review (`reports/agents/review_v5.md`). Everything was developed on options data up to 31 Dec 2021. The 2022–2026 options sample stays locked until the frozen tests below have been committed.*

## The question
Option prices reveal how uncertain the market is, where its fear points, and how tightly it expects contract months to move together. Do those quantities contain tradable or forecasting information beyond the futures curve? And what do they charge for the March−April squeeze risk found in v3 and v4?

## What survived review
1. **Outright monthly gas vol is fairly priced over an option's life.** Implied minus realised vol, 10–126 trading days out, averages −0.1 vol points with a CI of [−2.7, +2.2]. Shorting all vol earns nothing.
2. **Winter vol is rich when storage is loose.** It was rich in 7 of 7 such winters; within 6 months of expiry the gap is +3.6 vol points [0.5, 7.0]. When storage is tight it is cheap, but that rests on 2014, 2018-19 and 2019 (−3.7 without them). Read together: options price a roughly constant squeeze risk, so that risk is overpriced in loose years and underpriced in tight ones.
3. **Spread-option (CSO) vol is rich relative to realised spread vol** at the median: ln(implied/realised) is +0.33 [0.25, 0.44], positive in 9 of 10 gas years. That is restricted to spread options with open interest. It has **not** been tradable after costs: a short delta-hedged CSO straddle scores −$38 per trade with a CI of [−$298, +$186]. The apparent profit came from settlements on contracts nobody held.
4. **Neither the option surface (skew, term structure, vega shocks, open interest, gamma) nor weather-forecast disagreement leads futures or realised vol.** No test survives a multiple-testing correction across about 1,100 tests. Skew follows prices, and implied vol jumps only after squeezes.
5. **Costs dominate every systematic options book tested.** Cross-sectional vol relative value earned about 0.5–0.8 vol points a month before costs and lost about 2.3 after.
6. **Translation tool:** March−April CSO prices convert into an implied distribution of end-March storage. It is useful as a quantity to set against a desk's own inventory forecast. Whether its moderate-deficit tail exceeds physical models depends on the physical benchmark chosen (+0.035, CI spanning zero, against the better-calibrated one).

## Retracted or downgraded by review
- **The weekly-option "storage-report straddle" (+$225).** Panel bug: weekly expiry was set to the last settlement date, about 1 day early in 70% of series. Correctly scored, it is +$60 in 2014–16 and −$180 in 2017–21. Retracted.
- **"Unexplained straddle P&L predicts next-day futures" (Greeks P1).** It is a weekend artefact. Theta used calendar days while the next-day IV used a trading-day clock; with a weekend dummy, t falls to 0.5. Retracted.
- **The correlation risk premium.** Filtered on open interest, it is about half the reported size (+0.018), and it sits in the tight-storage years. It is the same underlying fact as point 2.
- **The short CSO straddle (+$137).** The profit came from entries on zero-open-interest settles. Retracted.

**Panel fixes to make before any further weekly or straddle-attribution work:**
- weekly expiry from the real calendar
- a single trading-day clock for theta and IV
- r > 0 for far-dated options. At 2023–25 rates, using zero rates biases IV low by up to about 2 vol points beyond 6 months.

## Thesis after v5
Gas options price outright risk efficiently and do not forecast. Their information is about **risk prices**:
- a persistent markup on spread (decoupling) vol
- a fixed squeeze premium that is too rich when storage is loose and too cheap when it is tight

For a long-vol desk, the possibly useful condition is **storage tightness**: winter vol was underpriced in tight winters, though that rests on about 3 episodes. It is not a forecasting signal.

---

## Frozen tests: locked 2022–2026 options sample (run once)
- **Runner:** `src/research/reviewer_v5/frozen_ref.py`, sha256 `133661db…ebd04`. It reproduces the development reference values exactly; the output is identical to `data/processed/research_reviewer_v5/frozen_dev_reference.txt`.
- **Inputs:** `data/processed/v5_locked/` (built by `src/v5/options_panel.py`, sha256 `51ba9b31…d04`), futures and EIA storage.
- **Command:** `python -m src.research.reviewer_v5.frozen_ref data/processed/v5_locked 2022-01-01 2026-08-15`. Positions are opened between 1 Jan 2022 and 15 Aug 2026; outcomes resolve by 13 Nov 2026, within the futures data (ends 17 Nov 2026).

| ID | Test (runner label) | Pass if | Development reference (2010–2021) |
| --- | --- | --- | --- |
| **P** | CSO vol richness: mean ln(nv / realised spread vol over the option's life), 5–126 trading days to expiry, previous day's CSO open interest > 0, gas-year block bootstrap ("PRIMARY") | Mean > 0 **and** 95% lower bound > 0 | +0.328 [0.245, 0.440], 9 of 10 gas years positive |
| S1 | Loose-storage winter vol: Dec–Mar delivery, ≤ 126 trading days, storage > 150 Bcf above the 5-year norm; IV − realised vol to expiry ("S2 loose td<=126") | Mean > 0 **and** lower bound > 0 | +3.6 [0.5, 7.0] |
| S2 | Short delta-hedged CSO straddle, 4% cost, open interest > 0 ("S1 short CSO straddle") | Mean > 0 **and** lower bound > 0 | −$38 [−298, 186]: **expected to fail** |
| S3 | Outright fairness: all deliveries, 10–126 trading days, IV − realised vol to expiry ("S3") | "Fair" if the CI includes 0 **and** \|mean\| < 2 vol points | −0.13 [−2.7, 2.2] |

The rest are not frozen: tight-storage cheapness (episode-driven), implied-storage gaps (benchmark-dependent), weekly and Greek signals (retracted), and weather-disagreement signals (null).

## Live 2026-27 (amendment to `reports/live_2026_27/PROTOCOL.md`)
- **LV-P:** the primary statistic on the G4X Mar-27 (H27−J27) spread options, for dates from 1 Dec 2026 to option expiry (23 Feb 2027), where the previous day's open interest > 0. **Pass:** mean ln(nv/RV) > 0. This is a single winter and complements v4's L3.
- **LV-S1:** scored **only if** the EIA report released in the week of 1 Dec 2026 shows storage > 150 Bcf above the 5-year norm. It applies S1 to the Jan-27, Feb-27 and Mar-27 contracts from 1 Dec 2026 to their expiries.
- **Data:** your Databento pull after 23 Feb 2027, plus Yahoo futures.
