# Holdout V1: does the curve forecast the next 2 weeks of weather? (`curve_weather`)

*Run once, 22 Nov 2026, on the re-used holdout. The spec, coefficients and runner were frozen before any 2022+ data was read. Raw output: `data/processed/research_curve_weather/holdout_end_2026-11-17/v1_results.json` and `v1_weekly.csv`. Run log: `data/processed/research_curve_weather/holdout_run_log.txt`.*

## Verdict: **PASS**

| Weeks | n | SS front + persistence | SS persistence | **ΔSS** | One-sided DM (HAC, 2 lags) | Winter-bootstrap 95% CI of ΔSS |
|---|---|---|---|---|---|---|
| Holdout: Nov–Feb Fridays, 2022-01-07 to 2026-02-27 | 77 | 0.129 | −0.091 | **+0.221** | t = −1.88, **p = 0.030** | +0.099 to +0.294 |
| Validation reproduction (end 2021-12-31) | 196 | 0.292 | 0.203 | +0.089 | t = −2.58, p = 0.005 | +0.032 to +0.165 |

The pass rule was ΔSS > 0 and p < 0.05, and both hold.

Secondary statistics on the holdout:

| Forecast | Correlation with realised | Sign hit rate | SS |
|---|---|---|---|
| Front + persistence | 0.45 | 65% | 0.129 |
| Persistence | 0.12 | 52% | −0.091 |
| Front month alone | 0.51 | 65% | **0.260** |

**Read with care.** Most of the gain comes from persistence collapsing, not from the combination improving:
- Persistence had a skill score of 0.20 on 2010–21 and −0.09 on the holdout.
- Front month alone scored 0.26, beating the frozen combination (0.13). The persistence term, with its frozen coefficient of 0.685, actively hurt on the holdout.

So the curve's weather signal replicated and grew stronger: correlation 0.51 against 0.35 in development. But the specific claim that the curve complements persistence did not. The frozen test compares the combination with persistence alone, so it passes. It would also have passed if it had compared front month alone with persistence.

**Additional caveats:**
- 5 winter blocks, one of them only 8 weeks (Jan–Feb 2022).
- The holdout was re-used, but this test's design never saw 2022+ data.

## Procedure
- **Runner:** `src/holdout/curve_weather.py`. It reuses `src/research/curve_weather/data.py` unchanged. Only the end date and the cache folder are re-pointed at run time, so the development cache was not touched. Coefficients come from `frozen_holdout_coefs.json`.
- **Check:** with end 2021-12-31, the panel is byte-identical to the development panel and reproduces ΔSS = +0.089, p = 0.005.
- **Holdout:** run once with end 2026-11-17 and start 2022-01-07. There were no crashes and no spec changes.
- **Sample:** Nov 2026 Fridays drop out because their 2-week-ahead weather is not yet available. The last usable Friday is 2026-02-27.

## By winter

| Winter | n | sd of wx2 (HDD) | SS front + persistence | SS persistence | SS front | ΔSS | Corr(front move, wx2) |
|---|---|---|---|---|---|---|---|
| 2021-22 (Jan–Feb) | 8 | 35 | −0.32 | 0.01 | −0.40 | **−0.33** | 0.12 |
| 2022-23 | 17 | 59 | 0.15 | −0.06 | 0.30 | +0.20 | 0.41 |
| 2023-24 | 17 | 49 | 0.41 | 0.07 | 0.42 | +0.35 | 0.51 |
| 2024-25 | 18 | 55 | −0.12 | −0.26 | 0.14 | +0.15 | 0.39 |
| 2025-26 | 17 | 68 | 0.11 | −0.16 | 0.27 | +0.26 | 0.61 |

- **ΔSS is positive in all 4 full winters.** It is negative only in the 8-week stub of 2021-22.
- **Front month alone has positive skill in every full winter.**

## Cold snaps
Rows show HDD anomalies for the next 2 weeks (wx2), the forecasts, and the weekly front-month move. No price levels are shown.

**Feb 2021, Uri** (development sample, for reference)

| Friday | Front move | a0 | Realised wx2 | Front + persistence | Persistence |
|---|---|---|---|---|---|
| 29 Jan | +4.3% | +12 | +55 | +16 | +2 |
| 5 Feb | +11.0% | +5 | **+119** | **+30** | −3 |
| 12 Feb | +1.7% | +50 | +72 | +36 | +31 |

The curve saw Uri a week before persistence did, but underpredicted it by about 4×.

**Jan 2024** (holdout)

| Friday | Front move | a0 | Realised wx2 | Front + persistence | Persistence | Front alone |
|---|---|---|---|---|---|---|
| 5 Jan | +14.0% | −15 | +48 | +24 | −18 | +38 |
| 12 Jan | +13.6% | −26 | +61 | +15 | −26 | +37 |
| 19 Jan | −27.4% | +75 | −74 | −26 | **+49** | **−91** |
| 26 Jan | −3.5% | −14 | −108 | −22 | −17 | −16 |

- The front month rose 14% in each of the two weeks before the mid-January Arctic blast. Persistence still pointed warm.
- At the peak of the cold, the curve fell 27% and correctly called the warm-up that followed. Persistence called for more cold.
- This is the clearest case of the curve trading forecasts rather than current weather.

**Jan 2025** (holdout)

| Friday | Front move | a0 | Realised wx2 | Front + persistence | Persistence | Front alone |
|---|---|---|---|---|---|---|
| 3 Jan | −0.9% | −40 | +71 | −33 | −37 | −8 |
| 10 Jan | +17.3% | +48 | **+110** | **+76** | +30 | +49 |
| 17 Jan | −1.0% | +23 | +70 | +9 | +10 | −9 |
| 24 Jan | +2.0% | +87 | −30 | +62 | +59 | +1 |

- **Missed:** the onset on 3 Jan, by every reading.
- **Caught:** the 10 Jan move was the curve's best call of the winter. The combined forecast gave +76 against +110 realised.
- **Persistence error:** on 24 Jan it lagged and pushed the combined forecast to the wrong sign.

**Also notable** (holdout, not requested)
- **Elliott, Dec 2022:**
  - On 16 Dec, ahead of +91 realised, the front month rose only 5.5% (front alone +12).
  - On 23 Dec it fell 26% at the peak of the cold and correctly anticipated the warm-up (−57 realised; persistence said +38).
- **Late Jan 2026:** on 23 Jan the front month rose 53%. Its implied anomaly was +164 HDD against **+166 realised**. On 16 Jan, ahead of +149, it was still flat.

**Pattern across events.** The curve reacts late to the onset of a cold snap (Jan 2025 and 2026), is strong once the event is in the 1–2 week forecast, and reliably prices the end of a cold snap before persistence does. The end-of-snap calls are what break persistence in the holdout.

## What this means
- The weekly front-month move carries a real 1–2-week weather forecast. Out of sample it was stronger than in development.
- The specific frozen form, which leans on persistence, was the weaker part. For the live 2026-27 log, front month alone should be reported next to the frozen combination. Reporting it is descriptive, not a spec change.
- This remains a weather-forecasting result. Development already showed that it does not translate into better storage forecasts, and V1 does not test that.
