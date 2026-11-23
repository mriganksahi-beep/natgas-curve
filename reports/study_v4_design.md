# Study v4 (draft): news vs premium in the gas curve

*Draft, 23 Nov 2026; data inventory confirmed the same evening. Nothing in it has yet been tested against outcomes.*

## Question
Every move in the gas curve is either **news**, a change in expected weather and therefore expected scarcity, or a **change in the risk premium**. News should persist and premium shocks should mean-revert.

Studies v2 and v3 could only split the two *after the fact*, because they used realised weather. With archived weather **forecasts**, the split can be made in real time. Three questions follow:
1. Does the unexplained part predictably reverse?
2. Does it tell a premium spike (16 Nov 2018) from a news-driven spike (24 Jan 2014) on the day?
3. Does it improve the risk/return of harvesting the March−April premium?

## Data (all free unless marked)
| Block | Source | Span |
| --- | --- | --- |
| Weather forecasts | GEFS v12 reforecast, 5 members daily (00Z), 2 m temperature at 17 cities, leads 0–15 days: **complete** (4,245 winter runs, none missing) | 2000–2019 |
| | GEFS operational (v11 to Sep 2020, then v12), same members: **complete** (1,447 winter runs) | Jan 2020 – now |
| | CPC 6–10 / 8–14-day outlooks (category and probability at 17 cities). Complete from 2015; about 70–85% of days in 2013–14 (the rest are unavailable from NOAA) | Sep 2012 – now |
| Realised weather | Open-Meteo archive, 17 cities, daily | 2000 – now |
| Prices | EIA front 4 contracts. The contract mapping was validated against CME: 97% of 2010–24 days match exactly; the rest are rounding and roll days | 1994 – Apr 2024 |
| | Databento full curve *(licensed, private)* | Jun 2010 – now |
| | Yahoo individual contracts (equal to CME settles; checked) | live |
| Storage | EIA weekly. No free pre-2010 weekly series was found, so holdout A runs without storage surprises | 2010 – now |
| Options *(licensed, private; $26.50)* | Databento daily settlements, near-the-money strikes. **ON**: monthly options on each contract, 2010-06 onward. **G4X**: 1-month calendar spread options (the March options are on March−April), winters 2013–2027 (2013–14 thin). **G3B**: 3-month spread options; **G6B**: 6-month. Underlyings verified against futures by put-call parity (median error $0.000) | 2010 / 2012 – now |

## Samples
- **Development:** winters 2010-11 to 2021-22, cut at 31 Dec 2021. This is the same cut as v2 and v3.
- **Clean holdout A, 1994–2010** (prices from the EIA front 4, forecasts from the 2000–2010 reforecast). This sample is **locked**: no agent reads pre-2010 prices during development. It contains the stressed winters 2000-01, 2002-03 (March−April went from 15¢ to $1.74), 2005-06 and 2006-07. It covers outright contracts all winter, and March−April from late November to expiry.
- **Clean holdout B: live 2026-27.** A dated amendment to `reports/live_2026_27/PROTOCOL.md` will be added before scoring.
- **2022–26:** descriptive only (already used twice).

## Timing (no look-ahead)
- The settle is at 14:30 ET on day d.
- The 00Z GEFS run of day d is public by about 06:00 ET, so it counts as known at the settle. The 12Z run of d is not.
- The CPC outlook issued on day d is published at 15:00–16:00 ET, **after** the settle, so it first counts at the d+1 settle.
- Storage reports count from the release day's settle.

## Core construction (built once by the orchestrator and shared)
- **Forecast anomaly.** For each run and lead ℓ: the gas-weighted HDD forecast minus a lead-specific model climatology, fitted on development years only. This removes model bias.
- **Window aggregation:** days 1–5, 6–10 and 11–15 ahead.
- **News** at d: the revision of the anomaly for the *same valid dates* between the run known at settle d and the run known at settle d−1, per window. The same is done for the CPC signed score.
- **Instruments:**
  - front, second
  - March−April (target winter)
  - winter strip
  - daily and weekly returns or changes, as in v2/v3

## Investigations (one agent each, then an independent review)
| # | Agent | Question | Key test |
| --- | --- | --- | --- |
| 1 | `news_channel` | How much of each curve move is forecast news? Split by lead window, scaled by tightness. Compared with realised weather (≈ 8% in v2) and with the CPC outlook | Walk-forward R²; which leads matter; stability across sub-periods |
| 2 | `private_info` | Does the unexplained part contain *better* weather information than GEFS (ECMWF, private forecasters)? | Does the residual predict realised weather beyond GEFS? If yes, part of the "premium" is news we can't see, and it should persist rather than reverse |
| 3 | `premium_shocks` | Does the forecast-unexplained move reverse while the news-explained move persists? Per instrument, at 1–20-day horizons. Classify spike episodes ex ante | Reversal slopes with HAC errors and winter bootstrap; episode table (2014, Nov 2018, 2021, …); a costed outright "fade premium shocks" rule, which gives many trades and statistical power |
| 4 | `widowmaker_v4` | Can the decomposition improve the March−April short, e.g. hold through premium spikes and cut only when the forecast turns persistently cold? | Premium kept per unit of worst drawdown vs plain short and **half size** (the benchmark nothing beat in v3) |
| 5 | `options` | What does the market charge for path risk? Black-76 implied vol of H, J and the front month (ON). Bachelier (normal) implied vol of H−J from G4X, in $/MMBtu, because spreads can be negative, vs realised spread volatility. Do premium shocks coincide with implied-vol spikes? | Spread variance risk premium by winter and lead; whether CSO implied vol prices the v3 drawdowns ex ante; the net premium of short H−J plus a long G4X call spread vs a naked short and half size. Samples: dev winters 2013–2021; 2022–26 descriptive; live March 2027 |

Agent 1 runs first, because its news definition feeds agents 2–4. Agents 2–5 run in parallel. Then the review, then frozen tests on holdout A, plus a live amendment.

## Pass criteria (to be frozen after the review, before holdout A)
- **Primary:** the reversal slope of the premium-shock component is < 0 at h = 5–10 trading days, with HAC one-sided p < 0.05, on holdout A. The news component's slope must not be negative.
- **Trading:** the outright fade rule's mean net P&L > 0 at 2 ticks per side over at least 20 trades on holdout A.
- **Widowmaker:** the decomposition rule beats half size on premium kept per unit of worst drawdown over holdout A's winters.
