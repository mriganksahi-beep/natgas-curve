# Squeeze-premium monitor: weekly readings, winter 2026–27

This page records the monitor described in section 5 of the [paper](../../README.md). Each row is one reading, taken from that day's settlement prices of the March–April 2027 spread options.

| Date | Movement implied by options (per year) | Movement in the past 20 trading days (per year) | Options: chance of a 400+ Bcf shortfall | Physical model: same chance | Options: chance March settles 25¢+ above April |
| --- | --- | --- | --- | --- | --- |
| 2026-11-06 | $0.45 | $0.10 | 12% | 28% | 4% |
| 2026-11-13 | $0.47 | $0.09 | 13% | 11% | 4% |
| 2026-11-17 | $0.47 | $0.10 | 12% | 9% | 3% |

**How to read the table.**
- The second and third columns compare the price of protection with the movement it protects against. In mid-November of past winters, the implied movement exceeded the movement that followed in 10 of 13 cases.
- The fourth and fifth columns compare the market's odds of a large shortfall at the end of March with the odds given by a model of storage and weather. A shortfall means end-March storage falling 400 billion cubic feet or more below its five-year average.
- The options' odds failed their pre-registered test on 2022–2026 ([result](../tool/GATE_RESULT.md)). They show what the market is pricing, not a forecast of storage.

**Updates.** The option prices are licensed and are collected in a separate step, so new readings are added when fresh data are collected, rather than automatically. The weekly storage forecast is fully automated and is recorded in [`reports/live_forecasts.csv`](../live_forecasts.csv).
