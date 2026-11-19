# What's in `reports/`

All storage figures are in **Bcf** (billion cubic feet) of Lower-48 working gas, as reported by EIA. A **change** is one week's net injection (positive) or net withdrawal (negative). A **level** is the total in storage at the end of a week. Typical sizes:
- weekly changes run from about −300 to +130 Bcf
- levels run from about 1,100 Bcf (late winter) to 4,000 Bcf (late autumn)

## Live records (written by the weekly GitHub Action)

### `live_forecasts.csv` — Model 1, next week's storage change
| Column | Meaning |
| --- | --- |
| `forecast_made_utc` | When the forecast was made (UTC). The bot's commit is the public timestamp |
| `week_ending` | The Friday ending the EIA storage week being forecast |
| `forecast_change_bcf` | Forecast **change** for that week (Bcf). +7 means a 7 Bcf net injection |
| `lo80_change_bcf`, `hi80_change_bcf` | 80% range for the change, from Model 1's past out-of-sample errors |
| `five_year_avg_change_bcf` | Average change for the same calendar week over the previous 5 years (the naive benchmark) |
| `prior_week_level_bcf` | Last storage **level** EIA had reported when the forecast was made |
| `forecast_level_bcf` | Implied **level** = prior level + forecast change |
| `hdd`, `cdd` | Gas-weighted heating / cooling degree days for the week (17 US cities) |
| `weather_source` | `archive` = Open-Meteo reanalysis. `forecast-api past_days` = Open-Meteo's recent-days analysis, used because the archive lags about 5 days |
| `model` | Model version that made the forecast |
| `actual_change_bcf`, `miss_bcf` | Filled after EIA reports: actual change, and actual minus forecast |
| `note` | Anything a reader needs to know about the row |

### `live_end_of_season.csv` — Model 2, storage at the end of the season
| Column | Meaning |
| --- | --- |
| `origin_week` | Latest EIA week the forecast starts from |
| `target_week` | Last EIA week of March (end of winter) or October (end of summer) |
| `weeks_to_go` | Weeks between origin and target |
| `p10_bcf`, `median_bcf`, `p90_bcf` | 10th percentile, median and 90th percentile of the end-of-season **level** across the weather × error scenarios |
| `five_year_path_bcf` | Today's level plus the 5-year-average change for every remaining week |
| `prob_below_five_year_path` | Share of scenarios finishing below that path |
| `n_scenarios` | Number of simulated paths |

## Backtest results
| File | Contents |
| --- | --- |
| `protocol.md` | Pre-registered rules, change log and results log. **Start here** |
| `model1_holdout_v3_own.csv` | Model 1 (current) holdout scores, 2022 onward |
| `model1_validation_v3_own.csv` | Model 1 settings compared on 2018–21 |
| `model1_bootstrap_v3_own.csv` | Block-bootstrap confidence intervals for Model 1's advantage |
| `figures/` | Charts used in the README and note (`python -m src.plots`) |
| `archive/` | Results from superseded versions, kept so every logged number can be traced |

Superseded backtest outputs for Model 1 v1/v2 and Model 2 v1 are in `data/processed/archive/`.
