# From weather to the gas curve

How far up the chain **weather → weekly storage → end-of-season storage → futures spread** can free public data forecast US natural gas, out of sample?

Everything here uses public data (EIA, Open-Meteo, and later CME settlements via Databento). Every test was written down in [`reports/protocol.md`](reports/protocol.md) before its results were seen. That file also logs every change and every model version's score, including the weaker ones.

## Results so far (updated 19 Nov 2026)
**Model 1: this week's storage change, from this week's weather.** Out of sample from 2022 (253 weeks):

| Forecast | Average miss (Bcf) |
| --- | --- |
| 5-year average | 31.3 |
| Ridge regression | **13.8** |

- Ridge's advantage over the 5-year average is 17.5 Bcf per week, with a 95% block-bootstrap interval of 11.0–24.3.
- Its 80% ranges contained the actual 85% of the time.
- Gradient boosting scores 14.6 Bcf. The two are **statistically tied**, so the simpler linear model is preferred.

**Model 2: storage at the end of the season, as a probability range.** Model 1 is simulated forward through about 20 past years of weather, each combined with Model 1's own past errors.
- The 80% ranges are calibrated: they covered the actual 85–93% of the time from 2022.
- The central forecast beats the 5-year-average path only within about 8 weeks of season end. Further out, weather is unforecastable and "normal" is as good a centre as any.

**Model 3 and Study v2: what the futures curve knows about storage.** Results from 2010–2021, then a single holdout run on 2022–2026 ([summary](reports/holdout/SUMMARY.md), [synthesis](reports/synthesis_v2.md)):
- **Storage surprises measured against norms barely move the March–April spread** (holdout R² 0.02).
- **The curve moves ahead of storage reports, mostly on weather** (holdout correlation −0.21 with the report two weeks out; about 70% of it is weather).
- **The winter curve behaves as one factor.**
- **No relative-value trading rule survives the holdout:** curve kinks, the gap between implied and forecast storage, and the storage-based fair value all fail.
- **One narrow pattern recurs:** early-winter scarcity spikes that partly unwind (2018, 2021, 2022, 2026). It is now tracked live as a hypothesis. It carries real tail risk: the 2014 polar-vortex spike kept rising.

## Live record
Every Wednesday at 12:00 UTC, before EIA's release, a [GitHub Action](.github/workflows/live_forecast.yml) does three things:
1. It refreshes storage from EIA's bulk file.
2. It scores last week's forecast against what EIA reported.
3. It commits this week's forecast to [`reports/live_forecasts.csv`](reports/live_forecasts.csv), and the end-of-season forecast to [`reports/live_end_of_season.csv`](reports/live_end_of_season.csv).

The bot's commit time is the timestamp. Column meanings are in [`reports/README.md`](reports/README.md).

## Repository map
| Path | What it is |
| --- | --- |
| `src/fetch.py` | Downloads storage, prices (EIA bulk file) and weather (Open-Meteo) |
| `src/model1.py` | Model 1 features, walk-forward test and scoring |
| `src/bootstrap.py` | Confidence intervals for Model 1 |
| `src/model2.py` | Model 2 scenario simulator, backtest and live forecast |
| `src/live.py` | Weekly live forecast and scoring |
| `src/prices.py` | Contract settlement prices (Databento) for Model 3 |
| `src/plots.py` | Figures |
| `tests/` | Checks that no forecast can see data from after its date |
| `reports/` | Protocol, results and live records ([guide](reports/README.md)) |

## Reproduce
```
pip install -r requirements.txt          # add requirements-model3.txt for Model 3
python -c "from src.fetch import from_bulk; from_bulk('NG.txt')"      # EIA bulk file, no key
python -c "from src.fetch import fetch_weather, weekly_degree_days; weekly_degree_days(fetch_weather())"
python -m pytest tests          # no-look-ahead checks
python -m src.model1            # Model 1 walk-forward (current version)
python -m src.bootstrap         # Model 1 confidence intervals
python -m src.model2            # Model 2 backtest
python -m src.plots             # figures
```

![Model 1 holdout](reports/figures/model1_holdout.png)

![Model 2 fans](reports/figures/model2_fans.png)
