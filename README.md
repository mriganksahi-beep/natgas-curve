# From weather to the gas curve

How far up the chain **weather → weekly storage → end-of-season storage → futures spread** can free public data forecast US natural gas, out of sample?

The rules are fixed in [`reports/protocol.md`](reports/protocol.md): timing, samples, benchmarks, success criteria, and a change log.

## Status (17 Nov 2026)
- **Model 1, weekly storage change:** done, v2.
  - Out-of-sample mean absolute error is **14.0 Bcf**, against **32.2** for the 5-year average, over 253 weeks from 2022 (own data pulls).
  - The 80% ranges cover 83.8% of outcomes.
  - Ridge beats gradient boosting.
- **Model 2, end-of-season storage scenarios:** next.
- **Model 3, futures spreads:** pending price data.

## Run
```
pip install -r requirements.txt
python -c "from src.fetch import from_bulk; from_bulk('NG.txt')"   # EIA bulk file, no key
python -c "from src.fetch import fetch_weather, weekly_degree_days; weekly_degree_days(fetch_weather())"
python -m src.model1 v2
python -m pytest tests            # no-look-ahead checks
```

![Model 1 holdout](reports/figures/model1_holdout.png)
