# Brief for research agents (read fully before starting)

You are one of several parallel research agents on a natural gas research project. The repo is at `/home/claude/natgas-curve`. Read these first:
- `reports/study_v2.md` (the study you serve)
- `reports/protocol.md` (history, especially the Model 2 and Model 3 sections)

## Hard rules
1. **No holdout.** Never use, print, plot or inspect any data dated on or after 2022-01-01. That covers prices, storage, weather and Model 2 rows (use the `origin` date for Model 2). Filter as soon as you load each file. Any fit, choice, threshold or evaluation uses data ≤ 2021-12-31 only.
2. **No look-ahead.** A prediction made on date t uses only information public by t. Storage for week ending Friday F is public the following Thursday at 10:30 ET. Futures settle at about 14:30 ET.
3. **Don't modify existing files.** Write only under your own folders:
   - code: `src/research/<your_name>/`
   - figures: `reports/agents/figures/<your_name>_*.png`
   - findings: `reports/agents/<your_name>.md`
   Don't run git commit or push.
4. **Licensed data.** CME prices (from Databento) must not appear in your report as raw tables of prices. Aggregates, statistics and charts are fine.
5. **Shared machine.** There are 2 CPUs and 7 GB RAM, shared with 3 other agents. Keep any single computation under about 10 minutes, and avoid heavy loops over all contracts × all days at once (vectorise, or subsample to weekly if you must). Cache intermediate results under `data/processed/research_<your_name>/`; that folder is git-ignored.
6. **Honesty over results.** A clear null result is a good result. Report effect sizes with confidence intervals (block bootstrap by winter or month, or HAC errors). Always compare with a simple benchmark. Count how many variants you tried and say so. Don't tune on the evaluation sample. If you compare variants, choose using an earlier window (e.g. 2011–2017) and report 2018–2021 as the check.

## Data (all local)
- **`data/raw/ng_curve_daily.csv`:** CME Henry Hub outright futures, one row per contract per trading day.
  - Columns: `date, symbol, contract_year, contract_month, settle ($/MMBtu), volume, open_interest`.
  - Covers 2010-06-04 to 2026-11-17. Dates are validated against EIA (99.97% match).
  - `settle` may be NaN for far contracts.
  - Volume and open interest start in Nov 2015.
  - 1 tick = $0.001/MMBtu = $10 per contract.
- **`data/raw/storage.csv`:** EIA weekly Lower-48 working gas, `week_ending` (Friday), `storage_bcf` (level). Starts 2010.
- **`data/raw/weekly_degree_days_own.csv`:** gas-weighted HDD/CDD per EIA week (17 cities), from Dec 2004.
- **`data/raw/henry_hub_spot.csv`:** daily spot price.
- **`data/processed/model2_backtest_v2b_from2016.csv`:** Model 2 walk-forward forecasts.
  - One row per origin week (Friday), November 2016 onward.
  - `target` is the end-March (winter) or end-October (summer) week.
  - Columns: `p10, median, p90` of the end-of-season storage level (Bcf); `benchmark` (5-year-average path); `actual`; `season`; `weeks_to_go`.
  - Model 2's median runs about 250 Bcf low on average (known bias; see the protocol).
- **`data/processed/model3_weekly.csv`:** the March−April weekly dataset (see `src/model3_data.py` for definitions).
- **Code you may import:**
  - `src/model1.py` (`build_features`, `load_raw`)
  - `src/model2.py` (`Simulator`: `Simulator().simulate(t)` returns a dict that includes `levels`, an array of about 4,000 simulated end-of-season storage levels, from storage and weather known at origin week t)
  - `src/model3_data.py`
  - `src/prices.py` (`spread_panel`)
- **Not available offline:** interest rates. Either treat financing as part of an estimated carry term, or use a documented constant (e.g. 2% and 5% as a sensitivity check). The internet is blocked except PyPI; `pip install --break-system-packages <pkg>` works (e.g. statsmodels, hmmlearn).
- **Physical facts you may use:**
  - Lower-48 working gas capacity is about 4.1–4.3 Tcf; the seasonal low is about 1.1–2.3 Tcf.
  - Weekly injections reach about +130 Bcf and withdrawals about −370 Bcf.
  - Contracts stop trading 3 business days before the first day of the delivery month.

## Your report (`reports/agents/<your_name>.md`), in this order
1. **Plain-English findings:** 3–6 bullets a trader could read in one minute.
2. **Method:** what you did and why, intuitively, with the key formulas.
3. **Results:** tables with confidence intervals, plus 1–3 figures, each one referenced in the text.
4. **What this tells us** about (a) the futures curve and (b) storage expectations.
5. **Caveats,** including the number of variants tried.
6. **Proposed holdout test:** one precise, frozen specification (inputs, target, horizon, statistic, pass/fail threshold) that could be run once on 2022+ data.
7. **How to rerun:** the command, and roughly how long it takes.

Keep the report under about 1,500 words. Be precise with numbers and units.
