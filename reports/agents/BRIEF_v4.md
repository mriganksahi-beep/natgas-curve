# Brief for study v4 research agents (read fully before starting)

You are a research agent on a natural gas futures project. The repo is at `/home/claude/natgas-curve`. Read first:
- `reports/study_v4_design.md`: the study you serve.
- `reports/synthesis_v2.md` and `reports/synthesis_v3.md`: what we already know. In short:
  - the curve prices weather 1–2 weeks before storage data confirms it
  - the March−April ("H−J") spread carries a premium of about 26¢ that decays into expiry, with mark-to-market drawdowns of $12–14k per spread (2014, 2018-19)
  - physical storage models add nothing beyond the spread itself
  - front-month moves forecast 2-week weather better than persistence

## The question of study v4
A curve move is **news** (a change in expected weather, so in expected scarcity) or a **premium shock**. Archived weather forecasts (GEFS, CPC) now let us split the two on the day.
- Does the forecast-unexplained part reverse while the news part persists?
- Does the split separate a premium spike (16 Nov 2018) from a news spike (24 Jan 2014) ex ante?
- Does it improve the March−April trade or its risk?
- What do options charge for this risk?

## Hard rules
1. **Development data only: dates 2010-06-07 to 2021-12-31.** Load `data/processed/v4/panel_dev.csv`, then **immediately** filter to `date <= 2021-12-31`. Never print, plot, fit or evaluate any row after that date. The same cut applies to every other file you read: options, weather, storage.
2. **Never open `data/processed/v4_locked/`** and never read `data/raw/eia_futures_c1_c4.csv` rows before 2010-06. These are the final exam (2000–2010). Do not compute anything on pre-2010 data from any source, except the forecasts' own climatology, which is already built in.
3. **No look-ahead.** The panel encodes timing: the GEFS 00Z run of day d is known at the d settle; CPC enters with a one-day lag; storage enters on its release day. Anything you fit (betas, thresholds) must be **walk-forward** (expanding window, refitted at most monthly), or leave-one-winter-out if you are only *describing* something. Say which.
4. **Choose on 2010-11 to 2016-17; check on 2017-18 to 2021 (Dec).** Report both windows. Count the variants you tried.
5. **Licensed data.** CME futures and options prices must not appear as raw tables. Aggregates, statistics and charts are fine. Don't modify files outside your own folders:
   - code: `src/research/v4_<name>/`
   - figures: `reports/agents/figures/v4_<name>_*.png`
   - report: `reports/agents/v4_<name>.md`
   - cache: `data/processed/research_v4_<name>/` (git-ignored)
   Don't run git commit or push.
6. **Shared machine.** 2 CPUs and about 7 GB RAM, shared with other agents. Keep any single step under about 10 minutes.
7. **Honesty over results.** A clear null is a good result. Give effect sizes with CIs: winter-block bootstrap, or HAC errors with lags at least the overlap. Always compare with a simple benchmark. A result that rests on 1–2 episodes must be labelled as such.

## Data
**`data/processed/v4/panel_dev.csv`** has one row per trading day; the column definitions are in the `src/v4/panel.py` docstring. Key columns:
- **Prices:**
  - `r_front`, `r_second`: % log changes on the same contract since the previous trading day
  - `hj`, `d_hj`: the target-winter March−April spread and its change ($/MMBtu); `hj_days_left`
  - `r_strip`; `px_front`; `front_days_left`
- **Forecast news (HDD-anomaly units; positive = colder = more demand), from the GEFS 5-member mean:**
  - `news_w0`: surprise on the days that just passed
  - `news_w1`, `news_w2`, `news_w3`: revisions to days 1–5, 6–10 and 11–15 ahead
  - `news_tot`
  - `_c00` versions use the control run
  - `lvl_w*`: forecast levels
  - `real_w*`: the **realised** anomalies over those windows. These are outcomes, never predictors.
- **CPC:** `cpc610`, `cpc814` (gas-weighted signed tercile probabilities, positive = cold, known at d) and their changes `d_cpc610`, `d_cpc814`. Available from late 2012.
- **Options:** `iv_front`, `iv_h`, `iv_j` (Black-76 ATM, annualised, from ON settles); `nv_hj` (Bachelier normal vol of the March−April spread, $/MMBtu/√yr, from G4X calendar spread options; winter 2013 onward, thin in 2013–14); `T_h`.
- **Storage:** `storage_surprise_5y` on release days (from 2015).
- **System flag:** `gefs_system` takes the values `reforecast_v12` (≤ 2019), `operational_v11` (Jan–Apr 2020, a different model) or `operational_v12`. Check that results do not hinge on the switch.

**Raw sources (all with the dev cut):**
- `data/forecasts/*.csv.gz`: GEFS per city and lead; CPC per city
- `data/weather/daily_temps_2000.csv`
- `data/raw/ng_curve_daily.csv`: licensed futures
- `data/raw/options/ng_options_near_money.csv.gz`: licensed options
  - columns: `date, root, cy, cm, cp, k_raw (strike × 1000), settle, volume, oi`
  - `root`: ON = options on futures month `cm`/`cy`; G4X = options on the 1-month calendar spread (month cm minus month cm+1), so `cm=3` is March−April; G3B = 3-month spreads; G6B = 6-month spreads
  - options expire 1 business day before the futures' last trading day, which is 3 business days before the delivery month
- **Shared helpers:** `src/v4/forecasts.py` (anomalies, `news()`) and `src/v4/panel.py`.

**Costs:**
- Outrights: 1 tick = $0.001 = $10 per contract; use 2 ticks per side (and report 4).
- Spreads: per leg, as for outrights.
- Options: assume half the bid−ask is about 1–2% of premium for near-ATM monthly options and 3–5% for calendar spread options, unless you can do better from volume and open interest. State what you assume.

## Report: `reports/agents/v4_<name>.md`, under about 1,500 words
1. **Plain-English findings:** 3–6 bullets a trader reads in one minute.
2. **Method,** intuitively, with the key formulas.
3. **Results:** tables with CIs and 1–3 figures, each referenced in the text.
4. **What this tells us** about news versus premium in the gas curve.
5. **Caveats,** including the variant count.
6. **Proposed frozen test** for the locked 2000–2010 sample, which has prices from the front 4 contracts only: outrights all winter, and H−J from late November to expiry; no options, no strip, no storage surprises, no CPC. Alternatively, if the test needs options, for live winter 2026-27. Give inputs, target, horizon, statistic and threshold.
7. **How to rerun:** the command and its runtime.
8. **New avenues:** 2–4 follow-ups, each with a line on its expected value.
