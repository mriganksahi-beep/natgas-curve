# Brief for study v5 research agents: natural gas OPTIONS (read fully before starting)

Repo: `/home/claude/natgas-curve`. Read first:
- `reports/synthesis_v3.md`: the March−April (H−J) premium and its path risk.
- `reports/synthesis_v4.md` and `reports/holdout/SUMMARY_v4.md`: news vs premium. Forecast news explains about 5% of daily and 20% of weekly moves. The premium-reversal result failed on 2000–10.
- `reports/agents/v4_options.md` and `reports/agents/review_v4.md`, the section on options:
  - H−J calendar-spread implied vol exceeded realised in 8 of 9 winters, but the all-winter mean gap has a CI that spans zero.
  - Outright implied vol is fairly priced.
  - Implied vol jumps only *after* blow-ups.
  - Protective calls tie with half size.

**Who reads this.** The target audience is Statar Capital, a natural-gas fund with a long-volatility bias that trades gas implied volatility. Its systematic lead trades cross-sectional futures and volatility relative value. Results must be tradable, costed and honest.

## Hard rules
1. **Development data only: dates ≤ 2021-12-31** for options, futures, weather and storage. Read options data **only** from `data/processed/v5/` (already cut). For futures (`data/raw/ng_curve_daily.csv`) or the v4 panel (`data/processed/v4/panel_dev.csv`), filter to `date ≤ 2021-12-31` immediately. You may use outcomes that resolve up to 31 Mar 2022 *only* for a position opened on or before 31 Dec 2021, for example the March 2022 expiry. Say so when you do.
2. **Never open** `data/processed/v5_locked/`, `data/processed/v4_locked/`, `data/raw/options/*` rows after 2021, or pre-2010 EIA prices. The 2022–2026 options sample is the final exam; no study has looked at it.
3. **No look-ahead.** Anything fitted (betas, thresholds, factor loadings, GARCH parameters) is walk-forward: an expanding window, refitted at most monthly. Leave-one-winter-out is allowed only for *description*. Say which you used.
4. **Choose on 2010–2016; check on 2017–2021.** For calendar-spread-option (CSO) work, choose on 2014–2016 (dense strikes) and check on 2017–2021 (thin). Report both windows. Count the variants you tried.
5. **Licensed data.** No raw price tables in reports; statistics and charts are fine.
   - code: `src/research/v5_<name>/`
   - report: `reports/agents/v5_<name>.md`
   - figures: `reports/agents/figures/v5_<name>_*.png`
   - cache: `data/processed/research_v5_<name>/` (git-ignored)

   No git commands, and no edits to other files.
6. **Shared machine.** 2 CPUs and about 7 GB, with up to 5 agents. Keep each step under 10 minutes and cache intermediate results.
7. **Honesty over results.** Every signal must be scored in tradable terms: the P&L of the position it implies, after costs, against a simple benchmark. Give CIs: winter-block bootstrap, or HAC errors with lags at least the overlap. Label anything that rests on 1–2 episodes (2014, Nov 2018, Feb 2021).

## Data: the shared options panel (`src/v5/options_panel.py`; read its docstring)
**`data/processed/v5/surface.csv`.** One row per (date, cid) for monthly ON options on futures month cid (YYYYMM), with T from 2 trading days to 18 months.
- `F` and `T` (years, counted in trading days/252)
- `atm_iv`, plus `iv_m20`, `iv_m10`, `iv_p10`, `iv_p20` at log-moneyness ±0.1 and ±0.2, from out-of-the-money options
- `skew` = iv_p10 − iv_m10 (> 0 means calls are richer, i.e. upside fear); `curv`
- ATM straddle `gamma`, `vega` (per 1.00 vol) and `theta` (per calendar day)
- `call_oi`, `put_oi`, `call_vol` and `put_vol` for near-money strikes. Open interest and volume are for that trading day; before 2016 their dates were inferred, matching the exchange label 99.8% of the time where both exist.
- `gex`: unsigned open-interest-weighted gamma notional
- `n_strikes`. About 18 contracts a day until 2016, 7–10 after.

**`dh_straddle.csv`.** Delta-hedged ATM straddle held one trading day at the same strike: `pnl`, `gamma_pnl`, `theta_pnl`, `vega_pnl`, `resid`, `iv0`, `iv1` and Greeks. The attribution R² is 0.88–0.96 in every year. P&L is in $/MMBtu per straddle; one contract is 10,000 MMBtu.

**`spreads.csv`.** One row per (date, cid) for 1-month calendar-spread options (G4X; cid is the near leg, so cm = 3 is March−April).
- `S` (futures spread) and `S_parity`; `nv` (Bachelier ATM normal vol, $/MMBtu/√yr)
- `iv_near` and `iv_far` (ON ATM vols of the two legs)
- `rho_impl`, the implied correlation between the legs, using the normal approximation. The far leg's vol runs to its own expiry, an approximation you should handle or bound.
- `rho_valid`; `cso_call_oi`, `cso_put_oi`
- Coverage is dense in 2014–2016 (2,500–4,500 rows a year) and thin in 2017–2021 (300–850).

**`weekly.csv`.** Weekly options (ON1–5, LN1–5): `series`, `expiry`, `tdays`, parity `F`, `atm_iv`. Coverage is 2014 onward and thin in 2017–2021.

**Raw near-money options** (`data/raw/options/ng_options_near_money_v2.csv.gz`). **Filter to date ≤ 2021-12-31 on load.**
- Columns: date, root (ON, G4X, G3B, G6B, G10), cy, cm, cp, k_raw (strike × 1000), settle, volume, oi, hi_bid, lo_offer.
- `hi_bid` and `lo_offer` are the session's highest bid and lowest offer, **not closing quotes**, so they are only a rough upper bound on spreads.
- Weekly raw options: `ng_weekly_options_v2.csv.gz`.

**Other inputs:**
- Futures: `data/raw/ng_curve_daily.csv`, licensed.
- The v4 daily panel (forecast news, levels, realised weather, storage surprises), with `src/v4/panel.py` documenting its columns.
- Weather forecasts: `data/forecasts/`.

**Costs** (state yours):
- Futures: 2 ticks per side ($0.002/MMBtu).
- Monthly ATM options: half-spread of 1–2% of premium, minimum $0.002.
- CSOs: 3–5% of premium.
- Weekly options: 3%.
- Delta hedging: futures cost per rebalance.

## Report: `reports/agents/v5_<name>.md`, at most about 1,800 words plus tables
1. **Plain-English findings:** 3–6 bullets.
2. **Method:** intuition first, then formulas.
3. **Results:** with CIs, 1–3 figures, choose and check windows, and the variant count.
4. **What it means for a gas volatility desk.**
5. **Caveats.**
6. **A proposed frozen test for the locked 2022–2026 options sample,** runnable unchanged on the same panel columns: inputs, target, statistic and threshold. Optionally also a live 2026-27 version.
7. **How to rerun.**
8. **New avenues.**

## When you finish, reply with a ~200-word summary of the key numbers.
