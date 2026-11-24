# v5_weather_uncertainty: does GEFS ensemble disagreement tell a long-vol fund when gas vol is cheap?

*Agent `weather_uncertainty`, 24 Nov 2026. Development data only: options and `weather_unc.csv` from `data/processed/v5/` (≤ 2021-12-31). Futures, storage and the v4 panel are cut at 2021-12-31 for every feature. No `*_locked` folder was opened. **Outcome exception (rule 1):** forward realised vol for rows dated in Dec 2021 uses futures settles up to late Jan 2022. Sample: Oct–Apr trading days, winters 2010-11 to 2021-22 (Oct–Dec 2021 only). Winter 2010-11 is burn-in for walk-forward fits and thresholds. **Choose window:** Oct 2011 – Dec 2016. **Check window:** 2017–2021. Units: vol pts = 0.01 annualised Black vol. Trade P&L is in vol pts per unit vega, so a book running $10k of vega per vol pt makes $10k per vol pt shown.*

## 1. Plain-English findings
- **Ensemble disagreement does not tell you when gas vol is cheap.** Across about 150 inferential tests, **none survives Holm on the choose window**, and nothing is significant in both windows.
- **Realised-vol forecasting.** Disagreement adds nothing once implied vol is in the model.
  - Beyond HAR + seasonality alone, disagreement *levels* add +1.4 to +2.3 pts of out-of-sample R² in the check window (Clark-West t 2.4–3.2 at 10–20 days). In the choose window the same blocks *lose* 1.7 to 2.2 pts.
  - Beyond HAR + IV, the gain is −2.8 to +1.0 pts. Beyond HAR + IV + forecast level + storage it is −3.9 to +1.4. Every bootstrap CI spans zero.
  - The one stable pattern is descriptive. Higher 11–15-day disagreement goes with slightly higher subsequent RV: an elasticity of about +0.2 to +0.3 (t ≈ 1.8–2.2), so +10% disagreement means about +1 vol pt of RV. Higher 1–5-day disagreement goes with slightly *lower* RV (−0.2, t ≈ −2). IV already carries most of it.
- **Implied vol does not react to disagreement, on the day or later.**
  - A 1-s.d. jump in 6–15-day disagreement moves front IV by −0.10 vol pts the same day (t −1.5) and −0.17 over the next 5 days (t −1.7).
  - Event paths after top-decile jumps disagree in sign: +0.8 vol pts for the w2/w3 mean, −1.3 for w3 alone. The positive path comes with a +1% futures rally, so it is the spot-vol correlation, not repricing of uncertainty.
  - There is no under-reaction to buy ahead of.
- **No disagreement-timed long-straddle rule makes money after costs.** Weekly-switched, delta-hedged long front straddles at 1.5% half-spread plus 2 ticks per hedge change:
  - "Long when disagreement is high" loses −4.9 vol pts per winter in the choose window and −6.9 in the check window.
  - It loses less than always-long (−7.7 / −28.8), but only because it is flat two-thirds of the time. Against exposure-matched random timing it sits at the 66th and 82nd percentiles.
  - Cost-free selection (weekly gross P&L when the signal is on minus when it is off) is +0.6 vol pts per week in the choose window [−1.6, 2.2] and +0.4 in the check window [−3.5, 4.6].
- **Q4 (tight storage plus high disagreement): no.** The two squeezes that pay a long-vol book split across the cells. **Jan–Feb 2014** came with *high* disagreement. **Nov 2018**, the biggest long-straddle week sum in the sample, came with *low* disagreement. "Tight plus high disagreement" earned +5.2 per winter net in the choose window (2 gas years, all of it 2013-14) and −7.8 in the check window (4 gas years). Tight storage *without* high disagreement did better (+4.8 / +7.4, mostly Nov 2018). **This rests on 2 episodes.**
- **One weak directional hint, the opposite of a long signal.** When disagreement is *low* and storage is not tight, front IV ran above the next 20 days' RV:
  - normal storage: +6.2 vol pts [3.1, 8.6];
  - loose storage: +2.8 [0.9, 4.3].
  - Long straddles lost in the normal-storage, low-disagreement cell: −1.3 vol pts per week [−1.9, −0.5], over 9 gas years.
  - The high-minus-low tercile gap (IV − RV20) is −3.5 (choose) and −2.9 (check), with CIs spanning zero.
  - This cell was found post hoc. At most it is a *when not to own vol* filter.

## 2. Method
**Intuition.** If the market underprices weather risk, ensemble spread should predict realised vol that IV has not priced, or IV should catch up with a lag. I test both, then trade it.

**Signals** (`weather_unc.csv`, known at the d settle):
- `unc_n_w1/w2/w3`: normalised GEFS 5-member s.d. for days 1–5, 6–10 and 11–15. 1.0 is a typical value.
- Because 5 members are noisy (AR(1) only 0.4–0.6), levels use a **3-day mean**, and the main index is the mean of w2 and w3 (`w23`).
- Changes are either the 1-run change in log `unc_n` or the 5-day change in its 3-day mean. I never use `d_unc`, which is mechanically negative.

**Q1, forecasting.** Pooled OLS on log forward RV (h = 5, 10, 20 days), fitted walk-forward: an expanding window refitted at each month start, using only rows whose target had resolved.
- Leg 1 is the nearest ON contract with more than h trading days to option expiry, so the window lies inside the option's life. Leg 2 is the next contract.
- HAR block: log RV5, RV21 and RV63 of the contract itself, log tenor, and calendar-month dummies.
- Five nested comparisons:
  - A: HAR → +levels
  - B: HAR → +changes
  - C: HAR + log IV → +levels + changes
  - D: HAR + IV + forecast level (lvl_w1–3 and |Σlvl|) + storage deviation (vol_rv's sdev) → +disagreement
  - E: HAR + IV → +w2/w3 levels only
- Tests: one-sided Clark-West with HAC lags = h, and a gas-year bootstrap CI of the R² gain. Holm across the 30 choose-window tests.

**Q2, IV reaction.** Front = nearest contract with more than 15 trading days left, fixed over d−6 to d+10.
- Regressions: ΔIV (same day, and d → d+k) on the standardised change in disagreement, controlling for r_d and |r_d|, with HAC errors.
- Event study: one-run jumps above the 90th percentile, declustered to at least 5 days apart (124–133 events per definition). Paths are measured against the same-calendar-month mean, with a gas-year bootstrap.

**Q3, trade engine.**
- Decision on the first trading day of each week. Target position in {−1, 0, +1} unit vega on the front contract (≥ 10 trading days to expiry).
- A straddle at K = F is **kept without re-striking** while the target persists. It is rolled when tenor falls below 10 days.
- Marked at the surface smile with vol_rv's pricer and delta-hedged daily. Validated against the settle-based `dh_straddle`: r = 0.78 on daily P&L. That is lower than vol_rv's 0.92 because the strike is fixed, so gamma drifts.
- Costs: 1.5% of premium per option side (min $0.002 per option; 1% and 2% also run) plus $0.002 × |Δhedge| at every rebalance.
- Thresholds are walk-forward expanding percentiles: high = above the 67th, combined rules use the 50th, cheap gap = below the 33rd.
- Benchmarks:
  - always long and always short;
  - vol_rv's storage-sign rule (long if sdev < 0, else short) and tight-only rule (long if sdev < −150);
  - 100 draws of random weekly timing with P(on) = 1/3 and 1/2.

**Q4.** Storage cells as in vol_rv (tight < −150 bcf, loose > +150), crossed with disagreement above or below its walk-forward median. Outcomes:
- the weekly gross P&L of a fresh long straddle;
- IV − RV20.

Each cell lists its contributing gas years.

## 3. Results
**Table 1. Q1: out-of-sample R² gain from adding disagreement** (pct pts; [gas-year 95% CI]; CW t).

| Comparison | h | Choose L1 | Choose L2 | Check L1 | Check L2 |
|---|---|---|---|---|---|
| A: HAR → +levels | 5 | +0.7 (1.2) | +1.2 (1.7) | +0.5 (1.7) | +0.4 (1.7) |
| | 10 | −1.9 (−0.6) | −1.7 (−0.4) | **+1.9 [0.2, 4.6] (3.2)** | **+2.3 [0.4, 5.8] (3.2)** |
| | 20 | −2.2 (−0.8) | −1.7 (−0.9) | +1.4 (2.4) | +1.7 (2.5) |
| C: HAR+IV → +lev+chg | 5 | −0.5 | −1.0 | −0.2 | −0.3 |
| | 10 | −2.2 | −2.7 | +0.5 (1.5) | +1.0 (2.1) |
| | 20 | −2.0 [−8.3, 4.6] | −2.8 | +0.4 (1.6) | +0.7 (1.8) |
| D: +level+storage → +disagr. | 20 | −0.1 | −3.7 | +0.8 (1.8) | +1.2 (2.0) |

- Base R² for HAR + IV at 20 days is 0.58 / 0.63 (choose) and 0.64 / 0.64 (check), consistent with vol_rv.
- B (changes only) and E (w2/w3 only) are within ±1.3 pts everywhere.
- **Holm-adjusted choose p = 1.0 for all 30.**
- Log IV's coefficient is 0.72–0.83 (t ≈ 9). IV dominates.

**Table 2. Q2: front IV reaction** (vol pts per 1-s.d. disagreement jump; HAC t).

| Index | Same day | d → d+1 | d → d+5 | d → d+10 |
|---|---|---|---|---|
| w2 change | −0.11 (−1.7) | +0.08 (1.3) | −0.15 (−1.4) | 0.00 (0.0) |
| w3 change | −0.04 (−0.6) | +0.01 (0.1) | −0.10 (−0.9) | +0.02 (0.2) |
| w23 change | −0.10 (−1.5) | +0.06 (0.8) | −0.17 (−1.7) | +0.01 (0.1) |
| w23 level (lagged only) | – | −0.12 (−1.5) | −0.19 (−0.6) | −0.23 (−0.4) |

- Choose and check splits flip sign in most cells: across 63 regressions the largest |t| is 1.9, and none is consistent across windows.
- Event study, top-decile w23 jumps (124 events): +0.0 vol pts on the day, +0.8 by d+2 [0.1, 1.5], +0.8 by d+10 [−0.8, 2.5]. Front futures rose +1.1% over the same days.
- w3-only events: −1.3 by d+4. Figure 1, right.

**Table 3. Q3: weekly-switched front straddles, net of 1.5% + 2 ticks** (vol pts per winter [gas-year CI]; positive winters / total where shown; max drawdown on daily cumulative net P&L, choose / check).

| Rule (weeks on) | Choose | Check | Max DD (window) | Pct vs random, choose / check |
|---|---|---|---|---|
| Always long (100%) | −7.7 [−22.6, 7.4] 3/6 | −28.8 [−41.6, −15.4] 0/6 | −75 / −203 | – |
| Always short | −22.6 [−36.8, −10.3] 0/6 | −3.7 [−14.1, 6.8] 3/6 | −145 / −89 | – |
| Storage sign (vol_rv) | −7.2 | −30.6 | −69 / −196 | – |
| Storage tight only (31%) | +6.1 [−14.9, 34.2] 1/3 | −13.7 [−23.9, −3.6] 1/5 | −31 / −86 | 99 / 54 |
| U2 high (33%) | +0.0 [−6.3, 6.3] | −8.9 | −36 / −92 | 87 / 72 |
| U3 high (29%) | −6.5 | −7.4 | −56 / −97 | 55 / 81 |
| **U23 high (35%)** | **−4.9 [−15.1, 6.2]** | **−6.9 [−21.7, 6.6]** | −61 / −81 | 66 / 82 |
| U23 rising (51%) | −13.8 | −12.9 | −88 / −106 | 17 / 65 |
| U23 high, long/short | −25.7 | −23.7 | −184 / −242 | – |
| U23 high and rising | −4.7 | −9.4 | −43 / −86 | 67 / 68 |
| Gap low: IV < HAR forecast (24%) | +4.4 [−4.5, 12.2] | −7.2 | −21 / −56 | 98 / 81 |
| Gap low vs HAR + disagreement | −2.0 | −5.4 | −30 / −56 | 76 / 85 |
| Gap low and U high (17%) | 0.0 | −8.1 [−15.1, −0.1] | −21 / −85 | 87 / 78 |
| Tight and U high (14%) | +5.2 [−2.6, 13.1] 1/2 | −7.8 [−19.2, 0.0] 1/4 | −26 / −53 | 99 / 80 |
| Tight and not U high (17%) | +4.8 [−5.2, 20.6] | +7.4 [−18.3, 47.3] | −13 / −55 | 99 / 94 |

- Random timing at P(on) = 1/3, net per winter: median −7.2 (choose) and −15.0 (check); 5th–95th percentiles −15.3 to +3.0 and −28.3 to +7.5. Random draws switch more often than persistent signals, so low-exposure rules are flattered.
- Holm on the 11 non-benchmark rules (net > 0, choose window): every p_holm = 1.0.
- The paired "beats always-long" differences are positive for every rule in the check window. That is purely lower exposure to a losing benchmark, not skill.
- At 1% half-spread the best disagreement rule, U2 high, nets +1.1 (choose) and −7.6 (check); its gross is +12.3 / +2.5. Most of the cost is hedging and front-month rolls, not the half-spread level.

**Table 4. Q4: storage × disagreement** (front IV − RV20, vol pts [CI]; weekly fresh-straddle gross, vol pts per week).

| Cell | IV − RV20 | Long straddle per week | Gas years | Driven by |
|---|---|---|---|---|
| Tight, high disagreement | −4.3 [−9.7, 4.4] | +0.97 [−0.8, 2.4] | 6 | 2013-14 (+42 of +43 total; 2018-19 −17) |
| Tight, low disagreement | −1.8 [−6.4, 9.5] | +1.96 [−1.6, 4.5] | 8 | 2018-19 (+110: Nov 2018); 2021-22 −26 |
| Normal, high | +2.6 [−0.3, 6.4] | +0.22 | 8 | – |
| Normal, low | **+6.2 [3.1, 8.6]** | **−1.34 [−1.9, −0.5]** | 9 | broad |
| Loose, high | +1.1 | +0.88 | 7 | 2019-20 |
| Loose, low | **+2.8 [0.9, 4.3]** | −0.23 | 7 | broad |

- Inside tight storage, high disagreement does **not** mark the underpriced weeks: low-disagreement weeks earned more.
- The tight cells' result is **Jan 2014 + Nov 2018 (2 episodes)**.

**Variant count:**
- Q1: 30 out-of-sample tests, plus 96 descriptive coefficients.
- Q2: 63 regressions and 3 event definitions.
- Q3: 15 rules × 4 cost levels, with 11 tested rules × 3 tests (net, vs long, selection), plus 200 random draws as a benchmark.
- Gap analysis: 9 cells and 3 differences. Q4: 12 cells.
- About 150 inferential tests in all. Nothing was tuned after looking: the thresholds (67th / 50th / 33rd percentiles), weekly decision days, the 3-day smoothing and the w23 index were fixed before any P&L was computed. The Q4 "low disagreement and not tight" cell is **post hoc**.

Figures:
- `figures/v5_weather_uncertainty_forecast_event.png`: Q1 gains and Q2 paths.
- `figures/v5_weather_uncertainty_trade.png`: cumulative net P&L.
- `figures/v5_weather_uncertainty_cells.png`: Q4 by gas year.

## 4. What it means for a gas vol desk
- **Do not use GEFS ensemble spread as a buy-vol trigger.** The options market already prices whatever disagreement tells you about future realised vol, through IV. IV does not lag it.
- **The free 5-member 00Z spread is a weak instrument.** v4 found the market trades a model run ahead of the free archive, and the market sees ECMWF's 51 members. A desk with the full ensembles might do better (Section 8), but nothing here suggests a large effect.
- **Squeezes are tightness events, not uncertainty events.** For a long-vol bias, vol_rv's storage-tightness tilt remains the better (still episode-thin) sizing input. Adding disagreement does not sharpen it.
- **If anything, use disagreement on the sell side.** Low disagreement in non-tight storage is where front IV ran richest versus subsequent RV. It is a "don't carry long gamma here" flag, and post hoc.
- **Costs dominate.** Always-long front straddles cost about 17 vol pts per winter in spreads and hedging. Any timing signal must clear that, and none of these does.

## 5. Caveats
- The 5-member spread is noisy. Reforecast (≤ 2019) and operational (2020+) runs differ, although their levels match within about 0.1°F.
- The spread is a gas-weighted average of per-city s.d.s, an upper bound on aggregate HDD uncertainty.
- The fixed-strike engine is validated at r = 0.78, not 0.92. Costs are assumed, not observed.
- Choose-window "winters" include partial blocks (Oct–Dec 2016 sits in gas year 2017). Oct–Dec 2021 is truncated.
- The tight-storage results rest on **Jan–Feb 2014 and Nov 2018**, and sdev is only defined from 2012.
- Walk-forward thresholds in 2011-12 use a single prior winter.

## 6. Frozen test for the locked 2022–2026 sample
**Runner and constants:**
- `src/research/v5_weather_uncertainty/frozen.py`. Constants are in `frozen_constants.json`, fitted on all development Oct–Apr rows:
  - unc_n_w23 3-day mean: q33 = 0.962, q50 = 1.018, q67 = 1.095;
  - 1-run change: mean 0.0005, s.d. 0.168;
  - HAR+IV and HAR+IV+U coefficients for h = 20, leg 1.
- Command: `V5_DIR=data/processed/v5_locked V5_CUT=<panel end> V5WU_CACHE=data/processed/research_v5_weather_uncertainty_locked python -m src.research.v5_weather_uncertainty.frozen --test --start 2022-01-01`.
- The same columns are used (`weather_unc`, `surface`, futures and storage, the last two cut at the panel end). Nothing is refitted.
- On dev data (`--start 2011-10-01`) it reproduces the reference values below.

| ID | Statistic | Holds / passes if | Dev reference |
|---|---|---|---|
| **T-WU1 (primary, falsification)** | ΔR² of HAR+IV+U over HAR+IV (frozen coefficients), log RV20 of leg 1, Oct–Apr; Clark-West t with 20 HAC lags | Conclusion **holds** unless ΔR² ≥ 0.010 **and** t ≥ 1.645 | In-sample +0.011 (t 2.9), an upper bound; walk-forward −0.001 |
| T-WU2 | Front ΔIV on the frozen-standardised 1-run change in w23 (controls r, \|r\|): same day and d → d+5 | Holds if both \|t\| < 2 | −0.10 (t −1.5); −0.18 (t −1.6) |
| T-WU3 | Weekly long front straddle when the w23 3-day mean > 1.095; engine and costs as in Table 3; net per winter, winter bootstrap | Holds if mean ≤ 0 or the 95% CI includes 0; also report on − off weekly gross | −7.1 [−16.9, 2.3]; selection +0.40 [−1.8, 2.7] |
| T-WU4 (secondary) | Front IV − RV20, high (> q67) minus low (< q33) disagreement | **Passes** if < 0 with one-sided p < 0.10 | −3.3 [−8.9, 1.6], p 0.11 (fails on dev) |
| T-WU5 (log) | Weekly gross in tight × high vs tight × low disagreement (q50) | Report; "not testable" if any cell has < 10 weeks | +0.92 (48 weeks) vs +2.07 (52 weeks) |

If T-WU1 or T-WU3 is falsified, disagreement carries priced information that 2010–21 did not show. Winter 2022 (the Ukraine-shock vol regime) is in the sample and should be reported separately.

**Live 2026-27 (optional):**
- Each trading day from 1 Dec 2026 to 30 Apr 2027, log w23 from the operational GEFS 00Z run (same normalisation), the front ATM IV, and the frozen high/low flags.
- At the end of April, score T-WU4's sign on all resolved days, and the weekly paper long-straddle P&L on high-flag weeks against all weeks.
- n is about 100 days, so this is a log, not a test.

## 7. How to rerun
`python -m src.research.v5_weather_uncertainty.run --force` takes about 2 minutes on 1 CPU. It writes caches to `data/processed/research_v5_weather_uncertainty/` (base, q1_*, q2_*, q3_*, q4_*) and the 3 figures. Modules: `data`, `forecast` (Q1), `event` (Q2), `trade` (engine), `analysis` (Q3/Q4), `figures` and `frozen`.

## 8. New avenues
- **Full ECMWF ENS (51 members) or GEFS 31-member spread,** including 06Z, against IV. The 5-member 00Z spread may simply be too noisy and too late. *Value: moderate.*
- **Disagreement as a short-vol filter:** sell front straddles or strangles in non-tight, low-disagreement weeks with banded hedging. This is post hoc, so freeze it before 2022. *Value: moderate.*
- **Bimodality rather than spread:** whether members split into cold and warm clusters, which matters for gas convexity, against call skew. *Value: unknown.*
- **Disagreement versus weekly-option implied variance around the 6–10-day horizon (2014+).** *Value: low to moderate*, given thin data after 2016.
