# Study v4 synthesis: news vs premium in the gas curve

*23 Nov 2026. Based on five agent reports (`reports/agents/v4_*.md`) and an independent review (`reports/agents/review_v4.md`). All evidence uses development data, 7 Jun 2010 to 31 Dec 2021. The 2000–2010 sample is still locked. The tests below were frozen and committed before it was opened.*

## What survived review
1. **Forecast news explains a modest share of curve moves.** Measured walk-forward, daily GEFS ensemble news explains about **5% of daily** and about **20% of weekly** front-month moves. The 11–15-day window carries the most weight, and CPC outlooks add nothing. Daily March−April moves are about 99% unexplained. Weekly, the share rises to 16–26% once moves are scaled by tightness.
2. **The free 00Z archive lags the market by about one model run.** Half of a midweek forecast revision, and a third of a weekend one, is already priced at the previous settle. The market sees the 06Z run (and ECMWF) first. So the "premium" part contains some early news, which biases daily reversal tests toward zero. This is a timing gap, not better forecasting: once the next run is known, the premium part predicts nothing more about the weather.
3. **The weekly forecast-unexplained move in the second month partly reverses over about 4 weeks.** The slope is −0.40, and it stays between −0.35 and −0.46 when any one winter is left out. The raw weekly move also reverses (−0.22). That the split *sharpens* the reversal is only suggestive (difference t = −1.75, and the 20-day horizon was chosen after seeing results).
4. **No ex-ante observable separates the spike that ran (24 Jan 2014) from the one that reversed (14 Nov 2018).** This holds for news, forecast level, implied vol and tightness. After the fact, 2014 was partly real cold (about a quarter of the move was justified by later weather) and Nov 2018 was pure premium.
5. **Nothing beats holding half a spread for the March−April short.** That covers forecast-based cuts, delayed entries, add-ons after squeezes, and out-of-the-money spread calls. The calls tie half size within cost uncertainty. Calendar-spread implied vol exceeded realised in 8 of 9 winters, but the all-winter mean gap (+0.06) has a CI spanning zero. Outright vol is fairly priced. Implied vol rises only *after* blow-ups.

## Overstated in the agent summaries (per the review)
- "Prices move the day before the revision" describes window misalignment, not anticipation.
- The volatility premium of +0.22 excludes the one adverse winter, 2013-14.
- "Expiry tails are overpriced (8% vs 0 of 11)": 0 hits in 9 winters is consistent with 8% (p = 0.47).
- The delayed-entry rule's edge is 2013-14 alone (1.09–1.16 with it, 0.99 without), and it fails its own 1.10 bar.

## Thesis after v4
The gas curve absorbs forecast news quickly, a model run ahead of the free archive. What remains is largely risk premium. Cold-scare premium partly mean-reverts over weeks, and the spread-options market charges a mild volatility markup for it. But neither forecasts nor options tell you **which** spike is the disaster, so for the widowmaker, position size remains the only reliable risk control.

---

## Frozen tests: locked 2000–2010 panel (run once)
**Frozen inputs:**

| File | sha256 |
| --- | --- |
| Coefficients `reports/frozen/v4_frozen_coefs.json` (copy of the agent's `frozen_coefs.json`, fitted on all development Nov–Feb rows) | `cd63189b…2960` |
| `src/research/v4_news_channel/decompose.py` | `829a2372…0629` |
| `src/research/v4_premium_shocks/frozen_test.py` | `03ac2f35…5e215dcd70a54c40480bb6437be7955` |

**General rules:**
- Locked panel `data/processed/v4_locked/panel_pre2010.csv`. Prices come from the EIA front 4 contracts; GEFS reforecast.
- Rows from 1 Oct 2000 onward. Test rows are Nov–Feb of winters 2000-01 to 2009-10.
- **First,** set r_front, r_second and d_hj to NaN wherever date − prev_date > 4 calendar days. These are EIA calendar gaps.
- Each test runs once. Runner: `src/holdout/v4_locked.py`. It is checked on development data first, where it must reproduce the development reference values.

| ID | Test | Pass rule | Dev reference |
| --- | --- | --- | --- |
| **P1 (primary)** | Weekly premium reversal, second month. `frozen_test.hps1`: fut20 ~ 1 + news_part + premium_part (weekly frozen split, Friday rows Nov–Feb), Newey-West with 5 lags. Also reported: naive slope; β_P − β_N; β_P at h = 5, 10, 15; leave-one-winter-out β_P | β_P < 0 with one-sided p < 0.05 **and** t(β_N) > −1.645 | β_P −0.35 (t −3.0) |
| S1 (gate) | Decomposition transfer, front month. R² = 1 − Σ(prem − mean)² / Σ(move − mean)² | Daily R² > 0.02 with winter-bootstrap 2.5% bound > 0; weekly R² > 0.08. If S1 fails, read P1 as a raw-move test | Daily 0.048 [0.026, 0.076]; weekly 0.224 |
| S2 | Controlled daily reversal, `frozen_test.hps1b` at h = 5 and 10. Uses future news as a control, so it is descriptive | β_P < 0 with one-sided p < 0.05 at both h | −0.24 (t −4.3); −0.20 (t −2.8) |
| S3 | March−April carry. Short the spread at the first row ≥ 1 Dec of Y−1 with hj_year = Y; exit at the last row with hj_year = Y; P&L = −ΔS × 10,000 − $80 (also report −$160) | Mean > 0 over the 10 winters. Also report the winter-bootstrap CI, positive-winter count, worst mark-to-market drawdown and the half-size row | v3: 10 of 11 winters positive |
| Logs (no pass/fail) | H-PS2 (costed fade vs naive fade); E-rule delayed entry (θ = 1.0, 5-day mean of (lvl_w2 + lvl_w3)/36.85 ≤ θ); H-HI-1 (premium_part(d) predicting news_part(d+1), front month, HAC) | — | — |

## Frozen tests: live winter 2026-27
These are added to `reports/live_2026_27/PROTOCOL.md` as a dated amendment before 1 Dec 2026.

| ID | Test | Data | Role |
| --- | --- | --- | --- |
| L3 = T-OPT1 | G = mean over d ∈ [1 Dec 2026, 22 Jan 2027] of nv_hj(d) − RV_life(d). nv_hj is computed as in `src/v4/panel.option_block` (ATM straddle, Bachelier, T = days to the Mar-27 option expiry / 365.25). RV_life(d) = √(Σ_{t>d} ΔS_t² / T_d) up to option expiry | G4X and H27/J27 settles, pulled by the user from Databento after expiry | Primary options test, n = 1. Pass if G > 0 |
| L4 | Protected short: short H27−J27 at the 1 Dec 2026 settle and buy the G4X Mar-27 call at K = F + $0.25 (rounded to $0.05) × 1.04 | As L3 | Descriptive |
| L5 | Plain 1 Dec short, half size and the E rule on H27−J27 (daily path, worst drawdown, expiry P&L) | Yahoo; GEFS operational archive | Log |
| L6 | Weekly news/premium log for the second month and H−J | Yahoo; GEFS | Log |
