# v5_implied_corr: implied correlation between adjacent gas months, and a correlation risk premium

*Agent `implied_corr`, 24 Nov 2026. Development data only (dates ≤ 2021-12-31). Options were read only from `data/processed/v5/`. Futures and the v4 panel were filtered on load; `panel_dev.csv` holds rows up to 2026, which were dropped. No `*_locked` folder was opened. No outcome after 2021-12-31 was used: winter 2021-22 has no expiry here, and contracts that expire after the cut have no realised values. Windows: **choose = 2014–2016 and check = 2017–2021**, by the row date or the trade entry date.*

## 1. Plain-English findings
- **The market prices about 2–3× more decoupling between adjacent months than actually happens.**
  - Across all adjacent-month CSOs at 5–126 trading days to expiry, median implied ρ is **0.981–0.984** against a realised **0.994**.
  - Implied 1−ρ is 0.010–0.020 against a realised 0.004–0.008.
  - The correlation risk premium CRP = ρ_realised − ρ_implied averages **+0.033 to +0.045** across all 6 horizon and invalid-row treatments. Every 95% winter-bootstrap CI excludes 0 on the full sample.
  - It is positive in both windows. For March−April it is larger, **+0.05 to +0.07**, and positive in 8 of 9 winters.
- **It is a spread-option premium, not an outright one.**
  - A short delta-hedged CSO straddle held to expiry keeps **23% of premium on average** [4%, 37%] (median 35%) before costs.
  - Delta-hedged ATM straddles on the legs keep about 0% (−2% near, −1% far).
  - This confirms v4's "spread vol rich, outright fair" across the whole curve, not only March−April.
- **The textbook "correlation trade" does not pay after costs.**
  - Hedging the short CSO with long leg straddles in ρ-neutral vega ratios earned +$329 per trade on the choose window and **−$89 on the check window** (entry 42 days out, chosen on choose).
  - The gamma-matched (covariance-swap) version needs 5–38 leg straddles per CSO. Leg vol and costs swamp it: **−$13k per trade**.
  - The plain short CSO straddle is the better trade: +$338 (choose) and **+$137 (check)** per trade net. It is small, with fat left tails: **−$3.7k in 2013-14**.
- **ρ_impl does not lead H−J squeezes once you control for the spread level.**
  - ρ_impl has a −0.85 rank correlation with CSO implied vol, and implied vol is about the spread level.
  - A rule that goes flat when ρ_impl is unusually low avoided Nov 2018. A price-only twin (flat when CSO vol is high) avoided it equally well and did better overall.
  - Out-of-sample incremental R² over level + trailing RV + nv is ≤ 0 for decoupling, premium decay and drawdown. *Nov 2018 is one episode.*
- **Data health.** 66% of all `spreads.csv` rows are exchange default settles (nv pinned at about 1.18 $/√yr, mostly more than 4 months out); I exclude them. The panel's `rho_valid=False` (9.3% of rows) is mostly a missing leg IV (74%). Where those IVs exist, the invalid rows are ρ>1 (nv < |σ_n − σ_f|) or ρ<−1, the latter almost all on default settles.

## 2. Method
**Intuition.** A 1-month CSO prices the spread's vol. Given the two legs' outright vols, the spread vol implies a correlation:

  nv² = σ_n² + σ_f² − 2ρ σ_n σ_f, with σ = F·IV (normal approximation, $/MMBtu/√yr).

If adjacent months move together more than the CSO implies, a seller of spread vol earns a "correlation premium."

**Horizon mismatch (step 1).** The far leg's ON option runs a month past the CSO, but the spread formula needs the far leg's vol over the *near* option's life [0, T_n]. I split the far leg's implied variance:

  σ_f²T_f = σ_h²T_n + σ_L²(T_f − T_n)

Here σ_L is the far contract's implied vol in its own last month, observed at the near expiry. Three treatments:
- **R0:** the panel's ρ (σ_h = σ_f).
- **R1, central:** σ_L = q̂·σ_f(d). q̂ is the walk-forward median of the realised ratio over *completed* past contracts of the same month pair and tenor bucket (expanding window, past information only).
- **R2:** the realised σ_L. This uses look-ahead and serves as a bound only.

q̂ is about 1.02 inside 21 days and 1.2–1.7 beyond 63 days (Samuelson). Rows with ρ > 1 are either **dropped** or **capped** at 1; both are reported, because dropping biases ρ_impl down.

**Realised (step 2).** Over (d, T_n] ("life") and the next 20 trading days, using zero-mean normal changes so the measure matches the implied formula:

  ρ_r = Σ dF_n dF_f / √(Σ dF_n² Σ dF_f²)

- CRP = ρ_r − ρ_impl.
- Sample: non-default settles, T = 5–126 trading days, ρ_impl ≥ −1.
- CIs come from a gas-year (Apr–Mar) block bootstrap with 4,000 draws. This is descriptive; there is no fitting beyond q̂.

**Variance decomposition.** The CSO's variance premium splits exactly into leg terms and a correlation term:

  nv² − RV_S² = (σ_n² − RV_n²) + (σ_h² − RV_f²) − 2(ρ_i σ_n σ_h − cov_r)

**Trade (step 3).**
- **Entry.** For each near month, enter at the first CSO quote 21, 42, 63 or 105 trading days before expiry (4 entry sets). Take the listed strike nearest S, within $0.10.
- **Positions and exits:**
  - Short 1 G4X straddle, held to expiry and settled at intrinsic |S_T − K| (CSO and ON expire together).
  - Long w_n near ON straddles, held to expiry.
  - Long w_f far ON straddles, closed at the near expiry at the raw settle (100% of exits had one).
- **Weights:**
  - **G:** dollar-gamma matched, w = Γ_CSO / Γ_leg. Then P&L ≈ Γ(Σ dF_n dF_f − ρ_i σ_n σ_f dt), a covariance swap.
  - **V:** ρ-neutral vega, w_n = V_S·(σ_n − ρσ_f)/nv / V_n. To first order this leaves only ρ exposure. Negative weights mean selling the leg.
- **Delta hedging.** Deltas at the entry IV. Three hedge rules:
  - daily, per leg, 2 ticks per leg
  - daily via the listed spread, 2 ticks per spread
  - **a 0.2-delta no-trade band (primary)**
- **Costs:**
  - CSO: **4% of premium** (3/5/8% as sensitivity)
  - ON: max(1.5%, $0.002) per option per side
  - exercise offsets: 2 ticks
- **Benchmarks:**
  - A: short CSO alone
  - B: short near ON straddles, premium-matched (or gamma-matched legs)
- **Selection.** Hedge rule and 4% were fixed a priori. The entry horizon (42) was chosen as the best choose-window t-stat of the correlation trade V. Marks for drawdowns use raw settles at the strike (CSO: 100% in 2014–16, 24% after) and otherwise the model at that day's vol.

**Indicator (step 4).**
- **Sample.** H−J target winter from the v4 panel (`hj`, `d_hj`). Weekly points from the first Friday of November to 20 days before expiry, on days with an H−J CSO quote: 59–72 points over 9 winters.
- **Signals:**
  - ρ_impl (R1, capped)
  - Δρ over 5–10 days
  - ρ_dev = ρ_impl minus the walk-forward median of past winters in the same tenor bucket
- **Targets:**
  - next-20-day realised decoupling 1−ρ_r20
  - Δhj20 (premium decay)
  - MAE20 (adverse excursion for a short)
- **Models.** Expanding-window OLS by winter against the benchmarks hj + RV20 and hj + RV20 + nv.
- **Overlay.** Short 1 H−J (v3 trade). Go flat for a week when ρ_dev < the walk-forward 20th percentile, against price-only twins (nv or hj above their walk-forward 80th percentile on the same days) and half size. $40 per side.

## 3. Results

**Table 1. Data audit (all 14,066 `spreads.csv` rows)**

| Item | Share |
|---|---|
| Exchange default settles (nv ∈ [1.15, 1.21]), excluded | 65.8% (8,196 rows; median T = 232 days) |
| Panel `rho_valid=False` | 9.3% of rows |
| … cause: leg IV missing (mostly far legs > 18 months) | 74% |
| … cause: ρ < −1 | 14%, nearly all on default settles |
| … cause: ρ > 1 (nv < \|σ_n − σ_f\|) | 12% |
| ρ>1 share in the analysis sample, R0: choose / check | 1.2% / 7.7% |
| ρ>1 share in the analysis sample, R1 | 16.6% / 12.2% |

ρ>1 rows are short-dated (median 27 days), thin (6.5 strikes against 9), and 74% fall in 2020–21. Shrinking the far vol (R1) widens |σ_n − σ_h|, so near-1 correlations tip over 1. **The horizon adjustment moves ρ by about 0.01, the same size as the median CRP**, which is why I report all treatments.

**Table 2. CRP = ρ_realised(life) − ρ_impl. Mean [95% winter-block CI], median, share of rows > 0**

| Sample | Window | R0 capped | R1 capped | R1 dropped | R2 (oracle) capped |
|---|---|---|---|---|---|
| All pairs | all (10 gas years) | +0.036 [0.015, 0.061] | **+0.033 [0.012, 0.057]**, med 0.007, 65% | +0.040 [0.017, 0.067] | +0.034 [0.014, 0.060] |
| All pairs | choose | +0.036 [0.019, 0.063] | +0.034 [0.013, 0.063] | +0.043 [0.020, 0.080] | +0.032 |
| All pairs | check | +0.036 [−0.004, 0.095] | +0.032 [−0.004, 0.088] | +0.038 [0.002, 0.094] | +0.041 [0.003, 0.099] |
| Winter pairs (Nov–Mar) | all | +0.032 [0.011, 0.056] | +0.027 [0.004, 0.052] | +0.035 [0.011, 0.059] | +0.028 |
| H−J (cm = 3) | all (9 winters) | +0.063 [0.034, 0.093] | **+0.051 [0.022, 0.083]**, med 0.018 | +0.065 [0.034, 0.097] | +0.052 |
| H−J | choose / check | +0.048 / +0.089 | +0.032 / +0.077 | +0.049 / +0.090 | +0.022 / +0.090 |

- **Fixed 20-day horizon (R1 capped):**
  - All pairs: +0.032 [0.018, 0.057]
  - Winter pairs: +0.038 [0.022, 0.067]
  - H−J: +0.061 [0.023, 0.111]
- **By tenor (R1 capped):** CRP is +0.067, +0.031, +0.027 and +0.017 at 0–21, 22–42, 43–63 and 64–126 days.
- **By month pair:** positive in 10 of 12. The exceptions are Oct−Nov (−0.008) and Nov−Dec (−0.034): the autumn storage-full pairs, where realised decoupling (ρ_r 0.965) exceeded implied.
- **H−J by winter:** +0.01 (2013), **+0.15 (2014, 14 days only)**, +0.07, +0.01, +0.01, +0.06, **+0.14 (2019)**, +0.08, −0.01 (2021). That is 8 of 9 positive.
- In 2014 and 2018-19 realised decoupling was large, yet implied decoupling was larger still. The CSO market over-reacts *after* the blow-up begins, consistent with v4.
- Figure: `figures/v5_implied_corr_rho.png`.

**Variance decomposition (R1, all pairs).** In ($/MMBtu)²/yr, the CSO variance premium is +0.058 [−0.12, 0.15]. It splits into:

| Term | Value |
|---|---|
| Correlation term | **+0.51** [−0.20, 1.51] |
| Near leg | −0.28 |
| Far leg | −0.18 |

On a variance basis the leg terms are negative: rare leg-vol spikes, such as 2014 and Oct 2021, are under-priced in *variance*, though fair in vol. **The correlation premium is real, but it comes bundled with short leg variance.** That is why hedging the legs costs more than it saves.

**Table 3. Trades, net $ per CSO straddle (10,000 MMBtu), band hedge, CSO 4%. Mean per trade; [CI] is the mean of winter means with winter bootstrap.**

| Strategy | Entry | Choose (n) | Check (n) | All, winter-mean [CI] | Worst trade |
|---|---|---|---|---|---|
| **A: short CSO straddle** | 42 | +338 (24) | **+137** (34) | −105 [−960, 446] | −3,663 (H14, Dec 2013) |
| A | all 4 entries | +402 (86) | +127 (120), t = 2.0 | −31 [−700, 380] | −3,663 |
| B: short near ON straddle, premium-matched | all | −83 | −15 | −225 [−661, 23] | −3,662 |
| **D: correlation trade, ρ-neutral vega** | **42 (selected)** | +329 | **−89** | +32 [−150, 226] | −1,904 |
| D | all 4 entries | +327 | −144 | −36 [−307, 209] | −6,199 |
| C: correlation trade, gamma-matched | 42 | −2,392 | −20,892 | −15,701 | −530,479 (Dec-21 CSO entered Oct 2021, leg IV 106%) |

- **Gross capture as a share of premium (mean):**

  | Leg | Choose | Check |
  |---|---|---|
  | CSO | 16% | 29% |
  | Near ON | −4% | −0% |
  | Far ON | −1% | −0% |

  - Realised spread vol was 0.61 × implied (median).
  - Costs per CSO trade are 9.5% of premium (option plus exercise) and 7% (band hedge).
  - With a daily per-leg hedge the check mean of A falls to +$34.
- **Cost sensitivity of A (check, entry 42):** +$148, 137, 126 and 94 at 3, 4, 5 and 8% CSO half-spread.
- **Portfolio, entry 42, all pairs (Figure `figures/v5_implied_corr_pnl.png`):**

  | Strategy | Choose: total / max DD | Check: total / max DD |
  |---|---|---|
  | A | +$8.1k / −$7.8k | +$4.7k / −$3.5k |
  | D | +$7.9k / −$3.4k | −$3.0k / −$5.0k |

- **Episodes:**

  | Episode | A | D | Note |
  |---|---|---|---|
  | 2013-14 | −$3.7k (H14 entered 23 Dec) | −$0.1k | 2-strike quotes, marks mostly model |
  | Nov 2018 | −$0.1k | −$0.5k | The H19 CSO at entry 105 (24 Sep) lost $2.5k as A and made +$0.2k as D; no H19 CSO quote at entry 42/63 before the squeeze |
  | Feb 2021 | −$0.5k | −$0.3k | Small: the near-month H21−J21 CSO stayed near ρ = 1 |

  All rest on 1–2 episodes.

**Table 4. Indicator (step 4), H−J weekly, walk-forward. Out-of-sample incremental R² over hj + RV20 (+ nv): choose / check / all [CI]**

| Signal → target | Pooled Spearman | Over hj + RV20 | Over hj + RV20 + nv |
|---|---|---|---|
| ρ_dev → 1−ρ_r20 (decoupling) | −0.63 | −0.07 / +0.01 / +0.01 [−1.8, 0.3] | −0.35 / +0.14 / +0.14 [−2.8, 0.4] |
| ρ_dev → Δhj20 (decay) | +0.48 | −0.14 / +0.18 / +0.14 [−0.6, 0.6] | 0.00 / −0.87 / −0.76 [−5.7, 0.2] |
| ρ_dev → MAE20 (drawdown) | −0.11 | +0.10 / −0.02 / −0.02 [−0.5, 0.5] | +0.10 / −1.30 / −1.22 [−14.5, 0.1] |
| ρ_impl → MAE20 | −0.08 | −2.42 / −0.04 / −0.14 | −1.59 / −0.79 / −0.91 [−6.0, −0.2] |
| Δρ5 → MAE20 | +0.09 | −0.05 / −0.03 / −0.03 [−0.15, −0.01] | −0.01 [−0.15, 0.03] |

- **Low ρ_impl forecasts more decoupling and more decay, but only as a level proxy.** The rank correlation of ρ_dev with nv is −0.85 and with hj −0.83. It never improves on nv.
- **Overlay over winters 2015–2021 (P&L per winter / worst drawdown, $ per spread):**

  | Rule | Choose (2015–17) | Check (2018–21) |
  |---|---|---|
  | Always short | +2,653 / −3,880 | +1,792 / −13,190 |
  | Flat when ρ_dev is low | identical (never fired) | +2,623 / −2,540 |
  | Twin: CSO vol high | identical | **+3,380 / −2,960** |
  | Twin: hj high | identical | +3,210 / −2,960 |
  | Half size | +1,327 / −1,940 | +896 / −6,595 |

  - All the gain is 2018-19: Nov 2018, ρ_impl falling 0.80 → 0.63 by 9 Nov, before the 13–14 Nov spike (Figure `figures/v5_implied_corr_episodes.png`).
  - In truncated 2021-22 the ρ rule cost $5.2k against always short.
  - ρ_impl was also low (0.67–0.78, ρ_dev ≈ −0.3) before the Jan 2014 polar vortex, but the overlay had no quotes on its decision days that winter.

**Variant count.**
- 108 descriptive CRP cells: 3 horizons × 2 invalid-row rules × 2 targets × 3 groups × 3 windows.
- 900 trade cells: 4 costs × 3 hedges × 5 strategies × 5 entry sets × 3 windows. One selection was made (entry 42).
- 18 indicator regressions, plus 5 overlay rules and 2 twins.

## 4. What it means for a gas volatility desk
- **The cheap leg of a squeeze is correlation.** CSO sellers are paid for a decoupling that realises at about 40% of what is priced, across all adjacent pairs and every tenor. That premium is the source of v4's H−J spread VRP.
- **Sell it naked and small rather than "hedged."** Leg-vol hedges at ρ-neutral ratios add cost and leg-variance noise. Gamma-matched dispersion is unusable at gas ratios (σ_n/nv ≈ 5).
  - The net edge is thin: about $100–400 per 10k-MMBtu straddle, or 10–20% of premium.
  - Execution decides it. A banded hedge, or hedging through the listed spread, is worth about $100 per trade over a daily per-leg hedge.
- **For a long-vol fund (Statar), the result reads in reverse.** Buying CSO spread vol as squeeze protection overpays by about 20–35% of premium on average. Outright leg straddles are the fairer-priced way to be long gas vol. A CSO is worth it only when the squeeze is *specific* (2014, Nov 2018).
- **Don't use ρ_impl as a squeeze alarm.** It is the spread level seen through options, the same conclusion v4 reached for nv.

## 5. Caveats
- **Thin data.** CSO coverage is dense only in 2014–16. After 2016 the quotes are thin: only 24% of CSO daily marks after 2016 are exchange settles at the strike, against 100% in 2014–16. 2013-14 H−J has 2 strikes and 14 usable days. CRP CIs use 4–6 blocks per window.
- **Horizon adjustment.** It shifts ρ by about 0.01, comparable to the median CRP. The mean CRP is robust across R0/R1/R2, but its size is uncertain.
- **Some marks are modelled.** Delta hedging uses the entry IV. Drawdowns use model marks where settles are missing.
- **Costs are assumed, not observed.** 4% for CSO is optimistic in 2013–15.
- **The "correlation trade" is held to expiry.** A vega-managed, rolling version was not tested.
- Overlay and episode conclusions rest on **Nov 2018** (and 2013-14).

## 6. Proposed frozen test for the locked 2022–2026 options sample
Run unchanged on `v5_locked/spreads.csv`, `surface.csv` and futures settles, using `src/research/v5_implied_corr` with `CUT` moved. The code paths are `rho.py`, `crp.py` and `trade.py`.

- **IC-L1 (primary): CRP.**
  - Rows dated 2022-01-01 onward whose CSO expires within the data.
  - Filters: nv ∉ [1.15, 1.21]; 5 ≤ tdays ≤ 126; `rho_impl` ≥ −1, capped at 1 (R0, the panel column, no fitting).
  - Statistic: the pooled mean of ρ_r,life − ρ_impl, with a gas-year block bootstrap.
  - **Pass if the mean > 0 and the 95% lower bound > 0 and the median > 0.** Dev: +0.036 [0.015, 0.061], median 0.008.
  - Secondary: H−J only, mean > 0 (dev +0.063).
- **IC-L2 (tradable).** Strategy A (short ATM CSO straddle, entry at the first quote with 37–42 trading days to expiry, strike nearest S within $0.10, band 0.2 hedge, 4% CSO cost) and strategy D (same plus ρ-neutral leg straddles).
  - **Pass A if the mean net per trade > 0 at 4% and at 8%.** Dev check: +$137 and +$94.
  - **Predict D − A ≤ 0** (dev check: −$226).
  - Report the worst trade and the drawdown per winter.
- **IC-L3 (descriptive).** H−J overlay "flat when ρ_dev < −0.188" (the dev 20th percentile). ρ_dev uses these frozen dev norms of ρ_impl (R1, capped), by tenor bucket:

  | Tenor (trading days) | Frozen norm |
  |---|---|
  | 0–42 | 0.981 |
  | 42–84 | 0.973 |
  | 84–126 | 0.895 |
  | > 126 | 0.960 |

  Compare with twin "flat when nv > 1.038". **The prediction is that the ρ rule does not beat the twin by more than $500 per winter.**

**Live 2026-27 (H27−J27 via G4X Mar-27, option expiry 23 Feb 2027).**
- **Daily log:** ρ_impl = (σ_H² + σ_J² − nv²)/(2σ_H σ_J), with nv the G4X Mar-27 ATM Bachelier straddle vol and σ = F × ON ATM IV for H27/J27 (R0).
- **IC-LIVE1:** G = mean over d ∈ [1 Dec 2026, 22 Jan 2027] of ρ_r,life(d) − min(ρ_impl(d), 1), with ρ_r,life summed from d+1 to 23 Feb 2027. **Pass if G > 0.**
- Log weekly ρ_dev against the norms above, and the nv-twin.

## 7. How to rerun
`python -m src.research.v5_implied_corr.run` takes about 30 seconds after the first build (about 2 minutes cold, 1 CPU).
- Writes `data/processed/research_v5_implied_corr/` (`summary.json`, `step1_*` to `step4_*.csv`, caches `rows.pkl`, `trades_*.pkl`, `hj_weekly.pkl`).
- Writes the three figures `reports/agents/figures/v5_implied_corr_{rho,pnl,episodes}.png`.
- Delete the `*.pkl` files to rebuild.

## 8. New avenues
- **Short CSO strangles vs straddles**, with vega-managed early exit at 50% of premium. *Value: moderate.* The premium is in the wings too, and early exit avoids expiry squeezes.
- **The autumn Oct−Nov / Nov−Dec pairs**, where realised decoupling exceeded implied: a candidate *long* spread-vol trade for a long-vol fund. *Value: moderate*, but it rests on 2–3 autumns.
- **Use G3B/G6B (2- and 6-month CSOs)** to build a ρ term structure. *Value: low to moderate.*
- **IC-L1 on the locked sample** is the cleanest next test: it involves no fitting and no selection. *Value: high.*
