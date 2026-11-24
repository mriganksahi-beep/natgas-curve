# Review of study v5 (natural gas options): independent reviewer

*24 Nov 2026. I used development data only (≤ 2021-12-31). No `*_locked` folder and no raw option row after 2021 was opened. Scratch code is in `src/research/reviewer_v5/`: `chk*.py` and `frozen_ref.py`. Outputs are in `data/processed/research_reviewer_v5/`.*

## 0. Verdict in one paragraph
v5 is mostly honest null work. Three of the five agents correctly report "no edge". But **three headline numbers do not survive audit.**
1. **The weekly EIA straddle premium (greeks S2, +$225) is a data bug.** The weekly panel's expiry is set to the last date with a settle. For about 70% of series that date is a day before the true expiry.
2. **greeks P1 passes its check window only because of a day-of-week artefact** in the panel's theta attribution.
3. **The short CSO straddle's net edge (implied_corr, +$137 check) comes from entries on CSO marks with zero open interest.** On genuine quotes it is about zero.

What survives:
- CSO spread vol is rich *at the median*.
- Outright vol is fair on average.
- Winter vol is rich when storage is loose.
- Everything that is "cheap vol when tight" or "ρ as an alarm" rests on 2014 and 2018-19.

## 1. Shared panels: audit

| Item | Finding | Severity |
|---|---|---|
| ON expiry = the trading day before futures LTD (`days[p-4]`) | **Correct.** On the raw file's last ON date (1 day before the panel expiry for 116/137 contracts) the ATM straddle still holds about 5.8¢ of time value. On the panel expiry day (15 contracts) it is at intrinsic. The raw file simply lacks most expiry-day settles. G4X behaves the same, so CSOs and ON options expire together. | OK |
| **Weekly expiry = last settle date** | **Wrong for most series.** On the "expiry" day the ATM straddle still carries a median 5–7¢ (2014–16, mostly Thursdays) or 10–12¢ (2017–21, mostly Fridays) of time value. 69% of greeks' S2 trades are affected. As a result `tdays` is too small by at least 1, short-dated weekly IV is inflated (about √2 at n = 1), and any hold-to-"expiry" trade skips the last day. | **Critical** for `weekly.csv` |
| T = trading days/252; theta per calendar day | **Inconsistent clock in `dh_straddle`.** `theta_pnl = theta0 × calendar days`, but `iv1` is solved on a trading-day clock. Mean `resid/P0` is −0.29% on 1-day holds and **+0.87% across weekends** (+1.5% on 4-day spans). Multi-day sums are roughly unbiased; daily residuals are not. | **High** for any daily use of `resid` |
| Black-76 with r = 0 | The "< ~1% of premium" bias holds for 2010–21 rates. In 2023–25 (4–5% rates) a 12-month IV is biased low by about 5% relative, roughly 2 vol pts. Any locked test of IV − RV should keep T ≤ 126 trading days (bias ≤ 1 vol pt, against the hypothesis). | Medium (locked) |
| OTM-only IV, ATM interpolation | Sound. The docstring says "straddle at two strikes", but the code interpolates OTM IVs in k; that is harmless. Flat smiles occur in < 0.2% of rows and stale ATM IVs in < 0.05%. | OK |
| Far-dated ON settles | Not pinned like CSOs, but they are **marks**: zero near-money volume in 50% of rows at 6–12 months and 83% at 12–18 months. `P1 == P0` despite \|ΔF\| > 5¢ in 2.4% of `dh` rows. This affects vol_rv's 6–18-month VRP cells and the "vega roll-up" (+0.098), not the ≤ 130-day books. | Medium |
| CSO Bachelier vol and ρ with the far-horizon mismatch | implied_corr handles it well (R0/R1/R2). **The default filter is fragile.** The "nv ∈ [1.15, 1.21]" rows (9,251 = 65.8%; the report's 8,196 is inconsistent) have zero CSO OI 99.6% of the time. Real CSO vol exceeds 1.0 in 8% of near-dated rows and will often fall in that band at 2022 price levels. **Use OI > 0 (lagged one day), not an nv band.** | **High** for locked tests |
| Pre-2016 OI/volume date inference; dev cut | Not material. One `dh` row has `date1` = 2022-01-03, an allowed outcome. Agents lag flows by one day. | OK |
| `weather_unc` | Normalised with dev-wide medians, a constant per lead, so the leak is negligible. 00Z on day d is available before the settle. | OK |
| `hi_bid`/`lo_offer` | When not crossed, these are a **lower** bound on the instantaneous spread, not an upper bound as the brief says. Non-crossed half-widths: G4X 3.5–6.5% at normal premiums, ≥ 28% below 1¢; weeklies ≥ 20% (n = 1,237, selected). G4X has both quotes on only 970 of 2.1M rows. | Costs (Section 3) |

## 2. Leakage and look-ahead
- **No agent opened locked data.** 2022 futures settles enter only as outcomes of positions opened in 2021 (implied_storage H22/J22 in Jan–Feb; vol_rv and weather RV to 31 Mar 2022), as the brief allows. No 2022 option settle is read, apart from the one allowed `dh` row.
- **Walk-forward claims hold** on inspection: implied_corr q̂ (completed contracts only), vol_rv HAR/GARCH (monthly refits on resolved targets), greeks z-scores and PCA, weather's expanding thresholds, and storage at week_ending + 6 days. greeks picks its trade direction on the choose window, so only its check window is honest.
- **Model 2 and P-lite are as-of-origin** and conservative (F against a price at F+7). The issue is **centring** (Section 4), not leakage.
- **Frozen constants** (weather, implied_corr ρ norms, implied_storage's 2022 map) are fitted on ≤ 2021 data, which is legitimate.

## 3. Settlement price versus tradable price
- **Short CSO straddle (implied_corr A).** I merged the trade cache with CSO OI on the entry day:

  | Entry quote | Window | Mean P&L per trade | n |
  |---|---|---|---|
  | **Zero OI** (no-trade marks) | choose | **+$974** | 31 |
  | Zero OI | check | **+$876** | 11 |
  | OI > 0 | choose | +$80 | 55 |
  | OI > 0 | check | +$52 | 109 |

  - Pooled, the OI > 0 trades earn **+$61, t 1.1**.
  - At entry 42 the OI > 0 subsets earn **−$55 (choose)** and **+$64 (check)**.
  - My panel-priced reconstruction (ATM at nv, K = S, entry at 37–42 days with lagged OI > 0, daily spread hedge, 4%) gives **−$38 per trade [−298, 186]**, n = 48; −$85 at 8%. The 2014 trade lost −$3.8k and the trades entered in Nov–Dec 2021 lost −$2.0k.
  - **The tradable CSO edge is not established.**
- **G4X March call spreads (implied_storage T1).**
  - Strikes priced off the fitted density (34% of K₁) earn +$364 (check) and +$414 (choose). Quoted strikes earn **+$147 and +$41**.
  - In the check window gross P&L equals premium on average: **no trade paid out in 2017–21**. The "+$231" is premium collected in five benign winters.
  - The breakeven extra half-spread is 0.7–1.1¢ per leg. The lower bound above (≥ 6.5% on 1–3¢ premiums) leaves it positive but thin.
- **Weekly straddles (greeks S2).** I re-scored them with expiry set to the next trading day wherever the last settle still holds time value:
  - report spans 2014–16: **+$60 [−50, 117]**, against the claimed +$225 [119, 296];
  - 2017–21: **−$180**.

  Spreads on weeklies are ≥ 20% of premium where observable, not 3%. **DROP.**
- **Monthly ATM (vol_rv, weather).** 1.5% is plausible at the money (non-crossed lower bound 1.9–2.9%). Both agents already find costs binding, so their nulls are robust to this.

## 4. Multiple testing and episode dependence

| Agent (cells) | What survives honestly | Episode-dependent or artefact |
|---|---|---|
| implied_corr (~1,030) | CSO vol rich in vol terms: ln(nv/RV_S) **+0.33 [0.25, 0.44]**, positive in 9/10 gas years. CRP > 0, but **+0.018 [0.003, 0.046]** with OI > 0 (half the headline), and **≈ 0.002 when ρ_impl > 0.95**: all of the mean comes from the 22% of rows with ρ_impl ≤ 0.95. | The trade edge (zero-OI marks). The ρ overlay rests on Nov 2018 only. |
| implied_storage (~110) | P&L is explained by the spread level (t 10.7), not the gap (t 0.19). This null is robust. | **S1 depends on centring.** Against the *raw* P-lite (the best Brier score, 0.082) the check-window gap at −400 is **+0.035 [−0.03, 0.10]**. The headline +0.21 uses the worse-calibrated centred model, and choose and check swap order under raw. Calibration rests on one event winter. |
| vol_rv (~200) | No unconditional VRP: −0.13 [−2.7, 2.2] at 10–126 days. IV is the best life forecaster. Loose storage makes winter IV rich, **+4.2 [2.7, 6.6], 7/7 gas years**. The term spread predicts relative RV. | Tight storage −10.4: it falls to **−3.7** without the Q1-2014 and 2018-19 winter contracts. T-VRV3 is post hoc and rests on about 3 spells. |
| greeks (~650) | The nulls (skew lags futures; no pinning; flows dead), with 0 survivors under Holm. | **P1:** adding a weekend dummy moves the check t from 1.69/1.78 to **0.53/−1.06** (raw/winsorised); all-sample winsorised t is 0.66. Choose survives (3.46/2.40), but that is 2014/2018. S2 is the expiry bug. |
| weather (~150) | A clean null, with 0 survivors under Holm. | The Q4 cells are 2014 vs Nov 2018. The "low disagreement means sell" cell is post hoc. |

## 5. CRP, storage premium and storage-regime vol: one fact or three?
They reduce to **two facts and one mirror image.**
- **Fact A: options price a roughly constant squeeze risk across storage states.** When storage is loose, winter outright IV is rich (+4.2) and the H−J upside tail is rich. implied_storage's gap is largest in 2016-17, 2017-18 and 2020-21 and near 0 in 2014-15, 2015-16 and 2018-19, and its P&L loads on the spread level. When storage is tight, realised legs overrun implied.
- **The CRP is largely the tight half of Fact A seen through a ratio.**
  - With OI > 0, CRP is −0.008 in loose storage, +0.023 in normal and **+0.072 in tight**, driven by gas years 2014 (+0.20) and 2019 (+0.13).
  - When leg vols are under-priced and the CSO is not, ρ_impl is mechanically too low. The agent's own variance decomposition shows this: spread variance premium ≈ 0 (I get −0.005 [−0.16, 0.11] with OI > 0), leg terms negative, correlation term positive.
  - "Correlation premium" is therefore a misleading label.
- **Fact B: CSO spread vol is rich at the median** (realised/implied ≈ 0.67) but about fair in variance. It is a short-tail profile and, after costs on genuine quotes, not monetisable in dev.

## 6. Proposed frozen tests: KEEP / MODIFY / DROP
| Agent | Verdicts |
|---|---|
| implied_corr | **IC-L1 MODIFY:** lagged CSO OI > 0 replaces the nv band; the vol-ratio form becomes the Primary. **IC-L2 MODIFY → S2:** OI > 0 at entry, panel pricing, honest prior ≈ 0; D − A becomes a log. **IC-L3 DROP** (Nov 2018 only). **IC-LIVE1 KEEP as a log** (duplicates v4 L1). |
| implied_storage | **S1 DROP as pass/fail** (depends on centring); log raw and centred gaps. **S2 DROP.** **T1 KEEP as a log:** needs raw G4X strikes, so not panel-only; a low-information "no squeeze" test. **T2, C1 and live 2026-27 KEEP as logs** (live uses the frozen 2022 map). |
| vol_rv | **T-VRV1 MODIFY → S1:** loose leg scored, ≤ 126 trading days because of r = 0, tight reported only. **T-VRV2 MODIFY → S3** (outright no-VRP); S_ewma book becomes a log. **T-VRV3 DROP** (post hoc). **T-VRV4 KEEP as a log.** `data.py` hard-codes `V5`/`CUT` with asserts, so it does **not** run unchanged on locked data. |
| greeks | **P1 DROP** (artefact); if logged at all, recompute `resid` with θ × 365.25/252 per trading day and add a weekend dummy. **S1 DROP.** **S2 DROP** (weekly expiry bug). Rebuilding `weekly.csv` from the CME expiry calendar is a prerequisite for any weekly work. |
| weather | **T-WU1 KEEP as a logged falsification** (runner supports `V5_DIR`/`V5_CUT`). **T-WU2, T-WU3 logs.** **T-WU4 DROP** (fails on dev). |

- **Cache hazard.** implied_corr (`rows.pkl`, `trades_*.pkl`) and vol_rv reuse dev caches silently if they are rerun with a moved CUT. Any locked rerun must use a fresh cache directory.

## 7. Recommended final frozen set (locked 2022–2026)
**Runner:** `python -m src.research.reviewer_v5.frozen_ref data/processed/v5_locked 2022-01-01 <panel_end>`. The dev reference is `data/processed/research_reviewer_v5/frozen_dev_reference.txt`, produced with no arguments.

**Inputs:**
- `spreads.csv` columns {date, cid, T, nv, cso_call_oi, cso_put_oi};
- `surface.csv` columns {date, cid, T, atm_iv};
- `data/raw/ng_curve_daily.csv`;
- `data/raw/storage.csv`.

**Common rules:**
- Expiry = `days[p−4]`.
- Rows are used only if their option expires within the futures data.
- Gas-year (Apr–Mar) block bootstrap, 4,000 draws, seed 0.
- "Not testable" if fewer than 3 gas-year blocks or fewer than 200 rows.

| ID | Spec | Pass rule | Dev reference |
|---|---|---|---|
| **P (primary): CSO spread vol rich** | Rows with 5 ≤ round(252T) ≤ 126, nv > 0, and the **previous panel row's** cso_call_oi + cso_put_oi > 0 (same cid). Statistic: L = ln(nv / RV_S), where RV_S = √(Σ ΔS² / (n/252)) over (d, expiry], S = F_cid − F_cid+1, n ≥ 5. | Pooled mean > 0 **and** 95% lower bound > 0; also report the median, H−J only, and each gas year | **+0.328 [0.245, 0.440]**, median 0.40; 9/10 gas years > 0 (2019: −0.23); choose 0.32, check 0.34 |
| **S1: loose storage ⇒ winter IV rich** | Delivery Dec–Mar, round(252T) ≤ 126. sdev = latest storage (week_ending + 6 days) minus the mean of the nearest same-date levels 1–5 years earlier (≥ 2 available). Loose means sdev > +150. Statistic: mean of 100·(atm_iv − RV_life), where RV_life = √(252·mean r²) over (d, expiry]. | Mean > 0 and lower bound > 0. Report the tight-storage mean (< −150) without scoring it. **Not testable** if fewer than 2 loose gas years | **+3.6 [0.5, 7.0]**, 6/7 gas years (full tenor +4.2 [2.7, 6.6]); tight −14.1 (episodic) |
| **S2: tradable CSO (expect a fail)** | For each cid, the first row with 37 ≤ round(252T) ≤ 42 and lagged OI > 0. Short a Bachelier ATM straddle (K = S₀, premium = nv√(2T/π)). Delta-hedge daily in the futures spread at the entry nv, $0.002 per unit; 4% of premium; $0.002 if exercised; held to expiry. | Mean net > 0 **and** lower bound > 0 (a tradable edge). Also report the 8% cost, the worst trade and the D − A log | **−$38 [−298, 186]**, n = 48: **fails** on dev |
| **S3: outright vol fairly priced** | All deliveries, 10 ≤ round(252T) ≤ 126, mean of 100·(atm_iv − RV_life) | "Fair" holds if the 95% CI includes 0 **and** \|mean\| < 2.0 | −0.13 [−2.68, 2.24]; choose +0.44, check −0.92 |

**Interpretation, written down before the run.**
- If P passes and S2 fails, the CSO premium is real but sits inside the cost and tail budget.
- For S1, 2022 rates bias it against passing by ≤ 1 vol pt.
- Report winter 2021-22 and the 2022 Ukraine-shock months separately, as a diagnostic only.

**Live 2026-27.** These need the user's Databento pulls after the 23 Feb 2027 expiry and EIA data.
- **L-P:** P restricted to G4X Mar-27 (H27−J27), daily 1 Dec – 22 Jan, using lagged OI > 0. Pass if the mean L > 0. This is the same object as v4's L1; run both.
- **L-S1:** record sdev at the 1 Dec 2026 report. If it exceeds +150, score the Z26/F27/G27/H27 mean IV − RV_life (≤ 126 days) > 0. Otherwise log it as "not loose".
- **Logs:** implied_storage's live Q₄₀₀ with the frozen map; weather T-WU1.

## 8. Robust v5 conclusions, honestly worded
1. **Outright monthly NG vol is fairly priced on average.** IV − RV_life is −0.1 [−2.7, 2.2] vol pts, and the short delta-hedged ATM straddle earns nothing gross. For a long-vol fund, owning outright gamma is not systematically expensive. That is consistent with v4.
2. **Winter vol is rich when storage is loose** (+4 vol pts, 7/7 gas years). The opposite claim, cheap when tight, rests on 2014 and 2018-19 and is a sizing hint at most.
3. **1-month CSO spread vol is rich at the median.** Realised is about 0.67× implied, positive in 9/10 gas years. In variance terms it is about fair, and a costed short on genuinely open contracts did not make money in development. The "correlation risk premium" is largely the same fact, plus under-priced legs in squeezes.
4. **The option surface does not lead futures.** Skew, flows, gamma, vega shocks and GEFS disagreement give 0 Holm survivors across roughly 800 tests. IV already contains what ensemble spread knows about future RV.
5. **Costs and hedging dominate every systematic vol book tested.** Option half-spreads plus daily hedging cost about 3 vol pts per month, against gross edges of 0.5–0.8.

**Overstated:**
- the weekly EIA premium (a bug);
- greeks P1 "replicates" (a weekend artefact);
- the short CSO straddle's net edge (zero-OI marks);
- the CRP's magnitude (halves with OI > 0, and concentrated in ρ_impl ≤ 0.95);
- implied_storage's "options price 359 Bcf below physics" (driven by centring; the raw gap is insignificant in check);
- vol_rv's tight-storage −10 vol pts (two episodes);
- the brief's "`hi_bid`/`lo_offer` is an upper bound" (when not crossed it is a lower bound, which makes the weekly and CSO-wing costs worse than assumed).
