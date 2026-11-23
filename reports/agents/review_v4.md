# Study v4: independent review (agent `reviewer_v4`)

*23 Nov 2026. I used development data only (≤ 2021-12-31). I did not open `v4_locked/`. From the EIA c1–c4 file I read only rows dated 2010-06-07 to 2021-12-31, to validate the mapping. Scratch code is in `src/research/reviewer_v4/`; each script runs in under 2 minutes.*

## 0. Verdict in one paragraph
- **No data leakage found.** The panel's timing is right and, if anything, conservative.
- **Selection leakage is real:**
  - premium_shocks added h = 20 and the up-only rules after seeing the check window;
  - widowmaker added family G after seeing Nov 2018;
  - options headlines the "ex-thin" VRP, which drops the one adverse winter.
- **v4's robust contribution is narrow:**
  - forecast news explains about 5% of daily and about 20% of weekly outright moves;
  - the 00Z archive lags the market by about a day;
  - the weekly forecast-unexplained move in the second month reverses over about 4 weeks;
  - none of this separates 24 Jan 2014 from 14 Nov 2018 ex ante.
- **Not robust:**
  - nothing improves the March−April trade beyond half size;
  - the "split beats raw" margin is not robust either.

## 1. Leakage and timing audit

**Panel (`src/v4/panel.py`, `forecasts.py`).** Clean.

- **Timing checks:**
  - The GEFS lead 0 of the 00Z run on day d is the ET day d (UTC 06/12/18 of d plus 00 of d+1).
  - news_w1..w3 compare the same valid dates between the runs of d and dp.
  - news_w0 uses realised weather for (dp, d) and the d run's lead 0 for d.
  - CPC is the last issue made on or before dp. This is conservative: weekend issues are ignored.
  - Storage enters on its release day.
- **Climatology** is fitted on 2010-10..2021-12 valid dates only.
- **Data completeness:**
  - Every run has lead 15 on every weekday, and n_mem = 5 throughout.
  - Wednesday reforecast news has normal variance: sd of news_w3 is 14.1 on Wednesdays against 14.1–16.1 on other days. The re-collection worked.
- **The timing test that matters:** the lead/lag of front moves on the frozen news index, by weekday of the news (my `leadlag_dow.py`).

  | News day | k = −1 (prior day's move) | k = 0 | k = +1 |
  |---|---|---|---|
  | Monday (Fri 00Z → Mon 00Z) | 0.57 (t 2.3) | **1.74 (t 6.0)** | −0.04 |
  | Wed–Fri | **0.90 (t 4.1)** | 0.51 (t 2.9) | 0.55 (t 2.5) |

  - This is what correct labels plus intraday runs predict:
    - A 00Z-to-00Z revision straddles the dp settle.
    - Midweek, about half of it comes from the dp 06Z/12Z runs, which are public before 14:30 ET.
    - For Monday news, only Friday's intraday runs precede the Friday settle: about 1/3 of a 3-day revision, and the observed ratio is 0.57/1.74 = 0.33.
  - A one-day label error would break this weekday pattern. **Verdict: not a bug.**
  - The design doc's premise that the 12Z run of d is not known at the settle is too strict. The GFS/GEFS 12Z output is normally out before 14:30 ET. The archive is therefore up to about 1.5 cycles stale. That is conservative, not look-ahead.
- **Unflagged issues:**
  1. **2020-01-02 news crosses systems** (reforecast to operational v11).
  2. **Operational v11 (Jan–Apr 2020) has a large cold level bias.** Mean L23 is +9 HDD against realised −26. Operational v12 is about 6 HDD warm. News, which is a difference, is barely affected. **Level rules** (widowmaker E, live 2026-27 on operational v12) are affected.
  3. **The locked panel uses the 2010–21 climatology.** If 2000s winters were colder, the level z-scores there are biased cold. No pre-2010 check is permitted, so the frozen level rule carries this risk.
- **EIA mapping (locked price source), checked on 2010-06..2021-12** (`eia_map_check.py`):
  - The contract mapping is right: 26 of 414k cells differ by more than 1¢.
  - **But the calendars differ:**
    - 47 prev_date mismatches;
    - 26 days where r_second differs by more than 1 percentage point (11 of them in Nov–Feb);
    - a weekly 99th-percentile difference of 3.6 pp.
  - Pre-2010 is unlikely to be cleaner. **Locked tests must drop rows with date − prev_date > 4 calendar days.**

**Walk-forward self-test.** Real but narrow. I reran it and it passes. It verifies the coefficient path (vintage < winter, and vandalising later moves leaves earlier news_part unchanged). It cannot detect feature-timing errors; the weekday test above covers those.

**Frozen coefficients:**
- `frozen_coefs.json` is reproduced exactly by `fit_frozen` on ≤ 2021 data.
- Walk-forward R² reproduces: front 0.050 / 0.038, H−J 0.008 / 0.005.

**Locked-column feasibility.** I restricted the dev panel to the locked column list and ran:
- `decompose`/`decompose_weekly(mode="frozen")` for r_front, r_second and d_hj;
- `hps1`, `hps2`.

All run; news_tot is rebuilt by `_with_tot` if absent. **One gotcha:** in frozen mode `train_mean` is NaN, so H-NC's R² must be defined explicitly (§5).

**No agent reads post-cut or locked rows.** Every loader cuts and asserts. Future-window targets (fut_h, rv20, real_w) become NaN past the cut.

## 2. Multiple testing and episode dependence

| Headline | Variants seen | Survives correction? | Rests on 1–2 episodes? |
|---|---|---|---|
| Front daily news R² 0.050/0.038; weekly 0.19/0.23 | ~100 | **Yes.** Positive in 9 of 10 OOS winters; bootstrap lower bounds > 0 | No |
| H−J weekly scaled news R² 0.26/0.16 | 3 weekly forms, c from daily grid | Weak: check CI [−0.008, 0.54] | Yes: CI width driven by 2014 and 2019 |
| Premium predicts the next 00Z revision (t 6.4) | ~260 | Yes, but it is **mechanical** (§3c) | No |
| Weekly second-month premium reversal at 20d, −0.40 (t −3.0) | ~500, h = 20 post hoc | Bonferroni over 288 regressions: no (needs \|t\| ≈ 3.7). Over a 6-horizon family: borderline | **No.** Leave-one-winter-out −0.35 to −0.46 |
| "Split beats raw" (−0.40 vs −0.22) | same | **No.** β_N − β_P has HAC t −1.75 at 20d and about −1 at 10–15d | – |
| H−J reversal (daily h = 5 −0.23) | same | No: 111% of the numerator from 10 days | **Yes** (2014, Nov 2018) |
| Widowmaker E, R 1.09–1.16 | 520 | No | **Yes: 2013-14 alone** |
| Options VRP +0.22 (ex-thin) | ~95 | The sign count (8/9 winters, p ≈ 0.02) survives. The mean does not: all-winter +0.06 [−0.29, 0.28] | 2014 flips the mean |
| Options protection, kept + cut 1.08 | 40 | No. P(> 1) = 0.64 | Yes (2014, 2018) |

## 3. Agent-specific findings

**(a) news_channel.** The "day before" claim is real (§1), but it is misworded. It is not anticipation: the 00Z-to-00Z revision and the settle-to-settle return cover offset windows. The fix is data (06Z/12Z archives), not modelling.
- Revisions are **negatively** autocorrelated (−0.12 to −0.14), so the k = +1 slope (0.38–0.55) is slow pricing, not revision momentum.
- The episode conclusion (both 2014 and 2018 are more than 95% "premium" daily) is correct and important.

**(b) premium_shocks.**
- **Not mechanical in the bad sense.** It does not hinge on expiry: weeks with the contract at most 27 days from expiry give −0.45, the rest −0.39. Leave-one-winter-out is stable.
- **But the −0.40 vs −0.22 comparison is not independent evidence.** The raw slope is roughly a variance-weighted average of β_P and β_N (news share 19%): 0.8·(−0.40) + 0.2·(+0.49) ≈ −0.22. So "split < raw" is equivalent to "β_N > β_P", and that difference is not significant (t −1.75).
- **The horizon profile is jumpy:** −0.10, −0.10, −0.14, −0.40, −0.40, −0.57 at h = 5, 10, 15, 20, 25, 30. The design's h = 5–10 shows nothing.
- **h = 20 was chosen post hoc.** That is legitimate for a fresh locked test only if disclosed.
- **HAC lags:** 5 weekly lags for a 4-week overlap are fine.
- **The Monday premium persists (+0.29).** The agent's interpretation (unseen news) is correct.
- **No trading rule survives:** the check window gives +$52 per trade, CI spanning 0.

**(c) hidden_info.**
- **A timing artefact, correctly diagnosed by the agent.** Premium_part at d contains the d 06Z/12Z information that the next 00Z run reports. Once the d+1 run is known, it has no skill (t 1.3).
- So H-HI-1 is close to guaranteed on any sample with the same archive. It is a diagnostic, not a finding.
- **The useful result** is the bound: more than 80% of large premium moves are never justified by measurable weather. One caveat: that bound uses the average weekly betas, so in stressed spreads it is an upper bound on "never justified".

**(d) widowmaker.** I reproduced E: R 1.09 (θ 1.0) and 1.16 (θ 0.5).
- E changes only 2 winters. **Without 2013-14, R = 0.99.**
- The gain is $531 off one drawdown (4.5%) plus +$960 in 2013-14, minus $231 in 2010-11.
- **Entry date ±7 days:** R 0.94–1.11. **Nov entry:** 0.98.
- **Grid** (k ∈ {1, 3, 5}, θ ∈ 0–2): 0.99–1.16. With L123: up to 1.30.
- **The proposed pass bar (R ≥ 1.10) fails on dev at the chosen θ = 1.0.**
- It is v3's "enter late" in a new disguise.
- The other conclusions (cold-forecast cuts don't beat half size; B fails at lag 1; D adds tail risk) are sound and well tested.

**(e) options.**
- **VRP:**
  - Horizon matching is consistent: both use calendar-year T, and RV_life sums squared daily ΔS to option expiry.
  - RV20 uses 252-day annualisation against calendar T. That is a ~2% vol mismatch, which is immaterial.
  - Each winter's season mean reuses one future path, so the effective n is 9. The winter bootstrap is right.
  - **The headline is selected:**
    - "Ex-thin" removes 2012-13 (+0.12) and 2013-14 (−1.15).
    - 2013-14's implied vol exists on only 14 days: with strikes 0.25/0.50 only, `atm_iv` needs F inside [0.25, 0.50], so the spike days are missing.
    - Honest summary: **positive in 8 of 9 winters, median about +0.18, all-winter mean +0.06 [−0.29, 0.28], negative in the choose window (−0.07).** That is a short-vol profile, not a demonstrated mean premium.
- **Implied tail (Breeden–Litzenberger 8%):**
  - It is computed from a smile that is linear in strike and **flat beyond the quoted range**, which likely understates upside wings.
  - In 2013-14 the +0.50 strike was never inside the quotes.
  - **"Expiry tails overpriced" is not supported:** with p = 0.08, P(0 of 9) = 0.47.
  - The path comparison (15% vs 2/9) is equally uninformative.
- **Protection:**
  - kept + cut 1.08 at a 4% half-spread; 1.04 at 8%; 0.99 at 15%.
  - Against DD-matched scaling it gains about $100 per winter.
  - It rests on 2014 (call quoted at K = 0.25, later marks extrapolated) and 2018.
  - Read it as **equal to half size within cost uncertainty**.
  - 4% is plausible for 2016+ but optimistic for 2013–15 G4X.

## 4. Verdicts on the proposed frozen tests

| Test | Verdict | Reason / change |
|---|---|---|
| H-NC (news transfer) | **MODIFY → S1** | Define R² explicitly; replace the H−J weekly leg (dominated by 2003) with front weekly; H−J weekly is reported only |
| H-HI-1/2/4 | **DROP as pass/fail; log** | Mechanical timing; near-certain pass; little information |
| H-HI-3 (null as pass) | **DROP** | A pass on non-rejection rewards low power |
| H-PS1 | **MODIFY → primary P1** | Add a calendar guard, pre-declared reporting of β_N − β_P and the h profile, and disclose that h = 20 is post hoc |
| H-PS1b | **KEEP → S2** | The design's h = 5–10 criterion in its honest (controlled, descriptive) form |
| H-PS2 (fade) | **DROP as pass/fail; log** | Dev check +$52 with CI spanning 0; a coin flip |
| H-WM4 (delay E) | **DROP as pass/fail; log** | Fails its own bar on dev; 2013-14 only; level bias risk on locked |
| H-OPT-A | **DROP** | ΔS20-on-level duplicates the premium decay; the MAE20 leg is null-as-pass |
| H-OPT-B (synthetic calls) | **DROP** | Model-on-model (the nv map), and 2003 would dominate |
| T-OPT1 (live VRP) | **KEEP, label as n = 1** | Prior pass rate ≈ 8/9; confirmation only |
| T-OPT2 | **MODIFY** | Entry on 6 Nov predates the freeze. Enter at the 1 Dec 2026 settle instead; descriptive |
| T-OPT3, live E rule, weekly alarm log | **KEEP as logs** | n = 1, no pass/fail |

## 5. Recommended frozen tests

**General rules on the locked panel:**
- Rows Nov–Feb of winters 2000-01..2009-10.
- Before anything else, set r_front, r_second and d_hj to NaN where date − prev_date > 4 calendar days.
- Coefficients from `frozen_coefs.json` (`created 2026-11-23`), unchanged.
- Run each test once.

**P1 (primary): weekly premium reversal, second month**
- `w = decompose_weekly(panel, "r_second", mode="frozen")`, keeping week_end months 11–2.
- Target: fut20 = 100·ln(P_{d+20}/P_d) for the contract that is `second` at Friday d, 20 panel rows ahead (`frozen_test.same_contract_fut`). Drop if missing.
- Regression: fut20 ~ 1 + news_part + premium_part, Newey-West (Bartlett) with 5 lags.
- **Pass:** β_P < 0 with one-sided p < 0.05, **and** t(β_N) > −1.645.
- **Pre-declared reports:**
  - naive slope (fut20 ~ move);
  - **β_P − β_N** from fut20 ~ move + premium_part (HAC t);
  - β_P at h = 5, 10, 15;
  - leave-one-winter-out β_P.
- **Dev reference:** frozen β_P −0.35 (t −3.0), walk-forward −0.40, difference t −1.75.
- **Reading:** pass = the forecast-unexplained weekly move mean-reverts over 4 weeks. Only a significant β_P − β_N supports "the split adds to the raw move".

**S1: decomposition transfer (gate)**
- R² = 1 − Σ(prem − mean prem)² / Σ(move − mean move)².
- Daily: `decompose(panel, "r_front", "frozen")`, in_core rows. **Pass:** R² > 0.02 and winter-block bootstrap (2,000) 2.5% bound > 0. Dev: 0.048 [0.026, 0.076].
- Weekly: `decompose_weekly(..., "r_front", "frozen")`. **Pass:** R² > 0.08. Dev: 0.224.
- Report H−J weekly (dev 0.284) without pass/fail.
- **If S1 fails, read P1 as a raw-move test.**

**S2: controlled daily reversal (design criterion)**
- `frozen_test.hps1b` unchanged: fut_h ~ news + prem + news_{d−1} + Σnews_{d+1..d+h} + Monday, NW lags h + 2, h = 5 and 10.
- **Pass:** β_P < 0 with one-sided p < 0.05 at both h. Dev: −0.24 (t −4.3) and −0.20 (t −2.8).
- **Descriptive:** it uses future news.

**S3: March−April carry on 2000–2010** (the v3 thesis on the locked sample; drop it if the orchestrator has already frozen it)
- Entry: the settle of the first row with date ≥ 1 Dec of Y−1, hj_year == Y and hj present.
- Exit: the last row with hj_year == Y.
- P&L: −(S_exit − S_entry) × 10,000 − $80 (2 ticks per leg per side). Also report $160.
- **Pass:** mean > 0 over the 10 winters.
- **Report:** winter-bootstrap CI, positive-winter count, worst MTM drawdown, half-size row, and the E-rule row (θ = 1.0, scale 36.85) **as a log only**.

**Logged on locked, no pass/fail:** H-HI-1 (timing), H-PS2 (fade versus naive), H-WM4.

**Live 2026-27 (freeze before 1 Dec 2026):**

| ID | Spec | Data | Role |
|---|---|---|---|
| L1 = T-OPT1 | G = mean over d ∈ [1 Dec 2026, 22 Jan 2027] of nv_hj(d) − RV_life(d). nv_hj as `panel.option_block` (ATM straddle, Bachelier, T = calendar days to 23 Feb 2027 / 365.25; days without strikes bracketing F are dropped and counted). RV_life = √(Σ_{t>d..23 Feb} ΔS_t² / T_d). Pass if G > 0; also report the median daily gap | G4X and H27/J27 settles via a Databento pull by the user after 23 Feb 2027; futures cross-checked against Yahoo | Primary live test; n = 1, prior ≈ 85% |
| L2 = T-OPT2′ | Short H27−J27 at the **1 Dec 2026** settle and long the G4X Mar-27 call at K = round₀.₀₅(F + 0.25) at that settle ×1.04. Report the protection cost and the protected/naked worst drawdown | as L1 | Descriptive |
| L3 | Plain 1 Dec short, half size and E rule (θ 1.0, 5-day mean of (lvl_w2 + lvl_w3)/36.85 ≤ θ). Daily path, worst drawdown, and expiry on 24 Feb 2027 | Yahoo H27/J27; GEFS operational 00Z (free, AWS) through the same `forecasts.py`. Note operational v12 is ~6 HDD warm against the reforecast climatology | Log |
| L4 | Weekly Friday log of `decompose_weekly` news/premium for second month and H−J, plus fut20 | Yahoo and GEFS | Log (about 12 weeks; no power) |
| L5 | T-OPT3 implied-tail log | Databento | Log |

## 6. What v4 robustly shows, and what is overstated

**Robust:**
1. **GEFS 00Z ensemble-mean news explains about 5% of daily and about 20% of weekly outright moves out of sample.**
   - The 11–15-day window carries the most.
   - CPC adds nothing.
   - H−J daily moves are about 99% unexplained.
2. **The free 00Z archive trails the market by about one model cycle.** Half of a midweek revision is priced at the previous settle, and a third of a weekend revision. "Premium" therefore contains early news, which biases daily reversal tests toward zero.
3. **The weekly forecast-unexplained second-month move partially reverses over about 4 weeks:** −0.40, stable when any winter is left out. The raw weekly move also reverses (−0.22). That the split *adds* is suggestive only.
4. **No ex-ante observable tells 24 Jan 2014 from 14 Nov 2018.** Not news, implied vol, forecast level or tightness. Ex post, 2014 was partly real cold; Nov 2018 was premium.
5. **Nothing beats half size for the H−J short.** That covers forecast cuts, delays, add-ons and OTM calls. The calls tie it within cost uncertainty. Spread implied vol exceeds realised in 8 of 9 winters, while outright vol is fair.

**Overstated:**
- **"Prices move the day before the revision"** is not anticipation. It is window misalignment.
- **"The weekly split sharpens reversal (−0.40 vs −0.22)":** the gap equals β_N − β_P, which has t −1.75. The horizon was chosen post hoc.
- **"VRP +0.22 [0.11, 0.33]"** excludes the adverse winter. The all-winter mean is +0.06 with a CI spanning 0.
- **"Expiry tails are overpriced (8% vs 0/11)":** 0/9 is consistent with 8% (p = 0.47).
- **"Protection works" / "the one overlay above scaling":** it beats scaling by about $100 per winter at an assumed 4% cost, on 2 episodes.
- **"E is the only robust cell":** it is 2013-14 alone (R 0.99 without it), and it fails its own 1.10 bar.
- **Down-moves "persist":** this is the winter drift, not evidence about premium.
