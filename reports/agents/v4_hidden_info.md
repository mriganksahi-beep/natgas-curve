# v4 agent `hidden_info`: is the unexplained part of curve moves hidden weather news?

*23 Nov 2026. Development data only: dates ≤ 2021-12-31, and realised-weather targets are dropped when their 15-day window ends after the cut. The news/premium split is agent 1's frozen walk-forward `decompose()` (cached `decomposed_dev.csv`, `decomposed_weekly_dev.csv`), so premium_part exists from winter 2012-13. Sample: November−February core days. Choice window 2012-13..2016-17 (5 winters); check window 2017-18..Dec 2021 (5 winters). The regressions are descriptive (pooled, with HAC and winter-block bootstrap CIs). Incremental predictive R² is walk-forward (expanding by winter, at least 2 training winters). Nothing was tuned.*

## 1. Plain-English findings
- **Part of the "premium" is weather information that arrives early.** A day's unexplained move predicts the next GEFS 00Z revisions and the realised weather beyond the forecast at d. The sign is right in 10 of 10 winters for the front month and in 9–10 of 10 for March−April.
  - Front month: +5.6 HDD of next-run revision per s.d. of premium (HAC t 6.4) and +5.3 HDD of realised surprise (t 4.5).
- **But it is a timing advantage, not better forecasting.** Once the next day's 00Z run is known, premium_part says nothing more about realised weather: t 1.3 front, 1.1 second, 1.1 March−April. The market sees the 06Z/12Z/ECMWF runs about a day before our archive does. It does not out-forecast the models.
- **The amount is small.** Future GEFS revisions plus realised surprises explain about 6% of the daily premium variance for outrights (3% leave-one-winter-out) and 3% for March−April (≈0% leave-one-winter-out). On big days (top decile of |premium|), only **14–18% of the move is later justified** by GEFS revisions over the next 5 days, and **6–9% by realised weather**. The placebo 95th percentile is 3–5%. **More than 80% of large premium moves is never justified by weather we can measure.**
- **Hidden news masks reversal.** Taking out the anticipated-news part (a diagnostic, because it uses future data) roughly doubles the 5-day reversal of premium:
  - front: −0.08 (t −1.0) → −0.17 (t −2.3);
  - March−April: −0.23 (t −1.9) → −0.25 (t −2.2).
  - The anticipated part keeps going instead (+1.4 to +1.9 per unit).
- **No ex-ante observable separates hidden-news days from premium days.** Implied-vol jump, GEFS level, tightness, CPC, Monday, "no GEFS news that day" and move size were all tried. Every AUC flips or sits near 0.5 between windows. The one consistent hint (moves *against* that day's GEFS news are more often later justified, AUC 0.31–0.45 in both windows) is weak and was found post hoc. No tradable "pure premium" proxy beats raw premium_part (correlation with the diagnostic pure premium: 0.97 for the proxy, 0.97 for raw).
- **Asymmetry is the opposite of the hypothesis, but not significant.** Cold-scare (up) moves are *more* often partly justified than down moves:
  - realised-weather share 15–18% up against 1–2.5% down; the up-minus-down CIs include 0;
  - in the slope test, up-premium predicts realised cold (front t 3.7), while down-premium does not predict warmth (t 0.3).
  - Down moves look like pure premium decay.

## 2. Method
- **Prediction (Test 1).** y = a + c′·GEFS-at-d + β·(P_d / sd(P)) + e.
  - P is premium_part (daily), or its d−4..d sum P5.
  - **Targets:**
    - future revisions FN_k = Σ_{j=1..k} news_tot_{d+j} (k = 1, 3, 5), plus the w1 and w3 parts of FN5;
    - realised real_w1, real_w2, real_w3 and real_tot;
    - real_tot over the *d+1 run's* windows, given the *d+1 run's* levels ("beyond the next run").
  - **Controls:** lvl_w1..w3. For revision targets, also news_w0..w3 at d and news_tot at d−1.
  - **Errors:** HAC lags of 5 (revisions) and 15 (realised). Winter-block bootstrap with 1,000 draws.
  - **Weekly version:** agent 1's Friday-to-Friday premium. Targets are next week's revisions and the realised weather from Friday.
- **Share (Test 2, DIAGNOSIS ONLY).** P_d is projected, leave-one-winter-out, on four blocks:
  - L: news at d−1 (late reaction);
  - F1: the d+1 run's news by window;
  - F23: the d+2 and d+3 runs;
  - R: realised surprise real_wk − lvl_wk.
  - **pure_premium_DIAG** = P − projection. **anticipated** = the F+R part. Both use future information and are not tradable.
  - The ex-ante proxy is a walk-forward fit of "anticipated" on P interacted with ex-ante state (Monday, |news_d|, lvl_tot, |P|, hj, Δiv/Δnv).
- **Episodes and big days (Tests 3–4).** Ex-post justification J converts HDD into price using agent 1's frozen **weekly** betas, because the daily betas are attenuated by the timing smear:
  - front/second: J = Σ_k b_k·X_wk;
  - March−April: J = 0.0047·(max(S_dp, 0) + 0.10)·X_tot;
  - X is either the later GEFS revisions (FN5) or the realised surprise.
  - **Class:** hidden_news if max(J/P) ≥ 0.5; partial if 0.2–0.5; premium otherwise.
  - **Dollar-weighted share** = Σ sign(P)·J / Σ|P|. The placebo pairs each big day with J from a day 20–195 rows away.
  - **Retention** = 1 + (move over d+1..d+10 − drift) / move.

## 3. Results
**Table 1. Does premium_part at d predict later weather information?** Daily, November−February, 2012-13..Dec 2021. β is in HDD per s.d. of P, with the bootstrap CI. ΔR² is in-sample, then walk-forward OOS for the choice / check windows. Figure: `figures/v4_hidden_info_predict.png`.

| Target (controls) | Front β [CI], t | ΔR² in / OOS ch / ck | 2nd β, t | March−April β [CI], t | ΔR² in / OOS ch / ck |
|---|---|---|---|---|---|
| Next run news_tot d+1 | **5.6 [3.8, 7.4], 6.4** | .054 / −.013 / **.085** | 5.1, 5.7 | **3.3 [2.7, 9.8], 3.0** | .019 / .013 / .029 |
| Runs d+1..d+5 (FN5) | 7.3 [4.6, 10.3], 4.6 | .021 / −.003 / .031 | 6.2, 4.2 | 6.0 [2.9, 11.8], 2.3 | .014 / .022 / −.008 |
| of which w3 (11–15 days) | 4.4, 4.6 | .019 / .014 / .022 | 3.9, 4.2 | 3.4, 2.6 | .011 / .013 / .005 |
| of which w1 (1–5 days) | 0.6, 1.3 | .002 | 0.6, 1.1 | 0.2, 0.5 | .000 |
| Realised real_tot (lvl at d) | **5.3 [3.1, 7.4], 4.5** | .006 / −.002 / .013 | 4.9, 4.4 | **4.0 [2.2, 8.1], 2.1** | .004 / .003 / −.001 |
| Realised real_tot (lvl at **d+1**) | 1.6 [−0.4, 3.8], 1.3 | .001 / −.001 / .001 | 1.3, 1.1 | 2.1 [−1.2, 5.9], 1.1 | .001 / .001 / −.007 |
| P5 (d−4..d) → FN5 | 5.7 [−2.3, 12.5], 2.2 | .012 / −.008 / −.011 | 4.7, 1.9 | 5.9 [−3.7, 11.0], 2.3 | .013 / −.025 / .008 |

- **Scale:** the s.d. of FN5 is 43 HDD and of real_tot − lvl_tot is 31 HDD. Per-winter β is positive in 10/10 winters (front: FN1, FN5, real_tot) and 10/10, 9/10, 9/10 (March−April).
- **Robustness:**
  - Reforecast-only rows (≤ 2019): front FN1 t 5.2, real_tot t 3.5.
  - The frozen-coefficient decomposition (the locked-test mode) on dev: front FN1 β 5.1 (t 6.6), real_tot 4.7 (t 4.6), beyond the next run 1.4 (t 1.3).
- **OOS gain is tiny,** and negative in the choice window. This is a description of where information sits, not a forecasting tool.
- **Weekly:** premium does **not** predict next week's revisions (front t 0.7; March−April t 1.6). It weakly predicts realised weather: real_tot t 2.3 (front), 2.4 (March−April); bootstrap lower bounds 0.25 and 0.01. The anticipation lives inside the week, which agent 1's weekly news_part already absorbs.

**Table 2. Share of daily premium variance that is anticipated news** (`t2_share.csv`; DIAGNOSIS)

| Blocks | Front R² in [CI] / LOWO | March−April R² in [CI] / LOWO |
|---|---|---|
| L (news d−1) | .014 / .002 | .012 / −.001 |
| F1 (next run) | .054 [.026, .092] / .040 | .018 / .008 |
| F1-3 + R | .061 [.038, .111] / .031 | .028 [.023, .067] / −.005 |
| L + F1-3 + R | .073 / .031 | .039 / −.005 |
| Weekly F + R | .077 / −.015 | .093 / −.066 |

- **Variance split** of front premium: 1.4% late news, 6.0% anticipated news, 96.7% pure premium (the components covary). March−April: 1.2%, 3.0%, ≈100%.

**Reversal over d+1..d+5,** controlling for news_part_d (HAC t; Fig. 2 right):

| Instrument | Raw P | Pure (diag.) | Anticipated (diag.) |
|---|---|---|---|
| Front | −0.08 (−1.0) | −0.17 (−2.3) | +1.88 (5.1) |
| March−April | −0.23 (−1.9) | −0.25 (−2.2) | +1.38 (2.9) |

- **The ex-ante proxy fails.** Walk-forward R² for predicting "anticipated" is −0.09 / +0.02 (front, choose / check) against −0.05 / +0.04 for plain scaling c·P. Its 5-day reversal slope is the same as raw P.

**Table 3. Episodes** (`t3_episodes.csv`). fracF = share justified by GEFS revisions d+1..d+5; fracR = share justified by realised weather; retain10 is drift-adjusted (1 = kept, 0 = reversed). The front month is in brackets.

| Date | P (March−April $) | fracF / fracR (front) | Class (front) | Retain10 March−April (front) | Ex ante: lvl_tot (HDD), Δiv |
|---|---|---|---|---|---|
| **2014-01-24** | +0.29 | .13/.21 (.10/.26) | partial (partial) | −0.01 (0.79) | 105, +0.60 |
| 2014-01-29 | +0.36 | .20/.36 (.17/.33) | partial | −0.67 (0.02) | 99 |
| 2014-02-04 | +0.31 | −.25/.37 | partial | 2.66 (2.76) | 80 |
| 2014-02-13 | +0.32 | .82/.61 (1.4/1.1) | **hidden news** | n/a (1.82) | −55 |
| 2014-02-19 | +0.40 | 1.34/.58 | **hidden news** | n/a | 58 |
| 2014-02-25 | −0.42 | −.34/.34 | partial | n/a | 140 |
| **2018-11-13** | +0.39 | .13/−.03 | premium (partial) | 2.14 (2.90) | 86, Δnv +0.9 |
| **2018-11-14** | +0.71 | .14/.05 (.12/.02) | **premium** | 0.59 (0.84) | 66, Δnv +1.8 |
| **2018-11-15** | −0.78 | −.31/−.11 | **premium** | 0.55 (0.18) | 54 |
| 2021-02-01 (Uri run-up; largest front day) | +0.05 | 1.07/.72 (.77/.60) | **hidden news** | 3.5 (2.1) | 35 |
| 2021-02-12 / 02-16 (Uri) | +0.01 / +0.11 | ≈0 | premium | n/a | 59 / 21 |
| 2021-10-05 / 10-06 | +0.34 / −0.36 | ≈0 | premium | −0.94 / 1.61 | −62 / −57 |

- **24 Jan 2014** was a partial news day:
  - the realised weather came in 33 HDD (about 1 s.d.) colder than the very cold forecast, which justifies about a fifth to a quarter of the move;
  - the spread then whipsawed: cumulative +0.29, +0.10, +0.36, +0.08 at 1, 2, 5 and 10 days, and +0.83 at 20 days;
  - its "persistence" is the rest of the winter's cold arriving, much of it as later GEFS news.
- **14–15 Nov 2018** was pure premium:
  - forecast cold levels were similar (66–86 HDD against 105) and realised weather matched the forecast (surprise −5 to +11 HDD);
  - **On the day, the two looked the same:** both had very cold forecast levels, no GEFS news and implied-vol jumps.
- **Uri (Feb 2021)** was barely a March−April event. The front month's big day came on 1 Feb and was hidden news (the cold arrived in the next runs).

**Big days and asymmetry** (top decile of |P|, 75–76 days per instrument; `t3_dollar_share.csv`, `t4_asym.csv`, `t4_slopes.csv`; Fig. 2 left):

| Instrument | Share justified by later GEFS, all / up / down | Share justified by realised, all / up / down | Placebo p95 (F/R) | "Hidden news" class, up / down |
|---|---|---|---|---|
| Front | .17 [.08, .25] / .23 / .12 | .09 [.01, .15] / .17 / .03 | .05/.03 | 27% / 28% |
| Second | .14 / .23 / .09 | .06 / .15 / .01 | .04/.05 | 31% / 26% |
| March−April | .18 [.11, .21] / .20 / .16 | .09 [−.02, .19] / .18 / .02 | .05/.03 | 39% / 40% |

- Every up-minus-down CI spans 0 (front realised: [−0.15, 0.49]).
- In the slope regressions, up-premium predicts realised cold, while down-premium does not predict warmth:
  - front: t 3.7 against 0.3;
  - March−April: t 1.9 against 0.3.
- **Ex-ante separation** (AUC for hidden_news against the rest among big days; `t3_auc.csv`; choose / check):
  - move size: 0.49 / 0.44 (front); 0.42 / 0.57 (March−April);
  - Δiv: 0.49 / 0.35; Δnv: 0.31 / 0.61;
  - hj: 0.68 / 0.32;
  - CPC: 0.50 / 0.63;
  - Monday: 0.42 / 0.61;
  - no GEFS news: 0.60 / 0.50;
  - same-day news against the move: 0.32 / 0.31 (front), 0.45 / 0.44 (March−April). This is the only feature consistent in both windows.

## 4. What this tells us
- The premium part is **mostly premium**. A thin, statistically clear layer of early weather news sits on top of it. That layer is the GEFS (and other model) runs the market sees roughly a day before our 00Z archive, not superior skill.
- That layer works against reversal tests: the anticipated part keeps going while the rest reverses. Agent 3's mandatory controls (future news parts over d+1..d+h, news at d−1) are therefore correct and sufficient. The leakage is concentrated in the next 1–3 runs.
- For the widowmaker, the ex-post classification separates 2014 (partly real cold, 20–60% justified on its spike days) from Nov 2018 (≈0% justified). **Nothing observable on the day did.** Weekly news_part remains the only ex-ante handle.

## 5. Caveats
- **Variant count, about 260 cells, none tuned:**
  - Test 1: 120 daily (60 main + 60 reforecast-only), 21 weekly, 12 asymmetry, 12 frozen-mode check;
  - Test 2: 27 share, 12 reversal, 3 proxies;
  - Tests 3–4: 54 AUCs, 15 dollar-share cells, 1 classification rule.
  - The 0.5/0.2 class thresholds and the weekly betas were set before looking.
- Justification in $ depends on agent 1's weekly betas, which are averages. In stressed spreads the true sensitivity is convex, so justified shares in 2014/2018 may be understated.
- The March−April CIs are driven by 2013-14 and 2018-19. The episode conclusions rest on 2–3 episodes.
- Only 00Z ensemble means are available. "Beyond the next run" uses the d+1 00Z run, which may itself trail what the market sees by d+1.
- Realised weather is the Open-Meteo 17-city composite, so measurement error biases J toward 0.

## 6. Proposed frozen test for the locked 2000–2010 panel (H-HI)
- **Inputs:** `decompose(panel, "r_front", mode="frozen")` (agent 1's frozen coefficients). Rows are November−February days with the needed columns. news_tot_l1 and future sums are built as in `common.daily()`: shift by trading row, with a 10-day gap guard. Realised targets are kept only if the window ends ≤ 2010-06-04.
- **H-HI-1 (primary):**
  - Regression: news_tot_{d+1} = a + c′[lvl_w1..w3, news_w0..w3, news_tot_{d−1}] + β·P_d/sd(P) + e.
  - Pass if β > 0 with HAC (Newey-West, 5 lags) one-sided p < 0.05.
  - Dev: β = 5.1 HDD, t 6.6 in frozen mode.
- **H-HI-2:**
  - Regression: real_tot = a + c′·lvl_w1..w3 + β·P/sd + e, with HAC 15 lags.
  - Pass if β > 0, one-sided p < 0.05 (dev t 4.6).
- **H-HI-3 (timing, not skill):**
  - Regression: real_tot over the d+1 windows on the d+1 levels + β·P_d.
  - Prediction: β is not significant, two-sided p > 0.05 (dev t 1.3).
  - If H-HI-3 rejects, premium carries forecasting skill beyond the next run, and reversal tests need longer future-news controls.
- **H-HI-4 (March−April, secondary):**
  - Same as H-HI-1 with P from `decompose(panel, "d_hj", mode="frozen")` and target FN5.
  - Pass if β > 0, one-sided p < 0.05 (dev t 2.3).
- **Reading:** if H-HI-1/2 pass, a reversal test on the locked panel must include future news controls. If they fail, premium_part there is cleaner and raw reversal tests suffice.

## 7. How to rerun
```
python -m src.research.v4_hidden_info.t1_predict    # Tests 1 and 4 slopes (~45 s)
python -m src.research.v4_hidden_info.t2_share      # Test 2, components, proxy (~5 s)
python -m src.research.v4_hidden_info.t3_episodes   # Tests 3-4, episodes, AUC, dollar shares (~3 min)
python -m src.research.v4_hidden_info.figures       # 2 figures (~3 s)
```
Outputs go to `data/processed/research_v4_hidden_info/` (including `daily_components.csv` with the DIAGNOSTIC pure premium).

## 8. New avenues
- **Archive the 06Z/12Z GEFS runs (and ECMWF if licensable).** This would move the anticipated layer from premium to news and likely double daily news R². It has the highest value: it cleans every reversal test.
- **Use the pure_premium_DIAG series as an upper bound for agent 3.** If even the cleaned premium reverses only about −0.2 in 5 days, no ex-ante rule can do better. It is cheap and sets expectations.
- **Level × surprise in stressed winters.** In 2014 and 2018, forecast levels were alike and the outcomes differed. Ensemble spread (member dispersion) at d may be the ex-ante proxy for "the forecast can still get colder". Medium value; 2 episodes.
- **Down-move asymmetry.** Down premium moves carry no weather information. A test of whether they are the premium's decay channel (drift concentrated on big down days) is low cost, with medium value for sizing the short.
