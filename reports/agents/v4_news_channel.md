# v4 agent `news_channel`: how much of the gas curve is weather-forecast news?

*23 Nov 2026. Development data only (dates ≤ 2021-12-31). Fits are walk-forward by winter: the first vintage is fitted on 2010-11 and 2011-12 and applied to 2012-13. Specifications were chosen on the out-of-sample (OOS) winters 2012-13 to 2016-17 and checked on 2017-18 to Dec 2021. CIs are winter-block bootstrap, 2,000 resamples.*

## 1. Plain-English findings
- **Daily GEFS news explains about 5% of daily outright moves, and almost none of the daily March−April move.**
  - Front month: OOS R² 0.050 [0.043, 0.056] in the choice window and 0.038 [0.006, 0.094] in the check window. The second month and the strip are the same.
  - March−April ($): 0.008 [0.002, 0.048] and 0.005 [−0.001, 0.059].
  - The 11–15-day window (w3) and the 1–5-day window (w1) carry the weight. The surprise on days already passed (w0) carries none.
- **At weekly frequency it is about 20% for outrights and 16–26% for March−April,** once the March−April sensitivity scales with how tight the spread is (max(S,0)+0.10).
  - The daily news is mistimed. The market prices about half of a 00Z revision the **day before** (from the 06Z/12Z or ECMWF runs, which we don't have) and another third the day after.
  - Summed over a week, moves respond 2.5× (outrights) to 3.8× (March−April) as strongly as the daily fit implies.
- **The ensemble mean beats the control run** (0.050 vs 0.020 in the choice window). **CPC outlooks add nothing** (CPC alone gives OOS R² < 0). **Realised weather in hindsight explains less than the forecasts do:** the market trades forecasts, not outcomes.
- **Monday carries the signal.** OOS R² is 0.11 on Mondays (weekend news) against 0.01 on Tuesday to Friday. Results do not hinge on the GEFS system switch. Storage-release days are no different, and the storage surprise adds nothing.
- **The split does not separate 24 Jan 2014 from 16 Nov 2018 on the day.** Both are more than 95% "premium" daily, and weekly too under walk-forward coefficients. All ten largest March−April days in development are premium by this definition.

## 2. Method
- **Daily model,** fitted on November−February rows:
  - move_d = a + Σ_k b_k·x_k,d + e_d
  - **news_part = Σ b_k·x_k,d.** The intercept (drift) is excluded.
  - **premium_part = move − news_part.**
  - Coefficients for winter W come only from winters < W (`decompose()`, refitted once a year). The `_self_test` asserts this. It also checks that vandalising post-2018 moves leaves earlier news_part values unchanged.
- **OOS R²** = 1 − SSE(model) / SSE(training-mean forecast), pooled over the OOS winters.
- **Frozen daily specification** (choice window, with a parsimony rule: extra terms must add at least 0.005 of R²):
  - Outrights (% log returns): x = news_w1, news_w2, news_w3.
  - March−April ($): x = news_tot, one beta.
- **Weekly companion** (`decompose_weekly()`, Friday to Friday, weeks of at least 4 days): the same terms, summed over the week. For March−April, x = Σ news_tot × (max(S at the prior Friday, 0) + 0.10).
- **Hindsight comparators** are not available ex ante:
  - **perfect-at-d revision:** realised window anomaly minus dp's forecast for the same days;
  - **realised next-2-week level** (v2-style).
- **Variant count:**
  - 59 daily specification variants (`variants.csv`: lead sets, control run, hj scaling with c ∈ {0.05, 0.1, 0.2, 0.4}, iv/price scaling, Monday interactions, Oct−Mar fit window, and 7 post-hoc lag variants);
  - 3 weekly March−April forms;
  - about 40 comparison and benchmark fits that were not used for selection.

## 3. Results

**Table 1 – Daily, November−February, walk-forward OOS R²** (`r2_table.csv`; Fig. 1 `figures/v4_news_channel_r2.png`)

| Model | Front % choose / check | Strip % choose / check | March−April $ choose / check |
|---|---|---|---|
| **GEFS ensemble (frozen)** | **0.050 [.043,.056] / 0.038 [.006,.094]** | 0.050 / 0.038 | **0.008 [.002,.048] / 0.005** |
| GEFS control run | 0.020 [−.029,.039] / 0.035 | 0.020 / 0.035 | 0.007 / 0.007 |
| CPC only / GEFS, CPC sample / GEFS + CPC (OOS 2014-15+) | −0.023 / 0.057 / 0.034 (check −0.002 / 0.039 / 0.042) | similar | −0.38 / 0.043 / −0.35 |
| HINDSIGHT revision | 0.026 / 0.053 | 0.027 / 0.044 | −0.016 / 0.008 |
| HINDSIGHT next-2-week level | 0.007 / 0.016 | 0.007 / 0.016 | −0.009 / −0.001 |
| DIAG.: + next day's 00Z news (look-ahead) | 0.065 / 0.105 | 0.062 / 0.097 | 0.002 / 0.027 |

- **Horizon weights** (front month, full-dev in-sample, HAC t; `coefs_insample.csv`): w1 0.060 %/HDD (t 2.9; 0.36% per s.d.), w2 0.017 (t 1.7; 0.21%), **w3 0.039 (t 5.0; 0.56%)**, w0 −0.02 (t −0.2).

- **March−April:** about 0.025¢ per HDD of news_tot, i.e. 0.6¢ per s.d. of news, against a daily s.d. of 4.3¢.
- **The w1 beta drifts up across vintages** (−0.01 → 0.066). The w3 beta is stable at 0.037–0.044 (`vintage_coefs.csv`).

**State dependence** (`state_dependence.csv`, descriptive: slope of the move on the frozen news index by tercile, full dev):
- **Outrights:** no tightness scaling. By hj tercile: 1.22 / 0.64 / 1.14. By price level: 1.14 / 1.05 / 0.87, so percentage sensitivity is roughly constant. By iv and by days to expiry: flat within CIs. February is weaker: 0.74 [0.04, 1.41].
- **March−April is strongly convex:** slope 0.48 [0.23, 0.73] in the low-hj tercile against 2.12 [1.37, 2.84] in the high tercile.
- **Daily, scaling by hj fails walk-forward** (c = 0.10: −0.024 in the choice window), because 2014-15 and 2019-20 blow up. Iv scaling is also worse for outrights (0.040).

**Table 2 – Weekly (Friday to Friday), November−February** (`weekly_frozen_spec.csv`, `weekly.csv`; Fig. 3)

| Instrument | Sum of daily news_part | Weekly refit (frozen weekly spec), choose / check | Hindsight next-2-week level, in-sample |
|---|---|---|---|
| Front % | 0.146 / 0.137 | **0.190 [.164,.227] / 0.227 [.160,.349]** | 0.105 |
| Strip % | 0.174 / 0.137 | 0.221 / 0.224 | 0.125 |
| March−April, linear | 0.039 / 0.045 | 0.094 [.058,.354] / 0.096 | **0.054 (v2: 0.081)** |
| **March−April × (max(S,0)+0.10)** | – | **0.264 [.214,.526] / 0.161 [−.008,.537]** | in-sample with news: 0.365 (v2 realised: 0.27) |
| March−April, v2 form (news, S, news×S) | – | −0.260 / 0.208 | – |

- **Weekly R² is about 3–4× the daily R²** for outrights.
- **Forecast news beats v2's realised-weather benchmark** for March−April: 0.13 against 0.054 in-sample on the same weeks (v2 reported 0.081 on its sample).
- **The weekly scaled March−April model is positive in 8 of 10 OOS winters.** The exceptions are 2018-19 (−0.02) and 2019-20 (−0.03).

**Timing** (`leadlag.csv`; slope of the move on day d+k on news_part_d, HAC t):

| Instrument | k = −1 | k = 0 | k = +1 | k = 2–5 |
|---|---|---|---|---|
| Front | 1.07 (5.2) | 1.08 (6.2) | 0.38 (2.1) | ≈ 0 |
| March−April | 1.80 (3.5) | 1.10 (3.9) | 0.93 (1.9) | ≈ 0 |

- The correlation of premium_part_d with news_part_d+1 is +0.23 for the front month and +0.13 for March−April.
- A post-hoc variant adding lagged news raised choice-window R² to 0.073 but **fell** to 0.033 in the check window. It was not adopted.

**Robustness** (`robustness.csv`; front-month OOS R²):
- **GEFS system:** reforecast v12 0.038 [.024, .052]; operational 0.061 [−.015, .125]. In-sample interaction t 1.5.
- **Monday:** **0.112 [.067, .154]** on Mondays against 0.013 [−.016, .042] Tuesday to Friday. In-sample interaction t **3.2**. A Monday-interaction specification did not improve the choice window (0.046 vs 0.050), so it was not adopted.
- **Storage releases (2015+):** −0.010 on release days against 0.053 on other days. Adding the storage surprise leaves R² at 0.004 (t −0.2).
- Same-day results hold for second month, strip and March−April (`robustness.csv`).

**Episodes, daily** (`episodes.csv`, `episodes_5d.csv`; walk-forward news_part; Fig. 2 `figures/v4_news_channel_episodes.png`)

| Date | ΔMarch−April | News part | Premium part | Front % (news part) |
|---|---|---|---|---|
| 2014-01-24 | +28.8¢ | −0.0¢ | +28.8¢ | +9.1 (0.07) |
| Other 2014 top-10 days (29 Jan, 4 Feb, 13 Feb) | +31 to +36¢ | ≈ 0 | ≈ all | +8 to +10 (≤ 0.4) |
| 2014-02-19 (Feb spike, top 10) | +40.1¢ | +0.1¢ | +40.0¢ | +10.2 (0.43) |
| 2018-11-13 / 14 / 15 (top 10) | +39.3 / **+71.1** / **−78.2¢** | +0.4 / −0.2 / −0.0¢ | ≈ all | +7.9 / +16.5 / −18.1 |
| 2021-02-12 | +0.4¢ | −0.7¢ | +1.1¢ | +1.5 (−0.84) |
| 2014-02-25; 2021-10-05 / 06 (top 10) | −42.0; +33.7 / −36.0¢ | ≈ 0 | ≈ all | – |

**Episodes, weekly** (walk-forward):

| Week ending | ΔMarch−April | News part | Front news part |
|---|---|---|---|
| 24 Jan 2014 | +39.9¢ | 2.0¢ | 2.3 of 18.1% |
| 21 Feb 2014 | +48.6¢ | 11.7¢ | – |
| 16 Nov 2018 | +67.8¢ | 6.1¢ | – |
| 23 Nov 2018 | +8.1¢ | 36.4¢ | – |

- Even with the frozen all-dev coefficients, the week to 24 Jan 2014 is 6¢ news out of 40¢.
- After Nov 2018, the weekly model attributes the large December moves partly to news, because the spread was wide (S ≈ $1).

## 4. What this tells us
- **The curve trades weather forecasts, and does so fast.** Revisions are priced within a day of the 00Z run, partly before it. The ensemble mean, not the control run, is what moves prices. Hindsight weather explains less than the forecast.
- **Daily H−J moves are about 99% unexplained by GEFS.** The polar-vortex days and the 2018 squeeze look alike on the day, so the ex-ante split that study v4 hoped for is **not** available at daily resolution with 00Z GEFS.
- **At weekly resolution, about a quarter of H−J moves are news,** with sensitivity proportional to tightness, as in v2. The rest, including all the extreme spike weeks, is unexplained.
- **The "premium" contains anticipated news** (+0.23 correlation with the next day's news). That biases reversal tests **toward finding no reversal**, a conservative direction for the premium hypothesis.

## 5. Caveats
- **Variants:** 62 specification variants plus about 40 comparison fits. The weekly H−J scaling form was suggested by v2 and the brief. Its constant c = 0.10 was taken from the daily grid, not tuned weekly.
- **The OOS winters are few:** 5 choice and 5 check. Some of the March−April CIs are wide (upper ends near 0.5) and are driven by 2013-14 and 2018-19.
- **Only 00Z ensemble-mean runs.** The missing 06Z/12Z/ECMWF runs are why the daily R² is low. Coefficients are attenuated by about 2.5×.
- **News for October, March and April rows is extrapolated** from the November−February fit. March−April rows exclude February days after the March expiry.
- **Two descriptive tables are full-dev in-sample:** state terciles and lead/lag.

## 6. Proposed frozen definition (for agents 2–4 and the locked test)
- **Daily (primary for 1–4-day tests):** `decompose(panel, ins)`.
  - Outrights: news_part = b₁·news_w1 + b₂·news_w2 + b₃·news_w3.
  - March−April: news_part = b·news_tot.
  - premium_part = move − news_part.
  - Development: walk-forward vintages (cached in `decomposed_dev.csv`; news_part from winter 2012-13).
  - Locked panel: `mode="frozen"`, using the dev coefficients in `frozen_coefs.json`:

    | Instrument | Coefficients |
    |---|---|
    | Front | 0.0598, 0.0173, 0.0387 %/HDD |
    | Second | 0.0501, 0.0118, 0.0356 %/HDD |
    | March−April | 0.000248 $/HDD |

    These never saw pre-2010 data, and the needed columns all exist there.
- **Weekly (primary for h ≥ 5-day tests, including the v4 pass criterion at h = 5–10):** `decompose_weekly(panel, ins)` (`decomposed_weekly_dev.csv`; frozen weekly coefficients in the json).
  - Outrights: weekly w1–w3.
  - March−April: news_tot × (max(S at the prior Friday, 0) + 0.10), slope 0.0047.
- **Mandatory controls** in any reversal regression of future moves on premium_part: the future news parts over the horizon (d+1..d+h) and news_part at d−1. A Monday dummy is also recommended.
- **Locked news-channel test (H-NC):**
  - With the frozen coefficients on 2000–2010, November−February:
    - (i) the daily front-month news R² > 0.02, with a winter-bootstrap lower bound > 0;
    - (ii) the weekly March−April scaled news R² > 0.05, and greater than the linear weekly form.
  - Pass both, or the decomposition does not transfer to the locked sample and the agents' locked reversal tests should be read as raw-move tests.

## 7. How to rerun
```
python -m src.research.v4_news_channel.decompose   # frozen_coefs.json, decomposed_*_dev.csv, self-test (~15 s)
python -m src.research.v4_news_channel.variants    # 59-variant scan (~5 s)
python -m src.research.v4_news_channel.run         # all tables and figures (~2.5 min)
```

## 8. New avenues
- **Archive the 06Z and 12Z GEFS runs (and ECMWF where licensable).** This should roughly double daily explained variance and fix the timing smear. It has the highest value for a daily split.
- **Weekly H−J news-vs-premium reversal using `decompose_weekly`.** This is the only resolution at which H−J news is visible, so it is the natural home for the widowmaker test.
- **Forecast-level (not revision) channel in stressed winters.** In Jan 2014 the curve rose on days of *warm* revisions, consistent with persistent cold levels and storage draws. Testing lvl_w* × tightness may tell 2014 from 2018 where revisions cannot. Medium value; risk of overfitting on 2 episodes.
- **Monday-only decomposition.** Weekend news is cleanly timed (R² 0.11), so it is a sharper test sample for premium reversal. Medium value; about 140 days per window.
