# v4 agent `premium_shocks`: does the premium part of curve moves reverse?

*23 Nov 2026. Development data only (dates ≤ 2021-12-31). The split is agent 1's frozen walk-forward `decompose` / `decompose_weekly`, so news_part exists from winter 2012-13. "Choose" means winters 2012-13..2016-17 and "check" means 2017-18..Dec 2021. Rows are Nov−Feb. Future moves are on the **same contract** (or the same H−J pair). Slopes: −1 = full reversal, 0 = persistence. CIs are 95% winter-block bootstrap; t-stats are Newey-West HAC with lags of at least the overlap (h+2 days, or k+1 weeks). Regressions are descriptive. Trading rules are chosen on the choose window only.*

## 1. Plain-English findings
- **Premium shocks reverse only partly and slowly, and on daily data the "premium part" is just the raw move.** The daily split removes about 5% of the variance, so fading the premium part is the same as fading the raw move. Second month, all dev:
  - h = 1 day: −0.12 [−0.17, −0.07]
  - h = 5: −0.09 [−0.17, 0.01]
  - h = 10: −0.05 [−0.20, 0.03]
  - The naive raw-move slopes match these to ±0.01.
- **Reversal becomes visible once future news is held fixed (descriptive only).** With future-news controls the premium slope is −0.23 [−0.32, −0.13] at h = 5 and −0.19 [−0.32, −0.08] at h = 10, with both windows negative. This is agent 1's "anticipated news" bias at work. The control is not tradable.
- **The weekly split does better than the raw move at 4 weeks.** Weekly second-month premium reverses −0.40 [−0.61, −0.22] over the next 20 days (choose −0.34, check −0.45), against −0.22 [−0.44, −0.04] for the raw weekly move. Weekly news persists: +0.49, not significant. This holds without 2014 and 2019 (−0.47). It is the most robust result here.
- **Up-shocks reverse and down-shocks do not.** Daily h = 10, second month: up −0.53 [−1.01, 0.18], down +0.09. H−J weekly 2-week: up −1.36 [−1.57, −0.71], down +0.14.
  - Big H−J reversals rest on 2014 and 2018-19. Without those winters the daily H−J h = 5 slope falls from −0.23 to −0.07.
- **No ex-ante feature tells a premium spike that reverses from one that runs, apart from its direction.** None of these separates 24 Jan 2014 from 13–14 Nov 2018: daily or weekly premium share, cumulative premium, tightness, implied vol, or forecast level. All have AUC 0.4–0.6 with CIs spanning 0, except spike direction and, for H−J only, iv level and the weekly premium share (both weak).
- **No tradable rule.** The best split rule (fade the 5-day weekly-split premium, z ≥ 1.5, hold 5 days) made +$445 per trade (choose) and **+$52 (check)**. The naive fade at the same parameters made −$114 and −$21. The split beats the naive fade in 58–92% of the 36 grid cells, but **every CI includes 0**. H−J fades lose money in the check window.

## 2. Method
- **Reversal regression:** fut_h(d) = a + β_N·news_d + β_P·prem_d [+ γ₁·news_{d−1} + γ₂·Σ_{d+1..d+h} news + Monday] + e.
  - Daily: h = 1, 2, 5, 10, 20.
  - Weekly: Friday rows of `decompose_weekly`, h = 5, 10, 20 trading days.
  - The benchmark replaces (news, prem) with the raw move.
- **Front month:** h = 20 is undefined (the contract expires), so the second month is the main instrument.
- **Heterogeneity:** separate premium slopes, each with its own intercept, by:
  - |prem| top decile vs the rest (choose-window cut-off)
  - sign
  - hj_prev tercile
  - month
- **Classification:** spikes are days with |move| ≥ the choose-window 90th percentile (97 second-month, 98 H−J).
  - Outcome: rev_h = −sign(move)·fut_h / sd60, plus a binary label: reversed (≥ 50% undone) vs ran (continued).
  - 13 ex-ante features. Spearman ρ and AUC are reported per window.
  - Nothing is fitted.
- **Rules:**
  - Signal: z = x_k / (sd60·√k), with sd60 = sd of daily moves over d−60..d−1.
  - x_k is one of: cumulative daily-split premium (PREM), rolling weekly-split premium using the walk-forward weekly betas applied to k-day news sums (WPREM), the raw move (RAW, the naive benchmark), or news (NEWS, follow).
  - If |z| ≥ thr, trade −sign(x) for 1 contract (or 1 spread) from the d settle to the d+h settle, one position at a time. "_UP" variants fade up-shocks only.
  - Grid: k ∈ {1, 3, 5}, thr ∈ {1, 1.5, 2}, h ∈ {2, 5, 10, 20}.
  - Selection: highest choose-window t of mean net at 2 ticks, with ≥ 30 trades (H−J: ≥ 20). The same parameters are then run on RAW.
  - Costs: $40 round trip per outright at 2 ticks/side ($80 at 4). H−J costs $80 ($160).

## 3. Results

**Table 1. Premium-part reversal slopes, all dev** (`step1_reversal.csv`; Fig. 1 `figures/v4_premium_shocks_slopes.png`)

| Instrument, h | Naive (raw) | Premium | News | Premium, future-news controls | Choose / check (premium, controls) |
|---|---|---|---|---|---|
| Second, 1d | −0.10 (t −2.4) | **−0.12 [−.17,−.07]** | +0.36 (t 2.0) | −0.17 (t −4.1) | −0.16 / −0.18 |
| Second, 5d | −0.09 | −0.09 [−.17,.01] | +0.07 | **−0.23 [−.32,−.13]** | −0.15 / −0.29 |
| Second, 10d | −0.04 | −0.05 [−.20,.03] | +0.20 | −0.19 [−.32,−.08] | −0.20 / −0.19 |
| Second, 20d | −0.06 | −0.11 [−.28,.06] | +1.12 | −0.27 [−.44,−.18] | −0.28 / −0.27 |
| H−J, 5d | −0.22 | −0.23 [−.31,−.06] | +0.22 | −0.28 (t −2.9) | −0.34 / −0.27 |
| H−J, 10d | −0.16 | −0.17 [−.58,.11] | +1.13 | −0.23 | −0.60 / −0.15 |
| **Second, weekly → 20d** | −0.22 [−.44,−.04] | **−0.40 [−.61,−.22] (t −3.0)** | +0.49 | −0.37 (t −3.2) | split: −0.34 / −0.45 |
| Second, weekly → 10d | −0.04 | −0.10 | +0.21 | −0.11 | −0.13 / −0.06 |
| H−J, weekly → 10d | −0.08 | −0.18 [−.64,.03] | +0.21 | −0.46 [−.70,−.11] | −0.60 / −0.06 (split) |

- Front month is the same as second at 1–5 days: premium −0.07 at 5d, −0.24 with controls.
- Future news enters with a slope of about 2.3, which is the daily attenuation that agent 1 found.
- **Robustness** (`step1_robustness.csv`):
  - The results do not hinge on the GEFS switch: without winter 2020 the weekly 20d slope is −0.40.
  - On **Monday rows only**, the premium slope is **positive**: +0.29 at h = 5 [0.09, 0.54] in both windows. The weekend "premium" persists, which suggests it is unseen news (ECMWF, 06/12Z), not premium.

**Table 2. Heterogeneity** (`step1_hetero.csv`, all dev, with controls)

| Slice | Second, daily h=10 | H−J, daily h=10 | H−J, weekly → 10d |
|---|---|---|---|
| Top-decile \|prem\| / rest | −0.17 / −0.28 | −0.27 [−.78,−.03] / −0.20 | −0.53 / −0.39 |
| Up / down | **−0.53 / +0.09** | **−0.71 / +0.32** | **−1.36 / +0.14** |
| Tight tercile (hj_prev) | −0.20 [−.43,−.04] | −0.26 [−.68,−.11] | −0.55 [−.80,−.26] |
| Jan / Feb | −0.31 / −0.44 | −0.58 / −0.82 | – |

- Big shocks reverse no more than ordinary ones in outrights.

**Concentration** (`step1_concentration.csv`, `step1_episodes.csv`):
- The 10 largest second-month premium days supply 81% of the h = 10 numerator. Removing them only moves the slope from −0.054 to −0.041.
- For H−J at h = 5, the 10 largest days supply **111%**. Without them the slope is −0.04.
- Those days are 13–16 and 28 Nov 2018, 24 and 29–30 Jan 2014, 4 Feb 2014, 29 Nov 2021, and 3 and 14 Dec 2018. Every one is more than 97% "premium". 13–14 Nov 2018 and 24 Jan 2014 both reversed within 10 days. 24 Jan 2014 ran again by day 20 (the February squeeze).

**Table 3. Classification of spikes** (`step2_classify.csv`)
- Spearman ρ with rev_h, all dev, winter-bootstrap CI. Choose / check AUC in brackets.

| Feature | Second, 10d | H−J, 10d | H−J, 20d |
|---|---|---|---|
| Daily premium share | −0.02 | 0.17 [.09,.30] (0.53/0.56) | 0.01 |
| 5-day cumulative premium (z) | 0.02 | 0.01 (0.64/**0.28**) | −0.01 |
| Rolling weekly premium (z) | 0.04 | 0.17 [−.13,.52] (0.72/**0.34**) | 0.07 |
| Weekly premium share | 0.01 | 0.27 [.06,.41] (0.70/0.61) | 0.10 |
| Weekly prem × tightness | −0.19 | 0.14 (0.78/**0.35**) | 0.01 |
| iv_front | 0.15 | 0.26 [.09,.52] (0.63/0.78) | 0.09 |
| Raw 5-day move (naive) | 0.02 | −0.01 | −0.02 |
| **Up spike** | 0.18 | **0.39 [.19,.66]** (0.63/0.91) | **0.59** |

- Base rates: the average spike **continues**, with mean rev10 −0.50 sd (second) and −0.63 sd (H−J). Down spikes run, which is the winter downward drift; removing the walk-forward drift barely changes this.
- Every feature that "works" in one window flips in the other, except direction.

**Table 4. Selected rules** (`step3_selected.csv`; Fig. 2 `figures/v4_premium_shocks_pnl.png`; Fig. 3 `figures/v4_premium_shocks_grid.png`)
- Mean net is $/trade at 2 ticks (4 ticks in brackets), with winter-bootstrap CI.

| Rule (chosen params) | Window | Trades | Mean net | Hit | Worst | Sharpe/yr | Split − naive |
|---|---|---|---|---|---|---|---|
| **2nd WPREM k5 z1.5 h5** | choose | 31 | +445 (+405) [−367, 1022] | 48% | −3,620 | 0.44 | +559 [−68, 1315] |
| | check | 24 | **+52** (+12) [−381, 512] | 50% | −5,740 | 0.06 | +73 [−450, 550] |
| naive RAW, same params | choose / check | 35 / 26 | −114 / −21 | 43% / 50% | −6,780 | – | – |
| 2nd PREM k3 z1 h10 | choose / check | 33 / 24 | +316 / +116 | 55% / 46% | −11,410 | 0.28 / 0.06 | +298 / **−309** |
| naive RAW, same params | choose / check | 33 / 27 | +17 / +424 | – | −11,410 | – | – |
| 2nd WPREM_UP k1 z1 h10 (post hoc) | choose / check | 32 / 23 | +213 / +1,389 [89, 3593] | 56% / 70% | −11,410 | 0.18 / 0.68 | +96 / +61 |
| 2nd NEWS follow k1 z1.5 h5 | choose / check | 39 / 33 | **+745 [358, 994]** / −59 | 67% / 45% | −4,240 | 1.18 / −0.06 | – |
| H−J WPREM k3 z1.5 h10 | choose / check | 22 / 15 | +168 / **−549** | 59% / 40% | −7,110 | 0.15 / −0.42 | +317 / +488 |
| H−J naive RAW, same params | choose / check | 23 / 14 | −149 / −1,037 | 43% / 29% | **−9,260** | – | – |
| H−J PREM k1 z1 h10 | choose / check | 28 / 21 | +60 / −144 | 36% / 48% | −4,770 | – | −281 / +402 |

- **Overlapping signal-day versions** (every qualifying day, HAC t) are weaker:
  - Second PREM: −$113 / −$423 per signal-day.
  - WPREM: +$126 / +$26 (t 0.3 / 0.1).
- **The post-hoc up-only check result rests on Nov–Dec 2021:** +$15.8k of the window's +$32k, with one trade of +$12.2k.
- **By winter, second WPREM** (2 ticks, $ sum): +4.0k, +5.6k, +9.3k, −6.8k, +1.7k | +5.5k, −2.1k, −1.8k, +0.6k, −0.9k.
- **2014 tail on H−J:** both rules lost $7,110 shorting on 14 Jan 2014. The weekly split avoided the naive fade's −$9,260 short of 5 Nov 2018, because that week's move carried news.
- **Grid (36 cells per family; Fig. 3):** share of cells where the split beats the naive fade, choose / check:

  | Family | Second | H−J |
  |---|---|---|
  | WPREM | 64% / 58% | 64% / **92%** |
  | PREM | 53% / 44% | 56% / 22% |

  The mean over cells is still ≤ 0 in the check window, except second-month WPREM_UP (+$355).

## 4. What this tells us
- At daily resolution the "premium part" is the move itself, so this study cannot test news against premium on daily data.
  - What we can say: daily moves overshoot by about 10–15% within 1–2 days, as in v2.
  - The non-news part reverses by about 20–25% over 1–4 weeks **once later news is held fixed**. Unconditionally, a premium shock is about as likely to be followed by more news in the same direction as to fade.
- The weekly split, which captures about 20% news, sharpens the 4-week reversal (−0.40 vs −0.22). This is the first evidence that removing forecast news isolates a mean-reverting component.
- Reversal is a property of **up-moves in tight markets in Jan–Feb**: a scarcity premium that fades. Down-moves persist (premium decay and drift).
- The Monday result is a warning: when news is cleanly timed, the "premium" persists. Part of what we call premium is still news we do not observe.

## 5. Caveats
- **Variant count:**
  - 288 descriptive regressions: 3 instruments × 8 horizons × 4 specifications × 3 windows.
  - 168 heterogeneity cells, plus 4 robustness cuts.
  - 52 classification cells (13 features × 2 instruments × 2 horizons).
  - **504 rule cells** (7 families × 36 × 2 instruments), with 7 selections per instrument.
  - h = 20 and the _UP families were added **after** seeing step 1 on both windows, so they are not clean out-of-sample. Neither h = 20 nor any _UP selection beat the pre-declared choices on the choose window.
- 10 OOS winters. H−J results rest on 2014 and 2018-19.
- Non-overlapping trades are path-dependent: a one-day shift changes cell means by hundreds of dollars.
- The future-news control uses the daily news_part (attenuated about 2.5×). The controlled slopes are descriptive, not tradable.
- Spike classification has about 40–55 spikes per window, clustered in a few episodes.

## 6. Proposed frozen test (locked 2000–2010, Nov−Feb, `src/research/v4_premium_shocks/frozen_test.py`)
- **Inputs:** `r_second`, `front`/`second` ids, `px_front`/`px_second` (all produced by the same `price_block` for the locked panel), and `news_w0..w3`. Coefficients come from `decompose(..., mode="frozen")` / `decompose_weekly(..., mode="frozen")` and `frozen_coefs.json`. No options, CPC, strip or storage.
- Weekly second-month moves (sd ≈ 6%) are far above the 2-decimal rounding (about 0.2–0.5% per cent-tick at 2000s prices).

**H-PS1 (primary, slope)**
- Rows: Friday weeks of `decompose_weekly(panel, "r_second", "frozen")`.
- Regression: fut20 (same contract, 100·ln P_{d+20}/P_d) = a + β_N·news_part + β_P·premium_part.
- NW lags 5.
- **Pass:** β_P < 0 with one-sided p < 0.05, and t(β_N) > −1.645.
- Report: the naive slope on the raw weekly move. The split "adds" if β_P < β_naive.
- Dev dry run (frozen coefficients, in-sample): β_P −0.35 (t −3.0), β_N +0.28, naive −0.21.
- This departs from the design's h = 5–10, because the evidence at 5–10 days needs future-news controls.

**H-PS1b (the design's criterion)**
- Daily split, h = 5 and 10: fut_h ~ news + prem + news_{d−1} + Σnews_{d+1..d+h} + Monday, NW lags h+2.
- Pass: β_P < 0 with one-sided p < 0.05 at both h. This is descriptive (future news).
- Dry run: −0.24 (t −4.3) and −0.20 (t −2.8).

**H-PS2 (trading)**
- Second-month fade of the rolling weekly-split premium: k = 5, z ≥ 1.5, h = 5, one position at a time, $40 per round trip.
- **Pass:** ≥ 20 trades, mean net > 0 at 2 ticks, **and** mean net > the naive raw fade at the same parameters. Also report 4 ticks.
- Dry run: +$258 (63 trades) vs +$7.
- Prior: inconclusive. The dev CI spans 0.

## 7. How to rerun
`python -m src.research.v4_premium_shocks.run`. This takes about 1 minute on 1 CPU (`classify` is about 40 s). It writes `data/processed/research_v4_premium_shocks/` (`step1_*`, `step2_*`, `step3_*`, `frozen_test_dryrun_dev.json`) and three figures.

## 8. New avenues
- **A Monday-cleaned split.** Treat weekend moves as news by default and test reversal on Tue–Fri premium only. *High value:* it targets the clearest sign that "premium" is unseen news.
- **06Z/12Z/ECMWF archives** (agent 1's top item). Without them the daily premium is the move. *High value* for any daily test.
- **An up-shock, tight-market, Jan–Feb weekly fade** (H−J and second month), frozen now and scored live in 2026-27. *Moderate value:* the effect is consistent, but it rests on 2014 and 2018-19.
- **Combine with `widowmaker_v4`.** The weekly split avoided the Nov 2018 naive short but not 14 Jan 2014. Test it as a *hold/cut* filter, not a fade. *Moderate value.*
