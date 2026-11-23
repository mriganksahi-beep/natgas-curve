# v4 agent `widowmaker_v4`: can weather-forecast information improve the short March−April trade?

*23 Nov 2026. Development data only (every loader cuts at 2021-12-31 and asserts it; nothing pre-2010 is read; `v4_locked/` is never opened). Rules are chosen on the 2010-11..2016-17 winters ("train") and checked on 2017-18..2020-21 ("check"). 2021-22 is truncated at 31 Dec 2021 and reported separately. P&L is per spread (10,000 MMBtu per leg), marked to market daily on settles, net of 2 ticks per leg per side ($40 per spread per side), with 4 ticks as a sensitivity. CIs are winter-block bootstrap (10,000 draws).*

## 1. Plain-English findings
- **Exiting when the forecast is cold does not beat holding half a spread.** Of the 24 "cut when the 6–15-day (or 1–15-day) forecast is persistently cold" cells, **none** beat half size in training, for either entry date. Of the 8 "size by forecast state" cells, none did either. The train-selected level rule keeps 0.82× (Nov entry) and 0.92× (Dec entry) of half size's premium per unit of drawdown over all 11 winters.
  - The reason is that the losses don't come when the forecast is cold. Only 5 of the 20 worst days for the short followed a cold forecast state.
  - The Feb 2014 expiry squeeze came with a **warm** 6–15-day forecast (z = −1.2 on 14 Feb).
  - The Nov 2018 squeeze came with only a mildly cold one (z = 0.86 on 13 Nov).
  - Cold states actually carried a slightly *better* next-day P&L for the short: +$86/day over warm states, CI [−4, +211].
- **"Cut on a cold forecast revision" looks good only on paper.** Cutting on the raw 5-day sum of news (B) gives the best training score (2.4× half size). That score comes from one-day flat periods that happened to straddle the 24 Jan and 3 Feb 2014 spikes. Executing one settle later turns the same rule into −0.95×, and 2013-14 becomes **−$16,800**. It is not usable.
- **Revisions scaled by tightness are the only weather signal that dodged Nov 2018.** This rule (G) is post-hoc. It cuts on the past week's news × (max(S,0)+0.10), which is the `news_channel` weekly form.
  - At 1 sd it was flat at the Nov 2018 entry and re-entered on 16 Nov, after the top. 2018-19 ended **+$6,790** with a −$2,960 drawdown.
  - It halved the 2014 drawdown, but 2013-14 still ended −$5,980 against the base's −$3,060.
  - It failed in training: 0.28× (Nov) and 0.03× (Dec). It rests on one episode, and it resembles v3's lucky "enter 2 weeks late".
  - A sign-flipped control, cutting on *warm* news, "wins" in training (1.84×). The training window cannot tell a cold signal from its opposite.
- **"Sell the squeeze" adds return and adds tail risk.** The rule adds one spread after a 2-day rise of at least 15¢ when the forecast is not cold. It doubles the mean (Nov entry: $3,028 against $1,377), but the worst drawdown grows to **−$17,760** (2014, added on 14 Feb at a warm forecast, just before the expiry squeeze).
  - It scores 1.64× half size, but 2 winters (2014-15, 2018-19) supply most of that.
  - It is leverage, not protection.
- **Delaying the 1 Dec entry until the 6–15-day forecast is no longer cold (E) is the only robust cell.** It scores 1.09× (threshold 1.0 sd) to 1.16× (0.5 sd), and the score holds with a 1-day lag, at 4 ticks and with a frozen scale. But it changes only 2–4 of 11 winters. Its gain is a few hundred dollars per winter, mostly from 2013-14.
- **Bottom line:** within 136 rule cells, weather-forecast information does not reliably beat half size. Half size remains the benchmark.

## 2. Method
- **Trade.** Short 1 H−J at the settle of the entry day, held to the H expiry settle.
  - Nov entry: the first trading day on or after the first Friday on or after 1 Nov (v3).
  - Dec entry: the first trading day on or after 1 Dec. This is runnable unchanged on the locked panel.
- **Daily P&L** = pos(t−1)·(−ΔS_t)·10,000 − |Δpos_t|·$40.
- **Forecast signals,** all known at the d settle (the GEFS 00Z run of d is public about 06:00 ET):
  - L23 = lvl_w2 + lvl_w3 and L123 = lvl_w1 + lvl_w2 + lvl_w3 (HDD anomaly).
  - Raw news: the k-day sum of news_tot.
  - Tightness-scaled news: Σ news_tot × (max(S_ref, 0) + 0.10), weekly (Friday to Friday, in force the following week) or as a rolling 5 days. The fitted slope from `news_channel` cancels under standardisation, so **nothing is fitted.**
  - **Walk-forward standardisation:** z = x / RMS(x over all Oct–Apr days before d). Anomalies are zero-mean against the model climatology. At least 20 prior days are required (8 weeks for the weekly series).
- **Execution.** Forecast-only signals trade at the same settle (lag 0). Lag 1 is reported. The add-on rule uses the price, so it always trades at lag 1.
- **Families** (× 2 entries × 2 lags × 2 cost levels = 520 scored variants):

| Family | Rule | Grid | Status |
|---|---|---|---|
| A cold cut | hold f while the k-day mean of z(L) > θ, re-enter when ≤ θ | L ∈ {L23, L123}, k ∈ {1, 5}, θ ∈ {0.5, 1, 1.5}, f ∈ {0, ½} (24) | pre-registered |
| B news cut | hold f while z(k-day Σ news_tot) > θ | k ∈ {5, 10}, θ ∈ {1, 1.5}, f ∈ {0, ½} (8) | pre-registered |
| C size | 3-state (1 / ½ / 0 at z ≤ 0 / ≤ θ / > θ) or continuous 1 − z/θ, rounded to ¼ | k ∈ {1, 5}, θ (8) | pre-registered |
| D add-on | add `add` spreads once if S rose ≥ X over 2 settles, z(L23) ≤ 0.5 and z(2-day news) ≤ 1; held to expiry | X ∈ {0.15, 0.30}, add ∈ {½, 1} (4) | pre-registered |
| E delay | enter on the first day ≥ the nominal entry with z(L23, 5-day) ≤ θ | θ ∈ {0, 0.5, 1} (3) | pre-registered |
| F hysteresis | flat at z(L23) > θ, re-enter only at z ≤ 0 | (4) | post-hoc |
| G tight news | hold f while the tightness-scaled news z > θ | weekly / rolling 5 days, θ ∈ {0.5, 1, 1.5}, f ∈ {0, ½} (12) | post-hoc, added after seeing A–E and the `news_channel` weekly episode table |
| control | G on *warm* news (sign flipped) | (3) | not eligible |

- **Score.** R = premium kept ÷ drawdown kept = (mean_rule / mean_base) ÷ (worstDD_rule / worstDD_base). This equals the ratio of Calmar ratios, so **half size = 1.00**.
  - P(beat half) is the share of winter-bootstrap resamples in which the rule's Calmar exceeds the base's.
  - Selection: per family, the best train Calmar with kept ≥ 0.4.

## 3. Results
**Table 1n. Nov entry** (2 ticks; the train-selected cell per family, plus G and its control). Figure: `figures/v4_widowmaker_frontier.png`.

| Rule | Train mean / worst DD | R train | Check mean / worst DD | R check | 11 winters: mean [CI] | Worst DD | Worst winter | R all | P(beat half) | R lag 1 | R 4 ticks |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 1,123 / −12,010 | 1 | 1,822 / −13,230 | 1 | 1,377 [223, 2,360] | −13,230 | −3,060 | 1 | – | – | 1 |
| half | 561 / −6,005 | 1 | 911 / −6,615 | 1 | 689 [131, 1,194] | −6,615 | −1,530 | 1 | – | – | 1 |
| A L123 k5 θ1.5 f½ | 827 / −11,170 | 0.79 | 1,676 / −13,230 | 0.92 | 1,136 [−307, 2,378] | −13,230 | −4,775 | 0.82 | 0.19 | 1.01 | 0.78 |
| B news k5 θ1.5 f0 | 1,399 / −6,290 | 2.38 | 1,685 / −9,740 | 1.26 | 1,503 [441, 2,677] | −9,740 | −1,700 | 1.48 | 0.95 | **−0.03** | 1.12 |
| D X0.15 add 1 | 2,371 / −17,760 | 1.43 | 4,177 / −13,230 | 2.29 | 3,028 [1,166, 5,441] | **−17,760** | +90 | 1.64 | 0.95 | (lag 1) | 1.68 |
| E θ0.5 | 1,099 / −12,010 | 0.98 | 1,612 / −13,230 | 0.88 | 1,285 [228, 2,241] | −13,230 | −3,060 | 0.93 | 0.08 | 1.01 | 0.93 |
| G weekly θ1.5 f0 (train pick) | 669 / −5,510 | 1.30 | 1,252 / −13,230 | 0.69 | 881 [−671, 2,145] | −13,230 | −5,460 | 0.64 | 0.15 | 0.56 | 0.57 |
| G weekly θ1.0 f0 | 160 / −6,030 | 0.28 | 2,515 / −2,960 | 6.17 | 1,016 [−728, 2,669] | −6,030 | −5,980 | 1.62 | 0.60 | 1.50 | 1.35 |
| control: warm news θ1.0 | 1,450 / −8,430 | 1.84 | −23 / −13,230 | −0.01 | 915 | −13,230 | −5,130 | 0.66 | 0.30 | 0.88 | 0.56 |

**Table 1d. Dec 1 entry.** The check window has no tail event (the Nov 2018 squeeze precedes entry), so R check is uninformative.

| Rule | R train | R check | 11 winters: mean [CI] | Worst DD | Worst winter | R all | P(beat half) | R lag 1 | R 4 ticks |
|---|---|---|---|---|---|---|---|---|---|
| base | 1 | 1 | 1,495 [138, 3,080] | −11,760 | −2,770 | 1 | – | – | 1 |
| A L123 k5 θ1.5 f½ | 0.60 | 1.01 | 1,276 | −10,940 | −4,550 | 0.92 | 0.18 | 1.00 | 0.87 |
| B news k5 θ1.5 f0 | 2.52 | 0.73 | 1,405 | −6,290 | −1,320 | 1.76 | 0.68 | **−0.14** | 1.43 |
| D X0.15 add 1 | 1.05 | 0.96 | 2,638 | −17,510 | +230 | 1.18 | 0.64 | (lag 1) | 1.20 |
| **E θ1.0** (train pick) | 1.18 | 1.00 | 1,562 [348, 3,127] | −11,230 | −1,810 | **1.09** | 0.65 | 1.09 | 1.10 |
| E θ0.5 | 1.08 | 1.19 | 1,657 | −11,230 | −2,160 | 1.16 | 0.87 | 1.13 | 1.17 |
| G weekly θ0.5 f½ (train pick) | 1.03 | 1.01 | 1,106 | −7,895 | −3,105 | 1.10 | 0.54 | 0.84 | 1.00 |
| G weekly θ1.0 f0 | 0.03 | 0.95 | 914 | −5,880 | −5,790 | 1.22 | 0.30 | 0.97 | 1.04 |

**Share of cells with R > 1, train → all 11 winters:**

| Entry | A | B | C | D | E | F | G |
|---|---|---|---|---|---|---|---|
| Nov | 0% → 21% | 25% → 75% | 0% → 0% | 50% → 50% | 0% → 33% | 0% → 0% | 33% → 50% |
| Dec | 0% → 0% | 25% → 62% | 0% → 0% | 100% → 50% | 67% → 100% | 0% → 0% | 8% → 67% |

**Table 2. Per winter, Nov entry: final P&L (worst MTM drawdown), $**

| Winter | Base | Half | A (pick) | B (pick) | D (pick) | G weekly θ1 |
|---|---|---|---|---|---|---|
| 2010-11 | 1,060 (−470) | 530 (−235) | 600 (−470) | 180 (−570) | 1,060 (−470) | 0 (−630) |
| 2011-12 | 1,520 (−590) | 760 | 1,520 | 1,520 | 1,520 | 1,520 |
| 2012-13 | 380 (−430) | 190 | 380 | 80 | 380 | 620 (−270) |
| **2013-14** | **−3,060 (−12,010)** | −1,530 (−6,005) | −4,775 (−11,170) | −1,700 (−6,290) | 90 (**−17,760**) | −5,980 (−6,030) |
| 2014-15 | 4,690 (−3,880) | 2,345 | 5,375 (−2,955) | 5,410 | 10,280 (−4,840) | 3,340 (−860) |
| 2015-16 | 1,120 (−1,110) | 560 | 1,120 | 1,140 | 1,120 | 1,120 |
| 2016-17 | 2,150 (−2,060) | 1,075 | 1,570 | 3,160 (−1,150) | 2,150 | 500 (−3,100) |
| 2017-18 | 2,040 (−1,990) | 1,020 | 2,230 | 1,590 | 2,040 | 420 (−2,690) |
| **2018-19** | **1,980 (−13,230)** | 990 (−6,615) | 1,280 (−13,230) | 3,880 (−9,740) | 10,950 (−13,230) | **6,790 (−2,960)** |
| 2019-20 | 2,680 (−550) | 1,340 | 2,025 | 930 (−750) | 2,680 | 2,200 |
| 2020-21 | 590 (−2,340) | 295 | 1,170 (−1,760) | 340 | 1,040 | 650 |
| 2021-22* | 10,010 (−2,430) | 5,005 | 10,010 | 10,010 | 15,500 (−3,340) | 10,010 |

Dec entry: base 2013-14 −2,770 (−11,760) and 2018-19 +7,950 (−3,060). E θ1.0 changes only 2010-11 (+820 vs +1,050) and 2013-14 (−1,810 (−11,230)). Full tables: `data/processed/research_v4_widowmaker/tables.md`.

**Episodes** (`episodes.csv`; Fig. `figures/v4_widowmaker_episodes.png`):
- **2013-14.**
  - **Level cut (A):** at ½ size from 21 Jan to 6 Feb, so it halved the 24 and 29 Jan spikes (−$5,385 against −$8,920 on 29 Jan). It returned to full size on the early-Feb warm spell and took the whole 14–20 Feb squeeze (−$11,120).
  - **G:** flat 17 Jan – 7 Feb, re-entered at S = 0.37, lost $3,820 by 14 Feb, then sat flat through the expiry collapse.
  - **D:** added a second spread on 14 Feb (forecast z −1.2), reaching −$17,710 on 20 Feb and finishing +$90.
  - **E (Dec):** waited out the cold Dec forecast, entered 17 Dec, then held through both spikes.
- **2018-19.**
  - **No level rule was flat on 13–14 Nov.** z peaked at 1.1 on 7 Nov and was 0.86 on 13 Nov.
  - **B:** happened to be flat on 12 Nov (lag 0) and back in on 13 Nov.
  - **G weekly:** flat from entry, because the tightness-scaled news in the week to 2 Nov was z 1.2. It entered on 16 Nov, after the top.
  - **D:** added on 6 Dec at S = 0.96 (+$8,970).
- **Oct–Dec 2021.** The 5–6 Oct spikes precede both entries. Over Nov–Dec 2021 the 6–15-day forecast was warm or normal (z −1.1 to +0.3). Every cut rule therefore held full size: +$10,010 (Nov entry) and +$1,700 (Dec entry) to 31 Dec. D added on 17 Nov (+$15,500).

**State diagnostic** (Nov-entry live days, next-day P&L of the short):

| State | Share of days | Mean $/day | SD | Share of 20 worst days |
|---|---|---|---|---|
| z(L23) ≤ 0 | 0.51 | −7 | 421 | 0.30 |
| 0 < z ≤ 1 | 0.32 | +28 | 848 | 0.45 |
| z > 1 | 0.17 | +82 | 835 | 0.25 |
| tight news z > 1 | 0.16 | +22 | 1,492 | **0.70** |
| S > 0.40 (price-only) | 0.07 | +246 | 2,155 | **0.60** |

Tightness-scaled news flags high-variance days, but those days carry the premium too (mean +$22 against +$19, CI of the difference [−121, +106]). The spread level alone does the same on fewer days. This is v3's "the risk sits where the premium is richest" once more.

## 4. What this tells us about news versus premium
- **The forecast level does not mark the dangerous days.**
  - The 2014 tail had two legs. January's was forecast-led: cold 6–15-day forecasts preceded it by about a week, and a level cut halved it. February's was a realised-scarcity expiry squeeze under a warm forecast, and no forecast rule can see that.
  - 2018 was a premium squeeze with only mildly cold forecasts.
  - So the split that study v4 hoped for is real for one of three tail legs, and a rule cannot exploit it without re-entering into the next leg.
- **News matters for H−J only in interaction with tightness,** consistent with `news_channel`.
  - The tightness-scaled weekly signal is the only one that was out before Nov 2018. Its value is mostly that of the spread level: it fires when S is already high, which is when v3's level and volatility triggers fired, at the same cost in premium.
  - The warm-news control shows that one tail winter cannot validate any of this.
- **The premium is not paid for cold forecasts.** Cold forecast states have the same or better forward drift for the short. Being short into a cold forecast is not, on average, the mistake. Being short into a tight spread late in the season is the risk, and that risk is the premium.

## 5. Caveats
- **Two tail events** (2013-14 in train, 2018-19 in check; Dec entry has only 2013-14). Every R is effectively driven by 1–2 winters, and labelled as such. Bootstrap P(beat half) above 0.9 (B, D) reflects resampling those winters, not robustness: B fails at lag 1, and D's tail grows.
- **Variants:** 66 rule cells per entry (47 pre-registered, 16 post-hoc, 3 controls) × 2 entries × 2 lags × 2 cost levels = **520** scored variants; 1 selection rule per family; plus 1 state diagnostic and 1 frozen-scale check. The G family was designed after seeing the A–E results and the `news_channel` weekly Nov 2018 values.
- **Lag 0 assumes** a settle order placed after the 06:00 ET GEFS run. The market partly prices that run the day before (`news_channel`), so lag 0 is not optimistic about information, only about execution.
- **2020** uses operational GEFS v11 and **2020-21 onward** v12 operational, against the reforecast climatology. Mean L23 in the operational-v12 days is −16 HDD, which could be a warm bias or just warm winters. Results do not hinge on 2020 (no rule changes the 2019-20 tail, which had none).
- **Fractional spreads** (½, ¼) stand for a multi-spread book.

## 6. Proposed frozen test (locked 2000-01 … 2009-10)
**H-WM4: "don't initiate the short into a cold 6–15-day forecast"** (E, θ = 1.0, the train-selected Dec cell, robust to lag and cost).
- **Signal:** z(d) = mean over trading days d−4..d of (lvl_w2 + lvl_w3) / **36.85** (frozen dev RMS, Oct–Apr, `frozen_constants.json`; the walk-forward scale gives the same trades in dev).
- **Entry:** the settle of the first trading day d ≥ the first trading day on or after 1 Dec of Y−1 with z(d) ≤ 1.0. If none occurs, no trade (P&L 0).
- **Exit:** the H_Y expiry settle (last day the locked panel's hj is on H_Y/J_Y).
- **Costs:** $40 per spread per side (2 ticks per leg). Also report 4 ticks.
- **Benchmarks:** the plain Dec-1 short and half of it, same exit.
- **Statistic:** R = (mean_rule / mean_base) ÷ (worstDD_rule / worstDD_base) over the 10 winters.
- **Pass:** R ≥ 1.10 **and** mean_rule > 0.
- **Inconclusive:** the base worst drawdown is smaller than $5,000, or the base mean is ≤ 0. In that case report the differences.
- **Also report** the bootstrap P(beat half) and the per-winter table, highlighting 2000-01 and 2002-03.
- **Expectation:** fail or inconclusive. Dev R is 1.09, driven by 2013-14.
- **Logged, not pass/fail:** G weekly θ1.0 f0 and D X0.15 add 1 (lag 1), on the same entry.

**Live 2026-27 variant** (freeze before 1 Dec 2026):
- The same E rule on H27−J27 with operational GEFS v12 (00Z of d) and the same 36.85 scale. Entry ≥ 1 Dec 2026, exit at the H27 expiry settle (24 Feb 2027).
- Report the final P&L and worst drawdown against the plain 1 Dec short and half size.
- Log the weekly tightness-scaled news z every Friday as an alarm series (descriptive, n = 1).

## 7. How to rerun
```
PYTHONPATH=. python -m src.research.v4_widowmaker.run      # ~15 s: winter_results, summary, selected, run_log
PYTHONPATH=. python -m src.research.v4_widowmaker.tables   # ~5 s: tables.md, episodes.csv, frozen_constants.json, 2 figures
```
Delete `data/processed/research_v4_widowmaker/winters_*.pkl` to rebuild the winter frames.

## 8. New avenues
1. **A price + forecast two-key exit** (S above the walk-forward 80th percentile **and** tightness-scaled news z > 1), evaluated only on the locked winters. It targets the 70% / 60% concentration of the worst days. *Expected value: low to medium; v3's level rule failed on premium kept.*
2. **Feb-expiry realised-scarcity guard** (storage draw versus the 5-year average in Feb only). The Feb 2014 leg is invisible to forecasts. *Expected value: medium, but there is no pre-2010 weekly storage, so it can only be tested live.*
3. **06Z/12Z or ECMWF runs** to sharpen the Jan-2014-type forecast-led leg, where the level cut helped. *Expected value: medium for that leg only.*
4. **Combine half size with the G4X call (`v4_options`)** instead of forecast rules. It is the only overlay that sat above the scaling line in both windows. *Expected value: medium.*
