# Risk overlays for the short March−April premium (agent `premium_risk_management`, study v3)

*22 Nov 2026. Only data dated on or before 2021-12-31 is used; every loader asserts this. Model 2 enters only through `scenario_risk`'s cached PHYS-M2 forecasts, which use origins Nov 2016 – Dec 2021. Winter 2021-22 is marked to market on 31 Dec 2021 and closed there. The 2022 expiry is never read. P&L is per spread (10,000 MMBtu per leg), net of 4 ticks per contract per side ($80 per spread per side).*

## 1. Plain-English findings
- **The base trade.** Short 1 H−J from the first November Friday, held to expiry. Over 2010-11 to 2020-21 it made **+$1,297 per winter** (95% winter-bootstrap CI 161–2,290). Including the truncated 2021-22 (+$9,930), the mean is +$2,017.
  - Worst peak-to-trough drawdowns: **−$13,270 (2018-19)** and **−$12,010 (2013-14)**. Every other winter's worst drawdown was smaller than −$3,900.
- **Tail protection costs at least as much premium as it removes.**
  - The honest benchmark is simply holding fewer spreads. Holding half a spread keeps 50% of the premium and cuts the worst drawdown by 50%.
  - Over all 12 winters, the **median variant in every overlay family falls below that line**. The physical trigger is the one exception (6 winters only). See `premium_risk_management_frontier.png`.
  - **No family beats the line consistently.** The volatility trigger beat it in 90% of its cells in training but only 40% in the check period. Staggered entry did the opposite: 5% in training, 95% in the check period.
- **Choosing on 2010–17 failed in 2017–21.** Training picked a $3,000 stop with re-entry after 4 weeks. It kept 158% of the premium and cut the drawdown by 53% in training, because its re-entry happened to land after the Feb 2014 peak.
  - In Nov 2018 the spread jumped about $11,000 in two days and then half reversed. The stop was triggered on 13 Nov, executed at the 14 Nov top, and turned a +$1,900 winter into **−$4,810**.
  - **Stops depend heavily on execution.** Exiting at the trigger day's own settle instead of the next one turns the same rule into the best point in the whole study: 128% kept, 52% drawdown cut. The Nov 2018 risk was a gap, not a trend.
- **The ex-ante triggers (PHYS-M2 ES, volatility, spread level) do protect.** They cut the worst drawdown by 64–82%.
  - They keep only 9–47% of the premium, because they go flat exactly when the spread, and so the premium, is richest (Nov 2018, Nov 2019, Nov 2021).
  - **PHYS-M2 at 2× its median** exited on 12 Nov 2018 at −$2,310, the only rule that was flat before the jump on a signal and not by luck. But it cleared its threshold by a hair (alarm 2.15), and m = 2.5 misses the episode entirely.
- **Hedging with next year's H−J is cheap but does not protect.**
  - The hedge ratio is 1.57 next-year spreads per front spread (pooled, 2010–17). It is unstable: 0.43–4.23 by winter, with a daily correlation of only 0.22–0.77.
  - It turned 2013-14 into a profit (+$6,260 at 1.5× the hedge ratio), but cut the Nov 2018 drawdown by only 9%.
- **The trade-off is not estimable from 2 events.** Every point on the frontier owes its place to the timing of one or two days in Jan 2014 and Nov 2018.

## 2. Method
- **Base trade.**
  - Entry at the settle of the first trading day on or after the first Friday on or after 1 November.
  - Daily P&L = pos(t−1) × −ΔS(t) × 10,000 − |Δpos| × $80. The position is flat at the expiry settle.
- **Execution.** Trigger-based exits (stops, volatility, physical, level) are seen at the close of day t and executed at the **next settle**. Scheduled changes (calendar windows, staggered tranches, timed re-entry) execute on their scheduled day. Same-settle stops are run as a sensitivity only.
- **Overlays.** Each overlay has at most 2 parameters, on a grid. There are 122 overlay variants plus the base.

| Family | Rule | Grid |
|---|---|---|
| (a) Stop | Flat when MTM since the last entry ≤ −X. Optional re-entry after R weeks, with the reference reset | X ∈ {2,3,4,5,6,8}k × R ∈ {none, 2, 4}; also run with same-settle exit (a0) |
| (b) Trailing stop | Flat when MTM ≤ its running peak − X | Same grid |
| (c) Volatility | Flat when trailing 10-day realised σ(ΔS) > k × its value on entry day. Re-entry (re = 1) when it falls back below | k ∈ {1.5, 2, 3, 4, 6} × re |
| (d) Physical | Flat when PHYS-M2 ES97.5 (h = 1 w, Friday d, Model 2 origin d−7) ÷ the median of all earlier PHYS-M2 ES > m. Needs ≥ 10 earlier forecasts. Model 2 era only | m ∈ {1.5, 2, 2.5, 3} × re; m = 2, re = 1 fixed a priori |
| (d2) Level (price-only twin) | Flat when S > L | L ∈ {0.25, 0.4, 0.6, 0.8} × re |
| (e) Stagger | n tranches of 1/n spread, one per week, the first `off` weeks after entry | n ∈ {1, 2, 4, 6, 8} × off ∈ {0, 2, 4, 6} |
| (f) Hedge | Long h·β of the hedge leg from entry to expiry. β = pooled OLS of daily ΔS on ΔX, 2010–17 live days | X ∈ {next-year H−J, front-March outright} × h ∈ {0.25, 0.5, 1, 1.5} |
| (g) Calendar | Flat in a window starting at a fixed date | Start ∈ {10 Nov, 20 Nov, 1 Dec, 15 Dec, 2 Jan} × {2, 4, 6} weeks |

- **Metrics, per sample.**
  - Mean net P&L per winter, with a winter-block bootstrap CI (4,000 draws), and the paired difference from the base.
  - Worst peak-to-trough drawdown within a winter.
  - Worst winter.
  - Sortino ratio: weekly P&L over all season weeks, flat weeks counted as 0, ×√52.
  - **Kept** = mean ÷ base mean on the same winters. **Cut** = 1 − worst DD ÷ base worst DD.
- **Benchmark: scaling.** Holding f spreads gives kept = f and cut = 1 − f. An overlay adds value only if kept + cut > 1.
- **Selection.** Per family, the cell with the highest train mean ÷ |worst DD| (Calmar) over 2010-11 to 2016-17. It is checked on 2017-18 to 2021-22.
  - The physical family is fixed at m = 2 with re-entry, because training contains only one Model 2 winter.
- **Timing placebo.** Each winter keeps the overlay's number of flat days, placed as one random contiguous block (2,000 draws). p_joint is the share of draws with both kept and cut at least as good as the overlay's.

## 3. Results
**Table 1. Train-selected overlays, plus the frontier variants (rows marked †)**

| Variant | Train mean / worst DD | Check mean [CI] / worst DD | All-winters kept / cut | Sortino (all) | Worst winter (all) |
|---|---|---|---|---|---|
| **Base** | $1,043 / −12,010 | $3,380 [1,206, 6,742] / −13,270 | 1.00 / 0 | 1.08 | −3,140 |
| (a) Stop $3k, re-entry 4 w | 1,644 / −5,700 | 2,038 [−1,974, 6,580] / −13,650 | 0.90 / −0.03 | 0.80 | −4,810 |
| (b) Trailing $4k, re-entry 4 w | Identical to (a) in every winter | | | | |
| (c) Volatility k = 1.5, re-entry | 631 / −1,550 | 1,304 [−2,150, 5,864] / −3,240 | 0.45 / 0.76 | 1.40 | −3,070 |
| (d) PHYS-M2 m = 2, re-entry (a priori; 2016–21 only) | n/a | 1,208 [−166, 2,654] / −4,740 | 0.43 / 0.64 | 1.48 | −1,380 |
| (d2) Level L = 0.25, no re-entry | 509 / −2,790 | −286 / −2,340 | 0.09 / 0.79 | 0.35 | −2,780 |
| (e) Enter 2 weeks late (n = 1, off = 2) | 1,053 / −11,990 | 4,240 [1,352, 7,732] / −3,020 | 1.18 / 0.10 | 1.74 | −3,120 |
| (f) Hedge 1.5β next-year H−J | 842 / −6,548 | 2,093 / −12,095 | 0.68 / 0.09 | 0.92 | −2,073 |
| (g) Flat 2 Jan + 4 w | 1,257 / −10,350 | 3,064 / −13,270 | 1.00 / 0.00 | 1.19 | +30 |
| † PHYS-M2 m = 2, no re-entry | n/a | 888 / −2,340 | 0.34 / 0.82 | 1.40 | −2,310 |
| † Volatility k = 1.5, no re-entry | 261 / −1,460 | 1,906 / −2,430 | 0.47 / 0.82 | 1.73 | −1,250 |
| † Stop $2k, re-entry 4 w | 1,479 / −6,860 | 3,678 [1,504, 6,998] / −6,320 | 1.19 / 0.48 | 1.77 | −90 |
| Same-settle stop $3k, re-entry 4 w (not implementable as tested) | 1,800 / −5,860 | 3,678 / −6,320 | 1.28 / 0.52 | 1.90 | +300 |

- **Paired differences from the base (all winters).**
  - The only significant ones are negative: volatility k = 1.5 with re-entry (CI −2,225 to −55), PHYS-M2 (−3,750 to −208) and level (−3,509 to −489).
  - No overlay's gain over the base has a CI excluding 0.
- **Without the truncated 2021-22** (2017-18 to 2020-21), the check-period base mean is $1,743. On that basis the volatility rule (−$853) and the level rule (−$595) lose money, the stop keeps 4%, and PHYS-M2 keeps 35%.

**The frontier** (`premium_risk_management_frontier.png`; Pareto set among next-settle variants, all 12 winters):

| Worst-DD cut | Premium kept | Variant |
|---|---|---|
| 0.83 | 0.18 | PHYS-M2 m = 1.5, no re-entry |
| 0.82 | 0.34 | PHYS-M2 m = 2, no re-entry |
| 0.82 | 0.47 | Volatility k = 1.5, no re-entry |
| 0.70 | 0.62 | Volatility k = 2, re-entry |
| 0.53 | 0.68 | Stop $2k, no re-entry |
| 0.48 | 1.19 | Stop $2k, re-entry 4 w |

- **Which overlays sit on the frontier:**
  - At deep cuts, the physical and volatility triggers (kept + cut = 1.16–1.29).
  - At moderate cuts, a $2k stop with re-entry. That stop is a cliff: the next grid cell, $3k, gives kept 0.90 and cut −0.03, because it exits one day later in Nov 2018.
- **Share of cells above the scaling line, all winters:**

  | Family | Share above the line |
  |---|---|
  | Same-settle stop | 61% |
  | PHYS-M2 | 38% |
  | Volatility | 40% |
  | Stagger | 26% |
  | Stop | 17% |
  | Calendar | 7% |
  | Trailing stop | 6% |
  | Hedge, level | 0% |

- **Check-period frontier.** It is dominated by stagger cells (entering 2 weeks late keeps 125% and cuts 77%). That is entirely because the 2018 jump fell in the second week of November. The same cells cut 0–6% in training.
- **Timing placebo** (14 overlays tested): only three beat random timing at p_joint < 0.05: volatility k = 1.5 with re-entry (0.018), stop $2k with 4-week re-entry (0.022) and entering 2 weeks late (0.034). About 0.7 would be expected by chance. PHYS-M2 m = 2 scores p_joint = 0.15.

**Episodes, day by day** (`premium_risk_management_episodes.png`; full logs in `episode_2014.csv` and `episode_2019.csv`):
- **2013-14.**
  - **Base path:** −$2,700 by 20 Dec, −$5,690 on 24 Jan, −$8,960 on 29 Jan, −$11,910 on 19 Feb, ending −$3,140.
  - **Stop $3k:** triggered on the 24 Jan gap and exited 27 Jan at −$3,880. It re-entered on 24 Feb, after the peak, and ended **+$1,070**.
  - **Volatility rule:** flat from 6 Dec at −$600.
  - **Level rule:** flat from 20 Dec at −$2,780.
  - **January calendar window:** flat 2–30 Jan, ending +$2,100 (worst DD −$10,350, because February was the worse part).
  - **Next-year hedge:** **+$6,260**. Next year's spread rose with the front through Feb 2014.
  - **Staggered entry:** identical to the base.
- **2018-19.**
  - **Base path:** −$2,230 on 12 Nov, −$6,160 on 13 Nov, **−$13,270 on 14 Nov**, −$5,450 on 15 Nov, ending +$1,900.
  - **PHYS-M2:** alarm 2.15 on Friday 9 Nov, flat on 12 Nov at −$2,310. With re-entry it went back in on 31 Dec and ended −$1,380.
  - **Volatility rule:** flat from 6 Nov at −$1,250. Its flat period was triggered by a minor early-November vol rise, not by any forecast.
  - **$3k next-settle stop:** exited on 14 Nov at the top, ending −$4,810.
  - **$2k stop:** triggered on 12 Nov, exited 13 Nov at −$6,160, re-entered 11 Dec, ending +$3,390.
  - **Same-settle $2k stop:** ended +$8,530.
  - **Next-year hedge:** worst DD −$12,100.
  - **Entering 2 weeks late:** entry on 16 Nov, after the top, ending +$9,970.

## 4. What this tells us
**(a) The futures curve.**
- The two tail events differ in kind:
  - **2014 was persistent scarcity.** Repeated spikes over five weeks, with next year's spread repricing too. Stops and hedges help.
  - **2018 was a two-day squeeze that half reversed within a day.** Next-settle stops sell the top, hedges barely move, and only being flat beforehand helps.
- No single simple rule handles both kinds.
- Because the premium is highest exactly when the tail is fattest, any ex-ante "too risky" signal also removes the rich premium. This is the same mechanism `scenario_risk` found for ES sizing.

**(b) Storage expectations.**
- PHYS-M2's lead in Nov 2018 is real but thin: the alarm was 2.15 against a threshold of 2.
- PHYS-M2 kept the book flat in Nov 2019 (alarm 3.3) and Nov 2021 (alarm 7–10), costing roughly $8,000 of premium in those two winters.
- Physical scarcity risk is informative about *when* the spread can jump, but not enough to pay for itself over 6 winters.

## 5. Caveats
- **Two tail events.** Worst-DD and "cut" statistics are effectively n = 1 per sample. CIs on means span 4–8 winters.
- **Execution timing.** Settle-only data cannot say where an intraday stop would have filled on 13 Nov 2018 or 24 Jan 2014. The lag-0 and lag-1 results bracket the truth, and they bracket it widely.
- **The 2021-22 truncation** (+$9,930 in two months) inflates every check-period mean. "test4" results are reported in the logs.
- **The PHYS-M2 alarm denominator** is the median of earlier forecasts, which are mostly calm late-winter weeks. That biases the alarm upward in November.
- **Variants tried.**
  - 123 strategy variants: 122 overlays plus the base, including 18 same-settle stops added after seeing the lag-1 stop results.
  - Each scored on 6 samples.
  - 14 placebo tests.
  - 1 selection rule.
  - One earlier identical run without the a0 family.
  - Same-settle stops were not eligible for selection.

## 6. Proposed holdout test (frozen; re-used holdout 2022-23 to 2025-26, then live 2026-27)
**H-PRM.**
- **Base.** Short 1 H−J at the settle of the first trading day on or after the first Friday on or after 1 Nov, held to the March expiry, 4 ticks per contract per side.
- **Overlay.** PHYS-M2 exactly as in `src/research/scenario_risk/models.py` (h = 1, ES97.5, walk-forward map), every Friday from 1 Nov to 15 Feb.
  - Alarm = ES ÷ **$1,530** (the median of the 79 development forecasts, frozen).
  - Alarm > 2.0: flat at the next settle. A later Friday with alarm ≤ 2.0: re-enter at the next settle.
- **Statistics, over all holdout winters:**
  - Kept = mean overlay P&L ÷ mean base P&L.
  - Cut = 1 − worst overlay DD ÷ worst base DD.
- **Pass:** kept + cut ≥ **1.10** and kept ≥ 0.40.
- **Fail:** otherwise.
- **Inconclusive** if the base's worst drawdown is under $5,000 (no tail event to protect against), or if the base mean is ≤ 0. In the second case, report the difference in means instead.
- **Reported alongside, not part of pass/fail:** the price-only $2k stop with 4-week re-entry (next-settle exit) and half-size scaling.
- **Expectation:** fail or inconclusive.

## 7. How to rerun
```
PYTHONPATH=. python -m src.research.premium_risk_management.run       # ~15 s; tables + run_log.txt
PYTHONPATH=. python -m src.research.premium_risk_management.figures   # ~5 s
```
- Outputs go to `data/processed/research_premium_risk_management/`: `summary.csv`, `winter_metrics.csv`, `placebo.csv`, `frontier_*.csv`, `episode_*.csv` and `hedge_betas.csv`.
- PHYS-M2 ES is read through `scenario_risk.models.run_all()`, which uses its cache.

## 8. New avenues
1. **Intraday execution for the two events** (1-minute H/J bars for 13–14 Nov 2018 and 24 Jan – 20 Feb 2014). The stop result swings from the worst to the best variant on fill timing, so this settles whether a stop is a viable overlay. *Expected value: high, and cheap.*
2. **Price option protection against this frontier.** Exits cost at least 1:1 (premium for drawdown). An H−J call spread (CSO) costing less than that would dominate every overlay here. *Expected value: high if CSO data can be obtained.*
3. **A two-mode overlay.** Combine the next-year hedge, which helped in 2014's persistent scarcity, with a pre-emptive physical or volatility flat for squeezes, which helped in 2018. Test it only on pre-2010 winters if Models 1–2 are back-cast. *Expected value: medium.*
4. **More tail events.** Extend the price history before 2010 (2003, 2006, 2008 winters). With two events, no frontier point is identified. *Expected value: high; shared with `premium_map` avenue 3.*
