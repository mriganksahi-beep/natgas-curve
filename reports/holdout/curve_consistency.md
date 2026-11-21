# Holdout H1: extreme-kink fade (`curve_consistency`). Verdict: **FAIL**

I ran this once, on 2022-01-03 to 2026-11-17. The spec is frozen as in `reports/synthesis_v2.md` (H1).

**Spec**
- Signal: fitted-curve butterfly kink fA, using variant A of the fair curve: a daily quadratic in months to delivery plus a 12-month seasonal shape.
- Seasonal shape: refit each 1 January from the previous 3 calendar years.
- Contracts: the first 18, excluding any within 5 trading days of expiry.
- Entry: when |fA| > 0.0424, one 1:−2:1 butterfly per centre contract per signal day, short the centre when it is rich, entered at that day's settle.
- Exit: the settle 20 trading days later. Trades whose exit would fall after 2026-11-17 are dropped.
- Cost: k ticks per contract per side, so a round trip costs 4 × 2 × k × $10 per butterfly ($320 at k = 4).

**Pass rule:** mean net > 0 at 4 ticks per side **and** at least 60% of episodes positive, with at least 3 episodes required.

## Verdict
- Mean net at 4 ticks per side is **−$54 per butterfly**, with a 95% CI of −$571 to +$508.
- Episodes positive: **3 of 7 (43%)**, below the 60% bar.
- Both conditions fail, so the result is **FAIL**.

## Results (1,108 trades, 43 months with trades; CI from a 3-month moving-block bootstrap)

| Cost | Mean net $ per butterfly | 95% CI |
|---|---|---|
| Gross | +266 | −219, +839 |
| 2 ticks per side | +106 | −417, +678 |
| **4 ticks per side** | **−54** | **−571, +508** |
| 6 ticks per side | −214 | −715, +403 |
| 10 ticks per side | −534 | −1,042, +28 |

## Episodes
Episodes are signal days separated by more than 20 trading days. Net is at 4 ticks per side.

| # | Entry month | Last entry | Trades | Main centre contracts | Mean net $ per butterfly |
|---|---|---|---|---|---|
| 1 | 2022-01 | 2022-10-07 | 271 | NGJ23, NGH23, NGG23 | +563 |
| 2 | 2022-11 | 2023-01-18 | 17 | NGJ23, NGH23 | +1,651 |
| 3 | 2023-02 | 2023-02-22 | 3 | NGV23 | +590 |
| 4 | 2023-04 | 2023-07-17 | 40 | NGV23 | −78 |
| 5 | 2023-10 | 2024-12-31 | 369 | NGH25, NGJ24, NGH24 | −535 |
| 6 | 2025-11 | 2025-12-31 | 35 | NGH26, NGJ26 | −1,109 |
| 7 | 2026-03 | 2026-10-20 | 373 | NGH27, NGF27, NGX26 | −8 |

**Descriptive breakdown** (not part of the test). Mean net by entry year was:

| Entry year | Mean net $ per butterfly |
|---|---|
| 2022 | +678 |
| 2023 | −588 |
| 2024 | −449 |
| 2025 | −1,109 |
| 2026 | −8 |

By centre contract: March/April centres (767 trades) averaged −$262, and all other centres (341 trades) averaged +$413.

## Comparison with validation

| | Validation 2018–21 (same runner, end 2021-12-31) | Holdout 2022–26 |
|---|---|---|
| Trades / episodes | 251 / 5 | 1,108 / 7 |
| Gross $ per butterfly | +2,326 | +266 |
| Net at 4 ticks per side | +2,006 (−568, +3,421) | −54 (−571, +508) |
| Net at 10 ticks per side | +1,526 | −534 |
| Episodes positive | 4 of 5 (80%) | 3 of 7 (43%) |
| Rule verdict | PASS | **FAIL** |

The check run reproduces the review's corrected validation numbers exactly: 251 trades, gross $2,326, +$2,006 at 4 ticks per side and +$1,526 at 10.

**Reading the result**
- **The edge was the scarcity-premium release in two winters.** In the holdout, extreme kinks still reverted a little on average: gross +$266, about one-ninth of validation. That does not cover realistic costs. Episode positivity fell below a coin flip after 2022.
- **The threshold fires much more often.** The frozen 424 bp threshold triggered about 4 times as often per year as in 2018–21. Kinks were larger after 2022 (mean |fA| on signal days was 536 bp), so the rule spent long stretches inside persistent kinks rather than catching brief spikes. This is consistent with the Q1 finding that most kinks are state information, not errors.
- **Only 2022 paid.** The 2022 energy-crisis episodes were profitable, and 2023–2025 gave the money back.

## Procedure notes
- **Runner.** `src/holdout/curve_consistency.py` wraps the research code without editing it. It copies `load_curve` with the cutoff at the end date plus 1 day, and copies `fit_all` (variant A) with the shape years extended to the end year.
- **No crashes and no spec changes.** The holdout was run exactly once, after the validation check passed.
- **Mechanical points, both by design:**
  - Contracts that had not expired by the end date get dte = 999, as in validation.
  - Trades without an exit row (an expiry window, or past 2026-11-17) are dropped. The latest exit is 2026-11-17.
- **Rerun.**
  - Validation check: `python -m src.holdout.curve_consistency check` (about 5 seconds).
  - Holdout: `python -m src.holdout.curve_consistency holdout` (about 5 seconds; already run once, do not use for choices).
  - Outputs go to `data/processed/research_curve_consistency/holdout_*`.
