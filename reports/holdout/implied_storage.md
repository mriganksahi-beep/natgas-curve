# Holdout test H3: implied-storage gap vs spread-only (`implied_storage`)

*Run once on 21 Nov 2026. The specification is frozen in `reports/agents/implied_storage.md` §6 and `reports/synthesis_v2.md`. There were no spec changes. Runner: `src/holdout/implied_storage.py`.*

## Verdict: **FAIL**

The test asks whether the gap model forecasts the 4-week change in the March−April spread better than the spread-only model.

| | Holdout (origins 2022-01-07 to Feb 2026) | Validation (2016-17 to Dec 2021), frozen coefficients |
|---|---|---|
| Weeks in the window | 53 | 70 |
| Readable weeks (implied s.e. ≤ 250 Bcf) | **9** (2022-23: 8 of 12; 2025-26: 1 of 12; Jan 2022: 0 of 3; 2023-24 and 2024-25: 0) | 43 |
| Test rows (with a 4-week target) | **8, all in winter 2022-23** | 43 (6 winters) |
| MSE, gap model ($/MMBtu)² | 0.0537 | 0.0566 |
| MSE, spread-only model | 0.0348 | 0.0372 |
| MSE, no change | 0.2364 | 0.1110 |
| DM mean (e²_gap − e²_spread) | +0.019 | +0.019 |
| DM t (HAC, lag 4), one-sided p (gap better) | t = 1.43, **p = 0.92** | t = 1.57, p = 0.94 |
| OOS R², gap model vs no change | **+0.77** | +0.49 |
| OOS R², spread-only vs no change | +0.85 | +0.66 |
| **Pass: p < 0.05 and R² > 0** | **No.** R² > 0 holds, but DM goes the wrong way. | No |

- **Why it fails:** the gap model beats "no change" but loses to the spread-only model, as it also does on validation. On both samples, the spread-level information in the gap explains the forecast, and the Model 2 part adds nothing.
- **How little the holdout can show:** 8 overlapping 4-week observations from a single winter. The test has almost no power, and a pass would not have been convincing either.

## Secondary (not pass/fail)

| | Holdout readable weeks (n = 9) | Validation (n = 43) |
|---|---|---|
| Mean implied level − 5-year path (Bcf) | **−454** (winter bootstrap −490 to −450 ¹) | −243 (−353 to −165) |
| Mean implied level − Model 2 median (Bcf) | −392 | −105 |
| Mean percentile in Model 2 | 0.24 (1 of 9 weeks below p10) | 0.44 (0 of 43 outside 10–90) |
| Accuracy against realised end-March, MAE (Bcf): implied / Model 2 / 5-year path | 732 / 388 / 297 | 339 / 292 / 183 |

¹ The bootstrap covers effectively two winters, one of which has a single row, so this interval is not meaningful. Read it as "8 weeks of one winter".

- **The pessimism repeated in the holdout and grew.** In Nov–Dec 2022 the spread reached 45–120% of April's price. The implied end-March level was 880–1,270 Bcf, against a 5-year path of about 1,500 Bcf and a realised level of **1,830 Bcf**.
- **The direction matches validation, and so does the lesson.** The market was more pessimistic than both the 5-year path and Model 2, and it was wrong by more than either. The 5-year path was again the most accurate of the three.

## Readability
- **Readability depends on the winter.** The spread was steep enough to invert only in a stressed winter (2022-23) and for one week in Jan 2026.
- **The four mild winters were mostly unreadable.** From Jan 2022 and in 2023-24 and 2024-25, every week had a spread too small to read, so the market-implied level is only a lower bound there.
- **This matches validation.** Then, 2019-20 and 2020-21 were also mostly unreadable (6 of 25 weeks).

## Figure
`reports/holdout/figures/implied_storage_holdout_headline.png` extends the headline figure through all ten winters. Holdout winters are labelled, and Jan–Feb 2022 sits to the right of the dotted line in the 2021-22 panel.

## Comparison with validation, and an admission
- **The proposed test was already failing on validation.** With the frozen coefficients applied to the validation rows, the gap model *loses* to the spread-only model: p = 0.94, and in-sample R² 0.28 for the gap model against 0.53 for the spread-only model.
- **The validation report pointed the other way.** It showed walk-forward OOS R² of 0.39 for the gap and 0.31 for the spread-only model, which made the comparison look like a contest. I froze the test without checking the frozen-coefficient comparison on validation.
- **The verdict stands either way.** The H3 failure confirms the validation conclusion: "departures predict the spread" is spread mean reversion, and Model 2 adds nothing to it.
- **The review's selection-bias point does not affect H3.** That point concerns the who-moves filter on week t+h. H3 selects rows only on readability at t, and the 4-week target exists whenever t+4 weeks falls before the March contract's last 25 days, which does not depend on prices.

## Procedure and integrity notes
- **Validation check (end date 2021-12-31):** reproduced the validation numbers exactly. 70 weeks, 43 readable; refitted coefficients −0.1103/0.0675 and 0.0129/−0.6326; −243 Bcf against the 5-year path; −105 Bcf against Model 2; mean percentile 0.44.
- **Model 2:**
  - Re-simulated with one `Simulator()` (seed 0) over every Nov–Feb origin from Nov 2016 to 27 Feb 2026, in date order: 172 origins, about 1 minute.
  - The first 95 origins match the validation cache exactly (maximum median difference 0.000 Bcf).
  - Cached in `data/processed/research_implied_storage/model2_percentiles_full_to_2026.csv`.
- **One mechanical fix**, made during the validation check before any holdout run: the DM helper indexed a NumPy array with `.iloc`. I replaced that with `np.asarray`, which changes no numbers.
- **Run once:** the holdout mode writes `H3_holdout_ran.txt` and refuses to run again. The week-level output is in `H3_weekly_all.csv`, in the same folder.
- **Rerun:** first `PYTHONPATH=. python -m src.holdout.implied_storage check` (about 10 s). The one-shot holdout command is `PYTHONPATH=. python -m src.holdout.implied_storage holdout`; it needs the flag file removed and about 1 minute of simulation if the cache is missing.
