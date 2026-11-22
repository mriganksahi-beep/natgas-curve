# V5 holdout: scenario-risk test H-SR (run once, 22 Nov 2026)

**Verdict: FAIL on criterion 1. This test is descriptive, and the holdout is a re-used one.**

PHYS-M2 breached its 1-week 99% VaR 3 times in 64 weeks (4.7%; Kupiec p = 0.031 < 0.05). Its ES backtest beat EWMA's, with Z2 1.37 against 2.36, but the difference has a winter-bootstrap CI of [−3.1, +1.8], which includes zero. The spread level ranks blow-up weeks better than PHYS-M2 does (AUC 0.68 against 0.58). A PHYS-M2-sized book had a Sharpe of 0.20, against 0.83 for holding 1 spread.

## Spec (frozen in `reports/agents/scenario_risk.md` §6)
- **Model:** PHYS-M2 exactly as coded in `src/research/scenario_risk/models.py`.
  - The research code is imported unchanged. The runner changes only the cutoff, the cache folder and the Model 2 source, which is `data/processed/research_implied_storage/model2_percentiles_full_to_2026.csv`, filtered to origin ≤ end.
  - The map, η and β are refit walk-forward on earlier winters. Model 2 is not rescaled.
- **Forecasts:** every Friday from 1 Nov to 15 Feb, horizon 1 week. Holdout weeks are those whose target window ends in 2022 or later (Friday ≥ 2021-12-31).
- **Weeks evaluated:** 64 weeks over 5 winters:

  | Winter | Weeks |
  |---|---|
  | 2021-22 | 7 (Jan–Feb only) |
  | 2022-23 | 14 |
  | 2023-24 | 14 |
  | 2024-25 | 15 |
  | 2025-26 | 14 |

  The Model 2 cache ends at origin 2026-02-27, so Nov 2026 is not covered.
- **Pass requires both:**
  1. Kupiec p ≥ 0.05 for the 99% VaR.
  2. Acerbi–Szekely Z2 for PHYS-M2 below EWMA's on the same weeks.
- **Comparators added per `review_v3.md`, declared before the holdout run:**
  - **HYB:** as coded in the research folder.
  - **LEVEL:** a spread-level-only twin. The loss is normal with s.d. max(S − a, 0.02) × s̄_z × √h, where s̄_z is the s.d. of the level-scaled weekly changes in earlier winters.
  - **Ranking scores** for the blow-up AUC (1-week loss > $1,000): the spread level S, and S − a.
  - **Sizing:** a PHYS-M2-sized book (N = $5,000 / ES) against 1 spread, at 4 ticks per side.
- **Descriptive only:** GARCH-t, HS-3y and PHYS-WX.

**Check run (end 2021-12-31).** The runner reproduced the dev numbers exactly:
- PHYS-M2: 1 breach in 79 weeks, Kupiec p 0.820, Z2 0.526 against 2.540 for EWMA.
- AUC: 0.719 for PHYS-M2 against 0.754 for the spread level.
- Sizing Sharpe: 0.74 for PHYS-M2 against 1.08 for constant size.

**One fix before the holdout run.** I fixed one bug found in the check run: the descriptive comparators were being dropped by the common-weeks filter. It did not touch any primary quantity. The spec was not changed after the holdout run.

**Correction to the dev report.** The dev M2 sample had 5 blow-up weeks in 4 winters, not 4.

## Results (holdout, n = 64)
| Model | 95% breaches | 99% breaches (n) | Kupiec p (99%) | Z2 | Mean ES $ | CRPS $ | Blow-up AUC |
|---|---|---|---|---|---|---|---|
| **PHYS-M2** | 6.2% | 4.7% (3) | **0.031** | **1.37** | 2,230 | 446 | 0.58 |
| EWMA | 7.8% | 4.7% (3) | 0.031 | 2.36 | 2,282 | 502 | 0.65 |
| HYB | 12.5% | 10.9% (7) | <0.001 | 7.92 | 1,195 | 443 | 0.63 |
| LEVEL | 6.2% | 6.2% (4) | 0.004 | 2.11 | 2,912 | 477 | 0.58 |
| GARCH-t | 9.4% | 3.1% (2) | 0.17 | 1.44 | 2,446 | 467 | **0.72** |
| HS-3y | 9.4% | 3.1% (2) | 0.17 | 2.69 | 3,254 | 508 | 0.31 |
| PHYS-WX | 9.4% | 4.7% (3) | 0.031 | 2.44 | 1,319 | 474 | 0.58 |
| Spread level S | | | | | | | 0.68 |
| S − a | | | | | | | 0.60 |

- **Blow-up count.** There were 5 blow-up weeks, in 3 winters.
- **AUC gap.** AUC(PHYS-M2) − AUC(S) = −0.09, with CI [−0.60, +0.02].

**Where PHYS-M2 breached.** All three breaches came in loose markets, when the spread sat near zero:

| Friday | Spread S | Loss | PHYS-M2 VaR99 |
|---|---|---|---|
| 21 Jan 2022 | $0.04 | $1,300 | $1,100 |
| 16 Jan 2026 | −$0.02 | $1,320 | $1,111 |
| 23 Jan 2026 | $0.11 | $2,189 | $1,551 |

- **Jan 2026 is one episode.** The last two breaches are consecutive weeks (Christoffersen p = 0.096). EWMA breached on both of those weeks as well. LEVEL also breached in Jan 2026, on both weeks.
- **The tight-market spike was covered.** In Nov 2022 the spread rose $2,480 and then $3,200 in successive weeks, from a level of $0.64–0.89. PHYS-M2's ES was $8,126 and $11,534 in those weeks. EWMA's ES of $4,837–4,899 also covered them.

**Sizing** (4 ticks per side, values per average spread held):

| Book | P&L | Worst week | Sharpe | Winters positive |
|---|---|---|---|---|
| Constant 1 spread | **+$8,120** | −$3,200 | **0.83** | 3/5 |
| PHYS-M2 | +$1,041 | −$2,134 | 0.20 | 2/5 |
| EWMA | +$1,190 | −$2,111 | 0.20 | 3/5 |
| LEVEL | −$1,564 | −$2,224 | −0.30 | 2/5 |
| HYB | −$1,002 | −$2,515 | −0.21 | 2/5 |

PHYS-M2's P&L by winter: −$4.7k (2021-22), +$11.4k (2022-23), +$4.8k (2023-24), −$5.1k (2024-25), −$2.9k (2025-26).

## Reading
1. **PHYS-M2's tail is built only for tight markets, and the holdout's breaches came from loose ones.** The convex map makes risk scale with (S − a), so jumps from a near-zero spread are under-forecast. The Jan 2022 and Jan 2026 jumps were cold-weather moves starting from ample storage, which is outside what the map can capture. In the one tight-market spike (Nov 2022) its wide ES covered the losses.
2. **The Z2 edge over EWMA repeats in direction (dev 0.53 vs 2.54; holdout 1.37 vs 2.36) but is not significant either time.** GARCH-t did as well as PHYS-M2 on Z2 (1.44) with fewer breaches, and it ranked blow-ups best.
3. **There is again no evidence that physics adds to the spread level.**
   - The spread level outranks PHYS-M2 for blow-ups.
   - The level-only twin's coverage is similar (4 breaches against 3).
4. **ES-budget sizing again loses to constant size, now in both periods.** Constant size had Sharpe 1.08 in dev and 0.83 in the holdout. The PHYS-M2 book had 0.74 in dev and 0.20 in the holdout.

This is consistent with synthesis_v3: no physical model beats the spread itself out of sample. The live alarm log for 2026-27 (PHYS-M2 alarm against the (S − a) alarm) should continue as planned. It is likely to add little, since loose-market jumps are the blind spot.

## How it was run
- **Commands:**
  - Check run: `PYTHONPATH=. python -m src.holdout.scenario_risk --end 2021-12-31` (about 40 s).
  - Holdout run: `PYTHONPATH=. python -m src.holdout.scenario_risk --end 2026-11-17` (run once, about 45 s).
- **Outputs:** in `data/processed/research_scenario_risk/holdout_end_<date>/` (`v5_result.json`, `v5_table.csv`, `v5_sizing*.csv`, `v5_weeks.csv`, `v5_big_weeks.csv`). The run log is `data/processed/research_scenario_risk/v5_holdout_run_log.txt`.
