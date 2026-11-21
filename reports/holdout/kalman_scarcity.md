# Holdout: frozen tests H4 and H5b (agent `kalman_scarcity`)

*Run once on 21 Nov 2026. Specifications are as frozen in `reports/synthesis_v2.md` and `reports/agents/kalman_scarcity.md` §6. No changes were made after the results were seen.*

## Verdicts
| Test | Statistic (holdout, Nov–Mar Fridays from 2022-01-07, 100 weeks) | 95% CI (winter bootstrap) | Pass rule | Verdict |
|---|---|---|---|---|
| **H4** | D = corr(s_t, storage surplus known) − corr(Mar−Apr, same) = **−0.017** | [−0.233, +0.134] | D < 0 and CI upper < 0 | **FAIL** (as expected) |
| **H5b** | corr(Δs_t, surprise published at t+3) = **−0.206** (n = 93) | [−0.371, −0.022] | ≤ −0.10 and CI upper < 0 | **PASS** |

- **H4.** On its own, the Kalman state correlates −0.49 [−0.69, +0.09] with storage known at the time, and Mar−Apr −0.48 [−0.68, +0.21]. The two are practically identical. In 2010–21 they were −0.64 and −0.56.
- **The Kalman state adds nothing over Mar−Apr,** in or out of sample.

## H5b report index (per the review)
- surprise_pub(t+k) is the EIA surprise (actual weekly change minus the prior-5-year average change) released in the week ending Friday t+k. It covers the storage week ending F + 7(k−1).
- **k = +1** is the report for the same physical week as the price move, released at F+6. **k = +3**, the H5b statistic, is two reports after that and covers the week ending F+14.
- So the true forecast lead is about 0–2 weeks, not 3.

## All lags, holdout (Nov–Mar; correlation of the weekly change with the surprise in report t+k)
| k | −2 | −1 | 0 | +1 (same week) | +2 | **+3** | +4 | +5 |
|---|---|---|---|---|---|---|---|---|
| Kalman Δs | −0.05 | +0.10 | +0.09 | +0.07 | −0.10 | **−0.21** [−0.37, −0.02] | +0.11 | −0.04 |
| Mar−Apr Δ | −0.04 | +0.10 | +0.14 | −0.11 | −0.12 | −0.18 [−0.38, −0.13] | +0.12 | −0.01 |
| 2010–21 Kalman Δs | 0.00 | −0.04 | +0.04 | −0.17 | −0.16 | −0.27 | −0.05 | +0.03 |

- **The peak at k = +3 replicates.** The week ending F+14 is again the strongest lead.
- **The same-week (k = +1) link did not hold** for the Kalman state (+0.07, against −0.17 before). It held weakly for Mar−Apr (−0.11).
- **The report released inside the price interval (k = 0) again shows no negative link:** +0.09 for the Kalman state, +0.14 for Mar−Apr.

**Weather-controlled version** (Nov–Mar, HAC with 4 lags; the coefficient is in Bcf per standard deviation of Δs_t, using the 2010–17 sd):

| Regression | Holdout | 2010–21 |
|---|---|---|
| Surprise at k=+3 on Δs_t | −14.3 (t −2.1) | −25.7 (t −3.7) |
| … + realised HDD anomalies for weeks t, t+1, t+2 | −4.4 (t −2.1) | −7.5 (t −2.0) |
| Surprise at k=+1 on Δs_t + HDD of week t | −2.6 (t −1.0) | −6.4 (t −2.0) |

Realised weather absorbs 69% of the lead-3 coefficient, against 71% before. That is consistent with the curve pricing weather forecasts about 2 weeks ahead. In both periods a small part of the lead, about −4 to −7 Bcf per sd, is not explained by realised weather.

## Figure
`reports/holdout/figures/kalman_scarcity_state_holdout.png` shows the filtered scarcity state (frozen random-walk parameters) against Mar−Apr from 2010 to 13 Nov 2026. It also shows the Markov-switching stressed-regime probability, filtered with the frozen 2010–17 parameters, which were stored rounded to 6 decimals.

What the figure shows (description only; not a test):
- **The state runs well above Mar−Apr from 2022.** It peaks at about $4.4 in 2022, against about $2 for Mar−Apr. From 2023–26 it stays about $0.3–1.0 above.
- **The likely cause is a shift in winter shape.** Since 2022, the early-winter spreads (Dec/Jan/Feb minus April) have been much steeper than Mar−Apr. The frozen loadings were fitted on 2010–17, when Mar−Apr moved with them, and they map that steepness into a high state. The one-factor structure fitted on 2010–17 therefore does not carry over unchanged in level.
- **The correlation-based tests are unaffected by this level offset.** Both signals still relate equally to known storage.
- **The regime probability is high for most of 2022 and the winters of 2025 and 2026.**

## Procedure and checks
- **Runner:** `src/holdout/kalman_scarcity.py`. It imports the validation code unchanged and overrides only the `END` cutoff and the cache folder at run time.
- **Frozen parameters:** `data/processed/research_kalman_scarcity/kf_params.json`, key `rw`.
- **Reproduction check** (`--end 2021-12-31`, run first):
  - The panel inputs match the validation panel exactly.
  - Every point estimate reproduces to the last digit: H4 D = −0.0813, H5b = −0.2648, all lags, and the regressions (−25.7, −7.5, −6.4).
  - The bootstrap CIs differ only by Monte Carlo noise (H4: [−0.228, 0.028] vs [−0.240, 0.036] originally), because of the random-number sequence.
- **Holdout run** (`--end 2026-11-17`): run once, exit 0, no crashes, no fixes. The last Friday is 2026-11-13. Bootstrap blocks are the target winters 2022–2027; the 2022 block holds only Jan–Feb 2022 and 2027 only Mar 2026 and Nov 2026.
- **Outputs:** `data/processed/research_kalman_scarcity/holdout_end_2026-11-17/holdout_results.json` and `holdout_run_log.txt`.

**Rerun:** `python -m src.holdout.kalman_scarcity --end 2026-11-17` (about 30 s).
