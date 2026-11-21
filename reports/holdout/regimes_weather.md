# Holdout: regimes_weather tests H2 and H5a (run once, 21 Nov 2026)

*The specs are frozen as in `reports/synthesis_v2.md` and `reports/agents/regimes_weather.md` §6. Runner: `src/holdout/regimes_weather.py`. It wraps the research code unchanged and overrides only the date cutoff, the cache folder and the cost.*

**Holdout sample:** winter Fridays from 2022-01-07 to 2026-02-20, 73 weeks.
- Winters: Jan–Feb 2022, then 2022-23 through 2025-26.
- The filter for 2021-22 starts in Nov 2021, as frozen.

## Verdicts
| Test | Statistic | Holdout | Threshold | Verdict |
|---|---|---|---|---|
| **H2 (1)** | Slope of fwd₄ on ΔS, weeks stressed the week before | **−0.455** (HAC t −1.81, one-sided p = 0.036; n = 12 weeks in 3 winters) | < 0 and p < 0.05 | Pass |
| **H2 (2)** | Spike-fade rule, mean net $ per spread at $160 round trip | **+$3,360** (2 trades, both winning; winter-bootstrap CI $2,480–4,240) | > 0 | Pass |
| **H2** | Both conditions | | | **PASS, but weak (see below)** |
| **H5a** | R² of ΔS on the storage surprise (5-year average) | **0.023** (bootstrap CI 0.003–0.161; n = 73) | ≤ 0.05 | **PASS** |

**Why H2 is weak despite passing:**
- **Bootstrap CI.** The winter-bootstrap CI for the slope is −0.50 to +13.6. There are only 3 winters, and a resample that drops 2022-23 leaves two stressed weeks with ΔS ≈ 0, so the slope is undefined.
- **One episode.** 10 of the 12 stressed-before weeks are Nov 2022–Jan 2023. Dropping that winter leaves only those two near-zero weeks, so the slope is not identified without it. Dropping either of the other winters gives −0.47 or −0.48.
- **Partly a falling spread.** In that episode, fwd₄ was negative in 11 of 12 weeks, including after down-jumps. The negative slope partly reflects the spread collapsing as winter 2022-23 turned out mild.
- **Two trades.** The P&L rests on 2 trades.

## H2 trades (short 1 March−April spread, $160 round trip)
| Entry (Friday) | Weekly jump | P(stressed) the week before | Hold | Spread change over hold | Net $ |
|---|---|---|---|---|---|
| 2022-11-18 | +$0.248 | 0.86 | 4 weeks | −$0.440 | **+4,240** |
| 2026-01-30 | +$0.219 | 0.86 | 3 weeks (March expiry) | −$0.264 | **+2,480** |

- **A missed trade.** On 2022-11-25 the spread jumped another +$0.320 and then fell $0.94 over 4 weeks. It was not traded because the first position was still open.
- **At the validation cost of $80:** +$3,440 per trade.

## H5a: news decomposition on the holdout
| Spec | R² (95% CI) |
|---|---|
| Storage surprise only | 0.023 (0.003–0.161) |
| + next-2-week HDD anomaly | 0.051 (0.013–0.268); weather t = 2.3 |
| Increase from weather | **+0.028** (0.001–0.135) |
| Both effects grow with how wide the spread is | 0.203 (0.18–0.36) |

- **Storage still explains almost nothing.** Its slope has t = 2.0, with the expected sign (a larger draw than normal widens the spread), but the R² is small.
- **Weather adds less than before:** +2.8 points of R², against +8.1 in 2012–21 and +12.9 in 2017–21.
- **The tightness-scaled spec still explains about 20%.** In the holdout, part of that comes from storage scaled by the spread (R² 0.125 on its own), not only weather.

## Regimes around the big 2022–2026 episodes
See the figure `figures_regimes_weather_regimes.png`.

- **Out of sample, the regime still separates volatility.** Next-week |ΔS| averaged 12.0¢ in stressed-before weeks (n = 17) and 4.5¢ in calm weeks (n = 51). In validation the figures were 11.5¢ and 5.6¢, using the regime filtered at t.
- **Jan–Feb 2022.** Calm until a +13¢ jump on 28 Jan. The filter then went to stressed (P 0.86 the week after) for the rest of February. No fwd₄ falls before expiry.
- **Nov 2022 – Jan 2023 (energy crisis, then a mild winter).**
  - Calm before the +18¢ move on 4 Nov (P = 0.19). Stressed before every week from 11 Nov to 6 Jan except 16 Dec (P = 0.41).
  - Weekly moves: −21¢, +25¢, +32¢, −56¢, then a steady decline.
  - The +32¢ spike on 25 Nov was the largest stressed-before spike.
- **Jan 2024 cold snap and Jan 2025.** The March−April spread barely moved (|ΔS| ≤ 5.5¢). P(stressed) stayed between 0.04 and 0.08. Neither winter had any stressed week (6% of 2023-24, 0% of 2024-25).
- **Dec 2025 – Feb 2026.**
  - Calm, then a −17¢ move on 12 Dec gave one stressed week.
  - Calm again until a +13¢ jump on 23 Jan (P = 0.05 before it).
  - Stressed from 30 Jan (the +22¢ spike, traded), then −13.5¢, −5¢ and −8¢.
- **Calm-before slope (secondary, predicted ≥ 0):** +0.12 (t 0.35, n = 41). Positive as predicted, but nothing like the validation's +0.77 continuation.

## Comparison with validation
| | Validation 2017–21 | Holdout 2022–26 |
|---|---|---|
| Stressed-before slope (HAC t, n) | −0.40 (−2.47, 24) | −0.455 (−1.81, 12) |
| Winters contributing stressed weeks with fwd₄ | 5 | 3 (10 of 12 weeks from 2022-23) |
| Rule trades, mean net at $160 | 5, +$3,130 | 2, +$3,360 |
| Storage-surprise R² | 0.001 | 0.023 |
| Weather adds (ΔR²) | +0.129 | +0.028 |
| Next-week \|ΔS\|, stressed vs calm | 11.5¢ vs 5.6¢ (regime filtered at t) | 12.0¢ vs 4.5¢ (regime from t−1) |

**Reading.** The direction and size of the stressed-week reversal replicated: slope about −0.4 to −0.5, and spikes faded after the regime was already stressed. But the holdout contains effectively one stressed episode plus one trade in 2026. Together with the review's multiple-testing note (best of 32 cells), this is supportive but thin evidence, not a confirmed edge. As in validation, the money came from the release of a scarcity premium in November–December (2018, 2021, 2022). H5a replicates cleanly: storage reports explain about nothing of weekly spread moves. The weather contribution was smaller in this sample of mostly calm winters.

## Procedure and deviations
1. **Check run, end 2021-12-31:**
   - Rebuilt 190 rows.
   - Filtered probabilities matched validation to 2.8e-15.
   - Reproduced the 2017–21 slope: −0.40, t −2.47, n = 24.
   - Reproduced the rule: 5 trades, +$3,210 at $80 and +$3,130 at $160.
   - Reproduced H5a: R² 0.000, ΔR² 0.081 (2012–21).
2. **Holdout run once,** with data to 2026-11-17, using the frozen JSON parameters (not refitted). There were no crashes and no spec changes.
3. **The one change after the run was cosmetic:** the winter labels on the figure were clipped. I redrew the figure from the saved panel without rescoring.
4. **Cost.** At the coordinator's per-side convention, $160 per round trip. Validation used $80, and both are shown.
5. **Mechanical note.** With the extended data, the 2021-22 March expiry is now the true date (24 Feb 2022) rather than the truncated 2021-12-31. This affects only which fwd₄ values exist.

Outputs:
- Results: `data/processed/research_regimes_weather/holdout/holdout_results.json`
- Panel: `data/processed/research_regimes_weather/holdout/panel_holdout.csv`

Rerun: `python -m src.holdout.regimes_weather holdout` (about 30 s).
