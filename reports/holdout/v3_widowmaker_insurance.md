# V4: widowmaker W1 vs its price-only twin (re-used holdout, run once, 22 Nov 2026)

*Descriptive test (`reports/synthesis_v3.md`, `reports/agents/review_v3.md`). The spec is W1 from `reports/agents/widowmaker_insurance.md` §6, with the review's change: the primary comparison is W1 vs the "spread > $0.17" twin. The runner is `src/holdout/widowmaker_insurance.py`. The research files were not edited. The runner was checked at end date 2021-12-31 before the single holdout run, and nothing was changed afterwards.*

## Verdict
- **Model 2 adds no value. Primary criterion: fail.**
  - On winters 2022-23 to 2025-26, W1 earned **+$2,647 per trade** net (15 trades; winter-block CI 1,141–3,964).
  - The spread > $0.17 twin earned **+$3,308** (11 trades).
  - W1 − twin = **−$662 per trade**, against a required +$500.
- **The insurance premium itself replicated.**
  - W1 met its own descriptive criterion: mean net > 0, and a decay slope of **1.04** (90% winter-block CI 0.45–1.41, which excludes 0).
  - The mean premium was $0.15. The spread fell by a mean of $0.17 from t to expiry.
- **The result is essentially one episode.** 31,710 of W1's 39,700 dollars came from Nov–Dec 2022. Only 2 of the 4 winters traded.

## Validation check (end 2021-12-31)
The check reproduces the research numbers exactly.
- The full Model 2 cache equals the dev cache on all 95 dev origins.
- Frozen curve: a = −0.133, b = 0.087, τ = 634 (n = 11).
- Mean premium $0.177; slope 1.042.
- W1: 29 trades, +$2,386 per trade.
- Twin at $0.15: 26 trades, +$2,645.
- **Twin at $0.17: 23 trades, +$2,913.** So W1 already trailed the $0.17 twin by $528 per trade in development. The holdout repeats that ordering.

## Holdout results, winters 2022-23 to 2025-26 (61 decision weeks)

| Rule (4-week hold or to expiry, $160 round trip) | Trades | Winters | Mean net $/trade (95% CI) | Hit rate | Worst trade | Worst MTM | Total |
|---|---|---|---|---|---|---|---|
| **W1: premium > $0.15** | 15 | 2 | **+2,647 (1,141–3,964)** | 93% | −160 | −7,530 | +39,700 |
| **Twin: spread > $0.17** | 11 | 2 | **+3,308 (1,560–3,964)** | 91% | −160 | −7,530 | +36,390 |
| Twin: spread > $0.15 | 12 | 2 | +3,132 (1,468–3,964) | 92% | −160 | −7,530 | +37,580 |

The CIs come from a bootstrap over only 2 trading winters, so they are nominal.

**By winter** (`figures/v3_widowmaker_insurance_holdout.png`):

| Winter | Mean spread | Mean AV | Mean premium | Mean P(expiry > $0.20) | Expiry spread | W1 trades / net | Twin $0.17 trades / net | W1 worst MTM | Short-all-winter worst MTM |
|---|---|---|---|---|---|---|---|---|---|
| 2022-23 | 0.32 | −0.03 | 0.35 | 0.03 | −0.10 | 8 / +31,710 | 8 / +31,710 | −7,530 | −7,530 |
| 2023-24 | 0.02 | −0.06 | 0.08 | 0.02 | −0.19 | 0 | 0 | – | −200 |
| 2024-25 | 0.01 | −0.04 | 0.05 | 0.03 | −0.05 | 0 | 0 | – | −870 |
| 2025-26 | 0.09 | −0.04 | 0.13 | 0.02 | +0.10 | 7 / +7,990 | 3 / +4,680 | −2,190 | −1,720 |

"Short-all-winter worst MTM" is a short opened at the first decision week and held to expiry. It is the path risk a seller carried.

**Decay test** (all 61 weeks with an expiry outcome): slope of (S_exp − S_t) on −premium = 1.04 (90% CI 0.45–1.41), intercept −0.014.

**Jan–Feb 2022, reported separately:** 7 weeks.
- W1 made 1 trade, +$2,230 (worst MTM −$2,910).
- The $0.17 twin did not trigger. The $0.15 twin made the same trade.

## Reading
- **The actuarial value barely moved:** −$0.03 to −$0.06 by winter, and P(large) was 2–3% throughout. W1's trigger was therefore just "spread > about $0.11–0.12", a lower bar than the twin's.
- The extra W1 trades all came in 2025-26 (4 more than the twin). They were profitable but small (+$828 each), so W1 earned more in total and less per trade.
- **The physics did not flag the only drawdown.** In Nov 2022 the spread spiked to $1.21 while the physical distribution stayed near zero, and the short carried a −$7,530 drawdown before paying out.
- This matches the development finding: the premium is real and decays roughly one-for-one, but Model 2 conditioning adds nothing over the spread level.
- **Caveats:**
  - The holdout is re-used, and the v3 thesis was formed with these winters in view (synthesis_v3 disclosure).
  - Only 2 of 4 winters traded, and one episode (Nov 2022) dominates.
  - The slope criterion is near-automatic when the actuarial value is nearly constant (review_v3).

## Rerun
`PYTHONPATH=. python -m src.holdout.widowmaker_insurance check`, then `... holdout`. Each takes about 25 s.
- Outputs go to `data/processed/research_widowmaker_insurance/holdout_v4_*.csv` and `holdout_v4_log.txt`.
- The figure is `reports/holdout/figures/v3_widowmaker_insurance_holdout.png`.
