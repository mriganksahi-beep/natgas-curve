# The widowmaker as weather insurance (agent `widowmaker_insurance`, study v3 idea 1)

*Data ≤ 2021-12-31 only. Every loader filters at that date. Model 2 rows come only from the dev cache (origins Nov 2016 – Dec 2021). The 2022 March contract's expiry is never read, so winter 2021-22 enters only through Dec 2021.*

## 1. Plain-English findings
- **The March−April spread carries a large, decaying premium over its physical value.** We push Model 2's end-March storage scenarios through a fitted expiry payout curve. The actuarial value of the expiry spread is about **−$0.02**, and it barely moves (sd $0.05). The market spread averaged **+$0.18 above it** (95% winter-block CI 0.09–0.30). The premium was positive in 92% of winter weeks. It decays from about $0.32–0.38 at 12–16 weeks to expiry, to about $0.15 at 8–10 weeks, to about $0.01 in the last month.
- **The premium is earned roughly one-for-one.** Regressing (expiry spread − spread at t) on −premium gives a slope of **1.04** (CI 0.92–1.22) and an intercept of about 0. The physical distribution is roughly calibrated: the realised expiry landed inside its 10–90% band in 74% of weeks.
- **But Model 2 adds almost nothing over the price itself.** Because the actuarial value is nearly constant, the premium is essentially the spread plus 2¢. The spread alone predicts the decay just as well (R² 0.91 vs 0.90). With the spread included, the actuarial value's coefficient is 0.32 (HAC s.e. 0.47).
- **The physical filter ("P(large expiry) low") cuts the 4-week tail but costs mean P&L.** In 2016–19, requiring P(expiry > $0.20) < 0.10 lowered the mean in every premium row, and it cut the worst 4-week loss from −$7,860 to −$1,050. It cannot be validated: 2016–21 contains no disaster winter.
- **On prices alone, 2010–21:** shorting H−J from 1 November to expiry made money in **10 of 11 winters**, a mean of **+$1,174 per spread** net (CI 170–1,907). 2013-14 is the only expiry loss (−$3,140). But **2018-19 had a worse mark-to-market drawdown (−$13,170) than 2014 (−$11,920)** before ending +$1,920. The disaster risk is the path, not the expiry.

## 2. Method
**Payout curve.**
- The response is the H−J settle on the March last trading day: the tradable payout of a hold-to-expiry short.
- The driver is x = end-March storage (last EIA week from 25–31 March) minus the average of the prior years' end-March levels.
- Storage starts in 2010, so the average uses 1, 2, 3 and 4 prior years for 2011–14, and 5 years from 2015. The 2011 point therefore rests on a single prior year. Requiring at least 2 prior years changes the mean premium by 0.004 (§3, Table 2).
- Form: y = a + b·exp(−x/τ), with b ≥ 0 and τ ∈ [150, 3000] Bcf, so the curve is convex and monotone. It is fitted by profile least squares over a τ grid.
- **Parameter uncertainty:** a 500-draw residual bootstrap. Residuals are inflated by √(n/(n−3)) and the curve is refitted on each draw.
- **Walk-forward fits:** winter Y uses only winters < Y (6 points for 2016-17, rising to 11). The leave-one-winter-out (LOWO) fits use all other winters 2011–21. LOWO gives a steadier curve and lets every winter be scored, including 2014. It borrows future winters, though, so it supports description and not any tradable claim.

**Physical distribution.**
- For each Model 2 origin F (decision at the settle on the first trading day ≥ F+7, origins 1 Nov – 14 Feb), take p1–p99 as 99 equally weighted scenarios.
- Floor each scenario at 700 Bcf, near the record low. Model 2's p1–p10 go negative in Nov 2018, and the floor keeps x inside the fitted range.
- Map the scenarios through each bootstrap curve and add a resampled residual: 49,500 draws per week.
- **Bias-robust version (primary):** the scenarios are re-centred so that their median equals Model 2's 5-year-path benchmark, keeping Model 2's spread of outcomes. Raw Model 2 is shown alongside.

**Premium** = S_t − AV_t, where AV is the mean payout. The test is ΔS = S_exp − S_t regressed on −premium; under "market = physical value + premium", the slope is 1 and the intercept 0.

**Costs:** 4 ticks per contract per side, so $160 per spread for the round trip. One spread is 10,000 MMBtu per leg.

## 3. Results
**Payout curve** (`widowmaker_insurance_payout.png`):
- Fit on 2011–21: a = −0.133, b = 0.087, τ = 634 Bcf (bootstrap 10–90%: 373–1,259). Residual sd is $0.064.
- Implied expiry spread: −$0.05 at x = 0, +$0.05 at −500 Bcf and +$0.31 at −1,031 (2014).
- The walk-forward curves are stable: τ ranges from 566 to 658 for 2016-17 through 2021-22.

**Table 1. Out-of-sample RMSE of the expiry spread ($/MMBtu)**

| Spec | WF (6 winters) convex / linear / mean | LOWO (11 winters) convex / linear / mean |
|---|---|---|
| **Expiry-day spread, $ (primary)** | **0.079** / 0.099 / 0.089 | **0.082** / 0.104 / 0.137 |
| Scaled by April at expiry | 0.066 / 0.067 / 0.084 | 0.063 / 0.071 / 0.142 |
| Last-5-day average, $ | 0.076 / 0.191 / 0.116 | 0.225 / 0.266 / 0.265 |

- The primary spec was fixed before this comparison.
- The scaled spec looks better only because it converts back to dollars with the *expiry* April price, which is unknown at t. Using the April price at t, its mean premium is 0.189, against 0.177 for the primary.
- The 5-day average is unusable because of 2014, where the 5-day average was 0.78 against 0.31 on the expiry day.
- The convexity is identified almost entirely by 2014. With 2014 left out, the curve predicts 0.15 for 2014 against an actual 0.31.

**Table 2. Premium, walk-forward curve (84 weeks, 6 winters)**

| Centre | Mean premium (CI) | Weeks > 0 | Mean AV | Expiry inside 10–90% band | Mean (S_exp − AV) |
|---|---|---|---|---|---|
| **5-yr path** | **0.177 (0.094–0.304)** | 92% | −0.018 | 74% | −0.001 |
| Raw Model 2 | 0.155 (0.064–0.291) | 85% | +0.004 | 82% | −0.025 |
| 5-yr path, LOWO curve | 0.165 (0.082–0.290) | 89% | | 66% | −0.014 |

- Sensitivity of the 5-yr-centred mean premium: storage floor 500 / 900 Bcf gives 0.169 / 0.186; at least 2 prior years gives 0.173.
- By winter, the mean premium was 0.09 (2016-17), 0.11, **0.44 (2018-19)**, 0.10, 0.08 and 0.31 (Nov–Dec 2021).
- Physically, the spread had only a 12–25% chance (by-winter averages, 5-yr centre) of expiring above the market spread. The market priced the right tail well above its physical frequency.

**Key figure: `widowmaker_insurance_bands.png`.** The market spread sits above the actuarial value in essentially every week. It leaves the 10–90% band in Nov–Dec 2018 and Nov 2021, and converges by expiry.

**Theta** (5-yr centre; 95% winter-block CI; `widowmaker_insurance_theta.png`, left):

| Weeks to expiry | 14–16 | 12–14 | 10–12 | 8–10 | 6–8 | 4–6 | 2–4 | 0–2 |
|---|---|---|---|---|---|---|---|---|
| Premium ($) | 0.32 | 0.38 | 0.25 | 0.16 | 0.09 | 0.10 | 0.01 | 0.02 |
| CI | .18–.51 | .13–.72 | .10–.49 | .08–.25 | .06–.12 | .03–.19 | −.02–.05 | −.08–.08 |

**Table 3. Is the premium earned? (n = 76 weeks, 5 winters with an outcome; winter-block bootstrap)**

| Regressor (x) | Target | Slope on −x (CI) | Intercept (CI) | R² |
|---|---|---|---|---|
| Premium, 5-yr centre | ΔS to expiry | **1.04 (0.92–1.22)** | 0.005 (−0.07–0.09) | 0.90 |
| Premium, raw Model 2 | ΔS to expiry | 1.05 (0.97–1.32) | −0.018 | 0.91 |
| Premium, 5-yr centre | ΔS over 4 weeks | 0.69 (0.44–0.96) | 0.024 | 0.57 |
| **Spread alone (price only)** | ΔS to expiry | 0.91 (0.83–1.20) | −0.034 | **0.91** |
| Spread alone, 2010–21 (11 winters) | ΔS to expiry | 0.78 (0.49–0.92) | −0.027 | 0.72 |

- With both regressors, the spread's coefficient is 0.96 and AV's is 0.32 (s.e. 0.47).
- The slope of S_exp on AV is 0.48 (CI −0.91 to 1.32).
- Realised short-to-expiry P&L by premium bin: ≤ $0.05: −$37 per trade (19 weeks); $0.05–0.15: +$476 (32); > $0.15: **+$3,934 (25, worst +$500)**.

**Conditional rule.** Short one spread when premium > X and P(S_exp > $0.20) < Y. The $0.20 threshold was fixed a priori, above every non-2014 expiry. X and Y are chosen on 2016-17 to 2018-19 (maximum mean net per trade, at least 5 trades) and checked on 2019-20 to 2021-22.
- **The selection picks the corner, X = 0.30 with no physical filter.** In the check period this gives 3 trades in a single winter at 4 weeks (+$5,767 each) and **no trades to expiry**. The check is therefore uninformative.
- **The full 6×6 grid** (`data/processed/research_widowmaker_insurance/rule_grid.csv`):
  - 4-week hold: positive in 30 of 33 non-empty check cells (median +$1,980).
  - To expiry: positive in 20 of 28 (median +$701).
  - Raw Model 2: 31/33 and 18/24.

**Table 4. 5-yr centre, X = 0.15: what the physical filter does**

| Y (max P(large)) | 4-week train: mean / worst (trades) | 4-week check: mean (trades) | To expiry, train / check |
|---|---|---|---|
| No filter | $2,176 / −$7,860 (21) | $2,935 (8, 3 winters) | $4,355 / $1,723 (4) |
| 0.10 | $1,178 / −$1,050 (8) | $2,395 (4) | $2,486 / $1,455 (2) |

**Table 5. Benchmarks on the same weeks, 2016–21**

| Rule | Hold | Trades | Mean net (CI) | Worst |
|---|---|---|---|---|
| Short in the first November week | Expiry | 5 | +$2,192 (1,280–2,982) | +$500 |
| Short in the first November week | 4 weeks | 6 | −$182 | −$7,860 |
| Short every week | Expiry | 76 | +$1,485 (277–2,972) | −$1,220 |
| Short after a weekly rise > $0.05 | Expiry | 12 | +$3,983 (all in 2016–19) | +$1,780 |
| Short after a weekly rise > $0.05 | 4 weeks | 13 | +$1,496 | −$7,860 |

**Price-only check, 2010–21** (`widowmaker_insurance_theta.png`, right):
- The November-average spread exceeded the expiry spread in 10 of 11 winters, by a mean of $0.184 (median $0.141).
- Short from 1 November to expiry: +$1,174 per winter (CI 170–1,907); ex-2014, +$1,605.
- **2014 is the only expiry loss.** The worst mark-to-market drawdowns were 2018-19 (−$13,170), 2013-14 (−$11,920) and 2014-15 (−$3,890).
- The November spread itself was $0.09 on average in 2011–16 and $0.28 in 2017–21, so the premium got richer after 2014.
- The price-only twin, "short when S > $0.15" at weekly decision points over 2010–21:
  - To expiry: +$3,137 per trade (CI 1,532–4,771); worst trade −$1,720; 2014 trades **+$9,110**, since they were entered into the spike and expiry was lower.
  - 4-week hold: +$1,942; worst −$7,860.

## 4. What this tells us
**(a) The futures curve.**
- The March−April spread behaves like an over-priced insurance policy. Its physical value at expiry is close to the storage carry (about −$0.02 to −$0.05). Everything above that is premium, and it decays roughly linearly over the last 12–14 weeks.
- The premium is **state-dependent in the market's own price, not in physics.** It was largest when the spread spiked (Nov 2018, Nov 2021). In those weeks the physical distribution moved only a few cents (2018-19 AV +$0.06).
- The insurance view therefore explains the spread's **level and decay**. It does not provide a signal beyond the spread itself.

**(b) Storage expectations.**
- Re-centring Model 2 on the 5-year path removes its bias in payout terms: the mean expiry miss is −0.001, against −0.025 for raw Model 2.
- Even a 500 Bcf deficit maps to only about +$0.05 at expiry. Only 2014-scale deficits (about 1 Tcf) produce a large payout.
- The physical model gave a large expiry spread (> $0.20) an average probability of 0–21% per winter in 2016–21, and none occurred.

## 5. Caveats
- **Few points.**
  - The curve has 11 points, and its convexity rests on 2014 alone.
  - The premium tests have 5 outcome winters, and 2018-19 dominates the slope.
  - The bootstrap CIs with 5–6 blocks are optimistic.
- **There is no disaster in the Model 2 era,** so the physical filter's main purpose (avoiding a 2014) is untestable here.
- **Model 2's percentiles are odd.** There is a bimodal jump between p16 and p17, and negative storage levels appear in Nov 2018. The floor handles the latter; sensitivity is ±0.01–0.02.
- **AV ignores** the correlation between April's price and the weather (relevant only to the scaled variant) and the tails beyond p1/p99.
- **Variants tried:**
  - 8 payout specs × 3 functional forms.
  - 4 AV versions (2 centres × 2 curve modes) plus 4 sensitivities.
  - 144 rule cells, 3 benchmarks and 10 price-only rule cells.
- **The holdout X = 0.15 was picked after seeing both the training and check grids.** It is a round, middle value, not the training optimum (0.30). Treat it as declared rather than validated.

## 6. Proposed holdout test (W1; frozen; re-used holdout 2022-23 to 2025-26, then live 2026-27)
- **Inputs:**
  - Weekly decision settles: first trading day ≥ F+7, origins 1 Nov – 14 Feb.
  - Model 2 v2 p1–p99 at origin F, re-centred so the median equals the 5-year-path benchmark, floored at 700 Bcf.
  - Payout curve **frozen at the 2011–21 fit:** a = −0.133, b = 0.087, τ = 634, with its 500 bootstrap draws and residuals (`curves()` seed 2022). x uses the prior-5-year average.
- **Signal:** premium = S_t − AV_t > **$0.15**. No P(large) filter.
- **Trade:** short 1 H−J and hold 4 weeks (exit at the March expiry if sooner), 4 ticks per contract per side.
- **Statistics:**
  1. Mean net $/trade.
  2. The slope of (S_exp − S_t) on −premium over all weeks, with a winter-block bootstrap.
- **Pass** if (1) > 0 **and** the slope's 90% CI excludes 0. Inconclusive if fewer than 2 winters trade.
- **Reported alongside, not as a pass criterion:** the price-only twin (S_t > $0.15). Model 2 "adds value" only if W1 beats the twin by more than $500 per trade.

## 7. How to rerun
`python -m src.research.widowmaker_insurance.run` takes about 2.5–4.5 minutes on 1 CPU. It prints every table (saved log: `data/processed/research_widowmaker_insurance/run_log.txt`), writes the CSVs to the same folder and redraws the 3 figures.

## 8. New avenues
1. **Back-cast Model 2 to winters 2011–16, including 2014** (needs pre-2010 EIA storage). This is the only way to test whether P(large) would have flagged Jan 2014. *Expected value: high. It is the whole case for physical conditioning, which is currently untested.*
2. **Price the path, not the expiry.** Fit a payout curve for the spread 4 weeks ahead against the storage path, and push the Model 2 paths through it. The real risk is the mark-to-market drawdown (−$13k in 2018, −$12k in 2014), not the expiry loss. *Expected value: a sizing and stop rule; likely risk reduction, not more return.*
3. **Cap the tail with options.** Express the short as a short H−J call spread (calendar-spread options), priced with the physical distribution, so the 2014 and 2018 paths are bounded. *Expected value: moderate. It shows whether the premium survives paying for tail cover.*
4. **Structural break after 2014.** The November spread tripled (from $0.09 to $0.28) after the polar vortex. Test whether the premium scales with the memory of recent disasters, using the holdout and live data, since a "fear premium" may fade. *Expected value: tells whether the 2017–21 harvest is persistent.*
