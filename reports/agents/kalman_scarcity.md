# Q3 part 1: a Kalman-filtered "market-expected scarcity" from the winter curve

*Agent `kalman_scarcity`, 21 Nov 2026. Data 2010-06-04 to 2021-12-31 only. Parameters estimated on target winters 2010-11 to 2016-17 and frozen; 2017-18 to 2021 is the check window.*

## 1. Plain-English findings
- **The winter curve is effectively one number.** PC1 explains 99.0% of the variance of the five standardised winter spreads. A one-factor Kalman state correlates 0.95 with the plain March−April spread and 0.97 with PC1. Combining the spreads adds almost nothing.
- **The filter is slightly smoother, not more informative.** Its weekly changes are 8% smaller relative to its level than those of March−April (0.475 vs 0.514). It tracks the eventual end-March surplus as well as March−April but not better: the difference in correlation is −0.02 to +0.01 at every horizon. Within 26 weeks of end-March, storage already known at the time is a better guide than any curve signal.
- **The curve prices weather 1–3 weeks before EIA reports it.** In Nov–Mar, a weekly rise in scarcity goes with larger-than-normal withdrawals in the reports published 1, 2 and 3 weeks later (correlations −0.17, −0.16, −0.27). It does not move on the report published that week (+0.04). About 70% of the 3-week lead disappears once the weather that actually followed is controlled for, so the market is pricing weather forecasts. March−April alone shows the same pattern.
- **Large unexplained jumps did not reliably reverse.** In 2011–2017 about 20% of a large jump reversed the following week. In 2018–2021 there was no reversal: 1–4 weeks later the slope is −0.01 to +0.18, which if anything means continuation.
- **The calm/stressed regime model tracks stress as it happens and gives no early warning.** It flags Nov 2018 in the week of the first jump: P(stressed) went from 0.02 to 1.00. It stayed flagged through all of winter 2013-14. Uri (Feb 2021) fell on the week the target winter rolled to 2021-22 and only shows up a week later. Stressed weeks rose from 18% of 2010–17 to 44% of 2017–21.

## 2. Method
**Observations.** Each Friday I take the first settlement on or after that day, the same convention as Model 3, so the Thursday storage report is already public. The target winter Y is the earliest whose March contract is more than 5 trading days from expiry. The five spreads in $/MMBtu are Dec−Apr, Jan−Apr, Feb−Apr, Mar−Apr, and the strip: the mean of the live Dec–Mar months minus the mean of Apr–Oct of year Y. Any leg within 5 trading days of expiry is dropped. Expiry is taken as the 3rd trading day before delivery, which matches the last observed settlement for 128 of 132 contracts; the other 4 are 1 day off. There are 605 weeks, 350 of them in the estimation window.

**Model** (linear-Gaussian, hand-written filter, missing legs skipped):

  s_t = s_{t−1} + η_t, with η ~ N(0, q), and N(0, q_roll) in the week the target winter rolls
  y_{i,t} = a_i + c_i τ_{i,t} + b_i e^{−κ τ_{i,t}} s_t + ε_{i,t}, with ε ~ N(0, h_i)

τ is years to delivery of the winter leg. For identification a_MarApr = 0 and b_MarApr = 1, so s_t is measured in $/MMBtu of a March−April spread at delivery. Parameters are fitted by maximum likelihood (L-BFGS, then Nelder–Mead) on the estimation window. The **filter**, not the smoother, then runs forward with frozen parameters through 31 Dec 2021.

Estimates for the primary random-walk model:
- **Loadings b:** Dec 1.20, Jan 1.17, Feb 1.13, Mar 1, Strip 1.36. κ = 0.69/yr, so loadings halve with each year to delivery.
- **State shocks:** sd 7.0¢/week, and 21¢ in roll weeks.
- **Noise sd:** Dec 3.5¢, Jan 1.0¢, Feb ~0, Mar 1.5¢, Strip 3.3¢.

The Feb−Apr noise variance hit zero (a boundary solution), so the unconstrained state is essentially a rescaled Feb−Apr spread whenever Feb is live. That is why I added the 1¢ noise-floor variant. The linear model did not "clearly fail", so I did not try an EKF or a convex g.

**Benchmarks.**
- The March−April spread of the target winter.
- PC1 of the standardised spreads, with loadings 0.43–0.47 from complete estimation-window weeks. When legs are missing, the score is the least-squares projection on the live legs.

Weekly changes exclude roll weeks.

**Storage.**
- **Surplus:** level minus the average of the same week in up to 5 prior years, requiring at least 2.
- **Surprise:** the weekly change minus the same average of changes. The surprise "published at t" is the one for the week ending t−7.
- **HDD anomaly:** relative to the prior 10 years.

**Inference.** Winter-block bootstrap (2,000 draws) for correlations; Newey–West HAC (lags h+2, or 4) for regressions.

**Markov switching.** Two regimes in the variance of Δs_t, with a common constant. Fitted on the estimation window, then *filtered* forward with frozen parameters. Roll-week changes are set to 0.

## 3. Results
**Table 1: correlation with the eventual end-March surplus** (ex post; target winters 2011-12 to 2020-21, 10 winters; 95% winter-bootstrap CI). Figure: `figures/kalman_scarcity_endmarch.png`.

| Weeks to end-March | KF state (RW) | Mar−Apr | PC1 | KF − Mar−Apr | Storage known at t |
|---|---|---|---|---|---|
| 5–8 (Feb), n=23 | −0.77 [−0.91, −0.37] | −0.74 [−0.89, −0.36] | −0.77 | −0.024 [−0.062, −0.005] | +0.91 [0.69, 0.98] |
| 9–17 (Dec–Jan), n=90 | −0.50 [−0.74, −0.29] | −0.50 [−0.74, −0.31] | −0.51 | +0.001 [−0.04, 0.06] | +0.76 [0.53, 0.90] |
| 18–26 (Oct–Nov), n=90 | −0.27 [−0.74, 0.36] | −0.25 [−0.68, 0.28] | −0.29 | −0.014 [−0.09, 0.08] | +0.44 [0.12, 0.82] |
| 27–56 (Mar–Sep), n=319 | −0.23 [−0.80, 0.20] | −0.24 [−0.79, 0.22] | −0.24 | +0.005 [−0.05, 0.07] | +0.18 [−0.28, 0.67] |

- **Controlling for storage known at t,** the partial correlations of every curve signal with the eventual surplus are not significant (KF: −0.30 [−0.73, 0.23] in Feb; +0.27 [−0.14, 0.44] in Dec–Jan). The scarcity *level* has no detectable information about end-March storage beyond what was already in the tank.
- **Estimation vs check windows:** in Dec–Jan the KF correlation is −0.65 in 2012–17 and −0.62 [−0.79, 0.37] in 2018–21, with only 4 winters in the latter.

**Table 2: (b) storage known, and lead/lag of changes vs EIA surprises** (Nov–Mar weeks, winter-bootstrap CI)

| | KF (RW) | Mar−Apr | PC1 |
|---|---|---|---|
| Level vs surplus known | −0.64 [−0.79, −0.41] | −0.56 [−0.72, −0.26] | −0.58 |
| KF − Mar−Apr (paired) | −0.08 [−0.24, 0.04] | | |
| ΔX_t vs surprise published t−1 | −0.04 [−0.18, 0.12] | −0.04 | −0.05 |
| ΔX_t vs surprise published t (this week's report) | +0.04 [−0.08, 0.16] | +0.03 | +0.03 |
| ΔX_t vs surprise published t+1 (week ending t) | −0.17 [−0.25, −0.05] | −0.19 [−0.27, −0.02] | −0.18 |
| ΔX_t vs surprise published t+2 | −0.16 [−0.30, 0.04] | −0.15 | −0.17 |
| ΔX_t vs surprise published t+3 | −0.27 [−0.34, −0.16] | −0.26 [−0.32, −0.18] | −0.27 |
| ΔX_t vs surprise published t+4 | −0.05 [−0.19, 0.10] | −0.06 | — |

A one-sd move in Δs_t goes with a surprise of −25.7 Bcf (t = −3.7) in the report 3 weeks later. Once the HDD anomalies that actually occurred in weeks t to t+2 are added, that falls to −7.5 Bcf (t = −2.0).

For next week's report, with this week's HDD controlled, the extra information in Δs_t differs by window:
- 2011–17: −17 Bcf per sd (t = −2.3)
- 2018–21: −1.2 Bcf per sd (t = −0.4)

**Table 3: (c) reversal after large unexplained jumps.** u_t is the residual of Δs_t on the surprise published that week and the HDD anomaly, each also interacted with Nov–Mar. That regression was fitted on 2011–17 and explains only 2.7% of the variance. A jump is |u| > 10.8¢, the 90th percentile in the estimation window. The table gives the slope of the later change on the jump size; negative means reversal.

| h (weeks) | KF, 2011–17 (24–25 jumps) | KF, 2018–21 (44–48 jumps) | Mar−Apr after KF jumps, 2018–21 |
|---|---|---|---|
| 1 | −0.20 (t −1.6; boot [−0.40, −0.06]) | −0.01 (t −0.1) | +0.05 (t 0.5) |
| 2 | −0.03 (t −0.3) | +0.05 (t 0.4) | +0.12 (t 1.0) |
| 3 | +0.06 (t 0.4) | +0.18 (t 1.5) | +0.26 (t 2.2) |
| 4 | −0.01 (t −0.0) | +0.11 (t 0.8) | +0.21 (t 2.0) |

The Mar−Apr spread's own jumps behave similarly: −0.29 (t −1.9) at h=1 in 2011–17, and −0.01 in 2018–21.

**Markov switching.**
- **Regimes:** calm sd 3.5¢/week, stressed sd 13.4¢/week. P(stay calm) = 0.978 and P(stay stressed) = 0.912, i.e. expected spells of about 45 and 11 weeks.
- **Jan–Feb 2014:** P(stressed) ≥ 0.84 from mid-Dec 2013 through Mar 2014, with one week at 0.49. It had already been high in Apr–Jun 2013.
- **Nov 2018:** 0.02 on 2 Nov, then 1.00 on 9 and 16 Nov (Δs = +27¢, then +58¢). It stayed at 0.73 or above until late Dec.
- **Feb 2021 (Uri):** 0.03 on 12 Feb. On 19 Feb the March-21 contract was inside the 5-day expiry window, so the target rolled to 2021-22 and that change is excluded. P(stressed) was 0.44 on 26 Feb and 1.00 on 5 Mar.

Figure: `figures/kalman_scarcity_state.png` shows the filtered state against Mar−Apr, with the episodes and the regime probability.

## 4. What this tells us
**(a) The futures curve.** Winter spreads move together almost perfectly. Up to a time-to-delivery adjustment, the "market-expected scarcity" is already visible in any one near-winter spread. The Kalman filter's job reduces to rescaling across the roll and smoothing a few cents of noise. Its loadings rise toward the front of the winter, and its noise is smallest for Jan−Apr and Feb−Apr. Those mid-winter spreads are the cleanest single read, cleaner than Mar−Apr.

**(b) Storage expectations.**
- The curve's scarcity level is mostly a re-reading of storage already reported (Table 2). It adds no measurable information about end-March storage beyond that.
- Its *changes* do anticipate EIA surprises by 1–3 weeks, mostly through weather forecasts.
- There is a small non-weather lead before 2018 that disappears after 2018.
- After 2017, scarcity moves are bigger and stressed weeks more common, but jumps did not overshoot in a way that reversed.

## 5. Caveats
- **Variants tried:**
  - 3 Kalman specifications, all reported: random walk (primary, chosen before results), AR(1), and random walk with a 1¢ noise floor.
  - 1 Markov specification (5 starting points).
  - 2 benchmarks.
  - One jump threshold (90th percentile) and one set of horizon buckets, both fixed before the results.
  - AR(1) beats the random walk in likelihood (LR = 28.2, φ = 0.90/week). Its signals are almost identical (correlation 0.999), so this does not change any conclusion.
- **Only 10 winters for (a)** and 4 in the check window, so the confidence intervals are wide.
- **Short storage history:** the storage file starts in 2010, so the "5-year" averages for 2011–2014 use 2–4 years.
- **Surprises are not the market's surprises:** they are measured against the 5-year average, not consensus. Weather is the realised HDD, not forecasts.
- **Boundary solution:** the Feb−Apr noise variance sits at zero, so the posterior state sd is about 0 and the 90% band in the figure is invisible. The noise-floor variant gives a state sd of 0.6–1.4¢ and the same results.
- **Rolls:** composition changes, such as a leg dropping out or the target-winter roll, are absorbed by the filter. Uri coincides with a roll.
- **s_t is not tradable.** The reversal result for the tradable Mar−Apr spread is the one in the last column of Table 3, before costs.

## 6. Proposed holdout test (frozen)
- **Inputs:** the frozen random-walk parameters in `data/processed/research_kalman_scarcity/kf_params.json` (key `rw`), with the filter and panel code exactly as here. Run the filter continuously from 2010-06 through the holdout. Storage surplus known at t is defined as in this report: the week ending F−7 minus the average of the same week in the 5 prior years.
- **Sample:** Fridays from 2022-01-07 to the last available, in Nov–Mar only.
- **Statistic:** D = corr(s_t, surplus_known_t) − corr(MarApr_t, surplus_known_t), with a 95% winter-block bootstrap CI (2,000 draws, blocks are target winters).
- **Pass** (the Kalman state is more informative than Mar−Apr): D < 0 and the CI upper bound < 0. Otherwise conclude that the Kalman filter adds nothing over Mar−Apr, as in 2010–21 (D = −0.08 [−0.24, 0.04]).
- **Secondary, reported but not a pass criterion:** corr(Δs_t, surprise published at t+3) in Nov–Mar, expected ≤ −0.10 with the CI upper bound < 0.

## 7. How to rerun
`cd /home/claude/natgas-curve && python -m src.research.kalman_scarcity.run`
- **Cached parameters:** about 3 minutes, almost all of it bootstrap.
- **Full refit** (`--refit`, three maximum-likelihood fits of about 5 minutes each): about 18 minutes.
- **Figures only:** `--figs`.

The panel is built by `src/research/kalman_scarcity/data.py` (about 6 s). Outputs go to `data/processed/research_kalman_scarcity/`: `eval_a.csv`, `eval_b.csv`, `eval_b_lead.csv`, `eval_c.csv` and `results.json`.
