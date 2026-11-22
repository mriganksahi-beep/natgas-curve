# Study v3: independent review (agent `reviewer_v3`)

*22 Nov 2026. Data ≤ 2021-12-31 only. Scratch code is in `src/research/reviewer_v3/`; each check runs in under 20 s.*

## 1. Leakage: **confirmed clean, with procedural caveats**
- **Cutoffs.** All seven codebases filter on load and assert the cutoff.
- **Model 2 dev cache.** Origins run from Nov 2016 to Dec 2021. Its 9 rows with a 2022 target have an empty `actual`.
- **Procedural breach.** `storage_space_option` calls `Simulator()`, which loads the raw storage and weather files through 2026 and runs Model 1 walk-forward over them. `simulate(t)` uses only past analogs, residuals and normals, and the cache ends at origin 2021-10-22 with no 2022 actuals. I found no contamination route, but this breaks hard rule 1.
- **Walk-forward fits and trailing inputs are correct.** widowmaker curves, scenario_risk maps and η use winters < Y. Storage and Model 2 enter at d−7. curve_weather trains on rows with F′+14 ≤ F, and its normals use prior years only. premium_map shifts storage to its public date. premium_risk_management triggers execute at the next settle.
- **Expiry spread and drift targets** appear only as regression targets or in P&L.
- **`fear_vs_physics`.** No future winter enters fear, physics (analog winters end before F) or the walk-forward PP. Tables 1–2 use full-sample z-scores and fits, so they are in-sample description.
- **Minor design look-ahead.** K = $0.20 was set "above every non-2014 expiry 2011–21". The P_tail −500 Bcf threshold came from widowmaker's full-sample curve.
- **The holdout is weaker than labelled.** `study_v3.md`'s motivating fact uses 2011–2026, so the thesis was formed with 2022–26 in view.

## 2. Headline premium: **confirmed**
Recomputed from raw settles:

| Claim | Reported | Recomputed |
|---|---|---|
| Mar−Apr L9 | +25.9¢ [15.6, 34.5], 10/11 years | +25.9¢ [16.0, 34.6], every year matches |
| Short from 1 Nov to expiry | +$1,174 | +$1,174 [159, 1,921] |
| Short from first Friday to expiry | +$1,297 | +$1,297 [169, 2,297] |

- **Conventions are right:** $10,000 per $1 of spread, $160 round trip per spread, and the expiry equals the official last trading day in every year.
- **The agents differ only in entry date.**
  - 2014-15 alone moves $2,070 between the two conventions.
  - The mean ranges from $1,155 to $1,448 across the first 8 November trading days.
- **They agree otherwise.** The 2018-19 drawdown is −$13.2k to −$13.3k in every report, and premium_map's 2016–21 L4 total is $11.8k against my $10.9k.
- **Caveat:** one disaster in 11 years, so 11-block CIs understate the tail.

## 3. "Volatility, not storage": **overstated**
- **The spread-vol measures track the spread level:** rv4 correlates 0.89 with S and maxrise 0.91. Level-normalised vol, sd(ΔS/(S−floor)), correlates only 0.13.
- **Level R² of S ~ wte + fear:**

  | Fear measure | R² |
  |---|---|
  | Reported index | 0.80 |
  | Front-month vol | 0.43 |
  | Normalised spread vol | **0.12** |
  | Physics alone (comparison) | 0.59 |

  About **45–85% of "80%" is mechanics**.
- **premium_map's vol slope** (+4.7%/sd) does not survive:

  | Specification | Vol slope (%/sd) |
  |---|---|
  | Controlling for spread level | **+1.2 [−0.7, 2.2]** |
  | Normalised vol | +1.4 [−1.5, 2.8] |
  | Front-month vol | +2.5 [−1.5, 5.9] |

  The level itself gives +5.4 [3.0, 7.6], which is near-tautological because drift = V − V(T).

## 4. Fear/physics decomposition: **wrong as stated**. FX rule: **overstated**
**Decomposition (−1.15 vs +0.63).**
- Every non-physics part decays by about −1: time −1.04, residual −0.89, and S alone −0.78. Given S, fear adds −0.04 [−0.11, 0.08].
- With level-neutral fear, **β_P = −0.58 [−0.91, −0.21]**, so physics decays too.
- **What survives:** given S, physics adds **+0.08 [0.01, 0.15]** to drift to expiry. That holds under every fear definition and when any single winter is left out (+0.04 without 2013-14). It is in-sample only and fails walk-forward.

**FX ≥ $0.10 (4-week hold).**
- **Where the money came from:** 2018-19 gave $37.1k and the truncated 2021-22 $17.3k, out of $55.1k.
  - Excluding 2018-19: +$2,000 per trade [−240, 4,800].
  - Excluding both: **+$117** on 6 trades.
- **Against S > $0.15:** FX made less in 4 of 5 check winters, **−$11.8k in total**. Its only edge is skipping 9 Nov 2018.
- **Multiple testing.** 142 check cells were viewed before FX was declared. The single-test bootstrap p is 0.03, and my CI lower bound is −$45. A Bonferroni correction over 8 family × hold choices gives p ≈ 0.25.
- **The 2014 escape is an artefact:** the frozen full-sample PP shorts 24 Jan 2014 (−$5,620).

## 5. Payout curve and actuarial value: **confirmed with caveat**
- **AV is stable while 2014 is in the fit.** Near-linear gives −$0.010, very convex −$0.027 and the mean of past expiries −$0.015. AV is effectively the historical mean expiry, which is why Model 2 adds nothing.
- **But the tail is unidentified.**
  - Without 2014, the convex fit explodes: mean AV +$0.35, max $2.63.
  - With 2014's last-5-day payout ($0.78), AV is +$0.022.
- **The premium is too large for the expiry tail to explain.** Breaking even on $0.18 needs about a 49% chance of a 2014-size expiry, so the market is pricing path risk.

## 6. scenario_risk calibration: **overstated**
- **Breach counts cannot separate the models.** At n = 79, Kupiec passes 0–3 breaches, so 1 vs 2 breaches is not a distinction. On the 69 common weeks after warm-up, EWMA also has 1 breach.
- **The Z2 gap (0.75 vs 2.29) hinges on 1–2 weeks.**
- **The Nov 2018 lead is a spread-level effect.** On 9 Nov 2018, ES/(S−a) was at its median (0.640 vs 0.632). The alarm of 2.15 matches the (S−a)-only alarm of 2.09.
- **Coverage is not level alone either.** A crude level-only twin breaches 11 of 69 weeks. PHYS-M2's coverage comes from being wider (η plus Model 2 dispersion).

## 7. premium_risk_management: **confirmed with caveat**
- **The scaling benchmark is right:** kept + cut = 1 for any size f.
- **"None beats it" is false in-sample.**
  - 17 of 105 implementable variants exceed 1, e.g. volatility k = 1.5 (1.29) and the $2k stop with re-entry (1.67).
  - 3 of 14 placebo tests have p < 0.05, against 0.7 expected (binomial p = 0.03).
- **Defensible reading:** none *demonstrably* beats scaling, because "cut" rests on a single event.

## Most robust
1. **Mar−Apr decay into expiry:** +26¢ at L9, 10 of 11 years, and +$1.2–1.3k per winter net, reproduced exactly.
2. **The tail is the path, not the expiry:** −$12k to −$13k drawdowns against a worst expiry loss of −$3.1k.
3. **No physical model beats the spread itself out of sample** (widowmaker, fear_vs_physics, curve_weather).

## Weakest
1. **The fear/physics split** and "fear explains 80%".
2. **The FX rule:** one week, one winter, chosen after viewing the check period.
3. **PHYS-M2's calibration and Nov 2018 lead.**

## Frozen tests worth running
**Re-used holdout.**
- **Run first:** curve_weather's ΔSS test. It never saw 2022+ and has about 55% power.
- **Also run:** storage_space_option's price-only test and premium_map's L9 test. Read L9 as weak confirmation, because the thesis was formed with those winters.
- **Run, but treat as descriptive:**
  - **W1:** make "beats the S > $0.17 twin by > $500" primary, since its slope criterion is near-automatic.
  - **H-SR:** add (S−a) and HYB comparators, since criterion 1 almost cannot fail.
  - **H-FvP:** the minimum detectable effect ($4.2k) exceeds the effect.
  - **H-PRM.**

**Live 2026-27, frozen before 1 Dec.**
1. The Mar−Apr 2027 drift from L9 and from 1 Nov to expiry, with the worst drawdown.
2. curve_weather's ΔSS.
3. A weekly log of the PHYS-M2 alarm against the (S−a) alarm.

**Most valuable new evidence:** a pre-2010 back-cast (2003, 2006, 2008) of the price-only drift and the analog physics.
