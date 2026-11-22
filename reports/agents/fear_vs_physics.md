# Fear vs physics in the March−April premium (agent `fear_vs_physics`, study v3 wave 2)

*22 Nov 2026. Data ≤ 2021-12-31 only; every loader asserts the cutoff. Model 2 is read only from the dev cache (origins ≤ 2021-12-31). The panel has 175 weekly decision points over 12 winters (2010-11 to Dec 2021): 167 have an expiry outcome and 171 a 4-week outcome. Code: `src/research/fear_vs_physics/`.*

## 1. Plain-English findings
- **Both fear and physics explain the level of the spread, and fear explains much more.** Recent spread turbulence plus time to expiry explains 80% of the variance in S. Storage physics plus time explains 59%; both together explain 82%. Given fear, physics still adds a little: **+5.8¢ per sd** (CI 1.8–8.7).
- **The fear part of the spread decays one-for-one. The physics part does not.** Split S into a fear part, a physics part, a time part and a residual. To expiry, the fear part's later change is **−1.15 × itself** (CI −1.47 to −0.57), i.e. it is fully given back. The physics part's is **+0.63** (CI −0.90 to +2.08): kept, but very imprecisely. Equivalently, for the same spread, one sd of extra physical tightness means **9¢ less decay** to expiry (CI 0.5–17¢). This is the "overpaid fear premium, fairly paid physics premium" pattern. It is clear for fear and only suggestive for physics.
- **Out of sample, none of this beats the spread itself.** In walk-forward forecasts of the drift, adding fear or physics to S does not improve the out-of-sample R² (to expiry −0.12 vs S alone, CI −2.6 to +0.45). The decomposition explains why the premium decays; it does not forecast the decay better.
- **The rule as literally posed ("fear high AND physical tail low") fails.** The physics filter removes exactly the rich weeks (tight storage and rich spreads coincide). It lowers mean P&L in every fear row. The one thing it does is cut the worst 4-week loss from −$7,860 to −$1,730.
- **What works better is "short the spread's excess over its physics-implied level" (FX = S − PP ≥ $0.10).** Over 4 weeks: train +$1,396 per trade (worst −$190); check 2017-21 **+$3,445** (CI 7 to 5,273; 16 trades, 5 winters, worst −$530). For comparison, S > $0.15 made +$2,675 with a worst trade of −$7,860. The rule sat out 9 Nov 2018, when S was below its physics level and then rose 77¢, and shorted from 16 Nov, when S was 65¢ above it. **In Jan 2014 it stayed out only because of a walk-forward fit to three calm winters; a full-sample fit would have shorted on 24 Jan 2014 and lost $5,620.** This family was chosen after seeing the first grid (§5).

## 2. Method
**Panel.** Each EIA storage week F with F between 1 Nov and 14 Feb is a decision point. The decision settle is the first trading day ≥ F+7, since storage for F is public at F+6. Definitions:
- S = H − J at the decision settle.
- Targets: ΔS_exp = S_expiry − S, and ΔS_4w = S(+28 d, or expiry if sooner) − S. ΔS_4w is NaN if the exit falls after 2021-12-31.

**Fear** (prices up to the decision settle):
- rv2, rv4, rv8: the s.d. of daily ΔS over 10, 20 and 40 days.
- maxrise: the largest 5-day rise in the last 20 days.
- fvol: 20-day front-month realised vol. The front month is the nearest contract with ≥ 2 days left; returns are within-contract.
- **Fear index** = mean of the five z-scores.

**Physics** (storage at F, weather before F):
- **dev_now:** storage minus the same calendar week's mean over the prior 1–5 years. Storage starts in Jan 2010, so there is no value for Nov–Dec 2010; Jan 2011 uses 1 year, and 5 years are used from 2015.
- **Weather-analog projection** (available in every year, and needs no Model 2):
  - Fit weekly Δstorage = a + b·HDD on the Nov–Mar weeks of the trailing 3 years (b ≈ −1.4 to −1.7 Bcf/HDD).
  - For each complete prior winter A from 2004-05 (6 analogs in 2010, 17 in 2021): x_A = storage_F + Σ_remaining weeks (a + b·HDD_A,k) − avg5_endMarch.
  - avg5 uses 1–5 prior end-Marches, so 2011 rests on 1 year.
  - **P_tail** = mean_A Φ((−500 − x_A)/(σ_res·√n_rem)), the analog probability that end-March falls more than 500 Bcf below its 5-year average. The −500 threshold was fixed a priori; −300 and −800 are robustness checks.
  - x_an = mean_A x_A.
- **Storage-draw analog** (secondary, from 2011): prior winters' actual draws.
- **Physics index** = mean z of (−dev_now, −x_an, P_tail).
- **Model 2** (Nov 2016+): P(x < −500) from p1–p99, re-centred on the 5-year path.

**Decomposition.**
- Level regression: S = c + a·wte + f·Fear + q·Phys + e.
- The parts are S_fear = f·(Fear − mean), S_phys, S_time, and the residual e.
- Drift regression: ΔS = k + β_F·S_fear + β_P·S_phys + β_T·S_time + β_e·e.
- Reading: β = −1 means that part decays fully, and β = 0 means it is retained.

**Inference and evaluation.**
- CIs: winter-block bootstrap with 2,000 draws.
- Walk-forward: z-scores and fits use winters < Y only.

**Rules.**
- Short 1 spread at each signal week. Overlapping trades are allowed and scored per trade. Holds are to expiry or 4 weeks, at 4 ticks per contract per side ($160 per spread round trip).
- Thresholds are chosen on train winters 2010-11 to 2016-17 (2012-13 to 2016-17 for walk-forward signals), by maximum mean net per trade with ≥ 10 trades and ≥ 3 winters. Check winters: 2017-18 to Dec 2021.
- FX = S − PP, where PP is the walk-forward fit of S on (wte, Phys index).
- **Control** TX = S − fit(S on wte): the same rule with physics removed.

## 3. Results
**Table 1. Level of S (n = 175, 12 winters; Fear and Phys correlate at 0.71)**

| Model | Fear ($/sd) | Phys ($/sd) | R² |
|---|---|---|---|
| wte + Fear | 0.225 [0.183, 0.257] | | 0.80 |
| wte + Phys | | 0.181 [0.097, 0.219] | 0.59 |
| wte + Fear + Phys | 0.182 [0.142, 0.205] | **0.058 [0.018, 0.087]** | 0.82 |

**Table 2. Which part decays?** (Fig. 1, `figures/fear_vs_physics_decomposition.png`). The component s.d.s are fear 16.7¢, physics 5.6¢, time 5.8¢ and residual 10.2¢.

| Target | S_fear | S_phys | S_time | Residual | R² |
|---|---|---|---|---|---|
| ΔS to expiry (n 167) | **−1.15 [−1.47, −0.57]** | +0.63 [−0.90, 2.08] | −1.04 | −0.89 | 0.79 |
| ΔS 4 weeks (n 171) | **−0.99 [−1.38, −0.62]** | +1.06 [−0.72, 2.60] | −0.38 | −0.53 | 0.53 |
| Given S (expiry): coefficient on Fear / Phys index | −0.04 [−0.11, 0.08] | **+0.090 [0.005, 0.171]** | | | 0.79 |
| Given S (4 weeks) | −0.091 [−0.17, −0.01] | +0.090 [−0.01, 0.18] | | | 0.53 |

**Robustness** (`decomp_variants.csv`):
- **Leaving out one winter** (2013-14, 2014-15, 2018-19 or 2021-22): β_F ranges from −0.94 to −1.15. β_P ranges from −0.18 to +0.68.
- **Every fear/physics definition** gives β_F between −0.84 and −1.39, with the CI excluding 0.
- **But with fvol alone as fear,** which is not mechanically tied to the spread level, physics takes more of the level, and **β_P = −0.34 [−0.92, 0.12]**: about a third of the physics part also decays.
- **"Physics is retained" is therefore not robust.** "Fear is overpaid" is.

**Table 3. Walk-forward out-of-sample R² of the drift** (test winters 2014-15 onward; 7 to expiry, 8 for 4 weeks)

| Forecast | Expiry: R² vs 0 | Expiry: vs S alone [CI] | 4 weeks: R² vs 0 | 4 weeks: vs S alone |
|---|---|---|---|---|
| S alone | **0.79** | — | **0.56** | — |
| Fear | 0.64 | −0.73 [−1.4, −0.4] | 0.54 | −0.05 |
| Phys | 0.30 | −2.4 | 0.15 | −0.93 |
| S + Fear + Phys | 0.77 | −0.12 [−2.6, 0.45] | 0.52 | −0.10 |
| Decomposition | 0.75 | −0.19 | 0.50 | −0.14 |

Starting in 2012-13 makes things worse: indices z-scored on two calm winters explode in 2013-14 (Phys R² −60).

**Model 2 era** (84 weeks, 6 winters): given wte and Fear, Model 2's tail adds 0.135 [−0.31, 0.30] to the level. Given S and Fear, it adds 0.108 [−0.25, 0.44] to the drift to expiry. Neither is distinguishable from 0.

**Table 4. Rules** (check 2017-18 to Dec 2021; per trade, net, with 95% CI; Fig. 3, `figures/fear_vs_physics_rules.png`)

| Rule | Hold | Train mean (worst) | Check mean [CI] | Check trades / winters | Check worst |
|---|---|---|---|---|---|
| Short every week | 4w | +88 (−5,620) | +945 [185, 2,177] | 65 / 5 | −7,860 |
| S > $0.15 | 4w | +796 (−5,620) | +2,675 [1,400, 3,790] | 25 / 4 | −7,860 |
| A: rv4 ≥ 0.02 & P_tail ≤ 0.5 (selected) | 4w | +851 (−710) | +952 [−18, 2,998] | 27 / 5 | −1,730 |
| Control TX ≥ $0.10 | 4w | +402 (−5,620) | +2,941 | 21 / 3 | −7,860 |
| **C: FX ≥ $0.10 (selected)** | 4w | **+1,396 (−190)** | **+3,445 [7, 5,273]** | 16 / 5 | **−530** |
| Short every week | exp | +478 (−3,150) | +1,400 [−14, 3,252] | 61 / 4 | −1,220 |
| S > $0.15 | exp | +1,840 (−1,720) | +4,126 [2,130, 5,510] | 21 / 3 | +1,010 |
| A selected: rv4 ≥ 0.03, no filter | exp | +1,341 (−2,160) | +3,112 [30, 4,247] | 24 / 3 | −1,060 |
| C: FX ≥ $0.10 | exp | +2,100 (−60) | +4,514 [−240, 7,485] | 13 / 4 | −530 |

- **Benchmarks.** 1 Nov → expiry, once per winter: +$1,174 (2011–21); 2014 −$3,140.
- **Grid A** (36 cells × 2 holds, `grid_A_rv4_ptail.csv`):
  - Check means are positive in 25/32 (expiry) and 25/33 (4w) cells.
  - At every fear level the physics filter lowers the check mean (to expiry: 1,400–4,322 unfiltered, against −49 to 575 averaged over the filtered cells).
  - At 4 weeks, the filter improves the worst trade from −7,860 to −1,730.
- **FX robustness** (`fx_robustness.csv`): 7 physics definitions × 3 thresholds × 2 holds.
  - The mean is positive in 42/42 cells, in both train and check.
  - **But the 4-week train worst trade is −5,620 (24 Jan 2014) in 5 of 7 definitions.** Only the composite and P_tail(−800) avoid it.
  - Costs from 0 to 8 ticks move every rule by ≤ $160.
- **FX 4-week P&L by winter:**

  | Winter | 2013-14 | 2014-15 | 2015-16 | 2016-17 | 2017-18 | 2018-19 | 2019-20 | 2020-21 | Nov–Dec 2021 |
  |---|---|---|---|---|---|---|---|---|---|
  | P&L ($) | +7,930 | +7,550 | +1,010 | +3,050 | +1,900 | **+37,120** | −310 | −890 | +17,300 |

  2018-19 dominates.

**Jan 2014** (Fig. 2, `figures/fear_vs_physics_episodes.png`). Analog physics turned before fear did.

| Date(s) | S | rv4 (percentile vs history) | dev_now | P_tail | FX (PP) | What happened |
|---|---|---|---|---|---|---|
| 8–29 Nov 2013 | 0.02–0.04 | 17–32% | −31 to −76 | 0.10–0.16 | ≈ −0.09 | |
| 20 Dec | 0.28 | 100% | −387 | 0.58 | −0.16 | |
| 17 Jan | 0.18 | 95% | −538 | 0.78 | −0.40 | |
| **24 Jan** | 0.58 | 100% | −430 | 0.78 | +0.02 (0.56) | S > $0.15 and TX shorted: −$5,620 over 4 weeks |
| 21 Feb | 1.12 | | | 0.90 | +0.47 | FX's only 2013-14 trade: +$7,930 |

From mid-December, fear was at record highs and would have sold into the vortex. Physics said the rally was backed. But PP's steep slope came from three calm training winters. The frozen full-sample fit puts 24 Jan at FX = +0.22, which is a trade.

**Nov 2018.**
- On 9 Nov, S was 0.39 against a PP of 0.43, so the rule did not trade. The spread then rose 77¢ (−$7,860 for S > 0.15).
- From 16 Nov to 21 Dec, FX ranged from +0.21 to +0.73: six trades, +$33,400.

**Model 2.** Its tail agreed (0.57–0.65 in Nov–Dec 2018) and added nothing beyond the analog.

## 4. What this tells us
**(a) The futures curve.**
- The March−April premium is mostly a charge for recent turbulence, and that charge is given back in full within weeks.
- The part of the spread that tracks physical tightness behaves differently. It is not reliably given back, and in tight winters (2014, Nov 2018) S sat at or below its physics-implied level before the moves that hurt shorts.
- So "rich" should be measured against physics, not against zero. The excess over physics is the harvestable premium. Its main value is avoiding shorting a spread that is merely fairly priced in a tight winter.
- For the mean, the spread level alone is as good a forecast as any decomposition.

**(b) Storage expectations.**
- A cheap weather-analog projection (no Model 2) carries the same information as Model 2's tail for this purpose: their correlation is 0.93 over 2016–21 (0.68–0.99 within winters).
- Physical tightness was visible in mid-December 2013, a month before the 24 Jan spike. The analog P_tail had risen from 0.14 to 0.58 by 20 Dec.

## 5. Caveats
- **Power:**
  - There are 11 outcome winters and one disaster.
  - FX's winter-mean P&L has an s.d. of $2,973, so over a 4-winter holdout the minimum detectable mean (80% power) is about $4,160, above the check mean.
  - P(≥ 1 disaster winter in 4) ≈ 0.32 at 1 in 11.
- **Fear partly measures the spread's own level.** rv4 correlates 0.89 with S, because spread vol scales with S − floor (see `scenario_risk`). The fvol-only variant is the cleanest, and there physics is partly decayed too.
- **The 2014 result is fragile** (§3). Physics coverage is thin in early years: avg5 uses 1–4 years for 2011–14, and dev_now is missing for Nov–Dec 2010.
- **The FX and TX families were added after seeing grid A's train and check results.** Within FX, $0.10 is the training choice. Treat the family choice as declared, not validated.
- **Variants tried:**
  - 9 level/drift specifications × 2 targets.
  - 7 decomposition definitions and 4 leave-one-winter-out runs.
  - 8 walk-forward models × 2 targets × 2 start years.
  - 9 Model 2 regressions.
  - 142 rule cells (A 72, B 50, C 10, TX 10), 42 FX robustness cells, 4 cost levels.

## 6. Proposed holdout test (H-FvP; frozen; re-used holdout 2022-23 to 2025-26, then live 2026-27)
- **Inputs:** the weekly panel from `data.py` with only the cutoff changed. Origins 1 Nov – 14 Feb; decision settle at the first trading day ≥ F+7. Physics index from the frozen z-parameters, and PP = −0.01049 + 0.013602·wte + 0.18091·Phys (`data/processed/research_fear_vs_physics/frozen_holdout.json`). No refitting.
- **Signal:** FX = S − PP ≥ **$0.10**. Short 1 H−J, hold 4 weeks (exit at the H expiry if sooner), 4 ticks per contract per side.
- **Statistics:**
  1. Mean net $ per trade, with a winter-block CI.
  2. Worst trade, compared with the S > $0.15 twin on the same weeks.
- **Pass** if (1) > 0 **and** FX's worst trade is better than the twin's worst trade, or neither loses more than $2,000. Inconclusive if fewer than 2 winters trade.
- **Secondary (not pass/fail):** with the frozen level coefficients, the slope of ΔS_4w on S_fear (expect about −1) and on S_phys (expect above −1).

## 7. How to rerun
```
python -m src.research.fear_vs_physics.run       # panel + all tables, ~2.5 min on 1 CPU; log in data/processed/research_fear_vs_physics/run_log.txt
python -m src.research.fear_vs_physics.figures   # 3 figures, ~5 s
```

## 8. New avenues
1. **Back-cast physics to pre-2010 winters** with the weather-analog projection, which needs only HDD from 2004 and an EIA storage history. This would test the excess-over-physics rule on 2005–10 spreads, if pre-2010 prices are obtainable. *Value: high; it is the only way to gain disaster winters.*
2. **Stop-out rule:** hold the plain November short, but flatten when FX < 0 while P_tail > 0.5 (spread no richer than physics in a tight winter). *Value: high for drawdown; this targets 24 Jan 2014 and 9 Nov 2018 directly.*
3. **Price-independent fear:** replace rv-based fear with options-implied vol or open-interest and positioning data (available from Nov 2015) to break the mechanical link between rv and S. *Value: medium; it decides whether "fear" is a real channel or a relabelling of the level.*
4. **Pool the physics map with storage theory** (a fixed convex slope, as in `widowmaker_insurance`'s payout curve) instead of a walk-forward OLS, so that PP does not depend on which calm winters it was trained on. *Value: medium; it removes the 2014 fragility.*
