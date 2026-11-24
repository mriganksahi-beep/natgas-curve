# rnd_composite: fallback risk-neutral density of S_T from the ON March and April surfaces plus a copula

*Agent `rnd_composite`, 24 Nov 2026. Development data only.*
- Options and futures were cut on load to dates ≤ 2021-12-31. No `*_locked` folder and no raw option row from 2022 on was opened.
- Scoring uses winters whose March option expired by Feb 2021 (2010-11 to 2020-21). Winter 2021-22 is computed for Nov–Dec 2021 but not scored.
- "Direct" means the shipped engine `src/tool/rnd_direct.py` (method `sln`), read from its daily cache `research_tool_rnd/final_*.pkl`. The v5_implied_storage mixture cache was used for the first comparison and the mixture-copula grid.
- Choose window: winters 2013-14 to 2015-16. Check window: 2016-17 to 2020-21.

## 1. Findings
- **The composite gives a reading on 99.7% of winter trading days** (888 of 891, Nov–Feb, 2010-11 to 2021-22).
  - It covers two winters that have no CSO data (2010-11, 2011-12).
  - On 28 days the raw file has no April ON quotes (Oct–Dec 2020). There the April smile is proxied from March; these days are flagged `april_proxy`.
  - In development the direct engine had usable quotes on 100% of days. So the fallback is needed because direct quotes can be poor, not because they are missing.
- **The composite is a cross-check, not a substitute.**
  - With ρ calibrated to the CSO ATM straddle, its spread distribution is wider in the body and less peaked than the direct density.
  - It overstates P(S_T > F + 0.10) by 6–8 points against direct, and P(> F + 0.25) by 2–3 points.
  - It matches the far tail: the bias in P(> F + 0.50) is −0.2 to +0.3 points, with MAE about 1 point.
  - On the same days it scores worse than direct in the check window: CRPS +0.023 [+0.005, +0.049] $/MMBtu.
- **The copula family hardly matters. The correlation level and the marginals do.**
  - At a fixed ATM straddle, the t₄ copula moves P(> F + 0.50) by +0.06 points and the decoupling mixture by +0.4 points (mean 5.3% → 5.6%).
  - Tail *dependence* (t) makes joint extremes co-move, so it does not add squeeze mass.
  - Decoupling (the mixture) adds a little, and the G4X wings do not ask for more: a 9-cell grid chose a mild mixture (5% weight on ρ = 0.3), and it is within noise of the Gaussian.
  - In a real squeeze (Nov 2018) the CSO-implied ρ itself falls to 0.1–0.4. The tail then comes from the March marginal: H at 90–100% IV against J at 30%. All copulas agree within 0.5 point.
- **Correlation fallbacks.**
  - Walk-forward EWMA ρ (half-life 20 days) minus a walk-forward premium offset tracks the CSO-calibrated ρ with a median error of −0.002 and MAD 0.030. The offset is +0.008 to +0.021: implied ρ sits below realised.
  - That error moves P(> F + 0.25) by 2.4 points (MAE). A fixed walk-forward average does worse (3.4 points).
- **The CSO is often Fréchet-infeasible with the legs.** The G4X ATM straddle is cheaper than the comonotone (ρ = 1) combination of the two ON marginals on:

  | Winter | Days infeasible |
  |---|---|
  | 2014-15 | 17% |
  | 2019-20 | 28% |
  | 2020-21 | 56% |

  These are the "ρ > 1" rows of v5_implied_corr. On these days the composite is capped at ρ = 1 and is too wide, and that error comes from the inputs. A composite cannot reproduce a CSO cheaper than its legs allow.
- **Recommended rule: use direct when it is usable and has ≥ 3 G4X quotes with previous-day open interest; otherwise use the composite.** Do not weight the two.
  - The rule changes nothing in 2017–21: 0.5% of days fall back.
  - In 2013-14 it switches 34% of choose-window days to the composite and cuts CRPS by 0.018 [−0.055, 0.000].
  - **This rests on one winter (2013-14, the polar vortex).** There, 2–4-strike G4X marks gave a bimodal direct density.
  - Every weighted blend was worse than direct in the check window.

## 2. Method
**Legs.** For each date:
- Black-76 IVs (r = 0) from out-of-the-money ON settles ≥ 0.002 on March (H) and April (J) of delivery year Y. There are 30–60 quotes per leg; the median is 47 for H and 41 for J.
- Raw SVI fitted to total variance w(k) with a robust (soft-L1) loss, in two passes. The second pass drops quotes more than max(4 MAD, 3 vol points) off the first fit; these are stale settles, such as 14 Nov 2018.
- Lee's wing bound b(1+|ρ|) ≤ 2 is imposed by the parametrisation. A penalty applies to Gatheral's g(k) < 0 within ±0.1 of the quoted range.
- Beyond the quoted strikes, IV is linear in k with slope |dσ/dk| ≤ 0.5. **The far squeeze tail beyond the last quoted strike, about +25–35% for H, is therefore model extrapolation.**
- The March marginal is Breeden–Litzenberger on the fit. Its options expire with the G4X: the trading day before H's last trade day. This was verified against futures data in every year 2011–21.
- Each marginal is shifted to mean F. The shift is ≤ 0.2¢ except on blow-up days (0.8¢).
- Residual negative BL mass is clipped and flagged. It is < 0.05 on 99.7% of days.

**April at the March expiry.** J's options expire about 21 trading days after T. Remove the April contract's expected variance over its last month:

  σ_h²·T = σ_J²·T_J − (q·σ_J)²·(T_J − T),   floor σ_h ≥ 0.5·σ_J.

The smile shape is carried in standardised moneyness:

  w_h(k) = w_J(k·√(W_J/W_h))·W_h/W_J.

- **Approximation.** The smile shape over [0, T] is assumed equal to that over [0, T_J].
- Two choices of q were tried:
  - the walk-forward implied ratio q̂ = IV_J(on the March expiry day)/IV_J(d), the median of past winters by tenor bucket; it runs 1.0–1.3;
  - q = 1.
- On dense choose-window days, q̂ made the CSO Fréchet-infeasible on 26% of days against 3% with q = 1, and fit the tails no better. **q = 1 was chosen.** With q = 1, σ_h = σ_J, which errs toward too much April variance. It therefore slightly understates decoupling for a given ρ.

**Copula and ρ.** Common random numbers (120k antithetic pairs). Families:
- Gaussian;
- t with ν = 4;
- mixture: with probability 5% the legs follow ρ = 0.3, otherwise ρ.

The ρ source (`corr_mode`):
- **cso:** ρ solved so the composite E|S − F| equals the G4X ATM Bachelier straddle. The straddle is valid only with previous-day near-money G4X OI > 0 and nv outside the 1.15–1.21 default band (the reviewer_v5 rule).
- **hist:** EWMA ρ of daily H/J log returns minus the median past-winter (EWMA − cso) offset.
- **fixed:** the past-winter median cso-ρ for the tenor bucket.
- **auto:** cso, then hist, then fixed.

All walk-forward fits use completed winters only, so every estimate uses data from before the date.

**Variants tried.**
- 3 families × 4 modes
- 2 choices of q
- a 9-cell mixture grid
- 8 blend rules
- 2 rounds of fit hardening (wing cap, butterfly penalty), made after the first run showed negative-density days. All tables come from the final code.

## 3. Results

**Table 1. Coverage, Nov–Feb trading days**

| Winter | Days | Composite usable | CSO-valid (ρ from CSO) | April proxy | Direct usable |
|---|---|---|---|---|---|
| 2010-11, 2011-12 | 79, 78 | 100%, 100% | 0% (no G4X) | 0 | — |
| 2012-13 to 2015-16 | 307 | 99.7% | 92–99% | 0 | 100% |
| 2016-17 to 2020-21 | 384 | 99.5% | 96–99% | 28 days (2020-21) | 100% |
| 2021-22 (Nov–Dec) | 43 | 100% | 100% | 0 | 100% |

**Table 2. Composite minus direct engine, mean difference [MAE] in P(S_T > F + x), percentage points**

| Window, configuration | +0.10 | +0.25 | +0.50 | q95 median diff ($) |
|---|---|---|---|---|
| Choose, Gaussian/auto (n = 230) | +7.6 [8.2] | +2.9 [4.0] | +0.3 [1.4] | +0.008 |
| Choose, mixture/auto | +6.9 [7.8] | +3.1 [3.8] | +0.7 [1.3] | +0.017 |
| Choose, Gaussian/hist | +6.0 [7.2] | +0.5 [2.9] | −0.9 [1.4] | −0.027 |
| Check, Gaussian/auto (n = 384) | +5.8 [5.9] | +1.8 [2.1] | −0.2 [0.9] | −0.012 |
| Check, t₄/auto | +5.6 [5.8] | +1.8 [2.2] | −0.2 [0.9] | −0.010 |
| Check, mixture/auto | +5.6 [5.8] | +2.1 [2.3] | +0.1 [0.8] | 0.000 |
| Check, Gaussian/fixed | +5.2 [5.9] | +1.1 [2.2] | −0.7 [1.4] | −0.005 |

The shoulder excess at +0.10 is a shape difference. Direct densities from G4X are peaked with fat wings: deep out-of-the-money G4X calls settle 2–10× the composite's price. The composite, matched at the money, is closer to normal. Figures `rnd_composite_examples.png` and `rnd_composite_tail.png`.

**Table 3. What the copula changes (all days of each ρ source; mean P, %)**

| ρ source | P(>F+0.25): Gauss / t₄ / mix | P(>F+0.50): Gauss / t₄ / mix | Median ρ: Gauss / t₄ / mix |
|---|---|---|---|
| cso (715 days) | 9.68 / 9.67 / 10.04 | 5.26 / 5.32 / 5.64 | 0.975 / 0.976 / 0.990 |
| hist (891) | 7.74 / 7.88 / 8.39 | 4.01 / 4.09 / 4.55 | — |

With ρ fixed (not recalibrated), the mixture adds 0.5 point at +0.50 and the t copula 0.1. Recalibrating to the CSO ATM absorbs most of it.

**Table 4. Calibration at realised S_T, same 691 days, 9 winters (2012-13 to 2020-21)**

| Route | CRPS (mean) | PIT mean | PIT outside [5, 95]% | P(>F+0.10): mean p / freq / Brier | P(>F+0.25): mean p / freq / Brier | P(>F+0.50): mean p / freq |
|---|---|---|---|---|---|---|
| Direct engine | 0.073 | 0.495 | 6.1% | 12.0 / 8.5 / 0.075 | 7.7 / 5.8 / 0.058 | 5.0 / **0** |
| Composite Gaussian/auto | 0.089 | 0.472 | 3.9% | 18.1 / 8.5 / 0.074 | 9.5 / 5.8 / 0.055 | 5.0 / **0** |

- Composite over all 11 winters: CRPS 0.077 [0.045, 0.116]; the t and mixture variants are within ±0.001.
- Composite minus direct, same days, winter-block bootstrap:
  - choose: −0.003 [−0.055, +0.041]
  - check: +0.023 [+0.005, +0.049]
- By winter, the composite is better only in 2013-14 (0.217 against 0.272) and ties in 2016-17. It is worst in 2018-19 (0.188 against 0.114), when CSO-implied ρ swung between 0.1 and 0.95 from day to day.
- **Both routes overprice the upside, which is the known H−J premium.** Realised S_T never exceeded F + 0.50 on any scored day: zero events in 11 winters. The far-tail calibration cannot be tested in development. The Brier scores at +0.10 and +0.25 tie.
- Figure `rnd_composite_calibration.png` (PIT; CRPS by winter). Daily rows overlap heavily within a winter, so there are only 9–11 effective observations.

**Table 5. Selection and blend rules, CRPS relative to direct-always (winter-block CI)**

| Rule | Choose (n = 230) | Check (n = 384) | Share on composite (choose / check) |
|---|---|---|---|
| Direct always | 0 | 0 | 0 / 0 |
| Composite always | −0.003 [−0.055, +0.041] | +0.023 [+0.005, +0.049] | 100 / 100% |
| 50/50 CDF blend | −0.004 [−0.031, +0.017] | +0.009 [+0.002, +0.021] | — |
| w = min(1, n_OI/8) | −0.015 [−0.044, 0.000] | 0.000 | — |
| **Direct if usable and n_OI ≥ 3, else composite** | **−0.018 [−0.055, 0.000]** | **0.000** | 34 / 0.5% |
| Direct if usable and n_OI ≥ 1, else composite | −0.000 | 0.000 | 0.4 / 0% |

Here n_OI counts G4X March quotes used by the engine that had open interest the previous quote day. The gain is all in 2013-14: one winter, with 2–4 strikes and zero OI on 6 days.

## 4. What it means for the tool
- **Primary route: direct. Fallback: composite.** Ship the switch `rnd_composite.select(direct, composite)`.
  - In development it fires on 12% of scored days (83 of 691); 77 of them are in 2013-14.
  - On the locked sample it guards against thin or marked-only G4X days.
  - Show the composite's P(> F + 0.25 / 0.50) alongside as a cross-check. A gap of more than 5 points between the routes is itself a data-quality warning; that happens on 11% of check days at +0.25.
- **For the storage map, only the deep-deficit (≤ −400 Bcf) threshold is informative.** That maps to the upper spread tail (P(> F + 0.25 to 0.50)). There the composite agrees with direct to about 1 point. Its +0.10 shoulder is biased, but that shoulder maps into the uninformative part of the storage range.
- **For a long-vol desk.** The "squeeze = decoupling" intuition holds, but the market expresses it through the *level* of ρ (0.1–0.4 in Nov 2018) and through March call skew, not through the copula's tail shape. A Gaussian copula at the right ρ is enough. Getting ρ right needs the CSO.

## 5. Caveats
- **Few winters.**
  - Scoring rests on 9–11 winters with no event beyond F + 0.50.
  - The blend gain rests on 2013-14.
  - The mixture choice and q = 1 were chosen on 2–3 winters.
- **Frechet-infeasible CSOs** (up to 56% of days in 2020-21, where CSO marks are thin) force ρ = 1. That composite is too wide; the flag is `rho_calib = rho_cap`.
- **The April horizon assumption** (q = 1 plus smile-shape scaling) is unverifiable without April options expiring at T.
- **The far tail** beyond the quoted ON strikes is extrapolated, with the IV slope capped at 0.5.
- F is the futures spread here; the direct engine uses the parity forward. They differ by up to 6.5¢ on blow-up days.
- The 2010-11 to 2012-13 rows used a slightly tighter ρ-solver tolerance (≤ 2e-4 in ρ). This is immaterial.

## 6. Proposed frozen test for 2022–2026 (log; the orchestrator decides the gate)
- Run `spread_density_composite(d, Y, 'auto', 'gauss')` with `RND_DATA_CUTOFF` or `COMPOSITE_CUT` set to the locked end, `COMPOSITE_CACHE` set to a fresh directory, and the options and futures paths pointed at the full files. Use every Nov–Feb trading day.
- **RC-1, coverage.** The composite is usable on ≥ 95% of days (dev: 99.7%).
- **RC-2, cross-check.** On days where direct is usable with n_OI ≥ 3, the mean |composite − direct| in P(> F + 0.50) is ≤ 2 points (dev check: 0.9).
- **RC-3, selection.** CRPS(select rule) − CRPS(direct always) ≤ +0.005 with a winter-block CI (dev check: 0.000).
- Log the Fréchet-infeasible share and the days that fell back.

## 7. How to rerun
Run from the repo root with `PYTHONPATH=.`. Each step takes under 10 minutes on 1 CPU:
```
python -m src.tool.rnd_composite_eval direct      # direct-mixture table (v5_implied_storage cache)
python -m src.tool.rnd_composite_eval mixgrid     # mixture-copula grid, 2014-16 (35 s)
python -m src.tool.rnd_composite_eval step1 2011 2012 2013 2014   # daily tables, ascending winters (~2 min/winter)
python -m src.tool.rnd_composite_eval step1 2015 ... 2022
python -m src.tool.rnd_composite_eval blendeval   # vs mixture direct (5 min)
python -m src.tool.rnd_composite_eval vs_engine   # vs shipped direct engine (7 min)
python -m src.tool.rnd_composite_analyse          # tables (t_*.csv) and 4 figures
```
- Caches are in `data/processed/research_tool_composite/`, tagged with the cut.
- Winters must run in ascending order, because hist and fixed read past winters' ρ tables.
- Engine comparison tables (`t_blend_engine.csv`, `t_compare_engine.csv`; Tables 2, 4 and 5): `python -m src.tool.rnd_composite_engine_cmp`, which takes about 10 s.

## 8. New avenues
- **Calibrate the April horizon jointly from the CSO**, by solving for q and ρ on dense days. Value: moderate. It would remove most infeasible days.
- **Fit the composite's shape to the G4X wings** (a mixture copula fitted to 2–3 OTM CSO strikes instead of only the ATM). Value: moderate. It turns the fallback into a partial-information direct fit.
- **Use the May contract** to bracket April's last-month variance. Value: low to moderate.
