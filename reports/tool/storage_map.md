# Tool link C: the spread → storage map, with uncertainty

*Agent `storage_map`, 24 Nov 2026.*

**Data rules.**
- Futures and storage rows dated ≤ 2021-12-31 only, filtered on load. Nothing from 2022 is read, including end-March 2022 storage.
- **Pre-2010 data is used** (allowed by SPEC: v4 spent it). EIA front-4 futures supply S_T for 1994–2010. EIA monthly US working gas (NG.N5020US2.M) supplies x for 1994–2021, with priors back to 1989.
- Walk-forward means winter Y is predicted from winters < Y. LOWO (leave-one-winter-out) is descriptive only.

## 1. Findings
- **The best map runs in the direction the tool uses: a direct regression of x on asinh(S_T/5¢).** On walk-forward 2004–21 (18 winters) its CRPS is **203 Bcf**, against:
  - climatology 272: **−69 [−137, −8]**
  - the v3/v5 exponential curve, Bayes-inverted: −20 [−46, +7]
  - a heteroskedastic exponential curve: −29 [−60, −2]

  The exponential forms fail on one winter. Hold out 2003 (S_T = $2.99 at x = −470) and they put it at −1,660 Bcf.
- **The map is noisy everywhere, not only in a "flat region".** Given any single S_T, x has a predictive s.d. of about **345 Bcf**, against 440 for climatology. The 80% interval is about 900 Bcf wide wherever S_T sits. The spread at expiry explains 44% of the variance in x.
- **Honest resolution of the tool:**
  - **It can tell "deficit of 400+ Bcf likely" from "normal or surplus".** P(x<−400) is 0.04 at S_T = −5¢, 0.12 at 0¢, 0.29 at +5¢, 0.58 at +20¢ and 0.81 at +80¢.
  - **It cannot tell −200 from 0.** The forward AUC is 0.61.
  - **It cannot tell −600 from −800.** Its −600/−800 odds are an extrapolation from 2 winters (1994, 2014).
- **The v5 deterministic map overstates the implied deficit by about 2×.** It sends every S_T > 1.2¢ to x < −400; this map gives P(x<−400 | S_T = 1.2¢) = 0.15. For a Q-density centred at 0 (s.d. 15¢), v5 says 0.46, this map **0.21 [0.11, 0.32]**. Most of v5's "storage premium" is map error.
- **Nothing beats a simple pooled fit.** None of these improves held-out scores: price-level controls (ln H, ln(H/J)), a pre/post-2010 break, scaling x by capacity or by the 5-yr level (%), or a heteroskedastic noise term.
- **More curve information would help.** The April−May spread at expiry, J−K, ranks storage much better than H−J does:
  - AUC for x<−200 is 0.97 against 0.84.
  - Walk-forward CRPS is 177 against 203 (difference −25 [−66, +5]).

  A J−K (G4X April) density is the natural second input to the tool.

## 2. Method
**Data (`src/tool/_map_data.py`).**
- **Winters.** One row per winter Y = 1994–2021 (28 winters).
- **S_T.** H−J settle on the March option expiry, the trading day before the futures LTD (LTD = 3 business days before 1 Mar).
  - EIA c1−c2 for Y ≤ 2010. At option expiry c1 = March and c2 = April, as in `src/v4/panel.pre2010_curve`.
  - CME curve file for 2011–21.
  - On the 2011–21 overlap the two sources agree exactly, on the same date and the same S_T.
  - EIA prices are rounded to 1¢, which is immaterial at this noise level.
- **x.** End-of-March US working gas minus the mean of the 5 prior end-March levels, in Bcf.

**Monthly US vs weekly Lower-48 (Table 1).** The weekly level runs about 21 Bcf below the monthly one, mainly because of the reporting date and Alaska. The offset cancels in x: on 2015–21, where weekly priors are complete, x_weekly − x_monthly = −11 ± 13 Bcf with correlation 0.9997. That is negligible against the 345 Bcf map noise. **The tool can therefore score realised outcomes with the weekly L48 x, as v5 does.**

| Table 1 | n | mean | s.d. |
|---|---|---|---|
| End-March level, weekly − monthly (2010–21) | 12 | −20.7 | 26.0 |
| x_weekly − x_monthly (2015–21) | 7 | −10.9 | 12.8 |

**Alternative x definitions** (Spearman ρ with S_T, 28 winters):

| Definition | ρ with S_T | Notes |
|---|---|---|
| 5-yr deviation (Bcf) | −0.64 | |
| % of 5-yr average | −0.64 | Walk-forward CRPS −6 [−29, +15] vs Bcf, so no gain |
| Scaled by working capacity | −0.64 | |
| Level minus 10-yr linear trend | −0.49 | |

Days-of-cover was not tested (no demand data). **I keep x in Bcf vs the 5-yr mean.** Starting in 1997 (avoiding pre-Order-636 priors) changed nothing material.

**Candidates (Table 2): 24 variants in total.**
- **Forward curves S_T | x, inverted by Bayes** with a climatological N(μ, σ) prior on x. Being Bayesian, they are never undefined below the asymptote; there the posterior falls back to the prior.
  - `homo_exp`: the v3/v5 curve.
  - `hetero_exp`: noise s.d.² = s0² + (k·b·e^(−x/τ))².
  - `hetero_hinge`: monotone, convex, piecewise linear.
  - Variants on ln(H/J), on x in %, and with 1997 and 2010 start years.
- **Direct inverse regressions** x | S_T ~ N(α + β·asinh(S_T/c), σ²):
  - c ∈ {2, 5, 10, 20}¢;
  - heteroskedastic log σ;
  - a % target;
  - ln(H/J);
  - with a ln H control;
  - with a post-2010 shift or slope;
  - post-2010 only;
  - a logistic regression per threshold, as a check.
- **Ceilings**, using curve information at expiry that an S_T density does not carry: J−K, alone and with S_T.

**Map uncertainty.** Pairs bootstrap over winters (B = 200 in the tool, 30–100 in evaluation). p(x | S_T) averages the draws' predictive normals (inflated by √(n/(n−2))); the tool reports P(x<x0) from this mixture plus a 5–95% band across draws.

**Tool integral.** P_Q(x<x0) = ∫ p(x | s) q(s) ds, where q is the RND of S_T, binned at 0.5¢ (5¢ above $1).

## 3. Results
**Table 2. Held-out prediction of x from S_T.**
- Walk-forward covers 2004–21 (18 winters); LOWO covers 1994–21 (28).
- CRPS and the median's MAE are in Bcf. The log score is per Bcf.
- Coverage is of the central 50/80/90% intervals.

| Spec | WF CRPS | WF log score | WF MAE | WF cover 50/80/90 | LOWO CRPS | LOWO cover 50/80/90 |
|---|---|---|---|---|---|---|
| Climatology | 272 | −7.65 | 364 | .50/.78/.78 | 256 | .54/.79/.86 |
| **inv_asinh, c = 5¢ (default)** | **203** | **−7.31** | **289** | .44/.72/.89 | **214** | .43/.75/.86 |
| inv, c = 2¢ / 10¢ / 20¢ | 208 / 203 / 207 | −7.34 / −7.31 / −7.34 | – | – | 206 / 224 / 237 | – |
| inv, heteroskedastic | 207 | −7.40 | – | .44/.72/.78 | 216 | .46/.71/.86 |
| inv, x in % of 5-yr average | 197 | −7.28 | – | .56/.72/.94 | 202 | .46/.71/.86 |
| inv, ln(H/J) | 205 | −7.31 | – | .44/.67/.78 | 202 | .43/.64/.89 |
| inv + ln H | 212 | −7.36 | 294 | .50/.72/.89 | 221 | .50/.75/.89 |
| inv + post-2010 dummy | 220 | −7.36 | 312 | .50/.67/.89 | 220 | .46/.75/.86 |
| homo_exp (v5 form, Bayes) | 223 | −7.35 | 314 | .50/.72/.83 | 243 | .57/.75/.86 |
| hetero_exp (Bayes) | 232 | −7.51 | 329 | .33/.67/.78 | 243 | .39/.71/.79 |
| hetero_hinge (Bayes) | 261 | −7.58 | 368 | .44/.61/.72 | 263 | .50/.71/.75 |
| *Ceiling: asinh(J−K)* | *177* | *−7.21* | *237* | .61/.78/.89 | *193* | .54/.82/.89 |
| *Ceiling: S_T + J−K* | *191* | *−7.29* | *262* | .56/.78/.89 | *197* | .54/.79/.89 |

- **Regime check.** Slope on asinh(S/5¢): pre-2010 −147 (s.e. 52), post-2010 −344 (s.e. 83, driven by 2014). Held-out 2013–21 CRPS: pooled 220, era interaction 222–225, post-2010 only 224. **I pool all 28 winters.**
- **The % variant** scores slightly better, within noise, but widens recent intervals (5-yr level now ~1,750 Bcf vs ~1,100 in the 1990s). Kept as a sensitivity.
- **Coverage** is slightly under nominal at 50% (0.43–0.44), within binomial noise for n = 18–28.

**Table 3. Threshold events, walk-forward Brier vs climatology** ([winter-block 95% CI] on the difference)

| Event | WF events (of 18) | Map | Climatology | Difference | LOWO events (of 28) | LOWO difference |
|---|---|---|---|---|---|---|
| x < −200 | 3 | 0.116 | 0.177 | −0.06 [−0.14, +0.01] | 7 | −0.08 [−0.15, −0.01] |
| x < −400 | 2 | 0.057 | 0.109 | −0.05 [−0.12, −0.01] | 5 | −0.09 [−0.17, −0.03] |
| x < −600 | 1 | 0.021 | 0.057 | −0.04 [−0.10, −0.00] | 2 | +0.01 [−0.07, +0.10] |

- A logistic regression fitted per threshold gives the same Brier scores (−400: 0.055 WF), so the Gaussian inverse is not losing classification skill.
- In-sample, S_T ranks x<−400 almost perfectly (AUC 0.98, 5 events). Only 2019 (+2¢, x = −469) sits among the normal winters.

**Table 4. The frozen map (fit on winters 1994–2021, for use from 2022). p(x | S_T), Bcf.**
- Parameters: α = 0, β = −206 (bootstrap 5–95% [−357, −138]), σ = 336.
- Climatology of x: mean +21, s.d. 440.
- Base rates: P(x<−200) = .25, P(x<−400) = .18, P(x<−600) = .07, P(x<−800) = .04.

| S_T ($) | x p10 | x p50 | x p90 | P(<−200) | P(<−400) | P(<−600) | P(<−800) |
|---|---|---|---|---|---|---|---|
| −0.15 | −56 | +392 | +844 | .04 | .01 | .00 | .00 |
| −0.05 | −249 | +180 | +615 | .12 | .04 | .01 | .00 |
| 0.00 | −446 | −20 | +416 | .28 | .12 | .04 | .01 |
| +0.02 | −538 | −108 | +332 | .38 | .18 | .07 | .02 |
| +0.05 | −659 | −219 | +232 | .51 | .29 | .13 | .04 |
| +0.10 | −803 | −344 | +121 | .64 | .43 | .23 | .10 |
| +0.20 | −978 | −489 | −2 | .77 | .58 | .38 | .20 |
| +0.40 | −1,169 | −638 | −126 | .86 | .71 | .53 | .34 |
| +0.80 | −1,366 | −786 | −248 | .92 | .81 | .65 | .48 |
| +3.00 | −1,710 | −1,064 | −473 | .97 | .92 | .83 | .69 |

**Forward direction (S_T from x).** This is needed to put physical scenarios into spread space, as in v5. Use `fit_map(Y, spec="hetero_exp")` with `forward()`.
- 2021 fit: a = −0.094, b = 0.102, τ = 319 Bcf, s0 = 3.3¢, k = 1.48.
- On walk-forward its median log score beats `homo_exp`: 1.21 against 0.89 per $. Its 80% coverage of S is 0.83, against 1.00 for `homo_exp`, which is too wide because of 2003.
- Separability of S_T at x vs at x = 0 (AUC): −200 → 0.61, −400 → 0.68, −600 → 0.71, −800 → 0.73. Below about −400, the deeper deficit adds almost nothing to the expected spread, relative to its noise.

**Illustrative Q-densities** (t₄, centre F, scale s): implied P(x<x0) [5–95% map band], against v5's deterministic map.

| F, s | P(<−200) | P(<−400) | P(<−600) | P(<−800) | x p10 / p50 / p90 | v5 P(<−400) |
|---|---|---|---|---|---|---|
| −5¢, 10¢ | .21 [.13,.27] | .10 [.05,.15] | .05 [.02,.08] | .02 [.00,.04] | −419 / +141 / +667 | .21 |
| 0¢, 15¢ | .35 [.24,.45] | .21 [.11,.32] | .11 [.04,.21] | .05 [.01,.11] | −644 / −18 / +594 | .46 |
| +5¢, 25¢ | .45 [.33,.57] | .31 [.17,.47] | .19 [.07,.36] | .10 [.02,.24] | −821 / −134 / +605 | .59 |

Figures:
- `figures/storage_map_scatter.png`: data, the default map's median and 10–90% band, the forward curve, and the v5 map.
- `figures/storage_map_resolution.png`: P(x<x0 | S_T) with bootstrap bands and base rates.
- `figures/storage_map_wf.png`: walk-forward intervals by winter.

## 4. Recommendations for the tool
1. **Use `inv_asinh` (the default).** Report P(x<−200) and **P(x<−400) as the headline thresholds.**
   - Show −600 and −800 greyed out, labelled "extrapolated, 1–2 events (1994, 2014)".
   - The map band for P(x<−800) is about as wide as the value itself.
2. **Display the map band alongside each probability.** At a typical S_T density it is ±0.10 for P(x<−400). That band is the floor on the tool's precision before any RND error.
3. **State the resolution in plain words:** "The options can distinguish a ≥400 Bcf deficit from a normal end-March. They cannot distinguish −200 from 0, or −600 from −800. Any single expiry spread leaves about ±450 Bcf (80%) of storage uncertainty."
4. **Drop the v5 "implied median x".** In the new map, the implied median x sits close to climatology unless the Q-mass is above about +5¢.
5. **Add a J−K (G4X April, cm = 4) density as a second input** if the RND link can deliver it: x | (S_T, JK_T). This is the one information gain found, and it is not yet significant: −25 [−66, +5] Bcf CRPS.
6. **Refit yearly in April** with `fit_map(Y−1)`; the frozen map for the locked test is `fit_map(2021)`. Label the output as a physical conditional p(x|s) averaged over the risk-neutral q(s).

## 5. Caveats
- **Few winters.** There are 28 in total, with 7/5/2/1 events at −200/−400/−600/−800. The tail of the map rests on 1994, 1996, 2003 and 2014.
- **The S_T–x relation is unstable:** 2003 squeezed to $2.99 at −470; 2001 sat at −4¢ at −357; 2019 at +2¢ at −469. End-February weather drives S_T as much as end-March storage.
- **Pre-2010 S_T uses EIA c1/c2 at a rule-based option expiry.** It matches CME exactly on 2011–21; before that, holidays could shift it by a day.
- **J−K at expiry is unknown at decision time**, like S_T; the ceiling rows bound the value of a second density.

## 6. How to rerun
```
python -c "from src.tool._map_data import winter_table; winter_table(refresh=True)"   # data cache
python -m src.tool.storage_map_eval --baselines     # seconds
python -m src.tool.storage_map_eval --spec inv_asinh   # and each spec in SPECS (2-3 min each, the exp ones)
python -m src.tool.storage_map_inverse_variants     # ~1.5 min
python -m src.tool.storage_map_eval --summary
python -m src.tool.storage_map_figures              # figures + data/processed/research_tool_map/resolution.json
```
API: `m = fit_map(last_winter)` and then `implied_storage(samples | (grid, cdf) | cdf_fn, m)`. It returns `P`, `P_band`, `q`, `density`, `mean`, `post_sd_mean` and `prior_sd`, in about 1 s. Cache: `data/processed/research_tool_map/`.
