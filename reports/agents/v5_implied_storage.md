# v5_implied_storage: what G4X March options say about end-of-March storage

*Agent `v5_implied_storage`, 24 Nov 2026. Development data only:*
- *Options come from the raw near-money file, filtered to G4X, cm = 3 and date ≤ 2021-12-31 on load.*
- *Storage, weather and Model 2 rows are all ≤ 2021-12-31.*
- *One exception, used under brief rule 1: for positions opened in Nov–Dec 2021, I read the H22/J22 futures settles from Jan–Feb 2022, up to the March-2022 option expiry. End-March 2022 storage is never read, so 2021-22 has a trade outcome but no storage outcome.*
- *Choose = decision dates in 2014–2016. Check = 2017–2021.*

## 0. The default-settlement warning
The orchestrator warned that many G4X settles are exchange defaults. I checked every G4X March quote I use.
- **Test.** For each quote I computed the Bachelier normal vol. A day was flagged as a default if at least half its usable out-of-the-money quotes are pinned in [1.15, 1.21], or if its vols are flat (IQR < 0.03) around 1.18.
- **Days flagged: 1 of 734 RND days** (1 Nov to option expiry, winters 2012-13 to Dec 2021). That day is excluded. The share of individual pinned quotes is 0–6% by winter.
- **Why so few here.** The pinning in `spreads.csv` sits mostly in far-dated rows: 11% of March-cid rows outside their own winter, against 2.4% inside it. Within its own winter, the March contract was traded (open interest about 25k–140k), and the in-season "1.18" ATM readings move day to day, as genuine prices do.
- **What remains a concern.** The share of zero-volume days is high in 2012-13 (64%), 2015-16 (38%), 2016-17 (26%) and 2020-21 (28%). On those days the settles are marks, not trades. I keep them, and the caveats return to this.

**Code:** `src/research/v5_implied_storage/defaults.py`. **Output:** `defaults_flag.csv`.

## 1. Plain-English findings
- **The options price a much lower end-March storage level than physics does.** Mapped through the walk-forward v3 payout curve:
  - In the check window, the market's *median* end-March storage sat **359 Bcf below** the 5-year path (winter-block CI −521 to −197).
  - It was 290–540 Bcf below the 5-year norm in every winter except 2015-16.
  - Physics (the 5-year path, or the P-lite and Model 2 medians) put it near or above the norm in four of those six winters.
- **The storage risk premium sits in the moderate tail, not the deep tail.** The headline event is a deficit of at least 400 Bcf below the 5-year norm.
  - In 2017–21 the options implied **P_Q = 0.45**, against **0.23** (P-lite) or **0.33** (Model 2, centred on the 5-year path).
  - That is a premium of **+0.21 [0.13, 0.30]** and **+0.12 [0.06, 0.18]**, measured in spread space with the map's noise included.
  - For a 2014-size deficit (−800 Bcf), the implied probability (0.125) is close to physical (0.06–0.15). Its gap CI spans zero against both Model 2 variants.
- **The storage translation only works where the map has slope.** At a −200 Bcf deficit, the expiry spread is within one residual standard deviation (about 5¢) of the "ample storage" value in 2020-21 and 2021-22, so it is not invertible there. On average, 15% of Q-mass sits where storage cannot be read: the flat region, S_T < y(+300 Bcf).
- **Physics was better calibrated, but this rests on one or two winters.** Across 7 winters, the winter-weighted Brier score for "x < −400" is:
  - 0.167 for the options
  - 0.110 for P-lite
  - 0.086 for Model 2 (5 winters)
  - 0.144 for climatology

  Only **one** winter (2018-19) had a deficit beyond −400 Bcf, and two (2017-18, 2018-19) beyond −200 Bcf.
- **Selling the tail pays, but the storage signal adds nothing.**
  - A weekly short G4X March call spread (ATM+0.25/+0.75) earned **+$113 per lot per trade [50, 190]** in the choose window and **+$231 [86, 390]** in the check window, after costs of 4% per leg.
  - The implied-minus-physical gap does not predict that P&L: HAC t = 0.19. The spread level does, at t = 10.7.
  - The gap filter chosen on 2014–16 *lowered* the check-window mean to +$189.
  - Per unit of drawdown, the short call spread (Calmar 0.03; 2014 drawdown −$4.9k) and the short call (0.08; Nov 2018 drawdown −$10.7k) are both worse than v4's short H−J or half size (0.115).

## 2. Method
**Intuition.** A G4X March call pays if H−J at expiry exceeds K. The v3 payout curve says H−J at expiry is high only when end-March storage x is low. So "the spread finishes above y(x₀)" is the same event as "storage finishes below x₀". The slope of call prices in K therefore gives the market's probability of a storage deficit.

1. **Risk-neutral density (`rnd.py`, `step1_rnd.py`).**
   - **Underlying.** S_T is the H−J futures spread on the option-expiry day. F is that day's H−J settle; the put-call-parity F agrees to within 0.5¢.
   - **Quotes.** Out-of-the-money settles with |K−F| ≤ $2.
   - **Model.** S_T ~ Σwᵢ N(mᵢ, sᵢ²), with Σwᵢmᵢ = F (r = 0). It uses 3 components on days with ≥ 15 quotes, 2 components on 5–14, and a plain Bachelier fit below 5.
   - **Breeden–Litzenberger.** A normal mixture is a proper density, so the fitted calls are monotone and convex by construction, and P_Q(S_T > k) = −∂C/∂K = Σwᵢ Φ((mᵢ−k)/sᵢ).
   - **Fit.** Weighted least squares (weights 1/(price + 0.003)), multi-start with warm starts.
   - **Model-free check.** Adjacent-strike slopes of the raw call curve, with a monotone (isotonic) projection.
2. **Map (`analysis.py`).**
   - **Curve.** y = a + b·exp(−x/τ), fitted with v3's `payout.fit`. Here y is H−J on the *option* expiry day, the thing the options settle on; v3's futures-LTD target is a sensitivity (it changes Q₄₀₀ by −0.02).
   - **Driver.** x = end-March storage minus the mean of up to 5 prior end-March levels.
   - **Walk-forward.** Winter Y is fitted on 2011..Y−1 and needs ≥ 4 winters, so the first is 2014-15. 2013-14 has no map. That is one reason it can only serve as a stress case, the other being its 2 strikes.
   - **Implied probabilities.** P_Q(x < x₀) = P_Q(S_T > y_Y(x₀)) for x₀ = −200, −400, −600 and −800.
   - **Informative threshold.** A threshold counts as informative if y(x₀) − y(+300) ≥ σ_e, the residual s.d. of the walk-forward fit.
   - **Implied quantiles.** x_q = y⁻¹(S_q), defined only where S_q > a.
3. **Physical distributions (`physical.py`).**
   - **Model 2.** From the shared dev cache (origins Nov 2016 – Dec 2021, p1–p99). It is a walk-forward cache built by the orchestrator. Every origin is ≤ 2021 and uses data ≤ origin, so there is no leakage. It cannot exist before 2016: Model 1's features need 5 years of storage, and its error pool starts in 2016.
   - **P-lite**, needed for the choose window. Built as-of each origin:
     - an OLS of the weekly Nov–Mar storage change on HDD, HDD², seasonal terms, a holiday dummy and a trend;
     - every prior weather year as an analog;
     - 4-week blocks of in-sample residuals.
   - **Agreement on the overlap.** Correlation of P(x<−400) with Model 2 is 0.79. P-lite's raw median sits 145 Bcf below the 5-year path, against Model 2's 263.
   - **Primary versions** are centred on the 5-year path (v3's bias-robust convention); raw versions are sensitivities.
   - **Spread space.** The fair comparison puts the physical distribution into spread space: P_P(S_T > y(x₀)) = mean over scenarios s of Φ((y(x_s) − y(x₀))/σ_e). This "noise-aware" version is the one I call the premium.
4. **Tradability (`trade.py`).**
   - **Entries.** On each weekly decision date (the Model 2 timing: storage week F, settle at the first trading day ≥ F+7, at least 5 trading days before expiry): sell a call spread K₁ = round₀.₀₅(F+0.25), K₂ = round₀.₀₅(F+0.75), or a single call at K₁.
   - **Pricing.** Each option is priced at its settle if that strike is quoted (66% of K₁, 57% of K₂), otherwise at the fitted density.
   - **Exit and cost.** Held to expiry. Cost is 4% of each leg's premium (3% and 5% as sensitivities, minimum one tick).
   - **Marks.** Daily, by the same rule, for drawdowns.
   - **Benchmarks.** v4's short H−J (first November decision date to the futures LTD, $80 round trip) and half size.
   - **Uncertainty.** Winter-block bootstrap CIs (4,000 draws) and HAC with 13 lags (the overlap is ≤ 15 weeks).

## 3. Results
**Fit quality (Table 1).** Median price RMSE in $/MMBtu.

| Winter | Quotes/day | RMSE (median / p90) | Butterfly violations in raw settles | Zero-volume days |
|---|---|---|---|---|
| 2013-14 (thin, 2 strikes) | 2 | 0.022 / 0.030 | – | 16% |
| 2014-15 (dense) | 48 | 0.004 / 0.013 | 20% | 3% |
| 2015-16 (dense) | 62 | 0.001 / 0.002 | 2% | 38% |
| 2016-17 | 11 | 0.004 / 0.008 | 10% | 26% |
| 2017-18 to 2021-22 (thin) | 8–13 | 0.002–0.011 / 0.003–0.022 | 0–13% | 9–28% |

- **Mixture versus model-free BL** at the −400 threshold, on 79 days with ≥ 15 quotes: mean |difference| 0.057, correlation 0.92. At −200: 0.043 and 0.95.
- **2013-14** cannot be fitted (2 strikes, 2¢ error). It enters only as a model-marked stress case.

**Map, walk-forward.**
- τ runs from 545–634 Bcf (2015–16) down to 373–434 (2019–22), with σ_e = 4.2–5.3¢.
- The −400 threshold is a spread of $0.00–0.04, and −800 is $0.18–0.23.
- The −400 threshold is informative in every winter; −200 is not in 2020-21 or 2021-22.
- 8% of Q-mass lies below the asymptote a, where no storage level can produce the spread.

**Table 2. Storage risk premium, P(x < x₀): implied minus physical** (spread space, noise included; [winter-block 95% CI])

| x₀ | Window (weeks, winters) | P_Q | P-lite (centred) | Gap vs P-lite | Model 2 (centred) | Gap vs M2 | Gap vs M2 raw |
|---|---|---|---|---|---|---|---|
| −200 | choose (36, 3) | 0.52 | 0.43 | +0.10 [−0.10, 0.51] | – | – | – |
| −200 | check (72, 6) | 0.67 | 0.40 | **+0.26 [0.18, 0.35]** | 0.48 | +0.19 [0.11, 0.26] | +0.12 [0.08, 0.17] |
| −400 | choose | 0.34 | 0.28 | +0.05 [−0.14, 0.48] | – | – | – |
| −400 | check | 0.45 | 0.23 | **+0.21 [0.13, 0.30]** | 0.33 | +0.12 [0.06, 0.18] | +0.05 [0.02, 0.11] |
| −800 | choose | 0.09 | 0.09 | +0.01 [−0.06, 0.12] | – | – | – |
| −800 | check | 0.13 | 0.06 | +0.06 [0.01, 0.13] | 0.11 | +0.01 [−0.04, 0.06] | −0.03 [−0.09, 0.04] |

- **Storage-space gaps** (no map noise) are larger, e.g. +0.26 at −400 in the check window against P-lite.
- **By lead time** (−400, P-lite, all winters):
  - more than 12 weeks: +0.20 [0.07, 0.33]
  - 8–12 weeks: +0.16 [−0.02, 0.32]
  - 4–8 weeks: +0.19 [0.07, 0.29]
  - under 4 weeks: +0.07 [−0.03, 0.18]

  The premium decays into expiry, consistent with v3's theta.
- **By winter** (Fig. `figures/v5_implied_storage_tails.png`):
  - The gap is largest in 2016-17 (implied 0.38 against 0.07–0.15), 2017-18, 2020-21 and Nov–Dec 2021 (0.59 against 0.21–0.33).
  - It is about zero in 2014-15, 2015-16 and 2018-19. In those three winters physics itself saw a deficit, or ample storage.
- **In Bcf** (Fig. `figures/v5_implied_storage_quantiles.png`):
  - Implied median minus the 5-year path: −152 [−844, +167] in the choose window; **−359 [−521, −197]** in the check window.
  - Implied 10th percentile minus Model 2's: −164 [−213, −112].

**Table 3. Calibration against the realised end-March x** (winter-weighted Brier score; lower is better; winters with an outcome 2014-15 to 2020-21)

| Event | Winters with event | Options (Q) | P-lite centred | P-lite raw | Model 2 centred | Model 2 raw | Climatology (walk-forward) |
|---|---|---|---|---|---|---|---|
| x < −200 | 2 of 7 | 0.276 | 0.250 | 0.138 | 0.113 | 0.185 | 0.254 |
| x < −400 | 1 of 7 | 0.167 | 0.111 | 0.082 | 0.086 | 0.125 | 0.144 |
| x < −800 | 0 of 7 | 0.023 | 0.021 | 0.031 | 0.026 | 0.045 | 0.027 |

- **PIT of the realised expiry spread under Q:** mean 0.47, ranging 0.06 (2016-17) to 0.71 (2020-21). In spread space the options' centre is unbiased.
- **The paradox, which is the key caveat.** Realised spreads near $0.00–0.06 map to x ≈ −300 to −450 Bcf. Yet 2020-21 had x = −34 at S_T = +$0.02. In the region where the market's median lives, the map's 5¢ noise is as large as the storage signal.
- **The rest of the "implied deficit" is the premium.** v3 measured it as the 18¢ gap between F and its actuarial value.

**Table 4. Tradability** ($ per lot = 10,000 MMBtu; net of costs at 4% per leg; winter-block CI)

| Strategy | Choose 2014–16 | Check 2017–21 | Worst trade | Worst mark-to-market drawdown |
|---|---|---|---|---|
| Short call spread +0.25/+0.75, every week | +113 [50, 190] (65) | +231 [86, 390] (64) | −1,371 (2014) | −4,931 (2014, model-marked) |
| Short call +0.25, every week | +386 [120, 735] | +1,011 [274, 2,145] | −1,360 | **−10,743 (Nov 2018)** |
| Short H−J futures, every week | +363 [−426, 1,305] | +1,532 [185, 3,194] | −3,070 | −12,010 |
| Short call spread, gap > 0.1 (chosen) | +173 (11, 2 winters) | +189 [102, 293] (45) | −20 | −589 |

- **Selection.** The gap threshold θ ∈ {none, 0, 0.1, 0.2, 0.3} was chosen on the choose window, requiring at least 5 trades in at least 2 winters. It picked θ = 0.1.
- **Costs** of 3% or 5% move the call-spread mean by only ±$9–17.
- **Regressions of weekly call-spread P&L** (2014-15 to 2020-21, HAC 13):
  - on the gap: t = 0.19, R² 0.00
  - on Q₄₀₀: t = 4.3
  - on gap + F: gap t = 0.9, F t = 10.7

  The P&L is the spread-level (premium) effect, not the storage-specific gap.

**Table 5. Once per winter, at the first November decision date (v4-comparable, 2012-13 to 2020-21)**

| Variant | Mean P&L per winter [CI] | Worst winter | Worst drawdown | Calmar (mean / \|worst drawdown\|) | 2021-22 (exception) |
|---|---|---|---|---|---|
| Naked short H−J | 1,387 [50, 2,496] | −3,070 (2014) | −12,010 (2014), −11,980 (2018) | 0.115 | +6,080 |
| Half size | 693 | −1,535 | −6,005 | 0.115 | +3,040 |
| Short call spread +0.25/+0.75 | 142 [−270, 440] | −1,362 (2014) | −4,922 (2014) | 0.029 | −141 (costs) |
| Short call +0.25 | 873 [122, 1,556] | −1,350 (2014) | −10,743 (2018) | 0.081 | +4,810 |

Fig. `figures/v5_implied_storage_pnl.png`.
- **2014 is the worst expiry case for every short.** H−J expired at $0.405 against an entry of $0.015.
- **Nov 2018 is the worst path for the short call.** The Nov 2018 entries were model-marked: K₁ was quoted on only 7% of 2018-19 entry days.

**Variants tried: about 110 cells.**
- RND: 2 (mixture, and the model-free check).
- Map targets: 2.
- Physical: 4 (P-lite / Model 2 × centred / raw) × 4 thresholds × 2 spaces.
- Calibration: 7 models × 4 thresholds.
- Rules: 2 signals × 5 θ × 2 instruments × 2 windows, plus 3 cost levels.
- Once-per-winter: 4.

## 4. What it means for a gas volatility desk (Statar)
- **The options do not price the 2014 tail more richly than physics; they price the middle.** An inventory-forecasting desk will find that G4X March prices imply a "moderate deficit" (−200 to −400 Bcf) about twice as often as weather scenarios do. For a 2014-size deficit the price is roughly fair.
- **The edge you can monetise is the level of the spread, not the storage disagreement.** The short call spread is cheap insurance *selling*: capped loss of about $5k, small mean (+$0.01–0.02/MMBtu per trade). Selecting entries on "implied ≫ physical" did not improve it.
- **Better inventory forecasts matter for one thing here: when *not* to sell.** In 2018-19 physics itself saw a deficit (implied and physical both about 0.65–0.75), and that is the winter whose path hurt the uncapped sale. That is a risk filter, untestable with one episode.
- **As a quoting tool**, "Q-implied end-March storage" is a useful translation of the curve into inventory language, but only in the −400 to −1,000 Bcf region. Above that, the 5¢ map noise swamps storage.

## 5. Caveats
- **Few winters.**
  - Choose has 3 (2013-14 has no map); check has 5 with outcomes.
  - Calibration rests on 1–2 event winters (2017-18, 2018-19).
  - The −800 comparisons never see a realised event, because 2013-14 has no walk-forward map.
- **Settle quality.** 2014-15 settles violate convexity in 20% of adjacent triples. Zero-volume days are common, so many settles are exchange marks. Thin years rely on 2-component fits with 8–13 quotes. 2013-14 (2 strikes) and the Nov 2018 wing strikes are model-marked.
- **The map is fitted on 4–11 points**, and its convexity rests on 2014.
- **P-lite is cruder than Model 2.** It has no supply-drift feedback and no weather recentring. On the overlap it is narrower in the left tail.
- **Information timing.** The physical models use data as of storage week F, while prices are at F+7, so the market knows one more week of weather and forecasts. This biases physical tails toward being *less* informed, not lower.
- **Costs.** Applied per leg (conservative for a package trade). Fills on thin G4X wings may be worse than 5%.

## 6. Frozen test for the locked 2022–2026 options sample
Run unchanged with `src/research/v5_implied_storage` on the locked panel's raw G4X cm = 3 rows. Settings:
- **Winters.** Y = 2022 (decision dates from 1 Jan 2022 only) to 2026.
- **Map.** Walk-forward on winters 2011..Y−1, target S on the option-expiry day, at least 4 winters.
- **RND.** Same rule as above.
- **Default filter.** Same rule.
- **Physical.** P-lite, centred on the 5-year path. Model 2 centred is reported alongside if its locked cache is opened by the orchestrator.

| ID | Statistic | Pass if |
|---|---|---|
| **S1 (primary): premium** | Mean over decision dates of G = Q₄₀₀ − Psp_P-lite,₄₀₀ (spread space, noise included) | Mean G > 0.10, **and** G > 0 in ≥ 3 of the 4 full winters 2022-23..2025-26 (dev: +0.21 check, +0.16 all) |
| S2: deep tail fair | Mean of Q₈₀₀ − Psp_M2c,₈₀₀ (or the P-lite version if Model 2 is unavailable) | \|mean\| < 0.07 (dev: +0.01 Model 2, +0.06 P-lite) |
| **T1 (primary, tradable)** | Weekly short call spread F+0.25/F+0.75, 4% per leg, held to expiry | Mean net > 0 per trade over all locked trades (dev +$113 / +$231) |
| T2: signal is useless | Mean(T1 restricted to G > 0.1) − mean(T1, all) | ≤ +$50 per trade (pre-registered *negative*: the gap adds nothing) |
| C1: calibration (descriptive) | Winter-weighted Brier for x < −400: Q vs P-lite centred vs climatology | Report; predicted Q worst. No pass/fail (few events) |

**Live 2026-27, G4X Mar-27** (option expiry 23 Feb 2027). Run weekly from the storage week ending 27 Nov 2026 (first decision about 4 Dec 2026) to 12 Feb 2027.
- **Frozen map.** The 2022 map: a = −0.083, b = 0.034, τ = 387 Bcf, σ_e = 0.053. This is fitted on ≤ 2021 data, so no locked outcomes are used. The −400 threshold is S_T > $0.012, and −800 is S_T > $0.184.
- **Log each week:**
  - Q₄₀₀ and Q₈₀₀;
  - the implied median and 10th-percentile x;
  - P-lite and a live Model 2 run (`Simulator().simulate(F)`), both centred;
  - G;
  - the call-spread entry price and its final P&L.
- **Pass if** the season-mean G > 0.10, and the season's aggregate call-spread P&L is > 0. Report the realised end-March 2027 x against all three distributions.

## 7. How to rerun
```
python -m src.research.v5_implied_storage.step1_rnd   # daily RND fits, about 14 min, cached per winter
python -m src.research.v5_implied_storage.run         # panel, premium, calibration, trades (<1 min) -> summary.json
python -m src.research.v5_implied_storage.figures
```
Cache: `data/processed/research_v5_implied_storage/`. It holds `summary.json`, `panel_weekly_final.csv`, `premium_table.csv`, `calibration.csv`, `rules.csv`, `trades_*.csv`, `gof_by_winter.csv` and `defaults_flag.csv`.

## 8. New avenues
- **Storage-conditional risk filter:** stop selling uncapped calls when *physical* P(x < −400) > 0.5. It would have avoided 2018-19's −$10.7k path, but rests on one episode. *Value: moderate (risk), low (evidence).*
- **Structural fit:** price G4X directly with x ~ N(μ_Q, σ_Q) pushed through y(x) + noise. This gives a one-number "market storage view" with the noise handled jointly. *Value: moderate, mostly for communication.*
- **Use the 2022–26 G4X winters** to get more than one tail event. *Value: high for calibration.*
- **Put side:** the 8% of Q-mass below the asymptote (spread below any storage-consistent level) suggests rich puts in ample-storage winters. Test a short put spread at a−0.05. *Value: low to moderate.*
