# v5_vol_rv: volatility relative value across the natural gas curve

*Agent `vol_rv`, 24 Nov 2026. Development data only: options from `data/processed/v5/` (≤ 2021-12-31). Futures, storage and the v4 panel are cut at 2021-12-31 for every feature. **Outcome exception (rule 1):** realised vol over the option's life uses futures settles from 1 Jan to 31 Mar 2022, but only for rows dated ≤ 2021-12-31 whose option expires by 31 Mar 2022. Trade positions are marked only on options data ≤ 2021-12-31, so the last period is truncated. Choose window: 2010/11–2016. Check window: 2017–2021. Units: "vol pts" = 0.01 of annualised Black vol. Trade P&L is in vol pts per unit vega per side, so a book running $10k of vega per vol point on each side makes or loses $10k × that number.*

## 1. Plain-English findings
- **The unconditional VRP in monthly NG options is about zero.** Over the option's life, ATM IV minus the realised vol of the same contract averages −0.2 vol pts in the choose window and −1.0 in the check window, with CIs of about ±3. A cost-free monthly short of every option in months 1–6, delta-hedged, earned −0.1 vol pts/month in both windows. The "IV ≫ RV" picture people often quote (+5 to +13 vol pts against next-20-day RV) is the Samuelson term structure of realised vol, not a premium.
- **The VRP that exists is local and depends on regime:**
  - **Non-winter front months are rich into expiry.** In the last month, IV beats life RV by +3.3 vol pts on shoulder contracts [1.0, 5.5] and +5.4 on summer contracts [2.8, 8.4]. Winter front months are fair (+0.3).
  - **Storage drives winter vol.** When storage is loose (more than 150 bcf above the prior-years norm), winter-delivery IV is **+4.3 vol pts rich [2.7, 6.9]**, rich in 7 of 7 gas years. When storage is tight (below −150 bcf), winter IV is **−10.1 vol pts cheap [−18.0, −2.7]**: realised exceeded implied in 5 of 8 gas years, mainly 2014, 2015, 2018 and 2019. Options do not price tightness.
- **No cross-sectional vol RV book survives the brief's costs.** Ranking months 1–6 on IV minus forecast RV and trading short-rich/long-cheap, vega-neutral, monthly and daily-hedged gives:

  | Book | Gross (vol pts/month) | Net at 1.5% + 2-tick hedges (vol pts/month) | Net Sharpe |
  |---|---|---|---|
  | Best signal (IV − EWMA) | +0.5 choose / +0.8 check, CIs span 0 | −2.3 / −2.4 | −1.5 / −1.0 |

  The signal beats the 95th percentile of random rankings in both windows, so the ranking carries information. But daily delta-hedging alone costs 0.9 vol pts/month, more than the gross edge. Option half-spreads add another 2.1.
- **Realised vol is forecastable and IV is the best single forecaster of life vol.**
  - Next 20 days, months 1–6: HAR with seasonality reaches out-of-sample R² 0.72/0.69 (EWMA 0.67/0.63). Adding IV lifts it to 0.78/0.76.
  - Over the option's life: IV alone gets 0.55/0.57 and HAR 0.24/0.32.
  - Storage and forecast-level features **hurt** out of sample (choose-window R² 0.65 vs 0.72).
- **The term structure tells you about realised vol, not about P&L.** The Jan − Apr IV spread predicts the Jan/Apr ratio of next-20-day realised vol (HAC t = 4.4 choose, 2.2 check). It does not predict the vega-neutral pair's P&L (t = 1.0 and −0.05). Tight storage widens the spread only weakly (t = −2.0 choose, −0.9 check).

## 2. Method
**VRP map (descriptive, no fitting).**
- For every surface row (d, cid): gap_life = atm_iv − RV_life and gap20 = atm_iv − RV20.
  - RV_life = √(252/n · Σ r_t²) over trading days (d, option expiry], using log returns of that futures contract. RV20 is the same over the next 20 days.
- Delta-hedged straddle P&L from `dh_straddle.csv`, normalised two ways:
  - per unit vega: pnl/vega0, in vol pts/day, split into carry (gamma + theta) and vega (IV change);
  - per premium: pnl/P0.
- Buckets:
  - delivery class: winter Z/F/G/H, shoulder J/K/V/X, summer M/N/Q/U;
  - tenor: <1m, 1–3m, 3–6m, 6–12m, 12–18m;
  - trading season: Nov–Mar vs Apr–Oct;
  - storage: sdev = latest public Lower-48 storage minus its same-date mean over the prior 2–5 years, defined from 2012; tight < −150 bcf, loose > +150 bcf;
  - price level: F ≤ 3, 3–4, > 4.
- CIs: 2,000 resamples of whole gas years (Apr–Mar), pooled mean.

**Forecasts (walk-forward).**
- Pooled OLS on log RV, expanding window, refitted at each month start on rows whose target had already resolved. First forecast month: 2011-07.
- HAR1: log RV5, RV21, RV63 of the contract itself, log tenor, winter delivery, winter trading, and winter × winter.
- HAR2 adds storage deviation and the GEFS forecast level (lvl_w1 + lvl_w2). HAR3 adds log IV.
- Benchmarks: EWMA (λ = 0.94), GARCH(1,1) and IV, each raw and with a walk-forward bias correction.
  - GARCH persistence is fitted walk-forward each month on front-month returns. The variance is targeted to each contract's own prior-year level.
- Out-of-sample R² is measured against the walk-forward training mean.

**Trade.**
- Rebalance on the first trading day after each monthly option expiry (126 dates, 2011-07 to 2021-12).
- Universe: contracts with 10–130 trading days to expiry, about 5–6 a month.
- Per contract: open a long ATM straddle at K = F, delta-hedge daily at the settle and hold to the next rebalance. The front month is held to expiry and pays |F − K|.
- Marking: Black-76 at the day's surface smile, interpolated in k = ln(K/F). Validated against the settle-based `dh_straddle`: day-1 P&L correlation 0.92, median absolute difference $0.0007 against a typical |P&L| of $0.006.
- Costs:
  - 1.5% of premium per option side, minimum $0.002 per option. No exit cost at expiry. Sensitivity at 1% and 2%.
  - Futures: $0.002 × |change in hedge| at every daily rebalance, including the initial hedge and the final unwind.
- Book: short the 2 contracts with the highest signal and long the 2 with the lowest, each side scaled to 1.00 vega.
- Signals: S_ewma = log IV − log EWMA (no fitting); S_20 = log IV − HAR1 forecast over 20 days; S_life = log IV − HAR1 forecast over the life.
- Benchmarks:
  - short-all;
  - long-all;
  - random ranking (300 draws);
  - term rule: short the 2 nearest, long the 2 farthest;
  - storage-sign: long all if sdev < 0, else short all. This one is **post hoc**, informed by the map.

**Term structure.**
- WS = IV(Jan of the coming winter) − IV(Apr of the same winter). FB = IV(front) − IV(≈12-month contract).
- Outcomes over the next 20 days:
  - Δlog RV20 between the two legs;
  - the summed per-vega `dh` P&L of leg 1 minus leg 2 (gross).
- Weekly samples. OLS with Newey-West errors (6 lags) and a gas-year bootstrap of the slope.

## 3. Results
**Table 1. IV − RV over life (vol pts; 95% gas-year CI).** All dates 2010–2021.

| Delivery | <1m | 1–3m | 3–6m | 6–12m | 12–18m |
|---|---|---|---|---|---|
| Winter (ZFGH) | +0.3 [−4.2, 4.2] | −1.3 [−6.3, 3.3] | −1.2 [−8.7, 4.9] | −0.7 | −0.8 |
| Shoulder (JKVX) | **+3.3 [1.0, 5.5]** | +1.0 | −0.2 | −0.1 | −0.9 |
| Summer (MNQU) | **+5.4 [2.8, 8.4]** | +1.2 | −0.7 | −1.4 | −1.9 [−4.0, 0.8] |

- Against next-20-day RV the same cells read +3 to +13, rising with tenor (for example winter 3–6m +13.0 [9.4, 16.8]). That is the term structure of realised vol, not money you can earn.
- By contract month (fig. 1, left), Feb and Mar are the cheapest (−2.2 and −1.8) and Nov the richest (+1.1). No single-month CI excludes 0.
- Winter delivery by storage regime:
  - tight: −10.1 [−18.0, −2.7]; leave-one-gas-year-out ranges from −5.7 to −11.9;
  - normal: +0.9;
  - loose: +4.3 [2.7, 6.9].
- Shoulder and summer show the same ordering, with smaller magnitudes (tight −2.6 and −2.4). Price level shows nothing.

**Per-day delta-hedged straddle P&L (all buckets):**
- A long straddle earns **+0.024 vol pts/day [−0.012, 0.064]** per unit vega.
- Carry (gamma + theta) is **−0.054 [−0.082, −0.029]**, so realised < implied on a daily basis.
- The vega term is **+0.098 [0.068, 0.137]**: each contract's IV rolls up the downward-sloping term structure as it ages.
- Net effect: short-vol harvests the carry but pays the roll-up. Per premium, the winter <1m bucket earns +1.0%/day [0.2, 2.0] for the long. That is the Jan 2014, Nov 2018 and Feb 2021 squeezes.

**Table 2. Out-of-sample R² of log RV (universe ≤ 130 tdays; choose / check).**

| Target | EWMA | EWMA+b | GARCH+b | HAR0 | HAR1 | HAR2 (+storage, wx level) | HAR3 (+IV) | IV raw | IV+b |
|---|---|---|---|---|---|---|---|---|---|
| Next 20d | 0.67 / 0.63 | 0.70 / 0.66 | 0.67 / 0.69 | 0.68 / 0.66 | 0.72 / 0.69 | 0.65 / 0.69 | **0.78 / 0.76** | 0.64 / 0.53 | 0.61 / 0.61 |
| Life | −0.45 / −0.25 | −0.18 / 0.11 | −0.23 / 0.15 | 0.22 / 0.27 | 0.24 / 0.32 | 0.11 / 0.31 | 0.49 / 0.54 | **0.55 / 0.57** | 0.47 / 0.56 |

**Table 3. Monthly vega-neutral books** (vol pts/month per unit vega per side; net = 1.5% half-spread + 2-tick hedges). Max DD is on daily cumulative net P&L over 2011–21.

| Rule | Gross, choose / check | Net, choose / check | Net Sharpe, choose / check | Max DD | Jan 14 / Nov 18 / Feb 21 (net) |
|---|---|---|---|---|---|
| S_ewma (L/S) | +0.53 / +0.78 (CIs −0.4..1.6, −0.8..2.3) | −2.32 / −2.37 | −1.54 / −1.04 | −308 | +14 / +11 / −10 |
| S_20 (HAR1) | −0.30 / +0.29 | −3.16 / −2.88 | −1.92 / −1.24 | −394 | +14 / −20 / +1 |
| S_life (HAR1) | −0.15 / +0.38 | −2.97 / −2.79 | −1.91 / −1.32 | −380 | −19 / −33 / −8 |
| Short all | −0.09 / −0.11 | −1.48 / −1.66 | −1.21 / −0.86 | −228 | −13 / −22 / +2 |
| Term (short front, long back) | +0.29 / −0.05 | −2.55 / −3.20 | −1.39 / −1.41 | −399 | −20 / −17 / −6 |
| Random (median of 300) | ≈ 0 | −2.9 | −2.65 / −2.01 (95th pct −1.77 / −1.31) | – | – |
| Storage sign (post hoc) | +0.68 / +0.51 | −0.61 / −1.04 | −0.53 / −0.55 | −111 | +10 / +19 / +2 |

- **Cost split for S_ewma:** options 2.06 and hedging 0.93 vol pts/month, against 0.65 gross. Each 1% of half-spread costs about 1.3 vol pts/month.
- **S_ewma is mostly a term bet.** Its rank correlation with tenor is +0.72: it sells the back months, where IV ≫ trailing RV, and buys the front.
- **Changing the holding period does not rescue it:**
  - Weekly: gross S_ewma +0.17 / +0.31 per week, Sharpe 0.39 / 0.46; net −2.4 / −2.6 per week.
  - Quarterly: gross −0.27 / +2.78, net −3.5 / −0.7.
- **Short front month to expiry, by delivery class, net:** winter −3.7 / −3.0; shoulder +1.6 / −2.2; summer −1.8 / −0.6. None holds in both windows.
- **Long winter-delivery straddles when storage is tight (post hoc):**
  - net +2.65 vol pts/month (choose; 13 months, 2 gas years; CI [−0.2, 9.1]);
  - +1.41 (check; 17 months, 4 gas years; [−3.6, 3.1]);
  - in loose storage the same long loses −2.4 / −3.6.
  - This is the only net-positive cell in both windows. It is post hoc and rests on about 3 tight spells (2014-15, 2018-19 and late 2021).

Figures: `figures/v5_vol_rv_vrp.png` (map and storage regimes), `figures/v5_vol_rv_trade.png` (cumulative net P&L with the random band) and `figures/v5_vol_rv_term.png`.

**Table 4. Term structure (weekly samples; slope, HAC t).**

| Regression | Choose | Check |
|---|---|---|
| WS on sdev (per bcf) | −0.001 (t −2.0) | −0.005 (−0.9) |
| FB on sdev | +0.005 (2.9) | +0.010 (0.9) |
| Δlog RV20 (Jan/Apr) on WS | **+0.029 (4.4)**, R² 0.23 | **+0.005 (2.2)** |
| Δlog RV20 (front/12m) on FB | +0.012 (4.5) | +0.007 (2.3) |
| Jan/Apr pair P&L on WS | +0.27 (1.0), CI [−0.19, 0.71] | −0.01 (−0.05) |
| Front/12m pair P&L on FB | +0.16 (0.7) | +0.34 (2.3, n = 42, CI spans 0) |
| Jan/Apr pair P&L on sdev | −0.001 (−2.0) | −0.012 (−1.7) |

The WS level is 4.8 ± 3.0 in the choose window and 17 ± 15 in the check window. The surface was thin after 2016, so the check-window WS is noisy.

**Variant count: about 200 cells.**
- VRP map: about 60 descriptive cells, nothing selected.
- Forecasts: 11 models × 2 targets × 2 universes = 44.
- Books: 18 rule × holding-period combinations × 4 cost levels = 72, plus 300 random draws as one benchmark, 3 front-month and 6 regime cells.
- Term structure: 8 regressions × 3 windows = 24.
- Selection: no parameter was tuned. HAR1 was fixed a priori; n = 2 per side; the universe of 10–130 days was fixed by coverage after 2016.

## 4. What this means for a gas volatility desk
- **Do not run a systematic short-vol or cross-sectional monthly ATM book on NG at these costs.** The gross edges are 0.5–0.8 vol pts/month. Round-trip options plus daily hedging cost 3 vol pts. Cutting hedge frequency (bands) and trading only into expiry are the levers, and even at zero option spread daily hedging alone eats the edge.
- **For a long-vol-biased fund there is a useful regime map.** Winter vol is cheap when storage is tight and rich when it is loose. Options do not price tightness, consistent with v4's finding that IV jumps only *after* blow-ups. The cost of carrying long winter gamma is lowest exactly when storage is below normal. Treat it as a sizing tilt, not a stand-alone strategy, because it rests on about 3 episodes.
- **Where to sell:** short-dated shoulder and summer months into expiry are the only consistently rich cells, but at +3 to +5 vol pts gross they do not clear daily-hedge costs at the front.
- **Use IV as the forecast of life vol and HAR + IV for the next month.** Storage and weather-level features did not help out of sample.

## 5. Caveats
- Costs are assumed, not observed. The front-month hedge cost of 2 ticks × Σ|Δdelta| is about 1 vol pt per position, the largest single drag; a desk hedging on bands or intraday will differ.
- Marks between settles use smile interpolation, validated at r = 0.92 against settle-based P&L. Strikes are at exact F, not the strike grid.
- After 2016 the surface holds 6–10 contracts a day, so the check-window universe is about 5 names and quartile ranks are coarse.
- Storage deviation uses prior 2–5 years from a 2010-start series, so it starts in 2012 and its "normal" drifts.
- Regime results rest on 2014-15, 2018-19 and 2021 (label: **1–3 episodes**). Jan 2014, Nov 2018 and Feb 2021 dominate the winter buckets.
- RV_life for 2021 rows uses Jan–Mar 2022 settles (allowed outcome window). Positions open at 31 Dec 2021 are truncated there.
- The per-day `dh` statistics weight long tenors heavily, because there are more rows per day.

## 6. Proposed frozen test (locked 2022–2026 options sample)
Run unchanged: `src/research/v5_vol_rv` code with `V5` pointed at the locked panel. Same columns (`surface.csv`, `dh_straddle.csv`). Futures and storage cut at the panel end. RV_life only for options that have expired.
- **T-VRV1 (primary): storage regime predicts the winter VRP.**
  - Statistic: Δ = mean gap_life(winter delivery, sdev > +150) − mean gap_life(winter delivery, sdev < −150), pooled over rows, with a gas-year bootstrap.
  - sdev uses the same prior-2–5-year rule.
  - **Pass if Δ > 0 and the loose-regime mean > 0.** Report the CI. If either regime has fewer than 100 rows, the test is "not testable", not a fail.
- **T-VRV2: no unconditional VRP, and no net cross-sectional book.** Run the monthly S_ewma L/S book and short-all with the frozen rules and costs (1.5%, $0.002 per option minimum, 2 ticks per hedge change).
  - Predictions: S_ewma net mean < 0 and short-all gross mean within ±1.0 vol pt/month.
  - Pass if both hold. A positive net S_ewma would *falsify* this study's conclusion.
- **T-VRV3 (post hoc, secondary): long winter-delivery straddles when sdev < −150** (`trade.regime_winter`).
  - Net mean per rebalance > 0.
  - Report months and gas years. Fewer than 6 tight months means "not testable".
- **T-VRV4: term spread predicts relative RV.** Slope of Δlog RV20(Jan/Apr) on WS > 0, HAC (6 lags) t > 2.

**Live 2026-27 (optional).**
- At each monthly rebalance from Dec 2026 to Mar 2027, log sdev from the latest EIA report and the regime.
- If tight: log a paper long book of winter-delivery straddles (10–130 days to expiry), with the T-VRV3 marks and costs.
- If loose: log a paper short of the same.
- Score: sign of the P&L matches the regime prediction.

## 7. How to rerun
`python -m src.research.v5_vol_rv.run` runs in about 2.5 minutes on 1 CPU; add `--force` to rebuild the caches. It writes:
- `data/processed/research_v5_vol_rv/` (`base.csv`, `s1_*`, `s2_*`, `s3_*`, `s4_*`);
- the 3 figures.

Single steps can be run on their own: `python -m src.research.v5_vol_rv.vrp`, and likewise `forecast`, `trade`, `term`.

## 8. New avenues
- **Hedging bands / weekly hedging for the front month.** This could roughly halve the 0.9 vol pts/month hedge drag. *Value: high*; it is the binding constraint.
- **Sell non-winter front-month strangles or straddles into expiry with sparse hedging.** Gross +3 to +5 vol pts. *Value: moderate.*
- **Storage-tightness long-vol tilt on winter months,** combined with v4's protective H−J calls. *Value: moderate*; few episodes, so the 2022–26 sample matters.
- **Cross-sectional ranking within a tenor bucket** (for example, winter vs shoulder at matched T) to strip out the term bet hidden in S_ewma. *Value: moderate.*
- **GEFS ensemble spread as a forecast of next-20-day RV** for winter contracts, when available. *Value: unknown.*
