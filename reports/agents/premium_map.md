# Idea 3: a premium map of the Henry Hub curve (agent `premium_map`)

*Inputs: CME settles from 2010-06-04 to 2021-12-31 and EIA storage before 2022. Every loader asserts the cutoff, and no 2022+ row was read. Weekly sampling uses Fridays. Code is in `src/research/premium_map/`.*

## 1. Plain-English findings
- **The curve's premium is the widowmaker.** Nine months before expiry, the March−April spread sits on average **+25.9¢/MMBtu (95% CI +15.6 to +34.5; 7.7% of price) above** where it expires. It fell in **10 of 11 years**; the exception was 2014. No other cell is within a factor of 5 of that.
- **Other premia are small, and the significant ones sit next to deep winter.**
  - April−May falls by about +4–5¢ and May−June by about +3¢ into expiry.
  - December−January *rises* by about 3–4¢ (long earns, 10 of 12 years).
  - July−August rises by about 3¢ in its last 1–3 months.
  - Everything else is ≤ 2¢ and insignificant, including the autumn contango pairs and both winter-vs-summer strips. The strips show large means (10–30¢) but huge dispersion (t ≤ 1.4).
- **The premium does not track scarcity or storage space.**
  - The March−April premium is about 10% of price both when storage is projected scarce and when it is projected full, against 4% when it is normal; the slope on the storage deviation is 0.
  - What does move it is **trailing realised spread volatility**: +4.7% of price per sd (CI +1.9 to +6.3). The market charges for recent turbulence, not for physical shortage.
- **Away from March−April the map is not stable.** Cell means from the 2010–15 expiries and the 2016–21 expiries are *negatively* correlated (r = −0.44 excluding March−April and the strips). Only March−April keeps a sizeable, same-sign premium in both halves.
- **A diversified "harvest the map" portfolio fails out of sample.**
  - Cells chosen on 2010–15 earn a gross 2016–21 Sharpe of 0.59 but **−1.63 net of 4 ticks per side**.
  - The hold-to-expiry version scores −0.79.
  - A plain short of March−April alone scores **+0.26 (from November) to +0.46 (from 12 months out)**.
  - The portfolios' correlation with the March−April short is only +0.05 to +0.20, but that low correlation is noise, not diversification. **The tradable premium is all the widowmaker.**

## 2. Method
- **Spreads.**
  - The 12 adjacent pairs (near minus far: Jan−Feb … Dec−Jan) plus Oct−Jan.
  - Two strip spreads, each the Nov–Mar average minus an Apr–Oct average:
    - **Win−nextSum:** winter minus the *following* summer (the winter premium).
    - **Win−prevSum:** winter minus the *preceding* summer (the storage spread).
- **Drift.** D(L) = V(t) − V(T).
  - T is the expiry of the near (first) leg, and L = round((T − t)/30.44 days), for L = 1…12.
  - For pairs, V(T) is the near leg's final settle minus the far leg's settle on that day.
  - For strips, V(T) is the hold-to-expiry payoff, i.e. each leg's own final settle.
  - D > 0 means the spread falls into expiry, so a short earns it.
  - Units are $/MMBtu and % of the average leg price at t.
  - Drift is averaged over the Fridays in each (spread, year, L) bucket, giving one observation per year per cell: 1,981 cells from 8,386 pair-Fridays.
- **Inference.** 95% CIs come from a gas-year-block bootstrap with 2,000 draws.
- **Storage state at t.**
  - The latest *public* EIA level (week ending F counts from F+6) minus its calendar-aligned 5-year average (prior years only, at least 3), so it is available from 2013.
  - The 5-year-average path gives projected end-March = avg5(31 Mar) + deviation and projected end-October likewise. Nothing realised is used.
  - **Scarce:** projected end-March < 1,500 Bcf. **Full:** projected end-October > 3,950 Bcf (near the 4.1–4.3 Tcf capacity).
- **Other regressors.** Regressions add lead fixed effects, the front-April price level and log trailing 13-week realised volatility of weekly spread changes, with gas-year cluster bootstrap CIs.
- **Portfolio.** Selection uses 2010–15 only; 2016–21 is the check. The universe is the 13 two-leg spreads.
  - **A (monthly cells):** each first Friday, hold every spread whose (pair, L) monthly step had a 2010–15 bootstrap |t| ≥ 2, in the sign of the mean step.
  - **B (hold to expiry):** for each pair, the entry lead L* = argmax |t| of cumulative drift on 2010–15 (requires |t| ≥ 2), held to expiry with size fixed at entry.
  - **Sizing:** equal ex-ante risk of $1,000 per spread per month, from the trailing 26-week sd of weekly changes (floored at 0.5¢).
  - **Costs:** 4 ticks × $10 per contract per side on every position change, 2 contracts per spread.
  - **Benchmarks:** short March−April from L = 4 (November) or from L = 12, held to expiry.

## 3. Results
**The map** (Fig. 1, `figures/premium_map_heatmap.png`). Of 180 cells, 42 have a CI excluding 0 in $ (44 in %). Biggest cells per pair (CIs in brackets; LOO = range of the mean when each year is left out):

| Cell | Mean drift | % of price | Years > 0 | LOO range | 2010–15 expiries | 2016–21 expiries |
|---|---|---|---|---|---|---|
| Mar−Apr, L9 | **+25.9¢** [+15.6, +34.5] | +7.7% [+4.6, +10.3] | 10/11 | +23.4 to +29.6 | +18.9 [+2.0, +34.3] | +31.7 [+24.7, +39.8] |
| Dec−Jan, L2 | −3.9¢ [−5.7, −1.7] | −1.2% [−1.7, −0.5] | 2/12 | −4.7 to −3.4 | −1.6 [−4.0, +1.3] | −6.1 [−7.6, −4.9] |
| Apr−May, L11 | +5.3¢ [+2.4, +7.9] | +1.8% [+0.8, +2.6] | 8/10 | +4.8 to +6.3 | +2.4 [−3.3, +8.2] | +7.2 [+5.8, +8.6] |
| Jul−Aug, L2 | −3.0¢ [−5.1, −0.8] | −1.2% [−2.0, −0.5] | 2/11 | −3.7 to −2.2 | −1.4 [−4.3, +1.9] | −4.3 [−7.0, −1.9] |

- **Year by year, Mar−Apr L9 in ¢:**

  | Year | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | Drift (¢) | +31 | +31 | +5 | **−12** | +40 | +24 | +38 | +51 | +31 | +24 | +23 |

  The full year tables for all four cells are in `top_cells_by_year.csv`.
- **Units.** $ and % give the same picture: the same cells are significant, with near-identical t.
- **Smaller cells.** The three smaller cells are significant only because of 2016–21; in 2010–15 their CIs span 0.

**Stability** (Fig. 2, `figures/premium_map_stability.png`).
- Across 180 cells, the 2010–15 and 2016–21 means correlate at r = −0.31 (Spearman −0.25), and −0.44 without March−April and the strips; in % units the figures are −0.34 and −0.42.
- Of the 17 cells with |t| ≥ 2 early, 12% keep their sign late.
- Cells within a pair share one price path, so the effective sample is about 15 spreads, not 180.

**Conditioning on storage** (2013+, % of price; scarce / normal / full):

| Family | Scarce | Normal | Full | Slope per +100 Bcf deviation | Realised-vol slope per sd | April-price slope per sd |
|---|---|---|---|---|---|---|
| Mar−Apr L3–12 | +9.7 [+8.1, +11.0] | +3.9 [−2.0, +8.9] | +10.4 [+4.6, +14.0] | +0.09 [−0.73, +1.18] | **+4.7 [+1.9, +6.3]** | −4.3 [−6.6, +0.3] |
| Win−nextSum | −7.0 [−19.6, +6.2] | +8.0 | +13.9 [−1.0, +36.1] | **+2.2 [+0.1, +6.2]** | +3.4 [−6.8, +10.6] | −4.1 |
| Win−prevSum | +14.8 [−3.3, +20.8] | −2.2 | −1.8 | −2.0 [−4.7, +0.6] | −0.4 | +13.0 [−2.4, +21.8] |
| Apr−May L3–12 | +1.9 | +0.9 | +2.8 | +0.08 [−0.07, +0.37] | +1.0 | −0.9 |
| Dec−Jan | −1.0 | −1.4 | −1.5 | −0.07 [−0.24, +0.15] | +0.1 | +0.8 [+0.2, +1.3] |
| Oct−Nov L6–12 | −0.8 | +6.1 | +0.5 | +0.3 [−0.5, +2.0] | +3.8 | −4.1 |

The answer is **neither scarcity insurance nor space rent**.
- The March−April premium is U-shaped in storage (with only 4–5 years per state) and scales with recent volatility.
- The autumn pairs show no excess space rent when full.
- The only storage slope whose CI excludes 0 is that Win−nextSum falls *more* when storage is loose, i.e. winter is overpriced most when it is least needed. That rests on 8 years.

**Portfolios** (Fig. 3, `figures/premium_map_portfolio.png`):

| Strategy (4 ticks/side) | 2016–21 net Sharpe [CI] | Gross Sharpe | Net total | Worst year | Worst month | Years > 0 |
|---|---|---|---|---|---|---|
| A, 17 cells, monthly | **−1.63** [−2.21, −1.15] | 0.59 | −$76.7k (costs $97k) | −$19.3k (2016) | −$9.1k (2019-08) | 0/6 |
| B, 7 pairs, hold to expiry | **−0.79** [−1.36, +0.18] | −0.46 | −$93.8k | −$51.6k (2020) | −$21.6k (2018-01) | 1/6 |
| Short Mar−Apr from Nov (L4) | +0.26 [+0.08, +1.41] | 0.28 | +$11.8k | −$0.7k (2021) | **−$12.4k (2018-11)** | 5/6 |
| Short Mar−Apr from L12 | +0.46 [+0.13, +1.30] | 0.49 | +$26.5k | −$2.3k (2021) | −$13.6k (2018-11) | 5/6 |

- **In-sample (2010–15) net Sharpes:** A −0.87, B +0.46, Mar−Apr +0.45 / +0.34.
- **Correlation of monthly net P&L with the short Mar−Apr (from L12):** A +0.21, B +0.08; within 2016–21, +0.20 and +0.07.
- **Robustness.**
  - A at |t| ≥ 1.5 (38 cells) scores −1.73, and at |t| ≥ 3 (2 cells) −0.49.
  - At 0, 1 and 2 ticks, A scores 0.59, −0.11 and −0.74, and B −0.46, −0.54 and −0.63.
- **Selection misses.** March−April had t = 1.97 at L11 in-sample on 4 years, so the |t| ≥ 2 rule *excluded* the one real premium from B. A held it only at L7–8.

## 4. What this tells us
**(a) The futures curve.**
- The curve carries one large, persistent risk premium, at the end-of-winter step.
- It is priced about 12 months out (about 25¢ at L12), but it is *earned* late. Only about 6¢ of it accrues between L12 and L3. About 15¢ goes between L3 and L1 (December–January, the weather-risk window), and the last 4¢ in the final month.
- The weaker neighbouring effects say the same thing. Jan cheapens against Dec, and Apr cheapens against May: deep-winter and end-of-winter legs are priced rich, and the premium leaks a few cents into the adjacent pairs.
- The rest of the curve, including the summer pairs, the autumn contango and the Nov−Dec pair, is fairly priced to within the noise of 11 years, and its small apparent premia flip sign between halves.

**(b) Storage expectations.**
- The premium is not a fee for a known physical shortage. It is the same size whether the 5-year-average path projects a tight or a full end of March.
- It moves with realised volatility, and with lower gas prices in % terms. That looks like an insurance price set by recent turbulence and by the risk capacity of the people selling insurance, not by expected storage.
- This agrees with v2: the market forecasts storage efficiently, and what remains is the premium.

## 5. Caveats
- **Few years.** 11–12 per cell, and 8–9 with storage conditioning. 2014 is the only disaster year in the sample, so the tail is under-sampled: the 2018-11 loss was 12× the monthly risk budget.
- **Variants tried.**
  - The map has 15 spreads × 12 leads × 2 units, which is descriptive.
  - Conditioning covered 8 families × (3 states + 1 slope + 3 multivariate).
  - There were 4 portfolio selection rules (A at three thresholds, and B), × 4 cost levels, plus 2 benchmarks fixed in advance.
  - The map was viewed *after* the rules were coded but *before* the results were seen. The L4 and L12 benchmarks were declared before the portfolio was run.
- **Sizing.** B fixes its size at entry. Long holds then carry winter volatility far above their entry sd (the 2018-01 and 2020 losses), which overstates B's tail relative to a rebalanced book.
- **Overlapping cells.** Cells overlap: cumulative drifts share paths across L, and the strips span up to 18 months.
- **Execution and trading cutoff.** Execution is at the settle. Positions with a near-expiry leg are assumed closed at the final settle.
- **Licensing.** No raw prices are reported.

## 6. Proposed holdout test (frozen)
- **Inputs.** CME settles for the Mar/Apr 2023, 2024, 2025 and 2026 contract pairs (4 winters). Mar/Apr 2022 is reported separately, because its entry falls in 2021.
- **Entry.** The settle on the last trading day on or before (March-contract expiry − 9 calendar months).
- **Target.** D = (Mar − Apr at entry) − (Mar final settle − Apr settle on the March expiry day), in $/MMBtu and in % of the average leg price at entry.
- **Trade and statistic.** Short 1 Mar−Apr spread at entry and hold it to expiry. Net P&L = D × 10,000 − $160 (4 ticks per contract per side, 2 contracts, round trip). Report each year and the mean.
- **Pass** requires **mean net P&L > 0 and D > 0 in at least 3 of 4 winters**. Otherwise **fail**.
- **Expected outcome.** Pass, unless a 2014-type winter occurs. The holdout is re-used (v2), so treat it as informative, not pristine.
- **Live check.** The Mar/Apr 2027 pair entered L ≤ 9 in mid-2026, so the cleanest fresh check is live tracking of the Mar−Apr 2027 drift to expiry, frozen now.

## 7. How to rerun
`python -m src.research.premium_map.run_all` from the repo root takes about 15 seconds on 1 CPU.
- **Tables:** `data/processed/research_premium_map/`, with all printed output in `results.txt` and CSVs per section.
- **Figures:** `reports/agents/figures/premium_map_{heatmap,stability,portfolio}.png`.
- **Modules:** `data.py`, `spreads.py`, `stats.py`, `portfolio.py`, `run_all.py`.

## 8. New avenues
1. **Price the premium against volatility.** Regress the March−April drift at L on trailing realised volatility (or on options-implied volatility, if obtainable), treating the spread like a variance-insurance seller's book. Expected value: if the premium per unit of vol is constant, it gives a sizing rule that cuts the 2014/2018-style tail without giving up the carry.
2. **Map the decay profile per unit of risk.** About 60% of the March−April drift (about 15¢) accrues between L3 and L1, exactly the December–January tail window, and only about 6¢ accrues between L12 and L3. Compare the Sharpe of the L12→L3 leg with the L3→expiry leg. Expected value: this shows whether the premium is pure compensation for winter tail risk (similar Sharpe in both legs) or whether there is a cheaper, calmer leg to harvest.
3. **Extend the map back to the pre-2010 curve** (EIA C1–C4 and any older settles) to add about 10 winters, including 2003, 2006 and 2008. Expected value: the main limitation here is 11 years with one disaster, so this is the cheapest way to size the tail properly.
4. **Win−nextSum overpriced when storage is loose.** Re-test with Model 2's winter percentiles, i.e. the physical probability of a tight end-March, instead of the 5-year path. Expected value: the only storage-dependent premium found (8 years, borderline CI); worth one focused check before discarding it.
