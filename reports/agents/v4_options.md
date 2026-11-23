# v4_options: what options charge for the March−April path risk

*Agent `v4_options`, 23 Nov 2026. Development data only (dates ≤ 2021-12-31). G4X March options exist from winter 2012-13. 2012-13 has 9 strikes a day and 2013-14 only **2 strikes** (0.25 / 0.50): both are flagged "thin". Winter 2021-22 is cut at 31 Dec 2021, so it has no expiry. Choose window = 2012-13..2016-17; check window = 2017-18..2021-22.*

## 1. Plain-English findings
- **The spread option market charges a real volatility premium, and the outright market does not.** In the trade season (early Nov to the March option expiry), H−J implied normal vol exceeded the vol realised over the option's remaining life in **8 of 9 winters**. Outside the two thin winters the gap is **+$0.22/MMBtu/√yr [CI 0.11, 0.33]**, so implied is about 1.5× realised. 2013-14 wiped this out: realised was 3.3 against 2.1 implied. ATM vols on the H and J outrights are fair (gap −0.01 and +0.00 over all winters).
- **Implied vol does not warn of the disasters.** Before the Jan 2014, Feb 2014 and Nov 2018 blow-ups, H−J implied vol rose 0.06–0.2 vol points (Feb 2014 is an exception, after January's spike). Over the 5 days after the blow-up began it rose 2–3.7 points. Implied vol is essentially the spread level in disguise (weekly correlation 0.98). Nothing in the options (vol level, implied-minus-realised gap, call skew) predicted the 20-day adverse excursion out of sample.
- **Buying the protection works and costs about what the premium earns.** A G4X March call at ATM+$0.25, bought at entry, cut the worst mark-to-market drawdown from **−$13,230 to −$2,660 (−80%)** and kept **28%** of the mean P&L, so kept + cut = 1.08. This was chosen on 2012-17 and held on the check window (1.08). Half size gives 1.00 by construction. The edge over scaling rests on 2 episodes (2014, 2018), with bootstrap P(kept + cut > 1) = 0.64. It is not significant.
- **The insurance is expensive at expiry and roughly fair for the path.** At entry the options imply about an 8% chance that H−J ends more than $0.50 above its November level. That happened in 0 of 11 dev winters. Through the reflection principle, the same prices imply a ~15% chance of *touching* +$0.50 before expiry. Physically it happened in 2 of 9 winters (22%). Held to expiry, calls at ATM+$0.25 returned 15¢ per $1 of premium.
- **The news/premium split adds nothing here.** Implied-vol jumps coincide with the premium part of d_hj (correlation 0.91, against 0.11 for news), but only because the news part is tiny (sd 0.005 vs 0.059 $/day). Per dollar of move, implied vol reacts the same to both.

## 2. Method
- **Data.** G4X March options (cm = 3) are options on H−J. Their settles pass put-call parity against the futures spread (median |error| $0.000, 95th percentile $0.002). Junk strikes more than $3 from F are dropped. T = calendar days to the option expiry (one trading day before the futures last trade) / 365.25, with r = 0.
- **Panel check.** My own `nv_hj` (Bachelier smile at K = F, 1,527 days) matches the panel: median difference 0.000, 90th percentile < 0.007 in 2013-2020. My own `iv_h` (300 random days) differs by a median 0.1%. Stale flat settles more than 11 months out are excluded. The panel's values are used.
- **Variance risk premium.** gap(d) = σ_imp(d) − RV(d), where RV_life(d) = √(Σ_{t=d+1}^{expiry} ΔS_t² / T_d) (zero-mean, so it matches the Bachelier variance) and RV20 = √(252/20 · Σ next-20 ΔS²). H and J use log returns. Winter season means, then a winter-block bootstrap (4,000 draws). Descriptive, no fitting.
- **Smile for off-grid strikes.** Bachelier vols from out-of-the-money settles, linear in strike, flat beyond the quoted range; on days with no quotes the last smile is carried forward. Backtest marks: 1,801 exact, 1,015 interpolated, 24 carried.
- **Prediction.** Weekly points (Fridays from entry until 20 days before expiry). Targets: MAE20 = max S over the next 20 days minus S, ΔS20, and RV20. Predictors: nv, nv − trailing RV20, and skew = C(F+δ)/C(F) for δ = 0.25 and 0.50. Each model is an expanding-window OLS, refitted by winter, and compared with (a) the training mean and (b) a benchmark using the level S plus trailing RV.
- **Protection backtest.** Short 1 H−J at the settle of the first trading day on or after the first Friday on or after 1 Nov, held to the futures expiry settle. Costs are 2 ticks per leg per side ($40 per side), with 4 ticks as a sensitivity. Calls are bought at the entry settle × (1 + 4%), with a minimum half-spread of $0.002, and held to expiry with intrinsic payoff. K = F_entry + δ, rounded to the $0.05 grid. Marked daily. Kept = mean / naked mean; cut = 1 − worst drawdown / naked worst drawdown (v3); scaling gives kept + cut = 1.
- **Implied distribution.** P_Q(S_T > K) = [C(K−0.05) − C(K+0.05)] / 0.10, averaged over the first 5 trading days on or after entry. Model-implied path probability = 2 × P_Q (reflection principle, zero drift).

## 3. Results
**Table 1. Season gap, implied minus realised over remaining life (winter-mean basis, 95% CI)**

| Instrument | Choose (5 w) | Check (4 w) | All (9 w) | Ex-thin (7 w) | Winters with gap > 0 |
|---|---|---|---|---|---|
| H−J nv ($/√yr) | −0.07 [−0.64, 0.32] | +0.22 [0.07, 0.35] | +0.06 [−0.29, 0.28] | **+0.22 [0.11, 0.33]**, ratio 1.51 | 8/9 |
| H (Black) | −0.06 [−0.20, 0.03] | +0.06 [0.05, 0.08] | −0.01 [−0.10, 0.05] | +0.04 [0.01, 0.06] | 6/9 |
| J (Black) | −0.00 | +0.01 | +0.00 [−0.04, 0.04] | +0.01 | 6/9 |

- Against the fixed 20-day horizon, H−J gives +0.20 [0.01, 0.43] over all 10 winters.
- By time to expiry (life basis), the H−J gap is about +0.08 to +0.10 at 2–11 months and about −0.03 to −0.06 inside 2 months, where the expiry squeezes sit. No bucket's CI excludes 0.
- Figure: `figures/v4_options_vrp.png`.

**Table 2. Episodes** (t0 = start of the largest 5-day rise in each window). Changes are in vol points.

| Episode | 5-day rise ($/spread) | Δnv, t0−20 → t0−1 | Δnv, t0−1 → t0+5 | Δiv_h, before / after | Quote quality |
|---|---|---|---|---|---|
| Jan 2014 polar vortex | 6,220 | +0.06 | +2.13 | +0.02 / +0.46 | Mostly extrapolated (2 strikes) |
| Feb 2014 expiry squeeze | 9,450 | +0.99 (after January) | +3.71 | +0.13 / +0.68 | Mostly extrapolated |
| Nov 2018 squeeze | 12,210 | +0.22 | +3.08 | +0.08 / +0.48 | Quoted |
| Sep 2021 (before the trade's entry) | 5,470 | n/a | +2.01 | n/a | Quoted |

- The standard deviation of a 20-day nv change is 0.69, so the pre-move rises are 0.1–0.3 sd.
- Same-day corr(Δnv, ΔS) is 0.92. At leads of 1–5 days the correlation is between −0.09 and +0.16, so there is no directional lead.
- Figure: `figures/v4_options_episodes.png`.

**Table 3. Walk-forward out-of-sample R² for MAE20 (adverse excursion)**
- nv vs the training mean: −0.04 (choose) and −0.02 (check).
- Gap: −0.06 and −0.03.
- Skew25: −0.01 and −0.01. Skew50: −0.03 and −0.01.
- Over the level + trailing-RV benchmark, every increment is ≤ +0.08, with CIs spanning 0.
- nv does "predict" ΔS20 (out-of-sample R² +0.48 in both windows; high nv → more decay). This is the level effect: given S and trailing RV, it adds −0.12 (choose) and +0.13 (check).

**Table 4. Protection, winters 2012-13..2020-21 (P&L), with drawdowns including the truncated 2021-22.** $ per spread; CIs from the winter bootstrap.

| Variant | Mean P&L | Δ vs naked | Worst drawdown | Worst winter | Kept / cut | Kept + cut, choose → check |
|---|---|---|---|---|---|---|
| Naked | 1,383 [32, 2,620] | – | −13,230 | −3,060 | 1 / 0 | 1 → 1 |
| Half | 692 | – | −6,615 | −1,530 | 0.50 / 0.50 | 1 → 1 |
| **Call ATM+0.25** | 382 [−299, 935] | −1,001 [−1,808, −223] | **−2,660** | −1,790 | 0.28 / 0.80 | **1.14 → 1.08** |
| Call ATM+0.50 | 427 | −957 | −4,931 | −3,260 | 0.31 / 0.63 | 0.82 → 1.21 |
| Call spread +0.25/+1.00 | 956 | −428 | −12,001 | −1,804 | 0.69 / 0.09 | 1.31 → **0.67** |

- Selected by choose-window Calmar: call +0.25 (0.145 vs 0.138 for the call spread).
- Nov 2018 ran straight through the short +1.00 strike of the call spread.
- Kept + cut for call +0.25 at different option half-spreads:
  - 1.09 at 2%
  - 1.04 at 8%
  - 0.99 at 15%
  - 1.00 with 4-tick futures costs and an 8% half-spread
- Matching call +0.25's worst drawdown by scaling means holding 0.20 spreads, which keeps $278 against $382: about +$100 per winter.
- Figure: `figures/v4_options_protection.png`.

**Table 5. Insurance price at entry (mean of winters 2012-13..2020-21) vs outcomes**

| Event | P_Q at expiry | Physical at expiry | Implied path (2·P_Q) | Physical path hit |
|---|---|---|---|---|
| S_T > F+0.25 | 0.11 | 1/9 (1/11 over 2011-21) | 0.22 | 2/9 |
| S_T > F+0.50 | 0.08 | 0/9 (0/11) | 0.15 | 2/9 |
| S_T > F+1.00 | 0.05 | 0/9 | 0.11 | 2/9 |

- Call payoff per $1 of premium:
  - ATM: 0.29
  - +0.25: 0.15, net −$953 per winter [−1,750, −186]
  - +0.50 and +1.00: 0 (no winter paid)
- The 2021-22 entry (truncated) priced extreme tails: nv about 3.1, and the far wings gave slightly negative implied densities, i.e. stale settles. It is descriptive only.

## 4. What this tells us
The options market confirms v3's thesis in its own prices:
- **Expiry tails are overpriced.** The puts and calls embed upside spread risk that never materialised at expiry in 11 winters.
- **The path is priced about fairly.** The reflection-implied probability of touching +$0.50 is close to the ~2-in-9 frequency of touching it.
- Spread implied vol carries a ~50% markup over realised, while outright vols do not. The "widowmaker" premium therefore sits in the **spread** specifically, in both the futures (the 26¢ drift) and the options (the vol gap). It is not general gas fear.

Options cannot tell news from premium ex ante: implied vol moves with the spread on the day and does not lead it. They do offer the one overlay so far that sits (slightly) above the scaling line: an OTM call that turns the 2018-type squeeze into a −$1.7k drawdown. That is the case v3 said only being flat beforehand could handle.

## 5. Caveats
- **Variant count: about 95 cells.**
  - VRP: 3 instruments × 2 RV definitions × windows/buckets, all descriptive.
  - Prediction: 3 targets × 6 predictors × 2 benchmarks = 30.
  - Protection: 5 variants × 4 option-cost levels × 2 futures-cost levels = 40.
  - Implied distribution: 5 thresholds + 4 strikes.
  - Step 5: 4 subsamples.
- **The protection result rests on 2 episodes.** 2014 is marked with 2 strikes, using extrapolated vols beyond 0.50. 2018 is well quoted.
- Option costs are assumed, not observed, and G4X volume is thin in 2013-15.
- With 9 winters, physical tail frequencies have very wide CIs (2/9 → roughly 6–55%).
- The reflection principle assumes zero-drift Bachelier dynamics. The spread drifts down, so this is a rough bound.
- Step 5 uses `v4_news_channel`'s walk-forward decomposition as it existed at 13:19 on 23 Nov 2026.

## 6. Proposed frozen tests
**Live winter 2026-27 (options).** Inputs: G4X March 2027 settles, H27/J27 futures settles. Option expiry 23 Feb 2027 (futures last trade 24 Feb).
- **T-OPT1 (primary): spread VRP.**
  - Statistic: G = mean over trading days d from 1 Dec 2026 to 22 Jan 2027 of [nv_hj(d) − RV_life(d)]. nv_hj is computed as in `src/v4/panel.py`. RV_life = √(Σ_{t>d}^{23 Feb} ΔS_t² / T_d).
  - Pass if G > 0. Also report the ratio mean(nv) / mean(RV).
  - The start is set after the 1 Dec freeze so that no observed day enters.
- **T-OPT2 (descriptive): protected vs naked.**
  - Short H27−J27 from the 6 Nov 2026 entry settle, long a G4X March 2027 call at K = round₀.₀₅(F + 0.25) bought at that settle +4%, with costs as above.
  - Report: protection cost = naked − protected final P&L (dev: −$1.3k in 2014, when the call paid, to +$3.4k; mean $1.0k), and drawdown ratio = protected worst drawdown / naked worst drawdown.
  - The prediction applies only if the naked drawdown is ≤ −$4,000: ratio ≤ 0.35.
  - Caveat: 6 Nov lies before the freeze. Settles from 6–23 Nov exist but were not looked at by this agent.
- **T-OPT3 (log): implied tail.** Log P_Q(S_T > F+0.50) at the entry, plus whether +0.50 is touched and whether it holds at expiry.

**Locked 2000–2010 (no options).**
- **H-OPT-A.** On late-Nov-to-expiry H−J, the weekly spread level predicts ΔS20 (OLS slope < 0, HAC lags ≥ 4, one-sided p < 0.05) but not MAE20 (slope CI includes 0). This tests "vol = level, and it predicts decay, not drawdowns".
- **H-OPT-B (synthetic protection).** Price the ATM+0.25 call by Bachelier with the frozen map nv = 0.270 + 3.291·max(S, 0) (dev corr 0.94) and T from the calendar. Apply the T-OPT2 rule to each locked winter from its first H−J date. Pass if kept + cut > 1 against the naked short over the locked winters, at a 4% half-spread. This is conditional on the map: 2003 (15¢ → $1.74) will dominate.

## 7. How to rerun
`python -m src.research.v4_options.run` takes about 2 minutes on 1 CPU and writes `data/processed/research_v4_options/` (`summary.json`, `step*_*.csv`) and the three figures.

## 8. New avenues
- **Call strike/timing** (e.g. +0.35, or a December purchase). *Value: moderate*; selection noise ≈ effect with 9 winters.
- **Sell H−J vol (delta-hedged G4X strangles)** instead of or alongside the drift. *Value: moderate to high*; the +0.22 gap is the cleanest premium found, but it carries the same 2014 tail.
- **G3B/G6B options as cheaper protection.** *Value: low to moderate.*
- **2022–26 G4X (descriptive)** to add winters to Table 5. *Value: moderate*; the only way to tighten 2-in-9 before live.
