# v5_greeks_signals: do option-market components carry information the futures price does not?

*Agent `greeks_signals`, study v5. Development data only: options from `data/processed/v5/`, futures and the v4 panel filtered to dates up to 31 Dec 2021, and raw option files filtered to 2021 on load. No locked folder was opened. Choose window 2010–2016, check window 2017–2021. Every fitted quantity is walk-forward: expanding windows refitted monthly using only earlier data.*

## 1. Plain-English findings
- **Overall, the answer is mostly no.** I split option moves into Greek P&L, surface factors and flows, and scored about 650 tests. Of 464 tradable rules, **none** survives Holm on the choose window, and none of the 13 rules that were nominally significant (p < 0.05) earns money in the check window after costs. What the option market "knows" on a given day, the futures already priced.
- **One weak, episodic exception: implied-vol jumps that the futures move does not explain predict a next-day giveback in futures.** After a jump in implied vol across the three nearest expiries (the vega P&L), the front future falls the next day: −0.22% per s.d. of the signal, t −3.2 over 2010–21 (check window −2.0). The attribution residual mirrors it: +0.25% per s.d., t +3.6 (check +1.7). **Both rest on two episodes, Jan–Feb 2014 and Nov 2018.** In other years the yearly t-stats sit between −1.1 and +1.2. Traded at |z| > 1 with a one-day hold, the residual rule nets about +$1k per year per front contract in the check window (t 0.4). That is real information, but not a tradable edge.
- **Call skew follows futures; it does not lead them.** The change in skew correlates +0.13 with *yesterday's* futures return and +0.03 with *tomorrow's*. Across 14 up-squeezes (5-day moves of +15% or more), skew was not elevated beforehand: the mean z was 0.6 at day −5, 0.6 at day 0 and 1.3 at day +5. Skew was already high going into Jan 2014, but it rose with the futures from 13 Jan onward. In Nov 2018 it rose only after the +8% move on 5 Nov. Forecast news adds at most 0.5 percentage points of R² to skew changes beyond the futures move.
- **Flows: no damping and no pinning.** A high open-interest gamma concentration in the final week does not damp realised vol. If anything the sign runs the other way, and it disappears once implied-vol level is controlled for. Settles land within 2.5¢ of the strike with the most open interest at 11 of 136 expiries, about the rate expected by chance (placebo 5–7%, p = 0.35). Put/call ratios and call-volume spikes predict nothing.
- **Storage-report vol: weekly options are rich, but not because of the report.** Selling the 1–2-day weekly straddle across the EIA Thursday earned **+$225 per straddle net** in 2014–16 (n = 237, CI [119, 296], wins 70% of trades). The same trade on spans with no report earned +$201 (n = 30) in 2014–16 and +$401 (n = 36) in 2017–21. One-day implied variance is the same with or without a report (medians 0.00148 vs 0.00142). The premium is general short-dated richness, or a settlement-mark artefact (Section 5), not overpriced storage risk.

## 2. Method
**Intuition.** If option traders know something futures traders do not, it should show up as option repricing that the same-day futures move cannot explain, and that repricing should lead later futures moves, realised vol, forecast news or storage surprises. Every candidate is therefore measured *net of the futures path* and scored as a trade.

**Greek attribution** (`dh_straddle.csv`). For each of the 3 nearest ON expiries k, the vol shock is
vshock_k = (iv1 − iv0) − β̂_k · ln(F1/F0), where β̂ is walk-forward OLS. Aggregates are:
- `vshock123`, the mean of vshock over the three expiries
- `vega_pnl123` = Σ vega0 (iv1 − iv0)
- `resid_rel123` = Σ resid / Σ P0, the attribution residual as a share of premium
- `gt_rel123` = Σ (gamma_pnl + theta_pnl) / Σ P0, realised minus implied variance. This is a futures-only quantity, used as a benchmark.

**Surface.** All changes are same-contract, with the rank fixed at t−1. The factors are:
- level: Δatm1
- term: Δ(atm1 − atm3)
- front skew Δskew1 and second-month skew Δskew2
- curvature Δcurv1
- the same five, residualised on the same-day front return
- levels of skew, term and curvature
- a walk-forward PCA (monthly refit) on the 6 ATM changes plus the 4 front wings. PC1 is a parallel shift, PC2 is front versus back, PC3 is the wings.

**Flows.** These are lagged one day, because CME open interest and final volume for day t are published on t+1. They are log(put OI / call OI) and log(put volume / call volume) for the two nearest expiries, their 5-day changes, and call and put volume spikes (log of volume over its prior 20-day mean).

**Scoring.** Each signal is converted to a walk-forward z. The direction is set by the sign of the choose-window slope and then frozen.
- **Positions:** sign(z), or sign(z) only when |z| > 1 ("thr1"), held 1 or 5 days as an overlapping portfolio.
- **Instruments:** front future, second future, H−J spread, and the front delta-hedged straddle.
- **Costs:** $20 per contract per side for futures; doubled for the spread, with roll costs. For the straddle, 1.5% of premium per side plus futures cost on each daily re-hedge.
- **Benchmarks:** zero; the same rule driven by the 1-day or 5-day futures return; always short the straddle; always long the future.
- **Information test:** the forward target is regressed on z, z(r_front_t) and z(r_front over t−4..t). Straddle targets add atm1 and past realised variance.
- **Statistics:** Newey-West errors with lags = max(5, h), and a 21-day block bootstrap for P&L CIs.
- **Multiple testing:** Holm across all option variants on the choose window, followed by the check window.

**Expiry gamma.** At A = 5 trading days before option expiry E, I record gex (unsigned open-interest-weighted gamma notional) and its walk-forward percentile among earlier contracts. "Concentration" is gex / (OI × ATM gamma × F²/100). The outcomes are:
- realised/implied vol ratio over (A, E]
- a pin hit, |F_E − K\*| ≤ 2.5¢, where K\* is the strike with the most open interest at A, compared with the same rate for K\* ± 10¢ and ± 15¢
- the short delta-hedged straddle from A to E−2.

**EIA event variance** (`weekly.csv`). Implied total variance is V = iv² · n/252 over n trading days containing n_E report days. Report day R is week_ending + 6 days, or the previous trading day after a holiday. I use two methods:
- **(A) direct:** n = 1 rows spanning a report versus not.
- **(B) per-date cross-section:** V_i = a(n_i − n_E,i) + b · n_E,i, giving an event excess of b − a.

Realised variance is the squared daily change of the matched underlying (the parity F matched to the curve to within 2¢). The trade sells, at the settle before R, the shortest weekly ATM straddle (n ≤ 2) whose life spans R, at the strike nearest the parity F, and holds it to expiry. Cost is 3% of premium, plus $0.002 if the option is exercised.

## 3. Results
**Variant count and multiple testing**

| Set | Variants | Raw p < 0.05 (choose) | Holm < 0.05 | Replicate in check |
|---|---|---|---|---|
| Tradable rules (29 signals × 8 targets × 2 rules), net P&L > 0 | 464 | 13 | **0** | **0** (one-sided p < 0.05) |
| Information (incremental slope, two-sided) | 232 | 34 | 6 (all Δlevel/Δterm → next-day front/second) | 0 of the 6. Of the 34, 4 replicate (vega_pnl123 and pc1 → front/second): one phenomenon |
| Descriptive (Part 1 robustness, storage, tails, expiry, EIA) | ~190 | — | — | — |

The Bonferroni bar for the tradable set is t > 3.70; the best tradable choose-window t is 2.35. The rank correlation of choose-window and check-window gross-P&L t-stats is 0.12, close to noise. For the information slopes it is 0.20.

**Part 1: what the vega shock predicts.** Controls throughout are today's and the 5-day front return.

| Target | vega_pnl123: choose t | check t | resid_rel123: choose t | check t |
|---|---|---|---|---|
| Front return t+1 (raw) | −3.14 | −2.01 | +3.65 | +1.69 |
| Front t+1, +\|r\| and r² controls | −3.17 | −1.94 | +3.57 | +1.84 |
| Front t+1, winsorised 1% | −2.50 | −0.86 | +3.26 | +1.75 |
| Second-month return t+1 | −2.91 | −1.94 | +3.28 | +1.42 |
| H−J change t+1 | −1.65 | −1.55 | +2.19 | +1.20 |
| Next-day forecast news (news_tot, Oct–Apr) | −0.42 | −1.63 | +0.34 | +1.04 |
| Next storage surprise (2015–16 / 2017–21) | +1.67 | +0.81 | −0.22 | −0.89 |
| Realised variance, next 5 days (+ iv², past RV) | −2.82 | +1.20 | +2.04 | −0.83 |
| Delta-hedged straddle P&L, t+1 | −2.53 | +1.95 | +1.86 | −1.22 |

Other aggregations (rank-1 vshock, the mean over three expiries, the 5-day sum) behave the same or weaker. The 5-day sum is null everywhere. Vega shocks do not anticipate weather revisions or storage numbers. They show mean reversion in vol on 2010–16 and momentum on 2017–21, so there is no stable vol signal. `resid_rel123` correlates −0.50 with Δatm1 and −0.38 with vega_pnl123, so the two next-day effects are largely one phenomenon: large implied-vol jumps are followed by a futures giveback.

Figure: `figures/v5_greeks_signals_attribution.png`. The cumulative net P&L of the |z| > 1 rules is flat apart from steps in Feb 2014 and Nov 2018. The yearly t-stats are large only in 2014, 2015 and 2018.

**Tradable view (net $/yr per contract, 21-day block-bootstrap CI)**

| Rule | 2010–16 | 2017–21 |
|---|---|---|
| resid_rel123 → front, \|z\|>1, 1 day | +7,587 [295, 16,813] | +1,079 [−3,098, 7,050] |
| vega_pnl123 → front, \|z\|>1, 1 day | +6,978 [−634, 16,629] | +1,892 [−6,658, 11,983] |
| pc1 → front, \|z\|>1, 1 day | +2,573 [−1,541, 7,686] | −424 [−5,048, 6,554] |
| skew2 level → front, \|z\|>1, 5 days | +2,905 [−3,217, 9,418] | −4,499 [−14,639, 6,907] |
| call_spike → front, \|z\|>1, 5 days | +225 [−1,608, 2,180] | −1,359 [−5,071, 1,868] |
| vshock123 → straddle, \|z\|>1, 1 day | −1,727 [−3,029, −240] | −8,694 [−13,616, −4,456] |
| Benchmark: futures-return rule → front, \|z\|>1 | +1,296 [−2,985, 5,400] | −1,315 [−8,804, 4,416] |
| Benchmark: always short front ATM straddle, delta-hedged | −2,999 (gross −786) | −3,146 (gross −865) |

**Part 2: surface factors.** On Oct–Apr rows, the same-day futures move explains:
- 8% of ATM level changes
- 3% of front-skew changes
- 5.5% of second-month-skew changes
- 12% of PC3.

Adding GEFS news raises R² by at most 0.5 percentage points. Δskew2 does load on news (t 2.3), but the effect is economically negligible. Skew therefore carries little independent news, and what it does carry does not lead futures. Lead-lag correlations are in `figures/v5_greeks_signals_skew_events.png`. In ex-ante tail tests, the probability of a next-5-day move of +10% or more is 3.8% in the top skew2 tercile versus 3.0% in the bottom on 2010–16 (difference CI [−3.5, +5.1] percentage points), and 7.2% versus 5.2% on 2017–21 (CI [−4.6, +7.8]). Skew1, its 5-day change and call spikes are all null. Only the vol *level* predicts big moves (5.9% vs 0.0%; 9.6% vs 0.7%), and it is non-directional.

**Episodes (rest on single events).**
- **Jan 2014:** front-skew z was already 2–3 from late December. It jumped to 5 on 13–16 Jan alongside +5% and +2% futures days, then to 18 on 24 Jan, the day of the spike itself.
- **Nov 2018:** second-month skew z drifted from 0.1 in mid-October to 0.6 on 31 Oct, and rose to 1.3–1.5 only after the +8% move on 5 Nov.
- **Feb 2021:** front skew rose from z 0.3 to 2.3 between 21 Jan and 5 Feb while futures rallied.

In every case skew moved with or after the futures rather than before them.

**Part 3: expiry gamma** (136 monthly expiries, 2010–21)

| Window | Group | n | Median RV/IV, final week | Pin hit | Placebo ±10¢ | Short-straddle net $ |
|---|---|---|---|---|---|---|
| 2010–16 | gex top / bottom tercile | 26 / 24 | 0.96 / 0.76 | 4% / 13% | 2% / 8% | −160 / −21 |
| 2017–21 | gex top / bottom tercile | 7 / 38 | 1.06 / 0.84 | 29% / 5% | 14% / 7% | −189 / −9 |
| All | concentration top / bottom | 52 / 40 | 0.82 / 0.86 | 8% / 8% | 8% / 5% | −11 / −102 |

The slope of log(RV/IV) on the gex percentile is +0.29 (t 2.1) on 2010–16 and +0.37 (t 2.4) on 2017–21. That is the opposite of damping. With log implied vol as a control, it falls to t 0.8 on 2017–21. Gex is ∝ OI·F/(σ√T), so it partly measures low implied vol, and realised vol over-runs low implied vol. Concentration is null (t −0.9 and −0.1). Overall the pin rate is 8.1% against 7.0% and 4.8% for the placebos (binomial p 0.35).

**Part 4: EIA event vol** (`figures/v5_greeks_signals_eia_event.png`)

| | 2014 | 2015 | 2016 |
|---|---|---|---|
| 1-day implied variance spanning report / realised r² on those days | 0.00104 / 0.00048 | 0.00185 / 0.00078 | 0.00214 / 0.00082 |
| 1-day implied variance, no report / realised (n = 10, 10, 5) | 0.00088 / 0.00075 | 0.00148 / 0.00063 | 0.00170 / 0.00108 |

Implied vol over-states realised by a factor of 1.5–1.6 on report days, and by a similar ratio on the few non-report days. In 2017–21 there are no one-day rows spanning a report. Realised report-day variance is only modestly elevated: the median r² on Thursdays is 1.4 times the Tue/Wed/Fri median. The mean excess over all non-report days is about zero (2014–16: −0.000007, CI [−0.00023, 0.00021]), because Monday weekend variance inflates the baseline.

Cross-section method (B) gives a median event excess b − a of 0.0011. However, it confounds the share of a span taken up by reports with tenor, because short weeklies are richer, so I do not rely on it.

The straddle trade:
- **Report spans:** +$225 net per straddle in 2014–16 (n = 237, CI [119, 296], t 4.7); +$34 in 2017–21 (n = 13, CI [−490, 152]).
- **No-report spans:** +$201 (n = 30) and +$401 (n = 36).
- **Same-day short of the delta-hedged *monthly* front straddle:** −$75 net. The monthly options carry no report premium.

## 4. What it means for a gas volatility desk
- Do not expect the vol surface, put/call positioning or gamma to tell you *where* futures go. Skew is a thermometer, not a barometer, and the 2014 and Nov 2018 squeezes were not visible in the surface beforehand.
- A sharp implied-vol jump not explained by the futures move has tended to precede a modest giveback in futures the next day. Treat this as a risk-management observation about panic days, not a strategy: it comes from 2–3 episodes and is worth about $1–2k per contract per year after costs out of sample.
- The front monthly ATM straddle, delta-hedged daily, has no vol premium to harvest gross (about −$0.8k per year short), which is consistent with v4's finding that outright vol is fairly priced. That is good news for a long-vol book: owning gamma is not systematically expensive.
- The only sizeable premium is in 1–2-day weekly options, and it is not specific to the EIA report. Before anyone sells it, verify it against real bid/offer quotes (Section 5).

## 5. Caveats
- **Single episodes:** the next-day vol-shock and residual effects are 2014, 2015 and Nov 2018. Squeeze descriptions are n = 1 each.
- **Option settles** are exchange marks, not trades. For the weekly trade, quotes exist for only 10 of 316 trades. In 9 of those 10, the settle is above the session's lowest offer, so the weekly premium may partly be a marking artefact. A 3% cost on a straddle worth about $0.10 may also be optimistic.
- **Thin post-2016 coverage:** the weekly check window has only 13 report trades, and 2017–21 has only 7 top-tercile gex expiries.
- **Data details:** open-interest dates before 2016 are inferred. Flows are lagged one day to stay conservative. Release days after holidays are approximated. Strike-level open interest is unsigned and the dealer side is unknown, so gex cannot distinguish long from short gamma.
- **In-sample direction choice:** the trading direction is chosen on 2010–16. The check-window numbers are the honest ones.

## 6. Frozen tests for the locked 2022–2026 sample
Code: `src/research/v5_greeks_signals/frozen_test.py --locked`. Dev run: `--dev`, with reference values in `data/processed/research_v5_greeks_signals/frozen_dev_reference.json`. The test reads `data/processed/v5_locked/{dh_straddle,weekly}.csv`, `data/raw/ng_curve_daily.csv`, the raw weekly options file and `data/raw/storage.csv`. Walk-forward z-scores use history from 2010, and the test rows run from 1 Jan 2022 onward. Run once.

| ID | Inputs / statistic | Pass rule | Dev reference (all / choose / check) |
|---|---|---|---|
| **P1 (primary)** | r_front(t+1) ~ z(resid_rel123) + z(r_front_t) + z(r_front over t−4..t), HAC with 5 lags. Raw, and winsorised (z clipped at ±2.33, y at the 1st/99th percentiles) | t ≥ +1.645 in **both** | raw t 3.59 / 3.63 / 1.69; winsorised 3.62 / 3.30 / 1.78 (passes in all three) |
| S1 | Same regression with z(vega_pnl123) | t ≤ −1.645 in both | raw −3.24 / −3.27 / −2.01; winsorised −2.28 / −2.59 / −0.84 (fails check) |
| S2 | Sell the shortest weekly ATM straddle (≤ 2 trading days) spanning each EIA report, at the prior settle; 3% cost | Mean net > 0, n ≥ 10, and 4-trade block-bootstrap 2.5% bound > 0 | +$215 [120, 301], n = 250 / +$225 / +$34 [−490, 152], n = 13 (fails check) |

P1 and S1 are correlated (ρ = −0.38) and should not be counted as independent confirmation. Log the no-report weekly trade alongside S2: if S2 passes but its report-minus-none difference is ≤ 0, the reading is "weekly richness", not "storage premium". No live 2026-27 version is proposed, because a one-day-horizon test needs a full season of daily option settles.

## 7. How to rerun
All commands run from the repo root, `python -W ignore -m src.research.v5_greeks_signals.<module>`. Each step takes under 1 minute. The cache is `data/processed/research_v5_greeks_signals/`.
1. `features`: builds the daily panel (`features.pkl`).
2. `score`: tradable and information scoring, Holm (`scores.csv`, `pnls.pkl`).
3. `part1_greeks`, `part2_surface`, `part3_flows`, `part4_weekly`: Parts 1–4 and two of the figures.
4. `summary`: key tables, variant counts and the attribution figure.
5. `frozen_test --dev`: dev reference values for Section 6.

## 8. New avenues
- **Intraday timing:** check whether the residual and vega-shock effect is a settlement-window artefact, where the options' 14:30 mark lags the futures close, by using futures TAS or 14:30–14:35 bars.
- **Signed dealer gamma:** use CME large-trader or block data, if licensed, to sign dealer gamma. Unsigned gex cannot test the pinning mechanism.
- **Weekly premium against real quotes:** test it with Databento MBP quotes for weeklies. If it holds, split it by day of week and by whether the expiry is the Friday after the report.
- **Calendar-spread surface:** extend the surface-factor tests to the CSO surface (`spreads.csv`, 2014–16), to see whether implied correlation, rather than outright skew, leads H−J moves.
