# Independent review of the four v2 agent reports (agent `reviewer`)

*Only data ≤ 2021-12-31 was read: the agents' cached outputs, each asserted < 2022 on load. Scratch code is in `src/research/reviewer/` (each check runs in under 1 minute).*

## 1. Leakage: **confirmed clean, minor caveats**
- **Cut-off.** All loaders cut at 2021-12-31 on load.
- **No full-sample normalisations.**
  - Seasonal shape and benchmark norms use the 3 prior years.
  - The kink threshold, PC1 loadings and Kalman/HMM parameters come from 2011–17 only.
  - All 5-year storage and HDD norms use prior years only.
- **Storage timing is right in all four:** storage is public at F+6, and prices are the settle after the report.
- **Regime and Kalman probabilities.** regimes_weather's `stress_pre` is the one-step predicted probability (data to t−1); I re-filtered from the frozen parameters and matched it to 3e-15. Kalman uses filtered output only.
- **implied_storage's mapping is genuinely walk-forward** (earlier winters only).
- **Minor:**
  - implied_storage's who-moves regression keeps a pair only if week t+h is also informative, which selects on the future spread (§5).
  - The 2021-22 March expiry is read from truncated data (regimes_weather), and curve_consistency gives Jan-22 `dte = 999`. Both are harmless.

## 2. curve_consistency, kink fade: **numbers confirmed; costs understated; claim overstated**
- **Independent recompute.** Threshold 423.96 bp (2011–17 q99). 2018–21: 251 trades, gross $2,326, net **$2,166** at the agent's cost. There are 5 episodes, 4 positive. 2011–17: 272 trades, +$677.
- **Arithmetic is correct:** P&L = −2·ΔB·10,000 per 1:−2:1 fly.
- **Costs are one side only.** The code charges 4 × ticks × $10 = $160 and calls it the round trip. Per contract per side, 4 ticks is $320:
  - OOS net **+$2,006**.
  - At 10 ticks: **+$1,526** (the report says +$1,926).
  - In-sample: +$517. Episodes positive fall to 9 of 11.
- **Units are consistent.** 424 bp is a log fly residual. Mean |fA|·F on signal days is about 24¢, and the gross P&L implies a mean ΔB of about 11.6¢, i.e. about 49% reversion.
- **Overstated.**
  - The 2018 and 2021 episodes supply $525k of the summed P&L; the other three average +$164 per fly.
  - 37% of OOS trades are Oct/Dec-centred, so this is not a "Mar/Apr" rule out of sample.

## 3. regimes_weather, reversal after a stressed week: **overstated**
- **Recomputed:**

| Sample | Slope | HAC t | n |
|---|---|---|---|
| Stressed before, 2010–21 | −0.49 | −3.3 | 41 |
| Stressed before, 2017–21 | −0.40 | −2.5 | 24 |
| **Stressed before, 2010–17 only** (not reported) | **−0.47** | **−1.6** | **17** |

  Dropping the largest week steepens the slope (−0.69), so it is not one-point-driven.
- **Test rule.** Confirmed: 5 trades, +$3,210. Without 19 Nov 2021 (+$6,830): +$2,305. Training: 4 trades, +$333. All 9 trades: about +$1,930. With per-side costs: +$3,130.
- **Multiple testing.** A within-winter label permutation gives p ≈ 0.0075 for the pooled slope. That is about 0.24 after Bonferroni × 32, and the 2017–21 cell is about 0.4. The split was chosen after seeing pooled data that includes 2017–21, so the out-of-sample label is not clean. Treat it as a hypothesis; the frozen holdout is the right test.
- **"Weather priced gradually" (+0.71) is likely mechanical:** the fitted part contains a₂, which also drives next week's move.

## 4. Curve leads storage reports by 1–3 weeks via weather: **confirmed with caveat**
- **Alignment is correct.** Kalman's "t+1" report is for the week ending F (out at F+6). regimes_weather's `surp5` covers the week ending F−7, released between the two settles. Both find about zero for the report released inside the price interval.
- **But "t+1" is the same physical week as the price move.** It reflects current weather, not forecasting. The real forecast lead is **0–2 weeks** (t+2: −0.16, CI spans 0; t+3: −0.27).
- **Both agents independently find the week ending F+14 strongest.**
- **Weather controls absorb 71%** of the t+3 coefficient (−25.7 → −7.5 Bcf per sd).
- **The non-weather lead is nil in 2018–21** (t −0.4).
- **"Via weather" rests on realised weather,** not forecasts.

## 5. implied_storage
- **Walk-forward mapping: confirmed.**
- **"Who moves" inconclusive: confirmed with caveat.** The i.i.d. noise bound (−0.55) is right and unidentifying. The informative-at-t+h filter drops 7 pairs whose spread fell (−2.3% vs −0.6% of April), biasing the market slope toward 0.
- **−105 Bcf vs Model 2: overstated.**
  - The 43 readable weeks are selected on a high spread: spread/April is 10.2% on them against 0.4% on the rest, where the median implied gap is +4 Bcf.
  - Model 2's median is itself 138 Bcf below the 5-year path and 193 Bcf below the outcome.
  - The gap is insignificant at s.e. ≤ 400 (−63, CI −179 to 26) or excluding floored weeks (−74, CI −213 to 9).
  - The defensible figure is −243 Bcf against the 5-year path, on tight-spread weeks.
- **"5-year path beats both": confirmed with caveat.**
  - It compares the same 39 weeks (MAE 183 / 292 / 339 Bcf, reproduced), and the benchmark equals avg5 + x_info to within 2 Bcf.
  - Only 5 winters, dominated by 2016-17 and 2018-19. The market beats the 5-year path in 2017-18 and 2020-21.
  - There is no significance test. On all 62 rows with an outcome, the 5-year path also beats Model 2 (210 vs 289 Bcf).

## 6. Cross-agent consistency: **no contradiction; shared-episode dependence**
- **Kalman's "no reversal in 2018–21" pools both signs and all months** (28 of its 44 jumps are Mar–Oct). I split its Nov–Feb jumps by regimes_weather's `stress_pre`:

| Regime before the jump | Slope of Mar−Apr fwd_4 | t | n |
|---|---|---|---|
| Stressed | **−0.35** | −4.2 | 9 |
| Calm | **+1.06** | 3.8 | 7 |

  That is the same pattern as regimes_weather, so the two results are compatible, but n is tiny.
- **curve_consistency's fade and regimes_weather's rule earn their OOS money in the same two episodes** (Nov–Dec 2018, Oct–Nov 2021). That is one scarcity-premium release, not two confirmations.
- **These agree:** Kalman (storage known at t beats the curve), implied_storage (5-year path most accurate) and curve_consistency (state-dependent Mar/Apr kinks; spread informative only when storage is tight).

## Most robust
1. **Storage surprises explain about nothing of weekly spread moves; weather 1–2 weeks ahead does, peaking at the week ending F+14.** Two codebases agree and the timing is verified.
2. **The winter curve is one factor (PC1 99%); the Kalman state adds nothing over Mar−Apr.**
3. **Small and medium kinks lose after costs, and storage known at t (or the 5-year path) beats curve-implied storage for end-March levels.**

## Weakest
1. **The stressed-week reversal and the 5-trade rule:** best of 32 cells, training-only t of −1.6, adjusted p about 0.2–0.4.
2. **The +$2,166 kink fade:** 5 episodes, 2 of which carry it; a CI that spans zero; +$2,006 at full round-trip costs; the same episodes as #1.
3. **The −105 Bcf gap to Model 2:** selected on tight-spread weeks, measured against a biased median, and fragile to the s.e. threshold.
