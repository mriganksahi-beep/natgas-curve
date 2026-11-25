# The winter squeeze premium in US natural gas

*Working paper, November 2026 · [pre-registration log](reports/protocol.md) · [holdout results](reports/holdout/)*

## Abstract

This paper asks where weather and storage information shows up in US natural gas futures and options. Mostly it does not show up where I expected: storage surprises barely move the March–April spread, and none of the relative-value rules I tested survived a holdout.

One effect survives every test. The market charges an insurance premium against an end-of-winter squeeze, and it appears in both futures and options:
- **Futures.** Shorting the March–April spread from 1 December to expiry earned money in 21 of 26 winters (2001–2026), with rare but large losses.
- **Options.** Calendar-spread options priced 39% more spread volatility than was realised in 2010–2021. On 2022–2026, a period held back and tested once, they priced 27% more (95% CI 7% to 52%).

Along the way I built a weekly storage forecast that runs live, and a monitor that reads the premium from option prices each week.

![Figure 1](reports/figures/hero.png)
*Figure 1. Left: P&L of one short March–April futures spread, held from 1 December to the March option expiry, by winter. Right: how much more spread volatility calendar-spread options implied than was later realised, in the development period and in the untouched holdout. Whiskers are 95% winter-block bootstrap intervals.*

## 1. Motivation

US working gas in storage is drawn down through the winter, and March is the last delivery month of the heating season. When end-March storage looks short, the March contract can trade far above April; traders call this spread the widowmaker. Anyone short winter gas faces a costly tail from a cold February or an unexpected supply loss, so they should be willing to pay to hedge it.

If hedgers consistently pay more than the risk is worth, the spread and its options should be overpriced on average. Sellers would then collect small gains most winters and take occasional severe losses.

I did not start from this hypothesis. I set out to test each link of the chain from weather to storage to the futures curve. The premium is what remained once the other links failed out of sample.

## 2. Data and testing discipline

**Data.**
- Physical data are public: EIA weekly and monthly storage, Open-Meteo temperatures, NOAA GEFS ensemble forecasts and CPC outlooks.
- Futures and option settlements are CME data obtained through Databento. They are licensed, so this repository contains only code and derived results.

**Discipline.** Each study followed the same rules:
1. **Pre-registration.** Hypotheses, statistics and pass/fail thresholds were committed here before results were seen, and frozen code is identified by its hash.
2. **Development vs holdout.** Models were developed on 2010–2021 and tested once on 2022–2026. The futures results were also tested once on 2000–2010.
3. **Inference.** Confidence intervals come from a bootstrap that resamples whole winters, because observations within a winter are not independent.
4. **Reporting.** Every holdout result, including the failures, is published in [`reports/holdout/`](reports/holdout/).

## 3. Results

### 3.1 The futures premium

Take a trader who sells one March–April spread (10,000 MMBtu) on the first trading day of December and holds it to the March option expiry.
- **Win rate.** The trade made money in 21 of 26 winters.
- **Size of gains.** The median gain was $1,200 and the mean $1,800.
- **The losses.** These are what make it a premium rather than an anomaly. In 2003 a late-winter squeeze cost $28,400 per spread, roughly sixteen average winters of gains. The 2014 polar vortex cost $3,600.
- **Holdout.** The pattern held in 2000–2010, a decade I examined only at the final test ([v4 summary](reports/holdout/SUMMARY_v4.md)).

### 3.2 The options premium

If the spread carries a squeeze premium, options on it should be priced as if the spread will move more than it does.

**Measure.** For each calendar-spread option with open interest, I compared its implied normal volatility with the volatility the spread actually realised over the option's remaining life.

**Results.**

| Sample | Mean log ratio (implied / realised) | Implied vs realised |
| --- | --- | --- |
| Development, 2010–2021 | +0.33 | 39% richer |
| Holdout, 2022–2026 (frozen in advance, run once; includes the 2022 energy crisis) | +0.24 | 27% richer (95% CI 7% to 52%) |
| Holdout, March–April only | +0.29 | about 34% richer |

**Can the premium be harvested?** That is not yet established. A short, delta-hedged straddle earned $176 per trade in the holdout, but the interval includes zero and the same trade lost money in development ([v5 summary](reports/holdout/SUMMARY_v5.md)).

### 3.3 Forecasting storage

The physical side of the chain is far easier to forecast than prices.

**Weekly change.** A ridge regression on gas-weighted degree days and seasonal terms predicts the weekly storage change.
- Out of sample, over 253 weeks, its mean absolute error is 13.8 Bcf.
- The five-year-average benchmark misses by 31.3 Bcf.
- The improvement is 11 to 24 Bcf (95% CI).

**End of season.** Simulating the model forward through historical weather years gives calibrated ranges: the 80% intervals covered the outcome 85–93% of the time.

**Live.** Since 18 November 2026, the model has published a forecast every Wednesday before the EIA report, timestamped by an automated commit ([live record](reports/live_forecasts.csv)).

## 4. What did not work

The negative results mark the boundaries of the premium:
- **Storage surprises.** Relative to normal, they explain almost none of the March–April spread's moves (holdout R² of 0.02). The curve has already moved on weather forecasts before the storage report arrives.
- **Relative-value rules.** None survived its holdout: not curve kinks, not implied versus forecast storage, not a storage-based fair value.
- **Fading price shocks.** Shocks not explained by forecasts partly reversed in 2010–2021 and continued in 2000–2010.
- **Options as predictors.** Option prices and Greeks did not predict futures returns.
- **Outright options.** Volatility on single-month contracts was roughly fair.
- **End-of-season forecast.** Its central forecast beats the five-year path only within about eight weeks of the season's end.

## 5. A live monitor

![Figure 2](reports/tool/figures/monitor.png)
*Figure 2. Left: end-March 2027 storage relative to its five-year average, implied by spread options (blue) and by a physical model (orange), on 17 November 2026. Right: each model's probability of a 400 Bcf deficit, averaged over the weekly readings of each winter.*

From November to mid-February the monitor reports two readings each week.

**Option richness.**
- On 17 November 2026, the March–April option implied a spread volatility of $0.47 a year. The spread had actually moved at $0.10 a year over the previous 20 days.
- At the same point in past winters, implied volatility exceeded what the spread went on to realise in 10 of 13 cases, typically by about two times.

**Scarcity odds.**
- The monitor converts the option-implied distribution of the spread into odds for end-March storage, and sets them beside a physical model built from storage, weather forecasts and analog winters.
- Today the options imply a 12% chance of a deficit of 400 Bcf or more; the physical model says 9%.

**Validation of the scarcity odds.** I tested these odds before using them ([test](reports/tool/GATE.md), [result](reports/tool/GATE_RESULT.md)).
- They lost to the historical base rate on 2022–2026: no deficit occurred, yet the options priced odds of 15–25% every year.
- That is consistent with a premium, but five quiet winters cannot separate a premium from a lack of skill.
- So the monitor describes what the market is pricing, not where storage will end up.

## 6. Conclusion

Weather and storage are forecastable, but by the time that information is confirmed, prices already reflect it. What survives out of sample is a premium for bearing squeeze risk, visible in both futures and options: profitable on average and dangerous in the tail.

Three questions remain open, and the monitor and live records are set up to answer them:
- Can the option premium be harvested after costs?
- Does its size vary predictably with storage and weather?
- Can a physical fair value time it?

<details>
<summary>Appendix: specifications, data and reproduction</summary>

**Weekly storage model.**
- **Features:**
  - gas-weighted heating and cooling degree days, with their anomalies and squares;
  - seasonal terms;
  - the five-year average change;
  - recent forecast errors;
  - a holiday flag.
- **Fitting:** ridge regression with α = 0.1, chosen by validation error on 2018–2021 from a grid of 0.001 to 100, and refit weekly.
- **Benchmark model:** gradient boosting (learning rate 0.03, 7 leaves, 300 iterations, minimum leaf 20, L2 1.0, refit every four weeks) scored 14.6 Bcf. The 0.9 Bcf difference is not significant (CI −0.9 to 2.8), so the simpler model is used.
- The scenario model and the live job use the same specification.

**End-of-season scenarios.**
- About 20 historical weather years, each re-centred on the latest 10-year normal.
- Out-of-sample errors are resampled in four-week blocks from the same half of the year.
- The result is an empirical scenario distribution, not a probabilistic weather forecast.

**Monitor.** Three components ([integration notes](reports/tool/integration.md)):
- a risk-neutral density of the spread, from calendar-spread option settlements;
- a walk-forward map from spread to storage, fitted on winters since 1994;
- an analog physical model.

| Path | Contents |
| --- | --- |
| `src/model1.py`, `src/model2.py`, `src/live.py` | Weekly model, scenarios, live job |
| `src/collect/` | Data collectors run by GitHub Actions |
| `src/v4/`, `src/v5/` | Weather-news and options studies |
| `src/tool/` | Monitor, gate test, figures |
| `tests/` | No-look-ahead checks |

```
pip install -r requirements.txt
python -m pytest tests
python -m src.model1
python -m src.model2
```

![Weekly model holdout](reports/figures/model1_holdout.png)
![End-of-season scenarios](reports/figures/model2_fans.png)
</details>
