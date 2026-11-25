# The winter squeeze premium in US natural gas

*Working paper, November 2026 · [pre-registration log](reports/protocol.md) · [holdout results](reports/holdout/)*

## Abstract

This paper asks whether public information about weather and gas storage can be used to anticipate prices in the US natural gas market. Storage proves to be quite predictable from the weather. Prices, however, respond to the weather well before the storage data confirm its effect, so a storage forecast says little about prices that the market does not already know. One effect nonetheless survives every test: the market pays a persistent premium for protection against a gas shortage at the end of winter. Selling the March–April futures spread in December and holding it to expiry earned money in 21 of 26 winters. Options on these spreads were priced for 27% to 39% more price movement than subsequently occurred, this pattern was also confirmed over the holdout period. The paper also describes a weekly monitor that measures the size of this premium from current option prices.

![Figure 1](reports/figures/hero.png)
*Figure 1. Left: the profit or loss from selling one March–April futures spread on 1 December and holding it to expiry, for each winter from 2001 to 2026. Right: the extent to which options on these spreads overstated the spread's subsequent movement, before and after 2022. The vertical lines show 95% confidence intervals.*

## 1. The question

Natural gas is stored through the summer and withdrawn through the winter, so the amount left in storage at the end of March summarises how demanding the winter has been. The links are straightforward. Weather drives demand, demand drives withdrawals from storage, and the expected level of storage should in turn be reflected in futures prices.

Each link in this chain can be measured with public data. The aim of the project was to find how far along the chain a forecast remains useful. A forecast of the weather may well predict storage, but it is a separate question whether a forecast of storage predicts prices. If it does, the chain yields a trading signal. If it does not, the point at which it breaks says something about how the market works.

## 2. Method

Every test followed the same procedure, designed to prevent the results from being shaped after the fact.

1. **Each hypothesis was written down before any results were seen**, together with the rule for judging whether it had succeeded. These records, and the exact code used, are committed to this repository with timestamps.
2. **The models were developed on data from 2010 to 2021.** The years 2022 to 2026 were withheld and used for a single final test. The futures results were also tested once on the earlier decade, 2000 to 2010.
3. **Uncertainty was measured winter by winter.** Daily observations within one winter are closely related, and treating them as independent would overstate the strength of the evidence.
4. **Every final result is published, whether or not it supported the hypothesis.**

The weather and storage data are public and come from the US Energy Information Administration and the US weather service. The futures and options prices are exchange data purchased from a data vendor. Because those prices are licensed, the repository contains the code and the results derived from them but not the prices themselves.

## 3. Where the chain holds and where it breaks

The physical links of the chain hold well.
- A simple regression predicts each week's change in US gas storage from temperatures, weighted by where gas is consumed. Over 253 weeks that the model had not seen, its average error was 13.8 billion cubic feet, compared with 31.3 for the conventional benchmark, the five-year average.
- When the same model is run forward through the weather of past winters, it produces ranges for end-of-season storage that contain the true outcome about as often as they claim to.
- Since 18 November 2026 the model has published a forecast every week, before each official storage report.

The link from storage to prices does not hold in any way that can be exploited.
- Storage reports that differ from normal explain almost none of the movement in the March–April price spread.
- The reason is timing. Prices respond to weather forecasts as soon as those forecasts are issued, often a week or more before the storage report confirms their effect. By the time storage is known, the information is already in the price.
- Consistent with this, none of the trading rules built on storage survived the final test. A rule that bet on the reversal of unexplained price moves worked from 2010 to 2021 but failed in the earlier decade.
- Option prices did not predict the direction of futures prices either.

What remained after these tests was not a way of forecasting prices but a regularity in how the market prices risk. That regularity is the subject of the next section.

## 4. What survives: a premium for protection against a shortage

March is the last month of the heating season. If storage runs short, the price of March gas can rise far above the price of April gas, and anyone committed to delivering gas in March can suffer large losses. Such participants have good reason to pay for protection against this outcome. If they consistently pay more than the protection is worth, those who sell it should earn small profits in most winters and suffer occasional severe losses. The evidence shows exactly this pattern, and it shows it in two separate markets.

**The futures market.** Selling one March–April futures spread on the first trading day of December and holding it to expiry earned money in 21 of the 26 winters from 2001 to 2026. The typical (median) gain was about $1,200 per contract and the average about $1,800. The losses reveal the nature of the trade. In 2003 a late-winter shortage cost $28,400 per contract, roughly sixteen average winters of gains. The pattern also held from 2000 to 2010, a decade examined only in the final test.

**The options market.** If the spread carries a premium for protection, options on the spread should be priced as though the spread will move more than it actually does. To test this, the movement implied by each option's price was compared with the movement the spread actually showed before the option expired. Between 2010 and 2021, options implied 39% more movement than occurred. For the withheld years 2022 to 2026, the test was fixed in advance and run once. Options implied 27% more movement than occurred, with a 95% confidence interval of 7% to 52%, and the result held through the 2022 energy crisis.

Whether this premium can be collected after trading costs has not been established. A strategy that sold these options was profitable in the final test, but not by a margin large enough to rule out chance, and it lost money in the earlier period.

## 5. A monitor of the premium

![Figure 2](reports/tool/figures/monitor.png)
*Figure 2. Left: the distribution of end-March 2027 storage relative to its five-year average, as implied by option prices (blue) and by a physical model of storage and weather (orange), on 17 November 2026. Right: each model's probability of a shortfall of 400 billion cubic feet or more, averaged over each winter.*

If the premium is real, its current size is worth knowing. The monitor measures it each week from November to mid-February, in two ways.

**First, it compares the movement implied by current option prices with the movement the spread is actually showing.**
- On 17 November 2026, the options implied movement of about 47 cents a year. Over the previous four weeks, the spread had moved at about 10 cents a year.
- At the same point in past winters, the options overstated the spread's subsequent movement in 10 of 13 cases, typically by a factor of two.

**Second, it translates option prices into the probability that the market assigns to a storage shortfall, and sets this beside the probability given by the physical model.**
- On 17 November 2026, the options implied a 12% chance that end-March storage would fall 400 billion cubic feet or more below its five-year average. The physical model gave 9%.
- This second measure was tested before use, and it failed.
  - From 2022 to 2026 no such shortfall occurred, yet the options assigned it a probability of 15% to 25% every year. A simple historical average therefore proved more accurate.
  - The failure is consistent with a premium, since the market once again paid for protection it did not need. Five uneventful winters, however, cannot distinguish a premium from a lack of skill.
  - The monitor therefore reports what the market is pricing, and makes no claim about where storage will end up.

## 6. Conclusion

The chain from weather to prices holds in its physical links and breaks where prices enter. Storage can be forecast, but prices have already responded to the weather that drives it. What survives is not a means of predicting prices but a regularity in how the market prices risk. Buyers of protection against a late-winter shortage pay more than the protection turns out to be worth, in futures and in options alike, and the sellers of that protection are occasionally exposed to severe losses.

Three questions remain open:
- whether the premium can be collected after costs;
- whether its size changes predictably with storage and weather;
- whether a physical estimate of fair value can indicate when the premium is largest.

The weekly monitor and the live forecast record are designed to answer these questions as new winters arrive.

<details>
<summary>Appendix: specifications, data and reproduction</summary>

**Weekly storage model.**
- *Inputs:*
  - heating and cooling degree days weighted by gas consumption, together with their deviations from normal and their squares;
  - seasonal terms;
  - the five-year average change;
  - recent forecast errors;
  - a holiday indicator.
- *Fitting:* ridge regression with a penalty of 0.1, chosen by validation error on 2018–2021 from a grid of 0.001 to 100, and refit weekly.
- *Comparison:* a gradient-boosting model (learning rate 0.03, 7 leaves, 300 iterations, minimum leaf size 20, L2 penalty 1.0, refit every four weeks) had an error of 14.6 billion cubic feet. The 0.9 difference from the ridge model is not statistically significant (interval −0.9 to 2.8), so the simpler model is used. The scenario model and the live job use the same specification.

**End-of-season scenarios.**
- About 20 historical weather years, each adjusted to the latest 10-year normal.
- Out-of-sample errors are resampled in four-week blocks from the same half of the year.
- The result is a distribution of historical scenarios rather than a probabilistic weather forecast.

**Monitor.** Three components ([integration notes](reports/tool/integration.md)):
- a probability distribution for the spread, derived from option prices;
- a map from the spread to end-March storage, fitted on winters since 1994 and updated each year using only past data;
- a physical model based on storage, weather forecasts and comparable past winters.

The frozen test and its result are in [GATE.md](reports/tool/GATE.md) and [GATE_RESULT.md](reports/tool/GATE_RESULT.md).

| Path | Contents |
| --- | --- |
| `src/model1.py`, `src/model2.py`, `src/live.py` | Weekly model, scenarios, live job |
| `src/collect/` | Data collection, run by GitHub Actions |
| `src/v4/`, `src/v5/` | Weather-forecast and options studies |
| `src/tool/` | Monitor, final test, figures |
| `tests/` | Checks that no forecast uses future data |

```
pip install -r requirements.txt
python -m pytest tests
python -m src.model1
python -m src.model2
```

![Weekly model holdout](reports/figures/model1_holdout.png)
![End-of-season scenarios](reports/figures/model2_fans.png)
</details>
