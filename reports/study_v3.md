# Study v3: the gas curve as a set of insurance contracts

*Written 22 Nov 2026, before any v3 analysis.*

## Starting point
Study v2 found that the curve prices weather 1–2 weeks before storage data confirms it, and that no relative-value rule built on public fundamentals survives. If the market prices storage well, what's left is the **risk premium**: what the market charges beyond the real-world (physical) expected payoff.

One fact motivates v3. Across 16 winters (2011–2026), the March−April spread averaged +$0.24 in November and −$0.03 at expiry, and it fell between the two in 14 winters. The exception was 2014, the polar vortex. The spread behaves like an insurance policy whose time value decays unless disaster strikes.

**The v3 thesis:** storage physics turns parts of the curve into insurance contracts.
- **Winter scarcity:** March−April, and more generally winter over spring.
- **Storage space:** autumn contango, the rent on storage capacity.

Our weather-to-storage simulation (Models 1–2) supplies the physical probabilities needed to price them. The gap between market price and physical value is a measurable premium, with a known, weather-driven tail risk.

## Ideas under investigation
| # | Idea | Question |
| --- | --- | --- |
| 1 | **The widowmaker as weather insurance** | Fit the expiry payout curve (expiry spread vs end-March storage relative to normal) and push Model 2's scenarios through it, giving the actuarial value. Market − actuarial = premium. Is the premium positive, how does it decay through winter, and when is it rich? Does "sell only when the premium is rich and the physical shortage probability is low" earn it while avoiding disasters? |
| 2 | **The storage-space option** | Is autumn contango (Oct−Nov, Sep−Oct) the rent on scarce space? Price it from Model 2's end-October distribution and the capacity constraint. Is it mispriced as fullness becomes clear? |
| 3 | **A premium map of the whole curve** (prices only) | For every calendar pair and lead time: how much does the spread drift between being far from delivery and expiry? Where is the curve's insurance premium concentrated, and how does it depend on storage conditions? |
| 4 | **Physical tail risk for spread positions** | Does scenario-based expected shortfall for short-spread positions warn of blow-ups better than volatility models (historical, EWMA, GARCH)? Is it calibrated? |
| 5 | **The curve as a weather forecaster** | Invert the curve's weather sensitivity into a market-implied HDD anomaly for the coming 1–2 weeks. How skilful is it against climatology? Does it reveal when the market's weather view is extreme? |

Each investigation reports **new avenues** it uncovers, so a second wave can follow them.

## Evaluation rules and the spent holdout
- **Development data is up to 31 Dec 2021 only.** Model 2 scenarios exist from Nov 2016 (winter) and are cached for development in `data/processed/shared/`.
- **The 2022+ holdout was used once already, in v2, for related spread tests.** v3 runs its frozen tests on it once more. It reports them as a **re-used holdout**: informative, but not pristine.
- **The truly fresh tests are:**
  - Live tracking in winter 2026–27, with rules frozen before 1 Dec 2026.
  - An extension of Models 1–2 to before 2016, if EIA pre-2010 storage history is obtained. The v3 trading rules have never touched those winters, including 2014.
- **Everything else is as in v2:**
  - walk-forward only
  - block bootstraps by winter, HAC errors, simple benchmarks
  - a count of variants tried
  - costs of k ticks per contract per side
  - one frozen holdout test per idea
