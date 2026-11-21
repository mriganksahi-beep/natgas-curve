# Study v2: What does the gas curve know about storage, and when does it get ahead of itself?

*Pre-registered 21 Nov 2026, before any v2 analysis. The holdout (all dates from 1 Jan 2022) is not used for any choice. It is scored once, after the v2 designs below are frozen.*

## The idea in one paragraph
Every Henry Hub futures price is a forecast of what gas will be worth in a given month. Gas can be stored but not borrowed from the future. So the gaps between neighbouring months, the spreads, have to tell one physically possible story about storage: how full it will be, and how scarce gas will be, month by month. We read that story out of the curve. Then we ask two things:
1. **Is the story internally consistent?** When one month's price implies something its neighbours contradict, does that kink correct?
2. **Does it agree with a public, weather-based storage forecast (Models 1–2)?** When the market's implied storage departs sharply from ours, who turns out to be right: does the price come back, or does our forecast move to meet it?

## The economics (no more than this)
- For two months, near and far: **far − near = storage cost + financing − convenience yield**, per month apart.
  - The convenience yield is the value of having gas in hand. It is near zero when storage is plentiful and large when storage is tight.
  - Rearranged, every spread reveals an **implied convenience yield**: the market's scarcity reading for that stretch of the calendar.
- **Summer bound.** Contango cannot exceed full carry, i.e. the cost of storing plus financing. Otherwise you could buy, store and sell at a riskless profit.
- **No winter bound.** Backwardation has no upper limit, because gas cannot be borrowed from the future.
- **Physics.** Storage moves smoothly, within capacity, at bounded injection and withdrawal rates. Scarcity readings along the curve must therefore line up with one feasible inventory path.
- **Insurance premium.** Prices include a premium for protection against running short. A *steady* difference between market-implied and forecast storage is that premium. *Sudden* departures are the candidate overreactions.

## Questions
- **Q1, curve consistency:** fit each day's curve with a smooth, physically consistent structure. Do the residuals ("kinks") shrink over the following 1–20 trading days by more than trading costs?
- **Q2, implied vs forecast storage:** each week, express the market's scarcity pricing as an implied end-of-season storage level, and locate it in Model 2's scenario distribution.
  - How large and how stable is the insurance premium?
  - Do large departures predict spread moves, or moves in our own forecast (who moves toward whom)?
- **Q3, a better summary of the curve's expectations:** can a state-space model (Kalman filter) combine all winter spreads into one latent "market-expected scarcity" series that is more stable and informative than any single spread? And can regime models (Markov switching) separate calm periods from stressed ones, so that we know *when* gaps tend to close?
- **Q4, what explains the gaps:** how much of the gap between market and forecast is explained by the weather that actually followed? That would mean the market priced forecasts we do not have.

## Rules for every analysis
- **Evaluation and model choice use dates up to 31 Dec 2021 only.**
  - Curve data starts June 2010.
  - Model 2 scenarios start November 2016.
- Every fit uses only information available on the date of the prediction (walk-forward).
- Uncertainty comes from block bootstraps by winter or by month, and from serial-correlation-robust (HAC) errors.
- Each result is reported against a simple benchmark: no change, or the previous value.
- Trading claims include costs of 2–10 ticks per leg, and a butterfly has 3 legs.
- Each analysis ends with **one precise test to run on the holdout**, frozen before any holdout row is touched.

## Deliverables
- One plain-English findings note per analysis (`reports/agents/`).
- A synthesis: what the curve knows about storage; when, if ever, it gets ahead of itself; and the frozen holdout tests.
