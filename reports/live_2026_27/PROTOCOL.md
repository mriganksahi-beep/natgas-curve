# Live tests, winter 2026–27 (frozen 23 Nov 2026, before 1 Dec 2026)

These rules are fixed. The code is `src/live2627.py`, and its frozen parameters block must not change after 30 Nov 2026. Results update weekly through `.github/workflows/live_2026_27.yml` into this folder. If a rule has to change (a data-source failure, say), the change is logged below with its date and reason, and the original rule's result is still reported.

**Why live.** The 2022–26 holdout has been used twice: once in study v2, and again in v3, whose thesis was formed with those years in view. Live data is the only sample that no part of the research has seen.

**Data.**
- **Prices:** Yahoo Finance daily closes of individual NYMEX contracts (`NGH27.NYM` etc.). Yahoo is free and redistributable, and no licensed CME data is stored here. On 23 Nov 2026, Yahoo closes were checked against CME settlements (via Databento) for 3–17 Nov 2026, 11 days and 6 contracts. They matched exactly, with a maximum absolute difference of $0.000.
- **Weather:** the Open-Meteo archive, topped up with Open-Meteo's analysis for the latest days, for the same 17 cities and gas weights as Model 1.

## L1: March−April 2027 premium path
- **Position:** short 1 March 2027 − April 2027 spread (10,000 MMBtu per leg), held to the March 2027 expiry. Costs are $160 per round trip (4 ticks per contract per side).
- **Entries**, both frozen in earlier studies:
  - **"may":** the last trading day on or before March expiry minus 9 months, i.e. **22 May 2026** (the `premium_map` / V2 spec).
  - **"nov":** the first trading day on or after the first Friday on or after 1 November, i.e. **6 Nov 2026** (the `premium_risk_management` base trade).
- **Prediction:** the spread falls into expiry, so net P&L at expiry is greater than 0 for each entry. This is a single draw, so it counts as one observation added to the record (10 of 11 development winters and 4 of 4 holdout winters), not as a test on its own.
- **Reported daily:** the spread, mark-to-market P&L, peak-to-trough drawdown, and the worst P&L against entry. No threshold applies to the drawdown. It is recorded because study v3's thesis is that the risk lies in the path, not the expiry.

## L2: the curve as a 2-week weather forecaster
- **Sample:** each Friday F from 6 Nov 2026 to 26 Feb 2027.
- **Price:** the close on the first trading day on or after F.
- **Front contract:** the earliest contract with at least 2 trading days to expiry.
- **Front-month move:** r_front = 100·Δlog(front) from the previous Friday's settle day, on the same contract.
- **Weather:**
  - a0 is the gas-weighted HDD anomaly of the week ending F.
  - wx2 is the sum of the anomalies for the weeks ending F+7 and F+14.
  - Anomalies are measured against the mean of the same calendar week in the prior 10 years (at least 5 required). This is the same definition as in development.
- **Frozen forecasts** (coefficients fitted on 2010–2021; identical to `frozen_holdout_coefs.json`):
  - front: ŵ = −5.308 + 3.109·r_front
  - persistence: ŵ = −6.616 + 0.751·a0
  - front + persistence: ŵ = −3.378 + 2.673·r_front + 0.685·a0
- **Score:** the skill score SS = 1 − Σ(y−ŷ)²/Σy², plus correlation.
- **Season predictions:**
  1. SS(front) > 0
  2. SS(front) > SS(persistence)
  3. SS(front + persistence) > SS(persistence), which is the V1 statistic
- **Decision rule:** about 17 weeks is too few for a standalone significance test. The decision test is the **pooled** one-sided Diebold–Mariano test of front + persistence against persistence over 2022–2027 (the holdout plus this winter), with p < 0.05. The front-alone versus persistence comparison is added because the holdout showed front alone to be stronger. That choice was made after the holdout, so on the pooled sample it is a secondary statistic, and only this winter is a clean test of it.

## Dropped from the v3 plan
- **The PHYS-M2 alarm log.** Both scenario-based tools failed their holdout tests: V5 (calibration) and V7 (overlay). Logging them adds no decision-relevant evidence, and Model 2's weekly end-of-season forecast keeps running in the live storage job anyway.

## Planned for study v4 (not frozen here)
NOAA keeps permanent public archives of the CPC 6–10 and 8–14-day outlooks and the GEFS forecasts, so this winter's forecasts can be collected at any time and need no daily snapshot. Any v4 live rule will be frozen in a dated amendment below **before** it is scored. Only weeks after that freeze date count as live evidence.

## Change log
- 2026-11-23: protocol frozen. First run committed (L1 from both entries; L2 with 1 week scored). Yahoo = CME settle check passed (see Data).
