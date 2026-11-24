# Tool spec: the Winter Squeeze-Risk Monitor (options-implied end-March storage odds)

*Orchestrator spec, 24 Nov 2026. It is the design contract for the parallel build investigations.*

## Purpose
Read the gas market's **odds for end-of-March storage** from option prices, compare them with physical (weather-scenario) odds, and show that the gap, the squeeze premium, is systematic. The audience is a natural-gas fund that forecasts inventory for a living. The tool is useful to them **because** they have their own forecast: it shows exactly what the market is pricing, in their units (Bcf), so they can see where they disagree.

## The chain (each link is one investigation)
```
 option settles ──(A) risk-neutral density of the March−April spread at option expiry, S_T ──┐
 (G4X spread options; and (B) ON March & April surfaces + implied correlation, as fallback)   │
                                                                                               ▼
 (C) map: spread at expiry  →  end-March storage deviation x (Bcf), WITH map uncertainty  ──►  implied P(x < x0)
                                                                                               │
 (D) physical as-of distribution of x (weather scenarios / analogs), calibrated  ─────────────►  physical P(x < x0)
                                                                                               ▼
                         premium = implied − physical;   calibration of both vs realised x
```

## Output (per date, and as "today" for winter 2026-27)
- The implied distribution of x: quantiles, and P(x < −200, −400, −600, −800 Bcf), each with an uncertainty band that includes the map error.
- The physical distribution of the same quantities.
- The spread-option vol-richness reading: ln(implied / forecast realised spread vol). Its premium was frozen-tested and PASSED on 2022–26.
- The spread's percentile against 2000–2026 at the same point in winter.

## Rules (same as `reports/agents/BRIEF_v5.md`, plus the following)
- **Development: options, futures and storage ≤ 2021-12-31.** The locked test sample is 2022–2026 (`data/processed/v5_locked/`, raw option rows from 2022 on, futures from 2022 on, storage from 2022 on). **No investigation opens it.** The orchestrator freezes the tool's gate test before it is run.
- **Pre-2010 data is no longer locked.** v4 spent it (run once, 23 Nov 2026). It **may** be used to fit the spread→storage map: EIA front-4 futures, which give March−April near the March expiry, plus EIA monthly storage from 1994. State this whenever you use it.
- Everything fitted is **walk-forward by winter.** The tool must run unchanged on any date given data up to that date.

## Success criteria for the finished tool, scored on 2022–2026 once frozen
1. **Calibration:** the implied P(x < x0) tails, and the full-distribution score (CRPS/PIT), are reported against climatology and the physical model. The gate test is specified by the orchestrator after the investigations.
2. **Coverage:** a usable reading on ≥ 90% of winter trading days (Nov–Feb), including thin-quote days.
3. **Stability:** day-to-day noise in P(x < x0) is small relative to its moves on news.
