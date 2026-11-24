# Tool integration notes (orchestrator, 24 Nov 2026)

## Link A: direct G4X density (`src/tool/rnd_direct.py`)
Agent A was stopped by an automated safeguard before writing its report. Its saved engine was **not
extended or re-researched**; it was only checked, reviewer-style, before use:

- **Runs** on 37 dev Fridays in winters 2014-15 to 2020-21 (every 4th Friday).
- **Usable** on 28 of 37. The 9 unusable dates are all after the March options expired (late Feb/March) or on a holiday (25 Dec). Both engines return nothing there, so the monitor's season is November to mid-February.
- **Mass and martingale.** Density mass was 0.996 to 1.024. The density mean matched the futures-implied forward F to within 0.001 on 24 of 28 dates. The largest gap was 7c (Nov 2014, F = 51c), where the right tail is truncated.
- **No look-ahead.** Rows are filtered to <= min(date, `RND_DATA_CUTOFF`). Open interest comes from the previous quote day. A date after the cut returns `no_data`.
- **Disagreement with the composite engine (B)** is largest when the spread is near zero. For example, on 2 Jan 2015 the probability that S_T > 0 was 0.24 direct and 0.56 composite. That day the direct smile is strongly right-skewed (median below zero, mean +5c), which the Gaussian copula cannot represent. The `select` rule from B's report decides which engine is used (direct when usable with >= 3 strikes carrying open interest). The direct engine was chosen on 122 of 154 dev readings.

## Link C: the map data cut
`src/tool/_map_data.py` was hard-cut at 2021. It now reads `MAP_DATA_CUTOFF` / `RND_DATA_CUTOFF`, so the default is unchanged and the dev cache is identical. Walk-forward is still enforced by `fit_map(Y - 1)`.

## Monitor (`src/tool/monitor.py`)
One reading per settlement date:
- **Market leg:** `select(direct, composite)`, then the `inv_asinh` lens, giving `Q_def400` = P_Q(x < -400). Also `Q_squeeze` = P_Q(S_T > 25c).
- **Physical leg:** the analog model gives `P_def400_raw`. Its samples go through the forward map S|x (`hetero_exp`) to give `P_squeeze` and `P_mean_S`.
- **Convention:** x follows the map's monthly convention; the physical model's weekly levels are shifted by the weekly-monthly offset.

**Design finding on dev, and why the headline is not a "Q minus P premium":**
- Comparing a lens-blurred Q reading with physical odds produces a fake premium; this was the v5 error.
- Pushing the physical leg through the forward map overshoots: its mean expected spread was 13c against 0.6c realised. The map's history is dominated by pre-shale squeezes (1996, 2001, 2003, 2014).
- So the "premium" computed that way is negative on average (−0.10) and does not line up with P&L (corr −0.18).
- The monitor therefore shows the market and physical readings side by side, each scored on its own, rather than a subtraction.
