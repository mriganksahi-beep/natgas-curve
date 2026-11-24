# Frozen gate test: Winter Squeeze-Risk Monitor on the locked 2022–2026 storage outcomes

Frozen 24 Nov 2026 and committed (with the sha256 of `src/tool/gate.py` and `src/tool/monitor.py`) **before** any
locked reading is computed. Run once. The results go in `reports/tool/GATE_RESULT.md` whatever they say.

## What is being tested
The monitor gives two market-implied probabilities each Friday from early November to 15 February:
- **Q_def400**: the options-implied probability that end-March storage ends 400+ Bcf below its 5-year average. Built from the March–April spread-option density and the walk-forward spread→storage map fitted on winters ≤ Y−1.
- **Q_squeeze**: the options-implied probability that March settles 25c+ over April at the March option expiry.

The physical analog model gives **P_def400_raw** as a reference.

## Sample
- Winters Y = 2022, 2023, 2024, 2025, 2026. Every Friday from 1 Nov (Y−1) to 15 Feb Y.
- If a Friday is a holiday or unusable, use the previous business day, up to 3 days back.
- Set the environment `RND_DATA_CUTOFF=2026-12-31`. Maps are walk-forward (`fit_map(Y−1)`).
- Outcomes:
  - x = EIA monthly end-March working gas minus the mean of the 5 prior end-March levels;
  - S_T = H − J at the March option expiry.

## Tests (rules fixed now)
| ID | Claim | Statistic | Pass rule | Dev 2012–21 (154 readings, 10 winters) |
| --- | --- | --- | --- | --- |
| **G1 (primary)** | Options-implied deficit odds carry information about storage | Brier skill vs walk-forward climatology (base rate of x < −400 over 1994..Y−1) | BSS > 0 | **+0.30**, 90% winter-block CI [−0.10, +0.37] |
| G2 | Options-implied squeeze odds are better than climatology | Brier skill vs climatology of S_T > 25c | BSS > 0 | +0.33, CI [−3.2, +0.44] |
| G3 | A physical storage model forecasts storage better than the options | Brier(Q_def400) − Brier(P_def400_raw) | > 0 | +0.052, CI [−0.03, +0.14] |
| X1 | (exploratory, no verdict) Does physical fair value add to the forward in predicting S_T? | corr(F, S_T), corr(P_mean_S, S_T) | — | 0.28, 0.52 |

## Honest limits, stated before the run
- **Power is very low.** Five winters, probably 0–2 deficit events and 0–1 squeeze events. A pass is consistent with skill, not proof of it, and a single event winter can decide every test. The dev intervals already include zero for all three tests.
- **S_T outcomes for 2022–26 are not fresh.** The futures record (21 of 25 winters) already used them in aggregate. The storage outcomes have never been scored against the tool, and the option densities for these dates have never been built.
- **Disclosure 1.** While benchmarking, the physical-model agent ran `physical_dist("2026-11-24")` once. That call read locked weekly storage, and it printed only the 5-year baseline (1,826 Bcf). No 2022–26 forecast or outcome was printed or analysed.
- **Disclosure 2.** The direct-density agent (A) was stopped by an automated safeguard before writing a report. Its saved engine was used as is, after the checks in `integration.md`.
- **No re-tuning.** No parameter of any link will be changed after this run. If G1 fails, the monitor is presented as a market-reading tool ("this is what the options price"), not as a storage forecaster.

## sha256 at freeze
```
03f6ab7a9fa38312b65bf0f6e4859a32c0d69ff0aa1dd9351691837461dbc834  src/tool/gate.py
5387958206c0864770fe56abe33cfe4d2cd3adfd3f574c3d34c093d9a527d4a9  src/tool/monitor.py
1a832b765118dfde1856d28b16e6e4c7f3b4bbb5d82b50ce206de0d3ff96d173  src/tool/rnd_direct.py
fd6b59c68a14fbaf682b874ebce7964b8fb32d61d1266df8bf4b580d009a0b50  src/tool/rnd_composite.py
d50c9fd9b0613c21db4bcb65d9b696eecf473da51a19eebdfc56f40ce4b627be  src/tool/storage_map.py
6afdeaab596c438da41c0232a6fece07d18c8239763c364fd71043d232643462  src/tool/physical.py
88a438bbefb528a06a965b6981f9955ccb93825d372dd781786f165dc448be8d  src/tool/_map_data.py
```
