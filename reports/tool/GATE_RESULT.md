# Gate test result: locked 2022–2026 (run once, 24 Nov 2026)

The test was frozen in `reports/tool/GATE.md` (commit c3cbc34) before any locked reading was built.

Run:
```
RND_DATA_CUTOFF=2026-12-31 python -m src.tool.gate data/processed/tool_locked/panel_locked.csv
```
Raw output: `reports/tool/gate_locked_output.json`. There were 76 readings, all through the direct G4X engine.

| ID | Test | Dev 2012–21 | **Locked 2022–26** | Verdict |
| --- | --- | --- | --- | --- |
| **G1** | Brier skill of options-implied P(x < −400) vs climatology | +0.30 [−0.10, 0.37] | **−0.66 [−1.09, −0.23]** | **FAIL** |
| G2 | Brier skill of options-implied P(S_T > 25c) vs climatology | +0.33 [−3.2, 0.44] | −2.20 [−3.50, −0.82] | FAIL |
| G3 | The physical model beats the options on storage (Brier gap) | +0.052 | +0.0007 [−0.03, 0.03] | PASS by the rule (point > 0), but a tie |
| X1 | corr(F, S_T) vs corr(physical fair spread, S_T); exploratory | 0.28 vs 0.52 | −0.23 vs 0.68 | — |

## What happened
- **There were no events.** None of the five locked winters ended 400+ Bcf short, and none squeezed 25c+ at expiry.
- **The options still priced the tail.** They put the deficit at about 15–25% and the squeeze at about 6–14% every winter.
  - Climatology (about 16% and 5%) priced it less, so climatology scored better.
  - In dev, the two event winters (2014, 2019) were where the options earned their skill. With no events, the only thing being scored is how much tail is priced, and the options price more.

## How to read it (no re-tuning, per the frozen rule)
1. **The monitor is not a storage forecaster.** On storage, the physical model was as good or better in both samples. The options-implied deficit odds lost to climatology in a quiet sample. From now on the monitor is presented as a **market-reading tool**: what the options price for end-March, next to what the fundamentals say.
2. **The failure points the same way as the rest of the project.**
   - The options carried tail probability that did not happen: deficit and squeeze odds above both climatology and outcome in 5 of 5 winters.
   - That fits a risk-neutral squeeze premium (Q ≠ P). It is the same direction as:
     - the March–April futures premium (21 of 25 winters);
     - CSO vol richness (+0.24 locked).
   - **But this test cannot tell "premium" from "no skill"**, so it is a consistent observation, not evidence.
3. **The exploratory lead (X1).** In both samples, the physical model's fair expected spread tracked the realised expiry spread better than the futures forward did.
   - Correlations: 0.52 vs 0.28 in dev, 0.68 vs −0.23 locked.
   - This is the trading-relevant direction: forward minus physical fair value as a short-spread signal.
   - It was not pre-registered, the S_T outcomes are not fresh, and there are 15 winters in total. It is the next thing to test properly, live.
