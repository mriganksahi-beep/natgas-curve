# Study v5 (options): results on the locked 2022–2026 options sample (run once, 24 Nov 2026)

The tests were frozen in `reports/synthesis_v5.md` (commit 07d0c99) before any 2022+ options row was analysed. The runner (`src/research/reviewer_v5/frozen_ref.py`) first reproduced the development reference values exactly. It was then run once:

`python -m src.research.reviewer_v5.frozen_ref data/processed/v5_locked 2022-01-01 2026-08-15`

Positions were opened between Jan 2022 and Aug 2026, covering gas years 2022–2027. Raw output: `reports/holdout/v5_locked_output.txt`.

| ID | Test | Dev 2010–21 | **Locked 2022–26** | Verdict |
| --- | --- | --- | --- | --- |
| **P** | Spread-option (CSO) vol richness, mean ln(implied/realised spread vol), open interest > 0 | +0.33 [0.25, 0.44] | **+0.24 [0.065, 0.42]**; median +0.36; 4 of 6 gas years positive (2022 −0.16, 2026 −0.04); March−April alone +0.29 | **PASS** |
| S1 | Loose-storage winter vol: IV − realised, ≤ 126 trading days | +3.6 [0.5, 7.0] | +4.7 [−2.7, 14.4]: 2023 +12.0, 2024 +14.5, 2025 −0.3, 2026 −3.0 | FAIL (right direction; the lower bound is below 0) |
| S2 | Short delta-hedged CSO straddle, 4% costs | −$38 [−298, 186] | +$176 [−40, 395] per trade (+$113 at 8%); worst −$1,947 | FAIL, as expected (lower bound below 0; point estimate positive) |
| S3 | Outright fairness: IV − realised, all deliveries | −0.13 [−2.7, 2.2] | +2.4 [−3.5, 7.7] | FAIL by the frozen rule (CI includes 0, but \|mean\| ≥ 2): mildly rich and noisy, not clearly fair |
| Log | Tight-storage winter vol (not frozen: the dev "cheap" result rested on 3 episodes) | −14.1 [−23.0, −0.5] | **+15.9 [8.3, 47.6]: rich, the opposite sign** | Dev hypothesis reversed |

## What this means
1. **The one robust options fact survives a sample nobody had seen.** Calendar-spread options on adjacent gas months price more spread volatility than is realised. The markup was about +0.24 to +0.33 in log terms (roughly 27–39% too rich) in both 2010–21 and 2022–26, including the 2022 energy crisis, and for March−April specifically. This is the options-market face of the March−April path-risk premium from v3/v4: the market pays for decoupling (squeeze) risk.
2. **Harvesting it is still not proven.** The short CSO straddle was positive in 2022–26 (+$176 per trade), but its interval includes zero and the dev sample was negative. Costs and thin open interest remain the binding constraint.
3. **The storage-regime story did not hold.**
   - Loose-storage winter vol was rich again in direction (+4.7), but inconclusive.
   - "Winter vol is cheap when storage is tight" **reversed**: in 2022–26 it was rich (+15.9), driven by the 2022 crisis and 2026.
   - So it was not a stable property, and it was right not to freeze it as an edge.
4. **Outright vol: no reliable mispricing.** It was rich on average in 2022–26 (+2.4 vol points) but with a wide interval. Together with dev (−0.1), the evidence says no stable outright vol premium.
