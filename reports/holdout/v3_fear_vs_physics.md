# V6: fear vs physics H-FvP on the re-used holdout (run once, 22 Nov 2026)

*Role: **descriptive**, per `reports/synthesis_v3.md`. The review (`reports/agents/review_v3.md`) judged two things: the fear/physics split is mostly mechanical, and the ≥ $0.10 rule was chosen after viewing 142 check cells. The minimum detectable effect over 4 winters (about $4,160 per trade) is larger than the effect. Runner: `src/holdout/fear_vs_physics.py`. Logs: `data/processed/research_fear_vs_physics/v6_devcheck_log.txt` and `v6_holdout_log.txt`.*

## Verdict
**PASS on paper. In substance it adds nothing over the price-only twin, and it rests on one episode.**

- **The FX rule made money.** Over 9 trades in 2 winters it made **+$3,372 per trade** net (95% winter-block CI 1,047–4,535). Its worst trade was **−$160**.
- **The twin, S > $0.15, on the same weeks, made +$3,132 per trade** over 12 trades in the same 2 winters. Its worst trade was also **−$160**.
- **The tail criterion passes only through the "neither loses more than $2,000" clause.** The two worst trades are the same trade, so there was no tail for physics to avoid.
- **FX earned less in total:** $30,350, against $37,580 for the twin.
- **Nov–Dec 2022 supplies 90% of FX's P&L:** $27,210 from 6 trades. 2023-24 and 2024-25 had no trades under either rule.
- **The holdout contains no 2014- or Nov-2018-type disaster.** The claimed benefit of physics, skipping a fairly priced spread in a tight winter, was never exercised.

## Procedure
1. **Spec:** exactly as frozen in `reports/agents/fear_vs_physics.md` §6, with the coefficients from `frozen_holdout.json`.
   - The panel is built as in `data.py`, with only the end date moved. Winter 2026-27 (the live test) is excluded.
   - Rule: FX = S − PP ≥ $0.10, where PP = −0.01049 + 0.013602·wte + 0.18091·Phys. Short 1 H−J, hold 4 weeks or to expiry if sooner, at 4 ticks per contract per side.
2. **Level-neutral fear** (for the secondary slopes):
   - Definition: nrv4 = s.d. over 20 days of ΔS_t / max(S_{t−1} + 0.20, 0.005).
   - The −$0.20 floor is scenario_risk's bound. It was fixed on dev data before the holdout run, because it reproduces the review's β_P: −0.56 [−0.90, −0.17] here, against the review's −0.58 [−0.91, −0.21].
   - Its z-scores and level coefficients were fitted on rows dated ≤ 2021-12-31 only.
3. **Dev check (end 2021-12-31):** the panel is identical to the agent's: 175 rows, max |ΔS| = 0, p_tail identical, S_4w identical.
   - The frozen rule in-sample: **28 trades, 9 winters, +$2,235 per trade, worst −$5,620**, as reported.
   - Slopes reproduced: fear part −0.99 (4 weeks) and −1.15 (expiry); physics part +1.06 and +0.63.
4. **Holdout:** run once with end 2026-11-17. No spec change afterwards.

## Results
**Rule (net $ per spread per trade; 4-week hold)**

| Segment | Rule | Trades / winters | Mean [95% CI] | Worst | Total |
|---|---|---|---|---|---|
| Dev 2010-11 to Dec 2021 (frozen, in-sample) | FX ≥ 0.10 | 28 / 9 | +2,235 [364, 4,400] | −5,620 | +62,570 |
| | S > 0.15 | 41 / 7 | +1,942 [651, 3,023] | −7,860 | +79,610 |
| **Jan–Feb 2022 (separate)** | FX ≥ 0.10 | 1 / 1 (28 Jan 2022) | +2,230 | +2,230 | +2,230 |
| | S > 0.15 | 1 / 1 (same week) | +2,230 | +2,230 | +2,230 |
| **Holdout 2022-23 to 2025-26** | **FX ≥ 0.10** | **9 / 2** | **+3,372 [1,047, 4,535]** | **−160** | +30,350 |
| | S > 0.15 | 12 / 2 | +3,132 [1,468, 3,964] | −160 | +37,580 |

**By winter.**
- **2022-23:** FX made +$27,210 on 6 trades (11 Nov – 16 Dec). The twin made +$31,710 on 8 trades.
- **2025-26:** FX made +$3,140 on 3 trades (23 Jan – 6 Feb). The twin made +$5,870 on 4 trades.
- **Neither rule traded in 2023-24 or 2024-25.**
- In Nov 2022, FX reached +0.94 while the analog P_tail stayed at about 0.36–0.42. The spike was priced well above physics, and it reverted, as in Nov 2018.

**Drift slopes on the decomposition parts** (holdout, 61 weeks, 4 winters; 95% winter-block CI; −1 = decays fully, 0 = retained)

| Fear definition | Target | Fear part | Physics part | Dev check (fear / physics) |
|---|---|---|---|---|
| Original fear index | ΔS 4 weeks | −0.54 [−1.10, −0.36] | −0.73 [−3.14, +0.39] | −0.99 / +1.06 |
| Original fear index | ΔS to expiry | −0.86 [−1.17, −0.25] | −0.37 [−4.14, +4.79] | −1.15 / +0.63 |
| Level-neutral fear | ΔS 4 weeks | +3.43 [+0.02, +5.65] | **−0.97 [−1.88, −0.67]** | −0.77 / −0.32 |
| Level-neutral fear | ΔS to expiry | +5.06 [−1.64, +6.95] | **−1.28 [−2.44, −0.34]** | −1.08 / −0.56 |

What the slopes show:
- **The fear part decays again** with the original index, though less than one-for-one over 4 weeks.
- **"The physics part is retained" is not supported.** With the original index the physics slope is uninformative. With level-neutral fear the physics part **decays fully** (about −1), as the review argued.
- **The level-neutral fear slope has the wrong sign and is unstable.** It rests on 4 winters, one of which is a spike.
- The residual decays at −0.8 to −1.2 under both definitions. As in dev, most of the spread above its time-of-season level decays, whatever it is attributed to.

## Notes
- **Run-to-run difference in dev rows.** In the holdout run, the dev segment's slopes (n = 175) differ slightly from the dev check (n = 167/171). Rows from Nov–Dec 2021 gain outcomes that fall in 2022. Trades are unaffected, since FX had no Dec 2021 entries. Use the dev-check numbers as the dev reference.
- **Thin winter-block CIs.** With 2 trading winters, the bootstrap resamples only 2 blocks, so the CIs are much too narrow to mean much.
- **Variants run on the holdout:** one rule, its pre-declared twin, and 2 fear definitions × 2 targets for the slopes. Nothing else.

## Bottom line
- **The fear-not-backed-by-physics short paid in the holdout, but so did the plain S > $0.15 short.** The twin earned more in total, with the same worst trade.
- **No disaster winter occurred,** so the one claimed advantage of physics was untested.
- **The decomposition replicated only one part:** the fear part decays. "Physics is retained" did not hold, and with level-neutral fear the physics part decays fully.
- **Status:** V6 stays descriptive. It is consistent with the synthesis view that the premium is a path-risk premium and that physics does not identify which part of it is safe to sell.
