# Holdout V3: autumn Nov−Oct contango drift, price only (agent `storage_space_option`)

*Run once on 22 Nov 2026 on the re-used holdout (autumns 2022–2026). The spec is frozen exactly as in `reports/agents/storage_space_option.md` §6 and was not changed after the run. Only prices were used: no storage data, no Model 2 and no simulator call. This is weak confirmation only: per `synthesis_v3.md`, the v3 thesis was formed with 2022–26 in view.*

## Spec (as frozen)
- **Drift.** D_Y = Nov−Oct contango at October's last trading day minus its average over trading days from 1 April to 31 August of Y.
- **Net P&L.** Short one contango spread at each of those days' settles and hold to October's last trade. Costs are 4 ticks per contract per side ($160 per spread round trip). Average the result within each year.
- **Pass:** D < 0 in at least 4 of 5 years **and** mean yearly net > 0.
- **Fail:** fewer than 3 years with D < 0, **or** mean yearly net < −$500.
- **Inconclusive:** anything else.
- **Tail flag:** any year with D > +15¢.

## Check before running (data ≤ 2021-12-31)
The runner reproduced the development numbers for every year 2010–2021:
- mean D −1.49¢, median −2.83¢
- 8 of 12 years negative
- mean yearly net −$11
- one tail year, 2020 (+36.5¢)

Those development numbers would themselves score INCONCLUSIVE under this rule. The only fix to the runner was a code bug (a column name clash). It was made before the holdout run and does not change the spec.

## Result (holdout, 2022–2026)

| Autumn | Entry days | D (¢) | Net $ per spread (yearly mean) | Tail |
|---|---|---|---|---|
| 2022 | 105 | +1.5 | −312 | no |
| 2023 | 106 | −28.6 | +2,704 | no |
| 2024 | 107 | −22.4 | +2,080 | no |
| 2025 | 105 | +2.0 | −360 | no |
| 2026 | 105 | −12.6 | +1,103 | no |

- **Years with D < 0:** 3 of 5. The pass rule needs at least 4.
- **D:** mean −12.0¢, median −12.6¢.
- **Mean yearly net:** **+$1,043** per spread.
- **Tail years:** none. The largest positive D was +2.0¢ (2025).

## Verdict: **INCONCLUSIVE**
The economics went in the predicted direction, and more strongly than in development: a mean fall of 12¢ against 1.5¢, and +$1,043 per year against −$11. The sign count fell one year short of the pass condition. In the two positive years (2022 and 2025), contango barely rose, by 1.5–2¢. Each cost about $310–360 per spread, mostly the round-trip cost. There was no 2020-style tail: the worst year lost $360.

**Reading.** This is consistent with autumn contango behaving as a sold-insurance premium that is usually earned, but it confirms nothing. The sample is 5 autumns, the holdout is re-used and the thesis was formed with these years in view. The large 2023–24 falls came from contango that was very wide in summer (roughly 5 times the 2013–19 norm), so the result says more about wide-contango summers than typical ones. The clean test is still pre-2010 autumns or live tracking.

## Rerun
`python -m src.holdout.storage_space_option --start 2022 --end 2026-11-17` takes a few seconds. The development check is `--start 2010 --end 2021-12-31`. The code is in `src/holdout/storage_space_option.py`.
