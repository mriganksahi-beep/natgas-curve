# Study v4: results on the locked 2000–2010 sample (run once, 23 Nov 2026)

The tests were frozen in `reports/synthesis_v4.md` (commit 264f4dc) before any locked row was read. The runner (`src/holdout/v4_locked.py`, commit 39c5810) first reproduced every development reference value: P1 −0.35 (t −3.0); S1 0.048 daily and 0.224 weekly; S2 −0.24 and −0.20. It was then run once on `panel_pre2010.csv`. Prices come from the EIA front 4 contracts; forecasts from the GEFS reforecast. The sample is winters 2000-01 to 2009-10, with 13 calendar-gap rows set to missing as frozen. Raw output: `reports/holdout/v4_locked_results.json`.

| ID | Test | Development reference | **Locked 2000–2010** | Verdict |
| --- | --- | --- | --- | --- |
| **P1** | Weekly premium reversal, second month, 20 days | β_P −0.35 (t −3.0) | **β_P +0.40 (t +1.9)**; news +0.12 (t 0.2); raw move +0.36 (t 1.7). Positive at every horizon (h5 +0.15, h10 +0.22, h15 +0.31) and in every leave-one-winter-out fit (+0.24 to +0.48) | **FAIL.** The 2000s show weekly *momentum*, not reversal |
| S1 | Does the news split transfer? | Daily R² 0.048 [0.026, 0.075]; weekly 0.224 | Daily 0.021 [−0.005, 0.038]; weekly 0.114; March−April weekly 0.027 | **FAIL** (daily bound ≤ 0). As pre-declared, P1 is therefore read as a raw-move test, and the raw move also continues (+0.36) |
| S2 | Controlled daily reversal (uses future news; descriptive) | −0.24 (t −4.3), −0.20 (t −2.8) | h = 5: **−0.21 (t −2.5)**; h = 10: −0.07 (t −0.8) | **FAIL** (passes at h = 5 only; the rule needs both) |
| S3 | March−April short from 1 Dec to expiry | +$1,544 per winter (dev, 1 Dec entry) | **+$3,396 per winter**, 7 of 10 winters positive, CI [−$2,653, +$9,650]. **2002-03: −$16,010 at expiry, worst drawdown −$28,630**; 2000-01 drawdown −$20,100 (still ended +$5,810); 2005-06 +$24,630. Half size: +$1,698, drawdown −$14,315 | **PASS** (mean > 0). The tail is far larger than in development |
| Log | Costed fade rule (H-PS2) | +$258 per trade (dev, in-sample) | Split −$256 per trade (33 trades); naive −$902 (41 trades) | Loses; the split loses less |
| Log | Delayed entry (E rule) | R 1.09–1.16 | +$5,424 per winter; helped 2000-01 and 2005-06; **did not avoid 2002-03** (−$16,010) | Not a protection |
| Log | Premium predicts the next 00Z revision (H-HI-1) | t 6.2 | t 2.6 | Timing effect present, weaker |

## What this means
1. **The March−April premium survives a decade nobody looked at, and so does its path risk, at twice the size.**
   - Short from 1 December, it earned on average in 2000–2010 too: +$3.4k per winter, positive in 7 of 10 winters.
   - 2002-03 is the winter v3 warned about: a −$16k loss at expiry and a −$29k drawdown. 2000-01 was −$20k underwater before ending profitable.
   - Together with 2010–21 and 2022–26, the record is now **21 of 25 winters positive**, counting expiry P&L. Development used the November entry, the locked sample the December entry. **The disasters are rare and large**, so the premium pays for bearing path risk. Position size remains the only reliable control.
2. **Premium-shock reversal is not a stable property of the gas curve.**
   - In 2010–21, weekly second-month moves not explained by forecasts partly reversed. In 2000–10 they *continued*, at every horizon and with every winter left out.
   - The 2000s were a different regime: a tighter, less liquid, pre-shale market, with trending supply shocks (2000-01, 2003, 2005 hurricanes).
   - Reversal looks regime-dependent. It is not a frozen edge.
3. **The forecast-news split transfers only weakly to the 2000s:** daily R² 0.02 against 0.05, weekly 0.11 against 0.22. This fits a less weather-driven, more supply-driven market and an older forecast system's lower skill. Climatology fitted on 2010–21 is a further caveat.
4. **Nothing found in v4 is a tradable edge.** The costed fade loses in the 2000s, and the forecast-delayed entry does not avoid 2003.
