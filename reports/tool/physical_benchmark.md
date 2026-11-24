# physical_benchmark: the physical as-of distribution of end-March storage (link D)

*Agent `physical_benchmark`, 24 Nov 2026. Development data only: every as-of date is ≤ 2021-12-31, and every input (storage, daily weather, GEFS runs) is filtered to ≤ the as-of date inside `src/tool/physical.py`. Outcomes are read with as-of 2021-12-31, so end-March 2022 is never read and 2021-22 is shown but not scored. Pre-2010 EIA **monthly** storage (NG.N5020US2.M, 1973+) is used for the climatology and the pre-2010 baseline years. SPEC allows this because it is not locked.*

## 1. Plain-English findings
- **The new analog model is the best physical distribution we have.**
  - **What it is.** The model starts from the storage already reported. It adds realised weather up to yesterday, then GEFS ensemble-mean HDD for days 0–15, perturbed with past forecast errors. Past weather years fill the rest of the winter. A walk-forward HDD→draw regression converts weather into draws, and its winter-specific "balance" term is updated from the current season's weeks, including Sep–Oct.
  - **Score.** Over 10 winters (172 weekly as-of dates), its CRPS is **134 Bcf**, against 165 for P-lite (raw), 347 for climatology and 186 for Model 2 on Model 2's five winters.
  - **Winter-block CIs.** It beats P-lite by **−31 [−59, −5]**, Model 2 by **−58 [−88, −27]** (5 winters) and climatology by −213 [−378, −67].
- **The gain over P-lite comes from information, not model structure.** I ran the same model with only storage-week information (no forecast, weather through F, as P-lite and Model 2 use). It ties P-lite at +5 [−27, +35]. The extra week of realised weather plus the 15-day GEFS forecast is worth −36 [−52, −21] Bcf of CRPS, and most of it shows in Dec–Feb.
- **Centring on the 5-year path is clearly wrong for every model.**
  - Centring adds **+158 [59, 257]** of CRPS to P-lite and +186 [78, 289] to the analog model, and the 80% coverage of centred models falls to 44–55%.
  - This confirms the v5 review: the "P-lite centred" baseline behind implied_storage's +0.21 premium is a badly calibrated physical model. **Use raw distributions.**
- **All weather-scenario models are too pessimistic in 2016-17 to 2020-21.**
  - In the check window, the realised x sat **above** the analog median by +169 Bcf on average. The PIT mean is 0.67, and P-lite and Model 2 show the same (+175, +154).
  - The likely causes are warmer recent winters than the 2000+ analog pool, plus supply growth. Linear HDD-trend scaling fixed the check-window bias (+82) but hurt the colder choose window (2014, 2015), so it was not chosen.
  - **Implication for the tool:** physical P(x < −400) leans high. The measured "squeeze premium" (implied − physical) is therefore, if anything, *understated*.
- **Tails are fine but rest on very few events.**
  - In 2011-12 to 2020-21, x < −200 happened in 3 winters, x < −400 in 2 (2013-14, 2018-19) and x < −600 in 1 (2013-14).
  - The analog model's Brier score for x < −400 is 0.065, tied with P-lite (0.065) and better than climatology (0.170) and Model 2 (0.122).
  - 80% intervals cover 86% of dates (10 winters).
- **Model 2 cannot be cleanly extended before 2016.** Its Model 1 features need 5 years of weekly storage, which starts in 2010, so features start in Feb 2015. Its miss pool is walk-forward from 2016. Starting the pool in 2015 would leave winter 2015-16 with about 6 winter misses, and nothing earlier is possible without weekly storage before 2010, which the repo does not have. On its own five winters, Model 2 is also over-wide: 80% coverage is 100%, and its 80% band is 1,296 Bcf wide against 754 for the analog model.

## 2. Method
**Target.** x = storage in the last EIA week of March (last Friday) minus the mean end-March level of the five prior years.
- From 2010 the prior levels are weekly. Before that they are the EIA monthly March value plus the mean weekly-minus-monthly offset, about −20 Bcf.
- No storage_map definition was published when I finished. `physical_dist(..., baseline_bcf=...)` takes any other baseline: x shifts by a constant, so the model comparison is unaffected.
- This definition always uses a full 5 years. implied_storage instead used "up to 5 years", so its x differs in 2011-12 to 2013-14.

**As-of grid.** For each storage week F, the as-of date is d = F+7 (the Friday after the Thursday release), in Nov–Feb. That gives 17–18 dates per winter, 172 scored and 9 unscored (2021-22).

**Information timing.**
- Storage week F is public at F+6.
- Realised daily HDD is used to d−1.
- The GEFS 00Z run initialised on d is used at d's settle.
- Days with no realised weather before the run's init (the live lag) use that day's own lead-0 run.
- P-lite, Model 2 and the ablation `an_F` use origin F.

**Analog model** (`src/tool/physical.py`; walk-forward, refitted at every as-of date):
1. **Daily HDD paths for days F+1 … T.**
   - Known days: realised HDD.
   - Days d … d+15: GEFS ensemble mean + (realised − forecast) from the run on the same calendar date k years earlier.
   - Later days: analog year k's realised HDD.
   - k runs over every year from 2000 to Y−1, giving 11–21 paths.
2. **Weekly regression on Nov–Mar weeks**, plus Sep–Oct weeks with their own constant, shape and CDD term:

   chg = a_season + b₁·HDD + b₂·HDD²/100 + c·holiday + d₁·sin + d₂·cos + e

   The intercept a_season is a fixed effect per season.
3. **The current season's intercept.**
   - Prior: a linear trend over past seasons' intercepts. It carries a Mundlak term γ·(mean HDD anomaly of the season), fitted once there are ≥ 5 past winters: cold winters draw less than the weekly slope implies.
   - Prior variance τ²: the one-step errors of that prior, shrunk to 5 Bcf/wk.
   - Normal-normal update with the season's weeks so far.
4. **Scenarios.** Analog paths × 200 residual paths, giving 2,100–4,200 samples. Each residual path draws an intercept from its posterior and 4-week blocks of past within-winter residuals.

**Comparators.**
- Climatology: x of every winter since 1980 from monthly data, before Y.
- P-lite, raw and centred: implied_storage's code, unchanged except one fix. Its `_season_end` sent October origins to the *past* March; I pointed them to the coming March. v5 never used October origins.
- Model 2, raw and centred: the shared dev cache, 99 percentiles.
- Centred = median moved to the 5-year path (the v5 convention).

**Scores.**
- CRPS from samples, PIT, Brier for P(x < −200/−400/−600), and 50/80/90% coverage.
- Every statistic is a winter-weighted mean (each winter counts once). CIs come from a winter-block bootstrap with 4,000 draws.

**Choose/check.** Analog variants were chosen on 2011-12 to 2015-16 and checked on 2016-17 to 2020-21.
- I tried **8 variants**: prior {last-3 mean, trend} × balance window {Nov–Mar, Sep–Mar}, + HDD-trend scaling ×2, + Mundlak ×2.
- The Mundlak pair was added after I saw a slope-overstatement pattern in winters 2012–16. I also had the full table in view at the time, so treat it as semi-post-hoc. Its gain over the next-best variant is small (137 vs 142 on choose).
- The ablation `an_F` is not counted as a variant.

## 3. Results

**Table 1. Scores** (winter-weighted; CRPS in Bcf; lower is better; [winter-block 95% CI])

| Model | Window (winters) | CRPS | ΔCRPS vs climatology | Brier x<−200 | Brier x<−400 | Brier x<−600 | Cov 50 / 80 / 90 | PIT mean | 80% width | Median bias (real − med) |
|---|---|---|---|---|---|---|---|---|---|---|
| Climatology | all (10) | 347 | – | 0.226 | 0.170 | 0.094 | 30 / 60 / 70 | 0.46 | 917 | −11 |
| P-lite raw | all (10) | 165 | −182 [−343, −45] | 0.101 | 0.065 | 0.064 | 46 / 84 / 91 | 0.60 | 938 | +134 |
| P-lite centred | all (10) | 323 | −24 [−127, 77] | 0.205 | 0.132 | 0.113 | 23 / 55 / 69 | 0.48 | 938 | −15 |
| **Analog (chosen)** | all (10) | **134** | **−213 [−378, −67]** | 0.083 | 0.065 | 0.050 | 54 / 86 / 98 | 0.61 | 726 | +115 |
| Analog centred | all (10) | 320 | −27 [−134, 75] | 0.210 | 0.131 | 0.110 | 19 / 44 / 54 | 0.48 | 726 | −15 |
| Analog, storage-week info only | all (10) | 170 | −177 [−331, −40] | 0.098 | 0.081 | 0.061 | 44 / 79 / 96 | 0.60 | 848 | +138 |
| Analog (chosen) | choose (5) | 137 | −353 [−594, −125] | 0.065 | 0.026 | 0.029 | 44 / 83 / 97 | 0.54 | 677 | +61 |
| P-lite raw | choose (5) | 175 | −316 [−546, −96] | 0.084 | 0.043 | 0.042 | 27 / 68 / 82 | 0.55 | 832 | +94 |
| Analog (chosen) | check (5) | 130 | −73 [−174, +29] | 0.100 | 0.104 | 0.071 | 63 / 89 / 99 | 0.67 | 776 | +169 |
| P-lite raw | check (5) | 155 | −48 [−144, +49] | 0.119 | 0.087 | 0.085 | 65 / 100 / 100 | 0.65 | 1,045 | +175 |
| Model 2 raw | check (5) | 186 | −17 [−117, +74] | 0.176 | 0.122 | 0.081 | 92 / 100 / 100 | 0.57 | 1,296 | +154 |
| Model 2 centred | check (5) | 212 | +9 [−94, +77] | 0.189 | 0.079 | 0.043 | 71 / 94 / 98 | 0.43 | 1,296 | −94 |

**Table 2. Paired CRPS differences** (row model minus reference; winter-block CI)

| Comparison | choose | check | all |
|---|---|---|---|
| Analog − P-lite raw | −38 [−52, −19] | −25 [−78, +24] | **−31 [−59, −5]** |
| Analog (storage-week info) − P-lite raw | +13 [2, 24] | −3 [−64, +58] | +5 [−27, +35] |
| Analog − analog (storage-week info): value of +1 week of weather and GEFS | −50 [−70, −24] | −22 [−37, −14] | **−36 [−52, −21]** |
| Analog − Model 2 raw (Model 2 dates) | – | **−58 [−88, −27]** | – |
| P-lite − Model 2 raw (Model 2 dates) | – | −33 [−48, −14] | – |
| Centred − raw, P-lite | +268 | +47 | **+158 [59, 257]** |

**Table 3. By as-of month** (all 10 winters; CRPS, with 80% coverage in brackets)

| As-of month | Climatology | P-lite raw | Analog | Analog, storage-week info | Model 2 (5 winters) |
|---|---|---|---|---|---|
| Nov | 347 (60%) | 211 (88%) | **189** (90%) | 207 (90%) | 238 (100%) |
| Dec | 347 | 188 (93%) | **144** (98%) | 185 (88%) | 240 (100%) |
| Jan | 347 | 141 (90%) | **115** (83%) | 156 (78%) | 166 (100%) |
| Feb | 347 | 116 (65%) | **79** (75%) | 128 (58%) | 103 (100%) |

- **November.** Every model's spread is wide (80% band about 1,000 Bcf), and the gain over climatology is −157 [−295, −38].
- **February.** The 80% band shrinks to about 360 Bcf. Coverage drops to 75%: late-winter intervals are slightly too narrow, and the model under-represents late-March forecast misses (2020-21, after Winter Storm Uri).

**By winter** (Fig. `figures/physical_crps_by_winter.png`).
- The analog model has the lowest CRPS in 7 of 10 winters, and it beats P-lite in 8 of 10.
- Climatology wins the three near-normal winters: 2012-13 (x = −36), 2016-17 (+269) and 2020-21 (−34).
- In 2016-17, every weather model expected a deficit (analog 200, Model 2 206, P-lite 153).

**Figures.**
- `figures/physical_fans.png`: the as-of fans through each winter, with realised x.
- `figures/physical_pit.png`: PIT histograms. The analog and P-lite histograms are skewed right (too pessimistic). Model 2's is hump-shaped (too wide). Climatology's is spread.

## 4. What it means for the tool and the desk
- **Use the analog model's raw output as "physical P(x < x₀)".** Do not centre it. Its tails are no worse than P-lite's, and its body is materially sharper, especially from December onward, once GEFS and the extra week of weather are in.
- **The comparison is timing-matched to the market.** The model uses everything public at d's settle: the latest EIA week, realised weather to yesterday, and the 00Z GEFS run. This removes implied_storage's caveat that "physics knows a week less than prices".
- **Read the premium with the known lean.** In 2017–21 physics leaned low by about 170 Bcf at the median. If that persists, part of any "implied below physical" gap is *smaller* than true. Show this bias next to the premium; do not correct it silently.

## 5. Caveats
- **Ten scored winters, with 1–3 tail events.** Dates within a winter share one outcome. CIs are winter-block, and the check-window advantage over climatology (−73 [−174, +29]) is not significant.
- **Early winters fit the regression on 1–3 winters of weekly storage (2010+).** The intercept prior falls back to τ0 = 5 Bcf/wk (an a-priori constant). The Mundlak γ is active only from winter 2014-15.
- **Parameter uncertainty in the slopes is ignored.** Weather analogs come from 2000+ only, with no climate-trend adjustment in the chosen variant.
- **Release-day timing.** I assume storage is public on Thursday at F+6. Holiday-week releases can slip to Friday, which is covered because d = F+7.
- **Data-rule disclosure.** During a smoke test I ran `physical_dist("2026-11-24")` once. That reads storage through Nov 2026, so it includes end-March 2022–26 levels in the baseline. Only that date's distribution summary was printed; no locked outcome was scored or inspected. After that I ran no live or locked dates. I tested the live data-lag code path on a dev date instead (as-of 2021-12-17, weather through 2021-12-09).

## 6. Proposed frozen use on 2022–26 (for the orchestrator)
- **Inputs.** `physical_dist(d)` with the defaults (`prior="trend", balance="fall", mundlak=True`), unchanged, on weekly d = F+7, Nov–Feb.
- **Scoring.** Log CRPS, PIT, Brier at −200/−400/−600/−800 and 80% coverage against climatology and P-lite raw.
- **Pre-registered expectations.**
  - Analog CRPS < climatology in ≥ 3 of 4 winters.
  - Analog CRPS < P-lite raw on the winter-weighted mean.
  - 80% coverage between 70% and 95%.

## 7. Rerun
```
python -m src.research.tool_physical.bench grid
python -m src.research.tool_physical.bench run clim m2 plite an_w_l3 an_w_tr an_f_l3 an_f_tr an_w_l3_ht an_f_l3_ht an_f_tr_mu an_f_l3_mu  # ~80 s per analog variant
python -m src.research.tool_physical.score      # writes chosen.txt; then:
python -m src.research.tool_physical.bench run an_F && python -m src.research.tool_physical.score   # ~3 min
python -m src.research.tool_physical.figures
```
- **Cache:** `data/processed/research_tool_physical/`. It holds `grid.csv`, `q_<model>.pkl` (1,000-point quantile grids per date), `scores_long.csv`, `table_*.csv`, `crps_by_winter.csv` and `chosen.txt`.
- **Tool call:** `physical_dist(date)` takes about 0.4 s after a 2 s data load.
