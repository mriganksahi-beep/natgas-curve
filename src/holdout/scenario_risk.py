"""V5 = frozen test H-SR (reports/agents/scenario_risk.md section 6), plus the review's comparators.

Run:  PYTHONPATH=. python -m src.holdout.scenario_risk --end 2021-12-31   (check: reproduce dev numbers)
      PYTHONPATH=. python -m src.holdout.scenario_risk --end 2026-11-17   (holdout, run ONCE)

The research code in src/research/scenario_risk/ is imported unchanged. Only three module attributes
are re-pointed at run time: the data cutoff, the cache folder, and the Model 2 percentile source
(the full cache with 2022+ origins, filtered to origin <= end).

Evaluation weeks
  check:    forecast Fridays >= 2016-11-01 (the dev "M2" sample)
  holdout:  forecast Fridays >= 2021-12-31, i.e. every 1-week target window ends in 2022 or later
  Nov-15 Feb Fridays only, h = 1, weeks common to PHYS-M2, EWMA, HYB and LEVEL.

Primary (pass needs both):  (1) Kupiec p >= 0.05 for PHYS-M2's 99% VaR;  (2) Z2(PHYS-M2) < Z2(EWMA).
Comparators added per review_v3 (declared before the holdout run):
  HYB    as in models.py
  LEVEL  spread-level-only twin: loss ~ N(0, [max(S - a, 0.02) * sbar_z]^2 * h), where sbar_z is the
         s.d. of z = dS / max(S_prev - a, 0.02) over Nov-Feb weeks of earlier winters (walk-forward;
         a from the same map). HYB with an unconditional, winter-only variance instead of EWMA.
Secondary: blow-up AUC (1-week loss > $1,000) of each model's ES and of the spread level S (and S - a);
           Sharpe of a PHYS-M2-sized book (N = $5,000 / ES) against 1 spread, 4 ticks per side.
"""
import argparse
import json

import numpy as np
import pandas as pd

from src.research.scenario_risk import data as D
from src.research.scenario_risk import models as M
from src.research.scenario_risk import evaluate as E
from src.research.scenario_risk.extra import auc

FULL_M2 = D.ROOT / "data/processed/research_implied_storage/model2_percentiles_full_to_2026.csv"
MODELS = ["PHYS-M2", "EWMA", "HYB", "LEVEL", "GARCH-t", "HS-3y", "PHYS-WX"]
CORE = ["PHYS-M2", "EWMA", "HYB", "LEVEL"]


def setup(end):
    end = pd.Timestamp(end)
    D.CUTOFF = end
    D.CACHE = D.ROOT / f"data/processed/research_scenario_risk/holdout_end_{end.date()}"
    D.CACHE.mkdir(parents=True, exist_ok=True)

    def load_model2():
        m = pd.read_csv(FULL_M2, parse_dates=["origin", "target"])
        m = m[m.origin <= end].copy()
        return m.drop(columns=["actual"]).set_index("origin")
    D.load_model2 = load_model2
    return end


def level_forecasts(fc):
    df = D.add_storage_state(D.weekly_panel())
    maps = M.fit_maps(df)
    win = df[df.index.month.isin([11, 12, 1, 2])]
    recs = []
    for _, r in fc[(fc.model == "EWMA") & fc.a.notna()].iterrows():
        Y = int(df.loc[r.date, "Y"])
        a = maps[Y]["a"]
        tr = win[win.Y < Y]
        prevS = df.S.shift(1).reindex(tr.index)
        same = (df.Y == df.Y.shift(1)).reindex(tr.index)
        z = (tr.dS_back1 / np.maximum(prevS - a, M.FLOOR_HYB)).where(same).dropna()
        q = max(r.S - a, M.FLOOR_HYB) * z.std() * np.sqrt(r.h) * M.ZU
        rec = r.to_dict()
        rec.update(model="LEVEL", q=q)
        recs.append(rec)
    return pd.DataFrame(recs)


def main(end):
    end = setup(end)
    holdout = end > pd.Timestamp("2021-12-31")
    fc = M.run_all()
    fc = pd.concat([fc, level_forecasts(fc)], ignore_index=True)
    fc = E.enrich(fc)
    assert fc.date.max() <= end
    start = "2021-12-31" if holdout else "2016-11-01"
    core_dates = E.common(fc, CORE, start, 1).date.unique()
    x = fc[(fc.h == 1) & fc.model.isin(MODELS) & fc.date.isin(core_dates)]
    ok = x.groupby("date").model.nunique()
    x = x[x.date.isin(ok[ok == len(MODELS)].index)]  # all listed models present
    x.drop(columns="q").to_csv(D.CACHE / "v5_weeks.csv", index=False)

    rows = []
    for m in MODELS + ["S level", "S - a level"]:
        g = x[x.model == ("EWMA" if "level" in m else m)].sort_values("date")
        y = g.loss > 1000
        if "level" in m:
            sc = g.S if m == "S level" else g.S - g.a
            rows.append(dict(model=m, auc=auc(sc.values, y.values)))
            continue
        b99, b95 = g.loss > g.VaR99, g.loss > g.VaR95
        rows.append(dict(model=m, n=len(g), winters=g.winter.nunique(), br95=b95.mean(), br99=b99.mean(),
                         n99=int(b99.sum()), kup99=E.kupiec(b99.values, 0.01), chr99=E.christoffersen(b99.values),
                         Z2=E.z2(g), n975=int((g.loss > g.VaR975).sum()), mean_ES=g.ES975.mean(),
                         crps=g.crps.mean(), tailpb=g.tailpb.mean(), auc=auc(g.ES975.values, y.values),
                         events=int(y.sum()), event_winters=g[y].winter.nunique()))
    tab = pd.DataFrame(rows)
    p = tab.set_index("model")
    # winter-bootstrap CI for Z2(PHYS-M2) - Z2(EWMA) and for AUC(PHYS-M2) - AUC(S)
    pm = x[x.model == "PHYS-M2"].set_index("date")
    ew = x[x.model == "EWMA"].set_index("date")
    j = pd.DataFrame({"winter": pm.winter, "lp": pm.loss, "vp": pm.VaR975, "ep": pm.ES975,
                      "ve": ew.VaR975, "ee": ew.ES975, "S": pm.S})

    def dz(d):
        zp = (d.lp * (d.lp > d.vp) / (0.025 * d.ep)).mean() - 1
        ze = (d.lp * (d.lp > d.ve) / (0.025 * d.ee)).mean() - 1
        return zp - ze

    def dauc(d):
        return auc(d.ep.values, (d.lp > 1000).values) - auc(d.S.values, (d.lp > 1000).values)
    ci_dz, ci_dauc = E.winter_boot(j, dz), E.winter_boot(j, dauc)

    sz, ser = E.sizing(fc[fc.model.isin(CORE)], CORE, start, h=1, k_ticks=4)
    sz = sz[sz.model.isin(["PHYS-M2", "CONST", "EWMA", "HYB", "LEVEL"])]
    ser["winter"] = [D.winter_label(d) for d in ser.index]
    by_w = ser.groupby("winter").sum()

    c1 = p.loc["PHYS-M2", "kup99"] >= 0.05
    c2 = p.loc["PHYS-M2", "Z2"] < p.loc["EWMA", "Z2"]
    res = dict(end=str(end.date()), mode="holdout" if holdout else "check", start=start,
               crit1_kupiec_p=float(p.loc["PHYS-M2", "kup99"]), crit1=bool(c1),
               crit2_Z2_phys=float(p.loc["PHYS-M2", "Z2"]), crit2_Z2_ewma=float(p.loc["EWMA", "Z2"]),
               crit2=bool(c2), PASS=bool(c1 and c2), dZ2_ci=list(map(float, ci_dz)),
               dAUC_phys_minus_S=float(p.loc["PHYS-M2", "auc"] - p.loc["S level", "auc"]),
               dAUC_ci=list(map(float, ci_dauc)))
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(tab.round(3).to_string())
    print(sz.round(2).to_string())
    print(by_w.round(0).T.to_string())
    print(json.dumps(res, indent=1))
    tab.to_csv(D.CACHE / "v5_table.csv", index=False)
    sz.to_csv(D.CACHE / "v5_sizing.csv", index=False)
    by_w.to_csv(D.CACHE / "v5_sizing_by_winter.csv")
    json.dump(res, open(D.CACHE / "v5_result.json", "w"), indent=1)
    # episode listing: the 5 largest 1-week losses in the evaluation weeks, with each model's loss/ES
    big = x[x.model == "EWMA"].nlargest(5, "loss").date
    ep = x[x.date.isin(big)].pivot_table(index="date", columns="model", values="ES975")
    ep.insert(0, "loss", x[(x.model == "EWMA") & x.date.isin(big)].set_index("date").loss)
    ep.insert(1, "S", x[(x.model == "EWMA") & x.date.isin(big)].set_index("date").S)
    print(ep.round(2).to_string())
    ep.to_csv(D.CACHE / "v5_big_weeks.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", required=True)
    main(ap.parse_args().end)
