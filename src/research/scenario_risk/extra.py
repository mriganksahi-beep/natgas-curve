"""Blow-up ranking (AUC) and sizing cost sensitivity. Run after evaluate.py.
Blow-up (fixed a priori): 1-week loss > $1,000 per spread (dS > $0.10/MMBtu); 4-week loss > $2,000.
Benchmark ranker: the spread level S itself (no model)."""
import numpy as np
import pandas as pd

from src.research.scenario_risk import data as D
from src.research.scenario_risk.evaluate import SAMPLES, common, enrich, sizing, RNG
from src.research.scenario_risk.models import run_all

THR = {1: 1000.0, 4: 2000.0}


def auc(score, y):
    s, y = np.asarray(score), np.asarray(y, bool)
    if y.sum() == 0 or (~y).sum() == 0:
        return np.nan
    r = pd.Series(s).rank().values
    return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())


def auc_table(fc):
    rows = []
    for sname, (models, start) in SAMPLES.items():
        for h in (1, 4):
            x = common(fc, models, start, h)
            for m in models + ["S level"]:
                g = x[x.model == (models[0] if m == "S level" else m)]
                sc = g.S if m == "S level" else g.ES975
                y = g.loss > THR[h]
                ws = g.winter.unique()
                bs = []
                for _ in range(1000):
                    pick = RNG.choice(ws, len(ws))
                    gg = pd.concat([g[g.winter == w] for w in pick])
                    bs.append(auc(gg.S if m == "S level" else gg.ES975, gg.loss > THR[h]))
                lo, hi = np.nanpercentile(bs, [2.5, 97.5])
                rows.append(dict(sample=sname, h=h, model=m, n=len(g), events=int(y.sum()),
                                 ev_winters=g[y].winter.nunique(), auc=auc(sc, y), lo=lo, hi=hi))
    return pd.DataFrame(rows)


def main():
    fc = enrich(run_all())
    pd.set_option("display.width", 250)
    a = auc_table(fc)
    a.to_csv(D.CACHE / "auc.csv", index=False)
    print(a.round(3).to_string())
    for k in (0, 10):
        models, start = SAMPLES["M2"]
        models = [m for m in models if m != "PHYS-M2-pure"]
        tab, _ = sizing(fc, models, start, h=1, k_ticks=k)
        tab.to_csv(D.CACHE / f"sizing_M2_h1_k{k}.csv", index=False)
        print("k", k); print(tab[["model", "mean_N", "total", "per_avg_spread", "worst_per_avg", "sharpe"]].round(2))
    # where does the CONST vs ES-sized gap come from: P&L by winter, M2 sample h=1 k=4
    ser = pd.read_csv(D.CACHE / "sizing_series_M2_h1.csv", index_col=0, parse_dates=True)
    ser["winter"] = [D.winter_label(d) for d in ser.index]
    print(ser.groupby("winter").sum().round(0).T)


if __name__ == "__main__":
    main()
