"""Misc: physics-given-S drift coefficient leave-one-winter-out; Model 2 summer cache dates."""
import pandas as pd
from src.research.reviewer_v3.common import ROOT, CUT
from src.research.fear_vs_physics import analysis as A
raw = pd.read_pickle(ROOT / "data/processed/research_fear_vs_physics/panel.pkl")
assert raw.obs.max() <= CUT
for drop in [None, "2013-14", "2018-19", "2014-15", "2017-18"]:
    q = raw if drop is None else raw[raw.winter != drop]
    p = A.add_indices(q)
    t = A.boot(p, "dS_exp", ["S", "Phys"], B=500)
    print(f"drop {drop}: dS_exp ~ S + Phys: Phys {t.loc['Phys', ['coef','lo','hi']].values.round(3)}")
m = pd.read_csv(ROOT / "data/processed/research_storage_space_option/model2_summer_percentiles_dev.csv",
                usecols=["origin", "target", "actual"], parse_dates=["origin", "target"])
print("summer cache origins", m.origin.min().date(), m.origin.max().date(), "targets max", m.target.max().date(),
      "actual non-null for targets>2021:", int(m[m.target > CUT].actual.notna().sum()))
