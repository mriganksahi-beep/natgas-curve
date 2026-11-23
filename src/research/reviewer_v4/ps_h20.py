import numpy as np, pandas as pd
from src.research.v4_premium_shocks.common import nw_ols
p = pd.read_pickle("data/processed/research_reviewer_v4/ps_build30.pkl")
w = p[p.wkprem_second.notna() & p.month.isin([11,12,1,2])].copy()
# trading days until the 'second' contract becomes too close to expiry: find index where front==this second and front_days_left
idx = {}
for i,(c) in enumerate(p.second):
    pass
# days to expiry of the contract that is 'second' at d: use the first row where front==that contract, plus its front_days_left
fr = p.groupby("front").apply(lambda g: g.index.max() + g.front_days_left.iloc[-1])
w["dte"] = w.second.map(fr) - w.index
w["inc"] = w.fut20_second - w.fut15_second
d = w.dropna(subset=["fut20_second","fut15_second"])
print("dte quantiles", d.dte.describe().round(1).to_dict())
for lab, m in [("dte<=27", d.dte<=27), ("dte>27", d.dte>27)]:
    dd=d[m]
    for y in ["fut15_second","fut20_second","inc"]:
        b,se=nw_ols(dd[y], dd[["wknews_second","wkprem_second"]],5)
        print(lab, len(dd), y, f"prem {b[2]:+.2f} t {b[2]/se[2]:+.1f}")
