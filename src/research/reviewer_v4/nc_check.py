import numpy as np, pandas as pd, json
from src.research.v4_news_channel.decompose import decompose, decompose_weekly, load_dev_panel, _self_test, fit_frozen, SPEC, FROZEN
p = load_dev_panel()
_self_test(p)
# frozen coefs reproducible from <=2021 data only?
fz = json.loads(FROZEN.read_text())
for ins in ["r_front","r_second","d_hj"]:
    f = fit_frozen(p, ins); print(ins, np.allclose(f["slopes"], fz[ins]["slopes"]), f["last_winter"])
# OOS R2 recompute
def r2(df): 
    e1=df.move-df.intercept-df.news_part; e0=df.move-df.train_mean; return 1-(e1**2).sum()/(e0**2).sum()
for ins in ["r_front","d_hj"]:
    d = decompose(p, ins); d=d[d.in_core & d.news_part.notna()]
    print(ins, "choose", round(r2(d[d.winter.between(2013,2017)]),3), "check", round(r2(d[d.winter.between(2018,2022)]),3),
          "per-winter", {w: round(r2(g),3) for w,g in d.groupby("winter")})
# Locked-column feasibility: emulate locked panel columns on dev and run frozen mode
cols = ["date","prev_date","front","second","front_days_left","px_front","r_front","px_second","r_second","hj_year","hj","d_hj","hj_days_left"]
cols += [c for c in p.columns if c.startswith(("lvl_w","news_w","real_")) or c.endswith("_c00") or c in ("real_d","gefs_system")]
q = p[cols].copy()
for ins in ["r_front","r_second","d_hj"]:
    a=decompose(q,ins,mode="frozen"); b=decompose_weekly(q,ins,mode="frozen"); print("locked-cols ok", ins, len(a), len(b))
from src.research.v4_premium_shocks.frozen_test import hps1, hps1b, hps2
print("hps1", {k:(round(v,3) if isinstance(v,float) else v) for k,v in hps1(q).items()})
print("hps2", hps2(q))
