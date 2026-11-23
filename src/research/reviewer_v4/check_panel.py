import pandas as pd, numpy as np
p = pd.read_csv("data/processed/v4/panel_dev.csv", parse_dates=["date","prev_date"])
p = p[p.date <= "2021-12-31"].reset_index(drop=True)
print(p.shape, p.date.min(), p.date.max())
print(list(p.columns))
# GEFS raw checks, dev cut
for f in ["gefs_reforecast_ens5.csv.gz","gefs_operational_ens5.csv.gz"]:
    g = pd.read_csv("data/forecasts/"+f, parse_dates=["init_date","valid_et_date"])
    g = g[(g.init_date >= "2010-06-01") & (g.init_date <= "2021-12-31")]
    print(f, g.n_mem.value_counts().to_dict())
    mx = g.groupby("init_date").lead_days.max()
    print(" max lead by weekday:", mx.groupby(mx.index.dayofweek).agg(["min","mean"]).to_dict())
p["dow"] = p.date.dt.dayofweek
w = p[p.date.dt.month.isin([11,12,1,2])]
print(w.groupby(["gefs_system","dow"])[["news_w1","news_w2","news_w3","news_w0"]].std().round(2))
print(w.groupby(["gefs_system","dow"])[["news_w3"]].mean().round(2))
# system switch days
for d in ["2020-01-02","2020-09-23","2020-09-24"]:
    print(p[p.date==d][["date","gefs_system","news_w1","news_w2","news_w3","news_w0"]])
print("news_w3 abs top days:"); print(w.loc[w.news_w3.abs().nlargest(8).index, ["date","dow","gefs_system","news_w3"]])
