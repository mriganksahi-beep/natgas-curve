import pandas as pd, numpy as np, json
import statsmodels.api as sm
p = pd.read_csv("data/processed/v4/panel_dev.csv", parse_dates=["date"])
p = p[p.date <= "2021-12-31"].reset_index(drop=True)
c = json.load(open("data/processed/research_v4_news_channel/frozen_coefs.json"))["r_front"]["slopes"]
p["np"] = p.news_w1*c[0]+p.news_w2*c[1]+p.news_w3*c[2]
p["dow"] = p.date.dt.dayofweek
p["r_prev"] = p.r_front.shift(1); p["r_next"] = p.r_front.shift(-1)
# same contract check for shifts
p["same_prev"] = p.front.shift(1)==p.front; p["same_next"]=p.front.shift(-1)==p.front
w = p[p.date.dt.month.isin([11,12,1,2]) & p.np.notna()]
def sl(df, y):
    d = df[[y,"np"]].dropna()
    m = sm.OLS(d[y], sm.add_constant(d.np)).fit(cov_type="HAC", cov_kwds={"maxlags":2})
    return f"{m.params.np:.2f} (t {m.tvalues.np:.1f}, n {len(d)})"
for lab, df in [("news day Mon", w[w.dow==0]), ("news day Tue", w[w.dow==1]), ("news day Wed-Fri", w[w.dow>=2])]:
    print(lab, "k=-1:", sl(df[df.same_prev], "r_prev"), " k=0:", sl(df, "r_front"), " k=+1:", sl(df[df.same_next], "r_next"))
# gap of prev day: news at d covers (dp 00Z, d 00Z]. Fraction of revision before dp settle.
