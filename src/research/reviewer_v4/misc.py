import numpy as np, pandas as pd, json
from src.research.v4_premium_shocks.common import nw_ols
p = pd.read_pickle("data/processed/research_reviewer_v4/ps_build30.pkl")
c = p[p.core_front==True]
print("corr news_part_d, news_part_d+1 (front):", round(c.news_front.corr(c.news_front.shift(-1)),3), " news_tot autocorr:", round(c.news_tot.corr(c.news_tot.shift(-1)),3))
# HAC t for news-minus-prem slope at weekly 20d: regress fut20 on (move, prem): coef on prem = bP - bN
w = p[p.wkprem_second.notna() & p.month.isin([11,12,1,2])].dropna(subset=["fut20_second"])
b,se = nw_ols(w.fut20_second, w[["wkmove_second","wkprem_second"]],5)
print("bP-bN", round(b[2],3), "HAC t", round(b[2]/se[2],2))
for h in [5,10]:
    d=p[p.wkprem_hj.notna() & p.month.isin([11,12,1,2])].dropna(subset=[f"fut{h}_hj"])
    b,se=nw_ols(d[f"fut{h}_hj"], d[["wkmove_hj","wkprem_hj"]], h//5+1); print("hj",h,"bP-bN",round(b[2],3),"t",round(b[2]/se[2],2))
