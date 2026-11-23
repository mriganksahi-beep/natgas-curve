set -e
pip -q install pandas numpy requests pygrib pyshp shapely
python - <<'PY'
import pandas as pd, time
from src.collect import gefs, cpc
for src, d in [("reforecast","2005-01-10"),("operational","2020-01-15"),("operational","2026-11-20")]:
    t=time.time(); v=gefs.run_values(pd.Timestamp(d), src); m=gefs.daily_means(pd.Timestamp(d), v)
    print(src, d, f"{time.time()-t:.1f}s", "hours", v.hour.nunique(), v.hour.min(), v.hour.max(), "days", m.lead_days.min(), m.lead_days.max())
    print(m[m.city=="Chicago"].head(4).to_string()); print(m[m.lead_days==1].head(17)[["city","temp_f"]].T.to_string())
for d in ["2013-01-15","2021-02-05","2026-11-20"]:
    for p in ("610","814"):
        rows=cpc.one(pd.Timestamp(d), p); print(d,p,[(r["city"][:6],r["cat"],r["prob"],r["signed"]) for r in rows][:17])
PY
