set -e
pip -q install pandas numpy requests pygrib
python - <<'PY'
import pandas as pd, time
from src.collect import gefs
for src, d in [("reforecast","2005-01-10"),("operational","2020-01-15"),("operational","2023-01-10")]:
    t=time.time(); r=gefs.fetch_run(pd.Timestamp(d), src)
    print(src, d, f"{time.time()-t:.1f}s", {m: len(v) for m, v in r.items()})
PY
