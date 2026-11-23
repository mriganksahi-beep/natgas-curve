set -e
pip -q install pandas numpy requests pygrib
nproc
python - <<'PY'
import pandas as pd, time
from pathlib import Path
from src.collect import gefs
t=time.time(); gefs.collect(list(pd.date_range("2005-01-03","2005-01-14")), "reforecast", Path("/tmp/t.csv"))
print("12 reforecast runs", f"{time.time()-t:.0f}s"); print(pd.read_csv("/tmp/t.csv").head(3).to_string())
PY
