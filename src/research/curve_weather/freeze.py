"""Freeze the holdout specification: calibrated wx2 ~ 1 + r_front + a0 and wx2 ~ 1 + a0,
fitted once on all panel weeks 2010-11 to Dec 2021 (data <= 2021-12-31 only)."""
import json
import numpy as np
from . import data

df = data.build()
assert df.d0.max() <= data.CUTOFF
out = {}
for name, xs in (("front+pers", ["r_front", "a0"]), ("pers", ["a0"]), ("front", ["r_front"])):
    d = df.dropna(subset=xs + ["wx2"])
    X = np.column_stack([np.ones(len(d))] + [d[x] for x in xs])
    b, *_ = np.linalg.lstsq(X, d.wx2.values, rcond=None)
    out[name] = {"features": xs, "coef": dict(zip(["const"] + xs, map(float, b))), "n": int(len(d))}
json.dump(out, open(data.CACHE / "frozen_holdout_coefs.json", "w"), indent=1)
print(json.dumps(out, indent=1))
