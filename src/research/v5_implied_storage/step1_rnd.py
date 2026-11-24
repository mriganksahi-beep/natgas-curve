"""Step 1: daily risk-neutral densities of S_T (H-J at the March option expiry), Nov 1 .. option expiry.
Writes data/processed/research_v5_implied_storage/rnd_daily.pkl (one row per trading day and Y)."""
import time

import numpy as np
import pandas as pd

from src.research.v5_implied_storage import data as D, rnd as R

KS_NP = np.round(np.arange(-0.25, 1.501, 0.05), 2)


def run(force=False):
    p = D.CACHE / "rnd_daily.pkl"
    if p.exists() and not force:
        return pd.read_pickle(p)
    o = D.load_options()
    fut = D.load_futures()
    ex = D.expiries(fut)
    t0 = time.time()
    allrows = []
    for Y in range(2013, 2023):
        py = D.CACHE / f"rnd_daily_{Y}.pkl"
        if py.exists() and not force:
            allrows.append(pd.read_pickle(py))
            continue
        rows, warm = [], None
        d = fut[Y]
        oe = ex.oexp[Y]
        days = d.index[(d.index >= pd.Timestamp(Y - 1, 11, 1)) & (d.index < oe) & (d.index <= D.CUT)]
        oy = o[o.Y == Y]
        g = dict(tuple(oy.groupby("date")))
        for t in days:  # noqa
            if t not in g:
                continue
            day = g[t]
            F = d.S[t]
            K, px, isc = R.prep_quotes(day, F)
            if len(K) == 0:
                continue
            f = R.fit_day(K, px, isc, F, warm=warm)
            warm = f["p"]
            bv, bn = R.butterfly_violations(K, px, isc, F)
            npt = R.nonparam_tail(K, px, isc, F, KS_NP) if len(K) >= 15 else np.full(len(KS_NP), np.nan)
            rows.append({"date": t, "Y": Y, "F": F, "F_par": R.parity_F(day), "T": (d.index.searchsorted(oe) -
                         d.index.searchsorted(t)) / 252.0, "nq": f["nq"], "ncomp": f["n"], "rmse": f["rmse"],
                         "maxerr": f["maxerr"], "rel_rmse": f["rel_rmse"], "bfly_viol": bv, "bfly_n": bn,
                         "kmin": K.min(), "kmax": K.max(), "theta": f["theta"], "np_tail": npt,
                         "quotes": (K, px, isc)})
        pd.DataFrame(rows).to_pickle(py)
        allrows.append(pd.DataFrame(rows))
        print(Y, len(rows), f"{time.time() - t0:.0f}s", flush=True)
    df = pd.concat(allrows, ignore_index=True)
    df.to_pickle(p)
    return df


if __name__ == "__main__":
    run()
