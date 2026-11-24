"""Steps 2-3: storage-implied tail probabilities from the RND (walk-forward payout map) vs physical models.

Map (v3 form, src/research/widowmaker_insurance/payout.py): y = a + b exp(-x/tau), fitted by profile LS.
Primary target: H-J settle on the OPTION expiry day (what G4X March settles on); v3's futures-LTD
settle is a sensitivity. Walk-forward: winter Y uses winters 2011..Y-1, at least 4 of them (Y >= 2015).

Event equivalence (monotone map, no noise):  {x < x0}  <=>  {S_T > y(x0)}, so
    P_Q(x < x0) = P_Q(S_T > y_Y(x0)) = sum_i w_i Phi((m_i - y_Y(x0)) / s_i).
Noise-aware comparison in spread space: P_P(S_T > y(x0)) = mean_s Phi((y(x_s) - y(x0)) / sigma_e).
A threshold is "informative" if y(x0) - y(+300) >= sigma_e (the map separates it from ample storage).
"""
import numpy as np
import pandas as pd
from scipy.special import ndtr

from src.research.v5_implied_storage import data as D, physical as P, rnd as R
from src.research.widowmaker_insurance import payout as PO

X0 = [-200, -400, -600, -800]
MIN_PRIOR_WINTERS = 4


def wf_maps(target="S_oexp", min_prior=MIN_PRIOR_WINTERS):
    t = D.payout_table()
    t = t[t[target].notna()]
    out = {}
    for Y in range(2013, 2023):
        tr = t[t.index < Y]
        if len(tr) < min_prior:
            continue
        a, b, tau = PO.fit(tr.x.values, tr[target].values)
        e = tr[target].values - (a + b * np.exp(-tr.x.values / tau))
        sig = float(np.sqrt((e ** 2).sum() / max(len(e) - 3, 1)))
        out[Y] = {"a": a, "b": b, "tau": tau, "sig": sig, "n": len(tr)}
    return out


def y_of(m, x):
    return m["a"] + m["b"] * np.exp(-np.asarray(x, float) / m["tau"])


def x_of(m, y):
    """Inverse map; NaN where y <= a (flat region: no storage level produces it)."""
    z = (np.asarray(y, float) - m["a"]) / m["b"] if m["b"] > 0 else np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(z > 0, -m["tau"] * np.log(z), np.nan)


def physical_scenarios(dd):
    """For each decision row: P-lite levels and Model 2 levels (if cached), plus the 5-yr path."""
    p = D.CACHE / "physical_scen.pkl"
    if p.exists():
        return pd.read_pickle(p)
    pl = P.PLite()
    m2 = P.m2_scenarios()
    s = D.load_storage()
    rows = {}
    for _, r in dd.iterrows():
        F = r.origin
        sim = pl.simulate(F)
        # 5-year path: level at F + mean remaining change from the same date in prior years (<= 5, >= 1)
        T = sim["target"]
        rem = []
        for k in range(1, 6):
            a0 = s.index[s.index.searchsorted(F - pd.DateOffset(years=k)) - 0]
            a1 = s.index[min(s.index.searchsorted(T - pd.DateOffset(years=k)), len(s) - 1)]
            if abs((a0 - (F - pd.DateOffset(years=k))).days) <= 3 and abs((a1 - (T - pd.DateOffset(years=k))).days) <= 3:
                rem.append(s[a1] - s[a0])
        bench = s[F] + np.mean(rem) if rem else np.nan
        rows[F] = {"plite": sim["levels"], "bench5": bench, "n_rem": len(rem),
                   "m2": m2[F][0] if F in m2 else None, "m2_bench": m2[F][1] if F in m2 else np.nan}
    pd.to_pickle(rows, p)
    return rows


def recentre(levels, bench):
    return levels - np.median(levels) + bench


def build_panel(rnd=None):
    dd = D.decision_dates()
    rnd = pd.read_pickle(D.CACHE / "rnd_daily.pkl") if rnd is None else rnd
    rnd = rnd.set_index(["date", "Y"])
    av = D.avg5_table()
    ex = D.expiries()
    maps = wf_maps()
    maps_ltd = wf_maps("S_ltd")
    phys = physical_scenarios(dd)
    rows = []
    for _, r in dd.iterrows():
        key = (r.obs, r.Y)
        if key not in rnd.index:
            continue
        q = rnd.loc[key]
        th = q.theta
        o = {"obs": r.obs, "origin": r.origin, "Y": r.Y, "wte": r.wte, "T": r["T"], "F": q.F, "nq": q.nq,
             "ncomp": q.ncomp, "rmse": q.rmse, "storage": r.storage, "avg5": av.avg5[r.Y],
             "x_real": av.x.get(r.Y, np.nan), "S_T": ex.S_oexp[r.Y] if r.Y in ex.index else np.nan,
             "window": "choose" if r.obs.year <= 2016 else "check"}
        if r.Y == 2022:
            o["x_real"] = np.nan        # end-March 2022 storage is never read
        # PIT of the realised option-expiry spread under Q
        o["pit_Q"] = float(R.mix_cdf(th, [o["S_T"]])[0]) if np.isfinite(o["S_T"]) else np.nan
        ph = phys[r.origin]
        scen = {"pl_raw": ph["plite"], "pl_rc": recentre(ph["plite"], ph["bench5"])}
        if ph["m2"] is not None:
            scen["m2_raw"] = ph["m2"]
            scen["m2_rc"] = recentre(ph["m2"], ph["m2_bench"])
        o["bench5"] = ph["bench5"]
        o["x_bench5"] = ph["bench5"] - o["avg5"]
        for nm, lv in scen.items():
            xs = lv - o["avg5"]
            o[f"xmed_{nm}"] = float(np.median(xs))
            if np.isfinite(o["x_real"]):
                o[f"pitx_{nm}"] = float((xs < o["x_real"]).mean())
            for x0 in X0:
                o[f"P_{nm}_{-x0}"] = float((xs < x0).mean())
        for tag, mp in (("", maps), ("ltd_", maps_ltd)):
            if r.Y not in mp:
                continue
            m = mp[r.Y]
            o[f"{tag}map_n"] = m["n"]
            for x0 in X0:
                k = float(y_of(m, x0))
                o[f"{tag}k_{-x0}"] = k
                o[f"{tag}inform_{-x0}"] = bool(k - y_of(m, 300) >= m["sig"])
                o[f"{tag}Q_{-x0}"] = float(R.mix_tail(th, [k])[0])
                if tag == "":
                    np_t = np.interp(k, np.round(np.arange(-0.25, 1.501, 0.05), 2), q.np_tail) \
                        if np.isfinite(q.np_tail).any() else np.nan
                    o[f"Qnp_{-x0}"] = float(np_t)
                    # noise-aware physical probability in spread space
                    for nm, lv in scen.items():
                        ys = y_of(m, lv - o["avg5"])
                        o[f"Psp_{nm}_{-x0}"] = float(ndtr((ys - k) / m["sig"]).mean())
            if tag == "":
                # implied quantiles of x in the invertible tail
                grid = np.linspace(-1.0, 2.5, 3501)
                tail = R.mix_tail(th, grid)
                for qq in (0.05, 0.10, 0.25, 0.50):
                    kq = grid[np.argmin(np.abs(tail - qq))]
                    o[f"xq_Q_{int(qq * 100)}"] = float(x_of(m, kq))
                    if "pl_rc" in scen:
                        o[f"xq_pl_rc_{int(qq * 100)}"] = float(np.quantile(scen["pl_rc"] - o["avg5"], qq))
                    if "m2_rc" in scen:
                        o[f"xq_m2_rc_{int(qq * 100)}"] = float(np.quantile(scen["m2_rc"] - o["avg5"], qq))
                # actuarial value of the spread under P (noise-aware) vs F
                for nm, lv in scen.items():
                    o[f"AV_{nm}"] = float(y_of(m, lv - o["avg5"]).mean())
                o["pit_P_sp_pl_rc"] = float(ndtr((o["S_T"] - y_of(m, scen["pl_rc"] - o["avg5"])) / m["sig"]).mean()) \
                    if np.isfinite(o["S_T"]) else np.nan
        rows.append(o)
    df = pd.DataFrame(rows)
    df.to_csv(D.CACHE / "panel_weekly.csv", index=False)
    return df, maps, maps_ltd
