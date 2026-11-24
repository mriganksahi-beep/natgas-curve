"""Winter Squeeze-Risk Monitor - integration of links A-D of reports/tool/SPEC.md (orchestrator, 24 Nov 2026).

One reading = one settlement date `as_of` and one winter Y (the March contract year):

  1. Market leg (Q).  Risk-neutral density of the March-April spread at the March option expiry, S_T:
     the direct G4X calendar-spread-option density (src/tool/rnd_direct.py, agent A's saved engine,
     verified in reports/tool/integration.md) when it is usable and has >= 3 strikes with previous-day open
     interest; else the composite H x J copula density (src/tool/rnd_composite.py) - the `select` rule.
  2. Physical leg (P).  The analog storage model (src/tool/physical.py): end-March storage samples from the
     last EIA week + realised weather + GEFS + analog years. It is translated into spread space with the
     forward map S_T | x ("hetero_exp", fit on winters <= Y-1), so P and Q are compared on the SAME variable.
  3. Storage lens.  Both legs are read in storage units through the SAME inverse map x | S_T ("inv_asinh",
     fit on winters <= Y-1). Pushing P through the same lens matters: the map is blurry (about +/-450 Bcf),
     and comparing a blurred Q reading with an unblurred P forecast manufactures a fake "premium" (the v5
     error). Headline numbers:
       Q_squeeze  = P_Q(S_T > S_STAR)              market odds of a squeeze at expiry
       P_squeeze  = P_P(S_T > S_STAR)              physical odds of the same event
       Q_def400   = P_Q(x < -400) through the lens   "the options imply a 400+ Bcf deficit with prob ..."
       P_def400   = P_P(x < -400) through the lens   same lens, physical input
       P_def400_raw = physical model's own P(x < -400) (no lens; shown for reference only)

x convention: end-March US working gas from the EIA MONTHLY survey minus the mean of the 5 prior end-March
monthly levels (the map's convention). The physical model works in weekly levels; it is converted with the
weekly-minus-monthly offset estimated on the overlap (about -20 Bcf; see _map_data.wm_offset).

Data cut: set RND_DATA_CUTOFF (read by A, B and _map_data) before importing for locked/live runs.
"""
from functools import lru_cache

import numpy as np
import pandas as pd

from src.tool import _map_data as MD
from src.tool import physical as PH
from src.tool import rnd_composite as RB
from src.tool import rnd_direct as RA
from src.tool import storage_map as SM

S_STAR = 0.25          # $/MMBtu: a "squeeze" = March settles 25c+ over April at the March option expiry
X_STAR = -400          # Bcf: the map's resolvable deficit threshold
S_GRID = np.linspace(-1.0, 4.0, 2001)


@lru_cache(maxsize=None)
def maps(last_winter: int, B: int = 200):
    return SM.fit_map(last_winter, B=B), SM.fit_map(last_winter, spec="hetero_exp", B=B)


@lru_cache(maxsize=None)
def wm_offset() -> float:
    em_w = MD.weekly_storage().to_dict()
    wg = MD.monthly_storage()["wg"]
    return float(np.nanmean([em_w[Y] - wg.get(Y * 100 + 3, np.nan) for Y in em_w]))


def avg5_monthly(Y: int) -> float:
    wg = MD.monthly_storage()["wg"]
    v = [wg.get((Y - k) * 100 + 3, np.nan) for k in range(1, 6)]
    return float(np.mean(v))


def realised(Y: int):
    """Realised (S_T, x) for winter Y, if inside the data cut; else (nan, nan)."""
    try:
        t = MD.winter_table()
    except Exception:
        return np.nan, np.nan
    if Y not in t.index:
        return np.nan, np.nan
    return float(t.at[Y, "S"]), float(t.at[Y, "x"])


def market_density(as_of: str, Y: int):
    a = RA.spread_density(as_of, Y)
    try:
        b = RB.spread_density_composite(as_of, Y)
    except Exception:
        b = None
    d, route = RB.select(a, b)
    return d, route, a, b


def physical_spread_cdf(x_samples, mf, grid=S_GRID):
    """CDF of S_T implied by physical x samples through the forward map (mixture of normals, draws averaged)."""
    f, s = SM.forward(mf, x_samples, draws=True)               # [draw, sample]
    z = (grid[None, None, :] - f[..., None]) / s[..., None]
    from scipy.special import ndtr
    cdf = ndtr(z).mean(axis=(0, 1))
    return grid, cdf


def reading(as_of: str, Y: int = None, B: int = 200, n_phys: int = 400) -> dict:
    as_of = pd.Timestamp(as_of)
    Y = Y or (as_of.year + 1 if as_of.month >= 4 else as_of.year)
    d, route, _, _ = market_density(as_of.strftime("%Y-%m-%d"), Y)
    out = {"as_of": as_of.date(), "Y": Y, "route": route}
    if d is None or len(d.get("grid", [])) == 0:
        out["usable"] = False
        return out
    mi, mf = maps(Y - 1, B)
    grid, cdf = np.asarray(d["grid"]), np.asarray(d["cdf"])
    imp = SM.implied_storage((grid, cdf), mi)
    # physical leg, in the map's monthly convention
    base = avg5_monthly(Y) + wm_offset()
    ph = PH.physical_dist(as_of, baseline_bcf=base)
    xs = np.asarray(ph["samples"])
    xs_thin = xs[np.linspace(0, len(xs) - 1, min(n_phys, len(xs))).astype(int)]
    g, cdf_p = physical_spread_cdf(xs_thin, mf)
    lens_p = SM.implied_storage((g, cdf_p), mi)
    Fq = float(d["F"])
    out.update({
        "usable": True, "F": Fq, "T": float(d.get("T", np.nan)),
        "Q_squeeze": float(1 - np.interp(S_STAR, grid, cdf, left=0, right=1)),
        "P_squeeze": float(1 - np.interp(S_STAR, g, cdf_p)),
        "Q_mean_S": float(np.trapezoid(grid * np.gradient(cdf, grid), grid)),
        "P_mean_S": float(np.trapezoid(g * np.gradient(cdf_p, g), g)),
        "Q_def400": imp["P"][X_STAR], "Q_def400_band": imp["P_band"][X_STAR],
        "P_def400": lens_p["P"][X_STAR], "P_def400_raw": ph["prob_below"](X_STAR),
        "Q_x_q": imp["q"], "P_x_q": {k: float(v) for k, v in ph["quantiles"].items()},
        "baseline_monthly": avg5_monthly(Y),
    })
    out["squeeze_premium"] = out["Q_squeeze"] - out["P_squeeze"]
    out["deficit_premium"] = out["Q_def400"] - out["P_def400"]
    S_T, x = realised(Y)
    out.update({"S_T": S_T, "x_real": x})
    return out


def friday_dates(Y: int):
    """Weekly Friday as-of dates of winter Y: first Friday of November (Y-1) to 15 February Y."""
    d = pd.date_range(f"{Y-1}-11-01", f"{Y}-02-15", freq="W-FRI")
    return [x.strftime("%Y-%m-%d") for x in d]


def panel(winters, **kw) -> pd.DataFrame:
    rows = []
    for Y in winters:
        for dte in friday_dates(Y):
            # Fridays that are holidays: step back to the previous business day
            for back in range(0, 4):
                dd = (pd.Timestamp(dte) - pd.offsets.BDay(back)).strftime("%Y-%m-%d") if back else dte
                r = reading(dd, Y, **kw)
                if r.get("usable"):
                    break
            r = {k: v for k, v in r.items() if not isinstance(v, (dict, tuple))}
            rows.append(r)
    return pd.DataFrame(rows)
