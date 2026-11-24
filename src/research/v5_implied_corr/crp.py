"""Step 1-2 tables: validity audit, CRP statistics, variance decomposition of the CSO vol premium."""
import numpy as np
import pandas as pd

from .common import boot_mean, gas_year
from .rho import build_rows

WINTER = {11, 12, 1, 2, 3}


def prep(s=None):
    s = build_rows() if s is None else s
    s = s.copy()
    s["blk"] = gas_year(s.date)
    s["yr"] = s.date.dt.year
    s["window"] = np.select([s.yr.between(2014, 2016), s.yr.between(2017, 2021)], ["choose", "check"], "pre2014")
    s["group"] = np.where(s.cm == 3, "HJ", np.where(s.cm.isin(WINTER), "winter_other", "summer_shoulder"))
    for v in ("R0", "R1", "R2"):
        s[f"rc_{v}"] = s[f"rho_{v}"].clip(-1, 1)          # capped variant (keeps rho>1 rows at 1)
    return s


def validity(s):
    base = s[s.tdays.between(5, 126)]
    out = []
    for (w, st), g in base.groupby(["window", "stale"]):
        r = {"window": w, "stale_default_settle": st, "n": len(g)}
        r.update(g.cause_R0.value_counts(normalize=True).round(4).add_prefix("R0:").to_dict())
        r.update(g.cause_R1.value_counts(normalize=True).round(4).add_prefix("R1:").to_dict())
        out.append(r)
    allr = s.copy()
    tot = {"window": "ALL rows (any T)", "n": len(allr), "stale_share": round(allr.stale.mean(), 4),
           "panel_rho_valid_false": round((~allr.rho_valid).mean(), 4)}
    inv = allr[~allr.rho_valid]
    tot.update(inv.cause_R0.value_counts(normalize=True).round(4).add_prefix("invalid_cause:").to_dict())
    tot["invalid_and_stale"] = round(inv.stale.mean(), 4)
    # what distinguishes rho>1 rows: nv vs |sn - sf|, n_strikes, tdays
    g1 = base[(base.cause_R0 == "rho_gt_1 (nv<|sn-sf|)") & ~base.stale]
    ok = base[(base.cause_R0 == "ok") & ~base.stale]
    diag = {"gt1_median_tdays": g1.tdays.median(), "ok_median_tdays": ok.tdays.median(),
            "gt1_median_nstrikes": g1.n_strikes.median(), "ok_median_nstrikes": ok.n_strikes.median(),
            "gt1_median_nv": g1.nv.median(), "gt1_median_abs_sn_sf": g1.dn_minus_df.median(),
            "gt1_share_2020_21": (g1.yr >= 2020).mean(), "gt1_share_cm3": (g1.cm == 3).mean()}
    return pd.DataFrame(out), tot, diag


def crp_table(s, tmin=5, tmax=126):
    rows = []
    base = s[(~s.stale) & s.tdays.between(tmin, tmax)]
    for v in ("R0", "R1", "R2"):
        for mode in ("drop", "cap"):
            if mode == "drop":
                b = base[base[f"rho_{v}"].between(-1, 1)].copy()
                ri = b[f"rho_{v}"]
            else:
                b = base[base[f"rho_{v}"].notna() & (base[f"rho_{v}"] >= -1)].copy()
                ri = b[f"rc_{v}"]
            b["crp_life"] = b.rho_r_life - ri
            b["crp20"] = b.rho_r20 - ri
            b["ri"] = ri
            for grp, gsel in (("all", None), ("winter_pairs", b.cm.isin(WINTER)), ("HJ", b.cm == 3)):
                g = b if gsel is None else b[gsel]
                for w in ("all", "choose", "check"):
                    gw = g[g.yr >= 2012] if w == "all" else g[g.window == w]
                    for tgt in ("crp_life", "crp20"):
                        m, lo, hi, nb = boot_mean(gw, tgt)
                        rows.append({"variant": v, "mode": mode, "group": grp, "window": w, "target": tgt,
                                     "n": int(gw[tgt].notna().sum()), "blocks": nb, "mean": m, "lo": lo, "hi": hi,
                                     "median": gw[tgt].median(), "share_pos": (gw[tgt] > 0).mean(),
                                     "rho_impl_med": gw.ri.median(),
                                     "rho_real_med": gw.rho_r_life.median() if tgt == "crp_life" else gw.rho_r20.median()})
    return pd.DataFrame(rows)


def by_cuts(s, v="R1"):
    b = s[(~s.stale) & s.tdays.between(2, 126) & s[f"rho_{v}"].notna() & (s[f"rho_{v}"] >= -1)].copy()
    b["ri"] = b[f"rc_{v}"]
    b["crp"] = b.rho_r_life - b.ri
    b["crp20"] = b.rho_r20 - b.ri
    b["dec_ratio"] = (1 - b.ri) / (1 - b.rho_r_life).clip(lower=1e-4)
    out = {}
    for name, key in (("bucket", "bucket"), ("cm", "cm"), ("year", "yr")):
        rows = []
        for k, g in b.groupby(key):
            m, lo, hi, nb = boot_mean(g, "crp")
            rows.append({key: k, "n": len(g), "rho_impl": g.ri.median(), "rho_real_life": g.rho_r_life.median(),
                         "rho_r20": g.rho_r20.median(), "crp_mean": m, "lo": lo, "hi": hi, "crp_med": g.crp.median(),
                         "share_pos": (g.crp > 0).mean(), "impl_decoupling_1mr": (1 - g.ri).median(),
                         "real_decoupling_1mr": (1 - g.rho_r_life).median()})
        out[name] = pd.DataFrame(rows)
    return out, b


def var_decomp(s, v="R1", tmin=5, tmax=126):
    """nv^2 - RV_S^2 = (sn^2 - RVn^2) + (sfh^2 - RVf^2) - 2 (rho_i sn sfh - cov_r), per year of variance.

    Only rows where rho_i is a valid correlation are used so the identity is exact.
    """
    b = s[(~s.stale) & s.tdays.between(tmin, tmax) & s[f"rho_{v}"].between(-1, 1) & s.vS_life.notna()].copy()
    T = b.n_life / 252.0
    sfh = b[f"sfh_{v}"]
    b["vrp_S"] = b.nv ** 2 - b.vS_life / T
    b["t_n"] = b.sn ** 2 - b.vn_life / T
    b["t_f"] = sfh ** 2 - b.vf_life / T
    b["t_c"] = -2 * (b[f"rho_{v}"] * b.sn * sfh - b.cv_life / T)
    b["resid"] = b.vrp_S - b.t_n - b.t_f - b.t_c
    rows = []
    for grp, g0 in (("all", b), ("winter_pairs", b[b.cm.isin(WINTER)]), ("HJ", b[b.cm == 3])):
        for w in ("all", "choose", "check"):
            g = g0[g0.yr >= 2012] if w == "all" else g0[g0.window == w]
            r = {"group": grp, "window": w, "n": len(g)}
            for c in ("vrp_S", "t_n", "t_f", "t_c"):
                m, lo, hi, _ = boot_mean(g, c)
                r.update({c: m, c + "_lo": lo, c + "_hi": hi})
            r["nv2_mean"] = (g.nv ** 2).mean()
            r["rv_S2_mean"] = (g.vS_life / (g.n_life / 252)).mean()
            rows.append(r)
    return pd.DataFrame(rows)
