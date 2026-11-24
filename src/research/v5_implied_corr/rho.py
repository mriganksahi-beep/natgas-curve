"""Steps 1-2: implied correlation (three horizon treatments), validity audit, realised correlation, CRP.

rho_impl = (sn^2 + sfh^2 - nv^2) / (2 sn sfh), sn = F_near*iv_near, sfh = F_far*sigma_far_h, where
sigma_far_h is the far leg's Black vol over the NEAR option's life [d, T_n]:

  R0 raw     : sigma_far_h = iv_far (vol to the far leg's own expiry; the panel's rho_impl)
  R1 wf      : forward-variance split  iv_far^2 T_f = sigma_h^2 T_n + sigma_L^2 (T_f - T_n),
               sigma_L (far leg's implied vol in its own last month, observed at the near expiry)
               predicted as q_hat * iv_far(d); q_hat = walk-forward median over COMPLETED past contracts
               of the same month pair and tdays bucket (expanding, refit daily - only uses outcomes
               known before d).                                          <- central estimate
  R2 oracle  : same split with the realised sigma_L (look-ahead; a descriptive bound only).

Realised (zero-mean, normal changes - matches the normal approximation):
  rho_r = sum dFn dFf / sqrt(sum dFn^2 sum dFf^2) over (d, T_n] ("life") and the next 20 days ("20d").
CRP = rho_r - rho_impl (positive => implied too low => spread vol too rich).
"""
import numpy as np
import pandas as pd

from .common import CACHE, futures, spreads, surface

BUCKETS = [0, 21, 42, 63, 126, 400]


def build_rows():
    p = CACHE / "rows.pkl"
    if p.exists():
        return pd.read_pickle(p)
    px, days, oexp = futures()
    s = spreads()
    sf = surface().set_index(["date", "cid"]).atm_iv
    pos = pd.Series(np.arange(len(days)), index=days)
    s["Fn"] = [px.at[d, c] if c in px.columns else np.nan for d, c in zip(s.date, s.cid)]
    s["Ff"] = [px.at[d, c] if c in px.columns else np.nan for d, c in zip(s.date, s.far)]
    s["sn"] = s.Fn * s.iv_near
    s["sf"] = s.Ff * s.iv_far
    s["exp_n"] = s.cid.map(oexp)
    s["exp_f"] = s.far.map(oexp)
    gap = np.array([np.busday_count(a.date(), b.date()) for a, b in zip(s.exp_n, s.exp_f)])
    s["Tn"] = s.tdays / 252.0
    s["Tf"] = s.Tn + gap / 252.0
    # far leg's own last-month implied vol, observed at the near expiry (or nearest prior day, <=3 days)
    last = {}
    for f, en in s[["far", "exp_n"]].drop_duplicates().itertuples(index=False):
        v = np.nan
        for k in range(4):
            i = days.searchsorted(en) - k
            if 0 <= i < len(days) and days[i] <= en:
                v = sf.get((days[i], f), np.nan)
                if np.isfinite(v):
                    break
        last[f] = v
    s["sigL"] = s.far.map(last)
    s["q_real"] = s.sigL / s.iv_far
    s["bucket"] = pd.cut(s.tdays, BUCKETS).astype(str)
    # walk-forward q_hat: contracts whose near expiry < d, same cm & bucket (contract means, median)
    cm_tab = (s.dropna(subset=["q_real"]).groupby(["cid", "cm", "bucket"])
              .agg(q=("q_real", "mean"), exp_n=("exp_n", "first")).reset_index())
    qh = np.full(len(s), np.nan)
    for i, (d, cm, b) in enumerate(zip(s.date, s.cm, s.bucket)):
        h = cm_tab[(cm_tab.exp_n < d) & (cm_tab.bucket == b)]
        hs = h[h.cm == cm]
        qh[i] = hs.q.median() if len(hs) >= 2 else (h.q.median() if len(h) >= 3 else 1.0)
    s["q_hat"] = qh

    def rho(sigh):
        sfh = s.Ff * sigh
        return (s.sn ** 2 + sfh ** 2 - s.nv ** 2) / (2 * s.sn * sfh), sfh

    def sig_h(q):
        v = (s.iv_far ** 2 * s.Tf - (q * s.iv_far) ** 2 * (s.Tf - s.Tn)) / s.Tn
        return np.sqrt(v.where(v > 0))

    s["rho_R0"], s["sfh_R0"] = rho(s.iv_far)
    s["rho_R1"], s["sfh_R1"] = rho(sig_h(s.q_hat))
    s["rho_R2"], s["sfh_R2"] = rho(sig_h(s.q_real))
    # validity cause (R0, as the panel's flag, and R1)
    for v in ("R0", "R1"):
        r = s[f"rho_{v}"]
        cause = np.select([s.iv_near.isna() | s.iv_far.isna() | s.Fn.isna() | s.Ff.isna(),
                           s[f"sfh_{v}"].isna(), r > 1, r < -1],
                          ["missing_leg_iv", "neg_fwd_var", "rho_gt_1 (nv<|sn-sf|)", "rho_lt_-1"], "ok")
        s[f"cause_{v}"] = cause
    s["dn_minus_df"] = (s.sn - s.sf).abs()

    # realised legs
    dF = px.diff()
    life, r20, lg = [], [], []
    for d, c, f, en in zip(s.date, s.cid, s.far, s.exp_n):
        i0 = pos.get(d)
        if c not in dF.columns or f not in dF.columns or en > days[-1]:
            life.append((np.nan,) * 4); r20.append((np.nan,) * 4); lg.append(np.nan)
            continue
        i1 = days.searchsorted(en)
        a = dF[c].values[i0 + 1:i1 + 1]
        b = dF[f].values[i0 + 1:i1 + 1]
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
        life.append((np.sum(a * a), np.sum(b * b), np.sum(a * b), len(a)))
        if i1 - i0 >= 20:
            a2, b2 = dF[c].values[i0 + 1:i0 + 21], dF[f].values[i0 + 1:i0 + 21]
            ok2 = np.isfinite(a2) & np.isfinite(b2)
            a2, b2 = a2[ok2], b2[ok2]
            r20.append((np.sum(a2 * a2), np.sum(b2 * b2), np.sum(a2 * b2), len(a2)))
        else:
            r20.append((np.nan,) * 4)
        la = np.diff(np.log(px[c].values[i0:i1 + 1]))
        lb = np.diff(np.log(px[f].values[i0:i1 + 1]))
        ok3 = np.isfinite(la) & np.isfinite(lb)
        lg.append(np.corrcoef(la[ok3], lb[ok3])[0, 1] if ok3.sum() >= 10 else np.nan)
    L = np.array(life, float)
    R = np.array(r20, float)
    s["vn_life"], s["vf_life"], s["cv_life"], s["n_life"] = L.T
    s["rho_r_life"] = s.cv_life / np.sqrt(s.vn_life * s.vf_life)
    s["rho_r20"] = R[:, 2] / np.sqrt(R[:, 0] * R[:, 1])
    s["vS_life"] = s.vn_life + s.vf_life - 2 * s.cv_life
    s["rho_r_log"] = lg
    s.loc[s.n_life < 5, ["rho_r_life", "rho_r_log"]] = np.nan
    s.to_pickle(p)
    return s


def sample(s, variant="R1", tmin=5, tmax=126):
    """Analysis sample: non-stale, valid under the variant, near-dated (dense) rows."""
    r = s[f"rho_{variant}"]
    ok = (~s.stale) & r.between(-1, 1) & s.tdays.between(tmin, tmax)
    return s[ok]
