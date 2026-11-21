"""Frozen holdout tests H4 and H5b for the kalman_scarcity study (reports/synthesis_v2.md).

    python -m src.holdout.kalman_scarcity --end 2021-12-31   # check: must reproduce the validation numbers
    python -m src.holdout.kalman_scarcity --end 2026-11-17   # the single holdout run

Nothing in the specification is changed. The validation code in src/research/kalman_scarcity
is imported unchanged; only its END cutoff and cache folder are overridden at run time
(the original cache is never touched).

H4  : Nov-Mar Fridays with surplus_known, holdout sample = Fridays >= 2022-01-07.
      D = corr(s_t, surplus_known_t) - corr(MarApr_t, surplus_known_t), 95% winter-block bootstrap
      (blocks = target winter, 2,000 draws). Pass: D < 0 and CI upper < 0.
H5b : corr(dS_t, surprise_pub(t+3)) on Nov-Mar weeks, same sample and bootstrap.
      Index: surprise_pub(t+k) is the storage surprise released in the week ending at Friday t+k,
      i.e. the EIA week ending F + 7(k-1). k=+1 is the report for the same physical week as the
      price move (out at F+6); k=+3 is two reports later (week ending F+14). Pass: <= -0.10 and CI upper < 0.
Also: lags -2..+5, the weather-controlled lead-3 regression, and the scarcity-state figure through END.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

import src.research.kalman_scarcity.data as D
from src.research.kalman_scarcity.kf import OneFactorKF

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "data/processed/research_kalman_scarcity"
HOLDOUT_START = pd.Timestamp("2022-01-07")
NBOOT = 2000
SEED = 20261121
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a85"


def winter_boot(df, stat, rng, n=NBOOT):
    g = df["winter"].values
    ug = np.unique(g)
    idx = {u: np.where(g == u)[0] for u in ug}
    pt = stat(df)
    bs = []
    for _ in range(n):
        pick = rng.choice(ug, len(ug), replace=True)
        rows = np.concatenate([idx[u] for u in pick])
        try:
            bs.append(stat(df.iloc[rows]))
        except Exception:
            pass
    lo, hi = np.nanpercentile(np.array(bs, float), [2.5, 97.5])
    return float(pt), float(lo), float(hi)


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return np.corrcoef(a[m], b[m])[0, 1]


def hac(y, X, lags):
    d = pd.concat([y, X], axis=1).dropna()
    return sm.OLS(d.iloc[:, 0], sm.add_constant(d.iloc[:, 1:])).fit(cov_type="HAC", cov_kwds={"maxlags": lags})


def build(end):
    D.END = pd.Timestamp(end)
    D.CACHE = FROZEN / f"holdout_end_{pd.Timestamp(end).date()}"
    D.CACHE.mkdir(parents=True, exist_ok=True)
    P = D.build_panel(force=True)
    theta = np.array(json.loads((FROZEN / "kf_params.json").read_text())["rw"]["theta"])
    m = OneFactorKF(ar1=False)
    Y = P[D.NAMES].values
    T = P[["tau_" + n for n in D.NAMES]].values
    F = m.filter(theta, Y, T, P["roll"].values)
    P["s_rw"] = F["s"]
    for c in ("s_rw", "MarApr"):
        d = P[c].diff()
        d[P["roll"].values] = np.nan
        P["d_" + c] = d
    P["season"] = P["date"].dt.month.isin([11, 12, 1, 2, 3])
    return P


def evaluate(P, sample_mask, label, rng):
    out = {"sample": label}
    S = P[sample_mask & P["surplus_known"].notna() & P["season"]]
    out["H4_n_weeks"], out["H4_winters"] = len(S), sorted(map(int, S.winter.unique()))
    out["H4_corr_KF"] = winter_boot(S, lambda d: corr(d.s_rw.values, d.surplus_known.values), rng)
    out["H4_corr_MarApr"] = winter_boot(S, lambda d: corr(d.MarApr.values, d.surplus_known.values), rng)
    out["H4_D"] = winter_boot(S, lambda d: corr(d.s_rw.values, d.surplus_known.values)
                              - corr(d.MarApr.values, d.surplus_known.values), rng)
    out["H4_pass"] = bool(out["H4_D"][0] < 0 and out["H4_D"][2] < 0)
    lags = {}
    for lag in (-2, -1, 0, 1, 2, 3, 4, 5):
        for c in ("s_rw", "MarApr"):
            X = pd.DataFrame({"d": P["d_" + c], "sur": P["surprise_pub"].shift(-lag), "winter": P["winter"]})
            X = X[sample_mask & P["season"]].dropna()
            lags[f"{c} lag{lag:+d}"] = winter_boot(X, lambda d: corr(d.d.values, d.sur.values), rng) + (len(X),)
    out["lags"] = lags
    out["H5b"] = lags["s_rw lag+3"]
    out["H5b_pass"] = bool(out["H5b"][0] <= -0.10 and out["H5b"][2] < 0)
    # weather-controlled lead-3 regression (as in the validation report), Nov-Mar, HAC 4
    X = pd.DataFrame({"sur3": P["surprise_pub"].shift(-3), "h0": P["hdd_anom"], "h1": P["hdd_anom"].shift(-1),
                      "h2": P["hdd_anom"].shift(-2), "next_sur": P["surprise_pub"].shift(-1),
                      "dx": P["d_s_rw"] / P["s_rw"][P["winter"] <= 2017].std()})[sample_mask & P["season"]]
    reg = {}
    for name, y, cols in (("lead3 no weather", "sur3", ["dx"]), ("lead3 + realised HDD t..t+2", "sur3", ["h0", "h1", "h2", "dx"]),
                          ("next report + HDD t", "next_sur", ["h0", "dx"])):
        r = hac(X[y], X[cols], 4)
        reg[name] = {"coef_bcf_per_sd": float(r.params["dx"]), "se": float(r.bse["dx"]),
                     "t": float(r.tvalues["dx"]), "n": int(r.nobs)}
    out["weather_regs"] = reg
    return out


def figure(P, end):
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    R = json.loads((FROZEN / "results.json").read_text())["markov"]
    pr = R["params"]
    params = np.array([pr["p[0->0]"], pr["p[1->0]"], pr["const"], pr["sigma2[0]"], pr["sigma2[1]"]])
    ms = MarkovRegression(P["d_s_rw"].fillna(0.0).values, k_regimes=2, trend="c", switching_trend=False,
                          switching_variance=True)
    p_stress = ms.filter(params).filtered_marginal_probabilities[:, R["stressed_regime"]]
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})
    x = P["date"]
    ax[0].plot(x, P["s_rw"], color=BLUE, lw=1.2, label="Filtered scarcity s_t (frozen RW params)")
    ax[0].plot(x, P["MarApr"], color=ORANGE, lw=1.0, label="Mar-Apr spread of the target winter")
    ax[0].axvline(pd.Timestamp("2017-02-24"), color="#333", lw=0.8)
    ax[0].axvline(HOLDOUT_START, color="#333", lw=1.4)
    ax[0].axvspan(HOLDOUT_START, x.max(), color="#f1efe8", lw=0, zorder=0)
    yl = ax[0].get_ylim()
    ax[0].text(pd.Timestamp("2017-03-10"), yl[1] * 0.93, "params frozen", fontsize=8)
    ax[0].text(HOLDOUT_START + pd.Timedelta(days=20), yl[1] * 0.93, "holdout (scored once)", fontsize=8)
    for a, b, n in (("2014-01-03", "2014-03-07", "Jan 2014"), ("2018-11-02", "2018-12-07", "Nov 2018"),
                    ("2021-02-05", "2021-02-26", "Uri")):
        ax[0].axvspan(pd.Timestamp(a), pd.Timestamp(b), color="#d0d0cc", alpha=0.6, lw=0)
        ax[0].text(pd.Timestamp(a), yl[1] * 0.8, n, fontsize=7.5)
    ax[0].axhline(0, color=GREY, lw=0.6)
    ax[0].set_ylabel("$/MMBtu")
    ax[0].legend(loc="upper left", frameon=False, fontsize=8)
    ax[0].set_title(f"Filtered market-expected scarcity vs Mar-Apr, 2010 to {pd.Timestamp(end).date()}", loc="left", fontsize=9.5)
    ax[1].plot(x, p_stress, color=AQUA, lw=1.0)
    ax[1].axvline(HOLDOUT_START, color="#333", lw=1.4)
    ax[1].set_ylim(0, 1)
    ax[1].set_ylabel("P(stressed)")
    ax[1].set_title("Markov-switching filtered P(high-variance regime), frozen 2010-17 parameters", loc="left", fontsize=9)
    fig.tight_layout()
    tag = "check" if pd.Timestamp(end) <= pd.Timestamp("2021-12-31") else "holdout"
    f = ROOT / f"reports/holdout/figures/kalman_scarcity_state_{tag}.png"
    f.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f, dpi=150)
    plt.close(fig)
    return f


def main(end):
    P = build(end)
    assert P["date"].max() <= pd.Timestamp(end)
    rng = np.random.default_rng(SEED)
    res = {"end": str(pd.Timestamp(end).date()), "last_friday": str(P.index.max().date()),
           "theta_source": str(FROZEN / "kf_params.json")}
    if pd.Timestamp(end) <= pd.Timestamp("2021-12-31"):
        res["validation_all"] = evaluate(P, P.index == P.index, "2010-2021 (reproduction)", rng)
    else:
        res["holdout"] = evaluate(P, P.index >= HOLDOUT_START, "holdout 2022-01-07+", rng)
    res["figure"] = str(figure(P, end))
    out = D.CACHE / "holdout_results.json"
    out.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", required=True)
    main(ap.parse_args().end)
