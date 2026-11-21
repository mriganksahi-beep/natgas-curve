"""Gaussian hidden Markov models on weekly March-April spread changes.

Fit on winters 2010-11 to 2016-17 (each winter a separate sequence), then run forward
with FILTERED probabilities only (no smoothing): P(state_t | dS_1..dS_t within the winter).
Each winter restarts from the fitted initial distribution.
"""
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.stats import norm

TRAIN_WINTERS = ["2010-11", "2011-12", "2012-13", "2013-14", "2014-15", "2015-16", "2016-17"]


def fit_hmm(df, k, col="dS", n_starts=30):
    tr = df[df.winter.isin(TRAIN_WINTERS)].sort_values("F")
    X = tr[[col]].values
    lengths = tr.groupby("winter", sort=False).size().loc[tr.winter.unique()].values
    best, best_ll = None, -np.inf
    for seed in range(n_starts):
        m = GaussianHMM(n_components=k, covariance_type="diag", n_iter=500, tol=1e-6,
                        random_state=seed, min_covar=1e-5)
        try:
            m.fit(X, lengths)
            ll = m.score(X, lengths)
        except Exception:
            continue
        if ll > best_ll and np.all(np.isfinite(m.transmat_)):
            best, best_ll = m, ll
    # order states by variance: 0 = calmest
    order = np.argsort(best.covars_.ravel())
    n_par = k * (k - 1) + (k - 1) + 2 * k  # transitions, initial, mean+var
    return dict(model=best, order=order, ll=best_ll, n=len(X),
                bic=-2 * best_ll + n_par * np.log(len(X)), aic=-2 * best_ll + 2 * n_par)


def params_table(fit):
    m, o = fit["model"], fit["order"]
    sd = np.sqrt(m.covars_.ravel())[o]
    mu = m.means_.ravel()[o]
    P = m.transmat_[np.ix_(o, o)]
    return pd.DataFrame({"mean_dS": mu, "sd_dS": sd, "stay_prob": np.diag(P),
                         "expected_weeks": 1 / (1 - np.diag(P)), "start_prob": m.startprob_[o]},
                        index=[f"state{i}" for i in range(len(o))]), P


def filtered_probs(df, fit, col="dS"):
    """Forward filter within each winter; returns array (n, k), states ordered calm->stressed.
    Also returns the one-step-ahead (predicted) probability for t made at t-1."""
    m, o = fit["model"], fit["order"]
    mu, sd = m.means_.ravel()[o], np.sqrt(m.covars_.ravel())[o]
    P, pi = m.transmat_[np.ix_(o, o)], m.startprob_[o]
    filt = np.full((len(df), len(o)), np.nan)
    pred = np.full((len(df), len(o)), np.nan)
    df = df.reset_index(drop=True)
    for w, g in df.groupby("winter", sort=False):
        prior = pi.copy()
        for i in g.sort_values("F").index:
            pred[i] = prior
            lik = norm.pdf(df.at[i, col], mu, sd)
            post = prior * lik
            post /= post.sum()
            filt[i] = post
            prior = post @ P
    return filt, pred
