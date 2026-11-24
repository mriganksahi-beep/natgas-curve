"""Risk-neutral density of the March-April spread at option expiry from G4X March settles.

Model: S_T ~ mixture of n normals (Bachelier mixture), sum_i w_i m_i = F (martingale, r = 0).
    C(K) = sum_i w_i [(m_i - K) Phi(d_i) + s_i phi(d_i)],  d_i = (m_i - K)/s_i ;  P(K) = C(K) - (F - K)
A mixture of normals is a proper density, so the fitted call curve is automatically monotone
decreasing and convex in K (no-arbitrage), and the Breeden-Litzenberger tail is
    P_Q(S_T > k) = -dC/dK = sum_i w_i Phi((m_i - k)/s_i).
n = 3 on dense days (>= 15 usable quotes), 2 on 5..14, 1 (plain Bachelier, mean F) below 5.
Fitted to out-of-the-money settles (calls K >= F, puts K < F), residuals weighted 1/(price + 0.003).

Non-parametric check (dense days): call curve on the strike grid from OTM settles (puts via parity),
finite-difference slope -dC/dK, isotonic (monotone) projection, clipped to [0, 1].
"""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import ndtr
from scipy.stats import norm

W_EPS = 0.003


def mix_call(theta, K):
    w, m, s = theta
    K = np.asarray(K, float)[:, None]
    d = (m[None, :] - K) / s[None, :]
    return ((m[None, :] - K) * ndtr(d) + s[None, :] * norm.pdf(d)) @ w


def mix_tail(theta, k):
    w, m, s = theta
    k = np.asarray(k, float)[:, None]
    return ndtr((m[None, :] - k) / s[None, :]) @ w


def mix_cdf(theta, k):
    return 1.0 - mix_tail(theta, k)


def _unpack(p, n, F):
    if n == 1:
        return np.array([1.0]), np.array([F]), np.array([np.exp(p[0])])
    z = np.concatenate([[0.0], p[: n - 1]])
    w = np.exp(z - z.max())
    w /= w.sum()
    m = p[n - 1: 2 * n - 1]
    m = m - (w @ m) + F                       # martingale shift
    s = np.exp(p[2 * n - 1: 3 * n - 1])
    return w, m, s


def prep_quotes(day, F, kmax=2.0):
    """OTM quotes for one date: returns K, price, is_call."""
    q = day[(np.abs(day.K - F) <= kmax)]
    c = q[(q.cp == "C") & (q.K >= F)][["K", "settle"]]
    p = q[(q.cp == "P") & (q.K < F)][["K", "settle"]]
    K = np.concatenate([c.K.values, p.K.values])
    px = np.concatenate([c.settle.values, p.settle.values])
    isc = np.concatenate([np.ones(len(c), bool), np.zeros(len(p), bool)])
    o = np.argsort(K)
    return K[o], px[o], isc[o]


def parity_F(day):
    c = day[day.cp == "C"].set_index("K").settle
    p = day[day.cp == "P"].set_index("K").settle
    j = c.index.intersection(p.index)
    if len(j) == 0:
        return np.nan
    diff = (c[j] - p[j])
    k = diff.abs().idxmin()
    return k + diff[k]


def fit_day(K, px, isc, F, n=None, seed=0, warm=None, starts=None):
    nq = len(K)
    if n is None:
        n = 3 if nq >= 15 else (2 if nq >= 5 else 1)
    wts = 1.0 / (px + W_EPS)

    def model(p):
        th = _unpack(p, n, F)
        c = mix_call(th, K)
        return np.where(isc, c, c - (F - K))

    def res(p):
        return (model(p) - px) * wts

    # starting values: ATM normal vol from the quote nearest F
    i = np.argmin(np.abs(K - F))
    s0 = max(px[i] * np.sqrt(2 * np.pi), 0.02)
    rng = np.random.default_rng(seed)
    best = None
    starts = (1 if n == 1 else (4 if warm is not None and len(warm) == 3 * n - 1 else 10)) if starts is None else starts
    for r in range(starts + (warm is not None and len(warm) == 3 * n - 1)):
        if r == starts:
            p0 = warm
        elif n == 1:
            p0 = [np.log(s0)]
        else:
            lw = rng.normal(-1.5, 0.8, n - 1)
            m0 = np.concatenate([[F], F + rng.uniform(0.05, 0.8, n - 1)])
            ls = np.log(np.concatenate([[s0 * rng.uniform(0.4, 1.0)], s0 * rng.uniform(1.5, 6, n - 1)]))
            p0 = np.concatenate([lw, m0, ls])
        try:
            sol = least_squares(res, p0, method="trf", max_nfev=3000)
        except Exception:
            continue
        if best is None or sol.cost < best.cost:
            best = sol
    th = _unpack(best.x, n, F)
    fit = model(best.x)
    err = fit - px
    return {"theta": th, "p": best.x, "n": n, "nq": nq, "rmse": float(np.sqrt(np.mean(err ** 2))),
            "maxerr": float(np.abs(err).max()), "rel_rmse": float(np.sqrt(np.mean((err / (px + W_EPS)) ** 2)))}


def nonparam_tail(K, px, isc, F, ks):
    """Model-free BL tail at points ks from adjacent-strike slopes (call curve), isotonic in K.
    Returns NaN for ks outside the quoted range."""
    C = np.where(isc, px, px + (F - K))       # calls via parity
    if len(K) < 4:
        return np.full(len(ks), np.nan)
    mid = 0.5 * (K[1:] + K[:-1])
    slope = -(np.diff(C) / np.diff(K))
    # isotonic (non-increasing) via pool-adjacent-violators on -slope
    from sklearn.isotonic import IsotonicRegression
    tail = IsotonicRegression(increasing=False, y_min=0.0, y_max=1.0).fit_transform(mid, slope)
    out = np.interp(ks, mid, tail, left=np.nan, right=np.nan)
    return out


def butterfly_violations(K, px, isc, F):
    """Share of adjacent-triple convexity violations in the raw call curve (equal spacing only)."""
    C = np.where(isc, px, px + (F - K))
    v = n = 0
    for i in range(1, len(K) - 1):
        if abs((K[i] - K[i - 1]) - (K[i + 1] - K[i])) < 1e-9:
            n += 1
            v += (C[i - 1] - 2 * C[i] + C[i + 1]) < -0.0011   # beyond one tick
    return v, n
