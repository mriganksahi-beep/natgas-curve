"""Linear-Gaussian one-factor state-space model with a hand-written Kalman filter.

State:   s_t = mu + phi (s_{t-1} - mu) + eta_t,   eta_t ~ N(0, q_t)
         q_t = q in ordinary weeks, q_roll in the week the target winter rolls forward.
         Random walk: phi = 1 (mu irrelevant).
Obs:     y_it = a_i + c_i tau_it + b_i exp(-kappa tau_it) s_t + e_it,   e_it ~ N(0, h_i)
         Identification: a_MarApr = 0, b_MarApr = 1, so s_t is in $/MMBtu of a
         March-April spread at delivery (tau = 0).
Variant h_floor: h_i = h_floor + exp(.), a floor on each measurement-noise variance.
Missing observations are skipped row by row (standard). Filtered (not smoothed) output.
"""
import numpy as np
from scipy.optimize import minimize

from .data import NAMES

K = len(NAMES)
IM = NAMES.index("MarApr")
FREE_A = [i for i in range(K) if i != IM]
FREE_B = [i for i in range(K) if i != IM]


class OneFactorKF:
    def __init__(self, ar1=False, h_floor=0.0):
        self.ar1 = ar1
        self.h_floor = h_floor

    # parameter vector <-> dict
    def unpack(self, th):
        th = np.asarray(th, float)
        p = {}
        j = 0
        a = np.zeros(K); a[FREE_A] = th[j:j + K - 1]; j += K - 1
        c = th[j:j + K]; j += K
        b = np.ones(K); b[FREE_B] = th[j:j + K - 1]; j += K - 1
        h = self.h_floor + np.exp(th[j:j + K]); j += K
        q = np.exp(th[j]); q_roll = np.exp(th[j + 1]); kappa = np.exp(th[j + 2]); j += 3
        if self.ar1:
            phi = 1 / (1 + np.exp(-th[j])); mu = th[j + 1]; j += 2
        else:
            phi, mu = 1.0, 0.0
        p.update(a=a, c=c, b=b, h=h, q=q, q_roll=q_roll, kappa=kappa, phi=phi, mu=mu)
        return p

    def start(self, Y, T):
        """Crude start values: s ~ MarApr, OLS of each series on [1, tau, MarApr]."""
        s = Y[:, IM]
        a0, c0, b0 = np.zeros(K), np.zeros(K), np.ones(K)
        h0 = np.zeros(K)
        for i in range(K):
            m = np.isfinite(Y[:, i]) & np.isfinite(s) & np.isfinite(T[:, i])
            X = np.column_stack([np.ones(m.sum()), T[m, i], s[m]])
            coef, *_ = np.linalg.lstsq(X, Y[m, i], rcond=None)
            a0[i], c0[i], b0[i] = coef
            h0[i] = np.var(Y[m, i] - X @ coef) + 1e-4
        h0[IM] = 0.002
        th = list(a0[FREE_A]) + list(c0 * 0) + list(b0[FREE_B]) + list(np.log(h0)) + \
            [np.log(0.003), np.log(0.05), np.log(0.5)]
        if self.ar1:
            th += [3.0, 0.2]
        return np.array(th)

    def filter(self, th, Y, T, roll, s0=0.2, P0=1.0):
        p = self.unpack(th)
        n = len(Y)
        s_f, P_f = np.full(n, np.nan), np.full(n, np.nan)
        s_p_out, P_p_out = np.full(n, np.nan), np.full(n, np.nan)
        innov = np.full((n, K), np.nan)
        ll = 0.0
        s, P = s0, P0
        for t in range(n):
            q = p["q_roll"] if roll[t] else p["q"]
            if t == 0:
                sp, Pp = s0, P0
            else:
                sp = p["mu"] + p["phi"] * (s - p["mu"])
                Pp = p["phi"] ** 2 * P + q
            s_p_out[t], P_p_out[t] = sp, Pp
            m = np.isfinite(Y[t]) & np.isfinite(T[t])
            if m.any():
                tau = np.maximum(T[t, m], 0.0)
                Z = p["b"][m] * np.exp(-p["kappa"] * tau)
                d = p["a"][m] + p["c"][m] * tau
                v = Y[t, m] - d - Z * sp
                F = Pp * np.outer(Z, Z) + np.diag(p["h"][m])
                Fi = np.linalg.inv(F)
                _, logdet = np.linalg.slogdet(F)
                ll += -0.5 * (m.sum() * np.log(2 * np.pi) + logdet + v @ Fi @ v)
                Kg = Pp * Z @ Fi
                s = sp + Kg @ v
                P = Pp - Pp * (Kg @ Z) * 1.0
                P = max(P, 1e-10)
                innov[t, m] = v
            else:
                s, P = sp, Pp
            s_f[t], P_f[t] = s, P
        return dict(s=s_f, P=P_f, s_pred=s_p_out, P_pred=P_p_out, innov=innov, ll=ll, params=p)

    def fit(self, Y, T, roll, maxiter=400):
        th0 = self.start(Y, T)
        f = lambda th: -self.filter(th, Y, T, roll)["ll"]
        best = None
        for method in ("L-BFGS-B", "Nelder-Mead"):
            x0 = th0 if best is None else best.x
            opts = {"maxiter": maxiter} if method == "L-BFGS-B" else {"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-3}
            r = minimize(f, x0, method=method, options=opts)
            if best is None or r.fun < best.fun:
                best = r
        # polish once more with L-BFGS-B
        r = minimize(f, best.x, method="L-BFGS-B", options={"maxiter": maxiter})
        if r.fun < best.fun:
            best = r
        self.theta, self.nll = best.x, best.fun
        return best
