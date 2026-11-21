"""Model 3, steps 3-4: storage-based fair value for the March-April spread, and tests.

For each test winter (walk-forward, refit each November on earlier winters only):
  1. Fair value F = f(inputs) for each method, fitted on all earlier winters.
  2. Gap G = spread - F.
  3. How much of a gap closes over h weeks, b_h, is estimated on earlier winters using
     leave-one-winter-out gaps (each earlier winter's gap from a fit that excluded it),
     so b is not inflated by in-sample residuals. b is clipped to [0, 1]: a gap either
     closes partly, fully, or not at all.
  4. Forecast of the spread change: dS_hat = -b_h * G.
A test winter needs at least two earlier winters (one to fit, one to estimate b).

Run:  python -m src.model3            # validation winters only
      python -m src.model3 holdout    # holdout, once per method version (see protocol)
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import SplineTransformer, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data/processed"
X_COLS = ["exp_surplus", "downside", "april", "weeks_to_expiry"]
HORIZONS = (1, 4)
TICK_USD = 10.0  # $0.001/MMBtu x 10,000 MMBtu
COST_TICKS = (0, 2, 4, 6, 10)  # round trip, per leg


# ---------------------------------------------------------------- fair-value methods
class Linear:
    def fit(self, X, y):
        self.m = make_pipeline(StandardScaler(), LinearRegression()).fit(X, y)
        return self

    def predict(self, X):
        return self.m.predict(X)


class Convex:
    """Theory-of-storage shape: S = c0 + c1 * exp(-c2 * surplus/100) + c3 * April price."""

    @staticmethod
    def f(Z, c0, c1, c2, c3):
        return c0 + c1 * np.exp(-c2 * Z[0]) + c3 * Z[1]

    def fit(self, X, y):
        Z = np.vstack([X["exp_surplus"].values / 100, X["april"].values])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.p, _ = curve_fit(self.f, Z, y.values, p0=(0.0, 0.05, 0.5, 0.05),
                                      bounds=([-5, 0, 0, -2], [5, 10, 5, 2]), maxfev=20000)
            except RuntimeError:
                self.p = None
                self.fallback = Linear().fit(X, y)
        return self

    def predict(self, X):
        if self.p is None:
            return self.fallback.predict(X)
        Z = np.vstack([X["exp_surplus"].values / 100, X["april"].values])
        return self.f(Z, *self.p)


class Spline:
    """Cubic spline in expected surplus (the GAM-style term) plus linear terms, ridge-penalised."""

    def fit(self, X, y):
        ct = ColumnTransformer([("spline", SplineTransformer(n_knots=4, degree=3), ["exp_surplus"]),
                                ("lin", StandardScaler(), ["downside", "april", "weeks_to_expiry"])])
        self.m = make_pipeline(ct, Ridge(alpha=1.0)).fit(X, y)
        return self

    def predict(self, X):
        return self.m.predict(X)


class GBM:
    def fit(self, X, y):
        self.m = HistGradientBoostingRegressor(max_depth=2, max_iter=100, learning_rate=0.05,
                                               min_samples_leaf=5, random_state=0).fit(X, y)
        return self

    def predict(self, X):
        return self.m.predict(X)


class MLP:
    """Small PyTorch network: 4 -> 16 -> 16 -> 1, weight decay, early stopping on the last
    training winter. Averaged over 5 seeds to reduce run-to-run noise."""

    def __init__(self, seeds=5, epochs=600, patience=40):
        self.seeds, self.epochs, self.patience = seeds, epochs, patience

    def fit(self, X, y, winters=None):
        import torch
        self.sx = StandardScaler().fit(X)
        self.my, self.sy = float(y.mean()), float(y.std() or 1.0)
        Xs = self.sx.transform(X).astype(np.float32)
        ys = ((y.values - self.my) / self.sy).astype(np.float32)
        last = winters == sorted(set(winters))[-1] if winters is not None and len(set(winters)) > 1 \
            else np.zeros(len(y), bool)
        tr, va = ~last, last
        self.nets = []
        for seed in range(self.seeds):
            torch.manual_seed(seed)
            net = torch.nn.Sequential(torch.nn.Linear(Xs.shape[1], 16), torch.nn.ReLU(),
                                      torch.nn.Linear(16, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
            opt = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=1e-3)
            xt, yt = torch.tensor(Xs[tr]), torch.tensor(ys[tr]).unsqueeze(1)
            xv, yv = (torch.tensor(Xs[va]), torch.tensor(ys[va]).unsqueeze(1)) if va.any() else (xt, yt)
            best, best_state, wait = np.inf, None, 0
            for _ in range(self.epochs):
                opt.zero_grad()
                loss = torch.nn.functional.mse_loss(net(xt), yt)
                loss.backward()
                opt.step()
                with torch.no_grad():
                    v = torch.nn.functional.mse_loss(net(xv), yv).item()
                if v < best - 1e-6:
                    best, best_state, wait = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
                else:
                    wait += 1
                    if wait >= self.patience:
                        break
            net.load_state_dict(best_state)
            self.nets.append(net)
        return self

    def predict(self, X):
        import torch
        xs = torch.tensor(self.sx.transform(X).astype(np.float32))
        with torch.no_grad():
            p = np.mean([n(xs).squeeze(1).numpy() for n in self.nets], axis=0)
        return p * self.sy + self.my


METHODS = {"linear": Linear, "convex": Convex, "spline": Spline, "gbm": GBM, "mlp": MLP}


def _fit(cls, d):
    m = cls()
    if isinstance(m, MLP):
        return m.fit(d[X_COLS], d["spread"], winters=d["winter"].values)
    return m.fit(d[X_COLS], d["spread"])


# ---------------------------------------------------------------- walk-forward
def walk_forward(df):
    winters = sorted(df["winter"].unique())
    out = []
    for i, w in enumerate(winters):
        test = df[df.winter == w]
        rec = test[["obs_date", "winter", "period", "spread"] + [f"dS_{h}w" for h in HORIZONS]].copy()
        if i < 2:
            continue  # need two earlier winters
        train = df[df.winter.isin(winters[:i])]
        for name, cls in METHODS.items():
            fv = _fit(cls, train).predict(test[X_COLS])
            # leave-one-winter-out gaps on the training winters, to estimate b
            lowo = []
            for k in winters[:i]:
                rest, held = train[train.winter != k], train[train.winter == k]
                g = held["spread"].values - _fit(cls, rest).predict(held[X_COLS])
                lowo.append(pd.DataFrame({"gap": g, **{f"dS_{h}w": held[f"dS_{h}w"].values for h in HORIZONS}}))
            lowo = pd.concat(lowo)
            rec[f"{name}_fair"] = fv
            rec[f"{name}_gap"] = test["spread"].values - fv
            for h in HORIZONS:
                ok = lowo[f"dS_{h}w"].notna()
                gg, dd = lowo.loc[ok, "gap"].values, lowo.loc[ok, f"dS_{h}w"].values
                b = float(np.clip(-(gg @ dd) / (gg @ gg), 0, 1)) if (gg @ gg) > 0 else 0.0
                rec[f"{name}_b{h}"] = b
                rec[f"{name}_fc{h}"] = -b * rec[f"{name}_gap"]
        out.append(rec)
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- evaluation
def newey_west_se(x, lags):
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    v = x @ x / n
    for L in range(1, lags + 1):
        v += 2 * (1 - L / (lags + 1)) * (x[L:] @ x[:-L]) / n
    return np.sqrt(v / n)


def evaluate(wf, period, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for h in HORIZONS:
        d = wf[(wf.period == period) & wf[f"dS_{h}w"].notna()]
        y = d[f"dS_{h}w"].values
        for name in METHODS:
            fc = d[f"{name}_fc{h}"].values
            e_rw, e_m = y ** 2, (y - fc) ** 2
            r2 = 1 - e_m.sum() / e_rw.sum()
            ld = e_rw - e_m  # > 0 when the model beats "no change"
            t = ld.mean() / newey_west_se(ld, h) if ld.std() > 0 else 0.0
            pos = -np.sign(d[f"{name}_gap"].values) * (d[f"{name}_b{h}"].values > 0)
            traded = pos != 0
            gross = pos * y * 10_000  # $ per spread (1 March + 1 April contract)
            r = {"h (weeks)": h, "method": name, "rows": len(d), "OOS R2 vs no change": r2,
                 "DM t": t, "DM p (one-sided)": 1 - norm.cdf(t),
                 "trades": int(traded.sum()),
                 "sign right (%)": 100 * np.mean(np.sign(fc[traded]) == np.sign(y[traded])) if traded.any() else np.nan,
                 "mean b": d[f"{name}_b{h}"].mean()}
            for c in COST_TICKS:
                pnl = gross[traded] - 2 * c * TICK_USD
                r[f"$/trade @{c}t"] = pnl.mean() if traded.any() else np.nan
            # winter-block bootstrap of mean P&L at 4 ticks
            if traded.any():
                pnl4 = pd.Series(gross - 2 * 4 * TICK_USD * traded, index=d.winter.values)[traded]
                ws = pnl4.index.unique()
                bs = [pnl4.loc[rng.choice(ws, len(ws))].mean() for _ in range(n_boot)]
                r["@4t 95% CI low"], r["@4t 95% CI high"] = np.percentile(bs, [2.5, 97.5])
            rows.append(r)
    return pd.DataFrame(rows)


def main(period="validation"):
    df = pd.read_csv(PROC / "model3_weekly.csv", parse_dates=["obs_date"])
    wf = walk_forward(df)
    wf.to_csv(PROC / "model3_walkforward.csv", index=False)
    res = evaluate(wf, period)
    res.to_csv(ROOT / f"reports/model3_results_{period}.csv", index=False)
    pd.set_option("display.width", 250)
    print(f"{period}: test winters {sorted(wf[wf.period == period].winter.unique())}")
    print(res.round(3).to_string(index=False))
    return wf, res


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "validation")
