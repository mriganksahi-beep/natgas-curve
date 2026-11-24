"""Reviewer v5: dev reference values for the recommended frozen tests (panel columns + futures only).
Usage: python -m src.research.reviewer_v5.frozen_ref [V5DIR] [START] [END]"""
import sys
import numpy as np, pandas as pd
from scipy.special import ndtr
V5 = sys.argv[1] if len(sys.argv) > 1 else 'data/processed/v5'
START = pd.Timestamp(sys.argv[2] if len(sys.argv) > 2 else '2010-01-01')
END = pd.Timestamp(sys.argv[3] if len(sys.argv) > 3 else '2021-12-31')
OUT_END = END + pd.Timedelta(days=90)   # outcomes may resolve <= 90 days after END for positions opened <= END
def gy(d): d = pd.DatetimeIndex(d); return np.where(d.month >= 4, d.year + 1, d.year)
def boot(v, blk, n=4000, seed=0):
    d = pd.DataFrame({'v': v, 'b': blk}).dropna(); g = d.groupby('b').v.agg(['sum', 'count'])
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(g), (n, len(g)))
    m = g['sum'].values[idx].sum(1) / g['count'].values[idx].sum(1)
    return d.v.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5), len(g), len(d)
c = pd.read_csv('data/raw/ng_curve_daily.csv', parse_dates=['date'], usecols=['date', 'contract_year', 'contract_month', 'settle'])
c = c[(c.date <= OUT_END) & (c.settle > 0)]; c['cid'] = c.contract_year * 100 + c.contract_month
px = c.pivot_table(index='date', columns='cid', values='settle').sort_index(); days = px.index
oexp = {}
for cid in px.columns:
    y, m = divmod(int(cid), 100); p = days.searchsorted(pd.Timestamp(y, m, 1))
    if 4 <= p < len(days): oexp[cid] = days[p - 4]
nxt = lambda cid: cid + 1 if cid % 100 < 12 else (cid // 100 + 1) * 100 + 1
# ---------------- PRIMARY: CSO vol richness, ln(nv / RV_S,life)
sp = pd.read_csv(f'{V5}/spreads.csv', parse_dates=['date'])
sp = sp[(sp.date >= START) & (sp.date <= END)].copy()
sp['oi'] = sp.cso_call_oi.fillna(0) + sp.cso_put_oi.fillna(0)
sp = sp.sort_values(['cid', 'date']); sp['oi'] = sp.groupby('cid').oi.shift(1)   # OI known at d: previous panel row (published next day)
sp['td'] = (sp['T'] * 252).round().astype(int)
sp = sp[(sp.oi > 0) & sp.td.between(5, 126) & (sp.nv > 0)]
rows = []
for r in sp.itertuples():
    e = oexp.get(r.cid); f = nxt(r.cid)
    if e is None or e > OUT_END or f not in px.columns or r.date not in days: continue
    i0, i1 = days.get_loc(r.date), days.get_loc(e)
    S = (px[r.cid] - px[f]).values[i0:i1 + 1]; dS = np.diff(S); dS = dS[np.isfinite(dS)]
    if len(dS) < 5: continue
    rv = np.sqrt(np.sum(dS ** 2) / (len(dS) / 252))
    rows.append((r.date, r.cid, r.nv, rv))
P = pd.DataFrame(rows, columns=['date', 'cid', 'nv', 'rv']); P = P[P.rv > 0]
P['lr'] = np.log(P.nv / P.rv); P['gy'] = gy(P.date)
m, lo, hi, nb, n = boot(P.lr.values, P.gy.values)
print(f'PRIMARY ln(nv/RV_S): mean {m:.3f} [{lo:.3f},{hi:.3f}] median {P.lr.median():.3f} blocks {nb} n {n}')
gm = P.groupby('gy').lr.mean()
for a, b in (('2010', '2016-12-31'), ('2017', '2021-12-31'), ('2022', '2027-12-31')):
    x = P[(P.date >= a) & (P.date <= b)]
    if len(x): print(f'   window {a}-{b[:4]}: mean {x.lr.mean():.3f} n {len(x)}')
print('  gas-year means', gm.round(2).to_dict(), ' share>0', round((gm > 0).mean(), 2))
hj = P[P.cid % 100 == 3]; print('  H-J only mean', round(hj.lr.mean(), 3), 'n', len(hj))
# ---------------- SECONDARY 1: short ATM CSO straddle, panel-priced (Bachelier at nv, K = S), daily hedge via spread
tr = []
for cid, g in sp.groupby('cid'):
    g = g[g.td.between(37, 42)].sort_values('date')
    if g.empty: continue
    r = g.iloc[0]; e = oexp.get(cid); f = nxt(cid)
    if e is None or e > OUT_END or f not in px.columns: continue
    i0, i1 = days.get_loc(r.date), days.get_loc(e)
    S = (px[cid] - px[f]).values[i0:i1 + 1]
    if not np.isfinite(S).all(): continue
    T0 = r.td / 252; K = S[0]; nv = r.nv
    prem = nv * np.sqrt(2 * T0 / np.pi)
    pnl, cost, dprev = prem, 0.04 * prem, 0.0
    for k in range(len(S) - 1):
        tau = max((r.td - k) / 252, 1e-6)
        dl = 2 * ndtr((S[k] - K) / (nv * np.sqrt(tau))) - 1      # straddle delta (short -> hedge +dl)
        cost += 0.002 * abs(dl - dprev); dprev = dl
        pnl += dl * (S[k + 1] - S[k])
    cost += 0.002 * abs(dprev)
    pay = abs(S[-1] - K); pnl -= pay; cost += 0.002 if pay > 0 else 0
    tr.append((r.date, cid, prem * 1e4, (pnl - cost) * 1e4, (pnl - cost - 0.04 * prem) * 1e4))
T = pd.DataFrame(tr, columns=['entry', 'cid', 'prem', 'net4', 'net8']); T['gy'] = gy(T.entry)
m, lo, hi, nb, n = boot(T.net4.values, T.gy.values)
print(f'S1 short CSO straddle net $/trade @4%: {m:.0f} [{lo:.0f},{hi:.0f}] n {n}; @8% {T.net8.mean():.0f}; worst {T.net4.min():.0f}; mean prem {T.prem.mean():.0f}')
for a, b in (('2010', '2016-12-31'), ('2017', '2021-12-31')):
    x = T[(T.entry >= a) & (T.entry <= b)]; print(f'   {a}-{b[:4]} n {len(x)} net4 {x.net4.mean():.0f}')
print('  by gas year', T.groupby('gy').net4.sum().round(0).to_dict())
# ---------------- SECONDARY 2: loose-storage winter-delivery IV - RV_life
s = pd.read_csv(f'{V5}/surface.csv', parse_dates=['date']); s = s[(s.date >= START) & (s.date <= END)].copy()
s['td'] = (s['T'] * 252).round()
s = s[((s.cid % 100).isin([12, 1, 2, 3])) | s.td.between(10, 126)].copy()
st = pd.read_csv('data/raw/storage.csv', parse_dates=['week_ending']); st = st[st.week_ending <= END].set_index('week_ending').storage_bcf.sort_index()
avg = []
for d, v in st.items():
    prior = [st.iloc[st.index.get_indexer([d - pd.DateOffset(years=k)], method='nearest')[0]] for k in range(1, 6)
             if (d - pd.DateOffset(years=k)) >= st.index[0] - pd.Timedelta(days=3)]
    avg.append(np.mean(prior) if len(prior) >= 2 else np.nan)
dev = pd.Series(st.values - np.array(avg), index=st.index + pd.Timedelta(days=6)).dropna()
s['sdev'] = dev.reindex(s.date.unique(), method='ffill').reindex(s.date).values
lp = np.log(px)
rvl = []
for r in s.itertuples():
    e = oexp.get(r.cid)
    if e is None or e > OUT_END or r.cid not in px.columns: rvl.append(np.nan); continue
    x = np.diff(lp[r.cid].values[days.get_loc(r.date):days.get_loc(e) + 1]); x = x[np.isfinite(x)]
    rvl.append(np.sqrt(252 * np.mean(x ** 2)) if len(x) >= 2 else np.nan)
s['gap'] = 100 * (s.atm_iv - np.array(rvl)); s['gy'] = gy(s.date)
x = s[s.td.between(10, 126)].dropna(subset=['gap']); m, lo, hi, nb, n = boot(x.gap.values, x.gy.values)
print(f'S3 all deliveries 10-126td IV-RV_life: {m:.2f} [{lo:.2f},{hi:.2f}] gas-years {nb} rows {n}')
for a, b in (('2010', '2016-12-31'), ('2017', '2021-12-31')):
    xx = x[(x.date >= a) & (x.date <= b)]; print(f'   {a}-{b[:4]} {xx.gap.mean():.2f} n {len(xx)}') if len(xx) else None
s = s[(s.cid % 100).isin([12, 1, 2, 3])].copy()
for reg, msk in (('loose', s.sdev > 150), ('tight', s.sdev < -150), ('loose td<=126', (s.sdev > 150) & (s.td <= 126)), ('tight td<=126', (s.sdev < -150) & (s.td <= 126))):
    x = s[msk].dropna(subset=['gap']); m, lo, hi, nb, n = boot(x.gap.values, x.gy.values)
    print(f'S2 {reg}: {m:.1f} [{lo:.1f},{hi:.1f}] gas-years {nb} rows {n}; by gy', x.groupby('gy').gap.mean().round(1).to_dict())
