# -*- coding: utf-8 -*-
"""
Where is there predictability in the 22 BTC chart features, and is it enough to beat costs?

POST-HOC RESEARCH. The 2025-01..2026-09 lockbox was opened on 2026-09-25 by the P0 study, so this
script reads all of 2014-2026 freely. Nothing here was pre-registered; treat every number as
exploratory. It writes only under btc/research/ and never calls btc.evaluate.guard (no audit-log write).

Parts
  1. Rank IC (Spearman) between feature X[t] and the forward open-to-open log return
       y_h[t] = ln(O[t+1+h] / O[t+1])      (entry = next open, like every fill in this repo)
     for h in {1, 6, 42, 180, 540} 4h bars, per era and pooled, with a Newey-West (Bartlett, lags=h)
     t-stat on the product of standardized ranks, plus (pooled only) a circular-shift null p-value
     that also absorbs feature persistence.
     Robustness: the same on a daily resampling (features recomputed on daily bars by the same
     FeatureEngine; h in {1, 7, 30, 90} days).
  2. Walk-forward (expanding from 2014-01, refit every 1 January 2017..2026) OLS / ridge forecasts of
     y_h from all 22 features; out-of-sample R^2 (vs. expanding historical mean, and vs. zero),
     Clark-West t; economic value of "long iff forecast > round-trip cost" with next-open fills and
     0.15% one-way cost, decided every 4h bar or once a day, vs. buy & hold.
  3. The same economics for single well-known signals (C>SMA1200, 1-month momentum, daily MACD
     histogram sign, SMA300/SMA1200 cross), plus their conditional forward-return spreads.
  4. Extras: quintile conditional means (nonlinearity), predictability of forward realized
     volatility (second moment).

  OMP_NUM_THREADS=1 python btc/research/predictability.py
  -> btc/research/predictability.json  (all numbers)   + tables on stdout
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import json
import math
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from btc import stats as S                                   # noqa: E402
from btc.env import simulate, daily_marks, BAR_SEC           # noqa: E402
from btc.features import NAMES, compute                      # noqa: E402
from btc.walkforward import load_phases                      # noqa: E402

OUT_JSON = os.path.join(HERE, "predictability.json")
COST = 0.0015
HURDLE = -2.0 * math.log(1.0 - COST)          # round trip in log units (0.3002%)
H4 = [1, 6, 42, 180, 540]                     # 4h, 1d, 1w, 1m, 3m
HD = [1, 7, 30, 90]                           # daily-bar robustness: 1d, 1w, 1m, 3m
DAY = 86400
TRAIN0 = "2014-01-01"
END = "2026-09-25"
ERAS = [("2014-2016", "2014-01-01", "2017-01-01"), ("2017-2020", "2017-01-01", "2021-01-01"),
        ("2021-2024", "2021-01-01", "2025-01-01"), ("2025-2026", "2025-01-01", END),
        ("pooled", "2014-01-01", END)]
OOS_ERAS = [("2017-2020", "2017-01-01", "2021-01-01"), ("2021-2024", "2021-01-01", "2025-01-01"),
            ("2025-2026", "2025-01-01", END), ("2017-2026", "2017-01-01", END)]
REFITS = list(range(2017, 2027))
LAMBDAS = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
SIGNALS = ["sma1200", "mom1m", "macd1d", "x300_1200"]
SIG_LABEL = {"sma1200": "C > SMA1200 (~200d MA)", "mom1m": "1-month momentum > 0",
             "macd1d": "daily MACD hist > 0", "x300_1200": "SMA300 > SMA1200 (~50/200d cross)"}


def _ts(s):
    return int(pd.Timestamp(s, tz="UTC").timestamp())


# ══════════ series ══════════
class Ser:
    """One bar series (4h phase 0, or its daily resampling) with features and forward returns."""

    def __init__(self, ts, o, h, l, c, v, X, sigma, forced_hold, bar_sec, horizons):
        self.ts, self.o, self.h, self.l, self.c, self.v = ts, o, h, l, c, v
        self.X, self.sigma, self.forced_hold = X.astype(float), sigma, forced_hold
        self.bar_sec = bar_sec
        self.T = len(ts)
        self.tau = ts + bar_sec                         # decision time (bar close)
        self.bpd = DAY // bar_sec
        self.y, self.texit = {}, {}
        for hh in horizons:
            self.y[hh], self.texit[hh] = forward(ts, o, hh, bar_sec, tol=DAY)
        # 1-bar open-to-open log returns r1[j] = ln(O[j+1]/O[j]) for realized-vol targets
        self.r1 = np.full(self.T, np.nan)
        self.r1[:-1] = np.log(o[1:] / o[:-1])


def forward(ts, o, h, bar_sec, tol):
    """
    y[t] = ln(O[exit]/O[t+1]); exit = first live bar with ts >= ts[t+1] + h*bar_sec (calendar-aligned
    so dead bars do not stretch the horizon). NaN if the entry bar is > tol after the decision or the
    exit bar is > tol after its target. texit[t] = open time of the exit bar (when y is known).
    """
    T = len(ts)
    y = np.full(T, np.nan)
    tx = np.full(T, np.iinfo(np.int64).max, dtype=np.int64)
    t = np.arange(T - 1)
    ent = t + 1
    target = ts[ent] + h * bar_sec
    ex = np.searchsorted(ts, target, side="left")
    ok = ex < T
    exc = np.minimum(ex, T - 1)
    ok &= (ts[exc] - target <= tol) & (ts[ent] - (ts[t] + bar_sec) <= tol)
    y[t[ok]] = np.log(o[exc[ok]] / o[ent[ok]])
    tx[t[ok]] = ts[exc[ok]]
    return y, tx


def daily_series(d):
    """Daily bars (UTC days) resampled from the phase-0 4h live bars; 22 features recomputed on them."""
    day = d.ts // DAY
    df = pd.DataFrame({"day": day, "open": d.o, "high": d.h, "low": d.l, "close": d.c, "volume": d.v,
                       "gap": d.gap})
    g = df.groupby("day", sort=True)
    b = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                      "close": g["close"].last(), "volume": g["volume"].sum(), "n": g["open"].count()})
    ts = b.index.to_numpy(np.int64) * DAY
    X, sig = compute(b)
    # forced hold after >= 1 missing day (mirrors the 4h rule: >= 1 day of dead bars)
    fh = np.zeros(len(ts), bool)
    fh[1:] = np.diff(ts) > DAY
    return Ser(ts, b["open"].to_numpy(float), b["high"].to_numpy(float), b["low"].to_numpy(float),
               b["close"].to_numpy(float), b["volume"].to_numpy(float), X, sig, fh, DAY, HD)


# ══════════ part 1: rank IC ══════════
def _z(a):
    a = np.asarray(a, float)
    sd = a.std(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a - a.mean(axis=0)) / sd


def nw_lrv(u, lags):
    """Bartlett (Newey-West) long-run variance of the columns of u (already demeaned)."""
    n = len(u)
    lrv = (u * u).sum(axis=0) / n
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv = lrv + 2.0 * w * (u[k:] * u[:-k]).sum(axis=0) / n
    return lrv


def rank_ic(Xs, y, lags):
    """Spearman IC per column and NW t-stat of mean(z_rank(x) * z_rank(y))."""
    rx = pd.DataFrame(Xs).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    zx, zy = _z(rx), _z(ry)
    p = zx * zy[:, None]
    ic = p.mean(axis=0)
    lrv = nw_lrv(p - ic, lags)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = ic / np.sqrt(np.maximum(lrv, 1e-300) / len(y))
    return ic, t, zx, zy


def shift_null(zx, zy, min_shift):
    """Circular-shift null for every column at once (FFT cross-correlation). Returns (p, sd_null)."""
    n = len(zy)
    zx = np.nan_to_num(zx)
    Fx = np.fft.rfft(zx, axis=0)
    Fy = np.fft.rfft(zy)
    cc = np.fft.irfft(Fx * np.conj(Fy)[:, None], n=n, axis=0) / n      # cc[s] = mean_t zx[t+s] zy[t]
    s = np.arange(n)
    adm = (s >= min_shift) & (s <= n - min_shift)
    null = cc[adm]
    p = (np.abs(null) >= np.abs(cc[0])[None, :]).mean(axis=0)
    return p, null.std(axis=0)


def ic_tables(ser, horizons, names, extra=None):
    """{era: {h: {feature: (ic, t)}}} + pooled shift-null p. extra={name: (T,) array} replaces the features."""
    if extra is None:
        X, names = ser.X, list(names)
    else:
        X, names = np.column_stack([extra[k] for k in extra]), list(extra)
    out = {}
    for era, lo, hi in ERAS:
        em = (ser.tau >= _ts(lo)) & (ser.tau < _ts(hi))
        out[era] = {}
        for hh in horizons:
            m = em & np.isfinite(ser.y[hh]) & np.all(np.isfinite(X), axis=1)
            ic, t, zx, zy = rank_ic(X[m], ser.y[hh][m], hh)
            row = {nm: dict(ic=float(ic[j]), t=float(t[j])) for j, nm in enumerate(names)}
            if era == "pooled":
                p, sdn = shift_null(zx, zy, max(2 * hh, 30 * ser.bpd))
                for j, nm in enumerate(names):
                    z = float(ic[j] / sdn[j]) if sdn[j] > 0 else 0.0
                    # neighbouring shifts are highly correlated, so the empirical p has coarse resolution
                    # (about h/n); report the larger of it and the normal approximation from the null sd
                    p_norm = 2.0 * (1.0 - S.norm_cdf(abs(z)))
                    row[nm].update(p_shift_emp=float(p[j]), z_shift=z, p_shift_norm=p_norm,
                                   p_shift=max(float(p[j]), p_norm))
            row["_n"] = int(m.sum())
            row["_n_indep"] = float(m.sum() / hh)
            out[era][hh] = row
    return out


def quintiles(ser, horizons):
    """Mean forward return by feature quintile (annualized log return), pooled 2014-2026 and per era."""
    ann = 365.0 * ser.bpd
    out = {}
    for era, a, b in ERAS:
        em = (ser.tau >= _ts(a)) & (ser.tau < _ts(b))
        out[era] = {}
        for hh in horizons:
            m = em & np.isfinite(ser.y[hh])
            y = ser.y[hh][m]
            out[era][hh] = {}
            for j, nm in enumerate(NAMES):
                x = ser.X[m, j]
                q = pd.Series(x).rank(pct=True, method="first").to_numpy()
                k = np.minimum((q * 5).astype(int), 4)
                out[era][hh][nm] = [float(y[k == i].mean() / hh * ann) if (k == i).any() else None
                                    for i in range(5)]
    return out


def realized_vol_ic(ser, horizons):
    """IC of features (and ln sigma) with forward realized vol sqrt(sum r1^2 over t+1..t+h), pooled."""
    r2 = np.nan_to_num(ser.r1 ** 2)
    cs = np.concatenate([[0.0], np.cumsum(r2)])
    X = np.column_stack([ser.X, np.log(ser.sigma)])
    names = NAMES + ["ln_sigma"]
    out = {}
    lo, hi = _ts(TRAIN0), _ts(END)
    for hh in horizons:
        T = ser.T
        rv = np.full(T, np.nan)
        t = np.arange(T - hh - 1)
        rv[t] = np.sqrt(cs[t + 1 + hh] - cs[t + 1])
        m = np.isfinite(ser.y[hh]) & np.isfinite(rv) & (ser.tau >= lo) & (ser.tau < hi)
        ic, tt, _, _ = rank_ic(X[m], rv[m], hh)
        out[hh] = {nm: dict(ic=float(ic[j]), t=float(tt[j])) for j, nm in enumerate(names)}
    return out


# ══════════ part 2: walk-forward forecasts ══════════
def ridge_fit(Z, y, lam):
    n, k = Z.shape
    zm, ym = Z.mean(axis=0), y.mean()
    Zc = Z - zm
    A = Zc.T @ Zc / n + (lam + 1e-9) * np.eye(k)
    b = np.linalg.solve(A, Zc.T @ (y - ym) / n)
    return ym - zm @ b, b


def _std(X):
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return mu, sd


def choose_lambda(X, y, tau, texit, idx):
    """Time-ordered inner split of the training window: first 70% fit (embargoed), last 30% validate."""
    idx = idx[np.argsort(tau[idx], kind="stable")]
    cut = int(0.7 * len(idx))
    v0 = tau[idx[cut]]
    fit = idx[:cut][texit[idx[:cut]] <= v0 - 0]
    val = idx[cut:]
    mu, sd = _std(X[fit])
    best, bl = math.inf, LAMBDAS[-1]
    for lam in LAMBDAS:
        a, b = ridge_fit((X[fit] - mu) / sd, y[fit], lam)
        mse = float(np.mean((y[val] - a - ((X[val] - mu) / sd) @ b) ** 2))
        if mse < best - 1e-15:
            best, bl = mse, lam
    return bl


def walk_forward(ser, hh, X, model):
    """
    Expanding-window forecasts of y_h, refit every 1 January (REFITS). Training rows: decision time
    >= 2014-01-01 and forward return fully known (exit open time <= refit time).
    model: 'ols' | 'ridge' (lambda by inner time split) | 'ridge_vs' (ridge on y/(sigma*sqrt(h)))
    Returns forecast f (log return over h bars), historical-mean benchmark, chosen lambdas.
    """
    y, tx, tau = ser.y[hh], ser.texit[hh], ser.tau
    X = X if X.ndim == 2 else X[:, None]
    scale = ser.sigma * math.sqrt(hh) if model == "ridge_vs" else np.ones(ser.T)
    yt = y / scale
    f = np.full(ser.T, np.nan)
    bench = np.full(ser.T, np.nan)
    lams = {}
    t0g = _ts(TRAIN0)
    for Y in REFITS:
        a0, a1 = _ts(f"{Y}-01-01"), min(_ts(f"{Y + 1}-01-01"), _ts(END))
        tr = np.nonzero((tau >= t0g) & np.isfinite(y) & (tx <= a0) & np.all(np.isfinite(X), axis=1))[0]
        te = np.nonzero((tau >= a0) & (tau < a1))[0]
        if len(te) == 0:
            continue
        lam = 0.0 if model == "ols" else choose_lambda(X, yt, tau, tx, tr)
        mu, sd = _std(X[tr])
        a, b = ridge_fit((X[tr] - mu) / sd, yt[tr], lam)
        f[te] = (a + ((X[te] - mu) / sd) @ b) * scale[te]
        bench[te] = y[tr].mean()
        lams[Y] = lam
    return f, bench, lams


def oos_stats(ser, hh, f, bench):
    y = ser.y[hh]
    out = {}
    for era, lo, hi in OOS_ERAS:
        m = (ser.tau >= _ts(lo)) & (ser.tau < _ts(hi)) & np.isfinite(y) & np.isfinite(f)
        if m.sum() < 10:
            out[era] = None
            continue
        e_m, e_b = y[m] - f[m], y[m] - bench[m]
        r2 = 1.0 - (e_m ** 2).sum() / (e_b ** 2).sum()
        r2_0 = 1.0 - (e_m ** 2).sum() / (y[m] ** 2).sum()
        adj = e_b ** 2 - (e_m ** 2 - (bench[m] - f[m]) ** 2)       # Clark-West
        u = (adj - adj.mean())[:, None]
        se = math.sqrt(max(float(nw_lrv(u, hh)[0]), 1e-300) / m.sum())
        # hit rate of sign, and share of forecasts above the hurdle
        out[era] = dict(r2_oos=float(r2), r2_vs_zero=float(r2_0), cw_t=float(adj.mean() / se),
                        corr=float(np.corrcoef(f[m], y[m])[0, 1]),
                        above_hurdle=float((f[m] > HURDLE).mean()), n=int(m.sum()))
    return out


# ══════════ economics ══════════
def decision_range_g(ts, bar_sec, lo, hi):
    close = ts + bar_sec
    a = int(np.searchsorted(close, lo, side="left"))
    b = int(np.searchsorted(close, hi, side="left"))
    return a, min(b, len(ts) - 1)


class Win:
    """Mirror of btc.evaluate.Window (same fills, marks, clipping) without the lockbox audit write."""

    def __init__(self, ser, lo, hi):
        self.s = ser
        self.lo, self.hi = _ts(lo), _ts(hi)
        a, b = decision_range_g(ser.ts, ser.bar_sec, self.lo, self.hi)
        while b > a and (ser.ts[b] + ser.bar_sec > self.hi or (b + 1 < ser.T and ser.ts[b + 1] > self.hi)):
            b -= 1
        self.a, self.b = a, b

    def run(self, targets, cost):
        s = self.s
        sim = simulate(targets, s.o, s.c, cost, self.a, self.b, forced_hold=s.forced_hold)
        days, eq = daily_marks(s.ts, sim["mark"], self.a, self.b, bar_sec=s.bar_sec, lo=self.lo, hi=self.hi)
        return dict(days=days[1:], r=eq[1:] / eq[:-1] - 1.0, pos=sim["pos"][self.a:self.b],
                    logr=sim["logr"][self.a:self.b])


def daily_only(ser, tg):
    """Keep targets only on the bar that closes at 00:00 UTC (one decision a day); NaN = hold."""
    out = np.full(ser.T, np.nan)
    k = (ser.tau % DAY) == 0
    out[k] = tg[k]
    return out


def econ(W, tg, r_bh, boot=True):
    """Net (0.15%) and gross performance of a target series, with era splits and dSharpe vs B&H."""
    net = W.run(tg, COST)
    gross = W.run(tg, 0.0)
    s = S.summary(net["r"])
    t = S.trade_stats(net["pos"], net["logr"], bars_per_day=W.s.bpd)
    res = dict(cagr=s["cagr"], sharpe=s["sharpe"], max_dd=s["max_dd"], exposure=t["exposure"],
               switches_per_year=t["switches_per_year"], avg_hold_days=t["avg_hold_days"],
               gross_sharpe=S.sharpe(gross["r"]), gross_cagr=S.summary(gross["r"])["cagr"])
    if r_bh is not None:
        res["d_sharpe"] = res["sharpe"] - S.sharpe(r_bh)
        if boot:
            bd = S.stationary_bootstrap_diff(net["r"], r_bh, mean_block=20, n_boot=4000, seed=1)
            res["d_sharpe_ci90"], res["p"] = bd["ci90"], bd["p"]
    eras = {}
    for era, lo, hi in OOS_ERAS[:-1]:
        m = (net["days"] > _ts(lo)) & (net["days"] <= _ts(hi))
        se = S.summary(net["r"][m])
        eras[era] = dict(cagr=se["cagr"], sharpe=se["sharpe"], max_dd=se["max_dd"])
        if r_bh is not None:
            sb = S.summary(r_bh[m])
            eras[era].update(bh_cagr=sb["cagr"], bh_sharpe=sb["sharpe"], bh_max_dd=sb["max_dd"])
    res["eras"] = eras
    res["_r"] = net["r"]
    return res


COST_GRID = [0.0, 0.0005, 0.001, 0.0015, 0.0025]


def band_targets(f, hurdle):
    """No-trade band: long if f > hurdle, flat if f < -hurdle, otherwise keep the current position."""
    tg = np.full(len(f), np.nan)
    tg[f > hurdle] = 1.0
    tg[f < -hurdle] = 0.0
    return tg


def cost_curve(W, f, daily):
    """Net Sharpe of 'long iff f > round trip(c)' as the cost c (and with it the hurdle) varies."""
    out = {}
    for c in COST_GRID:
        hc = -2.0 * math.log(1.0 - c)
        tg = np.where(np.isfinite(f), (f > hc).astype(float), np.nan)
        if daily:
            tg = daily_only(W.s, tg)
        r = W.run(tg, c)
        out[c] = dict(sharpe=S.sharpe(r["r"]), cagr=S.summary(r["r"])["cagr"])
    return out


def signal_arrays(ser):
    c = pd.Series(ser.c)
    fi = {n: i for i, n in enumerate(NAMES)}
    s1200 = c.rolling(1200).mean().to_numpy()
    s300 = c.rolling(300).mean().to_numpy()
    with np.errstate(invalid="ignore"):
        return {"sma1200": np.where(np.isfinite(s1200), (ser.c > s1200).astype(float), np.nan),
                "mom1m": (ser.X[:, fi["ret_180"]] > 0).astype(float),
                "macd1d": (ser.X[:, fi["macdh_1d"]] > 0).astype(float),
                "x300_1200": np.where(np.isfinite(s1200), (s300 > s1200).astype(float), np.nan)}


def signal_spread(ser, sig, hh):
    """E[y_h | s=1] - E[y_h | s=0] per era (annualized log return), NW t (lags=h) from OLS y ~ 1 + s."""
    ann = 365.0 * ser.bpd / hh
    out = {}
    for era, lo, hi in ERAS:
        m = (ser.tau >= _ts(lo)) & (ser.tau < _ts(hi)) & np.isfinite(ser.y[hh]) & np.isfinite(sig)
        s, y = sig[m], ser.y[hh][m]
        if s.min() == s.max():
            out[era] = None
            continue
        Xd = np.column_stack([np.ones(len(s)), s])
        beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
        e = y - Xd @ beta
        # full Newey-West covariance of the OLS moment conditions (Bartlett, lags = h)
        n = len(y)
        G = Xd * e[:, None]
        Om = G.T @ G / n
        for k in range(1, hh + 1):
            w = 1.0 - k / (hh + 1.0)
            C = G[k:].T @ G[:-k] / n
            Om += w * (C + C.T)
        Qi = np.linalg.inv(Xd.T @ Xd / n)
        V = Qi @ Om @ Qi / n
        out[era] = dict(on=float((beta[0] + beta[1]) * ann), off=float(beta[0] * ann),
                        spread=float(beta[1] * ann), t=float(beta[1] / math.sqrt(V[1, 1])),
                        share_on=float(s.mean()))
    return out


# ══════════ main ══════════
def main():
    t_start = time.time()
    d = load_phases(1)[0]
    s4 = Ser(d.ts, d.o, d.h, d.l, d.c, d.v, d.X, d.sigma, d.forced_hold, BAR_SEC, H4)
    sd = daily_series(d)
    res = dict(meta=dict(note="post-hoc; lockbox (2025-01..2026-09) already opened; exploratory",
                         cost=COST, hurdle=HURDLE, horizons_4h=H4, horizons_daily=HD, eras=ERAS,
                         refits=REFITS, lambdas=LAMBDAS, generated=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))

    # ── part 1
    print("part 1: IC ...", flush=True)
    sig4 = signal_arrays(s4)
    res["ic_4h"] = ic_tables(s4, H4, NAMES)
    res["ic_signals_4h"] = ic_tables(s4, H4, [], extra=sig4)
    res["ic_daily"] = ic_tables(sd, HD, NAMES)
    res["quintiles_4h"] = quintiles(s4, [42, 180])
    res["rv_ic_4h"] = realized_vol_ic(s4, [6, 42])
    res["signal_spread_4h"] = {k: {hh: signal_spread(s4, sig4[k], hh) for hh in H4} for k in SIGNALS}
    print(f"  {time.time() - t_start:.0f}s", flush=True)

    # ── part 2 + 3
    print("part 2: walk-forward ...", flush=True)
    W4 = Win(s4, "2017-01-01", END)
    Wd = Win(sd, "2017-01-01", END)
    bh_tg4 = np.ones(s4.T)
    bh4 = econ(W4, bh_tg4, None)
    bhd = econ(Wd, np.ones(sd.T), None)
    r_bh4 = bh4["_r"]
    res["bh_4h"] = {k: v for k, v in bh4.items() if k != "_r"}
    res["bh_daily_bars"] = {k: v for k, v in bhd.items() if k != "_r"}
    res["bh_days_equal"] = bool(len(bhd["_r"]) == len(r_bh4))
    res["bh_cost_curve"] = {c: S.sharpe(W4.run(bh_tg4, c)["r"]) for c in COST_GRID}
    # typical size of the forward return per horizon (to translate IC into an edge vs. the hurdle)
    res["y_sd_4h"] = {}
    for era, lo, hi in ERAS:
        m0 = (s4.tau >= _ts(lo)) & (s4.tau < _ts(hi))
        res["y_sd_4h"][era] = {hh: float(np.nanstd(s4.y[hh][m0])) for hh in H4}

    wf = {}
    for model in ("ols", "ridge", "ridge_vs"):
        for hh in H4:
            f, bench, lams = walk_forward(s4, hh, s4.X, model)
            key = f"{model}_h{hh}"
            ent = dict(model=model, h=hh, lambdas=lams, oos=oos_stats(s4, hh, f, bench))
            tg = np.where(np.isfinite(f), (f > HURDLE).astype(float), np.nan)
            tg0 = np.where(np.isfinite(f), (f > 0).astype(float), np.nan)
            e4 = econ(W4, tg, r_bh4)
            ed = econ(W4, daily_only(s4, tg), r_bh4)
            e0 = econ(W4, tg0, r_bh4, boot=False)
            eb = econ(W4, band_targets(f, HURDLE), r_bh4, boot=False)
            ebd = econ(W4, daily_only(s4, band_targets(f, HURDLE)), r_bh4, boot=False)
            ent["econ_4h_band"] = {k: v for k, v in eb.items() if k != "_r"}
            ent["econ_daily_band"] = {k: v for k, v in ebd.items() if k != "_r"}
            ent["cost_curve_4h"] = cost_curve(W4, f, False)
            ent["cost_curve_daily"] = cost_curve(W4, f, True)
            ent["econ_4h"] = {k: v for k, v in e4.items() if k != "_r"}
            ent["econ_daily_decision"] = {k: v for k, v in ed.items() if k != "_r"}
            ent["econ_4h_hurdle0"] = {k: v for k, v in e0.items() if k != "_r"}
            wf[key] = ent
            print(f"  {key:14s} R2oos={ent['oos']['2017-2026']['r2_oos']:+.4f} "
                  f"SR4h={e4['sharpe']:.2f} SRd={ed['sharpe']:.2f} sw={e4['switches_per_year']:.0f} "
                  f"lam={list(lams.values())}  {time.time() - t_start:.0f}s", flush=True)
    res["wf_4h"] = wf

    # band-width sensitivity of the no-trade-band rule (exploratory; shows how knife-edge it is)
    bs = {}
    for model, hh in (("ols", 1), ("ols", 6), ("ridge", 6), ("ridge_vs", 42), ("ridge", 180)):
        f, _, _ = walk_forward(s4, hh, s4.X, model)
        row = {}
        for band in (0.001, 0.002, 0.003, 0.004, 0.005, 0.0075, 0.01):
            r = W4.run(band_targets(f, band), COST)
            t = S.trade_stats(r["pos"], r["logr"])
            row[band] = dict(sharpe=S.sharpe(r["r"]), max_dd=S.summary(r["r"])["max_dd"],
                             exposure=t["exposure"], switches_per_year=t["switches_per_year"])
        bs[f"{model}_h{hh}"] = row
    res["band_sensitivity_4h"] = bs

    # daily-bar models (features recomputed on daily bars), daily decisions, same hurdle
    wfd = {}
    r_bhd = bhd["_r"]
    for model in ("ols", "ridge", "ridge_vs"):
        for hh in HD:
            f, bench, lams = walk_forward(sd, hh, sd.X, model)
            key = f"{model}_d{hh}"
            ent = dict(model=model, h_days=hh, lambdas=lams, oos=oos_stats(sd, hh, f, bench))
            tg = np.where(np.isfinite(f), (f > HURDLE).astype(float), np.nan)
            e = econ(Wd, tg, r_bhd)
            ent["econ"] = {k: v for k, v in e.items() if k != "_r"}
            wfd[key] = ent
            print(f"  {key:14s} R2oos={ent['oos']['2017-2026']['r2_oos']:+.4f} SR={e['sharpe']:.2f} "
                  f"sw={e['switches_per_year']:.0f}", flush=True)
    res["wf_daily_bars"] = wfd

    print("part 3: single signals ...", flush=True)
    sg = {}
    for k in SIGNALS:
        s = sig4[k]
        tg = np.where(np.isfinite(s), s, 0.0)
        ent = dict(label=SIG_LABEL[k])
        e4 = econ(W4, tg, r_bh4)
        ed = econ(W4, daily_only(s4, tg), r_bh4)
        ent["econ_4h"] = {q: v for q, v in e4.items() if q != "_r"}
        ent["econ_daily_decision"] = {q: v for q, v in ed.items() if q != "_r"}
        ent["cost_curve_4h"] = {c: S.sharpe(W4.run(tg, c)["r"]) for c in COST_GRID}
        ent["cost_curve_daily"] = {c: S.sharpe(W4.run(daily_only(s4, tg), c)["r"]) for c in COST_GRID}
        ent["oos"] = {}
        for hh in H4:
            f, bench, _ = walk_forward(s4, hh, np.nan_to_num(s, nan=0.0), "ols")
            ent["oos"][hh] = oos_stats(s4, hh, f, bench)
        sg[k] = ent
        print(f"  {k:10s} SR4h={e4['sharpe']:.2f} SRd={ed['sharpe']:.2f} gross={e4['gross_sharpe']:.2f} "
              f"sw={e4['switches_per_year']:.0f}", flush=True)
    res["signals"] = sg
    res["meta"]["runtime_s"] = time.time() - t_start

    def _clean(o):
        if isinstance(o, dict):
            return {str(k): _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            v = float(o)
            return None if not math.isfinite(v) else round(v, 6)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o

    with open(OUT_JSON, "w", encoding="utf-8") as fp:
        json.dump(_clean(res), fp, ensure_ascii=False, indent=1)
    print(f"wrote {OUT_JSON}  ({time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
