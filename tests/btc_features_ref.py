# -*- coding: utf-8 -*-
"""22개 시장 특징의 독립 배치 기준 구현 (independent batch reference).

btc/features.py(증분 FeatureEngine)는 보지 않고 명세(spec §2)와 interfaces.md 의 "Precise conventions"
만으로 작성했습니다. 두 구현을 t >= 1200 에서 대조해 공식 버그를 잡는 것이 목적입니다.

Written only from spec §2 + interfaces.md, with vectorised pandas/numpy (rolling, ewm, shift). The only
explicit loop is the Wilder RSI recursion, so that its SMA seeding follows the stated convention literally.

Conventions of THIS reference (all positional: t = 0 is the first row of df, whatever df.index is):
  * r_t = ln(C_t/C_{t-1}); NaN at t = 0.
  * E_t = EMA(r^2), alpha = 2/43, adjust=False, seeded E_1 = r_1^2;  sigma_t = max(0.002, sqrt(E_t)).
  * sigmaL_t = max(0.002, sqrt(mean(r^2_{t-539..t}))), exact window, first defined at t = 540.
  * Every feature uses the CURRENT sigma_t (which already contains r_t).
  * Warm-up: a feature is NaN until every quantity it needs is defined; FIRST_DEFINED gives the first
    finite index of each feature.  For t >= 1199 all 22 features are finite (given finite positive input).
  * Only open/high/low/close/volume are read; extra columns are ignored; `open` is not used by any feature.
"""
import numpy as np
import pandas as pd

NAMES = ["ret_1", "ret_6", "ret_42", "ret_180", "ma_20", "ma_50", "ma_200", "ma_1200", "x_50_200",
         "x_300_1200", "rsi_14", "rsi_84", "macd_4h", "macdh_4h", "macdh_1d", "vol_regime", "rel_vol",
         "vol_trend", "flow_20", "range", "clv", "dd_180"]

SIGMA_FLOOR = 0.002
E_ALPHA = 2.0 / 43.0          # span 42
SIGMA_L_WINDOW = 540
FLOW_EPS = 1e-12

# first positional index at which each reference feature is finite
FIRST_DEFINED = {
    "ret_1": 1, "ret_6": 6, "ret_42": 42, "ret_180": 180,
    "ma_20": 19, "ma_50": 49, "ma_200": 199, "ma_1200": 1199,
    "x_50_200": 199, "x_300_1200": 1199,
    "rsi_14": 14, "rsi_84": 84,
    "macd_4h": 1, "macdh_4h": 1, "macdh_1d": 1,
    "vol_regime": 540, "rel_vol": 180, "vol_trend": 179, "flow_20": 20,
    "range": 1, "clv": 0, "dd_180": 179,
}


# ---------------------------------------------------------------- primitives
def u(x):
    """u(x) = clip(x, -4, 4)/2 (NaN stays NaN)."""
    return np.clip(x, -4.0, 4.0) / 2.0


def lag(x, k):
    """x_{t-k} as an array aligned on t (NaN for t < k)."""
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan)
    if k < len(x):
        out[k:] = x[:len(x) - k]
    return out


def ewm_alpha(x, alpha):
    """y_0 = first defined x; y_t = (1-alpha) y_{t-1} + alpha x_t  (pandas ewm, adjust=False)."""
    return pd.Series(np.asarray(x, dtype=float)).ewm(alpha=alpha, adjust=False).mean().to_numpy()


def ema(x, n):
    """EMA_n: alpha = 2/(n+1), adjust=False, seeded with the first value."""
    return ewm_alpha(x, 2.0 / (n + 1.0))


def roll_mean(x, n):
    """Exact rolling mean over x_{t-n+1..t}; NaN until n values (or if any NaN in the window)."""
    return pd.Series(np.asarray(x, dtype=float)).rolling(n).mean().to_numpy()


def roll_sum(x, n):
    return pd.Series(np.asarray(x, dtype=float)).rolling(n).sum().to_numpy()


def roll_max(x, n):
    return pd.Series(np.asarray(x, dtype=float)).rolling(n).max().to_numpy()


def wilder_gl(close, n):
    """Wilder average gain/loss arrays (G, L).

    d_t = C_t - C_{t-1}; G_n, L_n = mean of gains/losses over d_1..d_n; for t > n
    G_t = (G_{t-1}(n-1) + g_t)/n (same for L).  NaN for t < n.
    """
    c = np.asarray(close, dtype=float)
    T = len(c)
    G = np.full(T, np.nan)
    L = np.full(T, np.nan)
    if T <= n:
        return G, L
    d = c[1:] - c[:-1]                 # d[t-1] = C_t - C_{t-1}
    gain = np.maximum(d, 0.0)          # np.maximum propagates NaN
    loss = np.maximum(-d, 0.0)
    g = gain[:n].mean()
    lo = loss[:n].mean()
    G[n], L[n] = g, lo
    for t in range(n + 1, T):
        g = (g * (n - 1) + gain[t - 1]) / n
        lo = (lo * (n - 1) + loss[t - 1]) / n
        G[t], L[t] = g, lo
    return G, L


def rsi_from_gl(G, L):
    """RSI = 100 - 100/(1+G/L); 100 if L == 0 (G > 0); 50 if G == L == 0; NaN if undefined."""
    G = np.asarray(G, dtype=float)
    L = np.asarray(L, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rsi = 100.0 - 100.0 / (1.0 + G / L)
    rsi = np.where(L == 0.0, np.where(G == 0.0, 50.0, 100.0), rsi)
    return np.where(np.isnan(G) | np.isnan(L), np.nan, rsi)


def wilder_rsi(close, n):
    """Wilder RSI_n with SMA seeding (first value at t = n)."""
    return rsi_from_gl(*wilder_gl(close, n))


# ---------------------------------------------------------------- reference
def _columns(df):
    return tuple(df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close", "volume"))


def helpers_ref(df):
    """Intermediate series (r, E, sigma, sigmaL, SMAs, EMAs, MACDs, RSIs, volume means) on df.index."""
    _o, h, _l, c, v = _columns(df)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(c / lag(c, 1))
        r2 = r * r
        E = ewm_alpha(r2, E_ALPHA)
        sigma = np.maximum(SIGMA_FLOOR, np.sqrt(E))                 # np.maximum keeps NaN
        sigmaL = np.maximum(SIGMA_FLOOR, np.sqrt(roll_mean(r2, SIGMA_L_WINDOW)))
        macd = ema(c, 12) - ema(c, 26)
        macd_d = ema(c, 72) - ema(c, 156)
        out = {
            "r": r, "E": E, "sigma": sigma, "sigmaL": sigmaL,
            "sma20": roll_mean(c, 20), "sma50": roll_mean(c, 50), "sma200": roll_mean(c, 200),
            "sma300": roll_mean(c, 300), "sma1200": roll_mean(c, 1200),
            "macd": macd, "macd_sig": ema(macd, 9), "macd_d": macd_d, "macd_d_sig": ema(macd_d, 54),
            "rsi14": wilder_rsi(c, 14), "rsi84": wilder_rsi(c, 84),
            "vbar20": roll_mean(v, 20), "vbar180": roll_mean(v, 180),
            "B180_prev": lag(roll_mean(v, 180), 1),
            "hmax180": roll_max(h, 180),
        }
    return pd.DataFrame(out, index=df.index)


def compute_ref(df):
    """22 features, columns exactly NAMES (in order), index = df.index, float64."""
    _o, h, l, c, v = _columns(df)
    H = helpers_ref(df)
    r, sigma, sigmaL = (H[k].to_numpy() for k in ("r", "sigma", "sigmaL"))
    f = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        # returns
        f["ret_1"] = u(r / sigma)
        for k in (6, 42, 180):
            f[f"ret_{k}"] = u(np.log(c / lag(c, k)) / (sigma * np.sqrt(k)))
        # price vs moving averages
        for n in (20, 50, 200):
            f[f"ma_{n}"] = u(np.log(c / H[f"sma{n}"].to_numpy()) / (sigma * np.sqrt(n / 3.0)))
        f["ma_1200"] = u(np.log(c / H["sma1200"].to_numpy()) / (sigma * np.sqrt(400.0)))
        f["x_50_200"] = u(np.log(H["sma50"].to_numpy() / H["sma200"].to_numpy())
                          / (sigma * np.sqrt(200.0 / 3.0)))
        f["x_300_1200"] = u(np.log(H["sma300"].to_numpy() / H["sma1200"].to_numpy())
                            / (sigma * np.sqrt(400.0)))
        # RSI
        f["rsi_14"] = np.clip((H["rsi14"].to_numpy() - 50.0) / 25.0, -2.0, 2.0)
        f["rsi_84"] = np.clip((H["rsi84"].to_numpy() - 50.0) / 10.0, -2.0, 2.0)
        # MACD
        cs = c * sigma
        macd, sig = H["macd"].to_numpy(), H["macd_sig"].to_numpy()
        f["macd_4h"] = np.clip(macd / cs, -8.0, 8.0) / 4.0
        f["macdh_4h"] = u((macd - sig) / cs)
        f["macdh_1d"] = u((H["macd_d"].to_numpy() - H["macd_d_sig"].to_numpy()) / (cs * np.sqrt(6.0)))
        # volatility regime
        f["vol_regime"] = np.clip(np.log(sigma / sigmaL), -2.0, 2.0)
        # volume
        B = H["B180_prev"].to_numpy()                    # mean(V_{t-180..t-1}), current bar excluded
        x = np.log((v + 0.01 * B) / B)
        x = np.where(B == 0.0, 0.0, x)                   # B NaN (warm-up) stays NaN
        f["rel_vol"] = np.clip(x, -3.0, 3.0) / 1.5
        vb20, vb180 = H["vbar20"].to_numpy(), H["vbar180"].to_numpy()
        eps = 0.01 * vb180
        x = np.log((vb20 + eps) / (vb180 + eps))
        x = np.where(vb180 == 0.0, 0.0, x)
        f["vol_trend"] = np.clip(x, -3.0, 3.0) / 1.5
        signed = np.sign(c - lag(c, 1)) * v              # sign(0) = 0; NaN at t = 0
        f["flow_20"] = roll_sum(signed, 20) / (roll_sum(v, 20) + FLOW_EPS)
        # bar shape
        f["range"] = np.clip(np.log(h / l) / sigma, 0.0, 6.0) / 2.0 - 1.0
        hl = h - l
        f["clv"] = np.where(hl == 0.0, 0.0, (2.0 * c - h - l) / hl)
        f["dd_180"] = np.clip(np.log(c / H["hmax180"].to_numpy()) / (sigma * np.sqrt(180.0)),
                              -4.0, 0.0) / 2.0
    out = pd.DataFrame({k: np.asarray(f[k], dtype=float) for k in NAMES}, index=df.index)
    return out[NAMES]


# ---------------------------------------------------------------- synthetic data
def synthetic_bars(n=5000, seed=0):
    """Regime-switching GBM with correlated volume (RangeIndex; columns open/high/low/close/volume).

    Includes: an ultra-quiet stretch (sigma and sigmaL floors bind), a turbulent stretch, two large shocks,
    exact ties (C_t == C_{t-1}), flat bars (H == L) and a few isolated zero-volume bars.
    """
    rng = np.random.default_rng(seed)

    def q(frac):
        return int(frac * n)

    vol = np.full(n, 0.015)
    vol[q(0.30):q(0.46)] = 0.0004
    vol[q(0.60):q(0.68)] = 0.045
    r = 0.0002 + vol * rng.standard_normal(n)
    r[0] = 0.0
    r[q(0.50)] += 0.25
    r[q(0.80)] -= 0.30
    c = 20000.0 * np.exp(np.cumsum(r))
    for t in np.sort(rng.choice(np.arange(1, n), size=max(1, n // 80), replace=False)):
        c[t] = c[t - 1]
    o = np.empty(n)
    o[0] = c[0]
    o[1:] = c[:-1] * np.exp(0.1 * vol[1:] * rng.standard_normal(n - 1))
    h = np.maximum(o, c) * np.exp(0.6 * vol * np.abs(rng.standard_normal(n)))
    lo = np.minimum(o, c) * np.exp(-0.6 * vol * np.abs(rng.standard_normal(n)))
    flat = rng.choice(np.arange(1, n), size=max(1, n // 250), replace=False)
    o[flat] = c[flat]
    h[flat] = c[flat]
    lo[flat] = c[flat]
    rr = np.abs(np.diff(np.log(c), prepend=np.log(c[0])))
    v = np.exp(3.0 + 0.7 * rng.standard_normal(n)) * (1.0 + 40.0 * rr)
    v[rng.choice(np.arange(1, n), size=max(1, n // 100), replace=False)] *= 8.0
    v[rng.choice(np.arange(1, n), size=3, replace=False)] = 0.0
    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "volume": v})
