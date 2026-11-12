# -*- coding: utf-8 -*-
"""독립 기준 구현(tests/btc_features_ref.py) 자체 검증 + 증분 엔진(btc.features)과의 대조.

python -m unittest tests.test_btc_features_ref -v

1. hand-computable self-tests of the reference,
2. engine (btc.features.compute) vs reference: max |diff| < 1e-9 for t >= 1200 on 5000 synthetic bars and,
   when the 15m cache exists, on the real phase-0 4h bars (skipped while btc.features / btc.data lack the names),
3. no-lookahead of the reference: truncating / garbling the input after t leaves features[<= t] bitwise equal.
"""
import math
import os
import sys
import unittest

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import btc_features_ref as ref  # noqa: E402

try:
    from btc import features as engine  # noqa: E402
    ENGINE_OK = hasattr(engine, "compute") and hasattr(engine, "NAMES")
    ENGINE_WHY = "btc.features lacks compute/NAMES"
except Exception as exc:  # not written yet / mid-edit
    engine, ENGINE_OK, ENGINE_WHY = None, False, f"btc.features not importable: {exc!r}"

try:
    from btc import data as bdata  # noqa: E402
    DATA_OK = hasattr(bdata, "load_15m") and hasattr(bdata, "bars_4h")
except Exception:
    bdata, DATA_OK = None, False
DATA_FILE = os.path.join(ROOT, "data", "btc", "btcusd_15m.csv.gz")
REAL_OK = ENGINE_OK and DATA_OK and os.path.exists(DATA_FILE)

T0 = 1200          # comparisons start here
TOL = 1e-9


def bars(close, high=None, low=None, volume=None, open_=None, index=None):
    c = np.asarray(close, dtype=float)
    return pd.DataFrame({
        "open": c if open_ is None else np.asarray(open_, dtype=float),
        "high": c if high is None else np.asarray(high, dtype=float),
        "low": c if low is None else np.asarray(low, dtype=float),
        "close": c,
        "volume": np.ones_like(c) if volume is None else np.asarray(volume, dtype=float),
    }, index=index)


def u(x):
    return max(-4.0, min(4.0, x)) / 2.0


# =================================================================== 1. self-tests
class ReferenceHandTests(unittest.TestCase):

    def test_columns_index_dtype(self):
        df = ref.synthetic_bars(300, seed=3)
        df.index = pd.date_range("2020-01-01", periods=len(df), freq="4h", tz="UTC")
        df["junk"] = "x"                                     # extra columns ignored
        F = ref.compute_ref(df)
        self.assertEqual(list(F.columns), ref.NAMES)
        self.assertEqual(len(ref.NAMES), 22)
        self.assertTrue(F.index.equals(df.index))
        self.assertTrue(all(F[k].dtype == np.float64 for k in ref.NAMES))

    def test_warmup_nan_pattern(self):
        F = ref.compute_ref(ref.synthetic_bars(1300, seed=4))
        for k in ref.NAMES:
            col = F[k].to_numpy()
            first = ref.FIRST_DEFINED[k]
            self.assertTrue(np.isnan(col[:first]).all(), k)
            self.assertTrue(np.isfinite(col[first:]).all(), k)

    def test_constant_price(self):
        T = 1300
        df = bars(np.full(T, 100.0), high=np.full(T, 101.0), low=np.full(T, 99.0), volume=np.full(T, 5.0))
        F = ref.compute_ref(df).iloc[T0:]
        H = ref.helpers_ref(df).iloc[T0:]
        self.assertTrue((H["sigma"] == 0.002).all() and (H["sigmaL"] == 0.002).all())
        for k in ("ret_1", "ret_6", "ret_42", "ret_180", "ma_20", "ma_50", "ma_200", "ma_1200",
                  "x_50_200", "x_300_1200", "rsi_14", "rsi_84", "macd_4h", "macdh_4h", "macdh_1d",
                  "vol_regime", "vol_trend", "flow_20", "clv"):
            np.testing.assert_array_equal(F[k].to_numpy(), 0.0, err_msg=k)   # RSI: G == L == 0 -> 50
        np.testing.assert_allclose(F["rel_vol"], math.log(1.01) / 1.5, rtol=0, atol=1e-15)
        np.testing.assert_allclose(F["range"], 2.0, rtol=0, atol=0)          # ln(101/99)/0.002 = 10 -> 6
        dd = max(-4.0, math.log(100 / 101) / (0.002 * math.sqrt(180))) / 2
        np.testing.assert_allclose(F["dd_180"], dd, rtol=0, atol=1e-14)

    def test_exponential_trend(self):
        g, T = 0.01, 1300
        t = np.arange(T)
        df = bars(100.0 * np.exp(g * t))
        F = ref.compute_ref(df)
        H = ref.helpers_ref(df)
        np.testing.assert_allclose(H["sigma"].iloc[1:], g, rtol=1e-12)
        np.testing.assert_allclose(F["vol_regime"].iloc[T0:], 0.0, atol=1e-12)
        at = F.iloc[T0:]
        np.testing.assert_allclose(at["ret_1"], 0.5, atol=1e-12)
        np.testing.assert_allclose(at["ret_6"], math.sqrt(6) / 2, atol=1e-12)
        np.testing.assert_allclose(at["ret_42"], 2.0, atol=0)
        np.testing.assert_allclose(at["ret_180"], 2.0, atol=0)

        def lsma(n):   # ln(C_t / SMA_n) on a pure exponential
            return -math.log((1 - math.exp(-g * n)) / (n * (1 - math.exp(-g))))
        for n in (20, 50, 200):
            np.testing.assert_allclose(at[f"ma_{n}"], u(lsma(n) / (g * math.sqrt(n / 3))), atol=1e-9)
        np.testing.assert_allclose(at["ma_1200"], u(lsma(1200) / (g * 20)), atol=1e-9)
        np.testing.assert_allclose(at["x_50_200"], u((lsma(200) - lsma(50)) / (g * math.sqrt(200 / 3))),
                                   atol=1e-9)
        np.testing.assert_allclose(at["x_300_1200"], u((lsma(1200) - lsma(300)) / (g * 20)), atol=1e-9)
        np.testing.assert_array_equal(at["rsi_14"], 2.0)                     # L == 0 -> RSI 100
        np.testing.assert_array_equal(at["rsi_84"], 2.0)
        np.testing.assert_array_equal(at["dd_180"], 0.0)                     # H = C = running max
        self.assertTrue((at["macd_4h"] > 0).all() and (at["macdh_4h"] >= -1e-12).all())

    def test_rsi_monotone(self):
        up = ref.compute_ref(bars(100.0 + np.arange(200.0)))
        dn = ref.compute_ref(bars(1000.0 - np.arange(200.0)))
        self.assertTrue(np.isnan(up["rsi_14"].iloc[:14]).all())
        np.testing.assert_array_equal(up["rsi_14"].iloc[14:], 2.0)
        np.testing.assert_array_equal(up["rsi_84"].iloc[84:], 2.0)
        np.testing.assert_array_equal(dn["rsi_14"].iloc[14:], -2.0)          # G == 0 -> RSI 0
        np.testing.assert_array_equal(dn["rsi_84"].iloc[84:], -2.0)
        np.testing.assert_allclose(up["flow_20"].iloc[20:], 1.0, atol=1e-11)
        np.testing.assert_allclose(dn["flow_20"].iloc[20:], -1.0, atol=1e-11)

    def test_wilder_rsi_hand(self):
        # d = +1, -0.5, +1, -0.5, +1 ; n = 3
        rsi = ref.wilder_rsi([10, 11, 10.5, 11.5, 11, 12], 3)
        self.assertTrue(np.isnan(rsi[:3]).all())
        # t=3: G=2/3, L=1/6 -> RS=4 ; t=4: G=4/9, L=5/18 -> RS=1.6 ; t=5: G=17/27, L=5/27 -> RS=3.4
        np.testing.assert_allclose(rsi[3:], [80.0, 100 - 100 / 2.6, 100 - 100 / 4.4], rtol=0, atol=1e-12)
        np.testing.assert_array_equal(ref.wilder_rsi(np.full(10, 7.0), 3)[3:], 50.0)

    def test_wilder_rsi_matches_alpha_form(self):
        # independent check of the loop: seed at t=n then ewm(alpha=1/n) over the later gains/losses
        rng = np.random.default_rng(7)
        c = 100 * np.exp(np.cumsum(0.01 * rng.standard_normal(3000)))
        for n in (14, 84):
            d = np.diff(c)
            gain, loss = np.maximum(d, 0), np.maximum(-d, 0)
            sg = np.concatenate([[gain[:n].mean()], gain[n:]])
            sl = np.concatenate([[loss[:n].mean()], loss[n:]])
            G = ref.ewm_alpha(sg, 1.0 / n)
            L = ref.ewm_alpha(sl, 1.0 / n)
            want = 100 - 100 / (1 + G / L)
            np.testing.assert_allclose(ref.wilder_rsi(c, n)[n:], want, rtol=0, atol=1e-10)

    def test_primitives_match_literal_definitions(self):
        df = ref.synthetic_bars(1500, seed=11)
        c = df["close"].to_numpy()
        H = ref.helpers_ref(df)
        # E_t: literal recursion seeded E_1 = r_1^2
        r = np.log(c[1:] / c[:-1])
        E = np.empty_like(r)
        E[0] = r[0] ** 2
        for i in range(1, len(r)):
            E[i] = (1 - 2 / 43) * E[i - 1] + (2 / 43) * r[i] ** 2
        self.assertTrue(np.isnan(H["E"].iloc[0]))
        np.testing.assert_allclose(H["E"].iloc[1:], E, rtol=1e-12, atol=0)
        # EMA seeded with first close, MACD signal seeded with MACD_0 = 0
        for n in (12, 156):
            e = np.empty_like(c)
            e[0] = c[0]
            for i in range(1, len(c)):
                e[i] = (1 - 2 / (n + 1)) * e[i - 1] + 2 / (n + 1) * c[i]
            np.testing.assert_allclose(ref.ema(c, n), e, rtol=1e-13, atol=0)
        self.assertEqual(H["macd"].iloc[0], 0.0)
        self.assertEqual(H["macd_sig"].iloc[0], 0.0)
        # exact windows vs brute force
        for t in (539, 540, 777, 1199, 1499):
            if t >= 540:
                want = max(0.002, math.sqrt(np.mean(r[t - 540:t] ** 2)))   # r[i-1] = r_i -> r_{t-539..t}
                self.assertAlmostEqual(H["sigmaL"].iloc[t], want, delta=1e-15)
            else:
                self.assertTrue(np.isnan(H["sigmaL"].iloc[t]))
            if t >= 1199:
                self.assertAlmostEqual(H["sma1200"].iloc[t] / np.mean(c[t - 1199:t + 1]), 1.0, delta=1e-14)
            self.assertAlmostEqual(H["hmax180"].iloc[t], df["high"].iloc[t - 179:t + 1].max(), delta=0)

    def test_clv_and_range_hand_bar(self):
        T = 50                        # constant close -> sigma = 0.002 (floor) exactly
        c = np.full(T, 103.0)
        h, lo = c.copy(), c.copy()
        h[30], lo[30] = 103.3, 102.9                                          # C=103 -> clv = -0.5
        h[31], lo[31] = 104.0, 100.0                                          # clv = +0.5, range clipped
        F = ref.compute_ref(bars(c, high=h, low=lo))
        self.assertAlmostEqual(F["clv"].iloc[30], -0.5, delta=1e-12)
        self.assertAlmostEqual(F["range"].iloc[30], math.log(103.3 / 102.9) / 0.002 / 2 - 1, delta=1e-12)
        self.assertEqual(F["clv"].iloc[31], 0.5)
        self.assertEqual(F["range"].iloc[31], 2.0)                            # ln(1.04)/0.002 = 19.6 -> 6
        self.assertEqual(F["clv"].iloc[29], 0.0)                              # H == L
        self.assertEqual(F["range"].iloc[29], -1.0)

    def test_dd_180_hand_path(self):
        T, t0, t1 = 900, 300, 600
        h = np.full(T, 100.0)
        h[t0], h[t1] = 110.0, 150.0
        F = ref.compute_ref(bars(np.full(T, 100.0), high=h))["dd_180"].to_numpy()
        want = math.log(100 / 110) / (0.002 * math.sqrt(180)) / 2             # -1.776 (sigma floor)
        self.assertEqual(F[t0 - 1], 0.0)
        np.testing.assert_allclose(F[t0:t0 + 180], want, rtol=0, atol=1e-14)  # current bar H counts
        self.assertEqual(F[t0 + 180], 0.0)                                    # window H_{t-179..t}
        np.testing.assert_array_equal(F[t1:t1 + 180], -2.0)                   # clip at -4 -> -2

    def test_rel_vol_excludes_current_bar(self):
        T, t0 = 500, 200
        v = np.full(T, 10.0)
        v[t0] = 30.0
        F = ref.compute_ref(bars(np.full(T, 100.0), volume=v))
        rv, vt = F["rel_vol"].to_numpy(), F["vol_trend"].to_numpy()
        self.assertAlmostEqual(rv[t0], math.log(30.1 / 10.0) / 1.5, delta=1e-14)   # B = 10: current excluded
        B = (179 * 10 + 30) / 180
        for t in (t0 + 1, t0 + 180):                                          # t0 still inside the window
            self.assertAlmostEqual(rv[t], math.log((10 + 0.01 * B) / B) / 1.5, delta=1e-14)
        self.assertAlmostEqual(rv[t0 + 181], math.log(1.01) / 1.5, delta=1e-14)
        vb20, vb180 = (19 * 10 + 30) / 20, B                                  # Vbar includes current bar
        eps = 0.01 * vb180
        self.assertAlmostEqual(vt[t0], math.log((vb20 + eps) / (vb180 + eps)) / 1.5, delta=1e-14)
        self.assertTrue(np.isnan(rv[179]) and np.isfinite(rv[180]))
        # B == 0 -> 0 ; Vbar180 == 0 -> 0
        v0 = np.zeros(T)
        v0[t0] = 5.0
        F0 = ref.compute_ref(bars(np.full(T, 100.0), volume=v0))
        self.assertEqual(F0["rel_vol"].iloc[t0], 0.0)
        self.assertEqual(F0["vol_trend"].iloc[t0 - 1], 0.0)
        self.assertEqual(F0["rel_vol"].iloc[t0 + 1], np.clip(math.log(0.01 * (5 / 180) / (5 / 180)), -3, 3) / 1.5)

    def test_flow_20_hand(self):
        T = 60
        t = np.arange(T)
        c = 100.0 + (t % 2)                                                   # up at odd t, down at even t
        v = np.where(t % 2 == 1, 2.0, 1.0)                                    # up bars carry volume 2
        F = ref.compute_ref(bars(c, volume=v))["flow_20"].to_numpy()
        self.assertTrue(np.isnan(F[19]))
        np.testing.assert_allclose(F[20:], 10.0 / (30.0 + 1e-12), rtol=1e-15)
        c2 = c.copy()
        c2[40] = c2[39]                                                       # a tie: sign(0) = 0
        F2 = ref.compute_ref(bars(c2, volume=v))["flow_20"].to_numpy()
        # window t-19..t with t=45 contains bar 40 (was a down bar with v=1 -> now 0) and bar 41 (up from
        # 101 to 101 -> tie too, v=2 -> 0)
        self.assertAlmostEqual(F2[45], (10 * 2 - 10 * 1 + 1 - 2) / 30.0, delta=1e-12)


# =================================================================== 3. no-lookahead of the reference
class ReferenceNoLookaheadTests(unittest.TestCase):

    def test_truncation_and_garbage_after_t(self):
        df = ref.synthetic_bars(1600, seed=21)
        full = ref.compute_ref(df).to_numpy()
        rng = np.random.default_rng(5)
        cuts = sorted({0, 1, 13, 14, 19, 20, 83, 84, 179, 180, 539, 540, 1198, 1199, 1200, 1599}
                      | set(rng.integers(0, 1600, 12).tolist()))
        for t in cuts:
            trunc = ref.compute_ref(df.iloc[:t + 1]).to_numpy()
            np.testing.assert_array_equal(trunc, full[:t + 1], err_msg=f"truncated at t={t}")
            if t + 1 < len(df):
                g = df.copy()
                n = len(g) - t - 1
                g.iloc[t + 1:, g.columns.get_loc("close")] *= rng.uniform(0.2, 5.0, n)
                g.iloc[t + 1:, g.columns.get_loc("high")] = np.nan
                g.iloc[t + 1:, g.columns.get_loc("low")] *= rng.uniform(0.2, 5.0, n)
                g.iloc[t + 1:, g.columns.get_loc("volume")] = 0.0
                g.iloc[t + 1::3, g.columns.get_loc("close")] = np.nan
                garb = ref.compute_ref(g).to_numpy()
                np.testing.assert_array_equal(garb[:t + 1], full[:t + 1], err_msg=f"garbage after t={t}")


# =================================================================== 2. engine vs reference
def diff_report(X, R, names, t0=T0):
    """Per-feature max |X - R| over t >= t0 (NaN/inf mismatch -> inf) and a printable table."""
    worst, lines = {}, []
    for j, k in enumerate(names):
        e, r = X[t0:, j], R[t0:, j]
        d = np.abs(e - r)
        d[~np.isfinite(d)] = np.inf
        i = int(np.argmax(d))
        worst[k] = float(d[i])
        lines.append(f"  {k:>11s}  max|diff|={d[i]:.3e}  at t={t0 + i}  engine={e[i]!r}  ref={r[i]!r}")
    return worst, "\n".join(lines)


@unittest.skipUnless(ENGINE_OK, ENGINE_WHY)
class EngineVsReferenceTests(unittest.TestCase):

    def check(self, df, label):
        X, sigma = engine.compute(df)
        X = np.asarray(X, dtype=float)
        R = ref.compute_ref(df)
        self.assertEqual(X.shape, (len(df), 22))
        self.assertTrue(np.isfinite(R.to_numpy()[T0:]).all(), "reference must be finite for t >= 1200")
        worst, table = diff_report(X, R[ref.NAMES].to_numpy(), ref.NAMES)
        bad = [k for k in ref.NAMES if not worst[k] < TOL]
        self.assertFalse(bad, f"{label}: features with max|engine-ref| >= {TOL} for t >= {T0}: {bad}\n{table}")
        s_ref = ref.helpers_ref(df)["sigma"].to_numpy()[T0:]
        s_eng = np.asarray(sigma, dtype=float)[T0:]
        ds = np.abs(s_eng - s_ref)
        ds[~np.isfinite(ds)] = np.inf
        self.assertLess(float(ds.max()), 1e-12, f"{label}: sigma max|diff| = {ds.max():.3e}")

    def test_names_match(self):
        self.assertEqual(list(engine.NAMES), ref.NAMES)

    def test_synthetic_gbm_5000(self):
        self.check(ref.synthetic_bars(5000, seed=2026), "synthetic GBM 5000")

    @unittest.skipUnless(REAL_OK, "real 15m cache or btc.data.load_15m/bars_4h unavailable")
    def test_real_phase0_4h(self):
        b = bdata.bars_4h(bdata.load_15m(), 0)
        self.assertGreater(len(b), T0 + 100)
        self.check(b, f"real phase-0 4h bars (T={len(b)})")


if __name__ == "__main__":
    unittest.main()
