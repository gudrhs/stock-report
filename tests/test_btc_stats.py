"""Tests for btc/stats.py: python -m unittest tests.test_btc_stats -v"""
import math
import time
import unittest
from statistics import NormalDist

import numpy as np

from btc import stats as S

ND = NormalDist()


class NormalTests(unittest.TestCase):
    def test_ppf_matches_stdlib_reference(self):
        grid = np.concatenate([np.logspace(-300, -1, 600), np.linspace(0.001, 0.999, 2001),
                               1.0 - np.logspace(-15, -1, 200)])
        worst = max(abs(S.norm_ppf(p) - ND.inv_cdf(p)) / max(1.0, abs(ND.inv_cdf(p))) for p in grid)
        self.assertLess(worst, 1e-9)
        self.assertLess(worst, 1e-12)      # Halley refinement gives near machine precision

    def test_ppf_known_values(self):
        self.assertAlmostEqual(S.norm_ppf(0.975), 1.959963984540054, places=13)
        self.assertAlmostEqual(S.norm_ppf(0.95), 1.6448536269514722, places=13)
        self.assertAlmostEqual(S.norm_ppf(0.5), 0.0, places=15)
        self.assertAlmostEqual(S.norm_ppf(0.025), -1.959963984540054, places=13)
        self.assertEqual(S.norm_ppf(0.0), -math.inf)
        self.assertEqual(S.norm_ppf(1.0), math.inf)
        with self.assertRaises(ValueError):
            S.norm_ppf(1.5)

    def test_cdf_roundtrip_and_tails(self):
        for p in (1e-12, 1e-6, 0.01, 0.3, 0.5, 0.8, 0.999):
            self.assertAlmostEqual(S.norm_cdf(S.norm_ppf(p)) / p, 1.0, places=12)
        self.assertAlmostEqual(S.norm_cdf(1.0), 0.8413447460685429, places=15)
        self.assertAlmostEqual(S.norm_cdf(-8.0) / 6.22096057427178e-16, 1.0, places=10)


class SummaryTests(unittest.TestCase):
    def test_sharpe_constructed_series(self):
        r = np.array([0.01, -0.01, 0.02, 0.0])
        sd = math.sqrt(5e-4 / 3.0)
        self.assertAlmostEqual(S.sharpe(r, 1), 0.005 / sd, places=12)
        self.assertAlmostEqual(S.sharpe(r), 0.005 / sd * math.sqrt(365), places=10)
        self.assertEqual(S.sharpe(np.zeros(10)), 0.0)
        out = S.summary(r)
        self.assertAlmostEqual(out["sharpe"], 0.005 / sd * math.sqrt(365), places=10)
        self.assertAlmostEqual(out["sortino"], math.sqrt(365), places=10)   # downside dev = 0.005
        self.assertAlmostEqual(out["vol"], sd * math.sqrt(365), places=12)

    def test_hand_path_drawdown_tuw_twm(self):
        r = np.array([0.1, -0.5, 0.2, 0.5, 0.2, -0.1])
        # equity 1.1, 0.55, 0.66, 0.99, 1.188, 1.0692 ; peak 1.1 x4, 1.188 x2
        out = S.summary(r)
        self.assertAlmostEqual(out["max_dd"], -0.5, places=12)
        self.assertAlmostEqual(S.max_drawdown(r), -0.5, places=12)
        self.assertEqual(out["tuw_days"], 3)
        self.assertAlmostEqual(out["twm"], 1.0692, places=12)
        cagr = 1.0692 ** (365 / 6) - 1
        self.assertAlmostEqual(out["cagr"], cagr, places=9)
        self.assertAlmostEqual(out["calmar"], cagr / 0.5, places=9)
        self.assertEqual(out["n_days"], 6)

    def test_drawdown_from_start_and_unrecovered(self):
        out = S.summary(np.array([-0.1, 0.05, -0.02, 0.01]))
        self.assertEqual(out["tuw_days"], 4)                 # never recovers above 1.0
        self.assertAlmostEqual(out["max_dd"], -0.1, places=12)         # trough is day 1 (0.9)
        up = S.summary(np.full(10, 0.01))
        self.assertEqual(up["tuw_days"], 0)
        self.assertEqual(up["max_dd"], 0.0)

    def test_rejects_nan(self):
        with self.assertRaises(ValueError):
            S.summary(np.array([0.01, np.nan]))


class TradeStatsTests(unittest.TestCase):
    def test_hand_example(self):
        lc = math.log(0.99)
        pos = np.array([0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0], dtype=float)
        lr = np.array([0, lc + 0.05, 0.02, -0.01, lc, 0, lc - 0.03, 0.0, lc, 0, 0, 0])
        out = S.trade_stats(pos, lr)
        t1, t2 = math.expm1(0.06 + 2 * lc), math.expm1(-0.03 + 2 * lc)
        self.assertEqual(out["n_trades"], 2)
        self.assertAlmostEqual(out["exposure"], 5 / 12, places=12)
        self.assertAlmostEqual(out["switches_per_year"], 4 / (2 / 365), places=9)
        self.assertAlmostEqual(out["avg_hold_days"], 2.5 / 6, places=12)
        self.assertAlmostEqual(out["hit_rate"], 0.5, places=12)
        self.assertAlmostEqual(out["profit_factor"], t1 / -t2, places=12)
        self.assertAlmostEqual(out["avg_trade"], (t1 + t2) / 2, places=12)

    def test_open_trade_at_end_and_no_losses(self):
        pos = np.array([1, 1, 1, 1, 1, 1], dtype=float)
        lr = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
        out = S.trade_stats(pos, lr)
        self.assertEqual(out["n_trades"], 1)
        self.assertEqual(out["profit_factor"], math.inf)
        self.assertAlmostEqual(out["switches_per_year"], 365.0)   # one entry in one day
        self.assertAlmostEqual(out["avg_hold_days"], 1.0)


class PsrDsrTests(unittest.TestCase):
    def test_psr_two_point_exact(self):
        # symmetric two-point returns: skew 0, kurtosis 1 -> denominator exactly 1
        r = np.tile([0.02, -0.01], 50)
        sr, g3, g4 = S.moments(r)
        self.assertAlmostEqual(g3, 0.0, places=12)
        self.assertAlmostEqual(g4, 1.0, places=12)
        self.assertAlmostEqual(sr * math.sqrt(99), 3.3, places=12)
        self.assertAlmostEqual(S.psr(r, 0.0), ND.cdf(3.3), places=12)

    def test_psr_half_at_own_sharpe_and_monotone(self):
        rng = np.random.default_rng(3)
        r = rng.standard_t(4, 2000) * 0.02 + 0.001
        sr = S.sharpe(r, 1)
        self.assertAlmostEqual(S.psr(r, sr), 0.5, places=12)
        vals = [S.psr(r, s) for s in np.linspace(-0.1, 0.2, 31)]
        self.assertTrue(all(a > b for a, b in zip(vals, vals[1:])))
        # more data with the same SR -> more confidence
        self.assertGreater(S.psr(np.tile(r, 4), 0.0), S.psr(r, 0.0))

    def test_expected_max_sharpe_formula(self):
        v, n = 0.0004, 100
        ref = math.sqrt(v) * ((1 - 0.5772156649015329) * ND.inv_cdf(1 - 1 / n)
                              + 0.5772156649015329 * ND.inv_cdf(1 - 1 / (n * math.e)))
        self.assertAlmostEqual(S.expected_max_sharpe(v, n), ref, places=13)
        self.assertEqual(S.expected_max_sharpe(v, 1), 0.0)

    def test_dsr_decreasing_in_trials(self):
        rng = np.random.default_rng(5)
        r = rng.normal(0.002, 0.03, 3000)
        vals = [S.dsr(r, n, 0.0004) for n in (1, 2, 5, 10, 24, 50, 100, 1000)]
        self.assertAlmostEqual(vals[0], S.psr(r, 0.0), places=15)
        self.assertTrue(all(a > b for a, b in zip(vals, vals[1:])))

    def test_trials_sr_var(self):
        rng = np.random.default_rng(6)
        M = rng.normal(0, 0.03, (500, 7))
        srs = [S.sharpe(M[:, j], 1) for j in range(7)]
        self.assertAlmostEqual(S.trials_sr_var(M), np.var(srs, ddof=1), places=14)


class HolmTests(unittest.TestCase):
    def test_textbook(self):
        adj = S.holm([0.01, 0.04, 0.03, 0.005])
        np.testing.assert_allclose(adj, [0.03, 0.06, 0.06, 0.02], rtol=0, atol=1e-15)

    def test_monotone_capped(self):
        np.testing.assert_allclose(S.holm([0.5, 0.6]), [1.0, 1.0])
        adj = S.holm([0.001, 0.2, 0.01, 0.04, 0.9])
        order = np.argsort([0.001, 0.2, 0.01, 0.04, 0.9])
        self.assertTrue(np.all(np.diff(adj[order]) >= 0))
        self.assertTrue(np.all(adj <= 1.0))
        np.testing.assert_allclose(adj, [0.005, 0.4, 0.04, 0.12, 0.9])   # sorted: 5p, 4p, 3p, 2p, 1p


class BootstrapTests(unittest.TestCase):
    def test_indices_geometric_blocks(self):
        rng = np.random.default_rng(0)
        idx = S.stationary_bootstrap_indices(1000, 200, 20, rng)
        self.assertEqual(idx.shape, (200, 1000))
        self.assertTrue(idx.min() >= 0 and idx.max() < 1000)
        cont = (idx[:, 1:] == (idx[:, :-1] + 1) % 1000)
        new_rate = 1.0 - cont.mean()
        self.assertAlmostEqual(new_rate, 1 / 20, delta=0.003)       # mean block ~ 20
        counts = np.bincount(idx.ravel(), minlength=1000)
        self.assertLess(counts.std() / counts.mean(), 0.2)           # roughly uniform coverage

    def test_strongly_better_series(self):
        rng = np.random.default_rng(1)
        rb = rng.normal(0.0005, 0.03, 2000)
        ra = rb + 0.003 + rng.normal(0, 0.005, 2000)
        out = S.stationary_bootstrap_diff(ra, rb, n_boot=2000, seed=0)
        self.assertGreater(out["obs"], 0.5)
        self.assertLess(out["p"], 0.01)
        self.assertGreater(out["ci90"][0], 0.0)
        self.assertLess(out["ci90"][0], out["obs"])
        self.assertGreater(out["ci90"][1], out["obs"])

    def test_identical_distribution_p_spread(self):
        ps = []
        for s in range(30):
            rng = np.random.default_rng(100 + s)
            ra = rng.normal(0.001, 0.03, 600)
            rb = rng.normal(0.001, 0.03, 600)
            ps.append(S.stationary_bootstrap_diff(ra, rb, n_boot=500, seed=s)["p"])
        ps = np.array(ps)
        self.assertTrue(0.3 < ps.mean() < 0.7, ps.mean())
        self.assertLessEqual(np.mean(ps < 0.05), 0.2)
        self.assertGreater(ps.max() - ps.min(), 0.5)

    def test_same_series_and_determinism(self):
        rng = np.random.default_rng(2)
        r = rng.normal(0.001, 0.03, 300)
        out = S.stationary_bootstrap_diff(r, r, n_boot=200)
        self.assertEqual(out["obs"], 0.0)
        self.assertEqual(out["p"], 1.0)
        rb = rng.normal(0.0, 0.03, 300)
        a = S.stationary_bootstrap_diff(r, rb, n_boot=300, seed=7)
        b = S.stationary_bootstrap_diff(r, rb, n_boot=300, seed=7)
        self.assertEqual(a, b)

    def test_speed_10000_on_3500_days(self):
        rng = np.random.default_rng(3)
        rb = rng.normal(0.002, 0.04, 3500)
        ra = 0.6 * rb + rng.normal(0.0005, 0.02, 3500)
        t0 = time.time()
        out = S.stationary_bootstrap_diff(ra, rb, mean_block=20, n_boot=10000, seed=0)
        self.assertLess(time.time() - t0, 30.0)
        self.assertEqual(out["n_boot"], 10000)
        self.assertTrue(0.0 <= out["p"] <= 1.0)


class LedoitWolfTests(unittest.TestCase):
    def test_iid_normal_matches_closed_form(self):
        # bandwidth 0 -> iid delta method; for bivariate normal returns the asymptotic
        # variance of SR_a - SR_b is (2 - 2 rho + 0.5 (SRa^2 + SRb^2 - 2 SRa SRb rho^2)) / T
        rng = np.random.default_rng(4)
        T, rho = 40000, 0.5
        z1, z2 = rng.standard_normal(T), rng.standard_normal(T)
        ra = 0.3 + z1
        rb = 0.1 + rho * z1 + math.sqrt(1 - rho ** 2) * z2
        out = S.lw_hac_test(ra, rb, bandwidth=0, periods=1)
        sa, sb = 0.3, 0.1
        se_ref = math.sqrt((2 - 2 * rho + 0.5 * (sa ** 2 + sb ** 2 - 2 * sa * sb * rho ** 2)) / T)
        self.assertAlmostEqual(out["se"] / se_ref, 1.0, delta=0.05)
        self.assertAlmostEqual(out["diff"], ra.mean() / ra.std() - rb.mean() / rb.std(), places=12)

    def test_identical_and_strong(self):
        rng = np.random.default_rng(5)
        rb = rng.normal(0.0005, 0.03, 2000)
        same = S.lw_hac_test(rb, rb.copy())
        self.assertEqual(same["diff"], 0.0)
        self.assertAlmostEqual(same["p_one_sided"], 0.5)
        ra = rb + 0.003 + rng.normal(0, 0.005, 2000)
        out = S.lw_hac_test(ra, rb)
        self.assertGreater(out["diff"], 0.5)
        self.assertLess(out["p_one_sided"], 0.01)
        self.assertGreater(out["bandwidth"], 0.0)
        worse = S.lw_hac_test(rb, ra)
        self.assertGreater(worse["p_one_sided"], 0.99)

    def test_agrees_with_bootstrap_se(self):
        rng = np.random.default_rng(6)
        rb = rng.normal(0.002, 0.04, 3000)
        ra = 0.5 * rb + rng.normal(0.001, 0.02, 3000)
        lw = S.lw_hac_test(ra, rb)
        bs = S.stationary_bootstrap_diff(ra, rb, n_boot=3000, seed=1)
        self.assertAlmostEqual(lw["se"] / bs["se"], 1.0, delta=0.2)


class CscvTests(unittest.TestCase):
    def test_pure_noise_pbo_high(self):
        pbos = []
        for s in range(40):
            M = np.random.default_rng(s).normal(0, 0.03, (800, 12))
            pbos.append(S.pbo_cscv(M, S=16)["pbo"])
        self.assertGreater(np.mean(pbos), 0.4)

    def test_persistent_edge_pbo_low(self):
        rng = np.random.default_rng(11)
        M = rng.normal(0, 0.03, (1600, 12))
        M[:, 3] += 0.006                                  # genuine, persistent edge in variant 3
        out = S.pbo_cscv(M, S=16)
        self.assertLess(out["pbo"], 0.2)
        self.assertEqual(out["n_splits"], 12870)
        self.assertEqual(len(out["logits"]), 12870)
        self.assertAlmostEqual(float(np.median(out["logits"])), math.log(12), places=12)
        self.assertEqual(out["p_oos_loss"], 0.0)

    def test_remainder_rows_dropped_at_start(self):
        rng = np.random.default_rng(12)
        core = rng.normal(0, 0.03, (16 * 20, 12))
        junk = rng.normal(0, 5.0, (7, 12))
        a = S.pbo_cscv(core, S=16)
        b = S.pbo_cscv(np.vstack([junk, core]), S=16)
        self.assertEqual(b["rows_used"], 320)
        self.assertEqual(a["pbo"], b["pbo"])
        np.testing.assert_allclose(a["logits"], b["logits"])

    def test_sharpes_match_direct_computation(self):
        rng = np.random.default_rng(13)
        M = rng.normal(0.001, 0.03, (8 * 30, 5))
        out = S.pbo_cscv(M, S=8)
        # first combination = blocks 0..3 in-sample
        is_rows, oos_rows = M[:120], M[120:]
        srs_is = [S.sharpe(is_rows[:, j]) for j in range(5)]
        best = int(np.argmax(srs_is))
        self.assertAlmostEqual(out["is_sr"][0], srs_is[best], places=9)
        self.assertAlmostEqual(out["oos_sr"][0], S.sharpe(oos_rows[:, best]), places=9)

    def test_speed_s16_n12(self):
        M = np.random.default_rng(14).normal(0, 0.03, (2922, 12))
        t0 = time.time()
        S.pbo_cscv(M, S=16)
        self.assertLess(time.time() - t0, 60.0)


class NullTests(unittest.TestCase):
    def test_markov_probs(self):
        p, q = S.markov_probs(0.6, 0.02)
        self.assertAlmostEqual(p / (p + q), 0.6, places=14)
        self.assertAlmostEqual(2 * 0.4 * p, 0.02, places=14)
        with self.assertRaises(ValueError):
            S.markov_probs(0.9, 0.5)

    def test_markov_calibration(self):
        e, s = 0.62, 0.015
        a = S.markov_positions(4, 200000, e, s, seed=1).astype(float)
        self.assertAlmostEqual(a.mean(), e, delta=0.02)
        sw = np.abs(np.diff(a, axis=1)).mean()
        self.assertAlmostEqual(sw / s, 1.0, delta=0.05)
        exp_, rate = S.exposure_switch_rate(a[0])
        self.assertAlmostEqual(exp_, e, delta=0.03)
        self.assertAlmostEqual(rate / s, 1.0, delta=0.1)

    def test_null_markov_matches_manual_path(self):
        rng = np.random.default_rng(2)
        T = 6 * 60
        m = rng.normal(0.0002, 0.015, T)
        day_id = np.arange(T) // 6
        day_id[day_id >= 30] += 2                          # two empty (outage) days
        c, e, s = 0.0015, 0.5, 0.05
        sh = S.null_markov(m, c, e, s, n=5, seed=9, day_id=day_id)
        pos = S.markov_positions(5, T, e, s, seed=9).astype(float)
        lc = math.log(1 - c)
        for i in range(5):
            a = pos[i]
            prev = np.concatenate([[0.0], a[:-1]])
            g = a * m + np.abs(a - prev) * lc
            g[-1] += a[-1] * lc
            daily = np.zeros(62)
            np.add.at(daily, day_id, g)
            self.assertAlmostEqual(sh[i], S.sharpe(np.expm1(daily)), places=10)
        # per-bar annualisation when no day map is given
        sh_bar = S.null_markov(m, c, e, s, n=5, seed=9)
        a = pos[0]
        g = a * m + np.abs(a - np.concatenate([[0.0], a[:-1]])) * lc
        g[-1] += a[-1] * lc
        self.assertAlmostEqual(sh_bar[0], S.sharpe(np.expm1(g), 6 * 365), places=10)

    def test_null_markov_costs_hurt(self):
        rng = np.random.default_rng(3)
        m = rng.normal(0.0, 0.015, 6 * 400)
        cheap = S.null_markov(m, 0.0, 0.5, 0.2, n=200, seed=1)
        dear = S.null_markov(m, 0.01, 0.5, 0.2, n=200, seed=1)
        self.assertTrue(np.all(dear < cheap))

    def test_circular_shift(self):
        rng = np.random.default_rng(4)
        T = 61
        r = rng.normal(0.001, 0.03, T)
        pos = (rng.random(T) < 0.5).astype(float)
        null = S.null_circular_shift(pos, r, n=200, min_shift=30, seed=0)
        allowed = {round(S.sharpe(np.roll(pos, k) * r), 12) for k in (30, 31)}
        self.assertTrue({round(v, 12) for v in null} <= allowed)
        # perfect-foresight positions: shifted versions lose the edge
        r = rng.normal(0.0, 0.03, 2000)
        pos = (r > 0).astype(float)
        obs = S.sharpe(pos * r)
        null = S.null_circular_shift(pos, r, n=500, min_shift=30, seed=1)
        self.assertLess(null.max(), obs / 3)
        self.assertEqual(S.null_pvalue(obs, null), 1 / 501)
        self.assertEqual(S.null_percentile(obs, null), 1.0)


class NeweyWestTests(unittest.TestCase):
    def test_beta_alpha_recovered(self):
        rng = np.random.default_rng(7)
        T = 5000
        x = rng.normal(0.001, 0.04, T)
        u = np.zeros(T)
        eps = rng.normal(0, 0.01, T)
        for t in range(1, T):
            u[t] = 0.3 * u[t - 1] + eps[t]
        y = 0.0008 + 0.6 * x + u
        out = S.newey_west_alpha_beta(y, x, lags=10)
        self.assertAlmostEqual(out["beta"], 0.6, delta=0.02)
        self.assertAlmostEqual(out["alpha_annual"], 0.0008 * 365, delta=3 * out["se_alpha_annual"])
        self.assertAlmostEqual(out["alpha_annual"], out["alpha_daily"] * 365, places=12)
        self.assertAlmostEqual(out["t_alpha"], out["alpha_annual"] / out["se_alpha_annual"], places=10)
        white = S.newey_west_alpha_beta(y, x, lags=0)
        self.assertLess(white["se_alpha_annual"], out["se_alpha_annual"])  # positive autocorr -> NW wider

    def test_lag0_equals_hc0(self):
        rng = np.random.default_rng(8)
        x = rng.normal(0, 1, 300)
        y = 0.1 + 0.5 * x + rng.normal(0, 1, 300) * (1 + np.abs(x))
        out = S.newey_west_alpha_beta(y, x, lags=0, periods=1)
        X = np.column_stack([np.ones(300), x])
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        e = y - X @ b
        inv = np.linalg.inv(X.T @ X)
        V = inv @ (X.T * e ** 2) @ X @ inv * 300 / 298
        self.assertAlmostEqual(out["beta"], b[1], places=12)
        self.assertAlmostEqual(out["se_beta"], math.sqrt(V[1, 1]), places=12)
        self.assertAlmostEqual(out["se_alpha_annual"], math.sqrt(V[0, 0]), places=12)


if __name__ == "__main__":
    unittest.main()
