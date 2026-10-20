"""W2 'dp_band' (btc/research/algos/dp_band.py) 단위 테스트 — 합성 데이터만, 1스레드 약 10초.

python -m unittest tests.test_btc_dp_band -v
검토 후 추가: 무한 문턱 기록(Logging), 독립 검증 도구(Independent), 저장 기록만으로 경로 복원(Reconstruct).
"""
import itertools
import json
import math
import os
import unittest

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from btc.env import PhaseData, BAR_SEC
from btc.walkforward import decision_range
from btc.research.algos import dp_band as M
from btc.research.rl import feat_index
from btc.research.variants import TREND8
from btc.research.walk import decision_mask

T0 = int(pd.Timestamp("2013-01-01", tz="UTC").timestamp())
T_END = int(pd.Timestamp("2017-03-05", tz="UTC").timestamp())


def _df(seed, offset=0):
    """합성 4시간봉 + 끈질긴 합성 지표 22개 (phase offset초만큼 밀림)"""
    n = (T_END - T0) // BAR_SEC
    rng = np.random.default_rng(seed)
    X = np.empty((n, 22))
    X[0] = rng.standard_normal(22)
    e = rng.standard_normal((n, 22))
    for t in range(1, n):
        X[t] = 0.995 * X[t - 1] + 0.1 * e[t]
    r = 0.0002 + 0.0005 * X[:, feat_index(["ma_200"])[0]] + 0.012 * rng.standard_normal(n)   # 약한 신호
    o = 1000 * np.exp(np.concatenate([[0], np.cumsum(r)[:-1]]))
    c = o * np.exp(r)
    df = pd.DataFrame(dict(ts=T0 + offset + BAR_SEC * np.arange(n), open=o, high=np.maximum(o, c) * 1.001,
                           low=np.minimum(o, c) * 0.999, close=c, volume=1.0, gap_before=0, forced_hold=False))
    return df, X


def _phase(seed, offset=0, df=None, X=None):
    if df is None:
        df, X = _df(seed, offset)
    return PhaseData(df, X.astype(np.float32), np.full(len(df), 0.01))


def _cfg(phases=2):
    from btc.research.dp_band_controls import proposed_cfg
    return dict(proposed_cfg(), phases=phases)


class Setup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.datas = [_phase(1, 0), _phase(2, 900)]
        cls.fi = feat_index(TREND8)


class LabelPurge(Setup):
    def test_no_label_beyond_Tk(self):
        Tk = int(pd.Timestamp("2016-06-01", tz="UTC").timestamp())
        X, y, idx, ends = M.make_samples(self.datas, Tk, self.fi, 20)
        self.assertGreater(len(y), 1000)
        self.assertTrue(np.all(ends <= Tk))                             # 레이블 끝 봉이 T_k 전에 마감
        self.assertGreater(ends.max(), Tk - 2 * 86400)                  # 너무 많이 버리지도 않음
        span = 20 * 6
        for (k, t), yy, ee in zip(idx[::97], y[::97], ends[::97]):
            d = self.datas[k]
            self.assertTrue(decision_mask(d, 6)[t])                       # 하루 판단봉
            self.assertGreaterEqual(d.ts[t], M.FIRST_TRAIN)
            self.assertGreaterEqual(t, M.WARMUP)
            self.assertEqual(ee, d.ts[t + 1 + span] + BAR_SEC)
            self.assertAlmostEqual(yy, math.log(d.o[t + 1 + span] / d.o[t + 1]) / 20, places=12)
        # 퍼지 없이 세면 T_k를 넘는 레이블이 있어야 (필터가 실제로 작동)
        X2, y2, _, e2 = M.make_samples(self.datas, Tk + 60 * 86400, self.fi, 20)
        self.assertTrue(np.any(e2 > Tk))

    def test_future_data_does_not_change_model(self):
        """T_k 이후 가격·지표를 망가뜨려도 T_k 모델은 똑같아야 (미래 정보 없음)"""
        Tk = int(pd.Timestamp("2016-06-01", tz="UTC").timestamp())
        bad = []
        for seed, off in ((1, 0), (2, 900)):
            df, X = _df(seed, off)
            fut = (df["ts"].to_numpy() + BAR_SEC) > Tk
            df.loc[fut, ["open", "high", "low", "close"]] *= 3.0
            X[fut] = 5.0
            bad.append(_phase(0, df=df, X=X))
        cfg = _cfg()
        m1, i1 = M.fit_month(cfg, self.datas, Tk)
        m2, i2 = M.fit_month(cfg, bad, Tk)
        np.testing.assert_array_equal(m1.beta, m2.beta)
        self.assertEqual(i1["q_in"], i2["q_in"])
        self.assertEqual(i1["q_out"], i2["q_out"])
        self.assertEqual(i1["ar_phi"], i2["ar_phi"])


class ValueIteration(unittest.TestCase):
    @staticmethod
    def brute(grid, P, kappa, gamma):
        """2상태 × 이전 포지션 2 = 4 결합 상태, 결정론적 정책 16개 전부 정확히 평가 → 원소별 최대"""
        states = [(i, prev) for i in range(2) for prev in (0, 1)]
        Vs = []
        for pol in itertools.product((0, 1), repeat=4):
            T = np.zeros((4, 4))
            r = np.zeros(4)
            for s, (i, prev) in enumerate(states):
                a = pol[s]
                r[s] = a * grid[i] - kappa * abs(a - prev)
                for j in range(2):
                    T[s, states.index((j, a))] += P[i, j]
            Vs.append(np.linalg.solve(np.eye(4) - gamma * T, r))
        Vb = np.max(Vs, axis=0).reshape(2, 2)                   # 최적 정책이 모든 정책을 원소별로 지배
        pb = np.zeros((2, 2), int)
        for i, prev in states:
            q = [a * grid[i] - kappa * abs(a - prev) + gamma * P[i] @ Vb[:, a] for a in (0, 1)]
            other = 1 - prev
            pb[i, prev] = other if q[other] > q[prev] + 1e-12 else prev   # 동점이면 유지
        return Vb, pb

    def test_against_brute_force(self):
        rng = np.random.default_rng(0)
        cases = [(np.array([-0.002, 0.003]), np.array([[0.9, 0.1], [0.2, 0.8]]), 0.003, 0.95),
                 (np.array([-0.001, 0.001]), np.array([[0.99, 0.01], [0.01, 0.99]]), -math.log(1 - 0.0015), 0.998)]
        for _ in range(6):
            g = np.sort(rng.normal(0, 0.002, 2))
            q = rng.uniform(0.5, 0.99, 2)
            cases.append((g, np.array([[q[0], 1 - q[0]], [1 - q[1], q[1]]]), rng.uniform(0.0005, 0.005), 0.97))
        for g, P, kappa, gamma in cases:
            V, pi, it, diff = M.value_iteration(g, P, kappa, gamma, 1e-12)
            self.assertLess(diff, 1e-12)
            Vb, pb = self.brute(g, P, kappa, gamma)
            np.testing.assert_allclose(V, Vb, atol=1e-9)
            np.testing.assert_array_equal(pi, pb)

    def test_grid_rows_are_distributions(self):
        g, P = M.ar1_grid(1e-5, 0.98, 2e-4, 201, 4.0)
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-12)
        self.assertEqual(P.shape, (201, 201))
        m = g @ P[100]                                                   # 가운데 칸의 다음 기대값 ≈ a + φ·p
        self.assertAlmostEqual(m, 1e-5 + 0.98 * g[100], delta=2e-5)


class Hysteresis(unittest.TestCase):
    def test_q_out_le_q_in(self):
        hp = M.hparams({})
        for a, phi, s in [(0.0, 0.98, 2e-4), (1e-5, 0.995, 5e-5), (-2e-5, 0.9, 5e-4), (5e-8, 0.999, 1e-5),
                          (0.0, 0.3, 1e-3)]:
            b = M.solve_band(a, phi, s, hp)
            self.assertTrue(b["mono"])
            self.assertLessEqual(b["q_out"], b["q_in"])
            self.assertTrue(np.all(b["pi"][:, 1] >= b["pi"][:, 0]))      # 보유 중이면 더 오래 들고 감
            if phi > 0.9:
                self.assertLess(b["q_out"], b["q_in"])                     # 느린 신호에는 실제로 띠가 있음

    def test_band_path(self):
        p = np.array([0.0, 2.0, 0.5, np.nan, -0.5, -2.0, 0.5, 1.5])
        w = M.band_path(p, 1.0, -1.0, 0.0)
        np.testing.assert_array_equal(w, [0, 1, 1, 1, 1, 0, 0, 1])
        np.testing.assert_array_equal(M.sign_path(p), [0, 1, 1, 1, 0, 0, 1, 1])

    def test_ar1_fit(self):
        rng = np.random.default_rng(3)
        x = np.empty(20000)
        x[0] = 0.0
        for t in range(1, len(x)):
            x[t] = 1e-4 + 0.95 * x[t - 1] + 3e-4 * rng.standard_normal()
        a, phi, s, clipped = M.fit_ar1(x[:-1], x[1:])
        self.assertAlmostEqual(phi, 0.95, delta=0.01)
        self.assertAlmostEqual(s, 3e-4, delta=1e-5)
        self.assertFalse(clipped)


class Monthly(Setup):
    def test_two_months(self):
        cfg = _cfg()
        d0 = self.datas[0]
        mask = decision_mask(d0, 6)
        T1, T2 = pd.Timestamp("2017-01-01", tz="UTC"), pd.Timestamp("2017-02-01", tz="UTC")
        m1, anc, e1 = M.monthly_update(cfg, self.datas, T1, (0, 2017, 1, 0), None, None)
        self.assertIsNone(anc)
        self.assertEqual((e1["month"], e1["kind"], e1["accepted"], e1["pos0"]), ("2017-01-01", "cold", True, 0.0))
        self.assertLessEqual(e1["label_end_max"], int(T1.timestamp()))
        self.assertTrue(e1["vi_converged"])
        a, b = decision_range(d0, int(T1.timestamp()), int(T2.timestamp()))
        sel = np.arange(a, b)[mask[a:b]]
        self.assertEqual(len(sel), 31)
        w = m1.weights(d0.X[sel])
        self.assertEqual(w.shape, (31,))
        self.assertTrue(set(np.unique(w)) <= {0.0, 1.0})
        np.testing.assert_array_equal(m1.weights(d0.X[sel]), w)       # 멱등
        self.assertEqual(m1.last_pos, w[-1])
        Xn = d0.X[sel].copy()
        Xn[5] = np.nan                                                   # NaN 행은 이전 포지션 유지
        wn = m1.weights(Xn)
        self.assertEqual(wn[5], wn[4])
        m1.weights(d0.X[sel])
        m2, _, e2 = M.monthly_update(cfg, self.datas, T2, (0, 2017, 2, 0), m1, None)
        self.assertEqual((e2["month"], e2["kind"], e2["accepted"]), ("2017-02-01", "finetune", True))
        self.assertEqual(e2["pos0"], w[-1])                              # 지난달 마지막 포지션 이어받기
        self.assertEqual(m2.pos0, w[-1])
        self.assertEqual(e2["prev_oos"]["n"], 31)
        self.assertGreater(e2["n_samples"], e1["n_samples"])
        json.dumps(e1), json.dumps(e2)                                   # 로그는 JSON으로 저장됨
        st = m2.state()
        self.assertTrue(all(isinstance(v, np.ndarray) for v in st.values()))
        # 결정론적: 시드가 달라도 같음
        m1b, _, e1b = M.monthly_update(cfg, self.datas, T1, (9, 2017, 1, 0), None, None)
        np.testing.assert_array_equal(m1b.weights(d0.X[sel]), w)
        self.assertEqual(e1b["q_in"], e1["q_in"])

    def test_walk_hook(self):
        """walk.monthly_update가 cfg['algo']로 이 모듈을 불러 쓰는지"""
        from btc.research import walk
        cfg = _cfg(phases=1)
        m, _, e = walk.monthly_update(cfg, self.datas, pd.Timestamp("2017-01-01", tz="UTC"), (0, 2017, 1, 0), None, None)
        self.assertIsInstance(m, M.DPBandModel)
        self.assertEqual(e["n_phases"], 1)


class Logging(unittest.TestCase):
    def test_infinite_thresholds_keep_sign(self):
        """항상 보유(−inf,−inf)와 항상 현금(+inf,+inf)이 기록에서 구분돼야 (검토 지적)"""
        hp = M.hparams({})
        up = M.solve_band(5e-5, 0.9, 1e-5, hp)
        dn = M.solve_band(-5e-5, 0.9, 1e-5, hp)
        self.assertEqual((up["q_in"], up["q_out"]), (-math.inf, -math.inf))
        self.assertEqual((dn["q_in"], dn["q_out"]), (math.inf, math.inf))
        self.assertEqual(M.policy_regime(up["q_in"], up["q_out"]), "always_long")
        self.assertEqual(M.policy_regime(dn["q_in"], dn["q_out"]), "never_long")
        self.assertEqual(M.policy_regime(0.001, -math.inf), "enter_never_exit")
        self.assertEqual(M.policy_regime(math.inf, 0.0), "exit_never_enter")
        self.assertEqual(M.policy_regime(0.001, -0.001), "band")
        enc = json.loads(json.dumps([M.fmt_q(up["q_in"]), M.fmt_q(dn["q_out"]), M.fmt_q(1.5e-4)]))
        self.assertEqual(enc, ["-inf", "+inf", 1.5e-4])
        self.assertEqual([M.parse_q(x) for x in enc], [-math.inf, math.inf, 1.5e-4])
        cu = M.policy_chain(up["grid"], up["P"], up["pi"], up["kappa"])
        cd = M.policy_chain(dn["grid"], dn["P"], dn["pi"], dn["kappa"])
        self.assertAlmostEqual(cu["exposure"], 1.0, places=9)
        self.assertAlmostEqual(cd["exposure"], 0.0, places=9)
        b = M.solve_band(0.0, 0.98, 2e-4, hp)                           # 평균 0 대칭 신호 → 노출 ½
        cb = M.policy_chain(b["grid"], b["P"], b["pi"], b["kappa"])
        self.assertAlmostEqual(cb["exposure"], 0.5, places=6)
        self.assertGreater(cb["rt_per_year"], 0.0)


class Independent(unittest.TestCase):
    """dp_band_controls의 독립 검증 도구 (정책 반복·몬테카를로 문턱 탐색)"""

    def test_policy_iteration_matches_value_iteration(self):
        from btc.research.dp_band_controls import policy_iteration
        hp = dict(M.hparams({}), dp_grid_n=61)
        for a, phi, s in [(0.0, 0.98, 2e-4), (1e-5, 0.995, 5e-5), (5e-8, 0.9995, 1e-5)]:
            b = M.solve_band(a, phi, s, hp)
            V, pi, it = policy_iteration(b["grid"], b["P"], b["kappa"], hp["dp_gamma"])
            np.testing.assert_array_equal(pi, b["pi"])
            np.testing.assert_allclose(V, b["V"], atol=1e-8)

    def test_mc_search_finds_dp_band(self):
        from btc.research.dp_band_controls import mc_band_search
        hp = M.hparams({})
        b = M.solve_band(0.0, 0.98, 2e-4, hp)
        r = mc_band_search(0.0, 0.98, 2e-4, b["kappa"], extra=[(b["q_in"], b["q_out"]), (0.0, 0.0)],
                           n_paths=40, n_days=4000, K=11, width=1.0)
        self.assertGreaterEqual(r["extra"][0] / r["best"][2], 0.97)     # DP 띠 ≈ 최선 (같은 난수)
        self.assertGreater(r["extra"][0], r["extra"][1])                 # DP 띠 > 부호 규칙


class Reconstruct(Setup):
    def test_paths_from_log_match_walk(self):
        """저장되는 것(U, log, last_state)만으로 DP를 비트 단위로 복원하고 부호 규칙을 다시 만들 수 있어야"""
        from btc.research import walk
        from btc.research.dp_band_controls import compare_paths
        from btc.evaluate import Window
        cfg = _cfg()
        end = pd.Timestamp("2017-03-01", tz="UTC")
        res = walk.run_replication(cfg, 0, datas=self.datas, end=end)
        log = json.loads(json.dumps(res["log"]))                         # 저장·불러오기와 같게
        self.assertEqual(len(log), 2)
        for e in log:
            for k in ("fi", "feat_mu", "feat_sd", "ridge_beta", "c0", "policy_regime", "policy_exposure"):
                self.assertIn(k, e)
        d0 = self.datas[0]
        paths = M.paths_from_log(log, d0, 6, end=end)
        U = res["U"][0.0][:, 0]
        sel = np.flatnonzero(np.isfinite(U))
        self.assertEqual(len(sel), 31 + 28)
        np.testing.assert_array_equal(np.flatnonzero(np.isfinite(paths["dp"])), sel)
        np.testing.assert_array_equal(paths["dp"][sel], U[sel])
        # 첫 달: 다음 달 entry의 prev_oos와 같아야
        po = log[1]["prev_oos"]
        np.testing.assert_array_equal(paths["dp"][sel[:31]], po["dp"])
        np.testing.assert_array_equal(paths["sign"][sel[:31]], po["sign"])
        np.testing.assert_allclose(paths["p"][sel[:31]], po["p"], rtol=0, atol=0)
        np.testing.assert_array_equal(paths["sign"][sel], (paths["p"][sel] > 0).astype(float))
        # 마지막 달: 다음 entry가 없으므로 last_state에 남아 있어야
        st = res["last_state"]
        np.testing.assert_array_equal(st["last_w"], U[sel[31:]])
        np.testing.assert_array_equal(st["last_sign"], paths["sign"][sel[31:]])
        np.testing.assert_array_equal(st["last_p"], paths["p"][sel[31:]])
        # screen과 같은 Window 평가가 돌아가는지
        W = Window(self.datas, "2017-01-01", "2017-03-01", "test dp_band compare")
        out = compare_paths(W, paths)
        for k in ("dp", "sign", "bh"):
            self.assertTrue(np.isfinite(out[k]["sharpe"]))
        self.assertIn("prereg_sharpe_ge_sign_plus_0p05", out)
        # 결정론적: rep(시드)가 달라도 U가 같음
        res3 = walk.run_replication(cfg, 3, datas=self.datas, end=end)
        np.testing.assert_array_equal(res3["U"][0.0], res["U"][0.0])


if __name__ == "__main__":
    unittest.main()
