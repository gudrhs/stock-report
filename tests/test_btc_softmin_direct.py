"""W1 softmin_direct(구간 강건 SoftMin 직접 정책) 단위 테스트 — 네트워크·실제 데이터 없음, 약 30초.

python -m unittest tests.test_btc_softmin_direct -v
  · 손실 전체(목적 + L2 + L2-SP)의 유한차분 기울기 검사 (float64, 상대오차 < 1e-4)
  · λ = 0 이면 direct.py 'log' 목적 × √365/σ_ref 와 같음 (값·기울기)
  · SoftMin 극한: τ → ∞ 이면 평균, τ → 0 이면 최솟값
  · λ 규칙, 월별 상수(σ_ref·λ)가 T_k 이후 가격에 영향받지 않음(누출 없음)
  · walk.monthly_update 플러그인 경로로 2달(처음부터 + 이어서) 실행, 비중 ∈ [0, 1]
"""
import os
import sys
import unittest

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from btc.nn import StackedMLP
from btc.research.variants import VARIANTS, TREND8
from btc.research.direct import DirectTrainer
from btc.research.algos import softmin_direct as SD


def w1_cfg(**kw):
    cfg = SD.proposed_cfg()
    cfg.update(kw)
    return cfg


_DATA = {}


def synth_phase(seed=1, n=7000, perturb_after=None):
    """합성 국면 전환 4시간봉(2012-01부터) → PhaseData. perturb_after가 있으면 그 시각 이후 가격만 흔듦"""
    key = (seed, n, perturb_after)
    if key in _DATA:
        return _DATA[key]
    from test_btc_controls import _series
    from btc import features as Fe
    from btc.env import PhaseData
    df, reg = _series("regime", seed, n=n)
    if perturb_after is not None:
        df = df.copy()
        late = df["ts"].to_numpy() >= perturb_after
        f = np.exp(np.cumsum(np.where(late, np.random.default_rng(9).normal(0, 0.05, len(df)), 0.0)))
        for c in ("open", "high", "low", "close"):
            df[c] = df[c] * f
    X, sig = Fe.compute(df)
    _DATA[key] = (PhaseData(df, X, sig), reg)
    return _DATA[key]


def _ts(s):
    return pd.Timestamp(s, tz="UTC")


class Objective(unittest.TestCase):
    def test_softmin_limits(self):
        z = np.array([[0.3, -1.2, 0.8, 2.0], [1.0, 1.0, 1.0, 1.0]])
        np.testing.assert_allclose(SD.softmin(z, 1e6, axis=1), z.mean(axis=1), atol=1e-5)
        np.testing.assert_allclose(SD.softmin(z, 1e-4, axis=1), [-1.2, 1.0], atol=1e-3)
        # 단조: τ가 커질수록 평균 쪽으로 (최솟값 ≤ SoftMin ≤ 평균)
        s = [SD.softmin(z[0], t) for t in (0.01, 0.1, 1.0, 10.0)]
        self.assertTrue(all(a <= b + 1e-12 for a, b in zip(s, s[1:])))
        self.assertTrue(z[0].min() - 1e-12 <= s[0] and s[-1] <= z[0].mean() + 1e-12)

    def test_lambda_rule(self):
        self.assertAlmostEqual(SD.lambda_rule(1.0, -10.0, 0.2), 0.05)
        self.assertAlmostEqual(SD.lambda_rule(1.0, -1.0, 0.2), 0.2)       # 상한
        self.assertAlmostEqual(SD.lambda_rule(-0.1, -1.0, 0.2), 0.2)      # P_BH ≤ 0
        self.assertAlmostEqual(SD.lambda_rule(0.5, 0.1, 0.2), 0.2)        # SoftMin_BH ≥ 0

    def _trainer64(self, M=2, G=2, K=2, L=7, lam=0.2, seed=0):
        cfg = w1_cfg(members=M, hidden=(6, 5), groups=G, seq_batch=G * K, seq_len=L, wd=3e-2, lambda_sp=1e-1)
        tr = SD.SoftminTrainer(cfg, (seed,))
        rng = np.random.default_rng(seed)
        tr.net = StackedMLP(M, [len(TREND8), 6, 5, 1], rng, last_scale=1.0, dtype=np.float64)
        for p in tr.net.params[1::2]:                   # 편향 0이면 죽은 행의 사전활성이 정확히 0(ReLU 꺾임)이 됨
            p += 0.3 * rng.standard_normal(p.shape)
        tr.sigma_ref, tr.lam, tr.tau = 0.03, lam, 0.2
        tr.anchor = [p + 0.1 * rng.standard_normal(p.shape) for p in tr.net.params]
        X = rng.standard_normal((G * K, L, len(TREND8)))
        m = 0.04 * rng.standard_normal((G * K, L))
        return tr, X, m

    def _full_loss(self, tr, X, m):
        c = tr.cfg
        loss, _, _ = tr.loss_grads(X, m)
        tot = float(loss.sum())
        for p, a in zip(tr.net.params, tr.anchor):
            if p.ndim == 3 and p.shape[1] > 1:
                tot += c["wd"] * float((p ** 2).sum())
            tot += c["lambda_sp"] * float(((p - a) ** 2).sum())
        return tot

    def test_finite_difference_gradient(self):
        tr, X, m = self._trainer64()
        _, grads, info = tr.loss_grads(X, m)
        self.assertGreater(info["q"].max(), 0.5 / tr.G)                     # 적 가중치가 실제로 작동
        rng = np.random.default_rng(1)
        eps = 1e-6
        checked = 0
        for i, p in enumerate(tr.net.params):
            for _ in range(4):
                idx = tuple(rng.integers(0, s) for s in p.shape)
                old = p[idx]
                p[idx] = old + eps
                fp = self._full_loss(tr, X, m)
                p[idx] = old - eps
                fm = self._full_loss(tr, X, m)
                p[idx] = old
                num = (fp - fm) / (2 * eps)
                ana = float(grads[i][idx])
                rel = abs(num - ana) / max(abs(num), abs(ana), 1e-6)
                self.assertLess(rel, 1e-4, f"param {i} {idx}: 수치 {num} 해석 {ana}")
                checked += 1
        self.assertEqual(checked, 4 * len(tr.net.params))

    def test_gradient_only_softmin_term(self):
        """λ만 바꾼 차이 = −λ·∂SoftMin — 비용 항이 t와 t+1 양쪽에 들어가는지 W 수준에서 직접 확인"""
        rng = np.random.default_rng(3)
        W = rng.uniform(0.05, 0.95, (2, 6, 9))
        m = 0.05 * rng.standard_normal((6, 9))
        args = (0.003, 0.025, 0.2, 0.2, 3)
        _, gW, _ = SD.softmin_objective(W, m, *args)
        eps = 1e-6
        for idx in [(0, 0, 0), (0, 2, 4), (1, 5, 8), (1, 3, 1)]:
            Wp, Wm = W.copy(), W.copy()
            Wp[idx] += eps
            Wm[idx] -= eps
            num = (SD.softmin_objective(Wp, m, *args)[0][idx[0]] - SD.softmin_objective(Wm, m, *args)[0][idx[0]]) / (2 * eps)
            self.assertLess(abs(num - gW[idx]) / max(abs(num), 1e-8), 1e-5)

    def test_lambda0_equals_direct_log_objective(self):
        cfg = w1_cfg(wd=0.0, lambda_sp=0.0, noise=0.0)
        rng = np.random.default_rng(5)
        n, L = 32, 63
        X = rng.standard_normal((n, L, len(TREND8))).astype(np.float32)
        m = (0.035 * rng.standard_normal((n, L))).astype(np.float32)

        class Cap:
            def step(self, g):
                self.g = g

        dt = DirectTrainer(cfg, (0, 1))
        dt.init_fresh()
        dt.L = L
        dt._seqs = lambda k: (X, m)
        st = SD.SoftminTrainer(cfg, (0, 1))
        st.init_fresh()
        st.net.set_params(dt.net.copy_params())
        st.L = L
        st._seqs = lambda k: (X, m)
        st.sigma_ref, st.lam = 0.031, 0.0
        cd, cs = Cap(), Cap()
        vd = dt.step(cd)
        vs = st.step(cs)
        k = np.sqrt(365.0) / st.sigma_ref
        self.assertAlmostEqual(vs / (k * vd), 1.0, places=4)
        for gd, gs in zip(cd.g, cs.g):
            np.testing.assert_allclose(gs, k * gd, rtol=2e-3, atol=1e-5 * float(np.abs(k * gd).max()))


class Config(unittest.TestCase):
    def test_proposed_cfg_is_r11_plus_w1_keys(self):
        """제안 설정은 R11과 이름·algo·output·seq_len·W1 키만 다름 (나머지 학습 설정 동일)"""
        w1, r11 = SD.proposed_cfg(), VARIANTS["R11_direct_loggrowth"]
        extra = {"name", "algo", "output", "seq_len", "groups", "softmin_tau", "lambda_max"}
        self.assertEqual({k for k in set(w1) | set(r11) if w1.get(k) != r11.get(k)}, extra)
        self.assertEqual((w1["algo"], w1["output"], w1["seq_len"], w1["groups"]), ("softmin_direct", "weights", 63, 8))
        self.assertEqual(w1["seq_batch"] % w1["groups"], 0)


class Monthly(unittest.TestCase):
    def test_pool_constants_no_leak(self):
        """σ_ref·P_BH·SoftMin_BH·λ 는 T_k 이후 가격을 바꿔도 같아야 함"""
        T_k = _ts("2015-01-01")
        Tk = int(T_k.timestamp())
        cfg = w1_cfg(phases=1, bh_batches=8)
        vals = []
        for pert in (None, Tk):
            d, _ = synth_phase(perturb_after=pert)
            tr = SD.SoftminTrainer(cfg, (0, 2015, 1, 0))
            n = tr.make_pool([d], Tk, SD.FIRST_TRAIN, SD.WARMUP)
            self.assertGreater(n, 500)
            span = tr.S * tr.L + 1
            last = d.ts[tr.starts[:, 1] + span] + 4 * 3600
            self.assertTrue(np.all(last <= Tk))
            vals.append((tr.sigma_ref, tr.P_bh, tr.sm_bh, tr.lam))
        np.testing.assert_allclose(vals[0], vals[1], rtol=0, atol=0)
        self.assertTrue(0 < vals[0][0] < 0.2 and 0 < vals[0][3] <= 0.2)

    def test_walk_plugin_two_months(self):
        from btc.research.walk import monthly_update
        d, _ = synth_phase()
        cfg = w1_cfg(phases=1, cold_steps=30, cold_split=20, ft_steps=10, bh_batches=4)
        ens, anchor, log = None, None, []
        for T_k in (_ts("2015-01-01"), _ts("2015-02-01")):
            ens, anchor, e = monthly_update(cfg, [d], T_k, (0, T_k.year, T_k.month, 0), ens, anchor)
            log.append(e)
        self.assertEqual([e["kind"] for e in log], ["cold", "finetune"])
        for e in log:
            self.assertTrue(e["accepted"])
            for k in ("lam", "P_BH", "SoftMin_BH", "sigma_ref", "adv_max", "adv_top3", "z_mean", "loss"):
                self.assertTrue(np.all(np.isfinite(e[k])), k)
            self.assertTrue(1.0 / 8 <= e["adv_max"] <= 1.0)
        self.assertEqual(log[0]["month"], "2015-01-01")
        sel = np.arange(d.T - 400, d.T - 10)
        w = ens.weights(d.X[sel])
        self.assertEqual(w.shape, (len(sel),))
        self.assertTrue(np.all(np.isfinite(w)) and w.min() >= 0.0 and w.max() <= 1.0)
        st = ens.state()
        self.assertTrue(all(isinstance(a, np.ndarray) for a in st.values()))
        # 이어학습이 1월 모델(앵커)에서 출발했는지: 앵커는 1월 모델 파라미터 그대로
        self.assertEqual(len(anchor), len(ens.net.params))


if __name__ == "__main__":
    unittest.main()
