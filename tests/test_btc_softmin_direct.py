"""W1 softmin_direct(구간 강건 SoftMin 직접 정책) 단위 테스트 — 네트워크·실제 데이터 없음, 약 30초.

python -m unittest tests.test_btc_softmin_direct -v
  · 손실 전체(목적 + L2 + L2-SP)의 유한차분 기울기 검사 (float64, 상대오차 < 1e-4)
  · λ = 0 이면 direct.py 'log' 목적 × √365/σ_ref 와 같음 (값·기울기),
    대안 loss_div_k=True 이면 λ = 0 에서 direct.py 기울기와 (배율 없이) 같음
  · λ = 0.2 에서 손실 값 = 반복문으로 따로 짠 공식 (nn.forward·softmin_objective 안 씀)
  · SoftMin 극한: τ → ∞ 이면 평균, τ → 0 이면 최솟값
  · λ 규칙, 월별 상수(σ_ref·λ)가 T_k 이후 가격에 영향받지 않음(누출 없음)
  · P_BH 정확식 = 표본 평균의 극한, λ 는 반복 씨앗과 무관(같은 달이면 같은 값)
  · 학습 전체(처음부터 달·이어학습 달)가 T_k 이후 가격·지표를 모두 바꿔도 파라미터가 비트 단위로 같음
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

    def test_value_matches_independent_formula(self):
        """λ = 0.2 에서 멤버별 손실 = 반복문 공식 (로짓은 직접 짠 순전파, SoftMin 도 직접)"""
        tr, X, m = self._trainer64(M=2, G=3, K=2, L=6, lam=0.2, seed=4)
        m = m + np.repeat(np.linspace(-0.03, 0.03, tr.G), 2)[:, None]       # 묶음마다 성적이 달라 적이 작동
        loss, _, info = tr.loss_grads(X, m)
        self.assertGreater(info["q"].max(), 0.5)
        params = tr.net.copy_params()
        n, L = m.shape
        K = n // tr.G
        c = tr.cfg["cost_train"]
        for j in range(len(loss)):
            h = X.reshape(n * L, -1)
            for i in range(0, len(params), 2):
                h = h @ params[i][j] + params[i + 1][j]
                if i < len(params) - 2:
                    h = np.maximum(h, 0.0)
            w = (1.0 / (1.0 + np.exp(-h[:, 0]))).reshape(n, L)
            g = np.zeros((n, L))
            for a in range(n):
                for t in range(L):
                    g[a, t] = np.log(1.0 + w[a, t] * (np.exp(m[a, t]) - 1.0))
                    if t > 0:
                        g[a, t] += np.log(1.0 - c * abs(w[a, t] - w[a, t - 1]))
            S = lambda x: x.mean() / tr.sigma_ref * np.sqrt(365.0)
            z = [S(g[b * K:(b + 1) * K]) for b in range(tr.G)]
            sm = -tr.tau * np.log(np.mean(np.exp(-np.array(z) / tr.tau)))
            self.assertAlmostEqual(loss[j] / (-S(g) - tr.lam * sm), 1.0, places=12)

    def test_loss_div_k_lambda0_equals_direct_gradients(self):
        """대안(기본 꺼짐) loss_div_k=True, λ = 0 → direct.py 'log' 와 같은 값·기울기 (k 배가 아님)"""
        cfg = w1_cfg(wd=1e-4, lambda_sp=1e-2, noise=0.0, loss_div_k=True)
        rng = np.random.default_rng(6)
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
        dt.anchor = [p + 0.05 for p in dt.net.copy_params()]
        st = SD.SoftminTrainer(cfg, (0, 1))
        st.init_fresh()
        st.net.set_params(dt.net.copy_params())
        st.L = L
        st._seqs = lambda k: (X, m)
        st.anchor = dt.anchor
        st.sigma_ref, st.lam = 0.031, 0.0
        cd, cs = Cap(), Cap()
        vd = dt.step(cd)
        vs = st.step(cs)
        self.assertAlmostEqual(vs / vd, 1.0, places=4)
        for gd, gs in zip(cd.g, cs.g):
            np.testing.assert_allclose(gs, gd, rtol=2e-3, atol=1e-5 * float(np.abs(gd).max()))
        # 기본(명세)은 꺼짐
        self.assertFalse(SD.proposed_cfg().get("loss_div_k", False))


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

    def test_lambda_deterministic_and_p_bh_exact(self):
        """λ·P_BH·SoftMin_BH 는 반복 씨앗과 무관, P_BH 정확식 = 큰 표본 평균"""
        T_k = _ts("2015-01-01")
        Tk = int(T_k.timestamp())
        d, _ = synth_phase()
        cfg = w1_cfg(phases=1, bh_batches=64)
        trs = []
        for seed in ((0, 2015, 1, 0), (3, 2015, 1, 0)):
            tr = SD.SoftminTrainer(cfg, seed)
            tr.make_pool([d], Tk, SD.FIRST_TRAIN, SD.WARMUP)
            trs.append(tr)
        a, b = trs
        self.assertEqual((a.P_bh, a.sm_bh, a.lam), (b.P_bh, b.sm_bh, b.lam))
        # 정확식 검증 1: 구간 평균 수익을 반복문으로 직접
        j = np.linspace(0, len(a.starts) - 1, 50).astype(int)
        for i in j:
            k, t = a.starts[i]
            idx = t + a.S * np.arange(a.L)
            self.assertAlmostEqual(a._seq_mean_m()[i], np.log(d.o[idx + a.S + 1] / d.o[idx + 1]).mean(), places=12)
        # 정확식 검증 2: 큰 표본의 w=1 점수 평균과 일치 (표본 오차 범위)
        mm = a._m_seqs(200000, np.random.default_rng(0))
        P = np.sqrt(365.0) / a.sigma_ref * mm.mean(axis=1)
        self.assertLess(abs(P.mean() - a.P_bh), 4 * P.std() / np.sqrt(len(P)))

    def test_training_no_leak_after_T_k(self):
        """T_k 이후(ts ≥ T_k) 가격·지표를 모두 바꿔도 학습된 파라미터·월 기록이 비트 단위로 같음 (처음부터 달·이어학습 달)"""
        from btc.research.walk import monthly_update
        from btc.env import PhaseData
        d0, _ = synth_phase(seed=3)                     # = _series("regime", 3, n=7000)
        T1, T2 = _ts("2015-01-01"), _ts("2015-02-01")
        cfg = w1_cfg(phases=2, cold_steps=30, cold_split=20, ft_steps=10, bh_batches=16)

        def perturbed(cut):
            from test_btc_controls import _series
            from btc import features as Fe
            df, _ = _series("regime", 3, n=7000)
            late = df["ts"].to_numpy() >= cut
            f = np.exp(np.cumsum(np.where(late, np.random.default_rng(9).normal(0, 0.2, len(df)), 0.0)))
            for c in ("open", "high", "low", "close"):
                df[c] = df[c] * f
            X, sig = Fe.compute(df)
            X = np.where(late[:, None], np.random.default_rng(1).standard_normal(X.shape), X)
            return PhaseData(df, X, sig)

        def run(d, months):
            ens, anc, log = None, None, []
            for T in months:
                ens, anc, e = monthly_update(cfg, [d, d], T, (0, T.year, T.month, 0), ens, anc)
                log.append(e)
            return ens, log

        for months in ((T1,), (T1, T2)):
            cut = int(months[-1].timestamp())
            e0, l0 = run(d0, months)
            e1, l1 = run(perturbed(cut), months)
            self.assertTrue(all(np.array_equal(p, q) for p, q in zip(e0.net.params, e1.net.params)), months)
            self.assertEqual(l0, l1)
        # 민감도: T_k 한 달 전부터 바꾸면 달라져야 함 (검사가 실제로 무언가를 봄)
        e0, _ = run(d0, (T1,))
        e2, _ = run(perturbed(int(_ts("2014-12-01").timestamp())), (T1,))
        self.assertFalse(all(np.array_equal(p, q) for p, q in zip(e0.net.params, e2.net.params)))

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
