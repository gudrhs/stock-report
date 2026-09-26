"""Q1 qrdqn(분포형 QR-DQN + CVaR 0.5 판단) 단위 테스트 — 합성 데이터만, 1스레드 약 20초.

python -m unittest tests.test_btc_qrdqn -v
  · 분위수 후버 손실의 유한차분 기울기 (단독, 그리고 보조·L2·L2-SP 를 더한 loss_grads 전체, float64)
  · 분위수 후버 손실 값 = 반복문으로 따로 짠 공식
  · 분포형 목표 = 반복문으로 따로 짠 공식 (비용 분해·행동 보강, 다음 행동은 온라인 '평균'으로, 분위수는 목표망)
  · CVaR 규칙: n=11, α=0.5 → 정렬 후 (하위 5개 + ½·6번째)/5.5, 분위수가 교차해도 정렬, CVaR ≤ 평균
  · 기준선 효과 (검토 지적): qr_base "mid"(사전 등록 식)는 CVaR 차이 = 평균 차이, "cash" 는 음의 왜도에서 CVaR 차이 < 평균 차이
  · 2상태 MDP (알려진 수익 분포, γ=0 과 γ=0.5 부트스트랩 + 전환 비용)에서 학습한 분위수가 참값에 가까움
  · proposed_cfg = R6 + Q1 키 (qr_base "cash" — 2026-09-25 결과 전 해석 정정), 같은 시드면 첫 두 층 초기값이 R6(KTrainer)와 같음
  · walk.monthly_update 플러그인 경로로 2달(처음부터 + 이어서), T_k 이후 가격·지표를 바꿔도 파라미터가 같음(누출 없음)
"""
import math
import os
import sys
import unittest

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from btc.env import KAPPA
from btc.nn import StackedMLP, Adam
from btc.research.rl import KTrainer, reward
from btc.research.variants import VARIANTS, TREND8
from btc.research.algos import qrdqn as Q


def q1_cfg(**kw):
    cfg = Q.proposed_cfg()
    cfg.update(kw)
    return cfg


def _ts(s):
    return pd.Timestamp(s, tz="UTC")


def _small_trainer(M=2, K=2, N=11, seed=0, base="cash"):
    """float64 작은 망 (입력 4, 은닉 5·4), ReLU 꺾임을 피하려고 편향 무작위"""
    cfg = q1_cfg(members=M, hidden=(5, 4), batch=6, acts=tuple(np.linspace(0, 1, K)), qr_base=base)
    tr = Q.QRTrainer(cfg, (1, 2, 3))
    rng = np.random.default_rng(seed)
    tr.net = StackedMLP(M, [4, 5, 4, K * N + 2], rng, last_scale=0.5, dtype=np.float64)
    for i in range(1, len(tr.net.params), 2):
        tr.net.params[i] = rng.normal(0, 0.3, tr.net.params[i].shape)
    tr.tgt = StackedMLP(M, [4, 5, 4, K * N + 2], rng, last_scale=0.5, dtype=np.float64)
    tr.tau_q = Q.taus(N)
    return tr, rng


class QuantileLoss(unittest.TestCase):
    def test_value_matches_loop(self):
        rng = np.random.default_rng(0)
        N = 11
        th = rng.normal(0, 2, (3, N))
        y = rng.normal(0, 2, (3, N))
        tau = Q.taus(N)
        loss, _ = Q.quantile_huber(th, y, tau)
        for r in range(3):
            s = 0.0
            for i in range(N):
                for j in range(N):
                    u = y[r, j] - th[r, i]
                    H = 0.5 * u * u if abs(u) <= 1 else abs(u) - 0.5
                    s += abs(tau[i] - (1.0 if u < 0 else 0.0)) * H
            self.assertAlmostEqual(loss[r], s / (N * N), places=12)

    def test_finite_difference_quantile_huber(self):
        rng = np.random.default_rng(1)
        th = rng.normal(0, 1.5, (4, 11))
        y = rng.normal(0, 1.5, (4, 11))
        tau = Q.taus(11)
        _, g = Q.quantile_huber(th, y, tau)
        for r in range(4):
            for i in range(11):
                o = th[r, i]
                th[r, i] = o + 1e-6
                a = Q.quantile_huber(th, y, tau)[0][r]
                th[r, i] = o - 1e-6
                b = Q.quantile_huber(th, y, tau)[0][r]
                th[r, i] = o
                fd = (a - b) / 2e-6
                self.assertLess(abs(fd - g[r, i]), 1e-7 * max(1.0, abs(fd)), (r, i))

    def test_finite_difference_full_loss(self):
        """loss_grads 전체(분위수 손실 + 보조 + L2 + L2-SP)의 파라미터 기울기"""
        M, K, N, B = 2, 2, 11, 6
        tr, rng = _small_trainer(M, K, N)
        anchor = [p + rng.normal(0, 0.1, p.shape) for p in tr.net.params]
        X0 = rng.normal(size=(M, B, 4))
        y = rng.normal(size=(M, B, K, N)) * 2
        z6, z42 = rng.normal(size=(M, B)), rng.normal(size=(M, B))
        ok = (rng.random((M, B)) > 0.3).astype(float)
        f = lambda: tr.loss_grads(X0, y, z6, z42, ok, anchor, 0.01)[0]
        _, g, _ = tr.loss_grads(X0, y, z6, z42, ok, anchor, 0.01)
        for pi, p in enumerate(tr.net.params):
            it = np.nditer(p, flags=["multi_index"])
            for _ in it:
                i = it.multi_index
                o = p[i]
                p[i] = o + 1e-6
                a = f()
                p[i] = o - 1e-6
                b = f()
                p[i] = o
                fd = (a - b) / 2e-6
                self.assertLess(abs(fd - g[pi][i]), 1e-6 * max(1.0, abs(fd)), (pi, i))

    def test_n1_tau_half_is_half_dqn_huber(self):
        """N = 1 (τ = ½) 이면 분위수 후버 = ½ × KTrainer 의 후버 (정규화 일관성 확인)"""
        d = np.array([[-3.0], [-0.4], [0.2], [2.5]])
        loss, g = Q.quantile_huber(np.zeros_like(d), d, Q.taus(1))
        hub = np.where(np.abs(d[:, 0]) <= 1, 0.5 * d[:, 0] ** 2, np.abs(d[:, 0]) - 0.5)
        np.testing.assert_allclose(loss, 0.5 * hub)
        np.testing.assert_allclose(g[:, 0], -0.5 * np.clip(d[:, 0], -1, 1))


class Targets(unittest.TestCase):
    def test_targets_match_loop(self):
        for kind in ("cash", "mid"):
            with self.subTest(qr_base=kind):
                self._targets_match_loop(kind)

    def _targets_match_loop(self, kind):
        M, K, N, B = 2, 3, 11, 5
        tr, rng = _small_trainer(M, K, N, seed=2, base=kind)
        tr.cfg["gamma"] = 0.967
        X1 = rng.normal(size=(M, B, 4))
        m = rng.normal(0, 0.03, (M, B)).astype(np.float32)
        cost = np.exp(rng.uniform(math.log(5e-4), math.log(0.02), (M, B))).astype(np.float32)
        y = tr.targets(X1, m, cost)
        self.assertEqual(y.shape, (M, B, K, N))
        Zon = tr.net.forward(X1, cache=False)[..., :K * N].reshape(M, B, K, N)
        Ztg = tr.tgt.forward(X1, cache=False)[..., :K * N].reshape(M, B, K, N)
        A = tr.acts.astype(np.float64)
        mid = A.mean() if kind == "mid" else 0.0                     # "cash": 기준선 = 현금 수익 0
        for mm in range(M):
            for b in range(B):
                lnc = KAPPA * math.log(1 - float(cost[mm, b]))
                for a in range(K):
                    base = KAPPA * (A[a] * m[mm, b] - mid * m[mm, b])
                    sc = [lnc * abs(A[a2] - A[a]) + Zon[mm, b, a2].mean() for a2 in range(K)]
                    s = int(np.argmax(sc))
                    ref = base + 0.967 * (lnc * abs(A[s] - A[a]) + Ztg[mm, b, s])
                    np.testing.assert_allclose(y[mm, b, a], ref, rtol=1e-5, atol=1e-5)

    def test_gamma0_broadcasts_immediate_reward(self):
        for kind, b0 in (("cash", 0.0), ("mid", 0.5)):
            with self.subTest(qr_base=kind):
                tr, rng = _small_trainer(2, 2, 11, seed=3, base=kind)
                tr.cfg["gamma"] = 0.0
                m = rng.normal(0, 0.03, (2, 4)).astype(np.float32)
                y = tr.targets(rng.normal(size=(2, 4, 4)), m, np.full((2, 4), 0.003, np.float32))
                base = KAPPA * (reward("lin", tr.acts[None, None, :], m[..., None]) - b0 * m[..., None])
                np.testing.assert_allclose(y, np.broadcast_to(base[..., None], y.shape), rtol=1e-6, atol=1e-6)


class CVaR(unittest.TestCase):
    def test_weights_exact(self):
        w = Q.cvar_weights(11, 0.5)
        np.testing.assert_allclose(w, np.array([1, 1, 1, 1, 1, 0.5, 0, 0, 0, 0, 0]) / 5.5)
        np.testing.assert_allclose(Q.cvar_weights(11, 1.0), np.full(11, 1 / 11))

    def _ens(self, params_scale=1.0, seed=0):
        from btc.features import NAMES
        rng = np.random.default_rng(seed)
        net = StackedMLP(3, [9, 8, 2 * 11 + 2], rng, last_scale=params_scale)
        return Q.QREnsemble(net, TREND8, (0.0, 1.0)), rng, len(NAMES)

    def test_values_rule_and_le_mean(self):
        ens, rng, nf = self._ens(params_scale=3.0)
        X = rng.normal(size=(50, nf))
        U, umax = ens.values(X, 0.003)
        Z = ens.quantiles(X, 0.003)                                  # (M,B,K,N)
        self.assertEqual(U.shape, (50, 2))
        ref = np.empty((3, 50, 2))
        for mm in range(3):
            for b in range(50):
                for a in range(2):
                    s = np.sort(Z[mm, b, a])                         # 교차해도 정렬 후 하위
                    ref[mm, b, a] = (s[:5].sum() + 0.5 * s[5]) / 5.5
        np.testing.assert_allclose(U, ref.mean(axis=0), rtol=1e-10)
        np.testing.assert_allclose(umax, np.abs(ref).max(axis=(0, 2)), rtol=1e-10)
        self.assertTrue(np.all(U <= ens.mean_values(X, 0.003) + 1e-12))
        self.assertTrue(np.all(ref <= Z.mean(axis=-1) + 1e-12))       # 멤버마다도 CVaR ≤ 평균
        st = ens.state()
        self.assertEqual(int(st["n_quant"]), 11)

    def test_point_mass_cvar_equals_value(self):
        """분위수가 모두 같으면 CVaR = 평균 = 그 값"""
        z = np.full(11, 1.7)
        self.assertAlmostEqual(float(np.sort(z) @ Q.cvar_weights(11, 0.5)), 1.7)


SC = 20.0


class TwoStateMDP(unittest.TestCase):
    """
    상태 0: 수익 m = SC·(0.02·U² − 0.004), U ~ 균등 → 다음 상태 1.   상태 1: m = +0.01 고정, 다음 상태 1 (흡수).
    SC = 20 으로 상태 0 분포 폭을 κ_huber(=1) 의 약 20배로 둠 — 분위수 후버 손실의 고정점은 |u| ≤ κ_huber 안에서는
    기대분위수(expectile) 쪽으로 눌리므로(폭 1 이면 실제로 30% 가량 좁게 학습됨), 폭 ≫ κ_huber 여야 참 분위수에 가까움.
    (실제 BTC 에서 하루 κ·½·m 의 표준편차 ≈ 1.7, γ 0.967 로 쌓인 수익 분포는 그보다 훨씬 넓음)
    acts (0, 1), 기준선 b (qr_base "mid" → ½, "cash" → 0), 비용 c = 0.003 고정 → pen = κ·ln(1 − c) ≈ −0.3005
      Z(1, 1) = κ(1 − b)·0.01/(1 − γ)             (계속 보유)
      Z(1, 0) = −κb·0.01 + γ·max(Z(1,0)', pen + Z(1,1))   (γ = 0.5 면 다음 날 보유로 전환이 나음)
      Z(0, a) = κ·(a − b)·m + γ·max_a' [pen·|a' − a| + mean Z(1, a')]   (상태 1 은 점질량이라 투영 오차 없음)
      "cash" 에서 Z(0, 0) 은 점질량 (현금은 확정 수익 0)
    """

    def _true(self, gamma, b0=0.5):
        pen = KAPPA * math.log(1 - 0.003)
        r1, r0 = KAPPA * (1 - b0) * 0.01, KAPPA * (-b0) * 0.01
        z11 = r1 / (1 - gamma) if gamma > 0 else r1
        if gamma > 0:
            stay0 = r0 / (1 - gamma)
            z10 = max(stay0, r0 + gamma * (pen + z11))
            V = {1: max(z11, pen + z10), 0: max(z10, pen + z11)}
        else:
            z10, V = r0, {0: 0.0, 1: 0.0}
        tau = Q.taus(11)
        qm = SC * (0.02 * tau ** 2 - 0.004)                           # m 의 분위수 (U² 는 U 의 증가함수)
        Z0 = {1: KAPPA * (1 - b0) * qm + gamma * V[1],
              0: KAPPA * (-b0) * qm[::-1] + gamma * V[0]}            # a = 0 은 −bκm → 뒤집힌 분위수 (b = 0 이면 점질량)
        return Z0, {1: z11, 0: z10}

    def _train(self, gamma, steps=2500, base="mid"):
        cfg = q1_cfg(members=3, batch=128, gamma=gamma, noise=0.0, tau=0.02, qr_base=base)
        tr = Q.QRTrainer(cfg, (7, 1, 1, 0))
        tr.init_fresh()
        rng = np.random.default_rng(5)
        M, B, nf = 3, 128, len(TREND8)
        ci = np.float32((math.log(0.003) - math.log(0.00316)) / 1.846)

        def enc(s):
            X = np.zeros(s.shape + (nf + 1,), np.float32)
            X[..., 0] = np.where(s == 0, 1.0, -1.0)
            X[..., nf] = ci
            return X

        def batch():
            s = (rng.random((M, B)) < 0.5).astype(int)
            m = np.where(s == 0, SC * (0.02 * rng.random((M, B)) ** 2 - 0.004), 0.01).astype(np.float32)
            z = np.zeros((M, B), np.float32)
            return enc(s), enc(np.ones_like(s)), m, np.full((M, B), 0.003, np.float32), z, z, z

        tr._batch = batch
        opt = Adam(tr.net.params, lr=3e-3, clip=5.0)
        for i in range(steps):
            if i == int(steps * 0.7):
                opt.lr = 5e-4
            tr.step(opt)
        out = tr.net.forward(enc(np.array([0, 1])), cache=False)[..., :22].reshape(3, 2, 2, 11).astype(np.float64)
        return out.mean(axis=0)                                       # (상태, 행동, 분위수) 멤버 평균

    def _check(self, gamma, base="mid"):
        Z = self._train(gamma, base=base)
        Z0, Z1 = self._true(gamma, 0.5 if base == "mid" else 0.0)
        ref_spread = max(Z0[a][-1] - Z0[a][0] for a in (0, 1))       # 점질량(현금, "cash")은 보유 쪽 폭으로 허용오차
        for a in (0, 1):
            err = np.abs(Z[0, a] - Z0[a])
            spread = max(Z0[a][-1] - Z0[a][0], ref_spread)             # "mid" ≈ 0.5·κ·SC·0.02 ≈ 20, "cash" 보유 ≈ 40
            self.assertLess(err.mean(), 0.04 * spread, (gamma, a, Z[0, a].round(3), Z0[a].round(3)))
            self.assertLess(err.max(), 0.08 * spread, (gamma, a))
            np.testing.assert_allclose(Z[1, a], Z1[a], atol=0.1, err_msg=f"state1 a={a} gamma={gamma}")
            # 학습한 분위수는 (거의) 증가 순서
            self.assertGreater(np.diff(Z[0, a]).min(), -0.02 * spread)
        # CVaR_0.5 규칙을 참 분위수와 학습 분위수에 적용한 값이 가까움
        w = Q.cvar_weights(11, 0.5)
        for a in (0, 1):
            self.assertLess(abs(np.sort(Z[0, a]) @ w - Z0[a] @ w), 0.03 * ref_spread)

    def test_gamma0(self):
        self._check(0.0)

    def test_gamma05_bootstrap_with_cost(self):
        self._check(0.5)

    def test_gamma0_cash(self):
        self._check(0.0, "cash")

    def test_gamma05_bootstrap_with_cost_cash(self):
        self._check(0.5, "cash")


class BaselineEffect(unittest.TestCase):
    """
    검토 지적(2026-09-25): 기준선 R(mid, m) 을 빼면 α = 0.5 에서 CVaR 차이(보유 − 현금) = 평균 차이 → 비관적 진입이 사라짐.
    qr_base "mid"(사전 등록 식, 기본값)는 그 한계를 그대로 보이고, "cash"(절대 수익)는 음의 왜도에서 CVaR 차이 < 평균 차이.
    정확 검사: m = −0.20 (확률 0.1) 아니면 +0.03 (평균 +0.007, 음의 왜도).
    학습 검사: m = −0.05 (확률 0.3) 아니면 +0.03 — 손실이 분위수 3개(τ 0.045·0.136·0.227)에 걸려 짧은 학습으로도 꼬리가 잡힘.
      11분위수 기준 참값 (보유, κ = 100): 절대 수익 −5 ×3, +3 ×8 → 분위수 평균 0.818, CVaR_0.5 = (−15 + 2·3 + ½·3)/5.5 = −1.36,
      현금 0. 'mid' 면 보유 −2.5 ×3·+1.5 ×8, 현금 −1.5 ×8·+2.5 ×3 → CVaR 차이 = 분위수 평균 차이 = 0.818.
    """

    @staticmethod
    def _m(rng, shape):
        return np.where(rng.random(shape) < 0.1, -0.20, 0.03).astype(np.float32)

    def test_targets_cvar_vs_mean_exact(self):
        """γ = 0 목표 표본 자체의 CVaR_0.5 (하위 절반 평균, 표본 수 짝수)"""
        rng = np.random.default_rng(11)
        m = self._m(rng, (1, 20000))
        cost = np.full(m.shape, 0.003, np.float32)
        diffs = {}
        for kind in ("mid", "cash"):
            tr, _ = _small_trainer(1, 2, 11, seed=4)
            tr.cfg["gamma"] = 0.0
            tr.base_kind = kind
            y = tr.targets(np.zeros((1, m.shape[1], 4)), m, cost)[0, :, :, 0].astype(np.float64)   # (B, K)
            lo = lambda v: np.sort(v)[: len(v) // 2].mean()
            dc = lo(y[:, 1]) - lo(y[:, 0])
            dm = y[:, 1].mean() - y[:, 0].mean()
            diffs[kind] = (dc, dm, lo(y[:, 0]))
        dc, dm, _ = diffs["mid"]
        self.assertAlmostEqual(dc, dm, places=3)                       # 한계: 평균과 똑같음
        dc, dm2, cash = diffs["cash"]
        self.assertAlmostEqual(dm2, dm, places=3)                      # 평균 차이는 기준선과 무관
        self.assertAlmostEqual(cash, 0.0, places=6)                    # 현금은 무위험
        self.assertLess(dc, dm - 1.0)                                  # 비관적: CVaR 차이 < 평균 차이
        self.assertLess(dc, 0.0)                                       # 이 분포면 CVaR 로는 보유하지 않음
        self.assertGreater(dm, 0.0)                                    # 평균으로는 보유

    def test_trained_gamma0(self):
        """실제 QRTrainer 학습(γ = 0, 한 상태)에서 CVaR 판단 가치의 차이"""
        res = {}
        for kind in ("mid", "cash"):
            cfg = q1_cfg(members=3, batch=128, gamma=0.0, noise=0.0, tau=0.02, qr_base=kind)
            tr = Q.QRTrainer(cfg, (7, 1, 1, 0))
            tr.init_fresh()
            rng = np.random.default_rng(5)
            M, B, nf = 3, 128, len(TREND8)
            X = np.zeros((M, B, nf + 1), np.float32)

            def batch():
                z = np.zeros((M, B), np.float32)
                m = np.where(rng.random((M, B)) < 0.3, -0.05, 0.03).astype(np.float32)
                return X, X, m, np.full((M, B), 0.003, np.float32), z, z, z

            tr._batch = batch
            opt = Adam(tr.net.params, lr=3e-3, clip=5.0)
            for i in range(1500):
                if i == 1000:
                    opt.lr = 5e-4
                tr.step(opt)
            Z = tr.net.forward(X[:, :1], cache=False)[:, 0, :22].reshape(3, 2, 11).astype(np.float64)
            cv = (np.sort(Z, axis=-1) @ Q.cvar_weights(11, 0.5)).mean(axis=0)     # (K,)
            mu = Z.mean(axis=-1).mean(axis=0)
            res[kind] = (cv[1] - cv[0], mu[1] - mu[0], cv)
            self.assertEqual(tr.ensemble().base_kind, kind)
        # (폭 8 이 κ_huber = 1 의 몇 배밖에 안 돼 분위수가 기대분위수 쪽으로 눌림 → 참값보다 좁음, 부호·대소만 봄.
        #  실측: mid 평균 차이 0.558 = CVaR 차이 0.558,  cash 평균 차이 0.96, 보유 CVaR −0.76, 현금 CVaR 0)
        dc, dm, _ = res["mid"]
        self.assertTrue(0.2 < dm < 1.2, dm)
        self.assertLess(abs(dc - dm), 0.1)                             # 한계: CVaR 차이 ≈ 평균 차이 (> 0 → 보유)
        self.assertGreater(dc, 0.0)
        dc, dm, cv = res["cash"]
        self.assertTrue(0.2 < dm < 1.2, dm)                            # 평균으로는 보유
        self.assertLess(abs(cv[0]), 0.1)                               # 현금 CVaR ≈ 0
        self.assertLess(cv[1], -0.3)                                   # 보유 CVaR 음수 (참값 −1.36)
        self.assertLess(dc, dm - 1.0)                                  # 보유 CVaR 가 크게 낮음
        self.assertLess(dc, 0.0)                                       # CVaR 판단이면 현금

    def test_bad_base_kind(self):
        with self.assertRaises(ValueError):
            Q.QRTrainer(q1_cfg(qr_base="avg"), (1, 2, 3))


class Walk(unittest.TestCase):
    def test_proposed_cfg_is_r6_plus_q1_keys(self):
        cfg = Q.proposed_cfg()
        r6 = VARIANTS["R6_daily_trend8_uniform"]
        extra = {"algo": "qrdqn", "n_quantiles": 11, "cvar_alpha": 0.5, "huber_k": 1.0, "qr_base": "cash"}
        self.assertEqual({k: v for k, v in cfg.items() if k not in extra and k != "name"},
                         {k: v for k, v in r6.items() if k != "name"})
        for k, v in extra.items():
            self.assertEqual(cfg[k], v)
        self.assertEqual(cfg["name"], "Q1_qrdqn_cvar")

    def test_same_seed_same_init_as_r6(self):
        """시드 사용법이 R6와 같음: 첫 두 층(모양이 같은 부분)의 초기값이 비트 단위로 같음"""
        seed = (0, 2017, 1, 0)
        a = KTrainer(VARIANTS["R6_daily_trend8_uniform"], seed)
        a.init_fresh()
        b = Q.QRTrainer(Q.proposed_cfg(), seed)
        b.init_fresh()
        for i in range(4):
            self.assertTrue(np.array_equal(a.net.params[i], b.net.params[i]), i)
        self.assertEqual(b.net.params[4].shape[-1], 2 * 11 + 2)

    @staticmethod
    def _phase(seed=3, n=7000, cut=None):
        from test_btc_controls import _series
        from btc import features as Fe
        from btc.env import PhaseData
        df, _ = _series("regime", seed, n=n)
        late = None
        if cut is not None:
            late = df["ts"].to_numpy() >= cut
            f = np.exp(np.cumsum(np.where(late, np.random.default_rng(9).normal(0, 0.2, len(df)), 0.0)))
            for c in ("open", "high", "low", "close"):
                df[c] = df[c] * f
        X, sig = Fe.compute(df)
        if cut is not None:
            X = np.where(late[:, None], np.random.default_rng(1).standard_normal(X.shape), X)
        return PhaseData(df, X, sig)

    def _run(self, cfg, d, months):
        from btc.research.walk import monthly_update
        ens, anc, log = None, None, []
        for T in months:
            ens, anc, e = monthly_update(cfg, [d, d], T, (0, T.year, T.month, 0), ens, anc)
            log.append(e)
        return ens, anc, log

    def test_walk_plugin_two_months(self):
        d = self._phase()
        months = (_ts("2015-01-01"), _ts("2015-02-01"))
        # (1) 사전 등록 점검 그대로('p0'): 흐름이 walk 의 DQN 경로 규칙을 따르는지
        cfg = q1_cfg(phases=2, cold_steps=30, cold_split=20, ft_steps=10)
        ens, anc, log = self._run(cfg, d, months)
        self.assertEqual(log[0]["kind"], "cold")
        for k in ("pool", "td", "accepted", "exposure", "switches_per_year", "umax"):
            self.assertIn(k, log[0])
        if log[0]["accepted"]:
            self.assertEqual(log[1]["kind"], "finetune")
            self.assertIn("accepted", log[1])
        else:
            self.assertEqual(log[0]["note"], "gate_failed_no_model_hold_cash")
            self.assertEqual(log[1]["kind"], "cold")                  # 모델이 없으면 다음 달도 처음부터
        # (2) 점검 'none'으로 이어학습 경로를 확실히 탐: 앵커 = 1월 모델, 판단 가치 (B, K)
        cfg = q1_cfg(phases=2, cold_steps=30, cold_split=20, ft_steps=10, gate="none")
        ens1, anc1, _ = self._run(cfg, d, months[:1])
        ens, anc, log = self._run(cfg, d, months)
        self.assertEqual([e["kind"] for e in log], ["cold", "finetune"])
        self.assertTrue(all(e["accepted"] for e in log))
        self.assertTrue(all(np.array_equal(p, q) for p, q in zip(anc, ens1.net.params)))
        self.assertFalse(all(np.array_equal(p, q) for p, q in zip(ens.net.params, ens1.net.params)))
        sel = np.arange(d.T - 400, d.T - 10)
        U, umax = ens.values(d.X[sel], 0.003)
        self.assertEqual(U.shape, (len(sel), 2))
        self.assertTrue(np.all(np.isfinite(U)) and np.all(np.isfinite(umax)))
        self.assertTrue(np.all(U <= ens.mean_values(d.X[sel], 0.003) + 1e-9))
        self.assertTrue(all(np.isfinite(e["td"]) for e in log))

    def test_training_no_leak_after_T_k(self):
        """T_k 이후(ts ≥ T_k) 가격·지표를 바꿔도 파라미터·월 기록이 비트 단위로 같음 (처음부터 달·이어학습 달)"""
        cfg = q1_cfg(phases=2, cold_steps=30, cold_split=20, ft_steps=10, gate="none")
        d0 = self._phase()
        T1, T2 = _ts("2015-01-01"), _ts("2015-02-01")
        for months in ((T1,), (T1, T2)):
            e0, _, l0 = self._run(cfg, d0, months)
            e1, _, l1 = self._run(cfg, self._phase(cut=int(months[-1].timestamp())), months)
            self.assertTrue(all(np.array_equal(p, q) for p, q in zip(e0.net.params, e1.net.params)), months)
            self.assertEqual([{k: v for k, v in e.items() if k != "secs"} for e in l0],
                             [{k: v for k, v in e.items() if k != "secs"} for e in l1])
        # 민감도: 한 달 전부터 바꾸면 달라져야 함
        e0, _, _ = self._run(cfg, d0, (T1,))
        e2, _, _ = self._run(cfg, self._phase(cut=int(_ts("2014-12-01").timestamp())), (T1,))
        self.assertFalse(all(np.array_equal(p, q) for p, q in zip(e0.net.params, e2.net.params)))


if __name__ == "__main__":
    unittest.main()
