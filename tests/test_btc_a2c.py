"""A1 a2c(정책경사 액터-크리틱) 단위 테스트 — 실제 데이터 없음, 합성 자료만. 약 30초.

python -m unittest tests.test_btc_a2c -v
  · 1단계 밴딧: 행동 표본 + GAE(길이 1) + 로짓 기울기로 만든 정책경사 추정이 정확한 기울기 π_a(r_a − J)의 불편추정
    (기준선 b가 있어도 없어도)
  · GAE = 반복문으로 따로 짠 공식, λ=1이면 부트스트랩 할인수익, λ=0이면 TD 오차
  · 액터 손실·비평가 손실·전체 손실(+L2·L2-SP)의 유한차분 기울기 (float64, 상대오차 < 1e-4)
  · 굴리기(rollout): 보상 = 공식, w_prev = 직전 표본 행동, 첫 w_prev ∈ 행동 목록, 부트스트랩은 61번째 날 지표
  · 합성 국면 경로에 90% 맞히는 신호를 TREND8[0](ret_42)에 심으면 짧은 학습으로 비중-국면 양의 상관
  · walk.monthly_update 플러그인으로 2달(처음부터 + 이어서), 비중 ∈ 25% 격자
  · 상태 있는 비중: 멱등, 달을 넘어 마지막 비중 이어받음, 행별 순전파 루프와 비트 단위로 같음, NaN 행은 유지
  · T_k 이후 가격·지표를 바꿔도 학습 결과가 비트 단위로 같음 (누출 없음)
"""
import os
import sys
import unittest

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from btc.nn import StackedMLP
from btc.features import NAMES
from btc.research.variants import VARIANTS, TREND8
from btc.research.algos import a2c as AC


def a1_cfg(**kw):
    cfg = AC.proposed_cfg()
    cfg.update(kw)
    return cfg


def _ts(s):
    return pd.Timestamp(s, tz="UTC")


_DATA = {}


def synth_phase(seed=1, n=7000, plant=False, perturb_after=None):
    """합성 국면 전환 4시간봉(2012-01부터) → (PhaseData, 국면). plant=True면 TREND8[0](ret_42) 자리에
    '다음 봉 국면'을 90% 맞히는 ±1 신호를 심음 (test_btc_controls와 같은 방식). perturb_after가 있으면 그 시각 이후
    가격·지표를 모두 흔듦"""
    key = (seed, n, plant, perturb_after)
    if key in _DATA:
        return _DATA[key]
    from test_btc_controls import _series
    from btc import features as Fe
    from btc.env import PhaseData
    df, reg = _series("regime", seed, n=n)
    late = None
    if perturb_after is not None:
        df = df.copy()
        late = df["ts"].to_numpy() >= perturb_after
        f = np.exp(np.cumsum(np.where(late, np.random.default_rng(9).normal(0, 0.2, len(df)), 0.0)))
        for c in ("open", "high", "low", "close"):
            df[c] = df[c] * f
    X, sig = Fe.compute(df)
    if plant:
        rng = np.random.default_rng(seed + 100)
        nxt = np.concatenate([reg[1:], reg[-1:]])
        noisy = np.where(rng.random(len(reg)) < 0.9, nxt, 1 - nxt)
        X[:, NAMES.index(TREND8[0])] = np.where(noisy == 1, 1.0, -1.0)
    if late is not None:
        X = np.where(late[:, None], np.random.default_rng(1).standard_normal(X.shape), X)
    _DATA[key] = (PhaseData(df, X, sig), reg)
    return _DATA[key]


class Estimator(unittest.TestCase):
    def test_bandit_policy_gradient_unbiased(self):
        """1단계 밴딧: E[−∂L/∂z] = ∂J/∂z = π_k(r_k − J),  J = Σ π_a r_a — 기준선이 있어도 불편"""
        rng = np.random.default_rng(0)
        z = np.array([0.3, -0.5, 1.0, 0.0, -1.2])
        pi = np.exp(AC.log_softmax(z))
        r_mean = np.array([0.02, -0.01, 0.005, 0.03, -0.02])
        J = float(pi @ r_mean)
        exact = pi * (r_mean - J)
        N = 400000
        zz = np.broadcast_to(z, (N, 5))
        a = AC.sample_actions(np.broadcast_to(pi, (N, 5)), rng.random(N))
        np.testing.assert_allclose(np.bincount(a, minlength=5) / N, pi, atol=4 * np.sqrt(pi * (1 - pi) / N).max())
        r = r_mean[a] + 0.01 * rng.standard_normal(N)
        for b in (0.0, 0.05):                                            # 기준선 V(s0) = b, 에피소드 끝 V = 0
            adv, ret = AC.gae(r[:, None], np.full((N, 1), b), np.zeros(N), 0.967, 0.95)
            np.testing.assert_allclose(adv[:, 0], r - b)
            np.testing.assert_allclose(ret[:, 0], r)
            _, _, g = AC.logit_grads(zz, a, adv[:, 0], 0.0, 1)             # 표본별 기울기 (N=1)
            est = -g.mean(axis=0)
            se = g.std(axis=0) / np.sqrt(N)
            self.assertTrue(np.all(np.abs(est - exact) < 4.5 * se), (b, est, exact, se))
        # 대조: 기울기가 0이 아닌 크기라서 위 검사가 실제로 무언가를 봄
        self.assertGreater(np.abs(exact).max(), 20 * se.max())

    def test_entropy_gradient(self):
        """엔트로피 항 기울기 = 유한차분 (정책경사 계수 0)"""
        rng = np.random.default_rng(1)
        z = rng.standard_normal((7, 5))
        a = rng.integers(0, 5, 7)
        _, H, g = AC.logit_grads(z, a, np.zeros(7), 0.3, 7)
        eps = 1e-6
        for i, k in [(0, 0), (3, 2), (6, 4)]:
            zp, zm = z.copy(), z.copy()
            zp[i, k] += eps
            zm[i, k] -= eps
            f = lambda zz: -0.3 * AC.logit_grads(zz, a, np.zeros(7), 0.3, 7)[1].mean()
            self.assertAlmostEqual((f(zp) - f(zm)) / (2 * eps), g[i, k], places=8)

    def test_gae_matches_loop(self):
        rng = np.random.default_rng(2)
        r, v, vb = rng.standard_normal((3, 9)), rng.standard_normal((3, 9)), rng.standard_normal(3)
        g, lam = 0.967, 0.95
        adv, ret = AC.gae(r, v, vb, g, lam)
        for i in range(3):
            vn = np.append(v[i, 1:], vb[i])
            for t in range(9):
                ref = sum((g * lam) ** k * (r[i, t + k] + g * vn[t + k] - v[i, t + k]) for k in range(9 - t))
                self.assertAlmostEqual(adv[i, t], ref, places=12)
        np.testing.assert_allclose(ret, adv + v)
        a1, _ = AC.gae(r, v, vb, g, 1.0)                                   # λ=1: 할인수익(부트스트랩) − V
        for t in (0, 4, 8):
            G = sum(g ** k * r[:, t + k] for k in range(9 - t)) + g ** (9 - t) * vb
            np.testing.assert_allclose(a1[:, t], G - v[:, t], atol=1e-12)
        a0, _ = AC.gae(r, v, vb, g, 0.0)                                   # λ=0: TD 오차
        np.testing.assert_allclose(a0, r + g * np.concatenate([v[:, 1:], vb[:, None]], 1) - v, atol=1e-12)

    def test_step_reward(self):
        a, wp, m, c = 0.75, 0.25, 0.04, 0.003
        self.assertAlmostEqual(AC.step_reward(a, wp, m, c),
                               np.log(1 + a * (np.exp(m) - 1)) + np.log(1 - c * abs(a - wp)), places=14)
        self.assertEqual(AC.step_reward(0.0, 0.0, 0.1, c), 0.0)


class Gradients(unittest.TestCase):
    def _setup(self, seed=0, M=2, n=3, L=5):
        cfg = a1_cfg(members=M, hidden=(6, 5), wd=3e-2, lambda_sp=1e-1)
        tr = AC.A2CTrainer(cfg, (seed,))
        rng = np.random.default_rng(seed)
        tr.net = StackedMLP(M, [tr.n_in, 6, 5, tr.K + 1], rng, last_scale=1.0, dtype=np.float64)
        for p in tr.net.params[1::2]:                   # 편향 0이면 ReLU 꺾임에 정확히 걸리는 행이 생김
            p += 0.3 * rng.standard_normal(p.shape)
        tr.anchor = [p + 0.1 * rng.standard_normal(p.shape) for p in tr.net.params]
        S = rng.standard_normal((M, n, L, tr.n_in))
        S[..., -1] = tr.acts[rng.integers(0, tr.K, (M, n, L))]
        a = rng.integers(0, tr.K, (M, n, L))
        adv = rng.standard_normal((M, n, L))
        ret = 0.2 * rng.standard_normal((M, n, L))
        return tr, S, a, adv, ret

    def _check(self, coefs, reg):
        tr, S, a, adv, ret = self._setup()
        if not reg:
            tr.cfg = dict(tr.cfg, wd=0.0, lambda_sp=0.0)
            tr.anchor = None
        _, grads, _ = tr.loss_grads(S, a, adv, ret, coefs)
        f = lambda: float(tr.loss_grads(S, a, adv, ret, coefs)[0].sum())
        rng = np.random.default_rng(7)
        eps = 1e-6
        for i, p in enumerate(tr.net.params):
            for _ in range(4):
                idx = tuple(rng.integers(0, s) for s in p.shape)
                old = p[idx]
                p[idx] = old + eps
                fp = f()
                p[idx] = old - eps
                fm = f()
                p[idx] = old
                num = (fp - fm) / (2 * eps)
                ana = float(grads[i][idx])
                rel = abs(num - ana) / max(abs(num), abs(ana), 1e-7)
                self.assertLess(rel, 1e-4, f"{coefs} param {i} {idx}: 수치 {num} 해석 {ana}")

    def test_actor_loss_fd(self):
        self._check((1.0, 0.0, 0.01), reg=False)

    def test_critic_loss_fd(self):
        self._check((0.0, 0.5, 0.0), reg=False)

    def test_full_loss_fd_with_l2_and_l2sp(self):
        self._check(None, reg=True)

    def test_loss_value_independent(self):
        """전체 손실 값 = 반복문으로 따로 짠 공식 (직접 짠 순전파)"""
        tr, S, a, adv, ret = self._setup(seed=3)
        loss, _, _ = tr.loss_grads(S, a, adv, ret)
        c = tr.cfg
        params = tr.net.copy_params()
        M = S.shape[0]
        for j in range(M):
            h = S[j].reshape(-1, S.shape[-1])
            for i in range(0, len(params), 2):
                h = h @ params[i][j] + params[i + 1][j]
                if i < len(params) - 2:
                    h = np.maximum(h, 0.0)
            tot = 0.0
            aa, dd, rr = a[j].ravel(), adv[j].ravel(), ret[j].ravel()
            N = len(aa)
            for b in range(N):
                zb = h[b, :tr.K]
                pb = np.exp(zb - zb.max())
                pb /= pb.sum()
                tot += -np.log(pb[aa[b]]) * dd[b] / N
                tot += 0.5 * (h[b, tr.K] - rr[b]) ** 2 / N
                tot -= 0.01 * -(pb * np.log(pb)).sum() / N
            for i, p in enumerate(params):
                if p.shape[1] > 1:
                    tot += c["wd"] * (p[j] ** 2).sum()
                tot += c["lambda_sp"] * ((p[j] - tr.anchor[i][j]) ** 2).sum()
            self.assertAlmostEqual(loss[j] / tot, 1.0, places=10)


class Rollout(unittest.TestCase):
    def test_rollout_consistency(self):
        d, _ = synth_phase()
        T_k = int(_ts("2015-01-01").timestamp())
        cfg = a1_cfg(phases=1, members=3, noise=0.0)
        tr = AC.A2CTrainer(cfg, (0, 2015, 1, 0))
        n = tr.make_pool([d], T_k, AC.FIRST_TRAIN, AC.WARMUP)
        self.assertGreater(n, 500)
        np.testing.assert_allclose(np.diff(tr.cdf), 1.0 / n, rtol=1e-6)      # 균등 추출 (recency_frac 0)
        tr.init_fresh()
        X, m = tr._seqs(4)
        self.assertEqual((X.shape, m.shape), ((4, 61, len(TREND8)), (4, 60)))
        ro = tr.rollout(X, m)
        A = tr.acts.astype(np.float64)
        S, a, r = ro["S"], ro["a"], ro["r"]
        self.assertTrue(np.all(np.isin(S[:, :, 0, -1], A)))              # 첫 w_prev ∈ 행동
        np.testing.assert_array_equal(S[:, :, 1:, -1], A[a[:, :, :-1]].astype(np.float32))   # w_prev = 직전 행동
        np.testing.assert_array_equal(S[:, :, :, :-1], np.broadcast_to(X[:, :60], S[..., :-1].shape))
        ref = np.log1p(A[a] * np.expm1(m[None].astype(np.float64))) + np.log1p(-0.003 * np.abs(A[a] - S[..., -1]))
        np.testing.assert_allclose(r, ref, rtol=1e-6, atol=1e-9)
        # V·부트스트랩 = 망 출력 (61번째 날 지표 + 마지막 행동)
        inp = np.concatenate([np.broadcast_to(X[:, 60], (3, 4, len(TREND8))), A[a[:, :, -1]][..., None]], axis=2)
        np.testing.assert_allclose(ro["v_boot"], tr.net.forward(inp.astype(np.float32), cache=False)[..., tr.K],
                                   rtol=1e-6)
        # 모든 구간이 T_k 이전에 끝남 (부트스트랩 날의 종가 포함)
        span = tr.S * tr.L + 1
        self.assertTrue(np.all(d.ts[tr.starts[:, 1] + span] + 4 * 3600 <= T_k))
        # 표본 m = 판단봉 다음 시가부터 하루 뒤 시가까지: 뽑힌 구간을 m으로 되찾아 확인
        r1 = np.log(d.o[7:] / d.o[1:-6])
        for i in range(4):
            hit = [t for t in tr.starts[:, 1] if np.allclose(r1[t + 6 * np.arange(60)], m[i], rtol=1e-5)]
            self.assertTrue(len(hit) >= 1)
            t = hit[0]
            np.testing.assert_allclose(X[i], d.X[t + 6 * np.arange(61)][:, tr.fi], rtol=1e-6)


class Learning(unittest.TestCase):
    def test_planted_signal_learned(self):
        """TREND8[0]에 90% 맞히는 국면 신호 → 짧은 학습(멤버 2개, 300걸음) 뒤 표본 밖 비중과 국면의 상관 > 0.4"""
        from btc.research.walk import decision_mask
        d, reg = synth_phase(seed=1, n=12000, plant=True)
        T = _ts("2016-01-01")
        Tk = int(T.timestamp())
        cfg = a1_cfg(phases=1, members=2, cold_steps=300, cold_split=200)
        tr = AC.A2CTrainer(cfg, (0, 2016, 1, 0))
        tr.make_pool([d], Tk, AC.FIRST_TRAIN, AC.WARMUP)
        tr.init_fresh()
        tr.fit_cold()
        mdl = AC.A2CModel(tr.net, tr.fi, tr.acts, 0.0)
        sel = np.flatnonzero(decision_mask(d, 6) & (d.ts + 4 * 3600 >= Tk))
        sel = sel[sel < d.T - 2]
        self.assertGreater(len(sel), 400)
        w = mdl.weights(d.X[sel])
        rg = reg[sel + 1]
        corr = np.corrcoef(w, rg)[0, 1]
        self.assertGreater(corr, 0.4, corr)
        self.assertGreater(w[rg == 1].mean() - w[rg == 0].mean(), 0.4)


class Monthly(unittest.TestCase):
    def test_walk_plugin_two_months_and_state(self):
        from btc.research.walk import monthly_update, decision_mask
        from btc.walkforward import decision_range
        d, _ = synth_phase()
        cfg = a1_cfg(phases=1, members=2, cold_steps=30, cold_split=20, ft_steps=10)
        ens, anchor, log, outs = None, None, [], []
        months = (_ts("2015-01-01"), _ts("2015-02-01"))
        mask = decision_mask(d, 6)
        for i, T_k in enumerate(months):
            ens, anchor, e = monthly_update(cfg, [d], T_k, (0, T_k.year, T_k.month, 0), ens, anchor)
            log.append(e)
            nxt = months[i + 1] if i + 1 < len(months) else _ts("2015-03-01")
            a, b = decision_range(d, int(T_k.timestamp()), int(nxt.timestamp()))
            sel = np.arange(a, b)[mask[a:b]]
            w = ens.weights(d.X[sel])
            outs.append((ens, sel, w))
        self.assertEqual([e["kind"] for e in log], ["cold", "finetune"])
        for e in log:
            self.assertTrue(e["accepted"])
            for k in ("loss", "pg", "vf", "ent", "reward", "exposure", "turnover"):
                self.assertTrue(np.isfinite(e[k]), k)
        self.assertEqual(len(anchor), len(ens.net.params))
        (m1, s1, w1), (m2, s2, w2) = outs
        self.assertTrue(len(s1) >= 28 and len(s2) >= 25)
        for w in (w1, w2):
            self.assertTrue(np.all(np.isin(w, AC.ACTS)))
        # 상태: 2월 모델은 1월 모델의 마지막 실행 비중에서 시작
        self.assertEqual(log[0]["pos0"], 0.0)
        self.assertEqual(log[1]["pos0"], w1[-1])
        self.assertEqual(m2.pos0, m1.last_pos)
        # 멱등: 같은 배치를 다시 불러도 같은 결과
        np.testing.assert_array_equal(m2.weights(d.X[s2]), w2)
        # 행별 순전파 루프(표 없이)와 같음
        for mdl, sel, w in outs:
            prev = mdl.pos0
            for i, t in enumerate(sel):
                e = mdl.expected_weight(d.X[t:t + 1], np.array([prev]))[0]
                prev = float(np.round(4 * e) / 4)
                self.assertEqual(prev, w[i])
        st = m2.state()
        self.assertTrue(all(isinstance(v, np.ndarray) for v in st.values()))
        np.testing.assert_array_equal(st["pos"], [m2.pos0, w2[-1]])

    def test_weights_nan_rows_hold_and_pos0(self):
        d, _ = synth_phase()
        cfg = a1_cfg(phases=1, members=2, cold_steps=20, cold_split=10)
        tr = AC.A2CTrainer(cfg, (1,))
        tr.make_pool([d], int(_ts("2015-01-01").timestamp()), AC.FIRST_TRAIN, AC.WARMUP)
        tr.init_fresh()
        for p in tr.net.params[-2:]:                                   # 출력이 w_prev·지표에 크게 반응하게
            p *= 300.0
        tr.net.params[-1][..., :tr.K] += np.linspace(-1, 1, tr.K).astype(np.float32)
        X = d.X[-60:].copy()
        mdl = AC.A2CModel(tr.net, tr.fi, tr.acts, 0.5)
        w = mdl.weights(X)
        self.assertGreater(len(np.unique(w)), 1)
        X[10, tr.fi[3]] = np.nan
        w2 = AC.A2CModel(tr.net, tr.fi, tr.acts, 0.5).weights(X)
        self.assertEqual(w2[10], w2[9])
        np.testing.assert_array_equal(w2[:10], w[:10])
        e = AC.A2CModel(tr.net, tr.fi, tr.acts, 0.5).weights(X[:0])
        self.assertEqual(len(e), 0)

    def test_training_no_leak_after_T_k(self):
        """T_k 이후(ts ≥ T_k) 가격·지표를 모두 바꿔도 학습된 파라미터·월 기록이 비트 단위로 같음 (처음부터·이어학습 달)"""
        from btc.research.walk import monthly_update
        T1, T2 = _ts("2015-01-01"), _ts("2015-02-01")
        cfg = a1_cfg(phases=2, members=2, cold_steps=20, cold_split=10, ft_steps=8)

        def run(d, months):
            ens, anc, log = None, None, []
            for T in months:
                ens, anc, e = monthly_update(cfg, [d, d], T, (0, T.year, T.month, 0), ens, anc)
                log.append(e)
            return ens, log

        d0, _ = synth_phase(seed=3)
        for months in ((T1,), (T1, T2)):
            cut = int(months[-1].timestamp())
            e0, l0 = run(d0, months)
            e1, l1 = run(synth_phase(seed=3, perturb_after=cut)[0], months)
            self.assertTrue(all(np.array_equal(p, q) for p, q in zip(e0.net.params, e1.net.params)), months)
            self.assertEqual(l0, l1)
        e0, _ = run(d0, (T1,))
        e2, _ = run(synth_phase(seed=3, perturb_after=int(_ts("2014-12-01").timestamp()))[0], (T1,))
        self.assertFalse(all(np.array_equal(p, q) for p, q in zip(e0.net.params, e2.net.params)))


class Config(unittest.TestCase):
    def test_proposed_cfg_is_r6_plus_a1_keys(self):
        a1, r6 = AC.proposed_cfg(), VARIANTS["R6_daily_trend8_uniform"]
        diff = {k for k in set(a1) | set(r6) if a1.get(k) != r6.get(k)}
        self.assertEqual(diff, {"name", "algo", "output", "acts", "reward", "cost_train", "seq_len", "seq_batch",
                                "gae_lambda", "ent_coef", "vf_coef", "adv_norm"})
        for k in ("stride", "gamma", "feat", "recency_frac", "members", "hidden", "cold_steps", "cold_lr1", "cold_lr2",
                  "cold_split", "ft_steps", "ft_lr", "lambda_sp", "wd", "noise", "clip", "phases"):
            self.assertEqual(a1[k], r6[k], k)
        self.assertEqual((a1["stride"], a1["gamma"], a1["recency_frac"], a1["members"]), (6, 0.967, 0.0, 5))
        self.assertEqual((a1["cold_steps"], a1["ft_steps"], a1["hidden"]), (2000, 300, (32, 32)))
        self.assertEqual(tuple(a1["acts"]), (0.0, 0.25, 0.5, 0.75, 1.0))
        self.assertEqual((a1["gae_lambda"], a1["ent_coef"], a1["vf_coef"], a1["cost_train"]), (0.95, 0.01, 0.5, 0.003))


if __name__ == "__main__":
    unittest.main()
