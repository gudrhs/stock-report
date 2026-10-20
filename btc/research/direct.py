# -*- coding: utf-8 -*-
"""
비교용(강화학습 아님): 샤프 비율을 직접 최대화하는 정책망 — Moody & Saffell, Lim et al.(2019) 방식.

  w_t = sigmoid(MLP(s_t)) ∈ [0, 1]            (보유 비중)
  r_t = w_t·m_t − c·|w_t − w_{t−1}|           (하루 단위, 비용 차감)
  손실 = −평균 로그성장(비용 차감) + ρ·평균|Δw|  (기본; 'sharpe'는 비중이 0으로 쪼그라드는 문제가 있어 참고용)

같은 입력(느린 추세 지표), 같은 재학습 일정(1월 처음부터·매월 이어서), 같은 5개 앙상블, 하루 한 번 판단.
이게 강화학습 변형과 비슷하면 'Q-러닝이라서 얻는 것'은 없다는 뜻입니다.
학습 표본: 학습 구간에서 60일 연속 구간을 무작위로(16격자 × 6시작) 뽑아 배치로 묶습니다.
"""
import numpy as np

from ..nn import StackedMLP, Adam
from .rl import feat_index, FEATURE_SETS


class DirectTrainer:
    def __init__(self, cfg, seed_seq):
        self.cfg = cfg
        self.feat_names = list(cfg.get("feat") or FEATURE_SETS["full22"])
        self.fi = feat_index(self.feat_names)
        ss = np.random.SeedSequence(seed_seq)
        a, b = ss.spawn(2)
        self.rng_init, self.rng = np.random.default_rng(a), np.random.default_rng(b)
        self.net = None

    def init_fresh(self):
        c = self.cfg
        self.net = StackedMLP(c["members"], [len(self.fi), *c["hidden"], 1], self.rng_init, last_scale=0.1)

    def init_from(self, params):
        self.init_fresh()
        self.net.set_params(params)

    def make_pool(self, datas, T_k, first_ts, warmup):
        """길이 L일 연속 구간의 시작점 목록 (모든 봉·격자 시작 가능, 구간 끝 O가 T_k 이전)"""
        c = self.cfg
        S, L = int(c.get("stride", 6)), int(c.get("seq_len", 60))
        self.S, self.L = S, L
        self.datas = datas[:c["phases"]]
        starts = []
        span = S * L + 1
        for k, d in enumerate(self.datas):
            t = np.arange(warmup, d.T - span - 1)
            t = t[d.ts[t] >= first_ts]
            t = t[(d.ts[t + span] + 4 * 3600) <= T_k]                           # 구간 끝(시가)이 T_k 이전
            t = t[(d.ts[t + span] - d.ts[t]) == span * 4 * 3600]               # 빠진 봉 없음
            starts.append(np.column_stack([np.full(len(t), k), t]))
        self.starts = np.concatenate(starts) if starts else np.zeros((0, 2), int)
        hl = c["half_life_years"] * 365.25 * 86400
        ts = np.array([self.datas[k].ts[t] for k, t in self.starts]) if len(self.starts) else np.zeros(0)
        wr = np.power(2.0, -(T_k - ts) / hl)
        w = c["recency_frac"] * wr / wr.sum() + (1 - c["recency_frac"]) / len(ts)
        self.cdf = np.cumsum(w / w.sum())
        return len(self.starts)

    def _seqs(self, n):
        j = np.minimum(np.searchsorted(self.cdf, self.rng.random(n)), len(self.cdf) - 1)
        S, L = self.S, self.L
        X = np.empty((n, L, len(self.fi)), np.float32)
        m = np.empty((n, L), np.float32)
        for i, (k, t) in enumerate(self.starts[j]):
            d = self.datas[k]
            idx = t + S * np.arange(L)
            X[i] = d.X[idx][:, self.fi]
            m[i] = np.log(d.o[idx + S + 1] / d.o[idx + 1])
        if self.cfg["noise"]:
            X += (self.cfg["noise"] * self.rng.standard_normal(X.shape)).astype(np.float32)
        return X, m

    def step(self, opt):
        c = self.cfg
        M = c["members"]
        n = c.get("seq_batch", 32)
        X, m = self._seqs(n)
        B = n * self.L
        Xf = X.reshape(B, -1)
        z = self.net.forward(np.broadcast_to(Xf, (M,) + Xf.shape).copy(), cache=True)[..., 0]   # (M, B)
        w = 1.0 / (1.0 + np.exp(-z))
        W = w.reshape(M, n, self.L)
        cost = c.get("cost_train", 0.003)
        dW = np.diff(np.concatenate([W[..., :1], W], axis=2), axis=2)                         # 첫날은 비용 0
        rho = c.get("turnover_pen", 0.0)
        N = n * self.L
        sgn = np.sign(dW)
        if c.get("obj", "log") == "sharpe":
            # 손실 = −샤프 + ρ·mean|Δw|  (주의: 샤프는 비중 크기에 무관해 비중이 0으로 쪼그라들 수 있음)
            r = W * m[None] - cost * np.abs(dW)
            mu = r.mean(axis=(1, 2), keepdims=True)
            sd = r.std(axis=(1, 2), keepdims=True) + 1e-8
            g_r = -(1.0 / (N * sd) - mu * (r - mu) / (N * sd ** 3))
            g_W = g_r * m[None]
            g_dW = g_r * (-cost * sgn) + rho * sgn / N
            val = -(mu / sd).mean()
        else:
            # 손실 = −평균 로그성장 + ρ·mean|Δw|,  성장 = ln(1 + w(e^m − 1)) + ln(1 − c|Δw|)
            em = np.expm1(m)[None]
            g_W = -(em / (1.0 + W * em)) / N
            g_dW = (cost * sgn / (1.0 - cost * np.abs(dW))) / N + rho * sgn / N
            val = -(np.log1p(W * em) + np.log1p(-cost * np.abs(dW))).mean()
        g_W += g_dW
        g_W[..., :-1] -= g_dW[..., 1:]
        g_z = (g_W.reshape(M, B) * w * (1 - w))[..., None].astype(np.float32)
        grads = self.net.backward(g_z)
        for i, p in enumerate(self.net.params):
            if p.ndim == 3 and p.shape[1] > 1:
                grads[i] = grads[i] + 2.0 * c["wd"] * p
            if getattr(self, "anchor", None) is not None:
                grads[i] = grads[i] + 2.0 * c["lambda_sp"] * (p - self.anchor[i])
        opt.step(grads)
        return float(val)

    def fit_cold(self):
        c = self.cfg
        self.anchor = None
        opt = Adam(self.net.params, lr=c["cold_lr1"], clip=c["clip"])
        split = c["cold_split"]
        out = []
        for s in range(c["cold_steps"]):
            if s == split:
                opt.lr = c["cold_lr2"]
            out.append(self.step(opt))
        return out

    def fit_finetune(self, anchor):
        c = self.cfg
        self.anchor = anchor
        opt = Adam(self.net.params, lr=c["ft_lr"], clip=c["clip"])
        return [self.step(opt) for _ in range(c["ft_steps"])]

    def weights(self, X):
        z = self.net.forward(X[:, self.fi].astype(np.float32), cache=False)[..., 0]
        return (1.0 / (1.0 + np.exp(-z))).mean(axis=0)
