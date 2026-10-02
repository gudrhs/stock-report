# -*- coding: utf-8 -*-
"""
강화학습 에이전트 — Double DQN, 행동 보강(AA), 비용 분해.

상태는 차트 지표 s(22개)와 현재 보유 p(현금/코인), 행동 a도 현금/코인 둘입니다.
작은 계좌는 가격에 영향을 주지 않으므로 다음 상태의 차트는 행동과 무관하고,
다음 보유는 곧 a입니다. 그래서 가치함수가 정확히 이렇게 나뉩니다:

  Q(s, p, a) = κ·|a−p|·ln(1−c)  +  U(s, c, a)
               └ 지금 갈아타는 비용 ┘  └ a를 들고 가는 미래 가치(신경망) ┘

  U(s,c,a) = κ·a·m + γ·max_a' [ κ·|a'−a|·ln(1−c) + U(s',c,a') ]

· 두 행동의 목표값을 표본마다 모두 정확히 계산할 수 있어(행동 보강) 탐험이 필요 없습니다.
· 판단: 현금이면 U1−U0 > −κ·ln(1−c) 일 때 매수, 보유 중이면 U1−U0 < κ·ln(1−c) 일 때 매도.
  비용만큼의 '문턱'이 저절로 생겨 잔매매를 하지 않습니다.
· 목표에서 ½·κ·m을 두 행동에 똑같이 빼 분산을 줄입니다(선택은 바뀌지 않음).
· 보조 출력(앞으로 6봉·42봉 수익률)은 규제용이고 판단에는 쓰지 않습니다.
"""
import math

import numpy as np

from .env import KAPPA
from .features import NAMES, MIN8, c_in, C_MIN, C_MAX
from .nn import StackedMLP, Adam

LN_CMIN, LN_CMAX = math.log(C_MIN), math.log(C_MAX)


def feat_index(features):
    names = NAMES if features == "full22" else MIN8
    return np.array([NAMES.index(n) for n in names])


class Ensemble:
    """학습된 앙상블 (판단 전용)"""

    def __init__(self, net, features):
        self.net = net
        self.features = features
        self.fi = feat_index(features)

    def inputs(self, X, cost):
        """X: (B, 22) 원시 특징 → (B, I) 입력 (선택 특징 + 비용)"""
        x = X[:, self.fi].astype(np.float32)
        ci = np.full((len(X), 1), c_in(cost), dtype=np.float32)
        return np.concatenate([x, ci], axis=1)

    def U(self, X, cost):
        """(M, B, 2) — 멤버별 U(s, c, a)"""
        out = self.net.forward(self.inputs(X, cost), cache=False)
        return out[..., :2]

    def delta(self, X, cost):
        """평균 U1 − 평균 U0 (B,) 와 봉별 |U| 최댓값 (B,) — 봉마다 따로 (다른 봉 값이 섞이지 않게)"""
        U = self.U(X, cost).astype(np.float64)
        mu = U.mean(axis=0)
        return mu[:, 1] - mu[:, 0], np.abs(U).max(axis=(0, 2)) if U.size else np.zeros(len(X))

    def state(self):
        st = self.net.state("p")
        st["features"] = np.frombuffer(self.features.encode(), dtype=np.uint8)
        return st

    @classmethod
    def from_state(cls, st):
        net = StackedMLP.from_state(st, "p")
        return cls(net, bytes(np.asarray(st["features"], dtype=np.uint8)).decode())


def threshold(cost):
    """갈아타는 문턱 = −κ·ln(1−c)"""
    return -KAPPA * math.log(1.0 - cost)


def policy_from_delta(delta, cost, forced_hold=None, p0=0):
    """
    Δ 계열 → 보유 계열 (결정 규칙: argmax_a κ|a−p|ln(1−c) + U(s,a)).
    NaN인 Δ는 판단 보류(직전 유지). 동점이면 현재 보유 유지.
    """
    th = threshold(cost)
    out = np.empty(len(delta))
    p = p0
    for i, d in enumerate(delta):
        if not (forced_hold is not None and forced_hold[i]) and not np.isnan(d):
            if p == 0 and d > th:
                p = 1
            elif p == 1 and d < -th:
                p = 0
        out[i] = p
    return out


_FLAT = {"datas": None, "flat": None}


def _flatten(datas):
    """
    여러 phase 배열을 하나로 이어 붙여 한 번의 인덱싱으로 표본을 꺼냅니다 (phase 사이 경계는 표본 조건이 막음).
    같은 객체 목록일 때만 재사용합니다 — id()만 비교하면 해제된 객체의 번호를 새 객체가 물려받아
    엉뚱한 데이터로 학습할 수 있습니다. 객체 자체를 잡아 두고 'is'로 비교합니다.
    """
    prev = _FLAT["datas"]
    if prev is None or len(prev) != len(datas) or any(a is not b for a, b in zip(prev, datas)):
        off = np.cumsum([0] + [d.T for d in datas[:-1]])
        _FLAT["datas"] = list(datas)
        _FLAT["flat"] = dict(
            off=off,
            X=np.concatenate([d.X for d in datas]).astype(np.float32),
            m=np.concatenate([d.m for d in datas]).astype(np.float32),
            z6=np.nan_to_num(np.concatenate([d.z[6] for d in datas])).astype(np.float32),
            z42=np.nan_to_num(np.concatenate([d.z[42] for d in datas])).astype(np.float32),
            ok=np.concatenate([d.aux_ok & ~np.isnan(d.z[42]) for d in datas]).astype(np.float32))
    return _FLAT["flat"]


class Trainer:
    """DDQN-AA 학습기 — 5개 멤버를 한꺼번에"""

    def __init__(self, cfg, seed_seq):
        self.cfg = cfg
        self.fi = feat_index(cfg["features"])
        self.n_in = len(self.fi) + 1
        ss = np.random.SeedSequence(seed_seq)
        s_init, s_samp = ss.spawn(2)
        self.rng_init = np.random.default_rng(s_init)
        self.rng = np.random.default_rng(s_samp)
        self.net = None
        self.tgt = None

    def init_fresh(self):
        c = self.cfg
        self.net = StackedMLP(c["members"], [self.n_in, *c["hidden"], 4], self.rng_init, last_scale=0.01)
        self.tgt = StackedMLP.__new__(StackedMLP)
        self.tgt.__dict__.update(self.net.__dict__)
        self.tgt.params = self.net.copy_params()

    def init_from(self, params):
        c = self.cfg
        self.net = StackedMLP(c["members"], [self.n_in, *c["hidden"], 4], self.rng_init, last_scale=0.01)
        self.net.set_params(params)
        self.tgt = StackedMLP.__new__(StackedMLP)
        self.tgt.__dict__.update(self.net.__dict__)
        self.tgt.params = self.net.copy_params()

    def ensemble(self):
        net = StackedMLP.__new__(StackedMLP)
        net.__dict__.update(self.net.__dict__)
        net.params = self.net.copy_params()
        net._cache = None
        return Ensemble(net, self.cfg["features"])

    # ── 표본 뽑기 ──
    def make_pool(self, datas, T_k, first_ts, warmup):
        """phase별 학습 가능 봉과 뽑힐 확률 (phase 균등, 그 안에서 최근 가중 70% + 균등 30%)"""
        c = self.cfg
        hl = c["half_life_years"] * 365.25 * 86400
        use = datas[:c["phases"]]
        flat = _flatten(use)
        g_idx, pool_w = [], []
        for k, d in enumerate(use):
            t = d.eligible(T_k, first_ts, warmup)
            if len(t) == 0:
                continue
            age = (T_k - d.ts[t]) / hl
            wr = np.power(2.0, -age)
            wr /= wr.sum()
            wu = np.full(len(t), 1.0 / len(t))
            w = c["recency_frac"] * wr + (1 - c["recency_frac"]) * wu
            g_idx.append(flat["off"][k] + t)
            pool_w.append(w / len(use))
        self.flat = flat
        fkey = "X_" + c["features"]
        if fkey not in flat:
            flat[fkey] = np.ascontiguousarray(flat["X"][:, self.fi])
        self.Xsel = flat[fkey]
        self.pool = np.concatenate(g_idx)
        w = np.concatenate(pool_w)
        self.pool_cdf = np.cumsum(w / w.sum())
        if self.cfg["permute"]:
            # 무작위 대조: 학습 표본의 특징을 시간상 섞어 보상과의 연결을 끊음 (N2 귀무모형)
            self.perm = self.rng.permutation(len(self.pool))
        else:
            self.perm = None
        return len(self.pool)

    def _batch(self):
        c = self.cfg
        M, B = c["members"], c["batch"]
        u = self.rng.random((M, B))
        j = np.minimum(np.searchsorted(self.pool_cdf, u), len(self.pool_cdf) - 1)
        gi = self.pool[j]                                          # 보상을 가져올 표본
        gf = self.pool[self.perm[j]] if self.perm is not None else gi   # 특징을 가져올 표본
        F = self.flat
        S0 = self.Xsel[gf]
        S1 = self.Xsel[gf + 1]
        m = F["m"][gi]
        z6 = F["z6"][gi]
        z42 = F["z42"][gi]
        ok = F["ok"][gi]
        nf = S0.shape[2]
        X0 = np.empty((M, B, nf + 1), np.float32)
        X1 = np.empty((M, B, nf + 1), np.float32)
        X0[..., :nf] = S0
        X1[..., :nf] = S1
        if c["noise"]:
            nz = self.rng.standard_normal((2, M, B, nf), dtype=np.float32)
            nz *= np.float32(c["noise"])
            X0[..., :nf] += nz[0]
            X1[..., :nf] += nz[1]
        lc = self.rng.uniform(LN_CMIN, LN_CMAX, size=(M, B))     # 비용 ~ 로그균등 [0.05%, 2%]
        cost = np.exp(lc).astype(np.float32)
        ci = ((lc - math.log(0.00316)) / 1.846).astype(np.float32)
        X0[..., nf] = ci
        X1[..., nf] = ci
        return X0, X1, m, cost, z6, z42, ok

    # ── 한 스텝 ──
    def targets(self, X1, m, cost):
        """y(a) = κ(a−½)m + γ[κ|a*−a|ln(1−c) + U_tgt(s', a*)],  a* = argmax_a' [κ|a'−a|ln(1−c) + U_on(s', a')]"""
        g = self.cfg["gamma"]
        lnc = KAPPA * np.log1p(-cost)                             # κ·ln(1−c) (음수)
        base = KAPPA * m
        y = np.empty(m.shape + (2,), np.float32)
        if g > 0:
            Uon = self.net.forward(X1, cache=False)[..., :2]
            Utg = self.tgt.forward(X1, cache=False)[..., :2]
            for a in (0, 1):
                q0 = abs(0 - a) * lnc + Uon[..., 0]
                q1 = abs(1 - a) * lnc + Uon[..., 1]
                boot = np.where(q1 > q0, abs(1 - a) * lnc + Utg[..., 1], abs(0 - a) * lnc + Utg[..., 0])
                y[..., a] = (a - 0.5) * base + g * boot
        else:
            y[..., 0] = -0.5 * base
            y[..., 1] = 0.5 * base
        return y

    def loss_grads(self, X0, y, z6, z42, ok, anchor=None, lam_sp=0.0):
        """
        손실(멤버 합)과 기울기. y는 상수로 둡니다(준경사 TD).
          Σ_m [ mean_{b,a} Huber(U−y) + w·mean_b ok·Huber(aux6−z6) + w·mean_b ok·Huber(aux42−z42) ]
          + wd·ΣW² + λ_sp·Σ(θ−θ_anchor)²
        """
        c = self.cfg
        hub = lambda d: np.where(np.abs(d) <= 1, 0.5 * d * d, np.abs(d) - 0.5)
        out = self.net.forward(X0, cache=True)
        B = y.shape[1]
        d = out[..., :2] - y
        a6 = out[..., 2] - z6
        a42 = out[..., 3] - z42
        w = c["aux_w"]
        loss = (hub(d).sum() / (2.0 * B) + w * (ok * hub(a6)).sum() / B + w * (ok * hub(a42)).sum() / B)
        gU = np.clip(d, -1.0, 1.0) / (2.0 * B)
        g6 = w * np.clip(a6, -1.0, 1.0) * ok / B
        g42 = w * np.clip(a42, -1.0, 1.0) * ok / B
        gout = np.concatenate([gU, g6[..., None], g42[..., None]], axis=2).astype(self.net.params[0].dtype)
        grads = self.net.backward(gout)
        for i, p in enumerate(self.net.params):
            if p.ndim == 3 and p.shape[1] > 1:                  # 가중치에만 L2
                grads[i] = grads[i] + 2.0 * c["wd"] * p
                loss += c["wd"] * float((p.astype(np.float64) ** 2).sum())
            if anchor is not None and lam_sp:
                diff = p - anchor[i]
                grads[i] = grads[i] + 2.0 * lam_sp * diff
                loss += lam_sp * float((diff.astype(np.float64) ** 2).sum())
        return float(loss), grads, float(hub(d).mean())

    def step(self, opt, anchor=None, lam_sp=0.0):
        X0, X1, m, cost, z6, z42, ok = self._batch()
        y = self.targets(X1, m, cost)
        _, grads, td = self.loss_grads(X0, y, z6, z42, ok, anchor, lam_sp)
        opt.step(grads)
        tau = self.cfg["tau"]
        for pt, p in zip(self.tgt.params, self.net.params):
            pt *= (1 - tau)
            pt += tau * p
        return td

    def fit_cold(self, steps=None):
        c = self.cfg
        steps = c["cold_steps"] if steps is None else steps
        split = int(round(steps * c["cold_split"] / c["cold_steps"]))
        opt = Adam(self.net.params, lr=c["cold_lr1"], clip=c["clip"])
        losses = []
        for s in range(steps):
            if s == split:
                opt.lr = c["cold_lr2"]
            losses.append(self.step(opt))
        return losses

    def fit_finetune(self, anchor):
        c = self.cfg
        # 이어 학습: Adam 모멘트 초기화, 목표망을 현재 가중치로
        self.tgt.params = self.net.copy_params()
        opt = Adam(self.net.params, lr=c["ft_lr"], clip=c["clip"])
        losses = []
        for _ in range(c["ft_steps"]):
            losses.append(self.step(opt, anchor=anchor, lam_sp=c["lambda_sp"]))
        return losses

    def td_loss(self, datas, idx_by_phase, cost=None, n=4096):
        """검증용: 주어진 봉들의 TD 손실 + 보조 손실 (예산 연구에만 사용, 손익은 보지 않음)"""
        c = self.cfg
        rng = np.random.default_rng(0)
        tot, cnt = 0.0, 0
        for k, idx in idx_by_phase.items():
            d = datas[k]
            if len(idx) == 0:
                continue
            tt = idx if len(idx) <= n else rng.choice(idx, n, replace=False)
            lc = rng.uniform(LN_CMIN, LN_CMAX, size=len(tt))
            ci = ((lc - math.log(0.00316)) / 1.846).astype(np.float32)[:, None]
            X0 = np.concatenate([d.X[tt][:, self.fi], ci], axis=1)
            X1 = np.concatenate([d.X[tt + 1][:, self.fi], ci], axis=1)
            lnc = KAPPA * np.log1p(-np.exp(lc))
            m = d.m[tt]
            g = c["gamma"]
            Uon = self.net.forward(X1, cache=False)[..., :2]
            Utg = self.tgt.forward(X1, cache=False)[..., :2]
            out = self.net.forward(X0, cache=False)
            for a in (0, 1):
                if g > 0:
                    q0 = abs(0 - a) * lnc + Uon[..., 0]
                    q1 = abs(1 - a) * lnc + Uon[..., 1]
                    boot = np.where(q1 > q0, abs(1 - a) * lnc + Utg[..., 1], abs(0 - a) * lnc + Utg[..., 0])
                else:
                    boot = 0.0
                y = (a - 0.5) * KAPPA * m + g * boot
                dd = out[..., a] - y
                tot += np.where(np.abs(dd) <= 1, 0.5 * dd * dd, np.abs(dd) - 0.5).mean() * len(tt) / 2
            ok = d.aux_ok[tt] & ~np.isnan(d.z[42][tt])
            for j, hh in ((2, 6), (3, 42)):
                dd = (out[..., j] - np.nan_to_num(d.z[hh][tt])) * ok
                tot += c["aux_w"] * np.where(np.abs(dd) <= 1, 0.5 * dd * dd, np.abs(dd) - 0.5).mean() * len(tt)
            cnt += len(tt)
        return tot / max(cnt, 1)
