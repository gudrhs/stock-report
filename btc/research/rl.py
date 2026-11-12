# -*- coding: utf-8 -*-
"""
연구용 일반화 에이전트 — 핵심 코드(btc/agent.py 등)는 건드리지 않고 확장합니다.
(핵심 파일을 고치면 이미 끝난 P0 실행 결과의 코드 해시가 달라져 재현 기록이 깨지기 때문)

일반화한 것
  · 행동 = 목표 비중 K개 (예: 0, 0.5, 1)
  · 보상 효용: lin  = a·m                       (P0와 같음)
               mv   = a·m − (λ/2)·a²·m²           (평균-분산: 변동성에 벌점)
               log  = ln(1 + a·(e^m − 1))         (로그 성장: 비율 보유의 자연스러운 위험 회피)
  · 비용: κ·|a − p|·ln(1 − c)  (거래한 비중만큼)

작은 계좌는 가격을 움직이지 못하므로 가치함수 분해가 그대로 성립합니다:
  Q(s, p, a) = κ·|a−p|·ln(1−c) + U(s, c, a)
  U(s, c, a) = κ·R(a, m) + γ·max_a' [κ·|a'−a|·ln(1−c) + U(s', c, a')]
(비율 보유는 봉 사이 가격 변동으로 비중이 흘러가는데, 목표 비중으로 되돌리는 비용은 근사에서 뺍니다.
 백테스트는 실제로 흘러간 비중·재조정 비용을 그대로 계산합니다.)
"""
import math

import numpy as np

from ..agent import _flatten, LN_CMIN, LN_CMAX
from ..env import KAPPA
from ..features import NAMES, MIN8, c_in
from ..nn import StackedMLP, Adam


def reward(kind, a, m, lam=0.0):
    """보상 R(a, m) — a: 목표 비중 (배열 가능), m: 시가→시가 로그수익"""
    if kind == "lin":
        return a * m
    if kind == "mv":
        return a * m - 0.5 * lam * (a * a) * (m * m)
    if kind == "log":
        return np.log1p(a * np.expm1(m))
    if kind == "down":                       # 하락만 벌점 (반분산)
        dn = np.minimum(m, 0.0)
        return a * m - 0.5 * lam * (a * a) * (dn * dn)
    raise ValueError(kind)


def feat_index(names):
    return np.array([NAMES.index(n) for n in names])


FEATURE_SETS = {"full22": NAMES, "min8": MIN8}


class KEnsemble:
    def __init__(self, net, feat_names, acts):
        self.net = net
        self.feat_names = list(feat_names)
        self.fi = feat_index(self.feat_names) if set(self.feat_names) <= set(NAMES) else None
        self.acts = np.asarray(acts, dtype=float)

    def inputs(self, X, cost):
        x = X[:, self.fi].astype(np.float32) if self.fi is not None else X.astype(np.float32)
        ci = np.full((len(X), 1), c_in(cost), dtype=np.float32)
        return np.concatenate([x, ci], axis=1)

    def values(self, X, cost):
        """(B, K) 멤버 평균 U, (B,) 봉별 |U| 최댓값"""
        K = len(self.acts)
        U = self.net.forward(self.inputs(X, cost), cache=False)[..., :K].astype(np.float64)
        return U.mean(axis=0), np.abs(U).max(axis=(0, 2))

    def state(self):
        st = self.net.state("p")
        st["acts"] = self.acts
        st["feat"] = np.frombuffer(",".join(self.feat_names).encode(), dtype=np.uint8)
        return st


def k_policy(U, acts, cost, forced_hold=None, p0_idx=0, allowed=None):
    """
    U: (T, K) 평균 가치 → 목표 비중 번호 계열. argmax_a κ|a−p|ln(1−c) + U(s,a), 동점·NaN이면 유지.
    allowed: (T, K) 불리언 — 허용되지 않은 행동은 고르지 않음. 지금 보유가 허용되지 않으면 허용된 것 중 최선으로 강제 이동.
    """
    acts = np.asarray(acts, dtype=float)
    lnc = KAPPA * math.log(1.0 - cost)
    pen = np.abs(acts[None, :] - acts[:, None]) * lnc          # pen[p, a]
    out = np.empty(len(U), dtype=np.int64)
    p = p0_idx
    for t in range(len(U)):
        u = U[t]
        if not (forced_hold is not None and forced_hold[t]) and np.all(np.isfinite(u)):
            q = u + pen[p]
            if allowed is not None:
                q = np.where(allowed[t], q, -np.inf)
                if not allowed[t][p]:
                    p = int(np.argmax(q))
                    out[t] = p
                    continue
            best = int(np.argmax(q))
            if q[best] > q[p] + 1e-12:
                p = best
        out[t] = p
    return out


def trend_filter(c, n=1200):
    """B2 신호: 종가 > n봉 단순이동평균 (1200봉 ≈ 200일). 인과적."""
    import pandas as pd
    sma = pd.Series(c).rolling(n).mean().to_numpy()
    return (c > sma).astype(np.float32)


def allowed_matrix(b2, acts):
    """거부권 방식: 추세 규칙이 '현금'이면 보유 불가(0만 허용), '보유'면 전부 허용"""
    acts = np.asarray(acts)
    return np.where(b2[:, None] > 0.5, True, acts[None, :] == 0.0)


class KTrainer:
    """
    DDQN + 행동 보강 + 비용 분해, K개 목표 비중·일반 보상. 표본 추출·잡음·보조 출력·L2·L2-SP는 P0와 같습니다.
    cfg 추가 키: acts (목표 비중 목록), reward ('lin'|'mv'|'log'), lam (평균-분산 위험회피), feat (특징 이름 목록)
    """

    def __init__(self, cfg, seed_seq):
        self.cfg = cfg
        self.acts = np.asarray(cfg.get("acts", (0.0, 1.0)), dtype=np.float32)
        self.K = len(self.acts)
        self.feat_names = list(cfg.get("feat") or FEATURE_SETS[cfg.get("features", "full22")])
        self.fi = feat_index(self.feat_names)
        self.n_in = len(self.fi) + 1
        ss = np.random.SeedSequence(seed_seq)
        s_init, s_samp = ss.spawn(2)
        self.rng_init = np.random.default_rng(s_init)
        self.rng = np.random.default_rng(s_samp)
        self.net = self.tgt = None
        mid = float(np.mean(self.acts))
        self._mid = mid

    # ── 망 ──
    def _new_net(self):
        c = self.cfg
        return StackedMLP(c["members"], [self.n_in, *c["hidden"], self.K + 2], self.rng_init, last_scale=0.01)

    def init_fresh(self):
        self.net = self._new_net()
        self._sync_target()

    def init_from(self, params):
        self.net = self._new_net()
        self.net.set_params(params)
        self._sync_target()

    def _sync_target(self):
        self.tgt = StackedMLP.__new__(StackedMLP)
        self.tgt.__dict__.update(self.net.__dict__)
        self.tgt.params = self.net.copy_params()

    def ensemble(self):
        net = StackedMLP.__new__(StackedMLP)
        net.__dict__.update(self.net.__dict__)
        net.params = self.net.copy_params()
        net._cache = None
        return KEnsemble(net, self.feat_names, self.acts)

    # ── 표본 ──
    def make_pool(self, datas, T_k, first_ts, warmup):
        """
        학습 표본. stride S>1이면 '하루 한 번 판단'용: 상태 s_t → s_{t+S}, 보상은 그 사이 S봉 시가→시가 수익 합.
        모든 4시간봉을 시작점으로 쓰므로(격자 16 × 시작 6) 표본 수는 줄지 않습니다.
        """
        c = self.cfg
        S = int(c.get("stride", 1))
        hl = c["half_life_years"] * 365.25 * 86400
        use = datas[:c["phases"]]
        flat = _flatten(use)
        mkey = f"m_S{S}"
        if mkey not in flat:
            parts = []
            for d in use:
                mS = np.full(d.T, np.nan, np.float32)
                if d.T > S + 1:
                    mS[:d.T - S - 1] = np.log(d.o[S + 1:] / d.o[1:d.T - S])
                parts.append(mS)
            flat[mkey] = np.concatenate(parts)
        self.mS = flat[mkey]
        self.S = S
        g_idx, pool_w = [], []
        for k, d in enumerate(use):
            t = d.eligible(T_k, first_ts, warmup)
            if S > 1 and len(t):
                t = t[t + S + 1 < d.T]
                t = t[(d.ts[t + S + 1] - d.ts[t + 1]) == S * 4 * 3600]     # 그 하루 안에 빠진 봉 없음
                t = t[(d.ts[t + S] + 4 * 3600) <= T_k]
            if len(t) == 0:
                continue
            age = (T_k - d.ts[t]) / hl
            wr = np.power(2.0, -age)
            wr /= wr.sum()
            wu = np.full(len(t), 1.0 / len(t))
            w = c["recency_frac"] * wr + (1 - c["recency_frac"]) * wu
            g_idx.append(flat["off"][k] + t)
            pool_w.append(w / len(use))
        # 기대수익(드리프트) 제거: 에이전트가 실제로 보는 표본 분포(최근 가중 포함)의 평균 수익 μ를 빼고
        # α·μ만 되돌려 줌 (α=1이면 원래대로, 0이면 '오를 거라는 사전 믿음' 없이 시점 선택만 배움)
        self.drift_alpha = float(c.get("drift_alpha", 1.0))
        self.mu = 0.0
        if self.drift_alpha != 1.0 and g_idx:
            allg = np.concatenate(g_idx)
            wts = np.concatenate(pool_w)
            vals = self.mS[allg].astype(np.float64)
            ok = np.isfinite(vals)
            self.mu = float(np.sum(wts[ok] * vals[ok]) / np.sum(wts[ok]))
        # 거부권 방식: 추세 규칙 신호 (행동 허용 여부)
        self.veto = bool(c.get("veto_b2", False))
        if self.veto and "b2" not in flat:
            flat["b2"] = np.concatenate([trend_filter(d.c) for d in use])
        self.flat = flat
        key = "X_" + ",".join(self.feat_names)
        if key not in flat:
            flat[key] = np.ascontiguousarray(flat["X"][:, self.fi])
        self.Xsel = flat[key]
        self.pool = np.concatenate(g_idx)
        w = np.concatenate(pool_w)
        self.pool_cdf = np.cumsum(w / w.sum())
        self.perm = self.rng.permutation(len(self.pool)) if c.get("permute") else None
        return len(self.pool)

    def _batch(self):
        c = self.cfg
        M, B = c["members"], c["batch"]
        u = self.rng.random((M, B))
        j = np.minimum(np.searchsorted(self.pool_cdf, u), len(self.pool_cdf) - 1)
        gi = self.pool[j]
        gf = self.pool[self.perm[j]] if self.perm is not None else gi
        F = self.flat
        nf = self.Xsel.shape[1]
        X0 = np.empty((M, B, nf + 1), np.float32)
        X1 = np.empty((M, B, nf + 1), np.float32)
        X0[..., :nf] = self.Xsel[gf]
        X1[..., :nf] = self.Xsel[gf + self.S]
        if c["noise"]:
            nz = self.rng.standard_normal((2, M, B, nf), dtype=np.float32)
            nz *= np.float32(c["noise"])
            X0[..., :nf] += nz[0]
            X1[..., :nf] += nz[1]
        lc = self.rng.uniform(LN_CMIN, LN_CMAX, size=(M, B))
        ci = ((lc - math.log(0.00316)) / 1.846).astype(np.float32)
        X0[..., nf] = ci
        X1[..., nf] = ci
        m = self.mS[gi]
        if self.drift_alpha != 1.0:
            m = (m - (1.0 - self.drift_alpha) * self.mu).astype(np.float32)
        self._b2_next = F["b2"][gi + self.S] if self.veto else None
        self._b2_now = F["b2"][gi] if self.veto else None
        return X0, X1, m, np.exp(lc).astype(np.float32), F["z6"][gi], F["z42"][gi], F["ok"][gi]

    # ── 목표·손실 ──
    def targets(self, X1, m, cost):
        c = self.cfg
        g = c["gamma"]
        A = self.acts
        K = self.K
        lnc = (KAPPA * np.log1p(-cost))[..., None, None]                     # (M,B,1,1)
        pen = np.abs(A[None, :] - A[:, None])[None, None] * lnc               # (M,B,K_from,K_to)
        R = reward(c.get("reward", "lin"), A[None, None, :], m[..., None], c.get("lam", 0.0))
        R0 = reward(c.get("reward", "lin"), np.float32(self._mid), m, c.get("lam", 0.0))[..., None]
        base = KAPPA * (R - R0)                                                # (M,B,K)
        if g > 0:
            Uon = self.net.forward(X1, cache=False)[..., :K]
            Utg = self.tgt.forward(X1, cache=False)[..., :K]
            q = Uon[..., None, :] + pen                                         # (M,B,from,to)
            if getattr(self, "_b2_next", None) is not None:
                # 다음 상태에서 추세 규칙이 '현금'이면 보유 행동은 고를 수 없음
                allow = np.where(self._b2_next[..., None] > 0.5, True, (A == 0.0)[None, None, :])
                q = np.where(allow[..., None, :], q, -np.inf)
            astar = q.argmax(axis=-1)                                           # (M,B,K)
            boot = np.take_along_axis(Utg[..., None, :] + pen, astar[..., None], axis=-1)[..., 0]
            y = base + g * boot
        else:
            y = base
        return y.astype(np.float32)

    def loss_grads(self, X0, y, z6, z42, ok, anchor=None, lam_sp=0.0):
        c = self.cfg
        K = self.K
        hub = lambda d: np.where(np.abs(d) <= 1, 0.5 * d * d, np.abs(d) - 0.5)
        out = self.net.forward(X0, cache=True)
        B = y.shape[1]
        d = out[..., :K] - y
        a6 = out[..., K] - z6
        a42 = out[..., K + 1] - z42
        w = c["aux_w"]
        loss = hub(d).sum() / (K * B) + w * (ok * hub(a6)).sum() / B + w * (ok * hub(a42)).sum() / B
        gU = np.clip(d, -1.0, 1.0) / (K * B)
        g6 = w * np.clip(a6, -1.0, 1.0) * ok / B
        g42 = w * np.clip(a42, -1.0, 1.0) * ok / B
        gout = np.concatenate([gU, g6[..., None], g42[..., None]], axis=2).astype(self.net.params[0].dtype)
        grads = self.net.backward(gout)
        for i, p in enumerate(self.net.params):
            if p.ndim == 3 and p.shape[1] > 1:
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
        out = []
        for s in range(steps):
            if s == split:
                opt.lr = c["cold_lr2"]
            out.append(self.step(opt))
        return out

    def fit_finetune(self, anchor):
        c = self.cfg
        self._sync_target()
        opt = Adam(self.net.params, lr=c["ft_lr"], clip=c["clip"])
        return [self.step(opt, anchor=anchor, lam_sp=c["lambda_sp"]) for _ in range(c["ft_steps"])]
