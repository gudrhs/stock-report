# -*- coding: utf-8 -*-
"""
W2 'dp_band' — 느린 추세 예측 하나를 벨만 방정식으로 '정확히' 거래하기 (모델 기반, 결정론적, 진단용).

btc/research/rl_effective_methods.md §3 W2의 사전 등록 설계를 그대로 구현합니다. 값을 조정하지 않습니다.
"느린 신호를 우리 비용(편도 0.15%)에서 최적으로 거래하면 얼마나 버나"에 대한 사실상의 상한 답입니다.

매월 T_k(1일 00:00 UTC) 재학습 — 매달 처음부터 다시 맞춤 (학습 시드 없음, 앙상블 없음)
  1. 학습 표본: phase 0..phases−1의 하루 판단봉(walk.decision_mask, stride 6) 중
     봉 시각 ≥ 2014-01-01(FIRST_TRAIN), 지표 준비(WARMUP) 이후, TREND8 8개가 모두 유한한 봉.
     목표 y_t = ln(O[t+1+120] / O[t+1]) / 20   (다음 봉 시가부터 20일 시가→시가 로그수익의 하루 평균)
     퍼지: 레이블 구간 끝 봉이 T_k 전에 마감한 표본만 — ts[t+1+120] + 4h ≤ T_k  (direct.py make_pool과 같은 규칙)
     그리고 레이블 구간에 빠진 봉이 없어야 함 (ts[t+121] − ts[t+1] = 20일).
  2. 릿지: TREND8을 학습 표본의 평균·표준편차로 표준화, α = 1.0 (sklearn Ridge와 같은 정의:
     ‖y − ȳ − Zβ‖² + α‖β‖², 절편은 벌점 없음 = ȳ).
  3. 예측: p_t = μ0 + (ŷ_t − mean_학습(ŷ)),  μ0 = 0.5 × mean_학습(y)  (같은 퍼지된 표본의 평균 일일 로그수익의 절반)
  4. AR(1): phase 0의 하루 p 계열(종가가 T_k 이전인 판단봉, 연속된 날짜 쌍만)에 최소제곱
     p_{t+1} = a + φ·p_t + e,  e ~ N(0, s²)
     (p는 지표만으로 계산하므로 레이블 퍼지와 무관하게 T_k 직전 날까지 씁니다 — 미래 정보 없음)
  5. 동적계획: p 격자 201칸(정상분포 평균 ± 4 표준편차), 포지션 {0, 1},
     하루 보상 = a·p (기대 로그성장), 전환 비용 = −ln(1 − 0.0015)·|a − 이전|, 할인 γ = 0.998,
     전이 = AR(1)을 격자 칸 경계(중점)에서 정규 CDF로 나눔 (Tauchen, 양 끝 칸은 무한대까지),
     가치 반복을 sup-노름 변화 < 1e-12 까지. 결과 pi[p칸, 이전 포지션].
     → 히스테리시스 문턱: q_in(현금일 때 p > q_in 이면 진입), q_out(보유일 때 p < q_out 이면 청산).
       문턱은 정책이 바뀌는 두 격자점의 중점입니다 (가까운 격자점 조회와 같은 결과).
  6. 부호 규칙(p > 0 이면 보유)은 진단으로만 기록합니다. 후보가 아닙니다.

상태가 있는 판단 (중요)
  히스테리시스는 '이전 포지션'에 따라 판단이 달라지므로 모델이 상태를 가집니다.
  walk.run_replication은 한 달치 phase 0 판단봉을 '시간 순서대로' 한 번에 weights(X)로 넘깁니다.
  weights(X)는 행을 순서대로 처리하며 자기 내부 포지션을 갱신합니다.
    · 시작 포지션 pos0 = 지난달 모델(ens)이 마지막으로 낸 포지션 (ens.last_pos), 없으면 0(현금)
    · weights(X)는 매번 pos0에서 다시 시작합니다 → 같은 배치로 여러 번 불러도 결과가 같습니다(멱등).
    · 호출이 끝나면 last_pos를 마지막 행의 포지션으로 둡니다 → 다음 달 monthly_update가 이어받습니다.
    · 지표가 NaN인 행은 판단을 보류하고 이전 포지션을 유지합니다.
  주의: 행이 시간 순서가 아니거나, 여러 달을 섞어 넘기면 결과가 틀립니다. 강제 유지(forced_hold)로 실제
  체결이 목표와 달라지는 드문 경우에도 내부 포지션은 목표 기준으로 이어집니다.

cfg: algo='dp_band', output='weights', stride=6, phases, feat(없으면 TREND8), 그리고 아래 dp_* 키(기본값 = 사전 등록값).
"""
import math

import numpy as np

from ...env import BAR_SEC
from ...features import WARMUP
from ...walkforward import FIRST_TRAIN
from ..rl import feat_index
from ..variants import TREND8
from ..walk import decision_mask

BPD = 86400 // BAR_SEC                       # 하루 봉 수 (6)
DEFAULTS = dict(                             # 사전 등록값 — 조정하지 않음
    dp_horizon_days=20,                      # 레이블: 앞으로 20일 로그수익 / 20
    dp_ridge_alpha=1.0,
    dp_mu0_frac=0.5,                         # μ0 = 0.5 × 학습 구간 평균 일일 로그수익
    dp_grid_n=201,
    dp_grid_sd=4.0,                          # 격자 = 정상 평균 ± 4 표준편차
    dp_cost=0.0015,                          # 편도 비용 (전환 비용 −ln(1−c))
    dp_gamma=0.998,
    dp_tol=1e-12,
)
PHI_MAX = 0.9995        # 수치 보호용만: φ̂ ≥ 1이면 정상분포가 없어 격자를 못 만듦 (잘랐으면 기록)
MAX_ITER = 200000
MIN_SAMPLES = 100


def hparams(cfg):
    return {k: cfg.get(k, v) for k, v in DEFAULTS.items()}


# ══════════ 1. 학습 표본 (퍼지) ══════════
def daily_rows(d, first_ts=FIRST_TRAIN, warmup=WARMUP):
    """phase d의 하루 판단봉 번호 (phase 0은 00:00 UTC 마감 봉), 지표 준비 이후·first_ts 이후"""
    t = np.flatnonzero(decision_mask(d, BPD))
    return t[(t >= warmup) & (d.ts[t] >= first_ts)]


def make_samples(datas, T_k, fi, h_days=20, first_ts=FIRST_TRAIN, warmup=WARMUP):
    """
    릿지 학습 표본. 반환 (X (n,F), y (n,), idx (n,2)=[phase, 봉 번호], label_end (n,) 레이블 끝 봉의 마감 시각).
    레이블 y_t = ln(O[t+1+20·6] / O[t+1]) / 20. 레이블 끝 봉(시가를 쓰는 봉)이 T_k 전에 마감해야 함 (퍼지).
    """
    span = h_days * BPD
    Xs, ys, ids, ends = [], [], [], []
    for k, d in enumerate(datas):
        t = daily_rows(d, first_ts, warmup)
        t = t[t + 1 + span < d.T]
        j0, j1 = t + 1, t + 1 + span
        ok = (d.ts[j1] - d.ts[j0]) == span * BAR_SEC                     # 빠진 봉 없음
        ok &= (d.ts[j1] + BAR_SEC) <= T_k                                 # 퍼지: 레이블 끝이 T_k 이전
        t = t[ok]
        X = d.X[t][:, fi].astype(np.float64)
        y = np.log(d.o[t + 1 + span] / d.o[t + 1]) / h_days
        fin = np.isfinite(X).all(axis=1) & np.isfinite(y)
        t = t[fin]
        Xs.append(X[fin])
        ys.append(y[fin])
        ids.append(np.column_stack([np.full(len(t), k), t]))
        ends.append(d.ts[t + 1 + span] + BAR_SEC)
    if not Xs:
        return np.zeros((0, len(fi))), np.zeros(0), np.zeros((0, 2), int), np.zeros(0, np.int64)
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ids), np.concatenate(ends)


# ══════════ 2. 릿지 ══════════
def fit_ridge(X, y, alpha):
    """표준화(학습 표본 평균·표준편차) 후 릿지. 절편(ȳ)은 벌점 없음. 반환 (mu, sd, beta, ybar)"""
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Z = (X - mu) / sd
    yb = float(y.mean())
    A = Z.T @ Z + alpha * np.eye(Z.shape[1])
    beta = np.linalg.solve(A, Z.T @ (y - yb))
    return mu, sd, beta, yb


# ══════════ 3. AR(1) ══════════
def fit_ar1(x, y):
    """y = a + φ·x + e 최소제곱. 반환 (a, φ, s, 잘랐는지). φ는 수치 보호로 ±PHI_MAX 안에 둠"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xm, ym = x.mean(), y.mean()
    vx = ((x - xm) ** 2).sum()
    phi = float(((x - xm) * (y - ym)).sum() / vx) if vx > 0 else 0.0
    clipped = abs(phi) > PHI_MAX
    phi = float(np.clip(phi, -PHI_MAX, PHI_MAX))
    a = float(ym - phi * xm)
    e = y - a - phi * x
    s = float(math.sqrt((e ** 2).sum() / max(len(x) - 2, 1)))
    s = max(s, 1e-12)
    return a, phi, s, bool(clipped)


# ══════════ 4. 격자·전이·가치 반복 ══════════
_ERF = np.frompyfunc(math.erf, 1, 1)


def norm_cdf(z):
    z = np.asarray(z, float)
    return 0.5 * (1.0 + _ERF(z / math.sqrt(2.0)).astype(float))


def ar1_grid(a, phi, s, n=201, width=4.0):
    """정상분포 평균 ± width 표준편차 격자와 전이행렬 P[i, j] (칸 경계 = 중점, 양 끝 칸은 무한대까지)"""
    m = a / (1.0 - phi)
    sd = s / math.sqrt(1.0 - phi * phi)
    g = np.linspace(m - width * sd, m + width * sd, n)
    edges = 0.5 * (g[1:] + g[:-1])
    F = norm_cdf((edges[None, :] - (a + phi * g)[:, None]) / s)          # (n, n−1)
    P = np.diff(np.concatenate([np.zeros((n, 1)), F, np.ones((n, 1))], axis=1), axis=1)
    return g, np.maximum(P, 0.0)


def value_iteration(grid, P, kappa, gamma, tol=1e-12, max_iter=MAX_ITER):
    """
    상태 (p칸 i, 이전 포지션 prev), 행동 a ∈ {0,1}:
      Q(i, prev, a) = a·p_i − κ·|a − prev| + γ·Σ_j P[i,j]·V(j, a),   V = max_a Q
    반환 (V (G,2), pi (G,2) int8 = 다음 포지션, 반복 수, 마지막 변화량). 동점이면 이전 포지션 유지.
    """
    G = len(grid)
    r = np.column_stack([np.zeros(G), np.asarray(grid, float)])             # r[i, a] = a·p_i
    V = np.zeros((G, 2))
    diff, it = float("inf"), 0
    for it in range(1, max_iter + 1):
        base = r + gamma * (P @ V)                                          # base[i, a]
        Vn = np.empty_like(V)
        Vn[:, 0] = np.maximum(base[:, 0], base[:, 1] - kappa)
        Vn[:, 1] = np.maximum(base[:, 1], base[:, 0] - kappa)
        diff = float(np.abs(Vn - V).max())
        V = Vn
        if diff < tol:
            break
    base = r + gamma * (P @ V)
    pi = np.empty((G, 2), np.int8)
    pi[:, 0] = (base[:, 1] - kappa) > base[:, 0]                            # 현금 → 진입?
    pi[:, 1] = ~((base[:, 0] - kappa) > base[:, 1])                         # 보유 → 유지?
    return V, pi, it, diff


def thresholds(grid, pi):
    """
    pi에서 히스테리시스 문턱. 현금일 때 p > q_in 이면 진입, 보유일 때 p < q_out 이면 청산.
    문턱 = 정책이 0→1로 바뀌는 두 격자점의 중점 (전부 1이면 −inf, 전부 0이면 +inf).
    반환 (q_in, q_out, 단조 여부) — 단조가 아니면 weights는 격자 조회로 대신 판단합니다.
    """
    out, mono = [], True
    for col in (0, 1):
        on = pi[:, col].astype(bool)
        mono &= bool(np.all(np.diff(on.astype(int)) >= 0))
        if on.all():
            out.append(-math.inf)
        elif not on.any():
            out.append(math.inf)
        else:
            i = int(np.argmax(on))
            out.append(0.5 * (grid[i - 1] + grid[i]) if i > 0 else -math.inf)
    return out[0], out[1], bool(mono)


def solve_band(a, phi, s, hp):
    """AR(1) 계수 → 격자·가치 반복·문턱 한 번에"""
    g, P = ar1_grid(a, phi, s, hp["dp_grid_n"], hp["dp_grid_sd"])
    kappa = -math.log(1.0 - hp["dp_cost"])
    V, pi, it, diff = value_iteration(g, P, kappa, hp["dp_gamma"], hp["dp_tol"])
    q_in, q_out, mono = thresholds(g, pi)
    return dict(grid=g, P=P, V=V, pi=pi, iters=it, resid=diff, q_in=q_in, q_out=q_out, mono=mono, kappa=kappa)


def band_path(p, q_in, q_out, pos0=0.0, grid=None, pi=None):
    """p 계열을 시간 순서대로 히스테리시스 규칙에 통과시킨 포지션 (NaN이면 유지).
    grid·pi를 주면 문턱 대신 가까운 격자점의 정책을 조회합니다 (정책이 단조가 아닐 때)."""
    out = np.empty(len(p))
    prev = float(pos0)
    for i, x in enumerate(p):
        if np.isfinite(x):
            if grid is not None:
                j = int(np.clip(np.searchsorted(grid, x), 1, len(grid) - 1))
                j = j - 1 if (x - grid[j - 1]) <= (grid[j] - x) else j
                prev = float(pi[j, int(prev)])
            elif prev == 0.0 and x > q_in:
                prev = 1.0
            elif prev == 1.0 and x < q_out:
                prev = 0.0
        out[i] = prev
    return out


def sign_path(p, pos0=0.0):
    """부호 규칙(진단용): p > 0 이면 보유, NaN이면 유지"""
    out = np.empty(len(p))
    prev = float(pos0)
    for i, x in enumerate(p):
        if np.isfinite(x):
            prev = 1.0 if x > 0 else 0.0
        out[i] = prev
    return out


def _path_stats(w, days):
    years = max(days / 365.0, 1e-9)
    turns = float(np.abs(np.diff(np.concatenate([[0.0], w]))).sum()) if len(w) else 0.0
    return dict(exposure=float(w.mean()) if len(w) else float("nan"), round_trips_per_year=turns / 2.0 / years)


def _f(x):
    """기록용 float (무한대는 None — JSON 호환)"""
    x = float(x)
    return x if math.isfinite(x) else None


# ══════════ 모델 ══════════
class DPBandModel:
    """월별 모델: 릿지 예측 p → 히스테리시스 정책. weights(X)는 상태가 있음 (모듈 설명 참고)"""

    def __init__(self, fi, mu, sd, beta, c0, ar, band, pos0=0.0):
        self.fi = np.asarray(fi)
        self.mu, self.sd, self.beta, self.c0 = mu, sd, beta, float(c0)
        self.ar = ar                                            # (a, φ, s)
        self.grid, self.pi = band["grid"], band["pi"]
        self.q_in, self.q_out, self.mono = band["q_in"], band["q_out"], band["mono"]
        self.pos0 = float(pos0)
        self.last_pos = float(pos0)
        self.last = None                                         # 마지막 weights 호출의 (p, 포지션, 부호 규칙)

    def forecast(self, X):
        """p = μ0 + (ŷ − mean_학습(ŷ)) = c0 + Z·β  (행마다 독립, 상태 없음)"""
        Z = (np.asarray(X, np.float64)[:, self.fi] - self.mu) / self.sd
        return self.c0 + Z @ self.beta

    def _band(self, p, pos0):
        if self.mono:
            return band_path(p, self.q_in, self.q_out, pos0)
        return band_path(p, self.q_in, self.q_out, pos0, self.grid, self.pi)

    def weights(self, X):
        """한 달치 판단봉을 시간 순서대로 → 포지션 (B,) ∈ {0, 1}. 매번 pos0에서 시작(멱등), last_pos 갱신"""
        p = self.forecast(X)
        w = self._band(p, self.pos0)
        s = sign_path(p, self.pos0)
        self.last_pos = float(w[-1]) if len(w) else self.pos0
        self.last = (p, w, s)
        return w

    def sign_weights(self, X):
        """부호 규칙 포지션 (진단용, 후보 아님)"""
        return sign_path(self.forecast(X), self.pos0)

    def state(self):
        a, phi, s = self.ar
        return dict(fi=np.asarray(self.fi), feat_mu=np.asarray(self.mu), feat_sd=np.asarray(self.sd),
                    beta=np.asarray(self.beta), c0=np.array([self.c0]), ar=np.array([a, phi, s]),
                    grid=np.asarray(self.grid), pi=np.asarray(self.pi),
                    q=np.array([self.q_in, self.q_out]), pos=np.array([self.pos0, self.last_pos]))


def fit_month(cfg, datas, Tk, pos0=0.0):
    """T_k(초) 시점의 모델 하나를 처음부터 맞춤. 반환 (모델, 진단 dict)"""
    hp = hparams(cfg)
    fi = feat_index(list(cfg.get("feat") or TREND8))
    X, y, idx, ends = make_samples(datas, Tk, fi, hp["dp_horizon_days"])
    if len(y) < MIN_SAMPLES:
        raise RuntimeError(f"dp_band: 학습 표본이 {len(y)}개뿐입니다 (T_k={Tk})")
    mu, sd, beta, yb = fit_ridge(X, y, hp["dp_ridge_alpha"])
    yhat_in = yb + ((X - mu) / sd) @ beta
    mu0 = hp["dp_mu0_frac"] * float(y.mean())
    c0 = mu0 + yb - float(yhat_in.mean())                       # p = c0 + Z·β
    # AR(1): phase 0의 하루 p 계열 (종가가 T_k 이전), 연속된 날짜 쌍만
    d0 = datas[0]
    t = daily_rows(d0)
    t = t[(d0.ts[t] + BAR_SEC) <= Tk]
    Xd = d0.X[t][:, fi].astype(np.float64)
    fin = np.isfinite(Xd).all(axis=1)
    t, Xd = t[fin], Xd[fin]
    p = c0 + ((Xd - mu) / sd) @ beta
    cons = np.flatnonzero(np.diff(d0.ts[t]) == 86400)
    if len(cons) < 30:
        raise RuntimeError(f"dp_band: AR(1)에 쓸 연속 날짜 쌍이 {len(cons)}개뿐입니다")
    a, phi, s, clipped = fit_ar1(p[cons], p[cons + 1])
    band = solve_band(a, phi, s, hp)
    model = DPBandModel(fi, mu, sd, beta, c0, (a, phi, s), band, pos0)
    days = float(len(p))
    w_in = model._band(p, 0.0)
    s_in = sign_path(p, 0.0)
    q_in, q_out = band["q_in"], band["q_out"]
    info = dict(
        n_samples=int(len(y)), n_phases=len(datas), label_end_max=int(ends.max()),
        ridge_beta=[float(b) for b in beta], ybar=yb, mu0=mu0, c0=c0,
        n_ar=int(len(cons)), ar_a=a, ar_phi=phi, ar_s=s, phi_clipped=clipped,
        p_stat_mean=a / (1 - phi), p_stat_sd=s / math.sqrt(1 - phi * phi),
        vi_iters=int(band["iters"]), vi_resid=float(band["resid"]), vi_converged=bool(band["resid"] < hp["dp_tol"]),
        q_in=_f(q_in), q_out=_f(q_out), policy_monotone=band["mono"],
        band_center=_f(0.5 * (q_in + q_out)), band_half=_f(0.5 * (q_in - q_out)),
        p_last=float(p[-1]), p_last_sd=float((p[-1] - a / (1 - phi)) / (s / math.sqrt(1 - phi * phi))),
        insample_dp=_path_stats(w_in, days), insample_sign=_path_stats(s_in, days),
    )
    return model, info


def _prev_oos(ens):
    """지난달 모델이 실제로 판단한 한 달(표본 밖) 요약 — 부호 규칙과 비교 (진단용)"""
    if ens is None or getattr(ens, "last", None) is None:
        return None
    p, w, s = ens.last
    tw = float(np.abs(np.diff(np.concatenate([[ens.pos0], w]))).sum())
    ts_ = float(np.abs(np.diff(np.concatenate([[ens.pos0], s]))).sum())
    return dict(n=int(len(p)), p_mean=float(np.nanmean(p)) if len(p) else None,
                dp_exposure=float(w.mean()) if len(w) else None, sign_exposure=float(s.mean()) if len(s) else None,
                dp_turns=tw, sign_turns=ts_, agree=float((w == s).mean()) if len(w) else None)


def monthly_update(cfg, datas, T_k, seed, ens, anchor):
    """
    walk.monthly_update 훅. 매달 처음부터 다시 맞추고(결정론적, seed 무시) 항상 채택합니다.
    kind는 다른 기법과 일정 표기를 맞추려고 1월·첫 달은 'cold', 나머지는 'finetune'이라 적지만
    절차는 같습니다(refit='full'). anchor는 쓰지 않고 그대로 돌려줍니다.
    시작 포지션은 지난달 모델의 마지막 포지션(ens.last_pos)을 이어받습니다.
    """
    Tk = int(T_k.timestamp())
    pos0 = float(getattr(ens, "last_pos", 0.0)) if ens is not None else 0.0
    model, info = fit_month(cfg, datas[:cfg.get("phases", len(datas))], Tk, pos0)
    entry = dict(month=str(T_k.date()), kind="cold" if (T_k.month == 1 or ens is None) else "finetune",
                 refit="full", accepted=True, pos0=pos0, prev_oos=_prev_oos(ens), **info)
    return model, anchor, entry
