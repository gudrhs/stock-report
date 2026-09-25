# -*- coding: utf-8 -*-
"""
W2 'dp_band' 합성 대조 실험 — 빠른 확인용 (전체 워크포워드 아님). 시험 수 N에 넣지 않습니다.
실패하면 구현 버그로 봅니다 (btc/research/rl_effective_methods.md §3 W2).

(1) 양성 대조 — 알려진 AR(1) 신호
    참 p: p_{t+1} = a + φ·p_t + β·ξ   (a = m·ε, φ = 1 − ε),  하루 수익 r_t = p_t + σ_ε·z_t
    경로마다 앞 N_TRAIN일의 p로 AR(1)을 적합 → 201칸 가치 반복 → 뒤 N_TEST일에 히스테리시스 적용.
      · 실현 순성장(비용 차감, 잡음 포함) ≥ 해석적 최적의 95%
        해석적 최적 g* = 참 계수로 푼 정책(801칸, ±6 표준편차)의 격자 마르코프 사슬 정상분포에서의
        하루 평균 기대 순성장 Σ π(i,prev)·[a·p_i − κ|a − prev|]
      · 적합한 문턱 반폭 (q_in − q_out)/2 가 de Lataillade·Deremble·Potters·Bouchaud(2012) 점근식
        q* = (1.5·Γ·β²)^{1/3} 과 20% 안 (적용 가능한 설정에서만)
    기호 대응 (그 논문 식 (8)과 목표):
      그들: p_{t+1} − p_t = −ε·p_t + β·ξ_t,  이득 Σ [π_t·p_t − Γ·|π_t − π_{t−1}|],  |π| ≤ M (최적은 π = ±M)
      우리: a ∈ {0, 1} = (1 + π)/2 (M = 1). a·p − κ|Δa| = p/2 + ½·[π·p − κ·|Δπ|]
            → p/2는 정책과 무관한 상수이므로 같은 문제이고 Γ = κ = −ln(1 − c), β = s (AR(1) 잔차 표준편차),
              ε = 1 − φ. 그들의 '+q*에서 롱 전환 / −q*에서 숏 전환'이 우리의 q_in = +q*, q_out = −q*.
              (평균 m = 0인 신호에서; 평균이 있으면 띠 중심이 움직이므로 반폭만 비교)
      점근식이 맞는 영역 (논문 §4.4): η = Γ·ε^{3/2}/β ≪ 1 이고 연속 극한 q* ≫ β.
      논문 §5는 q*/β ≈ 1.15·(Γ/β)^{1/3} 이 크지 않으면 이산 효과로 어긋난다고 밝힙니다. 이산 관측에서
      장벽이 약 0.5826·β 안쪽으로 당겨지는 효과(Broadie·Glasserman·Kou 1997 연속성 보정)를 참고로
      q* − 0.5826·β 도 함께 보고합니다.
      적용 가능 판정: m = 0, η < 0.1, q*/β ≥ 5 (아래 APPLICABLE; 설명은 반환 JSON 'rule'에).
(2) 음성 대조 — 예측력 없는 경로 (p = 일정한 드리프트 + 순수 잡음 지표)
    GARCH(1,1) 4시간봉 + 일정한 드리프트(하루 0.1% 로그). 지표는
      'price' : 그 가격 경로로 실제 TREND8 계산 (느리고 끈질긴 잡음 — 현실적인 경우)
      'iid'   : TREND8 자리에 독립 N(0,1) 잡음
    실제 훅(walk.run_replication, phase 1개)으로 2017-01 ~ 2025-01 월별 재학습.
    기준: 항상 보유(매수·보유) 대비 추가 왕복 거래 ≤ 연 1회.

  OMP_NUM_THREADS=1 python -m btc.research.dp_band_controls --part pos
  OMP_NUM_THREADS=1 python -m btc.research.dp_band_controls --part neg --seeds 2
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import json
import math
import time

import numpy as np
import pandas as pd

from .algos import dp_band as M

COST = 0.0015
BGK = 0.5826                               # −ζ(1/2)/√(2π)
# 사전 등록 설정 (실행 전 고정). A는 BTC 규모(예측 표준편차 ≈ 0.1%/일, 상관시간 50일)
SETTINGS = {
    "A_btc_like": dict(eps=0.02, beta=2e-4, m=0.0),
    "B_mid": dict(eps=0.005, beta=5e-5, m=0.0),
    "C_slow": dict(eps=0.002, beta=1e-5, m=0.0),
    "D_btc_drift": dict(eps=0.02, beta=2e-4, m=5e-4),
}
NOISE_MULT = 5.0                           # 수익 잡음 σ_ε = 5 × 예측의 정상 표준편차
N_PATHS, N_TRAIN, N_TEST = 20, 3650, 50000
APPLICABLE = dict(eta_max=0.1, q_over_beta_min=5.0)


# ══════════ 논문 식 ══════════
def dawson(x, n=4001):
    """D(x) = e^{−x²} ∫_0^x e^{v²} dv (사다리꼴 적분; x ≤ 5 정도면 충분히 정확)"""
    if x == 0:
        return 0.0
    v = np.linspace(0.0, x, n)
    f = np.exp(v * v - x * x)
    return float(np.sum((f[1:] + f[:-1]) * 0.5) * (v[1] - v[0]))


def q_eq12(eps, beta, gamma_cost):
    """논문 식 (12): q* = β/√ε · F^{-1}(Γ ε^{3/2} / β),  F(x) = x − D(x) (단조 증가 → 이분법)"""
    eta = gamma_cost * eps ** 1.5 / beta
    lo, hi = 0.0, max(10.0, 2 * eta)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid - dawson(mid) < eta:
            lo = mid
        else:
            hi = mid
    return beta / math.sqrt(eps) * 0.5 * (lo + hi)


def q_asym(beta, gamma_cost):
    """논문 식 (13): q* = (1.5 Γ β²)^{1/3}"""
    return (1.5 * gamma_cost * beta * beta) ** (1.0 / 3.0)


# ══════════ 해석적 최적 (참 계수, 고운 격자 사슬) ══════════
def analytic_gain(a, phi, s, hp, n=801, width=6.0):
    g, P = M.ar1_grid(a, phi, s, n, width)
    kappa = -math.log(1.0 - hp["dp_cost"])
    _, pi, _, _ = M.value_iteration(g, P, kappa, hp["dp_gamma"], hp["dp_tol"])
    G = len(g)
    # 결합 상태 (i, prev) → (j, a=pi[i,prev]) 전이
    T = np.zeros((2 * G, 2 * G))
    rew = np.zeros(2 * G)
    for prev in (0, 1):
        act = pi[:, prev].astype(int)
        rows = np.arange(G) * 2 + prev
        for av in (0, 1):
            sel = act == av
            T[np.ix_(rows[sel], np.arange(G) * 2 + av)] = P[sel]
        rew[rows] = act * g - kappa * np.abs(act - prev)
    A = T.T - np.eye(2 * G)
    A[-1] = 1.0
    b = np.zeros(2 * G)
    b[-1] = 1.0
    st = np.linalg.solve(A, b)
    q_in, q_out, _ = M.thresholds(g, pi)
    return float(st @ rew), q_in, q_out


def _band_paths(P, qin, qout):
    """경로별 문턱으로 히스테리시스 (시간 루프, 경로는 벡터). P: (경로, 날짜)"""
    n, T = P.shape
    W = np.empty((n, T))
    prev = np.zeros(n)
    for t in range(T):
        x = P[:, t]
        prev = np.where((prev == 0) & (x > qin), 1.0, np.where((prev == 1) & (x < qout), 0.0, prev))
        W[:, t] = prev
    return W


def _net(W, R):
    lnc = math.log(1.0 - COST)
    dW = np.abs(np.diff(np.concatenate([np.zeros((W.shape[0], 1)), W], axis=1), axis=1))
    return W * R + lnc * dW


def positive(settings=SETTINGS, n_paths=N_PATHS, n_train=N_TRAIN, n_test=N_TEST, seed=0):
    hp = M.hparams({})
    kappa = -math.log(1.0 - COST)
    out = {}
    for name, st in settings.items():
        t0 = time.time()
        eps, beta, m = st["eps"], st["beta"], st["m"]
        a_true, phi_true = m * eps, 1.0 - eps
        sd_p = beta / math.sqrt(1 - phi_true ** 2)
        sig_e = NOISE_MULT * sd_p
        g_star, qi_star, qo_star = analytic_gain(a_true, phi_true, beta, hp)
        b_true = M.solve_band(a_true, phi_true, beta, hp)
        rng = np.random.default_rng([seed, int(eps * 1e6), int(beta * 1e9), int(m * 1e7)])
        T = n_train + n_test
        Pth = np.empty((n_paths, T))
        Pth[:, 0] = m + sd_p * rng.standard_normal(n_paths)
        xi = rng.standard_normal((n_paths, T))
        for t in range(1, T):
            Pth[:, t] = a_true + phi_true * Pth[:, t - 1] + beta * xi[:, t]
        R = Pth + sig_e * rng.standard_normal((n_paths, T))
        qin, qout, fits = np.empty(n_paths), np.empty(n_paths), []
        for k in range(n_paths):
            x = Pth[k, :n_train]
            a, phi, s, clipped = M.fit_ar1(x[:-1], x[1:])
            b = M.solve_band(a, phi, s, hp)
            qin[k], qout[k] = b["q_in"], b["q_out"]
            fits.append(dict(phi=phi, s=s, a=a, clipped=clipped, mono=b["mono"]))
        Pt, Rt = Pth[:, n_train:], R[:, n_train:]
        W = _band_paths(Pt, qin, qout)
        Wo = _band_paths(Pt, np.full(n_paths, b_true["q_in"]), np.full(n_paths, b_true["q_out"]))
        net, net_o = _net(W, Rt), _net(Wo, Rt)
        exp_net = _net(W, Pt)                                              # 잡음 없는 기대 순성장
        per_path = net.mean(axis=1)
        realized = float(net.mean())
        se = float(per_path.std(ddof=1) / math.sqrt(n_paths))
        half = 0.5 * (qin - qout)
        qa, q12 = q_asym(beta, kappa), q_eq12(eps, beta, kappa)
        eta = kappa * eps ** 1.5 / beta
        applicable = bool(m == 0.0 and eta < APPLICABLE["eta_max"] and qa / beta >= APPLICABLE["q_over_beta_min"])
        hw = float(half.mean())
        res = dict(
            params=dict(eps=eps, beta=beta, m=m, sd_p=sd_p, sigma_noise=sig_e, cost=COST, kappa=kappa),
            g_star=g_star, realized=realized, realized_se=se, ratio=realized / g_star,
            ratio_expected_noise_free=float(exp_net.mean()) / g_star,
            oracle_realized=float(net_o.mean()), oracle_ratio=float(net_o.mean()) / g_star,
            always_long=float(Rt.mean()) - 0.0,
            exposure=float(W.mean()), round_trips_per_year=float(np.abs(np.diff(W, axis=1)).sum(axis=1).mean() / 2 / (n_test / 365)),
            half_fit_mean=hw, half_fit_sd=float(half.std()), center_fit_mean=float((0.5 * (qin + qout)).mean()),
            half_true_dp=0.5 * (b_true["q_in"] - b_true["q_out"]), half_fine_dp=0.5 * (qi_star - qo_star),
            q_asym=qa, q_eq12=q12, q_asym_bgk=qa - BGK * beta, eta=eta, q_over_beta=qa / beta,
            ratio_half_asym=hw / qa, ratio_half_eq12=hw / q12, applicable=applicable,
            phi_fit_mean=float(np.mean([f["phi"] for f in fits])), s_fit_mean=float(np.mean([f["s"] for f in fits])),
            any_clipped=any(f["clipped"] for f in fits), all_monotone=all(f["mono"] for f in fits),
        )
        res["pass_growth"] = bool(res["ratio"] >= 0.95)
        res["pass_band"] = bool(abs(res["ratio_half_asym"] - 1) <= 0.20) if applicable else None
        res["secs"] = round(time.time() - t0, 1)
        out[name] = res
        print(f"  {name:12s} g*={g_star:.3e}/일 실현 {realized:.3e}±{se:.1e} (비율 {res['ratio']:.3f}, 잡음 없이 "
              f"{res['ratio_expected_noise_free']:.3f}, 참계수 {res['oracle_ratio']:.3f}) | 반폭 {hw:.3e} "
              f"점근 {qa:.3e}(비 {res['ratio_half_asym']:.2f}) 식12 {q12:.3e}(비 {res['ratio_half_eq12']:.2f}) "
              f"보정 {qa - BGK * beta:.3e} η={eta:.3f} q/β={qa / beta:.2f} 적용 {applicable} "
              f"| 왕복 {res['round_trips_per_year']:.2f}/년 노출 {res['exposure']:.2f} ({res['secs']}s)", flush=True)
    return out


# ══════════ (2) 음성 대조 ══════════
def synth_bars(seed, drift_day=0.001):
    """합성 4시간봉 2012-01 ~ 2025-02: GARCH(1,1)(tests/test_btc_controls.py와 같은 계수) + 일정한 드리프트"""
    from ..env import BAR_SEC
    t0 = int(pd.Timestamp("2012-01-01", tz="UTC").timestamp())
    t1 = int(pd.Timestamp("2025-02-01", tz="UTC").timestamp())
    n = (t1 - t0) // BAR_SEC
    rng = np.random.default_rng(seed)
    w, a, b = 1.5e-6, 0.08, 0.90
    h = w / (1 - a - b)
    r = np.empty(n)
    z = rng.standard_normal(n)
    for i in range(n):
        e = math.sqrt(h) * z[i]
        r[i] = drift_day / 6.0 + e
        h = w + a * e * e + b * h
    o = 1000 * np.exp(np.concatenate([[0], np.cumsum(r)[:-1]]))
    c = o * np.exp(rng.normal(0, 0.002, n))
    hi = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.003, n)))
    lo = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.003, n)))
    v = rng.lognormal(3, 0.4, n)
    ts = t0 + BAR_SEC * np.arange(n)
    return pd.DataFrame(dict(ts=ts, open=o, high=hi, low=lo, close=c, volume=v, gap_before=0, forced_hold=False))


def negative_one(kind, seed, cfg=None):
    from .. import features as Fe
    from ..env import PhaseData
    from .rl import feat_index
    from .variants import TREND8
    from .walk import run_replication, decision_mask
    from ..walkforward import decision_range
    t0 = time.time()
    cfg = dict(cfg or proposed_cfg(), phases=1)
    df = synth_bars(1000 + seed)
    X, sig = Fe.compute(df)
    if kind == "iid":
        rng = np.random.default_rng(2000 + seed)
        fi = feat_index(TREND8)
        X[:, fi] = rng.standard_normal((len(X), len(fi)))
    d = PhaseData(df, X, sig)
    end = pd.Timestamp("2025-01-01", tz="UTC")
    res = run_replication(cfg, 0, datas=[d], end=end)
    lo = int(pd.Timestamp("2017-01-01", tz="UTC").timestamp())
    a, b = decision_range(d, lo, int(end.timestamp()))
    sel = np.arange(a, b)[decision_mask(d, 6)[a:b]]
    w = res["U"][0.0][sel, 0].astype(float)
    years = len(sel) / 365.0
    turns = float(np.abs(np.diff(np.concatenate([[0.0], w]))).sum())
    rt = turns / 2 / years
    rt_bh = 1.0 / 2 / years                                  # 항상 보유: 첫 진입 한 번
    log = res["log"]
    sign_turns = sum((e.get("prev_oos") or {}).get("sign_turns", 0.0) for e in log)
    out = dict(kind=kind, seed=seed, years=years, exposure=float(np.nanmean(w)), round_trips_per_year=rt,
               extra_round_trips_per_year=rt - rt_bh, sign_round_trips_per_year_lagged=sign_turns / 2 / years,
               mu0_mean=float(np.mean([e["mu0"] for e in log])),
               p_sd_mean=float(np.mean([e["p_stat_sd"] for e in log])),
               phi_mean=float(np.mean([e["ar_phi"] for e in log])),
               q_out_median=float(np.median([e["q_out"] if e["q_out"] is not None else -np.inf for e in log])),
               q_in_median=float(np.median([e["q_in"] if e["q_in"] is not None else np.inf for e in log])),
               months=len(log), secs=round(time.time() - t0, 1))
    out["pass"] = bool(out["extra_round_trips_per_year"] <= 1.0)
    print(f"  음성 {kind:5s} s{seed}: 왕복 {rt:.2f}/년 (추가 {out['extra_round_trips_per_year']:+.2f}) 노출 "
          f"{out['exposure']:.2f} μ0 {out['mu0_mean']:.2e} p_sd {out['p_sd_mean']:.2e} φ {out['phi_mean']:.4f} "
          f"q_out~{out['q_out_median']:.2e} | 부호 규칙 왕복 {out['sign_round_trips_per_year_lagged']:.2f}/년 "
          f"({out['secs']}s)", flush=True)
    return out


def proposed_cfg():
    from .variants import v, TREND8
    return v("W2_dp_band", algo="dp_band", output="weights", stride=6, feat=TREND8, **M.DEFAULTS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("pos", "neg", "all"), default="all")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--kinds", default="price,iid")
    ap.add_argument("--out", default=None, help="결과 JSON 경로 (없으면 출력만)")
    a = ap.parse_args()
    res = dict(rule=dict(
        growth="실현 순성장(잡음 포함, 전 경로·전 표본 밖 날짜 평균) ≥ 0.95 × g*",
        band="|반폭/점근식 − 1| ≤ 0.20, m = 0 · η < 0.1 · q*/β ≥ 5 인 설정에서만 (연속 극한 조건, 논문 §4.4·§5)",
        negative="매수·보유 대비 추가 왕복 ≤ 연 1회"))
    if a.part in ("pos", "all"):
        res["positive"] = positive()
    if a.part in ("neg", "all"):
        res["negative"] = [negative_one(k, s) for k in a.kinds.split(",") for s in range(a.seeds)]
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
