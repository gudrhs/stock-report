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
      ※ 이 판정 규칙은 참 계수 DP 문턱(A·B·C)을 본 '뒤에' 정한 것입니다 (사후 규칙 — 결과 JSON에도 표시).
        4개 설정 중 1개(C)만 해당하므로 띠 검증의 근거로 삼지 않습니다. 띠가 맞다는 근거는 아래 독립 검증입니다.
    독립 검증 (검토 후 추가, 사전 등록 기준이 아님 — g*가 같은 ar1_grid·value_iteration 코드를 쓰므로)
      · 정확한 정책 반복(Howard): 2G개 결합 상태의 선형방정식을 직접 풀어 평가·개선 — value_iteration과
        코드를 공유하지 않음. 참 계수 201칸에서 정책이 한 칸도 다르지 않아야 함 (pi_mismatch = 0).
      · 몬테카를로 문턱 탐색: 격자 없이 연속 AR(1)을 직접 시뮬레이션해 (q_in, q_out) 후보(정상 평균 ± 1.5 표준편차,
        0.1 표준편차 간격, q_out ≤ q_in) 전부를 같은 난수로 평가 → 최선 후보의 하루 기대 순성장 g_mc.
        참 계수 DP 띠의 같은 경로 성장 / g_mc ≥ 0.99 이어야 함 (선택 편향 때문에 1보다 약간 작을 수 있음).
        대칭 띠 중 최선의 반폭(half_mc_sym)도 보고 — A(BTC 규모)에서 DP 반폭이 점근식보다 작은 이유의 근거.
(2) 음성 대조 — 예측력 없는 경로 (p = 일정한 드리프트 + 순수 잡음 지표)
    GARCH(1,1) 4시간봉 + 일정한 드리프트(하루 0.1% 로그). 지표는
      'price' : 그 가격 경로로 실제 TREND8 계산 (느리고 끈질긴 잡음 — 현실적인 경우)
      'iid'   : TREND8 자리에 독립 N(0,1) 잡음
    실제 훅(walk.run_replication, phase 1개)으로 2017-01 ~ 2025-01 월별 재학습.
    기준: 항상 보유(매수·보유) 대비 추가 왕복 거래 ≤ 연 1회.
    사전 등록 문구('순수 잡음 지표')에 해당하는 판정은 'iid'입니다. 'price'는 사양상 실패할 수 있고(검토 재현: 시드 0에서
    연 +3.6회) 코드 버그가 아니라 릿지(α=1, 사실상 OLS)가 끈질긴 가격 지표의 잡음을 추세로 맞춘 결과입니다.
    그래서 'price'는 판정이 아니라 '가짜 거래율·가짜 이득의 기준선'으로 씁니다: 같은 Window 평가(다음 시가 체결,
    편도 0.15%)로 DP·부호 규칙·매수보유의 Sharpe·연 로그성장·왕복을 보고하고, 실제 BTC의 W2 결과는 이 기준선
    (여러 시드, --drift로 BTC 드리프트에 맞춤)과 나란히 읽어야 합니다.
(3) 부호 규칙 비교 (--part sign) — 저장된 W2_dp_band 실행(rep 0)의 월별 기록만으로 p를 복원(dp_band.paths_from_log)해
    DP와 부호 규칙을 screen과 같은 Window에서 평가. 사전 등록 예측: Sharpe(DP) ≥ Sharpe(부호) + 0.05, 회전(DP) ≤ ½·회전(부호).
    복원한 DP 포지션이 저장된 U와 같은지(dp_mismatch = 0)도 확인합니다. 잠금 구간을 넘으면 screen처럼 BTC_LOCKBOX_OPEN=1 필요.
참고: 이 기법은 결정론적이고 walk.run_replication은 rep마다 seed만 바꾸므로 rep 5개가 비트 단위로 같습니다 → --reps 1.

  OMP_NUM_THREADS=1 python -m btc.research.dp_band_controls --part pos
  OMP_NUM_THREADS=1 python -m btc.research.dp_band_controls --part neg --seeds 2 [--drift 0.001]
  OMP_NUM_THREADS=1 python -m btc.research.dp_band_controls --part sign [--name W2_dp_band --rep 0]
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


# ══════════ 해석적 최적 (참 계수, 고운 격자 사슬) — 사전 등록 기준 ══════════
def analytic_gain(a, phi, s, hp, n=801, width=6.0):
    """g* = 참 계수로 푼 정책(801칸)의 격자 사슬 정상분포에서의 하루 기대 순성장 (같은 DP 코드를 씀 — 자기참조)"""
    g, P = M.ar1_grid(a, phi, s, n, width)
    kappa = -math.log(1.0 - hp["dp_cost"])
    _, pi, _, _ = M.value_iteration(g, P, kappa, hp["dp_gamma"], hp["dp_tol"])
    ch = M.policy_chain(g, P, pi, kappa)
    q_in, q_out, _ = M.thresholds(g, pi)
    return ch["gain"], q_in, q_out


# ══════════ 독립 검증 1: 정확한 정책 반복 (value_iteration과 코드 공유 없음) ══════════
def policy_iteration(grid, P, kappa, gamma, max_iter=200):
    """
    Howard 정책 반복. 결합 상태 k = 2·i + prev. 정책 평가는 (I − γ·T_π) V = r_π 를 직접 풀고,
    개선은 Q(i, prev, a) = a·p_i − κ|a − prev| + γ·Σ_j P[i,j]·V(j, a) 의 argmax (동점이면 현재 행동 유지).
    반환 (V (G,2), pi (G,2), 반복 수).
    """
    G = len(grid)
    grid = np.asarray(grid, float)
    pol = np.tile(np.array([0, 1], np.int8), (G, 1))          # 시작: 이전 포지션 유지
    for it in range(1, max_iter + 1):
        T = np.zeros((2 * G, 2 * G))
        r = np.zeros(2 * G)
        for prev in (0, 1):
            for i in range(G):
                act = int(pol[i, prev])
                k = 2 * i + prev
                r[k] = act * grid[i] - kappa * abs(act - prev)
                T[k, act::2] = P[i]
        V = np.linalg.solve(np.eye(2 * G) - gamma * T, r).reshape(G, 2)
        cont = gamma * (P @ V)                                  # cont[i, a]
        new = pol.copy()
        for prev in (0, 1):
            q0 = 0.0 * grid - kappa * abs(0 - prev) + cont[:, 0]
            q1 = 1.0 * grid - kappa * abs(1 - prev) + cont[:, 1]
            cur = np.where(pol[:, prev] == 1, q1, q0)
            best = np.where(q1 > q0, 1, 0)
            bestq = np.maximum(q0, q1)
            new[:, prev] = np.where(bestq > cur + 1e-15 * (1 + np.abs(cur)), best, pol[:, prev])
        if np.array_equal(new, pol):
            return V, pol, it
        pol = new
    return V, pol, max_iter


# ══════════ 독립 검증 2: 연속 AR(1) 몬테카를로 문턱 탐색 (격자 없음) ══════════
def mc_band_search(a, phi, s, kappa, extra=(), n_paths=100, n_days=20000, K=31, width=1.5, seed=0):
    """
    연속 AR(1) p 경로(정상분포에서 시작)를 만들고, (q_in, q_out) 후보마다 히스테리시스를 돌려
    하루 기대 순성장 mean[a_t·p_t − κ|a_t − a_{t−1}|] 를 같은 난수로 비교 (수익 잡음은 기댓값에 영향이 없어 뺌).
    후보 = 정상 평균 + 정상 표준편차 × linspace(−width, width, K) 의 q_out ≤ q_in 쌍 + extra.
    반환 dict(best=(q_in, q_out, g), extra=[g...], sym=(반폭, g), gains=배열).
    """
    m = a / (1.0 - phi)
    sd = s / math.sqrt(1.0 - phi * phi)
    rng = np.random.default_rng([seed, 7919])
    x = m + sd * rng.standard_normal(n_paths)
    lv = m + sd * np.linspace(-width, width, K)
    pairs = [(lv[i], lv[j]) for i in range(K) for j in range(K) if j <= i]
    sym_idx = [pairs.index((lv[i], lv[K - 1 - i])) for i in range(K) if K - 1 - i <= i]
    pairs += list(extra)
    QI = np.array([q[0] for q in pairs])[:, None]
    QO = np.array([q[1] for q in pairs])[:, None]
    prev = np.zeros((len(pairs), n_paths))
    gain = np.zeros(len(pairs))
    for _ in range(n_days):
        new = np.where(prev == 0.0, (x > QI), ~(x < QO)).astype(float)
        gain += new @ x - kappa * np.abs(new - prev).sum(axis=1)
        prev = new
        x = a + phi * x + s * rng.standard_normal(n_paths)
    gain /= n_paths * n_days
    nb = len(pairs) - len(extra)
    k = int(np.argmax(gain[:nb]))
    ks = sym_idx[int(np.argmax(gain[sym_idx]))]
    return dict(best=(float(QI[k, 0]), float(QO[k, 0]), float(gain[k])), extra=[float(g) for g in gain[nb:]],
                sym=(float(0.5 * (QI[ks, 0] - QO[ks, 0])), float(gain[ks])), gains=gain)


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
        # 독립 검증 (사전 등록 기준 아님)
        V_pi, pi_pi, pi_it = policy_iteration(b_true["grid"], b_true["P"], kappa, hp["dp_gamma"])
        pi_mismatch = int((pi_pi != b_true["pi"]).sum())
        mc = mc_band_search(a_true, phi_true, beta, kappa, extra=[(b_true["q_in"], b_true["q_out"])], seed=seed)
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
            indep=dict(pi_mismatch=pi_mismatch, pi_iters=pi_it, V_maxdiff=float(np.abs(V_pi - b_true["V"]).max()),
                       g_mc_best=mc["best"][2], q_in_mc=mc["best"][0], q_out_mc=mc["best"][1],
                       half_mc=0.5 * (mc["best"][0] - mc["best"][1]), half_mc_sym=mc["sym"][0], g_mc_sym=mc["sym"][1],
                       g_mc_dp_true=mc["extra"][0], ratio_dp_true_vs_mc=mc["extra"][0] / mc["best"][2],
                       ratio_realized_vs_mc=realized / mc["best"][2]),
        )
        res["pass_growth"] = bool(res["ratio"] >= 0.95)
        res["pass_band"] = bool(abs(res["ratio_half_asym"] - 1) <= 0.20) if applicable else None
        res["pass_indep"] = bool(pi_mismatch == 0 and res["indep"]["ratio_dp_true_vs_mc"] >= 0.99)
        res["secs"] = round(time.time() - t0, 1)
        out[name] = res
        print(f"  {name:12s} g*={g_star:.3e}/일 실현 {realized:.3e}±{se:.1e} (비율 {res['ratio']:.3f}, 잡음 없이 "
              f"{res['ratio_expected_noise_free']:.3f}, 참계수 {res['oracle_ratio']:.3f}) | 반폭 {hw:.3e} "
              f"점근 {qa:.3e}(비 {res['ratio_half_asym']:.2f}) 식12 {q12:.3e}(비 {res['ratio_half_eq12']:.2f}) "
              f"보정 {qa - BGK * beta:.3e} η={eta:.3f} q/β={qa / beta:.2f} 적용 {applicable} "
              f"| 왕복 {res['round_trips_per_year']:.2f}/년 노출 {res['exposure']:.2f} ({res['secs']}s)", flush=True)
        ind = res["indep"]
        print(f"  {'':12s} 독립: 정책반복 불일치 {pi_mismatch}칸(V차 {ind['V_maxdiff']:.1e}) | MC 최선 "
              f"[{ind['q_out_mc']:.2e}, {ind['q_in_mc']:.2e}] g={ind['g_mc_best']:.3e}, 참DP띠/MC최선 "
              f"{ind['ratio_dp_true_vs_mc']:.4f}, 대칭 최선 반폭 {ind['half_mc_sym']:.2e} vs 참DP "
              f"{res['half_true_dp']:.2e} vs 점근 {qa:.2e} → {'통과' if res['pass_indep'] else '실패'}", flush=True)
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


def compare_paths(W, paths, cost=COST):
    """
    Window 위에서 DP·부호 규칙·매수보유를 screen과 같은 방식(판단봉 목표를 25% 단위로 반올림, 다음 시가 체결,
    편도 cost)으로 평가. paths = dp_band.paths_from_log 결과. 반환 dict(dp, sign, bh, 사전 등록 예측 판정).
    """
    from .. import stats as S
    d = W.datas[0]
    a, b = W.rng[0]
    out = {}
    tgs = {}
    for k in ("dp", "sign"):
        tg = np.full(d.T, np.nan)
        tg[a:b] = np.round(paths[k][a:b] * 4) / 4
        tgs[k] = tg
    bh = np.full(d.T, np.nan)
    bh[a:b] = 1.0
    tgs["bh"] = bh
    for k, tg in tgs.items():
        res = W.run(tg, cost)
        years = len(res["r"]) / 365.0
        pos = np.nan_to_num(res["pos"])
        turns = float(np.abs(np.diff(np.concatenate([[0.0], pos]))).sum())
        sm = S.summary(res["r"])
        out[k] = dict(sharpe=sm["sharpe"], cagr=sm["cagr"], max_dd=sm["max_dd"],
                      log_growth_per_year=float(np.log1p(res["r"]).sum() / years),
                      exposure=float(pos.mean()), round_trips_per_year=turns / 2.0 / years)
    out["sharpe_dp_minus_bh"] = out["dp"]["sharpe"] - out["bh"]["sharpe"]
    out["sharpe_sign_minus_bh"] = out["sign"]["sharpe"] - out["bh"]["sharpe"]
    out["sharpe_dp_minus_sign"] = out["dp"]["sharpe"] - out["sign"]["sharpe"]
    out["prereg_sharpe_ge_sign_plus_0p05"] = bool(out["sharpe_dp_minus_sign"] >= 0.05)
    out["prereg_turnover_le_half_sign"] = bool(out["dp"]["round_trips_per_year"] <= 0.5 * out["sign"]["round_trips_per_year"])
    return out


def negative_one(kind, seed, cfg=None, drift_day=0.001):
    from .. import features as Fe
    from ..env import PhaseData
    from ..evaluate import Window
    from .rl import feat_index
    from .variants import TREND8
    from .walk import run_replication, decision_mask
    from ..walkforward import decision_range
    t0 = time.time()
    cfg = dict(cfg or proposed_cfg(), phases=1)
    df = synth_bars(1000 + seed, drift_day)
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
    # 기록만으로 p·부호 규칙 복원 (DP는 저장된 U와 같아야 함)
    paths = M.paths_from_log(log, d, 6, end=end)
    dp_mismatch = int(np.sum(paths["dp"][sel] != w))
    sg = paths["sign"][sel]
    rt_sign = float(np.abs(np.diff(np.concatenate([[0.0], sg]))).sum()) / 2 / years
    W = Window([d], "2017-01-01", "2025-01-01", f"dp_band negative control {kind} s{seed}")
    cmp_ = compare_paths(W, paths)
    regimes = {}
    for e in log:
        regimes[e["policy_regime"]] = regimes.get(e["policy_regime"], 0) + 1
    out = dict(kind=kind, seed=seed, drift_day=drift_day, years=years, exposure=float(np.nanmean(w)),
               round_trips_per_year=rt, extra_round_trips_per_year=rt - rt_bh,
               sign_round_trips_per_year=rt_sign, dp_mismatch=dp_mismatch,
               mu0_mean=float(np.mean([e["mu0"] for e in log])),
               p_sd_mean=float(np.mean([e["p_stat_sd"] for e in log])),
               phi_mean=float(np.mean([e["ar_phi"] for e in log])),
               q_out_median=float(np.median([M.parse_q(e["q_out"]) for e in log])),
               q_in_median=float(np.median([M.parse_q(e["q_in"]) for e in log])),
               regimes=regimes, window=cmp_, months=len(log), secs=round(time.time() - t0, 1))
    out["pass"] = bool(out["extra_round_trips_per_year"] <= 1.0)
    out["prereg_judged"] = kind == "iid"                     # 사전 등록 문구('순수 잡음 지표')의 판정은 iid
    print(f"  음성 {kind:5s} s{seed}: 왕복 {rt:.2f}/년 (추가 {out['extra_round_trips_per_year']:+.2f}) 노출 "
          f"{out['exposure']:.2f} μ0 {out['mu0_mean']:.2e} p_sd {out['p_sd_mean']:.2e} φ {out['phi_mean']:.4f} "
          f"q_out~{out['q_out_median']:.2e} | 부호 규칙 왕복 {rt_sign:.2f}/년 | 가짜 Sharpe 이득 DP−BH "
          f"{cmp_['sharpe_dp_minus_bh']:+.3f}, 부호−BH {cmp_['sharpe_sign_minus_bh']:+.3f} | 복원 불일치 {dp_mismatch} "
          f"| {regimes} ({out['secs']}s)", flush=True)
    return out


def sign_report(name="W2_dp_band", r=0, lo="2017-01-01", hi="2026-09-25"):
    """(3) 저장된 실행의 기록만으로 부호 규칙을 복원해 DP와 비교 (screen과 같은 Window·비용)"""
    from ..evaluate import Window
    from ..walkforward import load_phases
    from .walk import load_run, decision_mask
    from .variants import VARIANTS
    cfg = VARIANTS.get(name, {})
    run = load_run(name, r)
    datas = load_phases()
    d = datas[0]
    paths = M.paths_from_log(run["log"], d, cfg.get("stride", 6))
    sel = np.flatnonzero(decision_mask(d, cfg.get("stride", 6)) & np.isfinite(paths["dp"]))
    U = run["U"][0.0][sel, 0].astype(float)
    mism = int(np.sum(paths["dp"][sel] != U))
    W = Window(datas, lo, hi, f"dp_band sign-rule report {name}")
    out = dict(name=name, rep=r, dp_mismatch=mism, **compare_paths(W, paths))
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return out


def proposed_cfg():
    from .variants import v, TREND8
    return v("W2_dp_band", algo="dp_band", output="weights", stride=6, feat=TREND8, **M.DEFAULTS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("pos", "neg", "all", "sign"), default="all")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--kinds", default="price,iid")
    ap.add_argument("--drift", type=float, default=0.001, help="음성 대조의 하루 로그 드리프트 (BTC에 맞춰 바꿔 보기)")
    ap.add_argument("--name", default="W2_dp_band")
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--out", default=None, help="결과 JSON 경로 (없으면 출력만)")
    a = ap.parse_args()
    res = dict(rule=dict(
        growth="실현 순성장(잡음 포함, 전 경로·전 표본 밖 날짜 평균) ≥ 0.95 × g*",
        band="|반폭/점근식 − 1| ≤ 0.20, m = 0 · η < 0.1 · q*/β ≥ 5 인 설정에서만 (연속 극한 조건, 논문 §4.4·§5). "
             "주의: 이 적용 판정 규칙은 참 계수 DP 문턱을 본 뒤 정한 사후 규칙이며 4개 중 1개만 해당 — 띠 검증 근거는 indep",
        indep="검토 후 추가(사전 등록 아님): 정확한 정책 반복과 정책 불일치 0칸, 연속 AR(1) 몬테카를로 최선 띠 대비 "
              "참 계수 DP 띠의 기대 순성장 ≥ 0.99",
        negative="매수·보유 대비 추가 왕복 ≤ 연 1회. 사전 등록 판정은 iid('순수 잡음 지표'); price는 판정이 아닌 "
                 "가짜 거래율·가짜 Sharpe 이득 기준선(실제 결과와 나란히 보고)",
        reps="결정론적 + walk가 rep마다 seed만 바꿈 → rep 5개가 같음. --reps 1 권장"))
    if a.part == "sign":
        res["sign"] = sign_report(a.name, a.rep)
    if a.part in ("pos", "all"):
        res["positive"] = positive()
    if a.part in ("neg", "all"):
        res["negative"] = [negative_one(k, s, drift_day=a.drift) for k in a.kinds.split(",") for s in range(a.seeds)]
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
