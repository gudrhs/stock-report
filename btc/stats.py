# -*- coding: utf-8 -*-
"""
성과 지표와 통계 검정 — scipy·statsmodels 없이 numpy·math만 씁니다.

입력 규칙
  별도 언급이 없으면 모든 수익률 인자는 같은 날짜에 정렬된 1차원 배열이며,
  00:00 UTC 기준 '일별 단순 순수익률'(비용 차감 후)입니다. 연율화는 √365 / 365일.

  summary()                     CAGR·변동성·샤프·소르티노·MDD·칼마·수중기간·최종배수
  trade_stats()                 노출도·연간 전환 횟수·평균 보유일·거래 적중률·손익비 (4h 봉 입력)
  sharpe() / psr() / dsr()      샤프, 확률적 샤프, 디플레이티드 샤프 (Bailey & López de Prado)
                                → psr·dsr 안의 샤프는 '일 단위, 연율화하지 않은 값'입니다
  stationary_bootstrap_diff()   두 전략 샤프 차이의 짝지은 정상(stationary) 부트스트랩 검정
  lw_hac_test()                 Ledoit-Wolf(2008) HAC 델타법 샤프 차이 검정 (교차확인용)
  holm()                        Holm 단계하강 다중검정 보정
  pbo_cscv()                    CSCV로 계산한 백테스트 과최적화 확률(PBO)과 IS→OOS 열화 기울기
  null_markov()                 N1 귀무: 노출도·전환율을 맞춘 무작위 2상태 마르코프 롱/현금 정책
  null_circular_shift()         N3 귀무: 일별 포지션을 30일 이상 원형 이동해 수익률과 어긋나게 함
  newey_west_alpha_beta()       매수·보유 대비 알파·베타 (Newey-West 표준오차)

날짜 ≥ 2025-01-01 (락박스) 차단은 날짜를 아는 호출 쪽(walkforward/report)에서 합니다.
여기 함수들은 날짜 없는 배열만 받습니다.
"""
import itertools
import math

import numpy as np

EULER = 0.5772156649015329           # 오일러-마스케로니 상수 γ_E
DAYS = 365                           # 연율화 기준 (코인은 365일 거래)
BARS_PER_DAY = 6                     # 4h 봉


# ───────────────────────── 정규분포 ─────────────────────────

def norm_cdf(x):
    """표준정규 누적분포 Φ(x). erfc를 써서 왼쪽 꼬리에서도 상대오차가 작습니다."""
    return 0.5 * math.erfc(-float(x) / math.sqrt(2.0))


_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
      3.754408661907416e+00)


def norm_ppf(p):
    """
    표준정규 역함수 Φ⁻¹(p).
    Acklam 유리근사(상대오차 1.15e-9)로 시작해 Halley 보정 한 번 → 1e-15 수준.
    p > 0.5는 대칭 −Φ⁻¹(1−p)로 계산합니다 (1−p는 이 구간에서 정확히 표현됨).
    """
    p = float(p)
    if not 0.0 <= p <= 1.0 or math.isnan(p):
        raise ValueError("p must be in [0, 1]")
    if p == 0.0:
        return -math.inf
    if p == 1.0:
        return math.inf
    if p > 0.5:
        return -norm_ppf(1.0 - p)
    if p < 0.02425:                                    # 왼쪽 꼬리
        q = math.sqrt(-2.0 * math.log(p))
        x = ((((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5])
             / ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0))
    else:                                              # 가운데
        q = p - 0.5
        r = q * q
        x = ((((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q
             / (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0))
    if 0.5 * x * x < 700.0:                            # Halley 한 단계 (exp 넘침 방지)
        e = norm_cdf(x) - p
        u = e * math.sqrt(2.0 * math.pi) * math.exp(0.5 * x * x)
        x = x - u / (1.0 + 0.5 * x * u)
    return x


# ───────────────────────── 기본 도구 ─────────────────────────

def _arr(r, name="r"):
    a = np.asarray(r, dtype=float).ravel()
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: non-finite values")
    return a


def sharpe(r, periods=DAYS):
    """
    샤프 = 평균 / 표본표준편차(ddof=1) × √periods (rf = 0).
    periods=365 → 연율화, periods=1 → 일 단위(연율화 안 함). 표준편차 0이면 0.
    """
    r = _arr(r)
    if len(r) < 2:
        return 0.0
    sd = r.std(ddof=1)
    return float(r.mean() / sd * math.sqrt(periods)) if sd > 0 else 0.0


def _row_sharpe(x, periods):
    """(k, n) 행렬의 행별 샤프 (ddof=1). 표준편차 0인 행은 0."""
    mu = x.mean(axis=1)
    sd = x.std(axis=1, ddof=1)
    out = np.zeros(len(mu))
    ok = sd > 0
    out[ok] = mu[ok] / sd[ok] * math.sqrt(periods)
    return out


def _col_sharpe(x, periods):
    return _row_sharpe(x.T, periods) if x.ndim == 2 else np.array([sharpe(x, periods)])


def moments(r):
    """(일 단위 샤프, 왜도 g3, 첨도 g4[비초과, 정규=3]). 왜도·첨도는 모집단 중심적률(ddof=0)."""
    r = _arr(r)
    m = r.mean()
    d = r - m
    m2 = float((d * d).mean())
    if m2 <= 0:
        return 0.0, 0.0, 3.0
    g3 = float((d ** 3).mean() / m2 ** 1.5)
    g4 = float((d ** 4).mean() / m2 ** 2)
    return sharpe(r, 1), g3, g4


# ───────────────────────── 요약 지표 ─────────────────────────

def max_drawdown(r):
    """일별 단순수익률 경로의 최대낙폭 (음수, 예: −0.834). 시작 자산 1.0을 고점 후보에 포함."""
    r = _arr(r)
    eq = np.concatenate([[1.0], np.cumprod(1.0 + r)])
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1.0).min())


def summary(r_daily, periods=DAYS):
    """
    일별 단순 순수익률 → 성과 요약 dict.
      cagr      최종배수^(365/일수) − 1
      vol       표본표준편차 × √365
      sharpe    평균/표준편차 × √365
      sortino   평균/하방편차 × √365, 하방편차 = √mean(min(r,0)²) (목표 0, 전체 표본 수로 나눔)
      max_dd    최대낙폭 (음수)
      calmar    cagr / |max_dd|
      tuw_days  가장 긴 '수중' 연속 일수 — 자산이 직전 최고치보다 낮았던 날이 연속된 최대 길이
                (끝까지 회복 못 한 구간도 포함)
      twm       최종 자산 배수 ∏(1+r)
    """
    r = _arr(r_daily, "r_daily")
    n = len(r)
    if n == 0:
        nan = float("nan")
        return dict(cagr=nan, vol=nan, sharpe=nan, sortino=nan, max_dd=0.0, calmar=nan,
                    tuw_days=0, twm=1.0, n_days=0)
    eq = np.concatenate([[1.0], np.cumprod(1.0 + r)])
    twm = float(eq[-1])
    cagr = float(twm ** (periods / n) - 1.0) if twm > 0 else -1.0
    vol = float(r.std(ddof=1) * math.sqrt(periods)) if n > 1 else 0.0
    down = math.sqrt(float((np.minimum(r, 0.0) ** 2).mean()))
    sortino = float(r.mean() / down * math.sqrt(periods)) if down > 0 else (math.inf if r.mean() > 0 else 0.0)
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    mdd = float(dd.min())
    under = eq[1:] < peak[1:] * (1.0 - 1e-12)
    tuw = _longest_run(under)
    return dict(cagr=cagr, vol=vol, sharpe=sharpe(r, periods), sortino=sortino, max_dd=mdd,
                calmar=float(cagr / abs(mdd)) if mdd < 0 else math.inf, tuw_days=int(tuw),
                twm=twm, n_days=int(n))


def _longest_run(mask):
    """불리언 배열에서 True가 연속된 최대 길이"""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return 0
    x = np.concatenate([[0], m.astype(np.int8), [0]])
    d = np.diff(x)
    starts = np.nonzero(d == 1)[0]
    ends = np.nonzero(d == -1)[0]
    return int((ends - starts).max())


def _runs(mask):
    """True 연속 구간의 (시작, 끝[포함하지 않음]) 목록"""
    m = np.asarray(mask, dtype=bool)
    x = np.concatenate([[0], m.astype(np.int8), [0]])
    d = np.diff(x)
    return np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]


def trade_stats(position_4h, bar_logr_4h, bars_per_day=BARS_PER_DAY, initial_position=0.0):
    """
    4h 봉 단위 매매 통계.
      position_4h[t]  t봉 구간에 (체결 직후) 들고 있던 비중 (0/1, 비율이면 >0을 보유로 봄)
      bar_logr_4h[t]  같은 구간의 순 로그수익 (비용은 비중이 바뀐 봉에 기록된다고 가정)
    거래 = 보유(>0)가 연속된 구간. 거래 수익 = 그 구간 로그수익 합 + 청산 봉(구간 직후 첫 현금 봉)의
    로그수익 — 현금은 0을 벌므로 청산 봉에는 매도 비용만 들어 있습니다. 끝까지 보유한 거래도 포함.
    반환:
      exposure           평균 비중
      switches_per_year  비중이 바뀐 횟수 / 연수 (첫 봉 이전 비중 = initial_position)
      avg_hold_days      보유 구간 평균 길이(봉) / bars_per_day
      hit_rate           수익 > 0 인 거래 비율
      profit_factor      이익 거래 단순수익 합 / |손실 거래 단순수익 합| (손실 없으면 inf, 거래 없으면 nan)
      n_trades, avg_trade(거래당 평균 단순수익)
    """
    pos = _arr(position_4h, "position_4h")
    lr = _arr(bar_logr_4h, "bar_logr_4h")
    if len(pos) != len(lr):
        raise ValueError("position_4h and bar_logr_4h must have the same length")
    T = len(pos)
    nan = float("nan")
    if T == 0:
        return dict(exposure=nan, switches_per_year=nan, avg_hold_days=nan, hit_rate=nan,
                    profit_factor=nan, n_trades=0, avg_trade=nan)
    years = T / (bars_per_day * DAYS)
    prev = np.concatenate([[float(initial_position)], pos[:-1]])
    n_sw = int(np.count_nonzero(pos != prev))
    held = pos > 0
    starts, ends = _runs(held)
    trades = []
    for s, e in zip(starts, ends):
        tot = lr[s:e].sum()
        if e < T:                                   # 청산 봉(매도 비용)
            tot += lr[e]
        trades.append(math.expm1(tot))
    trades = np.array(trades)
    if len(trades):
        wins, losses = trades[trades > 0], trades[trades < 0]
        hit = float((trades > 0).mean())
        pf = float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else (
            math.inf if len(wins) else nan)
        avg_hold = float((ends - starts).mean() / bars_per_day)
        avg_tr = float(trades.mean())
    else:
        hit = pf = avg_hold = avg_tr = nan
    return dict(exposure=float(pos.mean()), switches_per_year=float(n_sw / years),
                avg_hold_days=avg_hold, hit_rate=hit, profit_factor=pf,
                n_trades=int(len(trades)), avg_trade=avg_tr)


def exposure_switch_rate(position_4h, initial_position=0.0):
    """N1 보정용: (평균 노출도, 봉당 전환 횟수). 전환 = 직전 봉과 비중이 다른 봉."""
    pos = _arr(position_4h, "position_4h")
    prev = np.concatenate([[float(initial_position)], pos[:-1]])
    return float(pos.mean()), float(np.count_nonzero(pos != prev) / len(pos))


# ───────────────────────── PSR / DSR ─────────────────────────

def psr(r, sr_star=0.0):
    """
    확률적 샤프 PSR(SR*) = Φ( (SR − SR*)·√(T−1) / √(1 − g3·SR + (g4−1)/4·SR²) )
    SR, SR* 모두 '일 단위, 연율화하지 않은' 샤프. g4는 비초과 첨도(정규 = 3).
    """
    r = _arr(r)
    T = len(r)
    if T < 3:
        return float("nan")
    sr, g3, g4 = moments(r)
    den = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr * sr
    if den <= 0:
        den = 1e-12
    return norm_cdf((sr - sr_star) * math.sqrt(T - 1) / math.sqrt(den))


def expected_max_sharpe(sr_var, n):
    """
    n번 독립 시험에서 운만으로 기대되는 최대 샤프 (일 단위):
      SR0 = √V[SR] · ((1−γ_E)·Φ⁻¹(1−1/N) + γ_E·Φ⁻¹(1−1/(N·e)))
    sr_var = 시험들의 일 단위 샤프 분산. n ≤ 1이면 0.
    """
    if sr_var < 0:
        raise ValueError("sr_var must be >= 0")
    if n <= 1:
        return 0.0
    return math.sqrt(sr_var) * ((1.0 - EULER) * norm_ppf(1.0 - 1.0 / n)
                                + EULER * norm_ppf(1.0 - 1.0 / (n * math.e)))


def dsr(r, n_trials, sr_var):
    """디플레이티드 샤프 = PSR(SR0), SR0 = expected_max_sharpe(sr_var, n_trials) (일 단위)"""
    return psr(r, expected_max_sharpe(sr_var, n_trials))


def trials_sr_var(M):
    """시험(열)별 일 단위 샤프의 분산 (ddof=1). M: T×N 일별 수익률 행렬."""
    M = np.asarray(M, dtype=float)
    s = _col_sharpe(M, 1)
    return float(s.var(ddof=1)) if len(s) > 1 else 0.0


# ───────────────────────── 부트스트랩 ─────────────────────────

def stationary_bootstrap_indices(n, size, mean_block=20, rng=None):
    """
    Politis-Romano 정상 부트스트랩 인덱스 (size, n).
    각 위치에서 확률 1/mean_block로 새 블록을 무작위 시작점에서 열고(기하분포 블록 길이),
    아니면 직전 인덱스 + 1 (원형으로 이어 붙임).
    """
    rng = np.random.default_rng(rng)
    p = 1.0 / float(mean_block)
    t = np.arange(n)
    new = rng.random((size, n)) < p
    new[:, 0] = True
    starts = rng.integers(0, n, size=(size, n))
    last = np.maximum.accumulate(np.where(new, t, 0), axis=1)
    s = np.take_along_axis(starts, last, axis=1)
    return (s + (t - last)) % n


def stationary_bootstrap_diff(ra, rb, mean_block=20, n_boot=10000, seed=0, periods=DAYS):
    """
    짝지은 정상 부트스트랩으로 ΔSR = SR(a) − SR(b) (연율화) 검정.
    같은 인덱스를 a·b에 함께 써서 두 전략의 상관을 보존합니다.
      obs   관측 ΔSR
      ci90  부트스트랩 ΔSR의 5%·95% 분위수 (백분위 신뢰구간)
      p     단측 p (H1: SR(a) > SR(b)) = mean( (ΔSR_b − obs) ≥ obs )  — 중심화한 분포로 귀무 근사
      se    부트스트랩 ΔSR 표준편차
    """
    ra, rb = _arr(ra, "ra"), _arr(rb, "rb")
    if len(ra) != len(rb):
        raise ValueError("ra and rb must be aligned (same length)")
    n = len(ra)
    obs = sharpe(ra, periods) - sharpe(rb, periods)
    rng = np.random.default_rng(seed)
    chunk = max(1, 2_000_000 // max(n, 1))
    diffs = np.empty(n_boot)
    done = 0
    while done < n_boot:
        k = min(chunk, n_boot - done)
        idx = stationary_bootstrap_indices(n, k, mean_block, rng)
        diffs[done:done + k] = _row_sharpe(ra[idx], periods) - _row_sharpe(rb[idx], periods)
        done += k
    lo, hi = np.percentile(diffs, [5.0, 95.0])
    p = float(np.mean((diffs - obs) >= obs))
    return dict(obs=float(obs), ci90=(float(lo), float(hi)), p=p,
                se=float(diffs.std(ddof=1)), n_boot=int(n_boot), mean_block=float(mean_block))


# ───────────────────────── Ledoit-Wolf HAC ─────────────────────────

def parzen(x):
    """Parzen 커널 가중치"""
    x = np.abs(np.asarray(x, dtype=float))
    return np.where(x <= 0.5, 1.0 - 6.0 * x ** 2 + 6.0 * x ** 3,
                    np.where(x <= 1.0, 2.0 * (1.0 - x) ** 3, 0.0))


def _lrv_parzen(z, bandwidth=None):
    """
    스칼라 계열 z(평균 0)의 장기분산 (Parzen 커널).
    bandwidth=None → Andrews(1991) AR(1) 플러그인: S = 2.6614·(α(2)·T)^(1/5), α(2) = 4ρ²/(1−ρ)⁴.
    """
    T = len(z)
    if bandwidth is None:
        den = float(z[:-1] @ z[:-1])
        rho = float(z[1:] @ z[:-1]) / den if den > 0 else 0.0
        rho = min(max(rho, -0.97), 0.97)
        a2 = 4.0 * rho * rho / (1.0 - rho) ** 4
        bandwidth = 2.6614 * (a2 * T) ** 0.2
    S = float(bandwidth)
    lrv = float(z @ z) / T
    jmax = min(int(math.floor(S)), T - 1) if S > 0 else 0
    for j in range(1, jmax + 1):
        w = float(parzen(j / S))
        if w <= 0:
            break
        lrv += 2.0 * w * float(z[j:] @ z[:-j]) / T
    return lrv, S


def lw_hac_test(ra, rb, bandwidth=None, periods=DAYS):
    """
    Ledoit & Wolf (2008) 샤프 차이 HAC 검정.
    v = (μa, μb, E[a²], E[b²]),  f(v) = μa/√(E[a²]−μa²) − μb/√(E[b²]−μb²)
    SE² = ∇fᵀ Ψ ∇f / T, Ψ = y_t = (a−μa, b−μb, a²−E[a²], b²−E[b²])의 HAC 장기공분산
    (Parzen 커널, 대역폭 자동 또는 지정, 소표본 보정 T/(T−4)).
    ∇fᵀΨ∇f 는 스칼라 계열 z_t = ∇fᵀ y_t 의 장기분산과 같으므로 그것을 직접 계산합니다.
    반환: diff·se는 연율화(×√periods), z, p_one_sided (H1: SR(a) > SR(b)), p_two_sided, bandwidth.
    샤프 정의는 LW와 같이 모집단 표준편차(ddof=0)를 씁니다.
    """
    ra, rb = _arr(ra, "ra"), _arr(rb, "rb")
    if len(ra) != len(rb):
        raise ValueError("ra and rb must be aligned (same length)")
    T = len(ra)
    if T < 10:
        raise ValueError("need at least 10 observations")
    mua, mub = ra.mean(), rb.mean()
    ga, gb = (ra * ra).mean(), (rb * rb).mean()
    va, vb = ga - mua * mua, gb - mub * mub
    if va <= 0 or vb <= 0:
        raise ValueError("zero-variance series")
    diff = mua / math.sqrt(va) - mub / math.sqrt(vb)
    grad = np.array([ga / va ** 1.5, -gb / vb ** 1.5, -mua / (2.0 * va ** 1.5), mub / (2.0 * vb ** 1.5)])
    Y = np.column_stack([ra - mua, rb - mub, ra * ra - ga, rb * rb - gb])
    z = Y @ grad
    lrv, S = _lrv_parzen(z, bandwidth)
    lrv *= T / (T - 4.0)
    se = math.sqrt(max(lrv, 0.0) / T)
    zstat = diff / se if se > 0 else (math.inf if diff > 0 else (-math.inf if diff < 0 else 0.0))
    p1 = 0.5 * math.erfc(zstat / math.sqrt(2.0))           # 1 − Φ(z), 오른쪽 꼬리도 정확
    p2 = min(1.0, 2.0 * min(p1, 1.0 - p1))
    k = math.sqrt(periods)
    return dict(diff=float(diff * k), se=float(se * k), z=float(zstat), p_one_sided=float(p1),
                p_two_sided=float(p2), bandwidth=float(S))


# ───────────────────────── Holm ─────────────────────────

def holm(pvals):
    """Holm 단계하강 보정 p값 (입력 순서 그대로, 단조, 1로 상한)"""
    p = np.asarray(pvals, dtype=float).ravel()
    m = len(p)
    if m == 0:
        return p.copy()
    order = np.argsort(p, kind="mergesort")
    adj_sorted = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[order]))
    out = np.empty(m)
    out[order] = adj_sorted
    return out


# ───────────────────────── PBO (CSCV) ─────────────────────────

def pbo_cscv(M, S=16, periods=DAYS):
    """
    Bailey, Borwein, López de Prado, Zhu (2015) CSCV.
    M: T×N 일별 수익률 (N개 변형). 앞쪽 나머지 행을 버리고 S개 같은 길이 블록으로 나눈 뒤
    C(S, S/2)개 조합 각각을 표본내(IS), 나머지를 표본외(OOS)로 씁니다.
    IS 샤프 최고 변형 n*의 OOS 상대순위 w = rank/(N+1) (1 = 최하, 동점은 평균순위),
    logit λ = ln(w/(1−w)),  PBO = mean(λ ≤ 0).
    slope/intercept: 분할별 (IS 샤프, OOS 샤프) of n* 의 OLS 회귀 — 열화(degradation).
    샤프는 연율화 값 (순위·기울기는 척도와 무관).
    """
    M = np.asarray(M, dtype=float)
    if M.ndim != 2:
        raise ValueError("M must be T x N")
    if not np.all(np.isfinite(M)):
        raise ValueError("M: non-finite values")
    if S < 2 or S % 2:
        raise ValueError("S must be an even integer >= 2")
    T, N = M.shape
    L = T // S
    if L < 2:
        raise ValueError("too few rows for S blocks")
    X = M[T - L * S:]                                   # 앞쪽 나머지 행 버림
    c = X.mean(axis=0)
    X0 = X - c                                          # 수치 안정용 이동 (분산은 불변)
    blk_sum = X0.reshape(S, L, N).sum(axis=1)           # (S, N)
    blk_sq = (X0 * X0).reshape(S, L, N).sum(axis=1)
    combos = np.array(list(itertools.combinations(range(S), S // 2)), dtype=np.int64)
    C = np.zeros((len(combos), S))
    np.put_along_axis(C, combos, 1.0, axis=1)
    n_half = L * S // 2
    tot_sum, tot_sq = blk_sum.sum(axis=0), blk_sq.sum(axis=0)

    def _sr(s, q):
        m0 = s / n_half
        var = (q - n_half * m0 * m0) / (n_half - 1)
        sd = np.sqrt(np.maximum(var, 0.0))
        mu = m0 + c
        out = np.zeros_like(mu)
        ok = sd > 1e-15
        out[ok] = mu[ok] / sd[ok] * math.sqrt(periods)
        return out

    is_sum, is_sq = C @ blk_sum, C @ blk_sq
    sr_is = _sr(is_sum, is_sq)                          # (n_splits, N)
    sr_oos = _sr(tot_sum - is_sum, tot_sq - is_sq)
    best = np.argmax(sr_is, axis=1)
    rows = np.arange(len(best))
    sel_is = sr_is[rows, best]
    sel_oos = sr_oos[rows, best]
    less = (sr_oos < sel_oos[:, None]).sum(axis=1)
    eq = (sr_oos == sel_oos[:, None]).sum(axis=1)       # 자신 포함
    rank = less + (eq + 1) / 2.0
    w = rank / (N + 1.0)
    logits = np.log(w / (1.0 - w))
    vx = sel_is.var()
    if vx > 0:
        slope = float(((sel_is - sel_is.mean()) * (sel_oos - sel_oos.mean())).mean() / vx)
        intercept = float(sel_oos.mean() - slope * sel_is.mean())
    else:
        slope = intercept = float("nan")
    return dict(pbo=float(np.mean(logits <= 0)), logits=logits, slope=slope, intercept=intercept,
                is_sr=sel_is, oos_sr=sel_oos, p_oos_loss=float(np.mean(sel_oos < 0)),
                n_splits=int(len(combos)), block_len=int(L), rows_used=int(L * S))


# ───────────────────────── 귀무 분포 N1 / N3 ─────────────────────────

def markov_probs(exposure, switch_rate):
    """
    정상 노출도 e, 봉당 기대 전환 s를 맞추는 (P(0→1), P(1→0)).
    정상분포 π1 = p/(p+q) = e, 전환율 = π0·p + π1·q = 2·e·q  →  q = s/(2e), p = s/(2(1−e)).
    """
    e, s = float(exposure), float(switch_rate)
    if not 0.0 < e < 1.0:
        raise ValueError("exposure must be in (0, 1)")
    if s < 0 or s > 2.0 * min(e, 1.0 - e) + 1e-12:
        raise ValueError("switch_rate must be in [0, 2*min(e, 1-e)]")
    return min(1.0, s / (2.0 * (1.0 - e))), min(1.0, s / (2.0 * e))


def _markov_chunk(rng, k, T, e, p, q):
    """(T, k) int8 경로. 첫 상태는 정상분포 Bernoulli(e)에서 뽑습니다."""
    a = np.empty((T, k), dtype=np.int8)
    state = rng.random(k) < e
    U = rng.random((T, k))
    for t in range(T):
        a[t] = state
        state = state ^ (U[t] < np.where(state, q, p))
    return a


def _chunk_size(T):
    return max(1, 4_000_000 // max(T, 1))


def markov_positions(n, T, exposure, switch_rate, seed=0):
    """N1 경로만 따로 생성 (n, T) int8 — null_markov와 같은 seed면 같은 경로입니다."""
    p, q = markov_probs(exposure, switch_rate)
    rng = np.random.default_rng(seed)
    out = np.empty((n, T), dtype=np.int8)
    ch = _chunk_size(T)
    for s in range(0, n, ch):
        k = min(ch, n - s)
        out[s:s + k] = _markov_chunk(rng, k, T, exposure, p, q).T
    return out


def null_markov(m, cost, exposure, switch_rate, n=1000, seed=0, day_id=None,
                bars_per_day=BARS_PER_DAY, a_prev=0, close_out=True):
    """
    N1: 무작위 2상태 마르코프 롱/현금 정책 n개의 샤프(연율화) 배열.
      m[t]        t봉 결정이 버는 4h 시가→시가 로그수익 ln(O_{t+2}/O_{t+1})
      cost        편도 비용 c
      exposure    목표 정상 노출도 e,  switch_rate  봉당 기대 전환 횟수 s
    봉별 순 로그수익 g_t = a_t·m_t + |a_t − a_{t−1}|·ln(1−c)  (a_{−1} = a_prev,
    close_out=True면 마지막 봉에 청산 비용 |a_{T−1}|·ln(1−c)도 더함 — 백테스트 회계와 동일).
    day_id(길이 T, 비감소 정수)가 있으면 같은 날 로그수익을 합쳐 일별 단순수익으로 바꾸고 √365로
    연율화합니다. 봉이 하나도 없는 날(장애)은 수익 0인 날로 채웁니다. 없으면 봉 단위, √(6·365).
    """
    m = _arr(m, "m")
    T = len(m)
    p, q = markov_probs(exposure, switch_rate)
    lc = math.log1p(-float(cost))
    if day_id is not None:
        d = np.asarray(day_id, dtype=np.int64).ravel()
        if len(d) != T:
            raise ValueError("day_id must have the same length as m")
        if np.any(np.diff(d) < 0):
            raise ValueError("day_id must be non-decreasing")
        d = d - d[0]
        n_days = int(d[-1]) + 1
        brk = np.concatenate([[0], np.nonzero(np.diff(d))[0] + 1])
        day_of_brk = d[brk]
        periods = DAYS
    else:
        periods = bars_per_day * DAYS
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    ch = _chunk_size(T)
    for s in range(0, n, ch):
        k = min(ch, n - s)
        a = _markov_chunk(rng, k, T, exposure, p, q).astype(float)      # (T, k)
        prev = np.vstack([np.full((1, k), float(a_prev)), a[:-1]])
        g = a * m[:, None] + np.abs(a - prev) * lc
        if close_out:
            g[-1] += np.abs(a[-1]) * lc
        if day_id is not None:
            daily = np.zeros((n_days, k))
            daily[day_of_brk] = np.add.reduceat(g, brk, axis=0)
            x = np.expm1(daily)
        else:
            x = np.expm1(g)
        out[s:s + k] = _row_sharpe(x.T, periods)
    return out


def null_circular_shift(pos_daily, r_asset_daily, n=2000, min_shift=30, seed=0, periods=DAYS):
    """
    N3: 일별 포지션을 원형 이동(이동량 k ∈ [min_shift, T−min_shift] 균등)해 자산 수익률과 어긋나게 한 뒤
    전략 일수익 ≈ pos_shift[d]·r_asset[d]의 샤프(연율화) 배열. 이동해도 노출도 분포와 보유 구조는 그대로.
    """
    pos = _arr(pos_daily, "pos_daily")
    r = _arr(r_asset_daily, "r_asset_daily")
    T = len(pos)
    if len(r) != T:
        raise ValueError("pos_daily and r_asset_daily must be aligned")
    if T - min_shift < min_shift:
        raise ValueError("series too short for min_shift")
    rng = np.random.default_rng(seed)
    shifts = rng.integers(min_shift, T - min_shift + 1, size=n)
    t = np.arange(T)
    out = np.empty(n)
    ch = max(1, 2_000_000 // T)
    for s in range(0, n, ch):
        k = shifts[s:s + ch]
        idx = (t[None, :] - k[:, None]) % T                  # np.roll(pos, k)
        out[s:s + len(k)] = _row_sharpe(pos[idx] * r[None, :], periods)
    return out


def null_pvalue(obs, null):
    """귀무 분포 대비 단측 p = (1 + #(null ≥ obs)) / (1 + n)"""
    null = np.asarray(null, dtype=float)
    return float((1 + np.count_nonzero(null >= obs)) / (1 + len(null)))


def null_percentile(obs, null):
    """관측값의 귀무 분포 내 백분위 (0~1, 동점은 절반)"""
    null = np.asarray(null, dtype=float)
    return float(np.mean(null < obs) + 0.5 * np.mean(null == obs))


# ───────────────────────── 알파·베타 ─────────────────────────

def newey_west_alpha_beta(r, r_bh, lags=10, periods=DAYS):
    """
    r = α + β·r_bh + ε 의 OLS, Newey-West(Bartlett, w_l = 1 − l/(lags+1)) 표준오차.
    공분산에 소표본 보정 T/(T−2)을 곱합니다. alpha_annual = α × 365.
    """
    y = _arr(r, "r")
    x = _arr(r_bh, "r_bh")
    T = len(y)
    if len(x) != T:
        raise ValueError("r and r_bh must be aligned")
    if T < 3:
        raise ValueError("need at least 3 observations")
    X = np.column_stack([np.ones(T), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ (X.T @ y)
    e = y - X @ b
    Xe = X * e[:, None]
    Sm = Xe.T @ Xe
    for l in range(1, min(int(lags), T - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        G = Xe[l:].T @ Xe[:-l]
        Sm += w * (G + G.T)
    V = XtX_inv @ Sm @ XtX_inv * T / (T - 2.0)
    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    alpha, beta = float(b[0]), float(b[1])
    t_a = alpha / se[0] if se[0] > 0 else float("nan")
    t_b = beta / se[1] if se[1] > 0 else float("nan")
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(e @ e) / ss_tot if ss_tot > 0 else float("nan")
    return dict(alpha_annual=alpha * periods, beta=beta, t_alpha=float(t_a), t_beta=float(t_b),
                se_alpha_annual=float(se[0] * periods), se_beta=float(se[1]),
                p_alpha_two_sided=float(math.erfc(abs(t_a) / math.sqrt(2.0))) if se[0] > 0 else float("nan"),
                alpha_daily=alpha, r2=r2, n=int(T))
