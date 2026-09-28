# -*- coding: utf-8 -*-
"""
현실 점검 — btc/research/success_criteria.md '현실 점검 (2026-09-28 02:10 UTC 등록)' 그대로. 새 시험이 아닙니다.

이미 기록된 판단값(R6·R3·W2·C1)과 사전 등록한 기준 규칙을 한국에서 실제로 쓸 때의 조건으로 다시 셈합니다.
새 모델·설정·특징이 없으므로 시험 수에 더하지 않습니다.

대상 (모두 00:00 UTC 판단봉에서만 판단)
  · 기록된 판단: R6(반복 0~4, 범위는 0~9), R3(0~4), W2(결정론적, 반복 0), C1(R6 반복 0~9 다수결)
    판단값은 more_rl.rep_targets / more_rl.committee 와 같은 계산 (판단용 비용 0.3%, W2 띠 그대로)
  · 기준 규칙: B0 매수·보유, B2 200일선, B5 일봉 MACD, B6 1개월 모멘텀 (evaluate.baseline_targets 와 같은 식),
    B80 늘 80%(2%p 띠), SMA120(종가 > 720봉 평균, 사후 선택)

시나리오
  · 통화: USD(비트스탬프) / KRW — 봉 가격 × k(D), k(D) = 그날 00:00 UTC 원화 시가 ÷ 비트스탬프 00:00 시가
    (그날 안에서는 일정. 원화 시가: 2017-10-23까지 코빗 00:00 직전 체결가, 2017-10-24부터 업비트 일봉 시가)
  · 체결 시각: booked(등록대로 00:00 시가) · exec(00:30 15분봉 시가, 학습 모델의 월초 판단은 c(T) 뒤) ·
    lag1 · lag2 (판단을 판단봉 한 칸·두 칸 뒤로, 앞 칸은 현금)
  · 편도 비용: 0.15%(등록) · 0.05%(업비트 원화 수수료)

  BTC_LOCKBOX_OPEN=1 python -m btc.research.recost   → data/btc/research_runs/recost.json + 한국어 요약
"""
import argparse
import json
import math
import os

import numpy as np
import pandas as pd

from .. import stats as S
from ..env import simulate, simulate_weights, daily_marks, BAR_SEC
from ..evaluate import Window, FI
from .evaluate import daily_hold
from .walk import load_run, decision_mask, RUNS
from .more_rl import rep_targets, committee, cfg_of, headline_index

DAY = 86400
EXEC_OFFSET = 1800                    # 00:30 — 00:20 판단 뒤 첫 15분봉
DECIDE_DELAY = 1200                   # 판단 실행 시각 = 판단 가능 시각 + 20분 (CI 00:20)
EXEC_MAX_WAIT = 4 * 3600              # 00:30 봉이 비면 04:00 전 첫 체결 가능한 15분봉 시가
MIN_TRADE = 1e-12
N_BOOT, BOOT_SEED, MEAN_BLOCK = 4000, 1, 20

RULES = ("B0", "B2", "B5", "B6", "B80", "SMA120")
RULE_KO = {"B0": "매수·보유", "B2": "200일선", "B5": "일봉 MACD", "B6": "1개월 모멘텀", "B80": "늘 80%",
           "SMA120": "SMA120 (사후 선택)"}
FRAC_RULES = {"B80"}
MODELS = {"R6": ("R6_daily_trend8_uniform", 5, 10), "R3": ("R3_daily_trend8_log5", 5, 5), "W2": ("W2_dp_band", 1, 1)}
C1_REPS = 10
ORDER = ("R6", "R3", "W2", "C1") + RULES
TRAINED = {"R6", "R3", "W2", "C1"}   # 월간 학습이 있는 대상 → 실행 가능 시각에 c(T) 지연
CURRENCIES = ("USD", "KRW")
TIMINGS = ("booked", "exec", "lag1", "lag2")
COSTS = (0.0015, 0.0005)
WINDOWS = {
    "main": ("2017-01-01", "2026-09-25", "", "reality-check recost:main 2017-01-01..2026-09-25"),
    "early": ("2015-01-01", "2017-01-01", "__early", "reality-check recost:early 2015-01-01..2017-01-01"),
}
WINDOW_KO = {"main": "2017-01~2026-09", "early": "2015~2016"}
# 등록 계산(USD·booked·0.15%)이 기록된 값과 같은지 확인 (값, 소수 자리)
REF_CHECK = {
    "main": {"B0": (1.0148, 4), "B2": (1.0996, 4), "B5": (1.2523, 4), "B6": (1.1189, 4), "R6": (1.1487, 4),
             "R3": (1.0855, 4), "W2": (1.1920, 4), "C1": (1.1118, 4), "B80": (1.020, 3), "SMA120": (1.411, 3)},
    "early": {"B0": (1.2203, 4), "B2": (1.5920, 4), "B5": (1.7426, 4), "R6": (1.9044, 4), "R3": (1.9832, 4),
              "W2": (1.2957, 4), "C1": (1.8828, 4)},
}
KRW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "btc", "krw"))
KRW_FILE = os.path.join(KRW_DIR, "krw_daily.csv")
OUT = os.path.join(RUNS, "recost.json")


# ══════════ 기준 규칙 ══════════
def rule_signals(d):
    """봉마다 규칙 목표 (evaluate.baseline_targets 와 같은 식 + B80·SMA120). 구간·판단봉은 따로 자름"""
    c = pd.Series(d.c)
    sma720, sma1200 = c.rolling(720).mean().to_numpy(), c.rolling(1200).mean().to_numpy()
    X = d.X
    return {
        "B0": np.ones(d.T),
        "B2": np.where(d.c > sma1200, 1.0, 0.0),
        "B5": np.where(X[:, FI["macdh_1d"]] > 0, 1.0, 0.0),
        "B6": np.where(X[:, FI["ret_180"]] > 0, 1.0, 0.0),
        "B80": np.full(d.T, 0.8),
        "SMA120": np.where(d.c > sma720, 1.0, 0.0),
    }


def rule_targets(d, a, b, mask):
    """판단봉에서만 목표 (evaluate.daily_hold). 반환 {규칙: (목표, 비율 회계 여부)}"""
    sig = rule_signals(d)
    return {k: (daily_hold(v, mask, a, b), k in FRAC_RULES) for k, v in sig.items()}


# ══════════ 기록된 판단 ══════════
def model_targets(d, a, b, tag, loader=load_run):
    """
    {이름: [(목표, 비율 회계 여부), ...반복]} + C1 합의. more_rl.evaluate_window 와 같은 계산.
    R6는 반복 0~9를 모두 읽습니다 (대표값은 0~4, 범위와 C1에 0~9).
    """
    out = {}
    r6_held = None
    for key, (name, reps, n_read) in MODELS.items():
        lst, held = [], []
        for r in range(n_read):
            run = loader(name + tag, r)
            tg, frac, h, sel, nan_share = rep_targets(d, run_cfg(name, run), run, a, b)
            if nan_share > 0:
                raise ValueError(f"{name}{tag} rep{r}: 모델 없는 판단봉 {nan_share:.1%} — 이 점검은 모델이 다 있는 실행만 씁니다")
            lst.append((tg, frac))
            held.append(h)
        out[key] = lst
        if key == "R6":
            r6_held, r6_sel = held, sel
    votes = np.stack(r6_held[:C1_REPS], axis=1)
    cpos = committee(votes, d.forced_hold[r6_sel])
    tg = np.full(d.T, np.nan)
    tg[r6_sel] = cpos
    out["C1"] = [(tg, False)]
    return out


def run_cfg(name, run):
    return cfg_of(name, run)


# ══════════ 체결 시각 ══════════
def decision_bars(tg):
    return np.nonzero(np.isfinite(tg))[0]


def lag_targets(tg, k):
    """판단을 판단봉 k칸 뒤로 (앞 k칸은 현금 0). 목표가 있는 봉 = 판단봉"""
    if k == 0:
        return tg
    sel = decision_bars(tg)
    out = np.full(len(tg), np.nan)
    vals = tg[sel]
    out[sel[k:]] = vals[:len(vals) - k]
    out[sel[:k]] = 0.0
    return out


def month_start(ts):
    t = pd.Timestamp(int(ts), unit="s", tz="UTC")
    return int(pd.Timestamp(year=t.year, month=t.month, day=1, tz="UTC").timestamp())


def exec_targets(d, tg, cuts=None):
    """
    실행 가능 시각으로 옮긴 목표.
    판단봉 t(마감 X)의 목표는 E = (max(X, c(T)) + 20분) 이후 첫 00:30에 체결됩니다.
    그 체결 봉 j = E가 속한 날 00:00에 시작하는 phase 0 봉(없으면 그 뒤 첫 봉)이고, 목표는 j−1 봉에 둡니다
    (simulate 는 t봉 목표를 t+1봉 시가에 체결). 같은 봉으로 모이면 늦게 정한 판단이 이깁니다.
    cuts: {월 시작 T: c(T)} — 학습 모델만. None 이면 지연 없음(규칙).
    """
    sel = decision_bars(tg)
    out = np.full(len(tg), np.nan)
    close = d.ts + BAR_SEC
    for t in sel:                                   # 시간 순서 → 같은 봉이면 뒤의 판단이 덮어씀
        X = int(close[t])
        avail = X
        if cuts is not None:
            T = month_start(X)
            if T not in cuts:
                raise ValueError(f"c(T) 없음: {pd.Timestamp(T, unit='s', tz='UTC')}")
            avail = max(X, int(cuts[T]))
        e = avail + DECIDE_DELAY
        day = (e // DAY) * DAY
        if e > day + EXEC_OFFSET:
            day += DAY
        j = int(np.searchsorted(d.ts, day))         # 그날 00:00 시작 봉 (죽었으면 그 뒤 첫 봉)
        if j >= d.T or j < 1:
            continue
        out[j - 1] = tg[t]
    return out


def month_cuts(ph, lo, hi):
    """구간 안 각 달의 c(T) (paper._first_cut, 학습 phase 전체)"""
    from .paper import _first_cut
    cuts = {}
    T = month_start(lo)
    while T < hi:
        c = _first_cut(ph, T)
        if c is None:
            raise ValueError(f"c(T) 계산 불가: {pd.Timestamp(T, unit='s', tz='UTC')}")
        cuts[T] = c
        nxt = pd.Timestamp(T, unit="s", tz="UTC") + pd.offsets.MonthBegin(1)
        T = int(nxt.timestamp())
    return cuts


# ══════════ 가격 ══════════
def open_at(df15, day_ts, offset=EXEC_OFFSET, max_wait=EXEC_MAX_WAIT):
    """
    각 날짜(00:00 UTC 초)에 대해 day + offset 에 시작하는 15분봉 시가.
    그 봉이 비었으면 day + max_wait 전 첫 체결 가능한 15분봉 시가, 그래도 없으면 NaN. 반환 (가격, 대체 수)
    """
    ts = df15["ts"].to_numpy(dtype=np.int64)
    op = df15["open"].to_numpy(dtype=float)
    ok = np.isfinite(op)
    ts_ok, op_ok = ts[ok], op[ok]
    want = np.asarray(day_ts, dtype=np.int64) + offset
    i = np.searchsorted(ts_ok, want)
    px = np.full(len(want), np.nan)
    n_sub = 0
    for k, (w, j) in enumerate(zip(want, i)):
        if j < len(ts_ok) and ts_ok[j] < w - offset + max_wait:
            px[k] = op_ok[j]
            n_sub += int(ts_ok[j] != w)
    return px, n_sub


def exec_open(d, df15):
    """00:00 에 시작하는 phase 0 봉의 시가를 00:30 15분봉 시가로 바꾼 배열 (다른 봉은 그대로). 반환 (배열, 대체 수)"""
    o = d.o.copy()
    j = np.nonzero(d.ts % DAY == 0)[0]
    px, n_sub = open_at(df15, d.ts[j])
    ok = np.isfinite(px)
    o[j[ok]] = px[ok]
    return o, dict(n_bars=int(len(j)), n_substituted=int(n_sub), n_missing=int((~ok).sum()))


def load_krw(path=KRW_FILE):
    k = pd.read_csv(path, index_col=0, parse_dates=True)
    k.index = k.index.tz_localize("UTC")
    return k


def daily_open(df15):
    """날짜별 비트스탬프 첫 체결 가능한 15분봉 시가 (phase 0 00:00 봉 시가와 같은 정의)"""
    s = pd.Series(df15["open"].to_numpy(dtype=float),
                  index=pd.to_datetime(df15["ts"].to_numpy(dtype=np.int64), unit="s", utc=True))
    return s.dropna().resample("1D").first()


def krw_factor(d, df15, krw):
    """
    봉마다 원화 배수 k(그 봉이 시작한 날). k(D) = 원화 시가(D) ÷ 비트스탬프 시가(D). 없는 날은 직전 값.
    반환 (배수 배열, 정보)
    """
    bs = daily_open(df15)
    k = (krw["krw_open"] / bs.reindex(krw.index)).rename("k")
    n_ff = int(k.isna().sum())
    k = k.ffill()                                   # 원화 자료 시작 전 날짜는 NaN 그대로 (구간 안에서만 쓰는지 확인)
    days = pd.to_datetime((d.ts // DAY) * DAY, unit="s", utc=True)
    fac = k.reindex(days).to_numpy(dtype=float)
    return fac, dict(days_forward_filled=n_ff)


def krw_mark_check(d, fac, krw, lo, hi):
    """근사 점검: 그날 마지막 봉 종가 × k(D) 와 업비트 일봉 종가의 차이 (업비트 구간만)"""
    close = d.ts + BAR_SEC
    m = (d.ts % DAY == 20 * 3600) & (close > lo) & (close <= hi)      # 20:00 시작 봉 = 그날 마지막 봉
    j = np.nonzero(m)[0]
    days = pd.to_datetime((d.ts[j] // DAY) * DAY, unit="s", utc=True)
    up = krw["upbit_close"].reindex(days).to_numpy(dtype=float)
    ok = np.isfinite(up)
    if not ok.any():
        return dict(n=0)
    err = d.c[j[ok]] * fac[j[ok]] / up[ok] - 1.0
    return dict(n=int(ok.sum()), mean_abs_pct=float(np.mean(np.abs(err)) * 100),
                p99_abs_pct=float(np.percentile(np.abs(err), 99) * 100), max_abs_pct=float(np.max(np.abs(err)) * 100))


# ══════════ 한 칸 계산 ══════════
def run_one(d, a, b, lo, hi, tg, frac, cost, o, c):
    """Window.run 과 같은 계산 — 가격 배열(o: 체결가, c: 평가가)만 바꿀 수 있음"""
    sim = simulate_weights(tg, o, c, cost, a, b) if frac else \
        simulate(tg, o, c, cost, a, b, forced_hold=d.forced_hold)
    days, eq = daily_marks(d.ts, sim["mark"], a, b, lo=lo, hi=hi)
    pos = sim["pos"][a:b]
    prev = np.concatenate([[0.0], pos[:-1]])
    g = c[a:b] / np.concatenate([[c[a]], c[a:b - 1]])     # 봉 사이 가격 변동으로 흘러간 비중 (회전율용 근사)
    den = prev * g + (1.0 - prev)
    w_pre = np.where(den > 0, prev * g / np.where(den > 0, den, 1.0), 0.0)
    traded = np.abs(pos - w_pre)
    return dict(days=days, eq=eq, r=eq[1:] / eq[:-1] - 1.0, pos=pos, traded=traded)


def metrics(res):
    s = S.summary(res["r"])
    years = len(res["r"]) / 365.0
    return dict(sharpe=s["sharpe"], cagr=s["cagr"], max_dd=s["max_dd"], twm=s["twm"],
                exposure=float(res["pos"].mean()),
                turnover=float(res["traded"].sum() / years) if years > 0 else float("nan"),
                n_trades=int(np.sum(res["traded"] > MIN_TRADE)))


def boot(ra, rb):
    return S.stationary_bootstrap_diff(ra, rb, mean_block=MEAN_BLOCK, n_boot=N_BOOT, seed=BOOT_SEED)


# ══════════ 한 구간 ══════════
def apply_timing(d, name, tg, timing, cuts):
    if timing == "booked":
        return tg
    if timing == "exec":
        return exec_targets(d, tg, cuts if name in TRAINED else None)
    return lag_targets(tg, int(timing[-1]))


def build_series(d, a, b, tag, loader=load_run):
    series = {k: [v] for k, v in rule_targets(d, a, b, decision_mask(d, 6)).items()}
    series.update(model_targets(d, a, b, tag, loader))
    return series


def evaluate_window(W, ph, df15, krw, tag, loader=load_run, do_boot=True):
    d = W.datas[0]
    a, b = W.rng[0]
    lo, hi = W.lo, W.hi
    series = build_series(d, a, b, tag, loader)      # 이름 → [(목표, 비율)] (반복)
    cuts = month_cuts(ph, lo, hi)
    late = {pd.Timestamp(T, unit="s", tz="UTC").strftime("%Y-%m"): int((c - T) // DAY)
            for T, c in cuts.items() if c - T != DAY}

    o_ex, ex_info = exec_open(d, df15)
    fac, kinfo = krw_factor(d, df15, krw)
    if not np.all(np.isfinite(fac[a:b + 2])):
        raise ValueError("구간 안에 원화 배수가 없는 봉이 있습니다")
    kinfo["mark_check_vs_upbit_close"] = krw_mark_check(d, fac, krw, lo, hi)
    px = {("USD", False): (d.o, d.c), ("USD", True): (o_ex, d.c),
          ("KRW", False): (d.o * fac, d.c * fac), ("KRW", True): (o_ex * fac, d.c * fac)}

    def timed(name, tg, timing):
        return apply_timing(d, name, tg, timing, cuts)

    cells = {}
    rules_exec_with_cut = {}
    for cur in CURRENCIES:
        for timing in TIMINGS:
            o, c = px[(cur, timing == "exec")]
            for cost in COSTS:
                key = f"{cur}|{timing}|{cost:.4f}"
                res = {}
                for name, reps in series.items():
                    res[name] = [run_one(d, a, b, lo, hi, timed(name, tg, timing), frac, cost, o, c)
                                 for tg, frac in reps]
                bh = res["B0"][0]
                sb = S.sharpe(bh["r"])
                sb5 = S.sharpe(res["B5"][0]["r"])
                cell = {}
                for name in ORDER:
                    rr = res[name]
                    shs = [S.sharpe(x["r"]) for x in rr]
                    if name in MODELS and MODELS[name][1] > 1:
                        hi_rep = int(headline_index(shs[:MODELS[name][1]], sb))
                    else:
                        hi_rep = 0
                    m = metrics(rr[hi_rep])
                    m.update(headline_rep=hi_rep, d_vs_bh=m["sharpe"] - sb, d_vs_b5=m["sharpe"] - sb5)
                    if len(shs) > 1:
                        m["sharpe_reps"] = shs
                    if do_boot and name != "B0":
                        bt = boot(rr[hi_rep]["r"], bh["r"])
                        m.update(p_vs_bh=bt["p"], ci90_vs_bh=list(bt["ci90"]), se_vs_bh=bt["se"])
                    cell[name] = m
                cells[key] = cell
                if timing == "exec":                 # 규칙에 월초 지연을 넣었다면 (등록 문구의 다른 읽기) — 참고
                    alt = {}
                    for name in RULES:
                        tg, frac = series[name][0]
                        x = run_one(d, a, b, lo, hi, exec_targets(d, tg, cuts), frac, cost, o, c)
                        alt[name] = S.sharpe(x["r"]) - cell[name]["sharpe"]
                    rules_exec_with_cut[key] = alt
    n_days = len(run_one(d, a, b, lo, hi, series["B0"][0][0], False, COSTS[0], d.o, d.c)["r"])
    return dict(range=[lo, hi], n_days=int(n_days), month_cut_not_T_plus_1=late, exec_price=ex_info, krw=kinfo,
                rules_exec_with_month_cut_d_sharpe=rules_exec_with_cut, cells=cells)


# ══════════ 사후 점검 (등록 밖 — 결과를 본 뒤 추가, 주 결과를 바꾸지 않음) ══════════
POSTHOC_CELLS = (("booked", 0.0015), ("exec", 0.0005))
SUBPERIODS = (("2017-01-01", "2019-01-01"), ("2019-01-01", "2026-09-25"))


def day_factor(krw_col, d, when):
    """when: 'open' → 봉이 시작한 날의 값, 'close' → 봉이 마감한 시각의 날 값 (00:00 마감이면 다음 날)"""
    t = d.ts if when == "open" else d.ts + BAR_SEC
    days = pd.to_datetime((t // DAY) * DAY, unit="s", utc=True)
    return krw_col.reindex(days).to_numpy(dtype=float)


def posthoc(W, ph, df15, krw, tag, loader=load_run, do_boot=True):
    """
    원화 결과가 근사 방식이나 한 시기에 기대는지 점검:
      exact_marks — 평가가(종가)에 마감 시각의 날 k를 씀 → 매일 00:00 평가가 = 업비트 시가 (근사 오차 없음)
      fx_only     — 김치프리미엄 없이 환율만 (k = 원/달러)
      sub-period  — 2017~2018(프리미엄 거품 포함) / 2019~2026-09 로 나눈 매수·보유 대비 샤프 차이 (등록 원화·USD)
    """
    d = W.datas[0]
    a, b = W.rng[0]
    lo, hi = W.lo, W.hi
    series = build_series(d, a, b, tag, loader)
    cuts = month_cuts(ph, lo, hi)
    o_ex, _ = exec_open(d, df15)
    bs = daily_open(df15)
    k = (krw["krw_open"] / bs.reindex(krw.index)).ffill()
    fx = krw["usdkrw_0000"].ffill()
    k_open, k_close = day_factor(k, d, "open"), day_factor(k, d, "close")
    f_open = day_factor(fx, d, "open")
    pricing = {
        "USD": lambda ex: ((o_ex if ex else d.o), d.c),
        "KRW": lambda ex: ((o_ex if ex else d.o) * k_open, d.c * k_open),
        "KRW_exact_marks": lambda ex: ((o_ex if ex else d.o) * k_open, d.c * k_close),
        "KRW_fx_only": lambda ex: ((o_ex if ex else d.o) * f_open, d.c * f_open),
    }
    out = {}
    for pname, pf in pricing.items():
        for timing, cost in POSTHOC_CELLS:
            o, c = pf(timing == "exec")
            if not (np.all(np.isfinite(o[a:b + 2])) and np.all(np.isfinite(c[a:b + 2]))):
                raise ValueError(f"{pname}: 구간 안에 가격 배수가 없는 봉")
            res = {n: [run_one(d, a, b, lo, hi, apply_timing(d, n, tg, timing, cuts), fr, cost, o, c)
                       for tg, fr in reps] for n, reps in series.items()}
            bh = res["B0"][0]
            sb = S.sharpe(bh["r"])
            cell = {}
            for n in ORDER:
                rr = res[n]
                shs = [S.sharpe(x["r"]) for x in rr]
                hr = int(headline_index(shs[:MODELS[n][1]], sb)) if n in MODELS and MODELS[n][1] > 1 else 0
                x = rr[hr]
                m = dict(sharpe=shs[hr], d_vs_bh=shs[hr] - sb, headline_rep=hr,
                         max_dd=S.max_drawdown(x["r"]), twm=float(np.prod(1.0 + x["r"])))
                if do_boot and n != "B0":
                    m["p_vs_bh"] = boot(x["r"], bh["r"])["p"]
                if pname in ("USD", "KRW"):
                    subs = {}
                    days = bh["days"][1:]
                    for s0, s1 in SUBPERIODS:
                        sel = (days > _ts(s0)) & (days <= _ts(s1))
                        subs[f"{s0[:4]}-{s1[:4]}"] = dict(
                            d_vs_bh=S.sharpe(x["r"][sel]) - S.sharpe(bh["r"][sel]),
                            sharpe=S.sharpe(x["r"][sel]), bh_sharpe=S.sharpe(bh["r"][sel]),
                            p_vs_bh=(boot(x["r"][sel], bh["r"][sel])["p"] if do_boot and n != "B0" else None))
                    m["subperiods"] = subs
                cell[n] = m
            out[f"{pname}|{timing}|{cost:.4f}"] = cell
    return out


def _ts(s):
    return int(pd.Timestamp(s, tz="UTC").timestamp())


def ref_check(win_key, cells):
    out = {}
    cell = cells[f"USD|booked|{COSTS[0]:.4f}"]
    for name, (v, digits) in REF_CHECK[win_key].items():
        got = cell[name]["sharpe"]
        out[name] = dict(registered=v, computed=float(got), matches=bool(round(float(got), digits) == v))
    return out


def evaluate(loader=load_run, datas=None, df15=None, krw=None, do_boot=True):
    if datas is None:
        from ..walkforward import load_phases
        datas = load_phases()
    if df15 is None:
        from ..data import load_15m
        df15 = load_15m()
    krw = load_krw() if krw is None else krw
    wins = {}
    for key, (lo, hi, tag, what) in WINDOWS.items():
        W = Window(datas, lo, hi, what)             # 2017~2026 은 BTC_LOCKBOX_OPEN=1 필요 (evaluate.guard)
        w = evaluate_window(W, datas, df15, krw, tag, loader, do_boot)
        w["ref_check"] = ref_check(key, w["cells"])
        if key == "main":
            w["posthoc"] = posthoc(W, datas, df15, krw, tag, loader, do_boot)
        wins[key] = w
    return dict(registered="success_criteria.md '현실 점검 (2026-09-28 02:10 UTC 등록)'",
                costs=list(COSTS), timings=list(TIMINGS), currencies=list(CURRENCIES), n_boot=N_BOOT,
                boot_seed=BOOT_SEED, mean_block=MEAN_BLOCK, c_dec=0.003, windows=wins)


# ══════════ 출력 ══════════
def _clean(x):
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


TIMING_KO = {"booked": "등록(00:00)", "exec": "실행 가능(00:30)", "lag1": "하루 지연", "lag2": "이틀 지연"}
NAME_KO = dict(RULE_KO, R6="R6", R3="R3", W2="W2", C1="C1")


def report(out):
    pct = lambda x: "—" if x is None else f"{x * 100:.1f}%"
    lines = [f"현실 점검 (새 시험 아님) — 부트스트랩 {out['n_boot']}번·시드 {out['boot_seed']}·평균 블록 {out['mean_block']}"]
    for wk, w in out["windows"].items():
        rc = w["ref_check"]
        bad = [k for k, v in rc.items() if not v["matches"]]
        lines.append(f"\n[{WINDOW_KO[wk]}] 등록 계산 재현: " + ("모두 일치" if not bad else "다름! " + ", ".join(bad)))
        lines.append(f"  원화: 앞 날짜로 채운 날 {w['krw']['days_forward_filled']}, "
                     f"평가가 근사 오차(업비트 종가 대비) {w['krw']['mark_check_vs_upbit_close']}")
        lines.append(f"  00:30 체결가: {w['exec_price']}; c(T)≠T+1일 달: {w['month_cut_not_T_plus_1'] or '없음'}")
        for key, cell in w["cells"].items():
            cur, timing, cost = key.split("|")
            lines.append(f"\n  {cur} · {TIMING_KO[timing]} · 편도 {float(cost) * 100:.2f}%")
            lines.append(f"    {'이름':<16}{'샤프':>6}{'보유 대비':>9}{'p':>6}{'MACD 대비':>10}{'CAGR':>8}{'최대낙폭':>9}"
                         f"{'배수':>8}{'노출':>6}")
            for name in ORDER:
                m = cell[name]
                p = m.get("p_vs_bh")
                lines.append(f"    {NAME_KO[name]:<16}{m['sharpe']:>6.2f}{m['d_vs_bh']:>+9.2f}"
                             f"{'' if p is None else f'{p:.2f}':>6}{m['d_vs_b5']:>+10.2f}{pct(m['cagr']):>8}"
                             f"{pct(m['max_dd']):>9}{m['twm']:>8.1f}{pct(m['exposure']):>6}")
        ph = w.get("posthoc")
        if ph:
            lines.append("\n  [사후 점검 — 등록 밖, 결과를 본 뒤 추가] 원화 근사·시기 의존성")
            for key, cell in ph.items():
                lines.append(f"    {key}: " + ", ".join(
                    f"{NAME_KO[n].split(' ')[0]} {cell[n]['d_vs_bh']:+.2f}"
                    + (f"(p {cell[n]['p_vs_bh']:.2f})" if cell[n].get("p_vs_bh") is not None else "")
                    for n in ORDER if n != "B0"))
                if "subperiods" in cell["R6"]:
                    for sp in cell["R6"]["subperiods"]:
                        lines.append(f"      {sp}: " + ", ".join(
                            f"{NAME_KO[n].split(' ')[0]} {cell[n]['subperiods'][sp]['d_vs_bh']:+.2f}" for n in ORDER
                            if n != "B0") + f" (보유 샤프 {cell['R6']['subperiods'][sp]['bh_sharpe']:.2f})")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="현실 점검 (원화·실행 시각·지연·비용 민감도)")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--no-boot", action="store_true", help="부트스트랩 생략 (빠른 점검)")
    args = ap.parse_args()
    out = _clean(evaluate(do_boot=not args.no_boot))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    os.replace(tmp, args.out)
    print(report(out))


if __name__ == "__main__":
    main()
