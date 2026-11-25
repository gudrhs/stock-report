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
    (그날 안에서는 일정: 봉 시가는 시작한 날, 종가는 마감 시각이 속한 날의 k → 매일 00:00 평가가 = 원화 시가.
     원화 시가: 2017-10-23까지 코빗 00:00 직전 체결가, 2017-10-24부터 업비트 일봉 시가 — 이음 날 약 +0.7% 한 번 뜀)
    처음 구현(커밋 64ddc4d)은 00:00 평가가에 전날 k 를 붙인 버그였고, 그 방식은 사후 점검 KRW_start_day_marks 로만 남김
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
    그 봉이 비었으면 day + max_wait 전 첫 체결 가능한 15분봉 시가, 그래도 없으면 NaN.
    반환 (가격, 실제로 쓴 15분봉 시작 시각 — 없으면 −1)
    """
    ts = df15["ts"].to_numpy(dtype=np.int64)
    op = df15["open"].to_numpy(dtype=float)
    ok = np.isfinite(op)
    ts_ok, op_ok = ts[ok], op[ok]
    want = np.asarray(day_ts, dtype=np.int64) + offset
    i = np.searchsorted(ts_ok, want)
    px = np.full(len(want), np.nan)
    src = np.full(len(want), -1, dtype=np.int64)
    for k, (w, j) in enumerate(zip(want, i)):
        if j < len(ts_ok) and ts_ok[j] < w - offset + max_wait:
            px[k] = op_ok[j]
            src[k] = ts_ok[j]
    return px, src


def exec_open(d, df15):
    """
    00:00 에 시작하는 phase 0 봉의 시가를 00:30 15분봉 시가로 바꾼 배열 (다른 봉은 그대로).
    반환 (배열, 체결 시각 배열 — 봉마다 실제로 쓴 가격의 시각. 바꾸지 않은 봉은 봉 시작 시각)
    """
    o = d.o.copy()
    at = d.ts.copy()
    j = np.nonzero(d.ts % DAY == 0)[0]
    px, src = open_at(df15, d.ts[j])
    ok = np.isfinite(px)
    o[j[ok]] = px[ok]
    at[j[ok]] = src[ok]
    return o, at


def exec_info(d, at, a, b, lo, hi):
    """구간이 실제로 쓰는 체결 봉(a+1..b) 가운데 00:00 시작 봉의 00:30 체결가 사정 + 00:00 봉이 죽은 날"""
    j = np.arange(a + 1, b + 1)
    j = j[d.ts[j] % DAY == 0]
    sub = at[j] != d.ts[j] + EXEC_OFFSET
    missing = at[j] == d.ts[j]
    days = np.arange((lo // DAY) * DAY, hi, DAY)
    starts = set(int(x) for x in d.ts[(d.ts >= lo) & (d.ts < hi) & (d.ts % DAY == 0)])
    dead = [pd.Timestamp(int(x), unit="s", tz="UTC").strftime("%Y-%m-%d") for x in days if int(x) not in starts]
    return dict(n_bars=int(len(j)), n_substituted=int(np.sum(sub & ~missing)), n_missing=int(missing.sum()),
                substituted=[pd.Timestamp(int(t), unit="s", tz="UTC").strftime("%Y-%m-%d %H:%M")
                             for t in at[j][sub & ~missing]],
                dead_midnight_days=dead)


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
    원화 배수 k(D) = 원화 시가(D) ÷ 비트스탬프 00:00 시가(D) — 00:00 UTC 의 비율, 그날(D) 안에서는 일정.
    반환 dict:
      open  — 봉 시가에 쓰는 k (봉이 시작한 날)
      close — 봉 종가에 쓰는 k (종가 시각이 속한 날: 00:00 에 마감하는 봉은 다음 날 k → 매일 00:00 평가가 = 원화 시가)
      filled_open / filled_close — 그 k 가 없어 직전 값을 쓴 봉 (원화 값이나 비트스탬프 00:00 시가가 없는 날)
      start_day_close — 처음 구현의 종가 배수 (봉이 시작한 날의 k, 00:00 평가가에 전날 k) — 사후 점검용
    """
    bs = daily_open(df15)
    raw = (krw["krw_open"] / bs.reindex(krw.index)).rename("k")
    k = raw.ffill()                                 # 원화 자료 시작 전 날짜는 NaN 그대로 (구간 안에서만 쓰는지 확인)

    def by(t):
        days = pd.to_datetime((t // DAY) * DAY, unit="s", utc=True)
        return k.reindex(days).to_numpy(dtype=float), raw.reindex(days).isna().to_numpy()

    fo, mo = by(d.ts)
    fc, mc = by(d.ts + BAR_SEC)
    return dict(open=fo, close=fc, filled_open=mo, filled_close=mc, start_day_close=fo)


def krw_info(d, fac, krw, a, b, lo, hi):
    """구간이 쓰는 봉(a..b+1)에서 직전 k 로 채운 날 + 00:00 평가가(종가 × k)와 원화 시가의 차이 (근사 오차)"""
    sl = np.arange(a, min(b + 2, d.T))
    used = [(d.ts[sl], fac["filled_open"][sl]), (d.ts[sl] + BAR_SEC, fac["filled_close"][sl])]
    filled = sorted({pd.Timestamp(int((t // DAY) * DAY), unit="s", tz="UTC").strftime("%Y-%m-%d")
                     for ts_, m in used for t in ts_[m]})
    close = d.ts + BAR_SEC
    j = np.nonzero((close % DAY == 0) & (close > lo) & (close <= hi))[0]
    days = pd.to_datetime(close[j], unit="s", utc=True)
    ref = krw["krw_open"].reindex(days).to_numpy(dtype=float)
    ok = np.isfinite(ref)
    out = dict(days_forward_filled=len(filled), forward_filled=filled)
    if ok.any():
        err = d.c[j[ok]] * fac["close"][j[ok]] / ref[ok] - 1.0
        out["mark_vs_krw_open"] = dict(n=int(ok.sum()), mean_abs_pct=float(np.mean(np.abs(err)) * 100),
                                       max_abs_pct=float(np.max(np.abs(err)) * 100))
    return out


# ══════════ 한 칸 계산 ══════════
def run_one(d, a, b, lo, hi, tg, frac, cost, o, c):
    """Window.run 과 같은 계산 — 가격 배열(o: 체결가, c: 평가가)만 바꿀 수 있음"""
    sim = simulate_weights(tg, o, c, cost, a, b) if frac else \
        simulate(tg, o, c, cost, a, b, forced_hold=d.forced_hold)
    days, eq = daily_marks(d.ts, sim["mark"], a, b, lo=lo, hi=hi)
    pos = sim["pos"][a:b]
    prev = np.concatenate([[0.0], pos[:-1]])
    g = o[a + 1:b + 1] / o[a:b]                     # 직전 체결 뒤 체결가 변동으로 흘러간 비중 (paper.window_run 과 같음)
    den = prev * g + (1.0 - prev)
    w_pre = np.where(den > 0, prev * g / np.where(den > 0, den, 1.0), 0.0)
    traded = np.abs(pos - w_pre)
    return dict(days=days, eq=eq, r=eq[1:] / eq[:-1] - 1.0, pos=pos, w_pre=w_pre, traded=traded, a=a, b=b)


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

    o_ex, at_ex = exec_open(d, df15)
    fac = krw_factor(d, df15, krw)
    fo, fc = fac["open"], fac["close"]
    if not (np.all(np.isfinite(fo[a:b + 2])) and np.all(np.isfinite(fc[a:b + 2]))):
        raise ValueError("구간 안에 원화 배수가 없는 봉이 있습니다")
    ex_info = exec_info(d, at_ex, a, b, lo, hi)
    kinfo = krw_info(d, fac, krw, a, b, lo, hi)
    px = {("USD", False): (d.o, d.c), ("USD", True): (o_ex, d.c),
          ("KRW", False): (d.o * fo, d.c * fc), ("KRW", True): (o_ex * fo, d.c * fc)}

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
                    if frac_of(series, name):                # 비율 회계: 실제 재조정 수 (simulate_weights rebal 과 같음)
                        m["fractional"] = True
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


def frac_of(series, name):
    return bool(series[name][0][1])


# ══════════ 사후 점검 (등록 밖 — 결과를 본 뒤 추가, 주 결과를 바꾸지 않음) ══════════
POSTHOC_CELLS = (("booked", 0.0015), ("exec", 0.0005))
SUBPERIODS = (("2017-2018", "2017-01-01", "2019-01-01", False),
              ("2019-2026.09", "2019-01-01", "2026-09-25", False),
              ("excl 2017-12..2018-03", "2017-12-01", "2018-04-01", True))      # True = 그 구간을 뺀 나머지


def day_factor(krw_col, d, when):
    """when: 'open' → 봉이 시작한 날의 값, 'close' → 봉이 마감한 시각의 날 값 (00:00 마감이면 다음 날)"""
    t = d.ts if when == "open" else d.ts + BAR_SEC
    days = pd.to_datetime((t // DAY) * DAY, unit="s", utc=True)
    return krw_col.reindex(days).to_numpy(dtype=float)


def posthoc(W, ph, df15, krw, tag, loader=load_run, do_boot=True):
    """
    원화 결과가 계산 방식이나 한 시기에 기대는지 점검 (등록 밖):
      KRW                 — 등록 계산 (00:00 평가가 = 그날 k)
      KRW_start_day_marks — 처음 구현의 평가가 (00:00 평가가에 전날 k — 등록 정의와 다른 버그였던 방식)
      KRW_fx_only         — 김치프리미엄 없이 환율만 (k = 원/달러)
      기간 나눔           — 2017~2018 / 2019~2026-09 / 2017-12~2018-03 을 뺀 나머지 (USD·KRW)
    """
    d = W.datas[0]
    a, b = W.rng[0]
    lo, hi = W.lo, W.hi
    series = build_series(d, a, b, tag, loader)
    cuts = month_cuts(ph, lo, hi)
    o_ex, _ = exec_open(d, df15)
    fac = krw_factor(d, df15, krw)
    fx = krw["usdkrw_0000"].ffill()
    f_open, f_close = day_factor(fx, d, "open"), day_factor(fx, d, "close")
    pricing = {
        "USD": lambda ex: ((o_ex if ex else d.o), d.c),
        "KRW": lambda ex: ((o_ex if ex else d.o) * fac["open"], d.c * fac["close"]),
        "KRW_start_day_marks": lambda ex: ((o_ex if ex else d.o) * fac["open"], d.c * fac["start_day_close"]),
        "KRW_fx_only": lambda ex: ((o_ex if ex else d.o) * f_open, d.c * f_close),
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
                if len(shs) > 1:
                    m["sharpe_reps"] = shs
                if do_boot and n != "B0":
                    m["p_vs_bh"] = boot(x["r"], bh["r"])["p"]
                if pname in ("USD", "KRW"):
                    subs = {}
                    days = bh["days"][1:]                  # r[i] 는 days[i+1] 에 끝나는 하루
                    for label, s0, s1, excl in SUBPERIODS:
                        sel = (days > _ts(s0)) & (days <= _ts(s1))
                        if excl:
                            sel = ~sel
                        subs[label] = dict(
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
                boot_seed=BOOT_SEED, mean_block=MEAN_BLOCK, c_dec=0.003,
                multiplicity=multiplicity(wins) if do_boot else None, notes=NOTES, windows=wins)


NOTES = {
    "p_values": "칸마다 등록대로 보정 없이 적은 단측 p (매수·보유 대비). 같은 판단의 결정론적 변환이라 칸끼리 독립이 아님 — "
                "multiplicity 의 BH·Holm 은 참고용",
    "B80": "B80(늘 80%, 2%p 띠)은 매수·보유와 일별 상관이 약 0.9999라 부트스트랩 표준오차가 약 0.004 — "
           "±0.01 차이도 p≈0/1 로 나오므로 p값을 근거로 읽지 않음",
    "krw_marks": "원화 평가가는 매일 00:00 에 그날 k (= 원화 시가). 처음 구현(커밋 64ddc4d)은 00:00 평가가에 전날 k 를 붙인 "
                 "버그였음 — 그 값은 posthoc 의 KRW_start_day_marks",
    "rules_exec": "규칙은 학습이 없어 실행 가능 시각에 월초 c(T) 지연을 넣지 않음. 넣었을 때의 차이는 "
                  "rules_exec_with_month_cut_d_sharpe",
}


def multiplicity(wins):
    """등록 칸 p값 전체에 대한 참고용 BH q 최솟값·Holm 최솟값, 칸 안 Holm (보정은 등록 밖)"""
    ps, where = [], []
    for wk, w in wins.items():
        for key, cell in w["cells"].items():
            for n in ORDER:
                p = cell[n].get("p_vs_bh")
                if p is not None:
                    ps.append(float(p))
                    where.append((wk, key, n))
    ps = np.array(ps)
    m = len(ps)
    order = np.argsort(ps)
    bh = np.empty(m)
    run = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        run = min(run, ps[i] * m / (rank + 1))
        bh[i] = run
    holm = np.minimum(1.0, (m - np.arange(m)) * ps[order])
    holm = np.maximum.accumulate(holm)
    best = order[:5]
    return dict(n=int(m), n_below_05=int((ps < 0.05).sum()), n_below_01=int((ps < 0.01).sum()),
                min_bh_q=float(bh.min()), min_holm=float(holm[0]),
                smallest=[dict(window=where[i][0], cell=where[i][1], name=where[i][2], p=float(ps[i]), bh_q=float(bh[i]))
                          for i in best])


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
    short = lambda n: NAME_KO[n].split(" ")[0]
    lines = [f"현실 점검 (새 시험 아님) — 부트스트랩 {out['n_boot']}번·시드 {out['boot_seed']}·평균 블록 {out['mean_block']}"]
    for k, v in out.get("notes", {}).items():
        lines.append(f"  주의[{k}]: {v}")
    mu = out.get("multiplicity")
    if mu:
        lines.append(f"  p값 {mu['n']}개 가운데 0.05 미만 {mu['n_below_05']}, 0.01 미만 {mu['n_below_01']} — "
                     f"참고용 보정: BH q 최소 {mu['min_bh_q']:.3f}, Holm 최소 {mu['min_holm']:.3f}")
    for wk, w in out["windows"].items():
        rc = w["ref_check"]
        bad = [k for k, v in rc.items() if not v["matches"]]
        lines.append(f"\n[{WINDOW_KO[wk]}] 등록 계산 재현: " + ("모두 일치" if not bad else "다름! " + ", ".join(bad)))
        kr, ex = w["krw"], w["exec_price"]
        lines.append(f"  원화: 직전 k로 채운 날 {kr['days_forward_filled']} {kr['forward_filled']}, "
                     f"00:00 평가가 대 원화 시가 {kr.get('mark_vs_krw_open')}")
        lines.append(f"  00:30 체결가: 체결 봉 {ex['n_bars']}, 다른 15분봉으로 대체 {ex['n_substituted']} {ex['substituted']}, "
                     f"없음 {ex['n_missing']}, 00:00 봉이 죽은 날 {ex['dead_midnight_days']}; "
                     f"c(T)≠T+1일 달: {w['month_cut_not_T_plus_1'] or '없음'}")
        alt = w.get("rules_exec_with_month_cut_d_sharpe") or {}
        mx = max((abs(v) for c in alt.values() for v in c.values()), default=0.0)
        lines.append(f"  규칙에 월초 c(T) 지연을 넣으면 샤프 변화 최대 {mx:.3f} (학습이 없어 넣지 않음)")
        for key, cell in w["cells"].items():
            cur, timing, cost = key.split("|")
            lines.append(f"\n  {cur} · {TIMING_KO[timing]} · 편도 {float(cost) * 100:.2f}%")
            lines.append(f"    {'이름':<16}{'샤프':>6}{'보유 대비':>9}{'p':>7}{'MACD 대비':>10}{'CAGR':>8}{'최대낙폭':>9}"
                         f"{'배수':>8}{'노출':>8}{'회전/년':>8}")
            for name in ORDER:
                m = cell[name]
                p = m.get("p_vs_bh")
                lines.append(f"    {NAME_KO[name]:<16}{m['sharpe']:>6.2f}{m['d_vs_bh']:>+9.2f}"
                             f"{'' if p is None else f'{p:.3f}':>7}{m['d_vs_b5']:>+10.2f}{pct(m['cagr']):>8}"
                             f"{pct(m['max_dd']):>9}{m['twm']:>8.1f}{pct(m['exposure']):>8}{m['turnover']:>8.1f}")
            r6 = cell["R6"]
            sb = cell["B0"]["sharpe"]
            if r6.get("sharpe_reps"):
                rr = r6["sharpe_reps"]
                lines.append(f"    R6 반복 0~{len(rr) - 1} 보유 대비 범위 {min(rr) - sb:+.2f} ~ {max(rr) - sb:+.2f} "
                             f"(대표 반복 {r6['headline_rep']})")
        ph = w.get("posthoc")
        if ph:
            lines.append("\n  [사후 점검 — 등록 밖, 결과를 본 뒤 추가] 원화 계산 방식·환율만·기간 의존성")
            for key, cell in ph.items():
                lines.append(f"    {key}: " + ", ".join(
                    f"{short(n)} {cell[n]['d_vs_bh']:+.2f}"
                    + (f"(p {cell[n]['p_vs_bh']:.3f})" if cell[n].get("p_vs_bh") is not None else "")
                    for n in ORDER if n != "B0"))
                if "subperiods" in cell["R6"]:
                    for sp in cell["R6"]["subperiods"]:
                        lines.append(f"      {sp}: " + ", ".join(
                            f"{short(n)} {cell[n]['subperiods'][sp]['d_vs_bh']:+.2f}" for n in ORDER
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
