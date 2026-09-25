# -*- coding: utf-8 -*-
"""
체결 회계와 학습 표본 — 강화학습 보상, 백테스트, 기준전략, 모의매매가 모두 이 계산을 씁니다.

시점 규칙 (이 저장소의 원칙과 같습니다)
  t봉 종가에 지표를 보고 결정 → t+1봉 '시가'에 체결 → t+1봉 시가부터 t+2봉 시가까지 손익.
  t봉 종가로 t봉 종가에 샀다고 치면 실제보다 좋게 나오므로 그렇게 하지 않습니다.

  매수: 수량 = 자산 × (1−c) / O[t+1]      매도: 자산 = 수량 × O[t+1] × (1−c)
  c = 한 방향 거래비용(수수료+슬리피지). 현금 이자는 0.

보상 (강화학습이 최대화하는 값):
  R_t(p, a) = a·m_t + |a−p|·ln(1−c),   m_t = ln(O[t+2]/O[t+1])
  경로를 따라 더하면 정확히 ln(최종 자산)이 됩니다 (tests가 확인).
"""
import math

import numpy as np

BAR_SEC = 4 * 3600
KAPPA = 100.0          # 보상 단위: 로그수익 × 100 (≈ %)


def m_next(o):
    """m[t] = ln(O[t+2]/O[t+1]) — t봉 종가 결정이 버는 로그수익 (끝 두 개는 NaN)"""
    m = np.full(len(o), np.nan)
    m[:-2] = np.log(o[2:] / o[1:-1])
    return m


def simulate(targets, o, c, cost, start, end, forced_hold=None, exit_cost=True):
    """
    한 전략의 백테스트.

    targets[t] ∈ {0, 1, NaN}: t봉 종가에 정한 목표 (NaN = 직전 유지). t ∈ [start, end)만 봅니다.
    forced_hold[t] = True 이면 그 봉에서는 목표를 무시하고 유지합니다.
    반환:
      pos[t]    t봉 결정 후(t+1 시가 체결 후) 보유 (0/1)
      logr[t]   pos[t]·m[t] + |pos[t]−pos[t−1]|·ln(1−c)   (시가→시가 로그수익)
      mark[j]   j봉 종가 기준 자산 (j ∈ [start, end]; mark[start]=1, 첫 체결 전)
                마지막 mark에는 청산 비용을 반영합니다(보유 중이면 ×(1−c)).
      trades    청산된 거래별 수익률 (비용 포함)
    """
    T = len(o)
    if end >= T:
        raise ValueError("end 봉(마지막 결정의 체결 봉)이 있어야 합니다")
    pos = np.zeros(T)
    logr = np.zeros(T)
    mark = np.full(T, np.nan)
    lnc = math.log(1.0 - cost)
    cash, units = 1.0, 0.0
    mark[start] = 1.0
    prev = 0.0
    entry_eq = None
    trades = []
    for t in range(start, end):
        tg = targets[t]
        a = prev if (np.isnan(tg) or (forced_hold is not None and forced_hold[t])) else float(tg)
        j = t + 1                                   # 체결 봉
        if a == 1.0 and prev == 0.0:
            entry_eq = cash
            units = cash * (1.0 - cost) / o[j]
            cash = 0.0
        elif a == 0.0 and prev == 1.0:
            cash = units * o[j] * (1.0 - cost)
            units = 0.0
            trades.append(cash / entry_eq - 1.0)
            entry_eq = None
        pos[t] = a
        if j + 1 < T:
            logr[t] = a * math.log(o[j + 1] / o[j]) + abs(a - prev) * lnc
        mark[j] = cash + units * c[j]
        prev = a
    if exit_cost and prev == 1.0:
        mark[end] *= (1.0 - cost)
        trades.append(mark[end] / entry_eq - 1.0)
    return dict(pos=pos, logr=logr, mark=mark, trades=np.array(trades), start=start, end=end)


def daily_marks(ts, mark, start, end, bar_sec=BAR_SEC):
    """
    00:00 UTC 마다 자산을 찍어 일별 수익률을 만듭니다.
    그 시각 이전에 마감한 마지막 봉의 종가 기준 자산을 씁니다.
    반환 (날짜 배열 [UTC 초], 자산 배열)
    """
    close_t = ts[start:end + 1] + bar_sec
    m = mark[start:end + 1]
    d0 = int(math.ceil(close_t[0] / 86400.0)) * 86400
    d1 = int(close_t[-1] // 86400) * 86400
    days = np.arange(d0, d1 + 1, 86400, dtype=np.int64)
    k = np.searchsorted(close_t, days, side="right") - 1
    return days, m[k]


def daily_returns(ts, mark, start, end):
    days, eq = daily_marks(ts, mark, start, end)
    return days[1:], eq[1:] / eq[:-1] - 1.0


class PhaseData:
    """한 phase의 살아 있는 4시간봉과 특징·보상 재료"""

    def __init__(self, bars, X, sigma):
        self.ts = bars["ts"].to_numpy(np.int64)
        self.o = bars["open"].to_numpy(float)
        self.h = bars["high"].to_numpy(float)
        self.l = bars["low"].to_numpy(float)
        self.c = bars["close"].to_numpy(float)
        self.v = bars["volume"].to_numpy(float)
        self.gap = bars["gap_before"].to_numpy(np.int64)
        self.forced_hold = bars["forced_hold"].to_numpy(bool)
        self.X = X.astype(np.float32)
        self.sigma = sigma
        T = len(self.o)
        self.T = T
        self.m = m_next(self.o)
        # 보조 목표: 앞으로 h봉 수익률(변동성 정규화) — 규제용, 매매 판단에는 안 씀
        self.z = {}
        for hh in (6, 42):
            z = np.full(T, np.nan)
            if T > hh + 1:
                z[:T - 1 - hh] = np.log(self.o[1 + hh:] / self.o[1:T - hh]) / (sigma[:T - 1 - hh] * math.sqrt(hh))
            self.z[hh] = np.clip(z, -4, 4)
        # t, t+1, t+2 사이에 빠진 봉이 없어야 학습 표본으로 씀
        g = self.gap
        self.contig = np.zeros(T, bool)
        self.contig[:-2] = (g[1:-1] == 0) & (g[2:] == 0)
        # 보조 목표 구간(t..t+43)에 빠진 봉이 있으면 보조 손실 제외
        cg = np.concatenate([[0], np.cumsum(g > 0)])
        self.aux_ok = np.zeros(T, bool)
        n = T - 44
        if n > 0:
            self.aux_ok[:n] = (cg[np.arange(n) + 44] - cg[np.arange(n) + 1]) == 0

    def eligible(self, T_k, first_ts, warmup):
        """T_k 시점에 학습에 쓸 수 있는 봉 번호 — t+43봉 종가가 T_k 이전이어야 함 (정보 차단)"""
        t = np.arange(self.T)
        ok = (self.ts >= first_ts) & (t >= warmup) & self.contig & ~self.forced_hold
        ok[self.T - 43:] = False
        ok[:self.T - 43] &= (self.ts[43:] + BAR_SEC) <= T_k
        return np.nonzero(ok)[0]


def simulate_weights(weights, o, c, cost, start, end, band=0.02, exit_cost=True):
    """
    비율 보유(0~1) 전략용 — 기준전략 B7(변동성 목표)·B8(노출도 맞춘 고정비율)에만 씁니다.
    목표 비중과 현재(가격 변동으로 흘러간) 비중 차이가 band를 넘을 때만 거래합니다.
    거래비용 = cost × 거래 금액. 반환 형식은 simulate()와 같습니다.
    """
    T = len(o)
    if end >= T:
        raise ValueError("end 봉이 있어야 합니다")
    pos = np.zeros(T)
    logr = np.zeros(T)
    mark = np.full(T, np.nan)
    cash, units = 1.0, 0.0
    mark[start] = 1.0
    eq_open_prev = None
    for t in range(start, end):
        j = t + 1
        E = cash + units * o[j]
        w_cur = units * o[j] / E if E > 0 else 0.0
        tg = weights[t]
        if not np.isnan(tg) and abs(tg - w_cur) > band:
            V = tg * E - units * o[j]
            if V > 0:
                cash -= V
                units += V * (1.0 - cost) / o[j]
            else:
                units += V / o[j]
                cash += -V * (1.0 - cost)
        E_after = cash + units * o[j]
        pos[t] = units * o[j] / E_after if E_after > 0 else 0.0
        if eq_open_prev is not None:
            logr[t - 1] = math.log(E_after / eq_open_prev)
        eq_open_prev = E_after
        mark[j] = cash + units * c[j]
    if exit_cost and units > 0:
        mark[end] = cash + units * c[end] * (1.0 - cost)
    return dict(pos=pos, logr=logr, mark=mark, trades=np.array([]), start=start, end=end)
