# -*- coding: utf-8 -*-
"""
지표 시계열을 종목당 한 번만 계산하고, 임의의 날짜에서 값을 꺼내 쓰는 계층.

날짜마다 지표를 새로 계산하면 소급 실행이 감당이 안 됩니다
(350종목 × 120일 = 4만 회). 지표는 모두 누적 계산이라 전체 구간을
한 번 돌려두면 각 날짜 값은 인덱스로 바로 꺼낼 수 있습니다.
"""
import indi
from rules import SELL


def build_series(o, h, l, c, v):
    """종목 하나의 전체 지표 시계열. 데이터가 짧으면 None"""
    n = len(c)
    if n < 130:
        return None
    gb, gs = indi.golden(c, 32)
    T, Ts = indi.tsi(c)
    B = indi.rsi(c, 35)
    U, Us = indi.ultimate(h, l, c)
    mt, mb, red = indi.mesh(c)
    ma30 = indi.sma(c, SELL["MA_BREAK"])
    return dict(n=n, o=o, h=h, l=l, c=c, v=v,
                gb=gb, gs=gs, T=T, B=B, U=U, red=red, ma30=ma30)


def _hold_at(gs, i, lookback):
    """i일 기준 청선이 5선 이하를 며칠 연속 지켰는지"""
    hold = 0
    for j in range(i, max(i - lookback, -1), -1):
        if gs[j] <= indi.CFG["GS_MAX"]:
            hold += 1
        else:
            break
    return hold


def snapshot_at(S, i):
    """i일 기준 지표 묶음 — 매수·매도 판정과 로그에 공통으로 씁니다"""
    if i < 120 or i >= S["n"]:
        return None
    c, o, h, l, v = S["c"], S["o"], S["h"], S["l"], S["v"]
    gb, gs, ma30 = S["gb"], S["gs"], S["ma30"]
    if None in (S["T"][i], S["B"][i], S["U"][i], ma30[i]):
        return None

    hold = _hold_at(gs, i, indi.CFG["LOOKBACK"])
    seg = gs[i - hold + 1:i + 1] if hold else [gs[i]]

    return dict(
        close=int(c[i]), open=int(o[i]), high=int(h[i]), low=int(l[i]),
        gbuy=round(gb[i], 1), gsell=round(gs[i], 1),
        gb_slope5=round(gb[i] - gb[i - 5], 1),
        gs_hold=hold,
        gs_mean=round(sum(seg) / len(seg), 2), gs_std=round(indi.stdev(seg), 2),
        t=round(S["T"][i], 1), b=round(S["B"][i], 1), u=round(S["U"][i], 1),
        red=bool(S["red"][i]), ma30=int(ma30[i]),
        above_ma30=bool(c[i] >= ma30[i]),
        chg1=round((c[i] / c[i - 1] - 1) * 100, 2),
        chg5=round((c[i] / c[i - 5] - 1) * 100, 2),
        chg20=round((c[i] / c[i - 20] - 1) * 100, 2) if i >= 20 else 0.0,
        wick=round(indi.wick(o[i], h[i], l[i], c[i]), 2),
        vol=int(v[i]), amount=int(c[i] * v[i]),
    )


def buy_at(S, i):
    """i일 기준 매수 판정 — indi.evaluate()와 같은 규칙, 인덱스 기반"""
    if i < 120 or i >= S["n"]:
        return None
    C = indi.CFG
    c, o, h, l = S["c"], S["o"], S["h"], S["l"]
    gb, gs, ma30 = S["gb"], S["gs"], S["ma30"]
    if None in (S["T"][i], S["B"][i], S["U"][i], ma30[i]):
        return None

    hold = _hold_at(gs, i, C["LOOKBACK"])
    if hold < C["GS_DAYS"]:
        return None

    seg = gs[i - hold + 1:i + 1] if hold else [gs[i]]
    gs_mean = sum(seg) / len(seg)
    gs_std = indi.stdev(seg)
    if gs_std > C["GS_STD_MAX"]:
        return None

    look = gs[max(0, i - C["LOOKBACK"] + 1):i - hold + 1] if hold < C["LOOKBACK"] else []
    spike = bool(look and max(look) > C["GS_SPIKE"])

    if gb[i] < C["GB_MIN"]:
        return None
    slope5 = gb[i] - gb[i - 5]
    if slope5 < C["GB_SLOPE_MIN"]:
        return None

    bad_wick = any(indi.wick(o[j], h[j], l[j], c[j]) > C["WICK_MAX"] and h[j] > c[j]
                   for j in range(i - C["WICK_DAYS"] + 1, i + 1))
    big_down = any((c[j] / c[j - 1] - 1) * 100 <= C["BIG_DOWN"]
                   for j in range(i - C["WICK_DAYS"] + 1, i + 1))
    if bad_wick or big_down:
        return None
    if c[i] < ma30[i]:
        return None

    score = 0.0
    score += 22 * min(1, hold / C["GS_DAYS_GOOD"])
    score += 20 * max(0, (C["GS_MAX"] - gs_mean) / C["GS_MAX"])
    score += 14 * max(0, 1 - gs_std / C["GS_STD_MAX"])
    score += 16 * min(1, slope5 / 12)
    score += 8 * min(1, max(0, (gb[i] - 10) / 30))
    if not spike:
        score += 6
    if S["red"][i]:
        score += 4
    if S["U"][i] >= 50:
        score += 5
    if S["T"][i] >= 0:
        score += 5

    return dict(
        score=round(min(100, max(0, score)), 1),
        hold=hold, gs_mean=round(gs_mean, 2), gs_std=round(gs_std, 2),
        gbuy=round(gb[i], 1), gsell=round(gs[i], 1), slope5=round(slope5, 1),
        t=round(S["T"][i], 1), b=round(S["B"][i], 1), u=round(S["U"][i], 1),
        red=bool(S["red"][i]), ma30=int(ma30[i]), spike=spike,
        wick=bad_wick, big_down=big_down,
        r20=round((c[i] / c[i - 20] - 1) * 100, 1) if i >= 20 else 0,
    )
