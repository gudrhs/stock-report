# -*- coding: utf-8 -*-
"""
매수·매도 판정 규칙

매수는 indi.evaluate()의 강의 반영 조건을 그대로 쓰고,
매도는 여기서 정의합니다. 자동매매로 옮길 때 이 파일만 보면 되도록
모든 규칙을 한 곳에 모았습니다.
"""
import indi

# ── T·B 규칙 (일봉)
#    매수: T ≥ 10 이고 B ≥ 55 인 상태가 4일 이상 유지될 때
#    매도: 둘 중 하나라도 이탈할 때
#    홈런: 위 조건에 더해 G(청선) ≤ 5 — 여기서 큰 게 나옵니다
TB = dict(
    T_MIN=10.0,
    B_MIN=55.0,
    HOLD_MIN=4,       # 최소 유지일 (이 날부터 매수)
    HOLD_MAX=7,       # 이 날까지가 적기 — 넘으면 '지연' 표시만 하고 매수는 계속 허용
    HR_GS_MAX=5.0,    # 홈런 판정 청선 상한
    STOP_LOSS=-10.0,   # 손절선 (장중)
)

# ── 매도 규칙 파라미터
SELL = dict(
    GS_EXIT=5.0,       # 청선이 5선을 위로 이탈하면 추세 훼손
    GS_HARD=8.0,       # 8선 초과는 즉시 청산
    BIG_DOWN=-4.0,     # 장대음봉
    MA_BREAK=30,       # 30일선 이탈
    GB_DROP=-3.0,      # 적선 5일 기울기가 이만큼 꺾이면 청산
    STOP_LOSS=-7.0,    # 손절선 (%)
    TRAIL=-8.0,        # 고점 대비 이만큼 밀리면 청산 (%)
    MAX_DAYS=40,       # 최대 보유일
)


def snapshot(o, h, l, c, v):
    """지표 현재값 묶음 — 매수·매도 판정과 로그에 공통으로 씁니다"""
    n = len(c)
    if n < 120:
        return None
    gb, gs = indi.golden(c, 32)
    T, _ = indi.tsi(c)
    B = indi.rsi(c, 35)
    U, _ = indi.ultimate(h, l, c)
    mt, mb, red = indi.mesh(c)
    ma30 = indi.sma(c, SELL["MA_BREAK"])
    if None in (T[-1], B[-1], U[-1], ma30[-1]):
        return None

    hold = 0
    for i in range(n - 1, max(n - indi.CFG["LOOKBACK"] - 1, 0), -1):
        if gs[i] <= indi.CFG["GS_MAX"]:
            hold += 1
        else:
            break
    seg = gs[n - hold:] if hold else [gs[-1]]

    return dict(
        close=int(c[-1]),
        gbuy=round(gb[-1], 1), gsell=round(gs[-1], 1),
        gb_slope5=round(gb[-1] - gb[-6], 1),
        gs_hold=hold,
        gs_mean=round(sum(seg) / len(seg), 2), gs_std=round(indi.stdev(seg), 2),
        t=round(T[-1], 1), b=round(B[-1], 1), u=round(U[-1], 1),
        red=bool(red[-1]), ma30=int(ma30[-1]),
        above_ma30=bool(c[-1] >= ma30[-1]),
        chg1=round((c[-1] / c[-2] - 1) * 100, 2),
        chg5=round((c[-1] / c[-6] - 1) * 100, 2),
        chg20=round((c[-1] / c[-21] - 1) * 100, 2) if n > 21 else 0.0,
        wick=round(indi.wick(o[-1], h[-1], l[-1], c[-1]), 2),
        vol=int(v[-1]),
        amount=int(c[-1] * v[-1]),
    )


def sell_check(s, pos):
    """
    보유 종목 청산 판정.
    s   : snapshot() 결과 (오늘 지표)
    pos : {'entry': 진입가, 'entry_date':…, 'peak': 보유중 최고가, 'days': 보유일}
    반환: (action, reason, severity) — action은 'SELL' 또는 'HOLD'
    """
    entry = pos["entry"]
    px = s["close"]
    pnl = (px / entry - 1) * 100 if entry else 0.0
    peak = max(pos.get("peak", entry), px)
    from_peak = (px / peak - 1) * 100 if peak else 0.0

    # 손절은 장중 손절주문으로 따로 처리하므로 여기서는 다루지 않습니다
    # 강한 순서대로 — 먼저 걸리는 규칙이 청산 사유가 됩니다
    if s["gsell"] > SELL["GS_HARD"]:
        return "SELL", f"청선 {s['gsell']} — {SELL['GS_HARD']}선 초과 급이탈", "청산"
    if s["chg1"] <= SELL["BIG_DOWN"]:
        return "SELL", f"장대음봉 {s['chg1']:+.1f}%", "청산"
    if not s["above_ma30"]:
        return "SELL", f"30일선 이탈 (종가 {px:,} < {s['ma30']:,})", "청산"
    if s["gsell"] > SELL["GS_EXIT"]:
        return "SELL", f"청선 {s['gsell']} — 5선 이탈", "청산"
    if s["gb_slope5"] <= SELL["GB_DROP"]:
        return "SELL", f"적선 5일 기울기 {s['gb_slope5']:+.1f} — 상승탄력 꺾임", "청산"
    if from_peak <= SELL["TRAIL"]:
        return "SELL", f"고점 대비 {from_peak:+.1f}% 되밀림", "청산"
    if pos.get("days", 0) >= SELL["MAX_DAYS"]:
        return "SELL", f"보유 {pos['days']}일 경과 — 기간 청산", "기간"

    # 청산은 아니지만 주의가 필요한 상태
    warn = []
    if s["wick"] > indi.CFG["WICK_MAX"]:
        warn.append(f"위꼬리 {s['wick']}")
    if s["gsell"] > 3.5:
        warn.append(f"청선 {s['gsell']} 상승")
    if not s["red"]:
        warn.append("그물망 상승배열 아님")
    return "HOLD", (" · ".join(warn) if warn else "조건 유지"), ("주의" if warn else "정상")


def buy_check(o, h, l, c, v):
    """매수 판정 — 통과하면 evaluate() 결과, 아니면 None"""
    return indi.evaluate(o, h, l, c, v, strict=True)


# ══════════ T·B 규칙 ══════════
def tb_ok(t, b):
    return t is not None and b is not None and t >= TB["T_MIN"] and b >= TB["B_MIN"]


def tb_grade(gs, tb_hold):
    """
    등급 판정.
      홈런  — 청선 5 이하 (매수 집중 구간)
      일반  — 조건은 맞지만 청선이 높음
    """
    if gs <= TB["HR_GS_MAX"]:
        return "홈런"
    return "일반"


def tb_sell_check(s, pos):
    """T·B 규칙 보유 종목의 청산 판정"""
    if not tb_ok(s["t"], s["b"]):
        broke = []
        if s["t"] < TB["T_MIN"]:
            broke.append(f"T {s['t']} < {TB['T_MIN']:.0f}")
        if s["b"] < TB["B_MIN"]:
            broke.append(f"B {s['b']} < {TB['B_MIN']:.0f}")
        return "SELL", " · ".join(broke) + " 이탈", "이탈"

    warn = []
    if s["t"] < TB["T_MIN"] + 5:
        warn.append(f"T {s['t']} 기준선 근접")
    if s["b"] < TB["B_MIN"] + 3:
        warn.append(f"B {s['b']} 기준선 근접")
    if s["gsell"] > TB["HR_GS_MAX"]:
        warn.append(f"청선 {s['gsell']}")
    return "HOLD", (" · ".join(warn) if warn else "T·B 유지"), ("주의" if warn else "정상")
