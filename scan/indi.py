# -*- coding: utf-8 -*-
"""지표 계산 — numpy 없이 순수 파이썬 (32비트 환경 의존성 최소화)"""


def sma(v, n):
    out = [None] * len(v)
    if len(v) < n: return out
    s = sum(v[:n]); out[n - 1] = s / n
    for i in range(n, len(v)):
        s += v[i] - v[i - n]; out[i] = s / n
    return out


def ema(v, n):
    out = [None] * len(v)
    if len(v) < n: return out
    k = 2.0 / (n + 1); s = sum(v[:n]) / n; out[n - 1] = s; p = s
    for i in range(n, len(v)):
        p = v[i] * k + p * (1 - k); out[i] = p
    return out


def tsi(c, r=10, s=30, sig=8):
    mom = [0.0] + [c[i] - c[i - 1] for i in range(1, len(c))]
    am = [abs(x) for x in mom]
    e1 = [x for x in ema(mom, r) if x is not None]; e2 = ema(e1, s)
    a1 = [x for x in ema(am, r) if x is not None];  a2 = ema(a1, s)
    pad = len(c) - len(e2); out = [None] * pad
    for i in range(len(e2)):
        out.append(None if (e2[i] is None or a2[i] is None or a2[i] == 0)
                   else 100.0 * e2[i] / a2[i])
    tv = [x for x in out if x is not None]; sg = ema(tv, sig)
    return out, [None] * (len(out) - len(sg)) + sg


def rsi(c, n=35):
    """B (BPDI Hilo 근사)"""
    d = [0.0] + [c[i] - c[i - 1] for i in range(1, len(c))]
    up = [max(x, 0) for x in d]; dn = [max(-x, 0) for x in d]
    au, ad = ema(up, n), ema(dn, n)
    out = []
    for i in range(len(c)):
        if au[i] is None or ad[i] is None or (au[i] + ad[i]) == 0:
            out.append(None)
        else:
            out.append(100.0 * au[i] / (au[i] + ad[i]))
    return out


def ultimate(h, l, c, p1=7, p2=14, p3=28, sig=8):
    n = len(c)
    bp, tr = [0.0] * n, [0.0] * n
    for i in range(n):
        pc = c[i - 1] if i else c[0]
        bp[i] = c[i] - min(l[i], pc)
        tr[i] = max(h[i], pc) - min(l[i], pc)
    def rs(p):
        out = [None] * n
        for i in range(p - 1, n):
            st = sum(tr[i - p + 1:i + 1])
            out[i] = (sum(bp[i - p + 1:i + 1]) / st) if st else None
        return out
    r1, r2, r3 = rs(p1), rs(p2), rs(p3)
    out = []
    for i in range(n):
        if None in (r1[i], r2[i], r3[i]): out.append(None)
        else: out.append(100.0 * (4 * r1[i] + 2 * r2[i] + r3[i]) / 7)
    uv = [x for x in out if x is not None]; sg = sma(uv, sig)
    return out, [None] * (len(out) - len(sg)) + sg


def golden(c, n=32):
    """G 적선(Buy) / 청선(Sell)"""
    buy, sell = [], []
    for i in range(len(c)):
        w = c[max(0, i - n + 1):i + 1]
        lo, hi = min(w), max(w)
        buy.append((c[i] / lo - 1) * 100 if lo else 0.0)
        sell.append((hi / c[i] - 1) * 100 if c[i] else 0.0)
    return buy, sell


def mesh(c, start=10, step=5, cnt=6):
    mas = [sma(c, start + step * i) for i in range(cnt)]
    top, bot, red = [], [], []
    for i in range(len(c)):
        vs = [m[i] for m in mas if m[i] is not None]
        if len(vs) == cnt:
            top.append(max(vs)); bot.append(min(vs))
            red.append(mas[0][i] > mas[-1][i])
        else:
            top.append(None); bot.append(None); red.append(False)
    return top, bot, red


def stdev(v):
    if len(v) < 2: return 0.0
    m = sum(v) / len(v)
    return (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5


# ══════════ 강의 반영 판정 ══════════
CFG = dict(
    GS_MAX=5.0, GS_DAYS=3, GS_DAYS_GOOD=5, GS_STD_MAX=2.0, GS_SPIKE=8.0,
    GB_MIN=10.0, GB_SLOPE_MIN=1.0, WICK_MAX=0.5, WICK_DAYS=3, BIG_DOWN=-4.0,
    MA_YELLOW=30, LOOKBACK=15,
)


def wick(o, h, l, c):
    body = abs(c - o); upper = h - max(o, c)
    return upper / body if body > 0 else (999.0 if upper > 0 else 0.0)


def evaluate(o, h, l, c, v, strict=True):
    """일봉 판정. 조건 미달이면 None, 충족이면 dict"""
    n = len(c)
    if n < 120: return None
    gb, gs = golden(c, 32)
    T, Ts = tsi(c); B = rsi(c, 35); U, Us = ultimate(h, l, c)
    mt, mb, red = mesh(c)
    ma30 = sma(c, CFG["MA_YELLOW"])
    if None in (T[-1], B[-1], U[-1], ma30[-1]): return None

    hold = 0
    for i in range(n - 1, max(n - CFG["LOOKBACK"] - 1, 0), -1):
        if gs[i] <= CFG["GS_MAX"]: hold += 1
        else: break
    if strict and hold < CFG["GS_DAYS"]: return None

    seg = gs[n - hold:] if hold else [gs[-1]]
    gs_mean = sum(seg) / len(seg); gs_std = stdev(seg)
    if strict and gs_std > CFG["GS_STD_MAX"]: return None

    look = gs[max(0, n - CFG["LOOKBACK"]):n - hold] if hold < CFG["LOOKBACK"] else []
    spike = bool(look and max(look) > CFG["GS_SPIKE"])

    if strict and gb[-1] < CFG["GB_MIN"]: return None
    slope5 = gb[-1] - gb[-6]
    if strict and slope5 < CFG["GB_SLOPE_MIN"]: return None

    bad_wick = any(wick(o[i], h[i], l[i], c[i]) > CFG["WICK_MAX"] and h[i] > c[i]
                   for i in range(n - CFG["WICK_DAYS"], n))
    big_down = any((c[i] / c[i - 1] - 1) * 100 <= CFG["BIG_DOWN"]
                   for i in range(n - CFG["WICK_DAYS"], n))
    if strict and (bad_wick or big_down): return None
    if strict and c[-1] < ma30[-1]: return None

    score = 0.0
    score += 22 * min(1, hold / CFG["GS_DAYS_GOOD"])
    score += 20 * max(0, (CFG["GS_MAX"] - gs_mean) / CFG["GS_MAX"])
    score += 14 * max(0, 1 - gs_std / CFG["GS_STD_MAX"])
    score += 16 * min(1, slope5 / 12)
    score += 8 * min(1, max(0, (gb[-1] - 10) / 30))
    if not spike: score += 6
    if red[-1]: score += 4
    if U[-1] and U[-1] >= 50: score += 5
    if T[-1] and T[-1] >= 0: score += 5

    return dict(
        score=round(min(100, max(0, score)), 1),
        hold=hold, gs_mean=round(gs_mean, 2), gs_std=round(gs_std, 2),
        gbuy=round(gb[-1], 1), gsell=round(gs[-1], 1), slope5=round(slope5, 1),
        t=round(T[-1], 1), b=round(B[-1], 1), u=round(U[-1], 1),
        red=bool(red[-1]), ma30=int(ma30[-1]), spike=spike,
        wick=bad_wick, big_down=big_down,
        r20=round((c[-1] / c[-21] - 1) * 100, 1) if n > 21 else 0,
    )


def timing_60m(o, h, l, c, v):
    """60분봉 진입 타이밍 — 강의: 매매 타이밍은 분봉으로"""
    n = len(c)
    if n < 60: return None
    gb, gs = golden(c, 100 if n >= 120 else 32)
    T, _ = tsi(c); B = rsi(c, 35); U, _ = ultimate(h, l, c)
    if None in (T[-1], B[-1], U[-1]): return None

    hold = 0
    for i in range(n - 1, max(n - 30, 0), -1):
        if gs[i] <= 5.0: hold += 1
        else: break
    seg = gs[n - hold:] if hold else [gs[-1]]
    ready = bool(hold >= 4 and stdev(seg) <= 2.0 and gb[-1] >= 10 and gb[-1] > gb[-5])

    return dict(
        ready=ready, hold=hold,
        gbuy=round(gb[-1], 1), gsell=round(gs[-1], 1),
        gs_mean=round(sum(seg) / len(seg), 2), gs_std=round(stdev(seg), 2),
        t=round(T[-1], 1), b=round(B[-1], 1), u=round(U[-1], 1),
    )
