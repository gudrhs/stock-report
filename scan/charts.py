# -*- coding: utf-8 -*-
"""
화면에서 종목을 눌렀을 때 띄울 차트 데이터를 만듭니다.

지표는 브라우저에서 계산합니다(index.html 안의 지표 함수). 여기서는 OHLCV만
담아 파일을 가볍게 유지합니다 — 지표까지 넣으면 용량이 서너 배가 됩니다.

60분봉은 네이버가 1분봉을 7거래일치만 주기 때문에, 매일 실행하며
data/m60.json에 **누적**합니다. 며칠 돌리면 화면에 충분한 길이가 쌓입니다.
"""
import re, urllib.request, warnings
warnings.filterwarnings("ignore")
from concurrent.futures import ThreadPoolExecutor

DAILY_BARS = 160     # 일봉 표시 구간
M60_KEEP = 240       # 종목당 보관할 60분봉 수
M60_CODES = 160      # 60분봉을 보관할 최대 종목 수
UA = {"User-Agent": "Mozilla/5.0"}


def _daily(df):
    d = df.tail(DAILY_BARS)
    return dict(
        t=[x.strftime("%Y-%m-%d") for x in d.index],
        o=[int(x) for x in d["Open"]], h=[int(x) for x in d["High"]],
        l=[int(x) for x in d["Low"]], c=[int(x) for x in d["Close"]],
        v=[int(x) for x in d["Volume"]],
    )


def fetch_m60(code):
    """
    네이버 1분봉을 받아 60분봉으로 묶습니다.
    응답은 종가와 '당일 누적' 거래량만 주므로, 거래량은 차분으로 되살립니다.
    반환: [(키, o, h, l, c, v), ...] — 키는 YYYYMMDDHH
    """
    url = (f"https://fchart.stock.naver.com/sise.nhn?symbol={code}"
           f"&timeframe=minute&count=3000&requestType=0")
    try:
        req = urllib.request.Request(url, headers=UA)
        txt = urllib.request.urlopen(req, timeout=15).read().decode("euc-kr", "ignore")
    except Exception:
        return []

    rows = []
    for s in re.findall(r'data="([^"]+)"', txt):
        p = s.split("|")
        if len(p) < 6 or p[4] in ("null", ""):
            continue
        try:
            rows.append((p[0], float(p[4]), float(p[5] or 0)))
        except ValueError:
            continue
    if len(rows) < 20:
        return []

    # 누적 거래량 → 분당 거래량 (날이 바뀌면 0부터 다시 시작)
    per, prev, pday = [], 0.0, None
    for ts, c, cum in rows:
        day = ts[:8]
        if day != pday:
            pday, prev = day, 0.0
        per.append((ts, c, max(cum - prev, 0.0)))
        prev = cum

    out, cur, key = [], None, None
    for ts, c, v in per:
        k = ts[:10]                        # YYYYMMDDHH
        if k != key:
            if cur:
                out.append(tuple(cur))
            key, cur = k, [k, c, c, c, c, v]
        else:
            cur[2] = max(cur[2], c)
            cur[3] = min(cur[3], c)
            cur[4] = c
            cur[5] += v
    if cur:
        out.append(tuple(cur))
    return out


def merge_m60(store, code, bars):
    """기존 보관분과 합쳐 시간순 정렬 후 최근 것만 남깁니다"""
    old = store.get(code) or {}
    seen = {k: (o, h, l, c, v) for k, o, h, l, c, v
            in zip(old.get("k", []), old.get("o", []), old.get("h", []),
                   old.get("l", []), old.get("c", []), old.get("v", []))}
    for k, o, h, l, c, v in bars:
        seen[k] = (o, h, l, c, v)          # 같은 시각은 새 값으로 덮어씀
    ks = sorted(seen)[-M60_KEEP:]
    store[code] = dict(
        k=ks,
        o=[int(seen[k][0]) for k in ks], h=[int(seen[k][1]) for k in ks],
        l=[int(seen[k][2]) for k in ks], c=[int(seen[k][3]) for k in ks],
        v=[int(seen[k][4]) for k in ks],
    )
    return store[code]


def label_m60(keys):
    return [f"{k[4:6]}/{k[6:8]} {k[8:10]}시" for k in keys]


def build(codes, data, meta, m60_store, workers=6, with_m60=True):
    """
    (charts, m60_store) 반환.
    charts = {code: {name, market, daily:{...}, m60:{...}}}
    """
    out = {}
    for code in codes:
        df = data.get(code)
        if df is None or len(df) < 30:
            continue
        name, market = meta.get(code, (code, ""))
        out[code] = dict(name=name, market=market, daily=_daily(df))

    if with_m60 and out:
        with ThreadPoolExecutor(workers) as ex:
            for cd, bars in ex.map(lambda c: (c, fetch_m60(c)), list(out.keys())):
                if bars:
                    merge_m60(m60_store, cd, bars)
        for cd in out:
            m = m60_store.get(cd)
            if m and len(m["k"]) >= 20:
                out[cd]["m60"] = dict(t=label_m60(m["k"]), o=m["o"], h=m["h"],
                                      l=m["l"], c=m["c"], v=m["v"])

    # 보관 종목 수 제한 — 이번에 쓰인 종목을 우선 남깁니다
    if len(m60_store) > M60_CODES:
        keep = set(out) | set(list(m60_store)[-(M60_CODES - len(out)):])
        m60_store = {k: v for k, v in m60_store.items() if k in keep}
    return out, m60_store
