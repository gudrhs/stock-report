# -*- coding: utf-8 -*-
"""
일일 스캔 엔진 — 전일 종가 기준으로 매수·매도 신호를 뽑고 기록합니다.

  python daily.py                     최신 영업일 기준 1회
  python daily.py --date 2026-08-11   특정 일자 기준
  python daily.py --backfill 90       최근 90영업일을 소급해 기록·로그 생성

출력 (모두 ../data/ 아래)
  d/YYYY-MM-DD.json   그날의 전체 스냅샷 (지표값 포함 — 백테스트 재현용)
  latest.json         가장 최근 결과
  dates.json          기록이 있는 날짜 목록
  signals.csv         매수·매도 이벤트 로그 (append-only, 자동매매용)
  positions.json      현재 보유 중인 가상 포지션
"""
import os, sys, json, csv, argparse, datetime, warnings
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))

import indi, rules, engine, universe, charts as chartmod
import FinanceDataReader as fdr
from concurrent.futures import ThreadPoolExecutor

CFG = dict(
    BARS=260,          # 1회 실행 시 확보할 일봉 수 (지표 안정에 120 이상 필요)
    MAX_POS=10,        # 동시 보유 가상 포지션 수 — 상위 점수부터 채웁니다
    MIN_AMOUNT=3e8,    # 최소 거래대금
    TOP_SHOW=40,       # 화면에 싣는 매수 후보 수
    HIST_DAYS=260,     # history.json에 보관할 날짜 수 (약 1년)
    HIST_TOP=15,       # 지난 날짜에 보관할 매수 후보 수 (최신일은 TOP_SHOW 전량)
    CHART_MAX=60,      # 차트를 준비할 종목 수 (홈런·후보·보유 우선)
)

CSV_COLS = [
    "date", "strategy", "code", "name", "market", "action", "severity", "grade",
    "reason", "price", "score", "rank",
    "entry_date", "entry_price", "days_held", "pnl_pct",
    "gbuy", "gsell", "gs_hold", "gs_mean", "gs_std", "gb_slope5",
    "t", "b", "u", "red", "ma30", "chg1", "chg5", "chg20", "vol", "amount",
]


# ══════════ 데이터 수집 ══════════
def fetch_all(codes, start, end, workers=8):
    """{code: DataFrame} — 실패한 종목은 빠집니다"""
    def one(cd):
        try:
            df = fdr.DataReader(cd, start, end)
            return cd, (df if len(df) >= 130 else None)
        except Exception:
            return cd, None
    out = {}
    with ThreadPoolExecutor(workers) as ex:
        for cd, df in ex.map(one, codes):
            if df is not None:
                out[cd] = df
    return out


def build_all(data):
    """{code: (지표시계열, 날짜→인덱스)} — 종목당 한 번만 계산합니다"""
    out = {}
    for code, df in data.items():
        o, h, l, c, v = (df[k].astype(float).tolist()
                         for k in ("Open", "High", "Low", "Close", "Volume"))
        S = engine.build_series(o, h, l, c, v)
        if S is None:
            continue
        idx = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(df.index)}
        out[code] = (S, idx)
    return out


# ══════════ 저장 ══════════
def load_json(path, default):
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return default


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=0)


def append_csv(rows):
    path = os.path.join(DATA, "signals.csv")
    os.makedirs(DATA, exist_ok=True)
    # 파일이 비어 있으면 헤더부터 (삭제 대신 비워지는 환경 대응)
    new = (not os.path.exists(path)) or os.path.getsize(path) == 0
    with open(path, "a", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def build_stats(strategy=None):
    """signals.csv를 훑어 누적 성과를 냅니다. strategy를 주면 그 전략만."""
    path = os.path.join(DATA, "signals.csv")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    sells = [r for r in csv.DictReader(open(path, encoding="utf-8-sig"))
             if r["action"] == "SELL" and r["pnl_pct"] not in ("", None)
             and (strategy is None or r.get("strategy") == strategy)]
    if not sells:
        return None
    p = [float(r["pnl_pct"]) for r in sells]
    win = [x for x in p if x > 0]
    eq = 1.0
    for x in p:
        eq *= (1 + x / 100 / CFG["MAX_POS"])
    by = {}
    for r in sells:
        k = next((w for w in ("손절", "청선", "장대음봉", "30일선", "적선", "고점", "기간")
                  if w in r["reason"]), "기타")
        by.setdefault(k, []).append(float(r["pnl_pct"]))
    hr = [float(r["pnl_pct"]) for r in sells if r.get("grade") == "홈런"]
    return dict(
        trades=len(p),
        homerun=(dict(n=len(hr), avg=round(sum(hr) / len(hr), 2),
                      win=round(sum(1 for x in hr if x > 0) / len(hr) * 100, 1),
                      best=round(max(hr), 1)) if hr else None),
        avg=round(sum(p) / len(p), 2),
        winrate=round(len(win) / len(p) * 100, 1),
        best=round(max(p), 1), worst=round(min(p), 1),
        cum=round((eq - 1) * 100, 2),
        avg_days=round(sum(int(r["days_held"] or 0) for r in sells) / len(sells), 1),
        by_reason={k: dict(n=len(v), avg=round(sum(v) / len(v), 2),
                           win=round(sum(1 for x in v if x > 0) / len(v) * 100, 0))
                   for k, v in sorted(by.items(), key=lambda x: -len(x[1]))},
        first=sells[0]["date"], last=sells[-1]["date"],
    )


def log_row(date, code, name, market, action, severity, reason, s,
            score="", rank="", pos=None, pnl="", strategy=""):
    r = dict(date=date, strategy=strategy, code=code, name=name, market=market,
             action=action, severity=severity, reason=reason, price=s["close"],
             score=score, rank=rank, grade=(pos or {}).get("grade", ""),
             entry_date=(pos or {}).get("entry_date", ""),
             entry_price=(pos or {}).get("entry", ""),
             days_held=(pos or {}).get("days", ""), pnl_pct=pnl)
    for k in ("gbuy", "gsell", "gs_hold", "gs_mean", "gs_std", "gb_slope5",
              "t", "b", "u", "red", "ma30", "chg1", "chg5", "chg20", "vol", "amount"):
        r[k] = s.get(k, "")
    return r


# ══════════ 하루치 처리 ══════════
def scan_day(date, series, meta):
    """그날 전 종목 지표 스냅샷 + 두 전략의 매수 후보"""
    snaps, buys, tbuys = {}, [], []
    for code, (S, idx) in series.items():
        i = idx.get(date)
        if i is None:            # 그날 거래가 없던 종목 (거래정지·상장 전)
            continue
        s = engine.snapshot_at(S, i)
        if not s:
            continue
        snaps[code] = s
        if s["amount"] < CFG["MIN_AMOUNT"]:
            continue
        name, market = meta.get(code, (code, ""))
        e = engine.buy_at(S, i)
        if e:
            buys.append(dict(code=code, name=name, market=market, **e, **{
                k: s[k] for k in ("close", "chg1", "chg5", "chg20", "amount", "vol")}))
        tb = engine.tb_buy_at(S, i)
        if tb:
            tbuys.append(dict(code=code, name=name, market=market, **tb, **{
                k: s[k] for k in ("close", "chg1", "chg5", "chg20", "amount", "vol")}))

    buys.sort(key=lambda x: -x["score"])
    for i, b in enumerate(buys, 1):
        b["rank"] = i
    # T·B는 홈런을 위로 올리고, 그 안에서 점수순
    tbuys.sort(key=lambda x: (not x["homerun"], -x["score"]))
    for i, b in enumerate(tbuys, 1):
        b["rank"] = i
    return snaps, buys, tbuys


def trade(date, snaps, buys, positions, pending, sell_fn, tag, max_pos, stop_loss):
    """
    한 전략의 하루 매매. 순서가 중요합니다.

      1) 어제 종가로 낸 주문을 오늘 **시가**에 체결   ← 실제로 살 수 있는 가격
      2) 보유 종목 장중 손절 확인
      3) 오늘 종가로 지표를 계산해 내일 낼 주문을 결정

    종가 신호를 그 종가에 체결하면 미래를 미리 본 것이 됩니다.
    """
    rows, filled = [], []

    # ── 1) 어제 주문을 오늘 시가에 체결
    for od in pending.get("sell", []):
        code = od["code"]
        pos, s = positions.get(code), snaps.get(code)
        if not pos or not s:
            continue
        px = s["open"]
        pnl = round((px / pos["entry"] - 1) * 100, 2) if pos["entry"] else 0.0
        rows.append(log_row(date, code, pos["name"], pos.get("market", ""), "SELL",
                            od["severity"],
                            f"{od['reason']} → {od['signal_date']} 종가신호, 시가 체결",
                            dict(s, close=px), pos=pos, pnl=pnl, strategy=tag))
        filled.append(dict(kind="매도", code=code, name=pos["name"], price=px, pnl=pnl,
                           days=pos["days"], entry=pos["entry"],
                           entry_date=pos["entry_date"], reason=od["reason"],
                           severity=od["severity"], grade=pos.get("grade", "")))
        positions.pop(code, None)

    for od in pending.get("buy", []):
        code = od["code"]
        s = snaps.get(code)
        if not s or code in positions or len(positions) >= max_pos:
            continue
        px = s["open"]
        positions[code] = dict(name=od["name"], market=od["market"], entry=px,
                               entry_date=date, peak=px, days=0, score=od["score"],
                               grade=od.get("grade", ""), signal_date=od["signal_date"])
        rows.append(log_row(date, code, od["name"], od["market"], "BUY",
                            od.get("grade") or "진입",
                            f"{od['signal_date']} 종가신호 {od['score']}점 "
                            f"{od['rank']}위 → 시가 체결",
                            dict(s, close=px), score=od["score"], rank=od["rank"],
                            pos=positions[code], pnl=0.0, strategy=tag))
        filled.append(dict(kind="매수", code=code, name=od["name"], price=px, pnl=0.0,
                           days=0, entry=px, entry_date=date,
                           reason=f"{od['signal_date']} 신호 {od['rank']}위",
                           severity=od.get("grade") or "진입",
                           grade=od.get("grade", "")))

    # ── 2) 장중 손절 (실제 자동매매는 손절주문을 미리 걸어둡니다)
    #     stop_loss가 None이면 손절 없이 규칙 이탈까지만 봅니다
    for code in (list(positions.keys()) if stop_loss is not None else []):
        pos, s = positions[code], snaps.get(code)
        if not s:
            continue
        stop = pos["entry"] * (1 + stop_loss / 100.0)
        if s["low"] > stop:
            continue
        px = int(min(stop, s["open"]))       # 갭하락이면 시가 체결
        pnl = round((px / pos["entry"] - 1) * 100, 2)
        reason = f"장중 손절 {pnl:+.1f}% (기준 {stop_loss}%)"
        rows.append(log_row(date, code, pos["name"], pos.get("market", ""), "SELL",
                            "손절", reason, dict(s, close=px), pos=pos, pnl=pnl,
                            strategy=tag))
        filled.append(dict(kind="매도", code=code, name=pos["name"], price=px, pnl=pnl,
                           days=pos.get("days", 0), entry=pos["entry"],
                           entry_date=pos["entry_date"], reason=reason,
                           severity="손절", grade=pos.get("grade", "")))
        positions.pop(code)

    # ── 3) 오늘 종가로 내일 낼 주문 결정
    nxt = dict(buy=[], sell=[])
    sells, holds = [], []
    for code in list(positions.keys()):
        pos, s = positions[code], snaps.get(code)
        if not s:
            continue
        pos["days"] = pos.get("days", 0) + 1
        pos["peak"] = max(pos.get("peak", pos["entry"]), s["close"])
        action, reason, sev = sell_fn(s, pos)
        pnl = round((s["close"] / pos["entry"] - 1) * 100, 2) if pos["entry"] else 0.0
        rec = dict(code=code, name=pos["name"], market=pos.get("market", ""),
                   entry=pos["entry"], entry_date=pos["entry_date"], days=pos["days"],
                   price=s["close"], pnl=pnl, reason=reason, severity=sev,
                   grade=pos.get("grade", ""),
                   gsell=s["gsell"], gbuy=s["gbuy"], gb_slope5=s["gb_slope5"],
                   t=s["t"], b=s["b"], above_ma30=s["above_ma30"], red=s["red"])
        if action == "SELL":
            sells.append(rec)
            nxt["sell"].append(dict(code=code, reason=reason, severity=sev,
                                    signal_date=date))
        else:
            holds.append(rec)
            rows.append(log_row(date, code, pos["name"], pos.get("market", ""), "HOLD",
                                sev, reason, s, pos=pos, pnl=pnl, strategy=tag))

    room = max_pos - len(positions) + len(nxt["sell"])
    for b in buys:
        if len(nxt["buy"]) >= room:
            break
        if b["code"] in positions:
            continue
        nxt["buy"].append(dict(code=b["code"], name=b["name"], market=b["market"],
                               score=b["score"], rank=b["rank"],
                               grade=b.get("grade", ""), signal_date=date))
    return dict(filled=filled, sells=sells, holds=holds, rows=rows, nxt=nxt)


def run_day(date, series, uni, state, quiet=False):
    """두 전략을 함께 돌리고 그날 스냅샷을 만듭니다"""
    meta = {c: (n, m) for c, n, m, _ in uni}
    snaps, buys, tbuys = scan_day(date, series, meta)

    a = trade(date, snaps, buys, state["positions"], state["pending"],
              rules.sell_check, "기존", CFG["MAX_POS"], rules.SELL["STOP_LOSS"])
    b = trade(date, snaps, tbuys, state["positions_tb"], state["pending_tb"],
              rules.tb_sell_check, "TB", CFG["MAX_POS"], rules.TB["STOP_LOSS"])
    state["pending"], state["pending_tb"] = a["nxt"], b["nxt"]

    homerun = [x for x in tbuys if x["homerun"]]
    snap = dict(
        date=date,
        generated=datetime.datetime.now().isoformat(timespec="seconds"),
        universe=dict(kospi200=sum(1 for x in uni if x[2] == "코스피200"),
                      kosdaq150=sum(1 for x in uni if x[2] == "코스닥150"),
                      scanned=len(snaps)),
        buy=buys[:CFG["TOP_SHOW"]], buy_total=len(buys),
        sell=a["sells"], hold=a["holds"], filled=a["filled"],
        order=dict(buy=[x["code"] for x in a["nxt"]["buy"]],
                   sell=[x["code"] for x in a["nxt"]["sell"]]),
        tb=dict(buy=tbuys[:CFG["TOP_SHOW"]], buy_total=len(tbuys),
                homerun=homerun[:CFG["TOP_SHOW"]], homerun_total=len(homerun),
                sell=b["sells"], hold=b["holds"], filled=b["filled"],
                order=dict(buy=[x["code"] for x in b["nxt"]["buy"]],
                           sell=[x["code"] for x in b["nxt"]["sell"]])),
        stats=dict(buy=len(buys), sell=len(a["sells"]), hold=len(a["holds"]),
                   filled=len(a["filled"]),
                   open_positions=len(state["positions"]),
                   tb_buy=len(tbuys), tb_homerun=len(homerun),
                   tb_sell=len(b["sells"]), tb_filled=len(b["filled"]),
                   tb_positions=len(state["positions_tb"])),
    )
    if not quiet:
        print(f"  {date}  기존 매수 {len(buys):3d}/체결 {len(a['filled']):2d} · "
              f"T·B 매수 {len(tbuys):3d}(홈런 {len(homerun):2d})/체결 {len(b['filled']):2d} · "
              f"보유 {len(state['positions'])}+{len(state['positions_tb'])}")
    return snap, state, a["rows"] + b["rows"]


# ══════════ 진입점 ══════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="기준일 YYYY-MM-DD (미지정 시 최신 영업일)")
    ap.add_argument("--backfill", type=int, default=0, help="최근 N영업일 소급 실행")
    ap.add_argument("--reset", action="store_true", help="기록·로그를 지우고 새로 시작")
    ap.add_argument("--no-m60", action="store_true", help="60분봉 수집 생략 (빠른 실행)")
    a = ap.parse_args()

    os.makedirs(DATA, exist_ok=True)
    if a.reset:
        def wipe(p):
            """삭제가 막힌 환경(마운트 폴더 등)에서는 내용을 비웁니다"""
            if not os.path.exists(p):
                return
            try:
                os.remove(p)
            except OSError:
                open(p, "w", encoding="utf-8").close()
        for f in ("signals.csv", "positions.json", "pending.json",
                  "positions_tb.json", "pending_tb.json", "charts.json",
                  "dates.json", "latest.json", "history.json", "stats.json"):
            wipe(os.path.join(DATA, f))
        print("기존 기록을 지웠습니다")

    print("종목 목록 산출 중…")
    uni = universe.build()
    print(f"  대상 {len(uni)}종목 (코스피200 200 · 코스닥150 150)")

    end = a.date or datetime.date.today().strftime("%Y-%m-%d")
    start = (datetime.date.fromisoformat(end)
             - datetime.timedelta(days=int((CFG["BARS"] + a.backfill) * 1.65) + 60))
    print(f"시세 수집 중… ({start} ~ {end})")
    data = fetch_all([c for c, _, _, _ in uni], start, end)
    print(f"  {len(data)}종목 수집 완료")
    if not data:
        print("수집 실패 — 네트워크를 확인하세요")
        return 1

    print("지표 계산 중… (종목당 1회)")
    series = build_all(data)
    print(f"  {len(series)}종목 준비 완료")

    # 실제 거래일 목록 (가장 많은 데이터를 가진 종목 기준)
    ref = max(data.values(), key=len)
    days = [d.strftime("%Y-%m-%d") for d in ref.index]
    if a.date:
        days = [d for d in days if d <= a.date]
    targets = days[-(a.backfill or 1):]

    state = dict(
        positions=load_json(os.path.join(DATA, "positions.json"), {}),
        pending=load_json(os.path.join(DATA, "pending.json"), dict(buy=[], sell=[])),
        positions_tb=load_json(os.path.join(DATA, "positions_tb.json"), {}),
        pending_tb=load_json(os.path.join(DATA, "pending_tb.json"), dict(buy=[], sell=[])),
    )
    dates = load_json(os.path.join(DATA, "dates.json"), [])
    all_rows, last = [], None

    # 날짜별 스냅샷은 한 파일에 모읍니다 (파일 수를 늘리지 않고, 화면 전환도 즉시)
    hist = load_json(os.path.join(DATA, "history.json"), {})

    print(f"판정 중… ({len(targets)}일)")
    for d in targets:
        snap, state, rows = run_day(d, series, uni, state)
        hist[d] = dict(snap, buy=snap["buy"][:CFG["HIST_TOP"]],
                       tb=dict(snap["tb"], buy=snap["tb"]["buy"][:CFG["HIST_TOP"]],
                               homerun=snap["tb"]["homerun"][:CFG["HIST_TOP"]]))
        all_rows += rows
        last = snap
        if d not in dates:
            dates.append(d)

    dates = sorted(set(dates))[-CFG["HIST_DAYS"]:]
    hist = {d: hist[d] for d in dates if d in hist}
    save_json(os.path.join(DATA, "dates.json"), dates)
    save_json(os.path.join(DATA, "positions.json"), state["positions"])
    save_json(os.path.join(DATA, "pending.json"), state["pending"])
    save_json(os.path.join(DATA, "positions_tb.json"), state["positions_tb"])
    save_json(os.path.join(DATA, "pending_tb.json"), state["pending_tb"])
    append_csv(all_rows)

    # 화면에서 종목을 눌렀을 때 띄울 차트 데이터 (일봉 + 60분봉)
    print("차트 데이터 준비 중…")
    codes = []
    for group in (last["tb"]["homerun"], last["tb"]["buy"], last["buy"],
                  last["tb"]["hold"], last["hold"],
                  last["tb"]["sell"], last["sell"]):
        for x in group:
            if x["code"] not in codes:
                codes.append(x["code"])
    codes = codes[:CFG["CHART_MAX"]]
    m60 = load_json(os.path.join(DATA, "m60.json"), {})
    ch, m60 = chartmod.build(codes, data, {c: (n, m) for c, n, m, _ in uni}, m60,
                             with_m60=not a.no_m60)
    save_json(os.path.join(DATA, "charts.json"), ch)
    save_json(os.path.join(DATA, "m60.json"), m60)
    got = sum(1 for v in ch.values() if "m60" in v)
    print(f"  {len(ch)}종목 (60분봉 {got}종목)")

    st = build_stats()
    if st:
        save_json(os.path.join(DATA, "stats.json"), st)
        last["perf"] = st
        hist[last["date"]] = dict(hist[last["date"]], perf=st)
        for tag, key in (("기존", "legacy"), ("TB", "tb")):
            s2 = build_stats(tag)
            if s2:
                last.setdefault("perf_by", {})[key] = s2
        print(f"  누적 성과 — 청산 {st['trades']}건 · 평균 {st['avg']:+.2f}% · "
              f"승률 {st['winrate']}% · 누적 {st['cum']:+.1f}%")
        for key, nm in (("legacy", "기존"), ("tb", "T·B")):
            p = (last.get("perf_by") or {}).get(key)
            if p:
                print(f"    {nm:<4s} 청산 {p['trades']:3d}건 · 평균 {p['avg']:+6.2f}% · "
                      f"승률 {p['winrate']:4.1f}% · 누적 {p['cum']:+7.2f}%")
        hist[last["date"]]["perf_by"] = last.get("perf_by", {})

    save_json(os.path.join(DATA, "latest.json"), last)
    save_json(os.path.join(DATA, "history.json"), hist)

    s = last["stats"]
    print(f"\n완료 — 기준일 {last['date']}")
    print(f"  기존 규칙 — 매수 {s['buy']} · 매도예정 {s['sell']} · 보유 {s['open_positions']}")
    print(f"  T·B 규칙 — 매수 {s['tb_buy']} (홈런 {s['tb_homerun']}) · "
          f"매도예정 {s['tb_sell']} · 보유 {s['tb_positions']}")
    print(f"  로그 {len(all_rows)}행 추가 · 기록된 날짜 {len(dates)}일")
    hr = last["tb"]["homerun"]
    if hr:
        print("\n  홈런 후보:")
        for b in hr[:8]:
            print(f"    {b['name']:<14s}({b['code']}) {b['score']:5.1f}점  "
                  f"{b['close']:>9,}원  G {b['gsell']:>4.1f} · T {b['t']:>5.1f} · "
                  f"B {b['b']:>4.1f} · {b['tb_hold']}일 유지")
    return 0


if __name__ == "__main__":
    sys.exit(main())
