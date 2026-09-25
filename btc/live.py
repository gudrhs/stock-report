# -*- coding: utf-8 -*-
"""
실시간 모의매매 — 백테스트와 같은 지표 엔진·판단 규칙·체결 회계를 그대로 씁니다.

  python -m btc.live run  --venue upbit            4시간마다 스스로 깨어나 판단 (Ctrl+C로 종료)
  python -m btc.live once --venue upbit            새로 마감된 봉만 처리하고 종료 (cron/작업 스케줄러용)
  python -m btc.live status                        현재 상태 출력
  python -m btc.live retrain --src <1분봉 저장소>   매월 1일: 모델 갱신 (1월은 처음부터)

흐름 (4시간봉 마감 + 20초마다)
  1. 거래소에서 1시간봉을 받아 직접 UTC 4시간봉(00·04·08·12·16·20시)으로 묶습니다.
     아직 끝나지 않은 봉은 버립니다. 4개 시간봉 중 하나라도 없거나 거래량이 0이면 '죽은 봉'.
  2. 직전에 낸 주문을 이번 봉 '시가'로 체결 (백테스트와 같은 가정: 수수료+슬리피지 c)
  3. 지표 갱신 → 앙상블 판단 → 다음 봉 시가에 낼 주문 등록
  4. 상태를 원자적으로 저장 (임시 파일에 쓰고 이름 바꾸기) + 로그 한 줄

안전장치
  · 데이터가 2봉 넘게 끊기면 새 매수 금지 · 한 봉 수익이 10σ를 넘으면 그 봉은 관망
  · 모델 출력이 NaN이거나 |U|>200이면 직전 모델로 되돌리고 관망
  · 모의 낙폭이 백테스트 최악 낙폭의 1.2배를 넘으면 학습 동결 + 경고

실제 주문은 넣지 않습니다(모의매매 전용). 원화(업비트) 가격으로 판단해도 지표는 가격 수준과
무관하게 만들어져 있지만, 백테스트는 달러(비트스탬프) 기준이라 환율·김치프리미엄은 반영되지 않습니다.
"""
import argparse
import json
import math
import os
import pickle
import time

import numpy as np
import pandas as pd

from . import config as C
from .agent import Ensemble, threshold
from .data import DATA_DIR, FORCED_HOLD_GAP
from .features import FeatureEngine, NAMES, WARMUP

LIVE_DIR = os.path.join(DATA_DIR, "live")
MODEL_DIR = os.path.join(DATA_DIR, "models")
H = 3600
BAR = 4 * H
FI = {n: i for i, n in enumerate(NAMES)}
BASELINES = ("B0", "B1", "B2", "B3", "B4", "B5", "B6")


# ══════════ 거래소 1시간봉 ══════════
def _get_json(url, timeout=15, tries=4):
    import urllib.request
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "stock-report-btc/1.0",
                                                       "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:                      # 2·4·8초 쉬고 다시
            last = e
            if k < tries - 1:
                time.sleep(2 ** (k + 1))
    raise RuntimeError(f"조회 실패: {url} ({last})")


class Venue:
    """1시간봉 공급자. fetch_hourly(n, now)는 '마감된' 봉만 ts 오름차순 DataFrame으로 돌려줍니다."""
    name = "base"

    def fetch_hourly(self, n, now=None):
        raise NotImplementedError

    @staticmethod
    def closed_only(df, now):
        df = df.drop_duplicates("ts", keep="last").sort_values("ts").reset_index(drop=True)
        return df[df["ts"] + H <= now].reset_index(drop=True)


class Upbit(Venue):
    name = "upbit"

    def __init__(self, market="KRW-BTC"):
        self.market = market

    def fetch_hourly(self, n, now=None):
        now = time.time() if now is None else now
        rows, to = [], None
        while len(rows) < n:
            k = min(200, n - len(rows))
            url = f"https://api.upbit.com/v1/candles/minutes/60?market={self.market}&count={k}"
            if to:
                url += "&to=" + to
            js = _get_json(url)
            if not js:
                break
            rows += js
            to = js[-1]["candle_date_time_utc"] + "Z"
            time.sleep(0.12)                          # 초당 10회 제한
        df = pd.DataFrame({
            "ts": [int(pd.Timestamp(r["candle_date_time_utc"], tz="UTC").timestamp()) for r in rows],
            "open": [float(r["opening_price"]) for r in rows], "high": [float(r["high_price"]) for r in rows],
            "low": [float(r["low_price"]) for r in rows], "close": [float(r["trade_price"]) for r in rows],
            "volume": [float(r["candle_acc_trade_volume"]) for r in rows]})
        return self.closed_only(df, now)


class Binance(Venue):
    name = "binance"

    def __init__(self, symbol="BTCUSDT"):
        self.symbol = symbol

    def fetch_hourly(self, n, now=None):
        now = time.time() if now is None else now
        rows, end = [], None
        while len(rows) < n:
            url = f"https://api.binance.com/api/v3/klines?symbol={self.symbol}&interval=1h&limit=1000"
            if end:
                url += f"&endTime={end}"
            js = _get_json(url)
            if not js:
                break
            rows = js + rows
            end = int(js[0][0]) - 1
            if len(js) < 1000:
                break
        df = pd.DataFrame({"ts": [int(r[0]) // 1000 for r in rows], "open": [float(r[1]) for r in rows],
                           "high": [float(r[2]) for r in rows], "low": [float(r[3]) for r in rows],
                           "close": [float(r[4]) for r in rows], "volume": [float(r[5]) for r in rows]})
        return self.closed_only(df, now).tail(n).reset_index(drop=True)


class Bitstamp(Venue):
    name = "bitstamp"

    def fetch_hourly(self, n, now=None):
        now = int(time.time() if now is None else now)
        rows, end = [], now
        while len(rows) < n:
            js = _get_json(f"https://www.bitstamp.net/api/v2/ohlc/btcusd/?step=3600&limit=1000&end={end}")
            got = js["data"]["ohlc"]
            if not got:
                break
            rows = got + rows
            end = int(got[0]["timestamp"]) - 1
            if len(got) < 1000:
                break
        df = pd.DataFrame(rows).astype(float).rename(columns={"timestamp": "ts"})
        df["ts"] = df["ts"].astype("int64")
        return self.closed_only(df[["ts", "open", "high", "low", "close", "volume"]], now).tail(n).reset_index(drop=True)


VENUES = {"upbit": Upbit, "binance": Binance, "bitstamp": Bitstamp}


# ══════════ 1시간봉 → UTC 4시간봉 ══════════
def hourly_to_4h(h):
    """
    UTC 4시간 격자로 묶습니다. 4개 시간봉이 모두 있고 거래량이 0보다 커야 살아 있는 봉.
    반환: 살아 있는 봉 DataFrame(ts, open, high, low, close, volume, gap_before, forced_hold)
    마지막 4시간 구간이 아직 다 차지 않았으면 제외합니다.
    """
    if len(h) == 0:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume", "gap_before", "forced_hold"])
    h = h.drop_duplicates("ts", keep="last").sort_values("ts")
    g = (h["ts"] // BAR) * BAR
    agg = h.groupby(g).agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                           close=("close", "last"), volume=("volume", "sum"), n=("ts", "count"))
    full = np.arange(agg.index.min(), agg.index.max() + BAR, BAR)
    agg = agg.reindex(full)
    alive = (agg["n"] == 4).to_numpy() & (agg["volume"] > 0).to_numpy()
    run, gap = 0, np.zeros(len(agg), dtype=np.int64)
    for i, ok in enumerate(alive):
        gap[i] = run
        run = 0 if ok else run + 1
    agg["gap_before"] = gap
    out = agg[alive].copy()
    out["forced_hold"] = out["gap_before"] >= FORCED_HOLD_GAP
    out.insert(0, "ts", out.index.astype("int64"))
    return out.drop(columns="n").reset_index(drop=True)


# ══════════ 모의 계좌 ══════════
class Account:
    """env.simulate()와 같은 체결 회계 (0/1 보유, 다음 봉 시가 체결)"""

    def __init__(self, cost):
        self.cost = cost
        self.cash, self.units, self.pos = 1.0, 0.0, 0
        self.pending = None
        self.entry_eq = None
        self.hwm, self.equity = 1.0, 1.0
        self.trades = []

    def fill(self, open_px):
        a = self.pending
        self.pending = None
        if a is None or a == self.pos:
            return None
        c = self.cost
        if a == 1:
            self.entry_eq = self.cash
            self.units = self.cash * (1 - c) / open_px
            self.cash = 0.0
        else:
            self.cash = self.units * open_px * (1 - c)
            self.units = 0.0
            if self.entry_eq:
                self.trades.append(self.cash / self.entry_eq - 1)
        self.pos = a
        return dict(side="BUY" if a == 1 else "SELL", price=open_px)

    def mark(self, close_px):
        self.equity = self.cash + self.units * close_px
        self.hwm = max(self.hwm, self.equity)
        return self.equity

    @property
    def drawdown(self):
        return self.equity / self.hwm - 1


class PaperTrader:
    """
    봉 하나씩 처리하는 모의매매기. 상태 전체를 pickle로 저장/복원합니다.
    판단 규칙: argmax_a κ|a−p|ln(1−c_dec) + 평균 U(s, c_dec, a)  (백테스트와 동일)
    """

    def __init__(self, ens, cost=C.PRIMARY_COST, c_mult=C.P0["c_mult"], model_tag="", fallback=None):
        self.ens = ens
        self.fallback = fallback
        self.model_tag = model_tag
        self.cost = cost
        self.c_dec = C.c_dec(cost, c_mult)
        self.eng = FeatureEngine()
        self.acct = Account(cost)
        self.base = {k: Account(cost) for k in BASELINES}
        self.b3 = 0
        self.last_ts = None
        self.n_bars = 0
        self.log = []
        self.alarms = []
        self.max_dd_limit = None
        self.learning_frozen = False

    def _baseline_targets(self, x, c):
        e = self.eng
        rsi = 50 + 25 * x[FI["rsi_14"]]
        if rsi < 30:
            self.b3 = 1
        elif rsi > 70:
            self.b3 = 0
        m300, m1200 = e.sma[300].mean(), e.sma[1200].mean()
        return {"B0": 1, "B1": int(m300 > m1200), "B2": int(c > m1200), "B3": self.b3,
                "B4": int(x[FI["macdh_4h"]] > 0), "B5": int(x[FI["macdh_1d"]] > 0),
                "B6": int(x[FI["ret_180"]] > 0)}

    def on_bar(self, bar, trade=True, now=None):
        """
        살아 있는 4시간봉 하나. trade=False면 지표만 갱신(재시작 후 놓친 봉 따라잡기 — 소급 매매 없음).
        반환: 로그 dict
        """
        ts = int(bar["ts"])
        if self.last_ts is not None and ts <= self.last_ts:
            return None                                      # 중복·역순 봉 무시
        o, h, l, c, v = (float(bar[k]) for k in ("open", "high", "low", "close", "volume"))
        rec = dict(ts=ts, time=time.strftime("%Y-%m-%d %H:%M", time.gmtime(ts)), open=o, close=c)
        if trade:
            rec["fill"] = self.acct.fill(o)
            for k, acc in self.base.items():
                acc.fill(o)
        else:
            self.acct.pending = None
            for acc in self.base.values():
                acc.pending = None
        prev_sig = self.eng.sigma
        prev_c = self.eng.prev_c
        x = self.eng.update(o, h, l, c, v)
        self.n_bars += 1
        self.last_ts = ts
        rec["equity"] = self.acct.mark(c)
        for acc in self.base.values():
            acc.mark(c)
        if not trade:
            return rec
        hold_reason = None
        if bool(bar.get("forced_hold", False)):
            hold_reason = "gap"
        elif not self.eng.ready:
            hold_reason = "warmup"
        elif prev_c is not None and abs(math.log(c / prev_c)) > 10 * prev_sig:
            hold_reason = "jump>10σ"
            self.alarms.append(dict(ts=ts, alarm="jump"))
        if now is not None and now - (ts + BAR) > 2 * BAR:
            hold_reason = hold_reason or "stale_data"
        delta, umax = self.ens.delta(x[None, :], self.c_dec)
        d = float(delta[0])
        if (not math.isfinite(d) or umax >= 200) and self.fallback is not None:
            self.alarms.append(dict(ts=ts, alarm="model_fallback", umax=umax))
            self.ens, self.fallback = self.fallback, None
            delta, umax = self.ens.delta(x[None, :], self.c_dec)
            d = float(delta[0])
            hold_reason = hold_reason or "model_fallback"
        th = threshold(self.c_dec)
        p = self.acct.pos
        a = p
        if hold_reason is None and math.isfinite(d):
            if p == 0 and d > th:
                a = 1
            elif p == 1 and d < -th:
                a = 0
        if hold_reason == "stale_data" and a == 1 and p == 0:
            a = 0                                               # 데이터가 끊겼으면 새 매수 금지
        self.acct.pending = a
        for k, tgt in self._baseline_targets(x, c).items():
            self.base[k].pending = tgt if not bar.get("forced_hold", False) else self.base[k].pos
        if self.max_dd_limit and self.acct.drawdown < self.max_dd_limit and not self.learning_frozen:
            self.learning_frozen = True
            self.alarms.append(dict(ts=ts, alarm="drawdown_kill", dd=self.acct.drawdown))
        rec.update(delta=d, threshold=th, pos=p, decision=a, hold=hold_reason,
                   dd=self.acct.drawdown, model=self.model_tag,
                   base={k: v.equity for k, v in self.base.items()})
        self.log.append(rec)
        return rec


# ══════════ 저장·복원 ══════════
def save_state(trader, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(trader, f)
    os.replace(tmp, path)


def load_state(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def latest_model():
    p = os.path.join(MODEL_DIR, "p0_latest.npz")
    if not os.path.exists(p):
        raise SystemExit("모델 파일이 없습니다: python -m btc.live export 로 워크포워드 결과에서 내보내세요")
    z = dict(np.load(p))
    tag = bytes(z.pop("tag")).decode() if "tag" in z else "p0"
    return Ensemble.from_state(z), tag


def process(trader, venue, now=None, backfill_hours=12000, retries=18, wait=10.0, sleep=time.sleep):
    """
    새로 마감된 봉을 처리합니다. 처음이면 과거 봉으로 지표를 채웁니다(매매 없음).
    방금 끝난 4시간봉의 마지막 시간봉이 아직 안 왔으면 10초 간격으로 최대 3분 다시 받습니다.
    그래도 없으면 그 봉은 이번에 건너뛰고(관망) 경고를 남깁니다.
    """
    now = time.time() if now is None else now
    need = backfill_hours if trader.last_ts is None else int((now - trader.last_ts) // H) + 12
    expect = (int(now) // BAR) * BAR - BAR            # 방금 마감됐어야 할 4시간봉 시작 시각
    for k in range(retries + 1):
        h = venue.fetch_hourly(max(need, 12), now=now)
        have = set(int(x) for x in h["ts"]) if len(h) else set()
        if all(expect + i * H in have for i in range(4)) or k == retries:
            break
        sleep(wait)
        now += wait
    else:
        pass
    if not all(expect + i * H in have for i in range(4)):
        trader.alarms.append(dict(ts=expect, alarm="bar_incomplete_after_retry"))
    bars = hourly_to_4h(h)
    bars = bars[bars["ts"] + BAR <= now]
    if trader.last_ts is not None:
        bars = bars[bars["ts"] > trader.last_ts]
    out = []
    if trader.last_ts is None:
        # 첫 실행: 마지막 봉만 판단, 나머지는 지표 워밍업
        for i, (_, b) in enumerate(bars.iterrows()):
            out.append(trader.on_bar(b, trade=(i == len(bars) - 1), now=now))
    else:
        missed = len(bars) - 1
        for i, (_, b) in enumerate(bars.iterrows()):
            # 여러 봉을 놓쳤으면 마지막 봉만 판단 (소급 매매 없음)
            out.append(trader.on_bar(b, trade=(i == len(bars) - 1) or missed <= 0, now=now))
    return [r for r in out if r]


def export_model(run_name="P0", rep=0):
    """워크포워드 반복 0의 마지막 모델을 실운용 모델로 내보냅니다 (성적으로 고르지 않음, 사전 등록)"""
    from .walkforward import load_run
    run = load_run(run_name, rep)
    st = {k: v for k, v in run["last_state"].items() if not k.startswith("anchor")}
    anc = {k: v for k, v in run["last_state"].items() if k.startswith("anchor")}
    month = run["log"][-1]["month"]
    os.makedirs(MODEL_DIR, exist_ok=True)
    tag = f"p0_r{rep}_{month}"
    np.savez_compressed(os.path.join(MODEL_DIR, f"{tag}.npz"), **st, **anc)
    np.savez_compressed(os.path.join(MODEL_DIR, "p0_latest.npz"), **st, **anc,
                        tag=np.frombuffer(tag.encode(), dtype=np.uint8))
    return tag


def retrain(src=None, when=None):
    """
    매월 1일 모델 갱신 — 백테스트의 monthly_update()를 그대로 호출합니다.
    학습 데이터는 백테스트와 같은 비트스탬프 달러 1분봉 (src = ff137 저장소 클론, 먼저 git pull).
    """
    from .data import build_cache, CACHE_15M
    from .walkforward import load_phases, monthly_update
    T_k = pd.Timestamp(when or pd.Timestamp.now(tz="UTC").strftime("%Y-%m-01"))
    T_k = T_k.tz_localize("UTC") if T_k.tzinfo is None else T_k
    if src:
        build_cache(src, CACHE_15M, cutoff=T_k)          # 그 달 1일까지의 새 데이터 포함
    datas = load_phases()
    z = dict(np.load(os.path.join(MODEL_DIR, "p0_latest.npz")))
    z.pop("tag", None)
    anchor = [z[k] for k in sorted((k for k in z if k.startswith("anchor")), key=lambda s: int(s[6:]))]
    ens = Ensemble.from_state({k: v for k, v in z.items() if not k.startswith("anchor")})
    ens2, anchor2, entry = monthly_update(C.P0, datas, T_k, (0, T_k.year, T_k.month, 0), ens, anchor)
    st = ens2.state()
    for i, a in enumerate(anchor2):
        st[f"anchor{i}"] = a
    tag = f"p0_r0_{T_k.date()}"
    np.savez_compressed(os.path.join(MODEL_DIR, f"{tag}.npz"), **st)
    np.savez_compressed(os.path.join(MODEL_DIR, "p0_latest.npz"), **st,
                        tag=np.frombuffer(tag.encode(), dtype=np.uint8))
    return entry


def main():
    ap = argparse.ArgumentParser(description="BTC 강화학습 모의매매")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "once"):
        p = sub.add_parser(name)
        p.add_argument("--venue", default="upbit", choices=list(VENUES))
        p.add_argument("--cost", type=float, default=None, help="편도 비용 (기본: 업비트 0.10%%)")
        p.add_argument("--state", default=os.path.join(LIVE_DIR, "state.pkl"))
    sub.add_parser("status").add_argument("--state", default=os.path.join(LIVE_DIR, "state.pkl"))
    e = sub.add_parser("export")
    e.add_argument("--rep", type=int, default=0)
    r = sub.add_parser("retrain")
    r.add_argument("--src", default=None)
    r.add_argument("--month", default=None, help="YYYY-MM-01 (기본: 이번 달 1일)")
    a = ap.parse_args()

    if a.cmd == "export":
        print("내보냄:", export_model(rep=a.rep))
        return
    if a.cmd == "retrain":
        print(json.dumps(retrain(a.src, a.month), ensure_ascii=False, default=str))
        return
    if a.cmd == "status":
        t = load_state(a.state)
        print(json.dumps(dict(last=t.log[-1] if t.log else None, pos=t.acct.pos, equity=t.acct.equity,
                              dd=t.acct.drawdown, alarms=t.alarms[-5:], model=t.model_tag),
                         ensure_ascii=False, indent=1, default=str))
        return
    venue = VENUES[a.venue]()
    cost = a.cost if a.cost is not None else (C.COSTS["upbit"] if a.venue == "upbit" else C.PRIMARY_COST)
    if os.path.exists(a.state):
        trader = load_state(a.state)
    else:
        ens, tag = latest_model()
        trader = PaperTrader(ens, cost=cost, model_tag=tag)
    while True:
        for rec in process(trader, venue):
            if rec.get("delta") is not None:
                print(f"{rec['time']} 종가 {rec['close']:,.0f}  Δ={rec['delta']:+.3f} (문턱 ±{rec['threshold']:.2f})  "
                      f"보유 {rec['pos']}→{rec['decision']}  자산 {rec['equity']:.4f}  "
                      f"{'관망:' + rec['hold'] if rec['hold'] else ''}", flush=True)
        save_state(trader, a.state)
        with open(os.path.join(LIVE_DIR, "log.jsonl"), "a", encoding="utf-8") as f:
            for rec in trader.log[-1:]:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        if a.cmd == "once":
            return
        nxt = (int(time.time()) // BAR + 1) * BAR + 20
        time.sleep(max(5, nxt - time.time()))


if __name__ == "__main__":
    main()
