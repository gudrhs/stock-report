# -*- coding: utf-8 -*-
"""
실시간 모의매매 — 백테스트와 같은 지표 엔진·판단 규칙·체결 회계를 그대로 씁니다.

  python -m btc.live run  --venue upbit --retrain-src <1분봉 저장소>   ← 권장 (월초 자동 재학습)
  python -m btc.live once --venue upbit --retrain-src <1분봉 저장소>   ← cron·작업 스케줄러용 (UTC 00·04·…·20시 +20초)
  python -m btc.live status
  python -m btc.live retrain --src <1분봉 저장소>                       ← 수동 재학습 (보통은 필요 없음)

흐름 (4시간봉 마감 + 20초마다)
  1. 거래소에서 1시간봉을 받아 직접 UTC 4시간봉(00·04·08·12·16·20시)으로 묶습니다. 아직 끝나지 않은
     봉은 버리고(거래소 서버 시각 기준), 4개 시간봉 중 하나라도 없거나 거래량이 0이면 '죽은 봉'입니다.
     마지막 시간봉이 늦으면 10초 간격으로 3분까지 다시 받습니다.
  2. 월초(1일 00:00 UTC 마감 봉)면 판단 전에 먼저 재학습합니다 — 백테스트와 같은 시점, 같은 코드.
     여러 달을 건너뛰었으면 빠진 달을 순서대로 모두 재학습합니다(1월은 처음부터).
  3. 직전에 낸 주문을 이번 봉 '시가'로 체결 (백테스트와 같은 가정: 편도 비용 c)
  4. 지표 갱신 → 앙상블 판단 → 다음 봉 시가에 낼 주문 등록
  5. 상태 저장(임시 파일 → 이름 바꾸기) + 이번에 판단한 봉만 로그에 한 줄씩

안전장치
  · 데이터가 2봉 넘게 끊기면 새 매수 금지(매도는 허용) + 경고
  · 한 봉 수익이 10σ를 넘으면 두 번째 거래소(--confirm-venue)로 확인. 확인 안 되면 그 봉은 관망
  · 모델 출력이 NaN이거나 |U|≥200이면 관망 (이전 모델이 있으면 그걸로 복귀)
  · 봉 마감 5분이 지나서야 판단하게 되면(프로그램이 늦게 깸) 그 봉은 매매하지 않음 — 지난 가격 체결 방지
  · 모의 낙폭이 백테스트 최악 낙폭의 1.2배를 넘으면 학습·모델 교체 동결 + 경고
  · 재학습·조회가 실패해도 멈추지 않고 경고를 남긴 뒤 이전 모델로 계속
  · 시작할 때 지표 엔진이 일괄 계산과 1e-8 안에서 같은지 확인
  · --gate-venue bitstamp: 같은 기간을 비트스탬프 가격으로 판단했을 때와 85% 이상 일치하는지 확인

아직 구현하지 않은 것 (설계서 §11 중): 호가창 VWAP 체결 비용 기록, 지표 분포 이탈(PSI) 감시,
봉마다 온라인 학습(X2) 모의 트랙. 실제 주문은 넣지 않습니다(모의매매 전용).
업비트는 원화, 백테스트는 달러(비트스탬프) 기준이라 환율·김치프리미엄은 반영되지 않습니다.
"""
import argparse
import email.utils
import json
import math
import os
import pickle
import time
import traceback

import numpy as np
import pandas as pd

from . import config as C
from .agent import Ensemble, threshold, policy_from_delta
from .data import DATA_DIR, FORCED_HOLD_GAP
from .features import FeatureEngine, NAMES, compute

LIVE_DIR = os.path.join(DATA_DIR, "live")
MODEL_DIR = os.path.join(DATA_DIR, "models")
H = 3600
BAR = 4 * H
LATE_SEC = 300              # 봉 마감 후 이보다 늦게 판단하면 그 봉은 매매하지 않음
FI = {n: i for i, n in enumerate(NAMES)}
BASELINES = ("B0", "B1", "B2", "B3", "B4", "B5", "B6")
SERVER = {"skew": None}     # 거래소 서버 시각 − 내 컴퓨터 시각 (초)


# ══════════ 거래소 1시간봉 ══════════
def _get_json(url, timeout=15, tries=4):
    import urllib.request
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "stock-report-btc/1.0",
                                                       "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = r.headers.get("Date")
                if d:
                    try:
                        SERVER["skew"] = email.utils.parsedate_to_datetime(d).timestamp() - time.time()
                    except Exception:
                        pass
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:                      # 2·4·8초 쉬고 다시
            last = e
            if k < tries - 1:
                time.sleep(2 ** (k + 1))
    raise RuntimeError(f"조회 실패: {url} ({last})")


def server_now():
    """거래소 서버 시각 (HTTP Date 헤더로 잰 차이를 반영). 모르면 내 시각"""
    sk = SERVER["skew"]
    return time.time() + (sk if sk is not None else 0.0)


EMPTY = pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])


class Venue:
    """1시간봉 공급자. fetch_hourly(n, now)는 '마감된' 봉만 ts 오름차순 DataFrame으로 돌려줍니다."""
    name = "base"
    currency = "?"

    def fetch_hourly(self, n, now=None):
        raise NotImplementedError

    @staticmethod
    def closed_only(df, now):
        if df is None or len(df) == 0:
            return EMPTY.copy()
        df = df.drop_duplicates("ts", keep="last").sort_values("ts").reset_index(drop=True)
        return df[df["ts"] + H <= now].reset_index(drop=True)


class Upbit(Venue):
    name, currency = "upbit", "KRW"

    def __init__(self, market="KRW-BTC"):
        self.market = market

    def fetch_hourly(self, n, now=None):
        now = server_now() if now is None else now
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
        if not rows:
            return EMPTY.copy()
        df = pd.DataFrame({
            "ts": [int(pd.Timestamp(r["candle_date_time_utc"], tz="UTC").timestamp()) for r in rows],
            "open": [float(r["opening_price"]) for r in rows], "high": [float(r["high_price"]) for r in rows],
            "low": [float(r["low_price"]) for r in rows], "close": [float(r["trade_price"]) for r in rows],
            "volume": [float(r["candle_acc_trade_volume"]) for r in rows]})
        return self.closed_only(df, now)


class Binance(Venue):
    name, currency = "binance", "USDT"

    def __init__(self, symbol="BTCUSDT"):
        self.symbol = symbol

    def fetch_hourly(self, n, now=None):
        now = server_now() if now is None else now
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
        if not rows:
            return EMPTY.copy()
        df = pd.DataFrame({"ts": [int(r[0]) // 1000 for r in rows], "open": [float(r[1]) for r in rows],
                           "high": [float(r[2]) for r in rows], "low": [float(r[3]) for r in rows],
                           "close": [float(r[4]) for r in rows], "volume": [float(r[5]) for r in rows]})
        return self.closed_only(df, now).tail(n).reset_index(drop=True)


class Bitstamp(Venue):
    name, currency = "bitstamp", "USD"

    def fetch_hourly(self, n, now=None):
        now = int(server_now() if now is None else now)
        rows, end = [], now
        while len(rows) < n:
            js = _get_json(f"https://www.bitstamp.net/api/v2/ohlc/btcusd/?step=3600&limit=1000&end={end}")
            got = js.get("data", {}).get("ohlc", [])
            if not got:
                break
            rows = got + rows
            end = int(got[0]["timestamp"]) - 1
            if len(got) < 1000:
                break
        if not rows:
            return EMPTY.copy()
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
    cols = ["ts", "open", "high", "low", "close", "volume", "gap_before", "forced_hold"]
    if len(h) == 0:
        return pd.DataFrame(columns=cols)
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

    def __init__(self, ens, cost=C.PRIMARY_COST, c_mult=C.P0["c_mult"], model_tag="", fallback=None,
                 venue="?", dd_limit=None):
        self.ens = ens
        self.fallback = fallback
        self.model_tag = model_tag
        self.venue = venue
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
        self.max_dd_limit = dd_limit          # 음수 (예: −0.9). 넘으면 학습 동결
        self.learning_frozen = False
        self.bars = []                         # 최근 봉 (시작 점검용)

    def alarm(self, **kw):
        kw.setdefault("ts", self.last_ts)
        kw["at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.alarms.append(kw)

    def _baseline_targets(self, x, c):
        """B0~B6 목표 — B3(RSI 상태)는 매매하지 않는 봉에서도 반드시 갱신"""
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

    def on_bar(self, bar, trade=True, now=None, confirm=None):
        """
        살아 있는 4시간봉 하나.
          trade=False: 지표만 갱신 (첫 실행 워밍업, 재시작·지연으로 밀린 봉). 다만 이미 실시간으로 낸
                       주문(pending)은 그 봉 시가에 체결합니다 — 주문은 제때 나갔고 데이터만 늦게 온 것.
          confirm(ts, r): 10σ 급변을 다른 거래소로 확인하는 함수 (True/False/None)
        반환: 로그 dict (매매 봉에는 decision 포함)
        """
        ts = int(bar["ts"])
        if self.last_ts is not None and ts <= self.last_ts:
            return None                                      # 중복·역순 봉 무시
        o, h, l, c, v = (float(bar[k]) for k in ("open", "high", "low", "close", "volume"))
        rec = dict(ts=ts, time=time.strftime("%Y-%m-%d %H:%M", time.gmtime(ts)), open=o, close=c)
        rec["fill"] = self.acct.fill(o)
        for acc in self.base.values():
            acc.fill(o)
        prev_sig, prev_c = self.eng.sigma, self.eng.prev_c
        x = self.eng.update(o, h, l, c, v)
        self.n_bars += 1
        self.last_ts = ts
        self.bars.append((ts, o, h, l, c, v))
        del self.bars[:-3000]
        rec["equity"] = self.acct.mark(c)
        for acc in self.base.values():
            acc.mark(c)
        tg_base = self._baseline_targets(x, c)
        if not trade:
            return rec
        hold_reason = None
        if bool(bar.get("forced_hold", False)):
            hold_reason = "gap"
        elif not self.eng.ready:
            hold_reason = "warmup"
        elif prev_c is not None and abs(math.log(c / prev_c)) > 10 * prev_sig:
            ok = confirm(ts, math.log(c / prev_c)) if confirm else None
            self.alarm(ts=ts, alarm="jump", r=math.log(c / prev_c), sigma=prev_sig, confirmed=ok)
            if not ok:
                hold_reason = "jump>10σ"
        delta, umax = self.ens.delta(x[None, :], self.c_dec)
        d, umax = float(delta[0]), float(umax[0])
        if not math.isfinite(d) or umax >= 200:
            # 모델 이상: 경고 + 이번 봉은 관망. 직전 모델이 있으면 그걸로 되돌림
            self.alarm(ts=ts, alarm="model_bad", umax=umax, delta=d)
            hold_reason = hold_reason or "model_bad"
            if self.fallback is not None:
                self.ens, self.fallback = self.fallback, None
                self.model_tag += "→fallback"
        th = threshold(self.c_dec)
        p = self.acct.pos
        a = p
        if hold_reason is None and math.isfinite(d):
            if p == 0 and d > th:
                a = 1
            elif p == 1 and d < -th:
                a = 0
        if now is not None and now - (ts + BAR) > 2 * BAR and a == 1 and p == 0:
            a = 0                                               # 데이터가 끊겼으면 새 매수만 금지 (매도는 허용)
            hold_reason = "stale_data_no_entry"
            self.alarm(ts=ts, alarm="stale_data", age_h=round((now - ts - BAR) / 3600, 1))
        self.acct.pending = a
        for k, tgt in tg_base.items():
            self.base[k].pending = tgt if not bar.get("forced_hold", False) else self.base[k].pos
        if self.max_dd_limit is not None and self.acct.drawdown < self.max_dd_limit and not self.learning_frozen:
            self.learning_frozen = True
            self.alarm(ts=ts, alarm="drawdown_kill", dd=self.acct.drawdown, limit=self.max_dd_limit)
        rec.update(delta=d, threshold=th, pos=p, decision=a, hold=hold_reason,
                   dd=self.acct.drawdown, model=self.model_tag,
                   decided_at=now, base={k: v.equity for k, v in self.base.items()})
        self.log.append(rec)
        return rec

    def parity_check(self):
        """증분 엔진 == 일괄 계산 (시작·재시작 때). 다르면 경고하고 일괄 계산 상태로 다시 만듦"""
        if len(self.bars) < 50:
            return True
        df = pd.DataFrame(self.bars, columns=["ts", "open", "high", "low", "close", "volume"])
        X, _ = compute(df)
        if self.eng.n != len(self.bars):
            return True                                          # 전체 이력을 다 들고 있지 않으면 생략
        diff = float(np.max(np.abs(X[-1] - self.eng.last)))
        if diff > 1e-8:
            self.alarm(alarm="parity", diff=diff)
            eng = FeatureEngine()
            for row in self.bars:
                eng.update(*row[1:])
            self.eng = eng
            return False
        return True


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


def _save_npz_atomic(path, **arrs):
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrs)
    os.replace(tmp, path)


def _read_model(path):
    z = dict(np.load(path))
    tag = bytes(z.pop("tag")).decode() if "tag" in z else None
    dd = float(z.pop("dd_limit")) if "dd_limit" in z else None
    for k in ("code_hash", "cfg_hash"):
        z.pop(k, None)
    st = {k: v for k, v in z.items() if not k.startswith("anchor")}
    anc = [z[k] for k in sorted((k for k in z if k.startswith("anchor")), key=lambda s: int(s[6:]))]
    return Ensemble.from_state(st), anc, tag, dd


def latest_model(model_dir=None):
    p = os.path.join(model_dir or MODEL_DIR, "p0_latest.npz")
    if not os.path.exists(p):
        raise SystemExit("모델 파일이 없습니다: python -m btc.live export 로 워크포워드 결과에서 내보내세요")
    ens, _, tag, dd = _read_model(p)
    return ens, tag or "p0", dd


def model_month(tag):
    """'p0_r0_2026-10-01' → 2026-10-01 00:00 UTC (초). 알 수 없으면 None"""
    try:
        return int(pd.Timestamp(tag.split("_")[-1].split("→")[0], tz="UTC").timestamp())
    except Exception:
        return None


def maybe_swap_model(trader, model_dir=None):
    """p0_latest.npz가 바뀌었으면 새 모델로 교체 (이전 모델은 비상용으로 보관). 읽기 실패는 경고만"""
    if trader.learning_frozen:
        return False
    path = os.path.join(model_dir or MODEL_DIR, "p0_latest.npz")
    if not os.path.exists(path):
        return False
    try:
        ens, _, tag, dd = _read_model(path)
    except Exception as e:
        trader.alarm(alarm="model_read_failed", err=str(e)[:200])
        return False
    if not tag or tag == trader.model_tag.split("→")[0]:
        return False
    trader.fallback, trader.ens = trader.ens, ens
    trader.alarm(alarm="model_update", old=trader.model_tag, new=tag)
    trader.model_tag = tag
    if dd is not None:
        trader.max_dd_limit = dd
    return True


def month_starts_between(after_ts, upto_ts):
    """after_ts 달 다음 달 1일부터 upto_ts가 속한 달 1일까지 (각 00:00 UTC)"""
    a = pd.Timestamp(after_ts, unit="s", tz="UTC")
    b = pd.Timestamp(upto_ts, unit="s", tz="UTC")
    return list(pd.date_range((a + pd.offsets.MonthBegin(1)).normalize(), b, freq="MS", tz="UTC"))


def process(trader, venue, now=None, backfill_hours=12000, retries=18, wait=10.0, sleep=time.sleep,
            retrain_fn=None, model_dir=None, confirm=None):
    """
    새로 마감된 봉을 처리합니다. 처음이면 과거 봉으로 지표를 채웁니다(매매 없음).
    방금 끝난 4시간봉의 마지막 시간봉이 아직 안 왔으면 10초 간격으로 최대 3분 다시 받습니다.
    반환: 이번에 처리한 봉 기록 목록
    """
    now = server_now() if now is None else now
    need = backfill_hours if trader.last_ts is None else int((now - trader.last_ts) // H) + 12
    expect = (int(now) // BAR) * BAR - BAR            # 방금 마감됐어야 할 4시간봉 시작 시각
    have = set()
    for k in range(retries + 1):
        h = venue.fetch_hourly(max(need, 12), now=now)
        have = set(int(x) for x in h["ts"]) if len(h) else set()
        if all(expect + i * H in have for i in range(4)) or k == retries:
            break
        sleep(wait)
        now += wait
    if not all(expect + i * H in have for i in range(4)):
        trader.alarm(ts=expect, alarm="bar_incomplete_after_retry")
    bars = hourly_to_4h(h)
    bars = bars[bars["ts"] + BAR <= now]
    if trader.last_ts is not None:
        bars = bars[bars["ts"] > trader.last_ts]
    if not len(bars):
        return []
    # 월초: 백테스트처럼 그달 1일 00:00에 마감하는 봉부터 새 모델로 판단 — 빠진 달은 순서대로 모두
    last_close = int(bars["ts"].iloc[-1]) + BAR
    mm = model_month(trader.model_tag)
    if mm is None:
        trader.alarm(alarm="model_month_unknown", tag=trader.model_tag)
    elif not trader.learning_frozen:
        todo = month_starts_between(mm, last_close) if mm < last_close else []
        todo = [T for T in todo if int(T.timestamp()) <= last_close]
        if todo and retrain_fn is None:
            trader.alarm(alarm="model_stale", model=trader.model_tag, months_behind=len(todo))
        for T in (todo if retrain_fn else []):
            try:
                retrain_fn(T)
                trader.alarm(alarm="retrained", month=str(T.date()))
            except BaseException as e:                     # 실패해도 매매는 이전 모델로 계속
                if isinstance(e, KeyboardInterrupt):
                    raise
                trader.alarm(alarm="retrain_failed", month=str(T.date()), err=str(e)[:300])
                break
    maybe_swap_model(trader, model_dir)
    out = []
    n = len(bars)
    first_run = trader.last_ts is None
    for i, (_, b) in enumerate(bars.iterrows()):
        is_last = i == n - 1
        late = now - (int(b["ts"]) + BAR) > LATE_SEC
        trade = is_last and not first_run and not late
        if is_last and late and not first_run:
            trader.alarm(ts=int(b["ts"]), alarm="late_decision", late_sec=int(now - int(b["ts"]) - BAR))
        if not is_last and not first_run:
            trader.alarm(ts=int(b["ts"]), alarm="replayed_bar")
        rec = trader.on_bar(b, trade=trade, now=now, confirm=confirm)
        if rec:
            out.append(rec)
    if first_run:
        trader.parity_check()
    return out


def export_model(run_name="P0", rep=0, report=None):
    """
    워크포워드 반복 0의 마지막 모델을 실운용 모델로 내보냅니다 (성적으로 고르지 않음, 사전 등록).
    코드·설정 해시가 지금 코드와 다르면 거부합니다. report(평가 결과)가 있으면 낙폭 한도(1.2×)를 같이 저장.
    """
    from .evaluate import load_run                       # 해시 확인하는 쪽
    from .walkforward import code_hash
    run = load_run(run_name, rep)
    st = dict(run["last_state"])
    month = run["log"][-1]["month"]
    tag = f"p0_r{rep}_{month}"
    extra = dict(tag=np.frombuffer(tag.encode(), dtype=np.uint8),
                 code_hash=np.frombuffer(code_hash().encode(), dtype=np.uint8),
                 cfg_hash=np.frombuffer(C.train_hash(C.P0).encode(), dtype=np.uint8))
    if report is not None:
        worst = min(r["max_dd"] for r in report["reps"] if r["rep"] == report["headline_rep"])
        extra["dd_limit"] = np.array(1.2 * worst)
    os.makedirs(MODEL_DIR, exist_ok=True)
    _save_npz_atomic(os.path.join(MODEL_DIR, f"{tag}.npz"), **st, **extra)
    _save_npz_atomic(os.path.join(MODEL_DIR, "p0_latest.npz"), **st, **extra)
    return tag


def retrain(src=None, when=None, model_dir=None, live_dir=None):
    """
    매월 1일 모델 갱신 — 백테스트의 monthly_update()를 그대로 호출합니다.
    학습 데이터는 백테스트와 같은 비트스탬프 달러 1분봉 (src = ff137 저장소 클론). 저장소에 아직 없는
    최근 분(하루 1회 갱신)은 비트스탬프 API에서 직접 받아 T_k 00:00까지 채웁니다.
    연구용 데이터 파일(data/btc/btcusd_15m.csv.gz)은 건드리지 않고 data/btc/live/ 에 따로 만듭니다.
    """
    from . import data as D
    from . import walkforward as W
    model_dir = model_dir or MODEL_DIR
    live_dir = live_dir or LIVE_DIR
    T_k = pd.Timestamp(when or pd.Timestamp.now(tz="UTC").strftime("%Y-%m-01"))
    T_k = T_k.tz_localize("UTC") if T_k.tzinfo is None else T_k
    tag = f"p0_r0_{T_k.date()}"
    latest = os.path.join(model_dir, "p0_latest.npz")
    ens, anchor, cur_tag, dd = _read_model(latest)
    if cur_tag == tag:
        return dict(month=str(T_k.date()), note="already_retrained")      # 두 번 돌려도 한 번만
    cache = os.path.join(live_dir, "btcusd_15m.csv.gz")
    if src:
        last = _last_minute(src)
        extra = fetch_bitstamp_minutes(last + 60, int(T_k.timestamp())) if last + 60 < T_k.timestamp() else None
        D.build_cache(src, cache, cutoff=T_k, extra_minutes=extra)       # 그 달 1일 00:00까지
    if not os.path.exists(cache):
        raise RuntimeError("실운용 학습 데이터가 없습니다 (--src 로 1분봉 저장소 경로를 주세요)")
    q = D.load_15m(cache)
    if int(q["ts"].iloc[-1]) + 900 < int(T_k.timestamp()) - 86400:
        raise RuntimeError(f"학습 데이터가 오래됐습니다 (마지막 {q.index[-1]}). 1분봉 저장소를 git pull 하세요.")
    # 백테스트 코드(load_phases)를 그대로 쓰되 실운용 데이터 파일·캐시를 보게 함
    saved = (W.DATA_DIR, W.CACHE_DIR, W.load_15m)
    try:
        W.DATA_DIR, W.CACHE_DIR = live_dir, os.path.join(live_dir, "cache")
        W.load_15m = lambda: D.load_15m(cache)
        datas = W.load_phases()
    finally:
        W.DATA_DIR, W.CACHE_DIR, W.load_15m = saved
    ens2, anchor2, entry = W.monthly_update(C.P0, datas, T_k, (0, T_k.year, T_k.month, 0), ens, anchor)
    st = ens2.state()
    for i, a in enumerate(anchor2):
        st[f"anchor{i}"] = a
    extra = dict(tag=np.frombuffer(tag.encode(), dtype=np.uint8))
    if dd is not None:
        extra["dd_limit"] = np.array(dd)
    _save_npz_atomic(os.path.join(model_dir, f"{tag}.npz"), **st, **extra)
    _save_npz_atomic(latest, **st, **extra)
    return entry


def _last_minute(src):
    upd = os.path.join(src, "data", "updates", "btcusd_bitstamp_1min_latest.csv")
    f = upd if os.path.exists(upd) else os.path.join(src, "data", "historical", "btcusd_bitstamp_1min_2012-2025.csv.gz")
    return int(pd.read_csv(f, usecols=["timestamp"])["timestamp"].max())


def fetch_bitstamp_minutes(start, end):
    """비트스탬프 1분봉 [start, end) — 저장소(하루 1회 갱신)에 아직 없는 최근 분만 직접 받습니다"""
    rows, t = [], int(start)
    while t < end:
        js = _get_json(f"https://www.bitstamp.net/api/v2/ohlc/btcusd/?step=60&limit=1000&start={t}")
        got = js.get("data", {}).get("ohlc", [])
        if not got:
            break
        rows += got
        t = int(got[-1]["timestamp"]) + 60
        time.sleep(0.2)
    if not rows:
        return None
    df = pd.DataFrame(rows).astype(float)
    df["timestamp"] = df["timestamp"].astype("int64")
    return df[(df["timestamp"] >= start) & (df["timestamp"] < end)].drop_duplicates("timestamp")


def venue_agreement(ens, venue_a, venue_b, c_dec, hours=12000, now=None):
    """
    두 거래소 가격으로 각각 판단했을 때 최근 봉들에서 판단이 얼마나 일치하는지 (설계서: 85% 이상).
    업비트(원화) 지표로 한 판단이 백테스트(비트스탬프 달러)와 다른 정책이 되지 않았는지 확인합니다.
    """
    pos = []
    for v in (venue_a, venue_b):
        b = hourly_to_4h(v.fetch_hourly(hours, now=now))
        X, _ = compute(b)
        dl, _ = ens.delta(X, c_dec)
        pos.append(pd.Series(policy_from_delta(dl, c_dec, b["forced_hold"].to_numpy()), index=b["ts"].to_numpy()))
    j = pos[0].index.intersection(pos[1].index)
    j = j[2400:] if len(j) > 2400 + 180 else j[-180:]
    return float((pos[0][j] == pos[1][j]).mean()) if len(j) else None


def confirm_with(venue2):
    """10σ 급변을 두 번째 거래소의 같은 봉 수익률로 확인 (차이 3%p 이내면 진짜로 봄)"""
    def f(ts, r):
        try:
            b = hourly_to_4h(venue2.fetch_hourly(24))
            b = b.set_index("ts")
            if ts not in b.index:
                return None
            i = b.index.get_loc(ts)
            if i == 0:
                return None
            r2 = math.log(b["close"].iloc[i] / b["close"].iloc[i - 1])
            return abs(r2 - r) <= 0.03
        except Exception:
            return None
    return f


def main():
    ap = argparse.ArgumentParser(description="BTC 강화학습 모의매매")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "once"):
        p = sub.add_parser(name)
        p.add_argument("--venue", default="upbit", choices=list(VENUES))
        p.add_argument("--cost", type=float, default=None, help="편도 비용 (기본: 업비트 0.10%%, 그 외 0.15%%)")
        p.add_argument("--state", default=os.path.join(LIVE_DIR, "state.pkl"))
        p.add_argument("--retrain-src", default=None,
                       help="1분봉 저장소 경로 — 주면 매월 1일 첫 판단 전에 자동으로 재학습 (백테스트와 같은 시점)")
        p.add_argument("--confirm-venue", default=None, choices=list(VENUES), help="10σ 급변 확인용 두 번째 거래소")
        p.add_argument("--gate-venue", default=None, choices=list(VENUES),
                       help="시작할 때 판단 일치율(≥85%%) 확인용 기준 거래소 (예: bitstamp)")
        p.add_argument("--new-track", action="store_true", help="거래소·비용이 다른 기존 상태가 있으면 새로 시작")
    sub.add_parser("status").add_argument("--state", default=os.path.join(LIVE_DIR, "state.pkl"))
    e = sub.add_parser("export")
    e.add_argument("--rep", type=int, default=0)
    r = sub.add_parser("retrain")
    r.add_argument("--src", default=None)
    r.add_argument("--month", default=None, help="YYYY-MM-01 (기본: 이번 달 1일)")
    a = ap.parse_args()

    if a.cmd == "export":
        rep_path = os.path.join(DATA_DIR, "report_full.json")
        report = json.load(open(rep_path, encoding="utf-8")) if os.path.exists(rep_path) else None
        print("내보냄:", export_model(rep=a.rep, report=report))
        return
    if a.cmd == "retrain":
        print(json.dumps(retrain(a.src, a.month), ensure_ascii=False, default=str))
        return
    if a.cmd == "status":
        t = load_state(a.state)
        print(json.dumps(dict(last=t.log[-1] if t.log else None, venue=t.venue, pos=t.acct.pos,
                              equity=t.acct.equity, dd=t.acct.drawdown, alarms=t.alarms[-8:], model=t.model_tag,
                              frozen=t.learning_frozen,
                              baselines={k: v.equity for k, v in t.base.items()}),
                         ensure_ascii=False, indent=1, default=str))
        return
    venue = VENUES[a.venue]()
    cost = a.cost if a.cost is not None else (C.COSTS["upbit"] if a.venue == "upbit" else C.PRIMARY_COST)
    trader = None
    if os.path.exists(a.state):
        trader = load_state(a.state)
        if (trader.venue != a.venue or abs(trader.cost - cost) > 1e-12):
            if not a.new_track:
                raise SystemExit(f"저장된 상태는 {trader.venue}/{trader.cost}입니다. 다른 거래소·비용으로 이어 가면 "
                                 f"지표가 섞입니다. 새로 시작하려면 --new-track (또는 --state 다른 파일)")
            os.replace(a.state, a.state + f".{int(time.time())}.bak")
            trader = None
    if trader is None:
        ens, tag, dd = latest_model()
        trader = PaperTrader(ens, cost=cost, model_tag=tag, venue=a.venue, dd_limit=dd)
        os.makedirs(LIVE_DIR, exist_ok=True)
        if a.gate_venue:
            agree = venue_agreement(ens, venue, VENUES[a.gate_venue](), trader.c_dec)
            trader.alarm(alarm="venue_agreement", value=agree, ok=bool(agree is not None and agree >= 0.85))
            print(f"판단 일치율 {a.venue} vs {a.gate_venue}: {agree}")

    def _retrain(T_k):
        import subprocess
        subprocess.run(["git", "-C", a.retrain_src, "pull", "--ff-only"], check=False)
        print("월간 재학습:", json.dumps(retrain(a.retrain_src, str(T_k.date())), ensure_ascii=False, default=str))
    retrain_fn = _retrain if a.retrain_src else None
    confirm = confirm_with(VENUES[a.confirm_venue]()) if a.confirm_venue else None
    log_path = os.path.join(LIVE_DIR, "log.jsonl")
    while True:
        try:
            recs = process(trader, venue, retrain_fn=retrain_fn, confirm=confirm)
            for rec in recs:
                if rec.get("delta") is not None:
                    print(f"{rec['time']} 종가 {rec['close']:,.0f}  Δ={rec['delta']:+.3f} (문턱 ±{rec['threshold']:.2f})  "
                          f"보유 {rec['pos']}→{rec['decision']}  자산 {rec['equity']:.4f}  "
                          f"{'관망:' + rec['hold'] if rec['hold'] else ''}", flush=True)
            save_state(trader, a.state)
            with open(log_path, "a", encoding="utf-8") as f:     # 이번에 판단한 봉만 (중복 없음)
                for rec in recs:
                    if rec.get("decision") is not None:
                        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        except KeyboardInterrupt:
            save_state(trader, a.state)
            raise
        except Exception as e:                                   # 조회 실패 등: 멈추지 않고 경고만
            trader.alarm(alarm="cycle_error", err=str(e)[:300], tb=traceback.format_exc()[-800:])
            save_state(trader, a.state)
            print("경고:", e, flush=True)
            if a.cmd == "once":
                raise SystemExit(1)
        if a.cmd == "once":
            return
        nxt = (int(server_now()) // BAR + 1) * BAR + 20
        time.sleep(max(5, nxt - server_now()))


if __name__ == "__main__":
    main()
