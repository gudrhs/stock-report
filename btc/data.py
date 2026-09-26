# -*- coding: utf-8 -*-
"""
시세 데이터 — 백테스트용 저장 파일과 4시간봉 만들기.

저장 파일: data/btc/btcusd_15m.csv.gz  (비트스탬프 BTC/USD 15분봉, 2012-01 ~ 2026-09-24)
  원본은 공개 저장소 ff137/bitstamp-btcusd-minute-data 의 1분봉입니다.
    git clone https://github.com/ff137/bitstamp-btcusd-minute-data
    python -m btc.data build --src bitstamp-btcusd-minute-data

  컬럼: ts(구간 시작 UTC 초), open, high, low, close, volume, active_min
  · 거래가 있었던 1분봉(active)만으로 만듭니다. 원본은 거래소 장애 구간을 거래량 0
    평봉으로 채워 두었는데, 그런 가짜 봉이 지표에 들어가지 않게 하려는 것입니다.
    open = 첫 거래 분의 시가, close = 마지막 거래 분의 종가, active_min = 거래 있던 분 수
  · 1분봉 튐 보정: 고가 ≤ 1.1×max(시가,종가), 저가 ≥ 0.9×min(시가,종가)

4시간봉: bars_4h(df15, phase)
  UTC 00·04·08·12·16·20시 경계(phase 0)가 실시간 매매 격자입니다. 학습에서는 15분씩
  밀린 격자 16개(phase 0~15)를 모두 써서 특정 봉 모양을 외우지 못하게 합니다.
  거래가 있던 분이 24분 미만인 4시간봉은 '죽은 봉'으로 보고 빼 버립니다.
  죽은 봉이 6개 이상 이어진 뒤 첫 봉에서는 매매하지 않습니다(forced_hold).
"""
import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(HERE, "..", "data", "btc"))
CACHE_15M = os.path.join(DATA_DIR, "btcusd_15m.csv.gz")

CUTOFF = pd.Timestamp("2026-09-25 00:00", tz="UTC")   # 이 시각 이후 데이터는 쓰지 않습니다
COLS = ["ts", "open", "high", "low", "close", "volume", "active_min"]
STALE_MIN = 24          # 4시간(240분) 중 거래가 있던 분이 이보다 적으면 죽은 봉
FORCED_HOLD_GAP = 6     # 죽은 봉이 이만큼 이어졌다가 살아난 첫 봉은 관망
N_PHASES = 16


def to_unix(idx):
    """DatetimeIndex → UTC 초 (pandas 버전마다 내부 해상도가 달라 뺄셈으로 계산)"""
    return np.asarray((idx - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1), dtype=np.int64)


def from_unix(ts):
    return pd.to_datetime(np.asarray(ts, dtype=np.int64), unit="s", utc=True)


# ══════════ 1분봉 → 15분봉 ══════════
def minutes_to_15m(m, cutoff=CUTOFF):
    """
    1분봉 DataFrame(timestamp, open, high, low, close, volume) → 15분봉 (거래 있던 분만 집계).
    거래가 한 번도 없던 15분 구간은 가격이 NaN, active_min 0 입니다.
    """
    m = m.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    m = m[m["timestamp"] < int(cutoff.timestamp())]
    idx = from_unix(m["timestamp"].to_numpy())
    o, c = m["open"].to_numpy(float), m["close"].to_numpy(float)
    hi = np.minimum(m["high"].to_numpy(float), 1.1 * np.maximum(o, c))
    lo = np.maximum(m["low"].to_numpy(float), 0.9 * np.minimum(o, c))
    v = m["volume"].to_numpy(float)
    act = v > 0
    a = pd.DataFrame({"open": o[act], "high": hi[act], "low": lo[act], "close": c[act], "volume": v[act]},
                     index=idx[act])
    g = a.resample("15min", label="left", closed="left")
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                        "close": g["close"].last(), "volume": g["volume"].sum(),
                        "active_min": g["open"].count()})
    # 1분봉이 존재하는 전 구간으로 넓힘 (거래 없는 구간도 행으로 남겨 두어야 봉이 완성됐는지 셀 수 있음)
    full = pd.date_range(idx[0].floor("15min"), (idx[-1]).floor("15min"), freq="15min", tz="UTC")
    n_min = pd.Series(1, index=idx).resample("15min", label="left", closed="left").count().reindex(full, fill_value=0)
    out = out.reindex(full)
    out["active_min"] = out["active_min"].fillna(0).astype(int)
    out["volume"] = out["volume"].fillna(0.0)
    out = out[n_min.to_numpy() == 15]          # 1분봉이 덜 찬 15분 구간(맨 처음·끝)은 제외
    out.insert(0, "ts", to_unix(out.index))
    return out


def build_cache(src_repo, out=CACHE_15M, cutoff=CUTOFF, extra_minutes=None):
    """
    cutoff: 이 시각 이후 1분봉은 버림. 연구용 캐시는 2026-09-25로 고정(재현성),
    실운용 월간 재학습은 그 달 1일 00:00 UTC를 넘겨 새 데이터까지 씁니다.
    extra_minutes: 저장소에 아직 안 올라온 최근 1분봉 (비트스탬프 API에서 직접 받은 것)
    """
    d = os.path.join(src_repo, "data")
    parts = [pd.read_csv(os.path.join(d, "historical", "btcusd_bitstamp_1min_2012-2025.csv.gz"))]
    upd = os.path.join(d, "updates", "btcusd_bitstamp_1min_latest.csv")
    if os.path.exists(upd):
        parts.append(pd.read_csv(upd))
    if extra_minutes is not None and len(extra_minutes):
        parts.append(extra_minutes[["timestamp", "open", "high", "low", "close", "volume"]])
    m = pd.concat(parts, ignore_index=True)
    q = minutes_to_15m(m, cutoff=cutoff)
    q = q.copy()
    for k in ("open", "high", "low", "close"):
        q[k] = q[k].round(2)
    q["volume"] = q["volume"].round(8)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    q[COLS].to_csv(out, index=False, compression={"method": "gzip", "mtime": 0})
    return q


def load_15m(path=CACHE_15M):
    df = pd.read_csv(path)
    df.index = from_unix(df["ts"].to_numpy())
    return df


# ══════════ 15분봉 → 4시간봉 ══════════
def bars_4h(df15, phase=0, stale_min=STALE_MIN):
    """
    phase k의 4시간봉 중 살아 있는 봉만. 컬럼:
      ts, open, high, low, close, volume, active_min, gap_before, forced_hold
    gap_before = 이 봉 바로 앞에서 빠진 죽은 봉 수
    """
    if not 0 <= phase < N_PHASES:
        raise ValueError("phase는 0~15")
    off = pd.Timedelta(minutes=15 * phase)
    g = df15.resample("4h", label="left", closed="left", offset=off)
    b = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                      "close": g["close"].last(), "volume": g["volume"].sum(),
                      "active_min": g["active_min"].sum(), "n15": g["ts"].count()})
    full = (b["n15"] == 16).to_numpy()
    if not full.any():
        return pd.DataFrame(columns=["ts"] + list(b.columns.drop("n15")) + ["gap_before", "forced_hold"])
    i0, i1 = int(np.argmax(full)), len(full) - int(np.argmax(full[::-1]))
    b = b.iloc[i0:i1]                            # 맨 앞·끝의 미완성 봉만 잘라냄
    # 가운데에서 15분봉이 빠진 봉(원본 1분봉 누락)도 '죽은 봉'으로 셉니다 — 조용히 사라지면
    # 그 앞뒤 봉이 붙어 있는 것처럼 보여 수익·연속성 계산이 틀어집니다.
    alive = ((b["n15"] == 16).to_numpy() & (b["active_min"] >= stale_min).to_numpy()
             & b["close"].notna().to_numpy())
    b = b.drop(columns="n15")
    # 살아 있는 봉마다 바로 앞에 연속으로 빠진 봉 수
    dead_run = np.zeros(len(b), dtype=np.int64)
    run = 0
    for i, ok in enumerate(alive):
        dead_run[i] = run
        run = 0 if ok else run + 1
    b["gap_before"] = dead_run
    b = b[alive].copy()
    b["forced_hold"] = b["gap_before"] >= FORCED_HOLD_GAP
    b.insert(0, "ts", to_unix(b.index))
    b["active_min"] = b["active_min"].astype(int)
    return b


def main():
    ap = argparse.ArgumentParser(description="BTC 15분봉 캐시 만들기")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="1분봉 저장소에서 15분봉 캐시 생성")
    b.add_argument("--src", required=True, help="ff137/bitstamp-btcusd-minute-data 클론 경로")
    b.add_argument("--out", default=CACHE_15M)
    a = ap.parse_args()
    if a.cmd == "build":
        q = build_cache(a.src, a.out)
        print(f"{len(q):,}개 15분봉 저장 → {a.out}  ({q.index[0]} ~ {q.index[-1]})")


if __name__ == "__main__":
    main()
