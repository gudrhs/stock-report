# -*- coding: utf-8 -*-
"""
확증 시험(2단계)용 보지 않은 코인 데이터 만들기 — data/btc/holdout/<자산>_15m.csv.gz

  원본은 모두 GitHub에 있는 공개 사본입니다 (출처·커밋은 data/btc/holdout/README.md).
  python -m btc.research.holdout_data ETH --src <내려받은 폴더>      (자산마다 한 번)

맹검 규칙: 이 파일은 수익률·샤프·낙폭·매수보유 성적을 계산하지 않습니다. 하는 일은
행 수, 시각 범위, 빠진 구간·중복, 가격 > 0, 고가·저가 모순, 거래량 ≥ 0, 원본끼리 같은 시각의
가격 '차이(%)', 그리고 한 봉만 50% 넘게 튀었다가 바로 돌아오는 '깨진 봉'의 개수뿐입니다.

15분봉 만들기 (btc/data.py 와 같은 규칙)
  · 1분봉 원본: btc.data.minutes_to_15m 을 그대로 씁니다 (거래 있던 분만 집계, 1분봉 튐 보정,
    1분봉 15개가 다 있는 15분 구간만, active_min = 거래 있던 분 수). 비트코인과 한 줄도 다르지 않습니다.
  · 5분봉 원본: 같은 규칙을 5분 단위로 (튐 보정도 5분봉 단위). active_min 은 '거래 있던 분 수'를 알 수
    없어서 추정합니다 — 거래 있던 5분봉 하나를 min(5, 체결 건수)분으로 셉니다(체결 건수가 없으면 5분).
    거래 있던 분 수의 상한이라, 한산한 봉이 비트코인보다 조금 덜 '죽은 봉'으로 걸러집니다.
      mode="rows" (바이낸스): 거래가 없어도 5분봉 행이 있음 → 5분봉 3개가 다 있는 15분 구간만 남김
      mode="omit" (크라켄): 거래가 없는 5분봉은 원본에 없음 → 첫 봉부터 끝 봉까지 모든 15분 구간을 남김
        (거래소 장애와 거래 없음을 구별할 수 없으므로 둘 다 '거래 없는 구간'으로 둡니다. 비트스탬프
         원본이 장애 구간을 거래량 0 평봉으로 채운 것과 같은 결과가 됩니다.)
"""
import argparse
import glob
import hashlib
import json
import os

import numpy as np
import pandas as pd

from ..data import minutes_to_15m, to_unix, from_unix, CUTOFF, DATA_DIR

HOLDOUT_DIR = os.path.join(DATA_DIR, "holdout")
COLS = ["ts", "open", "high", "low", "close", "volume", "active_min"]


# ══════════ 5분봉 → 15분봉 ══════════
def fivemin_to_15m(m, mode, cutoff=CUTOFF):
    """
    m: DataFrame(timestamp[초], open, high, low, close, volume[, trades]) 5분봉.
    반환 형식은 btc.data.minutes_to_15m 과 같음 (거래 없는 15분 구간은 가격 NaN, active_min 0).
    """
    if mode not in ("rows", "omit"):
        raise ValueError(mode)
    m = m.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    m = m[m["timestamp"] < int(cutoff.timestamp())]
    if (m["timestamp"] % 300 != 0).any():
        raise ValueError("5분 격자에 맞지 않는 시각")
    idx = from_unix(m["timestamp"].to_numpy())
    o, c = m["open"].to_numpy(float), m["close"].to_numpy(float)
    hi = np.minimum(m["high"].to_numpy(float), 1.1 * np.maximum(o, c))
    lo = np.maximum(m["low"].to_numpy(float), 0.9 * np.minimum(o, c))
    v = m["volume"].to_numpy(float)
    act = v > 0
    if "trades" in m:
        tr = m["trades"].to_numpy(float)
        amin = np.where(np.isfinite(tr), np.clip(tr, 1, 5), 5.0)
    else:
        amin = np.full(len(m), 5.0)
    a = pd.DataFrame({"open": o[act], "high": hi[act], "low": lo[act], "close": c[act], "volume": v[act],
                      "amin": amin[act]}, index=idx[act])
    g = a.resample("15min", label="left", closed="left")
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                        "close": g["close"].last(), "volume": g["volume"].sum(),
                        "active_min": g["amin"].sum()})
    full = pd.date_range(idx[0].floor("15min"), idx[-1].floor("15min"), freq="15min", tz="UTC")
    out = out.reindex(full)
    out["active_min"] = out["active_min"].fillna(0).astype(int)
    out["volume"] = out["volume"].fillna(0.0)
    if mode == "rows":
        n5 = pd.Series(1, index=idx).resample("15min", label="left", closed="left").count().reindex(full, fill_value=0)
        out = out[n5.to_numpy() == 3]
    out.insert(0, "ts", to_unix(out.index))
    return out


def join_15m(parts):
    """시간 순서로 이어 붙임 — 앞 조각이 끝난 뒤의 행만 뒤 조각에서 가져옴"""
    out = parts[0]
    for p in parts[1:]:
        out = pd.concat([out, p[p["ts"] > out["ts"].iloc[-1]]])
    return out


# ══════════ 원본 읽기 ══════════
def read_kraken_json(path):
    """eth-macro-data-bridge 의 크라켄 JSON (releases 연도 파일 또는 history/ 일별 파일)"""
    with open(path) as f:
        j = json.load(f)
    cols = j["columns"]
    df = pd.DataFrame(j["records"], columns=cols)
    out = pd.DataFrame({"timestamp": (df["open_time_ms"].astype(np.int64) // 1000)})
    for k in ("open", "high", "low", "close", "volume"):
        out[k] = df[k].astype(float)
    out["trades"] = df["trade_count"].astype(float) if "trade_count" in cols else np.nan
    return out


def read_speirsy_parquet(files):
    import pyarrow.parquet as pq
    parts = []
    for f in sorted(files):
        t = pq.read_table(f, columns=["timestamp", "open", "high", "low", "close", "volume"], use_threads=False)
        d = t.to_pandas()
        d["timestamp"] = to_unix(pd.DatetimeIndex(d["timestamp"]))
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def read_trademonkey(files):
    parts = []
    for f in sorted(files):
        d = pd.read_csv(f, usecols=["Unix Time", "Open", "High", "Low", "Close", "Volume"])
        # 2018-02-09~10 바이낸스 재가동 직후 봉 시작 시각이 16.812초 밀려 기록된 구간이 있어 분 단위로 내림
        parts.append(pd.DataFrame({"timestamp": (np.floor(d["Unix Time"].to_numpy(float) / 60.0) * 60).astype(np.int64),
                                   "open": d["Open"], "high": d["High"], "low": d["Low"], "close": d["Close"],
                                   "volume": d["Volume"]}))
    return pd.concat(parts, ignore_index=True)


def read_nfi_feather(path):
    d = pd.read_feather(path)
    return pd.DataFrame({"timestamp": to_unix(pd.DatetimeIndex(d["date"])), "open": d["open"], "high": d["high"],
                         "low": d["low"], "close": d["close"], "volume": d["volume"]})


# ══════════ 무결성 점검 (수익률 통계 없음) ══════════
def raw_checks(m, step):
    """원본 봉(1분·5분) 점검: 개수·범위·중복·빠진 칸·가격>0·고저 모순·거래량≥0·튐 보정 대상 수"""
    ts = m["timestamp"].to_numpy(np.int64)
    dup = int(len(ts) - len(np.unique(ts)))
    u = np.unique(ts)
    d = np.diff(u)
    o, h, l, c, v = (m[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
    return dict(rows=int(len(ts)), first=str(from_unix([u[0]])[0]), last=str(from_unix([u[-1]])[0]),
                duplicates=dup, off_grid=int((u % step != 0).sum()),
                missing_slots=int(((d // step) - 1)[d > step].sum()), gaps=int((d > step).sum()),
                longest_gap_hours=round(float(d.max() / 3600.0), 2) if len(d) else 0.0,
                price_nonpos=int(((o <= 0) | (h <= 0) | (l <= 0) | (c <= 0)).sum()),
                price_nan=int((~np.isfinite(o) | ~np.isfinite(h) | ~np.isfinite(l) | ~np.isfinite(c)).sum()),
                high_lt_maxoc=int((h < np.maximum(o, c)).sum()), low_gt_minoc=int((l > np.minimum(o, c)).sum()),
                volume_neg=int((v < 0).sum()), zero_volume_rows=int((v == 0).sum()),
                wick_clipped=int(((h > 1.1 * np.maximum(o, c)) | (l < 0.9 * np.minimum(o, c))).sum()))


def broken_bar_count(close, jump=1.5, back=1.1):
    """한 봉이 jump배 넘게 튀었다가(오르든 내리든) 다음 봉에 직전 수준 ±10% 안으로 돌아온 경우의 '개수'만"""
    c = np.asarray(close, float)
    c = c[np.isfinite(c)]
    if len(c) < 3:
        return 0
    up = c[1:-1] / c[:-2]
    rv = c[2:] / c[:-2]
    spike = (up > jump) | (up < 1.0 / jump)
    ret = (rv < back) & (rv > 1.0 / back)
    return int((spike & ret).sum())


def q15_checks(q):
    ok = q["close"].notna()
    o, h, l, c = (q.loc[ok, k].to_numpy(float) for k in ("open", "high", "low", "close"))
    ts = q["ts"].to_numpy(np.int64)
    d = np.diff(ts)
    return dict(rows=int(len(q)), first=str(from_unix([ts[0]])[0]), last=str(from_unix([ts[-1]])[0]),
                duplicates=int(len(ts) - len(np.unique(ts))), sorted=bool(np.all(d > 0)),
                missing_15m_slots=int(((d // 900) - 1)[d > 900].sum()), no_trade_rows=int((~ok).sum()),
                price_nonpos=int(((o <= 0) | (h <= 0) | (l <= 0) | (c <= 0)).sum()),
                high_lt_maxoc=int((h < np.maximum(o, c) - 1e-12).sum()),
                low_gt_minoc=int((l > np.minimum(o, c) + 1e-12).sum()),
                volume_neg=int((q["volume"].to_numpy(float) < 0).sum()),
                broken_bars_15m=broken_bar_count(c),
                no_trade_rows_by_year={int(y): int(n) for y, n in
                                       (~ok).groupby(from_unix(ts).year).sum().items()})


def pct_diff(a, b):
    """두 원본의 같은 시각 가격 차이 (%) — |a/b − 1| × 100 의 요약만 (가격 수준은 내보내지 않음)"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b) & (b > 0)
    x = np.abs(a[ok] / b[ok] - 1.0) * 100.0
    if len(x) == 0:
        return dict(n=0)
    return dict(n=int(len(x)), median_pct=round(float(np.median(x)), 5), max_pct=round(float(x.max()), 5),
                n_gt_0_01pct=int((x > 0.01).sum()), n_gt_0_5pct=int((x > 0.5).sum()))


# ══════════ 저장 ══════════
def save_15m(q, path):
    q = q[COLS].copy()
    for k in ("open", "high", "low", "close"):
        q[k] = q[k].round(8)
    q["volume"] = q["volume"].round(8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    q.to_csv(path, index=False, float_format="%.10g", compression={"method": "gzip", "mtime": 0})
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _dir_sha(files):
    h = hashlib.sha256()
    for f in sorted(files):
        with open(f, "rb") as fp:
            h.update(os.path.basename(f).encode())
            h.update(hashlib.sha256(fp.read()).digest())
    return h.hexdigest()


# ══════════ 자산별 조립 ══════════
def build_eth_kraken(src):
    """src/kraken--ETHUSD--5m--YYYY.json (releases history-kraken-spot-v3) + src/tail5m/YYYY-MM-DD.json"""
    rel_files = sorted(glob.glob(os.path.join(src, "kraken--ETHUSD--5m--*.json")))
    tail_files = sorted(glob.glob(os.path.join(src, "tail5m", "*.json")))
    rel = pd.concat([read_kraken_json(f) for f in rel_files], ignore_index=True)
    tail = pd.concat([read_kraken_json(f) for f in tail_files], ignore_index=True)
    rep = dict(raw_release=raw_checks(rel, 300), raw_tail=raw_checks(tail, 300))
    # 겹치는 구간에서 두 수집 경로의 가격 차이
    ov = rel.drop_duplicates("timestamp").merge(tail.drop_duplicates("timestamp"), on="timestamp", suffixes=("_r", "_t"))
    rep["release_vs_tail_overlap"] = {k: pct_diff(ov[f"{k}_t"], ov[f"{k}_r"]) for k in ("open", "high", "low", "close")}
    rep["release_vs_tail_overlap"]["volume"] = pct_diff(ov["volume_t"], ov["volume_r"])
    last_rel = int(rel["timestamp"].max())
    m = pd.concat([rel, tail[tail["timestamp"] > last_rel]], ignore_index=True)
    rep["raw_joined"] = raw_checks(m, 300)
    q = fivemin_to_15m(m, "omit")
    rep["sources_sha256"] = dict(release=_dir_sha(rel_files), tail=_dir_sha(tail_files))
    return q, rep


def build_speirsy(src, sym):
    files = sorted(glob.glob(os.path.join(src, f"{sym}-1m-*.parquet")))
    m = read_speirsy_parquet(files)
    rep = dict(raw=raw_checks(m, 60), n_files=len(files), sources_sha256=_dir_sha(files))
    q = minutes_to_15m(m)
    return q, rep


def build_ltc(tm_dir, nfi_path, switch="2025-08-01"):
    """1분봉(TradeMonkey, 바이낸스) ~ switch 전까지 + 5분봉(NFI, 바이낸스) switch부터"""
    files = sorted(glob.glob(os.path.join(tm_dir, "**", "*_LTC_USDT.csv"), recursive=True))
    m1 = read_trademonkey(files)
    raw_ts = pd.concat([pd.read_csv(f, usecols=["Unix Time"]) for f in files])["Unix Time"].to_numpy(float)
    realigned = int((np.abs(raw_ts / 60.0 - np.round(raw_ts / 60.0)) > 1e-9).sum())
    m5 = read_nfi_feather(nfi_path)
    sw = int(pd.Timestamp(switch, tz="UTC").timestamp())
    rep = dict(raw_1m=raw_checks(m1, 60), raw_5m=raw_checks(m5, 300), n_files_1m=len(files),
               raw_1m_rows_floored_to_minute=realigned,
               sources_sha256=dict(trademonkey=_dir_sha(files), nfi=_dir_sha([nfi_path])))
    # 겹치는 기간(NFI 시작 ~ 1분봉 끝)에서 1분봉을 5분으로 묶어 NFI 5분봉과 비교
    a = m1.drop_duplicates("timestamp").copy()
    a["b5"] = a["timestamp"] // 300 * 300
    g = a.sort_values("timestamp").groupby("b5")
    agg = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                        "close": g["close"].last(), "volume": g["volume"].sum(), "n": g["open"].count()})
    agg = agg[agg["n"] == 5]
    ov = agg.join(m5.drop_duplicates("timestamp").set_index("timestamp"), how="inner", rsuffix="_n")
    rep["tm1m_vs_nfi5m_overlap"] = {k: pct_diff(ov[k], ov[f"{k}_n"]) for k in ("open", "high", "low", "close")}
    rep["tm1m_vs_nfi5m_overlap"]["volume"] = pct_diff(ov["volume"], ov["volume_n"])
    q1 = minutes_to_15m(m1[m1["timestamp"] < sw], cutoff=pd.Timestamp(switch, tz="UTC"))
    q5 = fivemin_to_15m(m5[m5["timestamp"] >= sw], "rows")
    rep["switch_ts"] = switch
    q = join_15m([q1, q5])
    return q, rep


def main():
    ap = argparse.ArgumentParser(description="확증 시험용 코인 15분봉 만들기 (수익률 통계 없음)")
    ap.add_argument("asset", choices=["ETH", "XRP", "ADA", "LTC"])
    ap.add_argument("--src", required=True, help="원본 폴더 (ETH: 크라켄 JSON, XRP/ADA: parquet, LTC: TradeMonkey 폴더)")
    ap.add_argument("--nfi", help="LTC: NFI LTC_USDT-5m.feather")
    ap.add_argument("--report", help="점검 결과 JSON 저장 경로")
    a = ap.parse_args()
    if a.asset == "ETH":
        q, rep = build_eth_kraken(a.src)
    elif a.asset == "LTC":
        q, rep = build_ltc(a.src, a.nfi)
    else:
        q, rep = build_speirsy(a.src, f"{a.asset}USDT")
    rep["q15"] = q15_checks(q)
    out = os.path.join(HOLDOUT_DIR, f"{a.asset}_15m.csv.gz")
    rep["file"] = os.path.relpath(out, os.path.dirname(os.path.dirname(DATA_DIR)))
    rep["file_sha256"] = save_15m(q, out)
    rep["file_mb"] = round(os.path.getsize(out) / 1e6, 2)
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    if a.report:
        with open(a.report, "w") as f:
            json.dump(rep, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
