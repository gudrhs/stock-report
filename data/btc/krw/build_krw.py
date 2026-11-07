"""Build merged KRW daily series from downloaded sources and run sanity checks.

All inputs are files under this scratch dir, plus the read-only project file
data/btc/btcusd_15m.csv.gz (Bitstamp BTCUSD 15m bars, ts = bar start UTC).
Outputs go to ./out/.
"""
import glob
import gzip
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
BITSTAMP = "/home/user/stock-report/data/btc/btcusd_15m.csv.gz"


def p(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- Bitstamp
bs = pd.read_csv(BITSTAMP)
bs["t"] = pd.to_datetime(bs["ts"], unit="s", utc=True)
bs = bs.dropna(subset=["close"]).set_index("t").sort_index()
bs_d = pd.DataFrame({
    "bs_open": bs["open"].resample("1D").first(),
    "bs_close": bs["close"].resample("1D").last(),
})
# price at 00:30 UTC = open of the 00:30 bar (fallback: close of 00:15 bar)
b0030 = bs[bs.index.strftime("%H:%M") == "00:30"]["open"]
b0030.index = b0030.index.normalize()
bs_d["bs_0030"] = b0030
# price at 15:00 UTC (= KST midnight) for Bithumb/Korbit KST-day alignment tests
b1500 = bs[bs.index.strftime("%H:%M") == "15:00"]["open"]
b1500.index = b1500.index.normalize()
bs_d["bs_1500"] = b1500
bs_d.index = bs_d.index.tz_localize(None)
p("bitstamp daily", bs_d.index.min().date(), bs_d.index.max().date(), len(bs_d))

# ---------------------------------------------------------------- FX
fx = pd.read_csv(os.path.join(HERE, "fx_fred_datasets_exchange-rates_daily.csv"))
fx = fx[fx["Country"] == "South Korea"].copy()
fx["Date"] = pd.to_datetime(fx["Date"])
fx["fred"] = pd.to_numeric(fx["Exchange rate"], errors="coerce")
fred = fx.groupby("Date")["fred"].last()
p("FRED duplicate dates", int(fx["Date"].duplicated().sum()))
p("FRED KRW rows", fred.notna().sum(), fred.index.min().date(), fred.index.max().date(),
  "NaN rows", fred.isna().sum())
fred_2014 = fred["2014":]
bd = pd.bdate_range("2014-01-01", fred.index.max())
missing_bd = bd.difference(fred_2014.dropna().index)
p("FRED business days missing since 2014 (holidays incl.)", len(missing_bd))

oxr = pd.read_csv(os.path.join(HERE, "fx_openexchangerates_KwangYeol_exchange-now_KRW.csv"),
                  header=None, names=["date", "ccy", "rate"])
oxr["date"] = pd.to_datetime(oxr["date"])
oxr = oxr.groupby("date")["rate"].last()
p("OXR KRW rows", len(oxr), oxr.index.min().date(), oxr.index.max().date())
cal = pd.date_range("2013-09-01", "2026-09-30", freq="D")
fred_ff = fred.reindex(cal).ffill()
oxr_ff = oxr.reindex(cal).ffill()
both = pd.concat([fred.rename("fred"), oxr.rename("oxr")], axis=1).dropna()
d = (both["oxr"] / both["fred"] - 1) * 100
p("OXR vs FRED same-date diff %%: mean %.3f sd %.3f p1 %.3f p99 %.3f maxabs %.3f n %d" % (
    d.mean(), d.std(), d.quantile(.01), d.quantile(.99), d.abs().max(), len(d)))
# OXR labelled D is snapshot ~00:04 UTC of D; compare with FRED D-1 (NY noon previous day)
both2 = pd.concat([fred.shift(1, freq="D").rename("fred_prev"), oxr.rename("oxr")], axis=1).dropna()
d2 = (both2["oxr"] / both2["fred_prev"] - 1) * 100
p("OXR(D) vs FRED(D-1) diff %%: mean %.3f sd %.3f" % (d2.mean(), d2.std()))

# ---------------------------------------------------------------- Upbit sources
def load_pyupbit_csv(path, name):
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index.str.replace("﻿", "")).normalize()
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df["src"] = name
    return df


def load_upbit_json(path, name, utc_key="candle_date_time_utc", kst_key="candle_date_time_kst"):
    d = json.load(open(path))
    df = pd.DataFrame(d)
    if utc_key in df:
        idx = pd.to_datetime(df[utc_key]).dt.normalize()
    else:  # KST 09:00 == UTC 00:00 of same date
        idx = pd.to_datetime(df[kst_key]).dt.normalize()
    out = pd.DataFrame({"open": df["opening_price"].values, "high": df["high_price"].values,
                        "low": df["low_price"].values, "close": df["trade_price"].values,
                        "volume": df["candle_acc_trade_volume"].values}, index=idx.values)
    out["src"] = name
    return out.sort_index()


srcs = {}
srcs["prodong04"] = load_pyupbit_csv(os.path.join(HERE, "upbit_d_prodong04_ViT_for_finance.csv"), "prodong04")
srcs["JS0208"] = load_pyupbit_csv(os.path.join(HERE, "upbit_d_JS0208_MM.csv"), "JS0208")
srcs["yoon627"] = load_upbit_json(os.path.join(HERE, "upbit_d_yoon627_yearly.json"), "yoon627")
bits = []
for f in sorted(glob.glob(os.path.join(HERE, "upbit_bitableau_versions", "*.json"))):
    snap_date = pd.Timestamp(os.path.basename(f).split("_")[2])
    x = load_upbit_json(f, "bitableau")
    # the newest candle in each snapshot is still forming (crawl ~18:20-21:00 UTC)
    x = x[x.index < x.index.max()]
    bits.append(x)
bit = pd.concat(bits)
bit = bit[~bit.index.duplicated(keep="last")].sort_index()
srcs["bitableau"] = bit
w4 = pd.read_csv(os.path.join(HERE, "upbit_4h_WomkaLabs_Stream.csv"), parse_dates=["datetime"]).set_index("datetime")
cnt = w4["close"].resample("1D").count()
w4d = pd.DataFrame({"open": w4["open"].resample("1D").first(), "high": w4["high"].resample("1D").max(),
                    "low": w4["low"].resample("1D").min(), "close": w4["close"].resample("1D").last(),
                    "volume": w4["volume"].resample("1D").sum()})
w4d = w4d[cnt == 6]
w4d["src"] = "womka4h"
srcs["womka4h"] = w4d
for k, v in srcs.items():
    full = pd.date_range(v.index.min(), v.index.max(), freq="D")
    p(f"upbit src {k:10s} {v.index.min().date()} .. {v.index.max().date()} rows {len(v)} missing_days {len(full.difference(v.index))}")

# pairwise agreement on overlaps
keys = list(srcs)
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        a, b = srcs[keys[i]], srcs[keys[j]]
        ix = a.index.intersection(b.index)
        if len(ix) == 0:
            continue
        for col in ["open", "close"]:
            r = (a.loc[ix, col] / b.loc[ix, col] - 1).abs()
            p(f"  overlap {keys[i]} vs {keys[j]} {col}: n={len(ix)} exact={int((r == 0).sum())} max|rel|={r.max():.2e}")

# merge with priority (most direct API dumps first)
prio = ["bitableau", "yoon627", "JS0208", "prodong04", "womka4h"]
parts = []
for k in prio:
    parts.append(srcs[k])
up = pd.concat(parts)
up = up[~up.index.duplicated(keep="first")].sort_index()
up = up[up.index <= "2026-09-26"]
full = pd.date_range(up.index.min(), up.index.max(), freq="D")
p("MERGED upbit", up.index.min().date(), up.index.max().date(), len(up), "missing", list(full.difference(up.index).date))
p("  source counts", up["src"].value_counts().to_dict())
up.index.name = "date_utc"
up.to_csv(os.path.join(OUT, "upbit_krwbtc_daily_merged.csv"))

# ---------------------------------------------------------------- Bithumb
bi = pd.read_csv(os.path.join(HERE, "bithumb_d_investing_gwon9906.csv"))
bi.columns = ["date", "close", "open", "high", "low", "vol", "chg"]
bi["date"] = pd.to_datetime(bi["date"].str.replace(" ", ""), format="%Y-%m-%d")
for c in ["close", "open", "high", "low"]:
    bi[c] = bi[c].str.replace(",", "").astype(float)
bi = bi.set_index("date").sort_index()
fullb = pd.date_range(bi.index.min(), bi.index.max(), freq="D")
p("bithumb investing", bi.index.min().date(), bi.index.max().date(), len(bi), "missing", len(fullb.difference(bi.index)))
cdd = pd.read_csv(os.path.join(HERE, "bithumb_d_cryptodatadownload_HugoLy.csv"))
cdd["date"] = pd.to_datetime(cdd["Unix Timestamp"], unit="s")
cdd = cdd.set_index("date").sort_index()
fullc = pd.date_range(cdd.index.min(), cdd.index.max(), freq="D")
p("bithumb CDD", cdd.index.min().date(), cdd.index.max().date(), len(cdd), "missing", len(fullc.difference(cdd.index)))
ix = bi.index.intersection(cdd.index)
r = (bi.loc[ix, "close"] / cdd.loc[ix, "Close"] - 1).abs()
p(f"  bithumb investing vs CDD close: n={len(ix)} exact={(r==0).sum()} median|rel|={r.median():.2e} max={r.max():.2e}")
bi.index.name = "date"
bi[["open", "high", "low", "close"]].to_csv(os.path.join(OUT, "bithumb_krwbtc_daily_investing.csv"))

# ---------------------------------------------------------------- Korbit ticks
kb = pd.read_csv(os.path.join(HERE, "korbitKRW.csv.gz"), header=None, names=["ts", "price", "amt"])
kb["t"] = pd.to_datetime(kb["ts"], unit="s")
kb = kb.set_index("t").sort_index()
kbd = pd.DataFrame({"open": kb["price"].resample("1D").first(), "high": kb["price"].resample("1D").max(),
                    "low": kb["price"].resample("1D").min(), "close": kb["price"].resample("1D").last(),
                    "volume": kb["amt"].resample("1D").sum(), "ntrades": kb["price"].resample("1D").count()})
# price nearest 00:00 UTC: last trade before 00:00 UTC (= prior day's close)
p("korbit ticks", kb.index.min(), kb.index.max(), len(kb))
p("  korbit days with 0 trades by year:", kbd[kbd["ntrades"] == 0].groupby(kbd[kbd["ntrades"] == 0].index.year).size().to_dict())
p("  korbit median trades/day by year:", kbd.groupby(kbd.index.year)["ntrades"].median().to_dict())
kbd.index.name = "date_utc"
kbd.to_csv(os.path.join(OUT, "korbit_krwbtc_daily_utc_from_ticks.csv"))

# ---------------------------------------------------------------- premium
def prem_table(krw_close, usd_close, fxs, label):
    df = pd.concat([krw_close.rename("krw"), usd_close.rename("usd"), fxs.rename("fx")], axis=1).dropna()
    df["prem"] = (df["krw"] / (df["usd"] * df["fx"]) - 1) * 100
    g = df.groupby(df.index.year)["prem"]
    t = pd.DataFrame({"n": g.size(), "mean": g.mean(), "median": g.median(), "p5": g.quantile(.05),
                      "p95": g.quantile(.95), "min": g.min(), "max": g.max()}).round(2)
    p(f"\n=== premium % [{label}] ===")
    p(t.to_string())
    return df


fx_close = fred_ff.copy()  # FRED noon NY of D, ffilled (weekends/holidays -> previous business day)
fx_open = fred_ff.shift(1)  # at 00:00 UTC of D the latest NY noon fix is D-1
# upbit close (24:00 UTC) vs bitstamp close
up_prem = prem_table(up["close"], bs_d["bs_close"], fx_close, "Upbit close 24:00UTC vs Bitstamp close x FRED(D)")
up_prem_o = prem_table(up["open"], bs_d["bs_open"], fx_open, "Upbit open 00:00UTC vs Bitstamp open x FRED(D-1)")
up_prem_oxr = prem_table(up["close"], bs_d["bs_close"], oxr_ff.shift(-1), "Upbit close vs Bitstamp close x OXR(D+1 ~00:04UTC)")
bi_prem = prem_table(bi["close"], bs_d["bs_close"], fx_close, "Bithumb(investing) close vs Bitstamp close(UTC) x FRED")
bi_prem15 = prem_table(bi["close"], bs_d["bs_1500"], fx_close, "Bithumb(investing) close vs Bitstamp 15:00UTC(=KST 24:00) x FRED")
kb_prem = prem_table(kbd.loc[kbd["ntrades"] > 0, "close"], bs_d["bs_close"], fx_close, "Korbit tick close(UTC day) vs Bitstamp close x FRED")

# 2018-01 detail
for name, df in [("upbit", up_prem), ("bithumb", bi_prem), ("korbit", kb_prem)]:
    s = df.loc["2017-12-01":"2018-02-15", "prem"]
    if len(s):
        p(f"{name} 2017-12..2018-02 premium max {s.max():.1f}% on {s.idxmax().date()}, 2018-01 mean {df.loc['2018-01','prem'].mean():.1f}%")

# consistency: upbit close vs bithumb close (same UTC day)
ix = up.index.intersection(bi.index)
r = (up.loc[ix, "close"] / bi.loc[ix, "close"] - 1) * 100
p("\nUpbit close / Bithumb(investing) close - 1 [%%]: median %.2f mean-abs %.2f p99-abs %.2f n %d" % (r.median(), r.abs().mean(), r.abs().quantile(.99), len(r)))
ix = up.index.intersection(kbd.index[kbd["ntrades"] > 0])
r = (up.loc[ix, "close"] / kbd.loc[ix, "close"] - 1) * 100
p("Upbit close / Korbit close - 1 [%%]: median %.2f mean-abs %.2f n %d" % (r.median(), r.abs().mean(), len(r)))

# cross-check vs yoon627 kimp series (Upbit close / Binance close x ECB)
k = json.load(open(os.path.join(HERE, "kimp_btc_yoon627.json")))
ks = pd.Series({pd.Timestamp(x["date"]): x["value"] for x in k["data"]})
cc = pd.concat([up_prem["prem"].rename("ours"), ks.rename("yoon")], axis=1).dropna()
p("ours vs yoon627 kimp: n %d corr %.3f mean diff %.3f pp, mean|diff| %.3f pp" % (
    len(cc), cc.corr().iloc[0, 1], (cc.ours - cc.yoon).mean(), (cc.ours - cc.yoon).abs().mean()))

# bm snapshots cross-check (2026)
bm = pd.read_csv(os.path.join(HERE, "bm_market_history.csv"), parse_dates=["date"]).set_index("date")
cc = pd.concat([up_prem["prem"].rename("ours"), bm["kimchi_pct"].rename("bm")], axis=1).dropna()
if len(cc):
    p("ours vs bm kimchi_pct (snapshot ~07:00-09:00 KST): n %d corr %.3f mean diff %.3f pp" % (
        len(cc), cc.corr().iloc[0, 1], (cc.ours - cc.bm).mean()))

# ---------------------------------------------------------------- recommended daily KRW table
tab = pd.DataFrame(index=pd.date_range("2014-01-01", "2026-09-25", freq="D"))
tab.index.name = "date_utc"
tab = tab.join(bs_d)
tab["usdkrw_fred_ffill"] = fred_ff
tab["usdkrw_fred_prev_ffill"] = fred_ff.shift(1)
tab["usdkrw_oxr_0004utc"] = oxr.reindex(tab.index)
tab["upbit_open"] = up["open"]
tab["upbit_close"] = up["close"]
tab["upbit_src"] = up["src"]
tab["bithumb_close_investing"] = bi["close"]
tab["korbit_close_ticks_utc"] = kbd.loc[kbd["ntrades"] > 0, "close"]
tab["korbit_ntrades"] = kbd["ntrades"]
tab["prem_upbit_close_pct"] = up_prem["prem"]
tab["prem_upbit_open_pct"] = up_prem_o["prem"]
tab["prem_korbit_close_pct"] = kb_prem["prem"]
tab.to_csv(os.path.join(OUT, "krw_daily_panel.csv"), float_format="%.6g")
p("\nwrote", os.path.join(OUT, "krw_daily_panel.csv"), tab.shape)

# ---------------------------------------------------------------- extras
# Korbit price as of 00:00 and 00:30 UTC (last trade at or before), with staleness in minutes
kbp = kb["price"]
def asof_px(hhmm_min):
    grid = pd.date_range("2013-09-04", "2018-01-20", freq="D") + pd.Timedelta(minutes=hhmm_min)
    pos = kbp.index.searchsorted(grid, side="right") - 1
    ok = pos >= 0
    px = pd.Series(np.nan, index=grid.normalize()); age = pd.Series(np.nan, index=grid.normalize())
    px[ok] = kbp.values[pos[ok]]
    age[ok] = (grid[ok] - kbp.index[pos[ok]]).total_seconds() / 60
    return px, age
k00, a00 = asof_px(0)
k30, a30 = asof_px(30)
p("\nKorbit last-trade age at 00:00 UTC (minutes) median by year:", a00.groupby(a00.index.year).median().round(1).to_dict())
p("Korbit last-trade age at 00:00 UTC p90 by year:", a00.groupby(a00.index.year).quantile(.9).round(1).to_dict())
tab = pd.read_csv(os.path.join(OUT, "krw_daily_panel.csv"), index_col=0, parse_dates=True)
tab["korbit_px_0000utc"] = k00
tab["korbit_age_min_0000"] = a00
tab["korbit_px_0030utc"] = k30
tab["korbit_age_min_0030"] = a30
tab["prem_korbit_0000_pct"] = (tab["korbit_px_0000utc"] / (tab["bs_open"] * tab["usdkrw_fred_prev_ffill"]) - 1) * 100
tab["prem_korbit_0030_pct"] = (tab["korbit_px_0030utc"] / (tab["bs_0030"] * tab["usdkrw_fred_prev_ffill"]) - 1) * 100
tab.to_csv(os.path.join(OUT, "krw_daily_panel.csv"), float_format="%.6g")
g = tab["prem_korbit_0000_pct"].dropna()
p("Korbit 00:00 premium by year mean/median/p5/p95:")
p(g.groupby(g.index.year).agg(["size", "mean", "median", lambda s: s.quantile(.05), lambda s: s.quantile(.95)]).round(2).to_string())

m = up_prem["prem"]
mm = m.groupby([m.index.year, m.index.month]).mean()
p("\nUpbit close premium monthly mean 2017-10..2018-04:")
p(mm.loc[(2017, 10):(2018, 4)].round(1).to_string())
p("Top-8 Upbit close premium days:")
p(m.sort_values().tail(8).round(1).to_string())
p("Upbit 2021 monthly mean premium:")
p(mm.loc[(2021, 1):(2021, 12)].round(1).to_string())

# spliced FX for 00:00 UTC booking: FRED(D-1) while FRED has data, then OXR(D ~00:04 UTC)
fred_last = fred.dropna().index.max()
tab["usdkrw_0000utc_spliced"] = np.where(tab.index <= fred_last + pd.Timedelta(days=1),
                                         tab["usdkrw_fred_prev_ffill"], tab["usdkrw_oxr_0004utc"])
tab["prem_upbit_open_spliced_pct"] = (tab["upbit_open"] / (tab["bs_open"] * tab["usdkrw_0000utc_spliced"]) - 1) * 100
tab.to_csv(os.path.join(OUT, "krw_daily_panel.csv"), float_format="%.6g")
p("\nFRED last date", fred_last.date(), "; tail with spliced FX:")
p(tab.loc["2026-09-15":"2026-09-25", ["usdkrw_fred_prev_ffill", "usdkrw_oxr_0004utc", "usdkrw_0000utc_spliced", "prem_upbit_open_pct", "prem_upbit_open_spliced_pct"]].round(3).to_string())
