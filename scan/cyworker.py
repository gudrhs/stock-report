# -*- coding: utf-8 -*-
"""
32비트 전용 워커 — CYBOS 엔진으로 지표만 계산합니다.
pandas 불필요(csv만 사용). 64비트 daily.py가 subprocess로 부릅니다.

사용: python32 cyworker.py <in_prices.csv> <out_indi.csv>
  in : code,seq,open,high,low,close,volume   (종목별 시간순)
  out: code,seq,B,T,T_sig,U,U_sig            (계산 불가 구간은 빈칸)
"""
import sys, csv, traceback

DBLMAX = 1.7976931348623157e+308
SPEC = {
    "B": ("BPDL Hilo Index", dict(Term1=16)),
    "T": ("TSI", dict(Term1=10, Term2=30, Signal=18)),
    "U": ("Ultimate Oscillator", dict(Term1=7, Term2=14, Term3=28, Signal=8)),
}


def clean(x):
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return None if f >= DBLMAX * 0.99 else f


def main():
    src, dst = sys.argv[1], sys.argv[2]

    import pythoncom
    import win32com.client as wc
    pythoncom.CoInitialize()

    cb = wc.Dispatch("CpUtil.CpCybos")          # ★ 생성 순서 중요
    if not cb.IsConnect:
        print("NOT_CONNECTED", file=sys.stderr)
        sys.exit(2)
    wc.Dispatch("CpIndexes.CpIndex")            # ★ Series보다 먼저

    data = {}
    with open(src, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            data.setdefault(r["code"], []).append(
                (float(r["open"]), float(r["high"]), float(r["low"]),
                 float(r["close"]), float(r["volume"])))

    def calc(bars, key):
        name, terms = SPEC[key]
        idx = wc.Dispatch("CpIndexes.CpIndex")
        idx.put_IndexKind(name)
        idx.put_IndexDefault(name)
        for k, v in terms.items():
            try:
                setattr(idx, k, v)
            except Exception:
                pass
        ser = wc.Dispatch("CpIndexes.CpSeries")
        for o, h, l, c, v in bars:
            ser.Add(c, o, h, l, v)
        idx.Series = ser
        idx.Calculate()
        try:
            items = int(idx.ItemCount) or 1
        except Exception:
            items = 1
        lines = []
        for it in range(items):
            try:
                n = idx.GetCount(it)
            except Exception:
                continue
            if n:
                lines.append([clean(idx.GetResult(it, i)) for i in range(n)])
        return lines

    def fit(a, n):
        if not a:
            return [None] * n
        return ([None] * (n - len(a)) + a)[-n:]

    out = open(dst, "w", newline="", encoding="utf-8")
    W = csv.writer(out)
    W.writerow(["code", "seq", "B", "T", "T_sig", "U", "U_sig"])
    done = 0
    for code, bars in data.items():
        n = len(bars)
        cols = {}
        for key in ("B", "T", "U"):
            try:
                ls = calc(bars, key)
            except Exception:
                traceback.print_exc(file=sys.stderr)
                ls = []
            cols[key] = fit(ls[0] if ls else None, n)
            cols[key + "_sig"] = fit(ls[1] if len(ls) > 1 else None, n)
        for i in range(n):
            W.writerow([code, i] + [
                ("" if cols[k][i] is None else round(cols[k][i], 4))
                for k in ("B", "T", "T_sig", "U", "U_sig")])
        done += 1
        if done % 25 == 0:
            print(f"PROGRESS {done}/{len(data)}", file=sys.stderr)
            out.flush()
    out.close()
    print(f"OK {done}", file=sys.stderr)


if __name__ == "__main__":
    main()
