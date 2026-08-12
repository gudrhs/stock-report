# -*- coding: utf-8 -*-
"""
스캔 대상 종목 산출 — 코스피200 + 코스닥150

KRX가 지수 구성종목 API를 공개하지 않아(2026-08 기준 pykrx·FDR 모두 조회 실패)
**시가총액 상위**로 대신합니다. 코스피200·코스닥150 모두 시총·유동성 상위로
구성되므로 대부분 겹치지만, 지수의 산업별 배분 규칙까지 반영하지는 못합니다.
"""
import warnings
warnings.filterwarnings("ignore")
import FinanceDataReader as fdr

# 지수에 담기지 않는 종류들
EXCLUDE_WORDS = ("스팩", "리츠")


def _is_common(row):
    """보통주만 — 우선주·스팩·리츠·ETF 제외"""
    name = str(row["Name"])
    code = str(row["Code"])
    if any(w in name for w in EXCLUDE_WORDS):
        return False
    # 우선주는 종목코드 끝자리가 0이 아닙니다 (예: 005935 삼성전자우)
    if code[-1] != "0":
        return False
    if name.endswith("우") or name.endswith("우B") or name.endswith("우C"):
        return False
    return True


def build(kospi_n=200, kosdaq_n=150, min_price=1000):
    """[(code, name, market, marcap), ...] 반환"""
    df = fdr.StockListing("KRX")
    df = df[df["Market"].astype(str).str.startswith(("KOSPI", "KOSDAQ"))]
    df = df[df["Marcap"].notna() & (df["Marcap"] > 0)]
    df = df[df["Close"].notna() & (df["Close"] >= min_price)]
    df = df[df.apply(_is_common, axis=1)]

    out = []
    for mkt, n, label in (("KOSPI", kospi_n, "코스피200"),
                          ("KOSDAQ", kosdaq_n, "코스닥150")):
        sub = df[df["Market"].astype(str).str.startswith(mkt)]
        sub = sub.sort_values("Marcap", ascending=False).head(n)
        for _, r in sub.iterrows():
            out.append((str(r["Code"]), str(r["Name"]), label, int(r["Marcap"])))
    return out


if __name__ == "__main__":
    u = build()
    k = [x for x in u if x[2] == "코스피200"]
    q = [x for x in u if x[2] == "코스닥150"]
    print(f"총 {len(u)}종목 · 코스피200 {len(k)} · 코스닥150 {len(q)}")
    print("코스피 상위 5:", [x[1] for x in k[:5]])
    print("코스닥 상위 5:", [x[1] for x in q[:5]])
