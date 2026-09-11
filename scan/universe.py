# -*- coding: utf-8 -*-
"""
스캔 대상 종목 산출 — 코스피200 + 코스닥150

KRX가 지수 구성종목 API를 공개하지 않아(2026-08 기준 pykrx·FDR 모두 조회 실패)
**시가총액 상위**로 대신합니다. 코스피200·코스닥150 모두 시총·유동성 상위로
구성되므로 대부분 겹치지만, 지수의 산업별 배분 규칙까지 반영하지는 못합니다.
"""
import datetime as dt
import json
import logging
import os
from pathlib import Path
import tempfile
import time
import warnings
warnings.filterwarnings("ignore")
import FinanceDataReader as fdr
import pandas as pd

LOG = logging.getLogger(__name__)
CACHE_PATH = Path(__file__).resolve().parents[1] / "data" / "universe_cache.json"
CACHE_MAX_AGE = dt.timedelta(days=14)
REQUIRED_COLUMNS = ("Code", "Name", "Market", "Marcap", "Close")
MAX_ATTEMPTS = 5
# 전체 KRX 응답의 보수적인 하한. build()의 사용자 지정 선정 개수와는 별개입니다.
MIN_LISTING_SIZE = 1000
MIN_MARKET_SIZE = {"KOSPI": 200, "KOSDAQ": 150}


def _validate_listing(df):
    """실시간 응답과 캐시에 동일한 검증을 적용하고 코드의 앞자리 0을 보존합니다."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("KRX listing is not a DataFrame")
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"KRX listing missing required columns: {', '.join(missing)}")
    if not df.columns.is_unique:
        raise ValueError("KRX listing has duplicate columns")
    df = df.loc[:, list(REQUIRED_COLUMNS)].copy()
    if not df["Code"].map(lambda code: isinstance(code, str)).all():
        raise ValueError("KRX listing codes must be strings (preserve leading zeros)")
    if not df["Code"].str.fullmatch(r"[0-9A-Z]{6}").all() or df["Code"].duplicated().any():
        raise ValueError("KRX listing has invalid or duplicate stock codes")
    for col in ("Name", "Market"):
        if not df[col].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"KRX listing has empty or invalid {col}")
    for col in ("Marcap", "Close"):
        df[col] = pd.to_numeric(df[col], errors="raise")
        if df[col].isin([float("inf"), float("-inf")]).any():
            raise ValueError(f"KRX listing has non-finite {col}")
    markets = df["Market"].str.startswith(("KOSPI", "KOSDAQ"))
    # 결측/0 가격 행은 기존 build()에서 제외하지만, 대부분이 결측인 응답은 거절합니다.
    usable = markets & (df["Marcap"] > 0) & (df["Close"] > 0)
    if usable.sum() < MIN_LISTING_SIZE:
        raise ValueError(
            f"KRX listing abnormally small: {int(usable.sum())} usable stocks "
            f"(minimum {MIN_LISTING_SIZE})"
        )
    for market, minimum in MIN_MARKET_SIZE.items():
        count = int((usable & df["Market"].str.startswith(market)).sum())
        if count < minimum:
            raise ValueError(f"KRX listing abnormally small for {market}: {count} (minimum {minimum})")
    return df


def _save_cache(df):
    """검증된 목록을 임시 파일에 쓴 후 교체하여 기존 정상 캐시를 보호합니다."""
    payload = {
        "version": 1,
        "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "rows": json.loads(df.to_json(orient="records", force_ascii=False)),
    }
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=CACHE_PATH.parent,
            prefix="universe_cache.", suffix=".tmp", delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        os.replace(temp_path, CACHE_PATH)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _load_cache():
    with CACHE_PATH.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("Invalid universe cache format/version")
    timestamp = payload.get("fetched_at")
    if not isinstance(timestamp, str):
        raise ValueError("Universe cache is missing fetched_at")
    fetched_at = dt.datetime.fromisoformat(timestamp)
    if fetched_at.tzinfo is None:
        raise ValueError("Universe cache fetched_at must include a timezone")
    age = dt.datetime.now(dt.timezone.utc) - fetched_at
    if age < dt.timedelta(0) or age > CACHE_MAX_AGE:
        raise ValueError(f"Universe cache age {age} is outside the allowed 0-14 days")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("Universe cache rows must be a list of records")
    df = _validate_listing(pd.DataFrame(rows))
    LOG.warning("Using KRX universe cache fetched at %s (%s old)", timestamp, age)
    return df


def _get_listing():
    last_error = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            df = _validate_listing(fdr.StockListing("KRX"))
        except Exception as exc:
            last_error = exc
            LOG.warning("KRX listing attempt %d/%d failed: %s", attempt + 1, MAX_ATTEMPTS, exc)
            if attempt < MAX_ATTEMPTS - 1:
                time.sleep(2 ** (attempt + 1))
            continue
        try:
            _save_cache(df)
        except OSError as exc:
            # 유효한 실시간 조회는 캐시 저장 권한/디스크 문제로 폐기하지 않습니다.
            LOG.warning("Could not save KRX universe cache at %s: %s", CACHE_PATH, exc)
        return df
    try:
        return _load_cache()
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise RuntimeError(
            f"KRX listing failed after {MAX_ATTEMPTS} attempts; "
            f"last error: {last_error}. No valid universe cache within 14 days "
            f"at {CACHE_PATH}: {exc}"
        ) from exc

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
    df = _get_listing()
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
