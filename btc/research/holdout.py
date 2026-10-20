# -*- coding: utf-8 -*-
"""
2단계 확증 시험 — 비트코인에서 고정한 설정을 한 번도 쓰지 않은 코인에서 그대로 돌리고 판정합니다.
사전 등록: btc/research/success_criteria.md (H1~H3, 후보 최대 3개, 본페로니 0.10/3)
데이터: data/btc/holdout/<자산>_15m.csv.gz (출처·점검: data/btc/holdout/README.md)

  python -m btc.research.holdout info ETH LTC XRP ADA                 ← 봉 수·구간·첫 학습 달 (성적 없음)
  python -m btc.research.holdout run ETH R6_daily_trend8_uniform --reps 5 --workers 4
  python -m btc.research.holdout evaluate ETH LTC XRP ADA --variants R6_daily_trend8_uniform R3_daily_trend8_log5

데이터 → PhaseData (btc.walkforward.load_phases 와 같은 길)
  15분봉 파일 → btc.data.bars_4h(df15, phase) 16개 격자(15분씩 밀림) → btc.features.compute → PhaseData.
  15분봉 파일에 active_min(거래 있던 분 수)이 있으면 비트코인과 한 글자도 다르지 않게 처리합니다.
  · active_min 이 없는 파일(예: 다른 곳에서 받은 15분·1시간봉)은 '거래 있던 봉' × 봉 길이(분)로 대신합니다
    (거래 있던 분 수의 상한 → 한산한 봉이 덜 걸러짐). tests/test_btc_holdout.py 가 비트코인에서 그 차이를 잽니다.
  · 1시간봉 파일은 1시간씩 밀린 격자 4개(phase 0~3), 나머지 규칙(완성 봉 = 하위 봉 4개, 죽은 봉 24분 미만,
    6개 이상 죽은 봉 뒤 첫 봉 관망)은 같습니다.

학습 시작점
  btc.walkforward.FIRST_TRAIN(2014-01-01)은 모든 코인의 데이터 시작보다 앞이라, 학습 표본은 자연히 그 코인의
  데이터 시작(특징 예열 2400봉 이후)부터입니다. 핵심 파일은 고치지 않습니다.
  워크포워드 달력(btc.walkforward.months, 2017-01~)도 그대로 씁니다. 다만 학습 표본이 하나도 없는 달은
  학습 코드가 멈추므로(빈 배열), 그리고 DQN 계열에서 1월 점검이 '최근 2년 봉 부족'으로 반드시 거부되는 달은
  결과가 버려지므로, 그런 앞쪽 달들은 이 프로세스 안에서만 btc.research.walk.months 를 바꿔 건너뜁니다.
  매달 학습 난수 시드는 (반복, 연, 월)로 정해지고 달 사이에 이어지지 않으므로, 건너뛰어도 결과는 똑같습니다
  (그 달들은 '모델 없음 = 현금'으로 기록).

판정 (evaluate)
  · 기간: 데이터 시작 + 2년 이후 첫 1월 1일 ~ 데이터 끝(자정), 최대 2026-09-25
  · 하루 한 번 00:00 UTC 마감 봉에서 판단(walk.decision_mask), 다음 봉 시가 체결, 편도 0.15%, 판단비용 0.3%
    (1단계 선별 btc.research.screen 과 같은 계산)
  · 헤드라인 = 그 자산에서 매수·보유 대비 ΔSharpe의 작은 쪽 중앙값 반복
  · H1 자산마다 헤드라인 샤프 > 매수·보유, H3 자산마다 헤드라인 최대낙폭이 더 얕음,
    H2 자산 평균 ΔSharpe > 0 이고 정상 부트스트랩 단측 p < 0.10/3 (자산들을 같은 날짜로 함께 뽑는 짝지은 부트스트랩)
  · 확증 시험에 보낸 후보는 data/btc/research_runs/holdout/confirmatory.jsonl 에 순서대로 남기고 3개를 넘으면 거부.
    trials.jsonl(1단계 시험 수)에는 기록하지 않습니다.
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import glob
import hashlib
import json
import math
import multiprocessing as mp
import time

import numpy as np
import pandas as pd

from .. import stats as S
from ..data import DATA_DIR, bars_4h, from_unix, STALE_MIN, FORCED_HOLD_GAP
from ..env import PhaseData, BAR_SEC
from ..features import compute, WARMUP
from ..walkforward import months, decision_range, FIRST_TRAIN, OOS_END

HOLDOUT_DIR = os.path.join(DATA_DIR, "holdout")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
RUNS_H = os.path.join(DATA_DIR, "research_runs", "holdout")
LEDGER = os.path.join(RUNS_H, "confirmatory.jsonl")

ASSETS = ("ETH", "LTC", "XRP", "ADA")      # 데이터를 보기 전에 정한 확증 시험 자산 (이 순서)
MAX_CANDIDATES = 3
ALPHA = 0.10 / 3
COST = 0.0015                               # 편도 0.15%
C_DEC = 0.003                               # 판단비용 = 2 × 0.15% (1단계 선별과 같음)
N_BOOT, BOOT_SEED, MEAN_BLOCK = 4000, 1, 20
BARS_COLS = ("ts", "open", "high", "low", "close", "volume", "gap_before", "forced_hold")


# ══════════ 파일 → 하위 봉 DataFrame ══════════
def asset_file(asset):
    for res in ("15m", "1h"):
        p = os.path.join(HOLDOUT_DIR, f"{asset}_{res}.csv.gz")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{asset}: data/btc/holdout/{asset}_15m.csv.gz 또는 _1h.csv.gz 가 없습니다")


def resolution_sec(ts):
    """하위 봉 길이(초) — 가장 흔한 간격. 4시간을 나누어떨어지게 하는 15분·30분·1시간·2시간만 허용"""
    d = np.diff(np.asarray(ts, np.int64))
    if len(d) == 0:
        raise ValueError("봉이 2개 이상 필요합니다")
    vals, cnt = np.unique(d, return_counts=True)
    res = int(vals[np.argmax(cnt)])
    if res < 900 or BAR_SEC % res != 0:
        raise ValueError(f"하위 봉 길이 {res}초는 지원하지 않습니다 (15분 이상, 4시간의 약수) — 먼저 15분봉으로 묶으세요")
    return res


def prepare_sub(df):
    """
    ts·open·high·low·close·volume[·active_min] → (UTC 색인 DataFrame, 하위 봉 초, active_min 추정 여부).
    active_min 이 없으면 '거래 있던 하위 봉(가격이 있고 거래량 > 0) × 봉 길이(분)'로 채웁니다.
    """
    df = df.drop_duplicates("ts", keep="last").sort_values("ts").reset_index(drop=True)
    res = resolution_sec(df["ts"].to_numpy())
    est = "active_min" not in df.columns
    if est:
        live = df["close"].notna() & (df["volume"].fillna(0) > 0)
        df = df.copy()
        df["active_min"] = np.where(live, res // 60, 0).astype(int)
        # 거래가 없던 하위 봉은 가격을 비움 (btc.data.minutes_to_15m 과 같은 표현: 거래 없는 구간 = NaN)
        for k in ("open", "high", "low", "close"):
            df.loc[~live, k] = np.nan
        df.loc[~live, "volume"] = 0.0
    if (df["ts"] % res != 0).any():
        raise ValueError("하위 봉 시각이 격자에 맞지 않습니다")
    df.index = from_unix(df["ts"].to_numpy())
    return df, res, est


def read_sub(path):
    return prepare_sub(pd.read_csv(path))


# ══════════ 하위 봉 → 4시간봉 (격자 phase) ══════════
def n_phases_for(res):
    return BAR_SEC // res


def bars_4h_any(df, phase, res, stale_min=STALE_MIN):
    """
    btc.data.bars_4h 를 하위 봉 길이 res에 맞게 일반화 (res=900이면 bars_4h를 그대로 부름).
    phase k 격자는 res×k 만큼 밀림. 완성 봉 = 하위 봉 BAR_SEC/res개, 죽은 봉 = active_min < stale_min.
    """
    if res == 900:
        return bars_4h(df, phase, stale_min=stale_min)
    n_sub = BAR_SEC // res
    if not 0 <= phase < n_sub:
        raise ValueError(f"phase는 0~{n_sub - 1}")
    off = pd.Timedelta(seconds=res * phase)
    g = df.resample("4h", label="left", closed="left", offset=off)
    b = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                      "close": g["close"].last(), "volume": g["volume"].sum(),
                      "active_min": g["active_min"].sum(), "n15": g["ts"].count()})
    full = (b["n15"] == n_sub).to_numpy()
    if not full.any():
        return pd.DataFrame(columns=["ts"] + list(b.columns.drop("n15")) + ["gap_before", "forced_hold"])
    i0, i1 = int(np.argmax(full)), len(full) - int(np.argmax(full[::-1]))
    b = b.iloc[i0:i1]
    alive = ((b["n15"] == n_sub).to_numpy() & (b["active_min"] >= stale_min).to_numpy()
             & b["close"].notna().to_numpy())
    b = b.drop(columns="n15")
    dead_run = np.zeros(len(b), dtype=np.int64)
    run = 0
    for i, ok in enumerate(alive):
        dead_run[i] = run
        run = 0 if ok else run + 1
    b["gap_before"] = dead_run
    b = b[alive].copy()
    b["forced_hold"] = b["gap_before"] >= FORCED_HOLD_GAP
    b.insert(0, "ts", ((b.index - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)).to_numpy(np.int64))
    b["active_min"] = b["active_min"].astype(int)
    return b


def phase_bars(df, res, phases=None):
    n = n_phases_for(res)
    return [bars_4h_any(df, k, res) for k in (range(n) if phases is None else phases)]


def to_phase_data(bars_list):
    out = []
    for b in bars_list:
        X, sig = compute(b)
        out.append(PhaseData(b, X, sig))
    return out


def phases_from_frame(df, phases=None):
    """ts·OHLCV[·active_min] DataFrame → PhaseData 목록 (캐시 없음, 시험용)"""
    sub, res, _ = prepare_sub(df)
    return to_phase_data(phase_bars(sub, res, phases))


def _stamp(path):
    from .. import data as _d, features as _f, env as _e
    h = hashlib.sha1()
    with open(path, "rb") as fp:
        h.update(fp.read())
    for mod in (_d, _f, _e):
        with open(mod.__file__, "rb") as fp:
            h.update(fp.read())
    with open(os.path.abspath(__file__), "rb") as fp:
        h.update(fp.read())
    return h.hexdigest()[:16]


def file_sha256(path):
    with open(path, "rb") as fp:
        return hashlib.sha256(fp.read()).hexdigest()


def load_asset(asset, path=None):
    """자산의 PhaseData 목록 (15분봉이면 16개, 1시간봉이면 4개). 특징은 data/btc/cache에 캐시 (git 제외)."""
    path = path or asset_file(asset)
    os.makedirs(CACHE_DIR, exist_ok=True)
    cpath = os.path.join(CACHE_DIR, f"holdout_{asset}_{_stamp(path)}.npz")
    if os.path.exists(cpath):
        z = np.load(cpath)
        n = int(z["n_phases"])
        out = []
        for k in range(n):
            bars = pd.DataFrame({c: z[f"{k}_{c}"] for c in BARS_COLS})
            out.append(PhaseData(bars, z[f"{k}_X"], z[f"{k}_sig"]))
        return out
    sub, res, _ = read_sub(path)
    save = {"n_phases": np.int64(n_phases_for(res))}
    for k, b in enumerate(phase_bars(sub, res)):
        X, sig = compute(b)
        for c in BARS_COLS:
            save[f"{k}_{c}"] = b[c].to_numpy()
        save[f"{k}_X"], save[f"{k}_sig"] = X, sig
    tmp = f"{cpath}.{os.getpid()}.tmp.npz"
    np.savez(tmp, **save)
    os.replace(tmp, cpath)
    return load_asset(asset, path)


# ══════════ 기간 ══════════
def data_span(asset, path=None):
    """(첫 하위 봉 시작, 마지막 하위 봉 끝) — 가격은 읽지 않음"""
    ts = pd.read_csv(path or asset_file(asset), usecols=["ts"])["ts"].to_numpy(np.int64)
    res = resolution_sec(ts)
    return pd.Timestamp(int(ts[0]), unit="s", tz="UTC"), pd.Timestamp(int(ts[-1]) + res, unit="s", tz="UTC")


def eval_window(asset, path=None, datas=None):
    """
    판정 기간 (사전 등록, 2026-09-25 추가 규칙): 시작 = 다음 둘 중 늦은 날 이후 첫 1월 1일
      (1) 데이터 시작 + 2년,  (2) phase 0 지표 예열이 끝난 봉 + 2년 (학습 자료가 최소 2년 쌓인 뒤)
    끝 = 데이터 끝(자정으로 내림), 최대 OOS_END. 가격·수익은 읽지 않습니다(시각만).
    """
    start, end = data_span(asset, path)
    datas = datas or load_asset(asset, path)
    d0 = datas[0]
    warm = pd.Timestamp(int(d0.ts[min(WARMUP, d0.T - 1)]), unit="s", tz="UTC")
    return window_rule(start, end, warm)


def window_rule(start, end, warm):
    """eval_window 의 규칙만 (시험용): max(데이터 시작, 예열 끝) + 2년 이후 첫 1월 1일 ~ 데이터 끝(자정 내림, 최대 OOS_END)"""
    s2 = max(start, warm) + pd.DateOffset(years=2)
    lo = pd.Timestamp(f"{s2.year}-01-01", tz="UTC")
    if lo < s2:
        lo = pd.Timestamp(f"{s2.year + 1}-01-01", tz="UTC")
    hi = min(OOS_END, end.floor("D"))
    return lo, hi


# ══════════ 실행 (학습) ══════════
def _pool_nonempty(cfg, datas, Tk):
    """그 달 학습 표본이 하나라도 있는지 — KTrainer.make_pool / DirectTrainer.make_pool 의 조건 그대로"""
    S = int(cfg.get("stride", 1))
    for d in datas[:cfg["phases"]]:
        if cfg.get("algo") == "direct":
            span = S * int(cfg.get("seq_len", 60)) + 1
            t = np.arange(WARMUP, d.T - span - 1)
            t = t[d.ts[t] >= FIRST_TRAIN]
            t = t[(d.ts[t + span] + BAR_SEC) <= Tk]
            t = t[(d.ts[t + span] - d.ts[t]) == span * BAR_SEC]
        else:
            t = d.eligible(Tk, FIRST_TRAIN, WARMUP)
            if S > 1 and len(t):
                t = t[t + S + 1 < d.T]
                t = t[(d.ts[t + S + 1] - d.ts[t + 1]) == S * BAR_SEC]
                t = t[(d.ts[t + S] + BAR_SEC) <= Tk]
        if len(t):
            return True
    return False


def _gate_short(cfg, d0, Tk):
    """DQN 계열 1월 점검이 '최근 2년 봉 부족'으로 반드시 거부하는지 (walk.gate_cold 와 같은 조건)"""
    if cfg.get("algo") not in (None, "dqn") or cfg.get("gate", "p0") == "none":
        return False
    a, b = decision_range(d0, Tk - 2 * 365 * 86400, Tk)
    return b - a < 2 * 365 * 6 * 0.9


def first_month(cfg, datas, end):
    """학습 결과가 쓰일 수 있는 첫 재학습 달 (그 앞 달들은 결과가 '모델 없음'으로 정해져 있음)"""
    for T_k in months():
        if T_k >= end:
            break
        Tk = int(T_k.timestamp())
        if _pool_nonempty(cfg, datas, Tk) and not _gate_short(cfg, datas[0], Tk):
            return T_k
    return None


def holdout_code_hash():
    from . import walk
    h = hashlib.sha1(walk.code_hash().encode())
    with open(os.path.abspath(__file__), "rb") as fp:
        h.update(fp.read())
    return h.hexdigest()[:12]


def run_path(asset, name, r):
    return os.path.join(RUNS_H, asset, name, f"rep{r:02d}.npz")


def run_replication(asset, cfg, r, datas=None, path=None):
    """자산 하나·반복 하나 — btc.research.walk.run_replication 을 그대로 부르고 앞쪽 빈 달만 건너뜀"""
    from . import walk
    path = path or asset_file(asset)
    datas = datas or load_asset(asset, path)
    lo, hi = eval_window(asset, path, datas)
    end = hi
    fm = first_month(cfg, datas, end)
    all_m = [m for m in months() if m < end]
    if fm is None:
        raise RuntimeError(f"{asset}: 학습할 수 있는 달이 없습니다")
    orig = walk.months
    walk.months = lambda *a, **k: [m for m in orig(*a, **k) if m >= fm]     # 이 프로세스 안에서만
    try:
        res = walk.run_replication(cfg, r, datas=datas, end=end)
    finally:
        walk.months = orig
    skipped = [dict(month=str(m.date()), kind="skipped", accepted=None, note="no_pool_or_gate_short_window_hold_cash")
               for m in all_m if m < fm]
    res["log"] = skipped + res["log"]
    res["meta_extra"] = dict(asset=asset, data_file=os.path.relpath(path, os.path.dirname(os.path.dirname(DATA_DIR))),
                             data_sha256=file_sha256(path), first_month=str(fm.date()), end=str(end),
                             eval_window=[str(lo), str(hi)])
    return res


def save_run(asset, cfg, r, res, ch):
    """walk.save_run 과 같은 형식 (U_판단비용, s_상태, log, meta) — 경로만 holdout/<자산>/<변형>/"""
    from .walk import cfg_hash
    path = run_path(asset, cfg["name"], r)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    flat = {f"U_{cd:.4f}": v for cd, v in res["U"].items()}
    for k, v in res["last_state"].items():
        flat[f"s_{k}"] = v
    flat["log"] = np.frombuffer(json.dumps(res["log"]).encode(), dtype=np.uint8)
    meta = dict(cfg=cfg, cfg_hash=cfg_hash(cfg), code_hash=ch, **res.get("meta_extra", {}))
    flat["meta"] = np.frombuffer(json.dumps(meta, default=list).encode(), dtype=np.uint8)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **flat)
    os.replace(tmp, path)


def load_run(asset, name, r):
    z = np.load(run_path(asset, name, r))
    meta = json.loads(bytes(z["meta"]).decode())
    U = {float(k[2:]): z[k] for k in z.files if k.startswith("U_")}
    return dict(U=U, log=json.loads(bytes(z["log"]).decode()), meta=meta)


def _job(args):
    asset, cfg, r = args
    path = run_path(asset, cfg["name"], r)
    if os.path.exists(path):
        from .walk import cfg_hash
        meta = load_run(asset, cfg["name"], r)["meta"]
        if meta.get("cfg_hash") != cfg_hash(cfg) or meta.get("data_sha256") != file_sha256(asset_file(asset)):
            return asset, cfg["name"], r, "STALE (설정이나 데이터가 다름 — 지우고 다시 돌려야 함)"
        return asset, cfg["name"], r, "cached"
    t = time.time()
    ch = holdout_code_hash()
    try:
        res = run_replication(asset, cfg, r)
    except Exception:
        import traceback
        return asset, cfg["name"], r, "FAILED\n" + traceback.format_exc()
    save_run(asset, cfg, r, res, ch)
    return asset, cfg["name"], r, round(time.time() - t, 1)


def run_many(assets, names, reps, workers=4):
    from .variants import VARIANTS
    for n in names:
        if n not in VARIANTS:
            raise SystemExit(f"{n}: btc/research/variants.py 에 없는 변형입니다 (확증 시험은 고정한 설정만)")
    for a in assets:
        load_asset(a)                                   # 특징 캐시를 먼저 만들어 둠
    jobs = [(a, VARIANTS[n], r) for a in assets for n in names for r in range(reps)]
    if workers <= 1:
        for j in jobs:
            print("  완료", _job(j), flush=True)
        return
    with mp.get_context("spawn").Pool(workers) as pool:
        for out in pool.imap_unordered(_job, jobs):
            print("  완료", out, flush=True)


# ══════════ 판정 ══════════
class HoldoutWindow:
    """btc.evaluate.Window 와 같은 계산. 비트코인 잠금 구간 감사(guard)는 비트코인 전용이라 뺐습니다."""

    def __init__(self, datas, lo, hi):
        from ..evaluate import Window
        self._W = Window.__new__(Window)
        self._W.datas, self._W.lo, self._W.hi = datas, int(lo.timestamp()), int(hi.timestamp())
        self._W.rng = {}
        for k, d in enumerate(datas[:1]):
            a, b = decision_range(d, self._W.lo, self._W.hi)
            while b > a and (d.ts[b] + BAR_SEC > self._W.hi or (b + 1 < d.T and d.ts[b + 1] > self._W.hi)):
                b -= 1
            self._W.rng[k] = (a, b)
        self.datas, self.rng, self.lo, self.hi = datas, self._W.rng, self._W.lo, self._W.hi

    def run(self, targets, cost, weights=False):
        return self._W.run(targets, cost, phase=0, weights=weights)


def variant_targets(W, cfg, run):
    """btc.research.screen.targets 와 같은 계산 (판단비용 C_DEC 한 가지)"""
    from .rl import k_policy, trend_filter, allowed_matrix
    from .walk import decision_mask
    d = W.datas[0]
    a, b = W.rng[0]
    sel = np.arange(a, b)[decision_mask(d, cfg.get("stride", 1))[a:b]]
    tg = np.full(d.T, np.nan)
    acts = np.asarray(cfg.get("acts", (0.0, 1.0)), dtype=float)
    if cfg.get("algo") == "direct" or cfg.get("output") == "weights":
        tg[sel] = np.round(run["U"][0.0][sel, 0] * 4) / 4
        return tg, True
    U = run["U"][C_DEC][sel]
    allowed = allowed_matrix(trend_filter(d.c)[sel], acts) if cfg.get("veto_b2") else None
    w = acts[k_policy(U, acts, C_DEC, d.forced_hold[sel], allowed=allowed)].astype(float)
    w[np.isnan(U).any(axis=1)] = np.nan
    tg[sel] = w
    frac = bool(np.any((acts > 0) & (acts < 1))) or bool(np.any(acts < 0))
    return tg, frac


def pooled_bootstrap(series, n_boot=N_BOOT, seed=BOOT_SEED, mean_block=MEAN_BLOCK):
    """
    자산 여러 개의 평균 ΔSharpe 에 대한 짝지은 정상 부트스트랩.
    series: [(days, r_전략, r_매수보유)] — 자산마다 기간이 달라도 됨.
    모든 자산을 하나의 날짜 축(합집합)에 놓고 같은 날짜 색인으로 함께 뽑아 자산 간 상관을 보존합니다.
    각 표본에서 자산마다 자기 기간에 들어온 날만으로 ΔSR을 계산해 평균.
    자산이 하나면 btc.stats.stationary_bootstrap_diff 와 같은 값(같은 시드)입니다.
    p = mean( (ΔSR_b − obs) ≥ obs )  (stats 와 같은 중심화 단측 p)
    """
    step = 86400
    d0 = min(int(s[0][0]) for s in series)
    d1 = max(int(s[0][-1]) for s in series)
    n = (d1 - d0) // step + 1
    A, B = [], []
    for days, ra, rb in series:
        fa, fb = np.full(n, np.nan), np.full(n, np.nan)
        i = (np.asarray(days, np.int64) - d0) // step
        fa[i], fb[i] = ra, rb
        A.append(fa)
        B.append(fb)
    obs = float(np.mean([S.sharpe(ra) - S.sharpe(rb) for _, ra, rb in series]))
    rng = np.random.default_rng(seed)
    chunk = max(1, 2_000_000 // max(n, 1))
    diffs = np.empty(n_boot)
    done = 0

    def nan_sharpe(x):
        cnt = np.sum(np.isfinite(x), axis=1)
        mu = np.nanmean(x, axis=1)
        sd = np.sqrt(np.nansum((x - mu[:, None]) ** 2, axis=1) / np.maximum(cnt - 1, 1))
        out = np.zeros(len(mu))
        ok = (sd > 0) & (cnt > 1)
        out[ok] = mu[ok] / sd[ok] * math.sqrt(S.DAYS)
        return out

    while done < n_boot:
        k = min(chunk, n_boot - done)
        idx = S.stationary_bootstrap_indices(n, k, mean_block, rng)
        acc = np.zeros(k)
        for fa, fb in zip(A, B):
            acc += nan_sharpe(fa[idx]) - nan_sharpe(fb[idx])
        diffs[done:done + k] = acc / len(A)
        done += k
    lo, hi = np.percentile(diffs, [5.0, 95.0])
    return dict(obs=obs, p=float(np.mean((diffs - obs) >= obs)), ci90=(float(lo), float(hi)),
                se=float(diffs.std(ddof=1)), n_boot=int(n_boot), mean_block=float(mean_block), n_days_union=int(n))


def _ledger_rows():
    if not os.path.exists(LEDGER):
        return []
    with open(LEDGER, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def candidates_so_far():
    out = []
    for row in _ledger_rows():
        for v in row["variants"]:
            if v not in out:
                out.append(v)
    return out


SLOT_ORDER = ("R6_daily_trend8_uniform", "R3_daily_trend8_log5")   # success_criteria.md 추가 조항 (10:15 UTC)
SLOT3_MIN_SHARPE = 1.149                                              # 3번째 자리: 1단계 통과 + 헤드라인 샤프 > R6


def _slot_check(prev, new):
    order = prev + new
    for i, n in enumerate(order[:2]):
        if n != SLOT_ORDER[i]:
            raise SystemExit(f"확증 시험 순서는 {SLOT_ORDER} 입니다 (받은 순서: {order})")
    if len(order) >= 3:
        from .walk import RUNS
        with open(os.path.join(RUNS, "screen.json"), encoding="utf-8") as f:
            v = json.load(f)["variants"].get(order[2])
        if not v or not v.get("passed") or not (v.get("sharpe", 0) > SLOT3_MIN_SHARPE):
            raise SystemExit(f"3번째 후보 {order[2]}는 1단계 통과와 헤드라인 샤프 > {SLOT3_MIN_SHARPE} 조건을 만족해야 합니다")


def evaluate(assets, names, reps=5, record=True, exploratory=False):
    from ..evaluate import baseline_targets, lower_median
    from .variants import VARIANTS
    from .walk import cfg_hash, decision_mask
    prev = candidates_so_far()
    new = [n for n in names if n not in prev]
    if len(prev) + len(new) > MAX_CANDIDATES:
        raise SystemExit(f"확증 시험 후보는 최대 {MAX_CANDIDATES}개입니다. 이미 보낸 후보: {prev}, 새 후보: {new}")
    for n in names:
        if n not in VARIANTS:
            raise SystemExit(f"{n}: 고정된 변형이 아닙니다")
    if not exploratory:
        if sorted(assets) != sorted(ASSETS) or reps != 5:
            raise SystemExit(f"확증 판정은 자산 {ASSETS} 전부, 반복 5로만 합니다 (탐색용은 --exploratory, 기록은 남고 판정에는 안 씀)")
        _slot_check(prev, new)
    # 계산 전에 의도를 먼저 기록 (결과를 보고 기록을 빼는 일이 없게)
    if record:
        os.makedirs(RUNS_H, exist_ok=True)
        with open(LEDGER, "a", encoding="utf-8") as f:
            f.write(json.dumps(dict(kind="intent", time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                    assets=list(assets), variants=list(names), reps=reps,
                                    exploratory=bool(exploratory)), ensure_ascii=False) + "\n")
    # 모든 반복이 있고, 지금 데이터 파일로 만든 것인지 먼저 확인 (일부만 보고 판정하지 않음)
    for a in assets:
        sha = file_sha256(asset_file(a))
        for n in names:
            for r in range(reps):
                p = run_path(a, n, r)
                if not os.path.exists(p):
                    raise SystemExit(f"{a} {n} rep{r:02d} 실행 결과가 없습니다: {p}")
                meta = load_run(a, n, r)["meta"]
                if meta.get("data_sha256") != sha:
                    raise SystemExit(f"{a} {n} rep{r:02d}: 지금 데이터 파일과 다른 파일로 학습한 결과입니다")
                if meta.get("cfg_hash") != cfg_hash(VARIANTS[n]):
                    raise SystemExit(f"{a} {n} rep{r:02d}: 지금 변형 설정과 다른 설정으로 학습한 결과입니다")
    code_hashes = {n: sorted({load_run(a, n, r)["meta"].get("code_hash") for a in assets for r in range(reps)})
                   for n in names}
    for n, hs in code_hashes.items():
        if len(hs) != 1:
            raise SystemExit(f"{n}: 반복·자산마다 코드 지문이 다릅니다 {hs}")
    out = dict(kind="exploratory" if exploratory else "confirmatory",
               time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), assets=list(assets), variants=list(names),
               code_hashes=code_hashes, cfg_hashes={n: cfg_hash(VARIANTS[n]) for n in names},
               reps=reps, cost=COST, c_dec=C_DEC, alpha=ALPHA, predeclared_assets=list(ASSETS),
               asset_set_matches_predeclared=sorted(assets) == sorted(ASSETS),
               candidate_order=prev + new, per_asset={}, results={})
    ctx = {}
    for a in assets:
        datas = load_asset(a)
        lo, hi = eval_window(a, datas=datas)
        W = HoldoutWindow(datas, lo, hi)
        bh = W.run(baseline_targets(W)["B0"], COST)
        sb = S.summary(bh["r"])
        ctx[a] = (W, bh)
        out["per_asset"][a] = dict(window=[str(lo.date()), str(hi.date())], n_days=len(bh["r"]),
                                   bh=dict(sharpe=sb["sharpe"], cagr=sb["cagr"], max_dd=sb["max_dd"]))
    for n in names:
        cfg = VARIANTS[n]
        per, series = {}, []
        for a in assets:
            W, bh = ctx[a]
            sb = S.summary(bh["r"])
            res, no_model = [], []
            d = W.datas[0]
            wa, wb = W.rng[0]
            sel = np.arange(wa, wb)[decision_mask(d, cfg.get("stride", 1))[wa:wb]]
            for r in range(reps):
                run = load_run(a, n, r)
                tg, frac = variant_targets(W, cfg, run)
                U0 = next(iter(run["U"].values())) if cfg.get("algo") == "direct" or cfg.get("output") == "weights" \
                    else run["U"][C_DEC]
                no_model.append(float(np.isnan(U0[sel]).any(axis=1).mean()))       # 모델 없는(현금) 판단일 비율
                res.append(W.run(tg, COST, weights=frac))
            srs = [S.sharpe(x["r"]) for x in res]
            dsr = [s - sb["sharpe"] for s in srs]
            lm = lower_median(dsr)
            h = res[lm]
            sh = S.summary(h["r"])
            boot = S.stationary_bootstrap_diff(h["r"], bh["r"], mean_block=MEAN_BLOCK, n_boot=N_BOOT, seed=BOOT_SEED)
            per[a] = dict(headline_rep=lm, sharpe_reps=srs, d_sharpe_reps=dsr, sharpe=sh["sharpe"], cagr=sh["cagr"],
                          max_dd=sh["max_dd"], exposure=float(np.nanmean(h["pos"])), d_sharpe=dsr[lm],
                          p_single=boot["p"], no_model_share=no_model, no_model_flag=bool(max(no_model) > 0.05),
                          H1=bool(sh["sharpe"] > sb["sharpe"]),
                          H3=bool(sh["max_dd"] > sb["max_dd"]))
            series.append((h["days"], h["r"], bh["r"]))
        pb = pooled_bootstrap(series)
        H2 = bool(pb["obs"] > 0 and pb["p"] < ALPHA)
        H1 = all(v["H1"] for v in per.values())
        H3 = all(v["H3"] for v in per.values())
        out["results"][n] = dict(per_asset=per, pooled=pb, H1=H1, H2=H2, H3=H3, passed=bool(H1 and H2 and H3))
    if record:
        with open(LEDGER, "a", encoding="utf-8") as f:
            f.write(json.dumps(_clean(out), ensure_ascii=False) + "\n")
    return out


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def print_report(out):
    ok = lambda b: "통과" if b else "실패"
    print(f"확증 시험 (사전 등록: btc/research/success_criteria.md) — 편도 {COST*100:.2f}%, 하루 한 번 판단, "
          f"유의수준 0.10/3 = {ALPHA:.4f}")
    print(f"후보 순서: {out['candidate_order']}  (최대 {MAX_CANDIDATES}개)")
    if not out["asset_set_matches_predeclared"]:
        print(f"주의: 자산 목록 {out['assets']} 이 사전에 정한 {out['predeclared_assets']} 와 다릅니다")
    for a, v in out["per_asset"].items():
        b = v["bh"]
        print(f"  {a}: 기간 {v['window'][0]} ~ {v['window'][1]} ({v['n_days']}일)  매수·보유 샤프 {b['sharpe']:.2f} "
              f"CAGR {b['cagr']*100:.1f}% MDD {b['max_dd']*100:.1f}%")
    for n, res in out["results"].items():
        print(f"\n{n}")
        print(f"  {'자산':5s} {'헤드라인':>6s} {'샤프':>6s} {'ΔSR':>6s} {'반복 샤프':24s} {'MDD':>7s} {'보유MDD':>7s} "
              f"{'노출':>5s} {'p(단일)':>7s} H1 H3")
        for a, v in res["per_asset"].items():
            bh = out["per_asset"][a]["bh"]
            reps = ",".join(f"{s:.2f}" for s in v["sharpe_reps"])
            print(f"  {a:5s} {'rep%02d' % v['headline_rep']:>6s} {v['sharpe']:6.2f} {v['d_sharpe']:+6.2f} {reps:24s} "
                  f"{v['max_dd']*100:6.1f}% {bh['max_dd']*100:6.1f}% {v['exposure']*100:4.0f}% {v['p_single']:7.3f} "
                  f"{'○' if v['H1'] else '×'}  {'○' if v['H3'] else '×'}")
        pb = res["pooled"]
        print(f"  평균 ΔSharpe {pb['obs']:+.3f}  부트스트랩 단측 p = {pb['p']:.4f} (기준 < {ALPHA:.4f}), "
              f"90% 구간 [{pb['ci90'][0]:+.2f}, {pb['ci90'][1]:+.2f}]")
        print(f"  H1 {ok(res['H1'])} · H2 {ok(res['H2'])} · H3 {ok(res['H3'])}  →  "
              f"{'2단계 통과' if res['passed'] else '2단계 실패: 매수·보유를 이긴다는 증거 없음'}")


def info(assets):
    """봉 수·기간·죽은 봉·첫 학습 달 — 가격 경로 통계는 계산하지 않음"""
    from .variants import VARIANTS
    cfg = VARIANTS["R6_daily_trend8_uniform"]
    for a in assets:
        path = asset_file(a)
        datas = load_asset(a, path)
        lo, hi = eval_window(a, path)
        start, end = data_span(a, path)
        d0 = datas[0]
        fm = first_month(cfg, datas, hi)
        print(f"{a}: {os.path.basename(path)} sha256 {file_sha256(path)[:16]}  {start} ~ {end}  phases {len(datas)}")
        print(f"   4시간봉(phase 0) {d0.T}개, 죽은 봉 뒤 봉 {int((d0.gap > 0).sum())}개, 관망 봉 {int(d0.forced_hold.sum())}개, "
              f"예열 끝 {pd.Timestamp(int(d0.ts[min(WARMUP, d0.T - 1)]), unit='s', tz='UTC').date()}")
        print(f"   판정 기간 {lo.date()} ~ {hi.date()}, 첫 학습 달(R6 기준) {fm.date() if fm is not None else None}")


def main():
    ap = argparse.ArgumentParser(description="2단계 확증 시험 (보지 않은 코인)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("info")
    i.add_argument("assets", nargs="+")
    r = sub.add_parser("run")
    r.add_argument("asset")
    r.add_argument("variant", nargs="+")
    r.add_argument("--reps", type=int, default=5)
    r.add_argument("--workers", type=int, default=4)
    e = sub.add_parser("evaluate")
    e.add_argument("assets", nargs="*", default=list(ASSETS))
    e.add_argument("--variants", nargs="+", required=True)
    e.add_argument("--reps", type=int, default=5)
    e.add_argument("--out", default=os.path.join(RUNS_H, "report.json"))
    e.add_argument("--exploratory", action="store_true", help="판정에 쓰지 않는 탐색용 (기록은 남음)")
    a = ap.parse_args()
    if a.cmd == "info":
        info(a.assets)
    elif a.cmd == "run":
        assets = list(ASSETS) if a.asset == "all" else [a.asset]
        run_many(assets, a.variant, a.reps, a.workers)
    else:
        out = evaluate(a.assets or list(ASSETS), a.variants, a.reps, exploratory=a.exploratory)
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(_clean(out), f, ensure_ascii=False, indent=1)
        print_report(out)


if __name__ == "__main__":
    main()
