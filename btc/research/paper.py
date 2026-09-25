# -*- coding: utf-8 -*-
"""
3단계 — 앞으로의 비트코인 모의매매 (btc/research/success_criteria.md, 2026-09-25 11:06 UTC 추가 조항)

코드 지문과 설정을 고정한 후보 하나를 2026-09-25부터 모의매매로 돌립니다. 백테스트(walk.run_replication +
screen.targets + evaluate.Window.run)와 **같은 코드·같은 시드·같은 판단 규칙·같은 체결 회계**를 쓰고,
판단 시각에 이미 있던 데이터만 봅니다. 점검일은 2027-03-25(중간, 판정 없음)와 2027-09-25(판정)입니다.

  python -m btc.research.paper register R6_daily_trend8_uniform [--reps 5] [--start 2026-09-25]
  python -m btc.research.paper step   [--now 2026-10-02T00:20Z] [--offline] [--max-minutes 35] [--workers 1]
  python -m btc.research.paper report [--now ...]

흐름 (step 한 번)
  1. (--offline이 아니면) 비트스탬프 1분봉을 btc/live.py 의 fetch_bitstamp_minutes 로 받아
     data/btc/paper/btcusd_15m_ext.csv.gz 에 15분봉으로 덧붙입니다. 연구용 15분봉 파일
     (data/btc/btcusd_15m.csv.gz, 2026-09-25 00:00 UTC에서 끊김)은 건드리지 않고, 그 뒤만 따로 둡니다.
     · 덧붙이기만 합니다(한 번 쓴 15분봉은 고치지 않음) — '그 시각에 있던 데이터'가 그대로 남습니다.
     · 받은 마지막 1분봉이 끝나는 15분 구간까지만 씁니다. 그 사이에 빠진 분은 원본 저장소처럼
       거래량 0으로 채웁니다(→ 거래 없는 분, 봉이 모자라면 '죽은 봉').
     · 조회가 실패하면 멈추지 않고 경고를 남긴 뒤 있는 데이터로 계속합니다(live.py 와 같은 원칙).
  2. 연구용 파일 + 덧붙인 파일을 now 에서 자르고(종가 ≤ now인 15분봉만) phase별 4시간봉·지표를 다시 만듭니다.
     load_phases 와 같은 함수(bars_4h, compute, PhaseData)를 쓰고, 디스크 캐시는 두지 않습니다
     (캐시 무효화 문제가 아예 없음. phase 0은 약 1초, 16개 전부는 약 11초이며 재학습하는 달에만 16개를 만듭니다).
  3. 등록된 변형 × 반복마다 월간 재학습 사슬을 이어 갑니다 — walk.monthly_update 를 run_replication 과
     똑같이(같은 시드 (r, 연, 월, 0), 1월 처음부터·나머지 이어학습·점검, ens·anchor를 달마다 넘김) 부릅니다.
     · 첫 모의매매 달의 모델은 백테스트 첫 달(2017-01)부터 사슬을 다시 돌려 만듭니다(아래 '왜 다시 돌리나').
     · 달마다 학습 직후 모델 상태(ens, anchor)를 저장하므로, 중간에 끊겨도 다음 실행이 같은 결과로 이어 갑니다.
     · 다음 달을 학습하기 직전에 지난달 판단봉 전체로 run_replication 과 같은 호출을 한 번 합니다(마감 호출).
       dp_band 처럼 판단 호출에 상태(지난달 마지막 포지션)가 있는 모델도 백테스트와 똑같이 이어집니다.
  4. 새로 마감된 판단봉(phase 0, 00:00 UTC 마감, walk.decision_mask(d, 6))마다 screen.targets 와 같은 규칙으로
     목표 비중을 정합니다: U 출력은 k_policy(판단비용 0.3%, 직전 포지션을 이어받음), 비중 출력은 25% 단위 반올림.
     · 판단은 그 봉 종가까지의 지표만 씁니다. 체결가(다음 봉 시가)가 아직 없어도 판단은 먼저 기록하고,
       체결·손익은 다음 봉이 생긴 뒤 장부에 반영됩니다.
     · 백테스트는 한 달치 판단봉을 한 번에 신경망에 넣습니다. float32 행렬곱은 묶음 크기에 따라 마지막 비트가
       달라질 수 있어, 달 중간에는 아직 오지 않은 판단봉 자리를 0으로 채워 백테스트와 같은 묶음 크기로 계산합니다
       (행마다 결과는 다른 행의 값과 무관). 달이 끝나면 마감 호출 값과 비교해 다르면 경고합니다.
  5. 반복마다 장부(편도 0.15%, 다음 4시간봉 시가 체결)를 env.simulate / simulate_weights 와
     evaluate.Window.run 과 같은 방식으로 처음부터 다시 계산합니다 — 저장된 판단만으로 결정되므로 멱등입니다.
     매수·보유 장부도 같이 만듭니다.

밀린 실행 따라잡기: 며칠을 건너뛰어도 step 한 번이 빠진 날을 순서대로 처리합니다. 판단은 그 봉 종가까지,
학습은 T_k 이전 데이터만 쓰므로 실시간으로 매일 돌린 것과 같습니다(tests/test_btc_paper.py 가 확인).
같은 now로 두 번 돌리면 아무것도 바뀌지 않습니다.

왜 저장된 연구 실행(research_runs/<변형>/repNN.npz)을 이어 쓰지 않고 2017-01부터 다시 돌리나
  · 저장 파일에는 마지막 달의 ens 상태만 있고 anchor(1월 모델, 이어학습의 L2-SP 기준)가 없습니다.
    10~12월 이어학습에는 anchor가 꼭 필요합니다. dp_band 는 지난달 포지션(last_pos)도 필요합니다.
  · 연구 실행들의 코드 지문(ebe75b0d04e2 등)은 지금 코드(walk.code_hash())와 다릅니다.
  그래서 등록 뒤 첫 step 에서 사슬 전체를 다시 돌립니다(반복마다 R6 약 3~4분, W1 약 7분, 1스레드).
  run_replication 과 비트 단위로 같다는 것은 테스트가 확인합니다. 그 뒤로는 달에 한 번 반복마다 몇 초입니다.

파일 (data/btc/paper/)
  registry.json                 등록부 (설정 전체, 설정·코드 지문, 시작일, 점검일, 판정 규칙)
  btcusd_15m_ext.csv.gz         2026-09-25 이후 15분봉 (덧붙이기만)
  status.json · report.json     마지막 step 상태 · 보고서
  <변형>/state/repNN.pkl        마지막으로 학습한 달의 모델 상태 (다음 실행이 이어 가는 데 필요 — 커밋 대상)
  <변형>/models/repNN/YYYY-MM.pkl  모의매매 기간의 달별 모델 보관본 (재현·감사용, 커밋하지 않아도 됨)
  <변형>/train_log_repNN.jsonl  월간 갱신 기록 (사슬 전체)
  <변형>/decisions_repNN.csv    판단 기록 (덧붙이기만: 판단 시각, U 또는 비중, 목표)
  <변형>/ledger_repNN.csv · fills_repNN.csv · bh_ledger.csv   일별 자산 · 체결 · 매수·보유 (매번 다시 계산)

모델 상태는 pickle 입니다(live.py 의 상태 저장과 같은 방식). 플러그인 기법(dp_band 등)은 모델 객체에 상태가 있어
npz 한 가지 형식으로는 담기 어렵습니다. 이 저장소가 직접 쓴 파일만 읽습니다.
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import copy
import csv
import hashlib
import json
import math
import pickle
import time

import numpy as np
import pandas as pd

from .. import stats as S
from ..data import DATA_DIR, COLS, N_PHASES, bars_4h, minutes_to_15m, from_unix
from ..env import PhaseData, BAR_SEC, simulate, simulate_weights, daily_marks
from ..features import compute
from ..walkforward import months, decision_range, OOS_START
from . import walk
from .rl import k_policy, trend_filter, allowed_matrix

PAPER_DIR = os.path.join(DATA_DIR, "paper")
BASE_15M = os.path.join(DATA_DIR, "btcusd_15m.csv.gz")
EXT_NAME = "btcusd_15m_ext.csv.gz"
DAY = 86400
Q15 = 900
COST = 0.0015            # 편도 비용 (1·2단계와 같음)
C_DEC = 0.003            # 판단비용 = 2 × 0.15% (screen.targets 가 쓰는 값)
W_ROUND = 4              # 비중 출력은 25% 단위 (screen.targets)
START = "2026-09-25"
CHECKPOINTS = ("2027-03-25", "2027-09-25")
VERDICT_RULE = (
    "사전 등록 판정 (success_criteria.md 11:06 UTC 추가 조항): 등록 시작일 00:00 UTC부터 마지막 점검일 00:00 UTC "
    "직전에 마감한 판단봉까지, 하루 한 번(00:00 UTC 마감 봉) 판단·다음 4시간봉 시가 체결·편도 0.15%로 계산한 "
    "일별 수익에서, 반복들 가운데 ΔSharpe(매수·보유 대비)가 작은 쪽 중앙값인 반복을 헤드라인으로 삼아 "
    "헤드라인 샤프 > 매수·보유 샤프 이고 헤드라인 최대낙폭이 매수·보유보다 얕으면 'pass'(앞으로도 유지됨), "
    "아니면 'fail'. 마지막 점검일 전에는 'pending'. 중간 점검(2027-03-25)은 기록만 하고 판정하지 않음. "
    "1년치라 통계적 확증이 아니라 방향성 확인입니다.")
FROZEN = ("cfg_hash", "code_hash", "algo_hash", "reps", "start", "checkpoints", "base_sha256")
HERE = os.path.dirname(os.path.abspath(__file__))


class PaperError(RuntimeError):
    pass


# ══════════ 작은 도구 ══════════
def _ts(s):
    """ISO 문자열·Timestamp → UTC 초 (시간대 없으면 UTC)"""
    t = pd.Timestamp(s)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return int(t.timestamp())


def _iso(ts):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(int(ts)))


def _date(ts):
    return time.strftime("%Y-%m-%d", time.gmtime(int(ts)))


def _utc(ts):
    return pd.Timestamp(int(ts), unit="s", tz="UTC")


def _file_hash(paths, n=12, algo="sha1"):
    h = hashlib.new(algo)
    for p in paths:
        with open(p, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:n] if n else h.hexdigest()


def algo_hash(cfg):
    """플러그인 기법 파일의 지문 (walk.code_hash()에는 algos/*.py·direct.py 가 들어 있지 않음)"""
    a = cfg.get("algo")
    if a in (None, "dqn"):
        return None
    if a == "direct":
        return _file_hash([os.path.join(HERE, "direct.py")])
    return _file_hash([os.path.join(HERE, "algos", f"{a}.py")])


def _cfg_from_json(c):
    """JSON에서 읽은 설정 — 목록은 원래처럼 튜플로 (feat만 목록). cfg_hash 는 둘을 같게 봅니다."""
    return {k: (tuple(v) if isinstance(v, list) and k != "feat" else v) for k, v in c.items()}


def _write_atomic(path, data, mode="w"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, mode + ("b" if isinstance(data, bytes) else ""), **({} if isinstance(data, bytes) else
                                                                       dict(encoding="utf-8", newline=""))) as f:
        f.write(data)
    os.replace(tmp, path)


def _write_json(path, obj):
    _write_atomic(path, json.dumps(obj, ensure_ascii=False, indent=1, default=_json_default) + "\n")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (tuple, set)):
        return list(o)
    return str(o)


def _read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _num(x):
    """CSV용 숫자 — NaN은 빈칸, 나머지는 repr (왕복 시 비트 단위로 같음)"""
    x = float(x)
    return "" if math.isnan(x) else repr(x)


def _write_csv(path, header, rows):
    import io
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    for r in rows:
        w.writerow([r.get(k, "") for k in header])
    _write_atomic(path, buf.getvalue())


def _read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _f(s):
    return float(s) if s not in ("", None) else float("nan")


# ══════════ 등록 ══════════
def reg_path(paper_dir=None):
    return os.path.join(paper_dir or PAPER_DIR, "registry.json")


def load_registry(paper_dir=None):
    return _read_json(reg_path(paper_dir), {"variants": {}})


def register(name, reps=5, start=START, cfg=None, paper_dir=None, base_path=BASE_15M, checkpoints=CHECKPOINTS):
    """
    변형 하나를 고정해 등록합니다. 같은 이름이 이미 있으면 고정 항목(설정·코드·기법 지문, 반복 수, 시작일,
    점검일, 연구용 15분봉 파일 지문)이 모두 같을 때만 그대로 두고, 하나라도 다르면 거부합니다.
    cfg: 기본은 variants.VARIANTS[name]. (테스트에서만 학습 스텝을 줄인 설정을 넘김)
    checkpoints: 기본 (2027-03-25, 2027-09-25). 마지막 것이 판정일입니다. (CLI에서는 바꿀 수 없음)
    """
    if cfg is None:
        from .variants import VARIANTS
        if name not in VARIANTS:
            raise PaperError(f"알 수 없는 변형: {name}")
        cfg = VARIANTS[name]
    cfg = dict(cfg)
    cfg["name"] = name
    st = _ts(start)
    st = st - st % DAY                                        # 시작일 00:00 UTC
    cfg_js = json.loads(json.dumps(cfg, default=list))
    ent = dict(
        name=name, cfg=cfg_js, cfg_hash=walk.cfg_hash(cfg), code_hash=walk.code_hash(),
        algo=cfg.get("algo") or "dqn", algo_hash=algo_hash(cfg), reps=int(reps), start=_date(st),
        first_decision=_iso(st) + " 이후 처음 마감하는 00:00 UTC phase 0 봉(살아 있는 봉)",
        chain_from=str(OOS_START.date()),
        checkpoints=[_date(_ts(c)) for c in checkpoints], final_checkpoint=_date(_ts(checkpoints[-1])),
        verdict_rule=VERDICT_RULE,
        policy=dict(cost=COST, c_dec=C_DEC, stride=int(cfg.get("stride", 1)), weights_round=1.0 / W_ROUND,
                    weights_band=0.02, fill="다음 4시간봉 시가", headline="ΔSharpe 작은 쪽 중앙값 반복"),
        base_15m=os.path.relpath(os.path.abspath(base_path), os.path.dirname(os.path.dirname(DATA_DIR))),
        base_sha256=_file_hash([base_path], n=None, algo="sha256"),
        paper_hash=_file_hash([os.path.abspath(__file__)]),
        walkforward_hash=_file_hash([os.path.join(os.path.dirname(HERE), "walkforward.py")]),
        registered_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    reg = load_registry(paper_dir)
    old = reg["variants"].get(name)
    if old is not None:
        diff = [k for k in FROZEN if old.get(k) != ent[k]]
        if diff:
            raise PaperError(f"{name}: 이미 다른 내용으로 등록돼 있습니다 ({', '.join(diff)}). "
                             f"고정한 후보는 바꿀 수 없습니다 — 새 이름으로 등록하세요.")
        return old
    reg["variants"][name] = ent
    _write_json(reg_path(paper_dir), reg)
    return ent


def check_entry(ent, base_path=BASE_15M, check_base=True):
    """지금 코드·설정이 등록 당시와 같은지. 다르면 (막는 문제 목록, 경고 목록)"""
    cfg = _cfg_from_json(ent["cfg"])
    bad, warn = [], []
    if walk.cfg_hash(cfg) != ent["cfg_hash"]:
        bad.append("cfg_hash")
    if walk.code_hash() != ent["code_hash"]:
        bad.append("code_hash")
    if algo_hash(cfg) != ent["algo_hash"]:
        bad.append("algo_hash")
    if check_base and _file_hash([base_path], n=None, algo="sha256") != ent["base_sha256"]:
        bad.append("base_sha256")
    if _file_hash([os.path.abspath(__file__)]) != ent.get("paper_hash"):
        warn.append("paper_hash")
    if _file_hash([os.path.join(os.path.dirname(HERE), "walkforward.py")]) != ent.get("walkforward_hash"):
        warn.append("walkforward_hash")
    return bad, warn


# ══════════ 데이터: 연구용 15분봉 + 덧붙인 15분봉 ══════════
_BASE = {"key": None, "df": None}


def _read_base(base_path):
    st = os.stat(base_path)
    key = (os.path.abspath(base_path), st.st_size, st.st_mtime_ns)
    if _BASE["key"] != key:
        _BASE["key"], _BASE["df"] = key, pd.read_csv(base_path)
    return _BASE["df"]


def ext_path(paper_dir=None):
    return os.path.join(paper_dir or PAPER_DIR, EXT_NAME)


def load_15m_all(paper_dir=None, base_path=BASE_15M):
    """연구용 15분봉(2026-09-25 00:00 UTC에서 끊김) 뒤에 덧붙인 15분봉을 이어 붙임 — load_15m 과 같은 형식"""
    df = _read_base(base_path)
    p = ext_path(paper_dir)
    if os.path.exists(p):
        ext = pd.read_csv(p)
        if len(ext):
            if int(ext["ts"].iloc[0]) <= int(df["ts"].iloc[-1]):
                raise PaperError("덧붙인 15분봉이 연구용 파일과 겹칩니다")
            df = pd.concat([df, ext[df.columns]], ignore_index=True)
    df = df.copy()
    df.index = from_unix(df["ts"].to_numpy())
    return df


def truncate(df15, now):
    """now 시각에 이미 끝난 15분봉만 (종가 ≤ now)"""
    return df15[df15["ts"].to_numpy() + Q15 <= int(now)]


def extend_cache(paper_dir=None, base_path=BASE_15M, now=None, fetch=None):
    """
    비트스탬프 1분봉으로 15분봉을 now까지 덧붙입니다 (btc.live.fetch_bitstamp_minutes 사용).
    반환 dict(added=새 15분봉 수, until=덧붙인 끝 시각). 한 번 쓴 봉은 고치지 않습니다.
    fetch(start, end): 테스트용 대체 함수 (기본 live.fetch_bitstamp_minutes)
    """
    if fetch is None:
        from ..live import fetch_bitstamp_minutes as fetch
    now = int(time.time() if now is None else now)
    p = ext_path(paper_dir)
    base_last = int(_read_base(base_path)["ts"].iloc[-1])
    ext = pd.read_csv(p) if os.path.exists(p) else None
    last = int(ext["ts"].iloc[-1]) if ext is not None and len(ext) else base_last
    start = last + Q15
    end = ((now - 60) // Q15) * Q15                  # 진행 중인 분은 빼고, 다 끝난 15분 구간까지만
    if end - start < Q15:
        return dict(added=0, until=_iso(start))
    m = fetch(start, end)
    if m is None or not len(m):
        return dict(added=0, until=_iso(start), note="no_minutes")
    m = m[(m["timestamp"] >= start) & (m["timestamp"] < end)]
    if not len(m):
        return dict(added=0, until=_iso(start), note="no_minutes")
    got_end = min(end, ((int(m["timestamp"].max()) + 60) // Q15) * Q15)   # 받은 마지막 분이 끝나는 15분 구간까지
    if got_end <= start:
        return dict(added=0, until=_iso(start), note="partial_interval")
    grid = np.arange(start, got_end, 60, dtype=np.int64)
    m = (m.drop_duplicates("timestamp", keep="last").set_index("timestamp")
         [["open", "high", "low", "close", "volume"]].reindex(grid))
    m["volume"] = m["volume"].fillna(0.0)                                # 빠진 분 = 거래 없음 (원본 저장소와 같은 규칙)
    m = m.rename_axis("timestamp").reset_index()
    q = minutes_to_15m(m, cutoff=_utc(got_end)).copy()
    for k in ("open", "high", "low", "close"):                           # build_cache 와 같은 반올림
        q[k] = q[k].round(2)
    q["volume"] = q["volume"].round(8)
    q = q[COLS]
    q = q[q["ts"] >= start]
    if not len(q):
        return dict(added=0, until=_iso(start))
    out = pd.concat([ext, q], ignore_index=True) if ext is not None and len(ext) else q
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp.csv.gz"
    out.to_csv(tmp, index=False, compression={"method": "gzip", "mtime": 0})
    os.replace(tmp, p)
    return dict(added=int(len(q)), until=_iso(int(q["ts"].iloc[-1]) + Q15))


# ══════════ phase 데이터 (load_phases 와 같은 계산, 필요한 phase만) ══════════
_PH = {"key": None, "ph": {}}


def _df_key(df15):
    h = hashlib.sha1(pd.util.hash_pandas_object(df15[COLS], index=False).to_numpy().tobytes())
    return (len(df15), h.hexdigest())


def phases_for(df15, n, key=None):
    """
    잘린 15분봉 → phase 0..n−1 의 PhaseData. load_phases 와 같은 함수(bars_4h → compute → PhaseData)이고
    값도 비트 단위로 같습니다(테스트가 확인). 같은 데이터면 프로세스 안에서 객체를 재사용합니다
    (agent._flatten 이 객체 동일성으로 캐시하므로).
    """
    key = key or _df_key(df15)
    if _PH["key"] != key:
        _PH["key"], _PH["ph"] = key, {}
    for k in range(n):
        if k not in _PH["ph"]:
            b = bars_4h(df15, k)
            X, sig = compute(b)
            _PH["ph"][k] = PhaseData(b, X, sig)
    return [_PH["ph"][k] for k in range(n)]


# ══════════ 모델 상태 ══════════
def _slim(ens):
    """저장·복사용 — _DirectModel 은 학습기(tr, 학습 데이터 전체)를 붙들고 있어 떼어 냄 (판단에는 안 씀)"""
    if ens is not None and hasattr(ens, "tr"):
        ens = copy.copy(ens)
        del ens.tr
    return ens


def _state_path(vdir, r):
    return os.path.join(vdir, "state", f"rep{r:02d}.pkl")


def _load_state(vdir, r):
    p = _state_path(vdir, r)
    if not os.path.exists(p):
        return dict(month=None, ens=None, anchor=None, entry=None)
    with open(p, "rb") as f:
        return pickle.load(f)


def _save_state(vdir, r, st, archive=False):
    data = pickle.dumps(st, protocol=4)
    _write_atomic(_state_path(vdir, r), data)
    if archive:
        _write_atomic(os.path.join(vdir, "models", f"rep{r:02d}", st["month"][:7] + ".pkl"), data)


def _append_log(vdir, r, entry):
    """월간 갱신 기록 한 줄 (이미 그 달이 마지막 줄이면 건너뜀 — 저장 도중 끊긴 경우)"""
    p = os.path.join(vdir, f"train_log_rep{r:02d}.jsonl")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            lines = [l for l in f.read().splitlines() if l.strip()]
        if lines and json.loads(lines[-1]).get("month") == entry.get("month"):
            return
    os.makedirs(vdir, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=_json_default) + "\n")


def _log_months(vdir, r):
    p = os.path.join(vdir, f"train_log_rep{r:02d}.jsonl")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(l)["month"] for l in f if l.strip()]


# ══════════ 판단 ══════════
def _is_weights(cfg):
    return cfg.get("algo") == "direct" or cfg.get("output") == "weights"


def _is_frac(cfg):
    """screen.targets 와 같은 기준: 비중 출력이거나 행동에 0·1 말고 다른 비중이 있으면 simulate_weights"""
    if _is_weights(cfg):
        return True
    acts = np.asarray(cfg.get("acts", (0.0, 1.0)), dtype=float)
    return bool(np.any((acts > 0) & (acts < 1))) or bool(np.any(acts < 0))


def dec_header(cfg):
    K = len(cfg.get("acts", (0.0, 1.0)))
    mid = ["w_raw"] if _is_weights(cfg) else [f"u{i}" for i in range(K)] + ["act_idx"]
    return ["close_utc", "ts_close", "model_month"] + mid + ["target", "forced_hold", "decided_at", "lag_h"]


def _dec_path(vdir, r):
    return os.path.join(vdir, f"decisions_rep{r:02d}.csv")


def _month_sel(d0, mask, Tk, Tn):
    """종가가 [Tk, Tn)인 판단봉 (run_replication 의 decision_range + decision_mask 와 같음; 데이터 끝 제한 없이)"""
    close = d0.ts + BAR_SEC
    a = int(np.searchsorted(close, Tk, side="left"))
    b = int(np.searchsorted(close, Tn, side="left"))
    return np.arange(a, b)[mask[a:b]]


def _future_slots(last_close, Tn, stride):
    """아직 마감하지 않은(종가 > last_close) 이달 판단봉 자리 수 — phase 0은 종가가 stride×4시간의 배수"""
    P = stride * BAR_SEC
    first = (int(last_close) // P + 1) * P
    return max(0, -(-(int(Tn) - first) // P)) if first < Tn else 0


def _outputs(ens, X, direct):
    """판단용 출력 (모델 사본으로 — 상태 있는 모델도 저장본은 그대로 둠). U는 판단비용 0.3%"""
    m = copy.deepcopy(ens)
    if direct:
        return np.asarray(m.weights(X), dtype=float)
    U, _ = m.values(X, C_DEC)
    return U


def _finalize(ens, d0, sel, direct):
    """run_replication 과 똑같은 달 마감 호출 (실제 객체로 — 상태 있는 모델의 다음 달 시작 상태가 여기서 정해짐)"""
    if ens is None or len(sel) == 0:
        return None
    if direct:
        return np.asarray(ens.weights(d0.X[sel]), dtype=float)
    out = None
    for cd in walk.C_DECS:
        u, _ = ens.values(d0.X[sel], cd)
        if cd == C_DEC:
            out = u
    return out


def _decide(cfg, ens, d0, mask, Tk, Tn, lo, last_dec, p_idx, now, model_month):
    """
    이달 판단봉 가운데 아직 판단하지 않은 것(종가 ≥ lo, > last_dec). 반환 (행 목록, 새 p_idx).
    신경망 출력은 백테스트와 같은 묶음 크기(이달 판단봉 수)로 계산 — 아직 오지 않은 자리는 0으로 채움.
    """
    close = d0.ts + BAR_SEC
    sel = _month_sel(d0, mask, Tk, Tn)
    new_pos = np.nonzero((close[sel] >= lo) & (close[sel] > last_dec))[0]
    if not len(new_pos):
        return [], p_idx
    new = sel[new_pos]
    direct = _is_weights(cfg)
    stride = int(cfg.get("stride", 1))
    n_fut = _future_slots(close[-1], Tn, stride)
    if ens is None:
        out = np.full((len(new),) if direct else (len(new), len(cfg.get("acts", (0.0, 1.0)))), np.nan)
    else:
        Xb = d0.X[sel]
        if n_fut:
            Xb = np.concatenate([Xb, np.zeros((n_fut, Xb.shape[1]), Xb.dtype)])
        out = _outputs(ens, Xb, direct)[:len(sel)][new_pos]
    # 백테스트는 출력을 float32 배열에 저장한 뒤 판단합니다(run_replication → screen.targets) — 똑같이
    out = np.asarray(out, np.float32)
    fh = d0.forced_hold[new]
    rows = []
    if direct:
        tg = (np.round(out * W_ROUND) / W_ROUND).astype(float)
        idx = None
    else:
        acts = np.asarray(cfg.get("acts", (0.0, 1.0)), dtype=float)
        allowed = allowed_matrix(trend_filter(d0.c)[new], acts) if cfg.get("veto_b2") else None
        idx = k_policy(out, acts, C_DEC, fh, p0_idx=p_idx, allowed=allowed)
        tg = acts[idx].astype(float)
        tg[np.isnan(out).any(axis=1)] = np.nan
        p_idx = int(idx[-1])
    for j, i in enumerate(new):
        row = dict(close_utc=_iso(close[i]), ts_close=int(close[i]), model_month=model_month if ens is not None else "",
                   target=_num(tg[j]), forced_hold=int(bool(fh[j])), decided_at=_iso(now),
                   lag_h="%.2f" % ((now - close[i]) / 3600.0))
        if direct:
            row["w_raw"] = _num(out[j])
        else:
            for k, v in enumerate(out[j]):
                row[f"u{k}"] = _num(v)
            row["act_idx"] = int(idx[j])
        rows.append(row)
    return rows, p_idx


def process_rep(ent, r, df15, now, paper_dir=None, deadline=None, key=None):
    """
    반복 하나: 월간 재학습 사슬을 now까지 이어 가며 새 판단봉을 판단합니다. 반환 dict(요약, alarms).
    run_replication 과 같은 순서: [T_k 학습] → [T_k 달 판단] → [마감 호출] → [T_(k+1) 학습] …
    """
    cfg = _cfg_from_json(ent["cfg"])
    name = ent["name"]
    vdir = os.path.join(paper_dir or PAPER_DIR, name)
    alarms = []
    stride = int(cfg.get("stride", 1))
    direct = _is_weights(cfg)
    n_ph = int(cfg.get("phases", N_PHASES))
    d0 = phases_for(df15, 1, key)[0]
    mask = walk.decision_mask(d0, stride)
    close = d0.ts + BAR_SEC
    data_end = int(df15["ts"].iloc[-1]) + Q15
    lo = _ts(ent["start"])
    Ms = list(months(OOS_START, _utc(min(data_end, int(now)))))       # 데이터가 T까지 다 있는 달만 학습
    st = _load_state(vdir, r)
    k = -1
    if st["month"] is not None:
        mm = [str(T.date()) for T in Ms]
        if st["month"] not in mm:
            raise PaperError(f"{name} rep{r}: 저장된 모델 달 {st['month']}이 지금 데이터 범위 밖입니다")
        k = mm.index(st["month"])
        logged = _log_months(vdir, r)
        if st.get("entry") and (not logged or logged[-1] < st["month"]):
            _append_log(vdir, r, st["entry"])                      # 상태 저장 직후 끊긴 경우 기록 복구
    ens, anchor = st["ens"], st["anchor"]
    header = dec_header(cfg)
    decs = _read_csv(_dec_path(vdir, r))
    last_dec = int(decs[-1]["ts_close"]) if decs else -1
    p_idx = int(decs[-1]["act_idx"]) if decs and not direct else 0
    new_rows = []
    lo_month = _utc(lo).replace(day=1)
    trained = 0
    while True:
        if k >= 0:
            Tk = Ms[k]
            Tn = int((Tk + pd.offsets.MonthBegin(1)).timestamp())
            if Tn > lo:
                rows, p_idx = _decide(cfg, ens, d0, mask, int(Tk.timestamp()), Tn, lo, last_dec, p_idx, int(now),
                                      str(Tk.date()))
                if rows:
                    new_rows += rows
                    last_dec = rows[-1]["ts_close"]
        if k + 1 >= len(Ms):
            break
        if deadline is not None and time.time() > deadline:
            alarms.append(dict(alarm="time_budget", variant=name, rep=r, next_month=str(Ms[k + 1].date())))
            break
        datas = phases_for(df15, n_ph, key)
        if k >= 0 and ens is not None:
            Tk = Ms[k]
            sel = _month_sel(d0, mask, int(Tk.timestamp()), int(Ms[k + 1].timestamp()))
            fin = _finalize(ens, d0, sel, direct)
            if fin is not None:
                _recheck(fin, sel, close, decs + new_rows, direct, alarms, name, r)
        T = Ms[k + 1]
        t0 = time.time()
        try:
            ens, anchor, entry = walk.monthly_update(cfg, datas, T, (int(r), T.year, T.month, 0), ens, anchor)
        except Exception as e:                                          # 경고만 남기고 이 반복은 다음 실행에서 다시
            import traceback
            alarms.append(dict(alarm="retrain_failed", variant=name, rep=r, month=str(T.date()),
                               err=str(e)[:300], tb=traceback.format_exc()[-800:]))
            break
        entry["secs"] = round(time.time() - t0, 2)
        ens = _slim(ens)
        k += 1
        _save_state(vdir, r, dict(month=str(T.date()), ens=ens, anchor=anchor, entry=entry),
                    archive=T >= lo_month)
        _append_log(vdir, r, entry)
        trained += 1
    if new_rows:
        _write_csv(_dec_path(vdir, r), header, decs + new_rows)
    exp = np.nonzero(mask & (close >= lo))[0]
    return dict(variant=name, rep=r, trained_now=trained, trained_through=str(Ms[k].date()) if k >= 0 else None,
                n_decisions=len(decs) + len(new_rows), new_decisions=len(new_rows),
                undecided=int(np.sum(close[exp] > last_dec)), alarms=alarms)


def _recheck(fin, sel, close, rows, direct, alarms, name, r):
    """달 마감 호출 값 == 달 중간에 기록한 판단 값 (float32 비트 단위). 다르면 경고 (판단은 이미 기록된 대로)"""
    by = {int(x["ts_close"]): x for x in rows}
    fin = np.asarray(fin, np.float32)
    bad = 0
    for j, i in enumerate(sel):
        x = by.get(int(close[i]))
        if x is None or not x["model_month"]:
            continue
        rec = np.array([_f(x["w_raw"])] if direct else
                       [_f(x[f"u{k}"]) for k in range(fin.shape[1])], np.float32)
        now_v = np.atleast_1d(fin[j])
        if not np.array_equal(rec, now_v, equal_nan=True):
            bad += 1
    if bad:
        alarms.append(dict(alarm="recheck_mismatch", variant=name, rep=r, n=bad,
                           note="달 중간 판단값과 달 마감 호출값이 다름 (죽은 판단봉 등으로 묶음 크기가 달라진 경우)"))


# ══════════ 장부 (evaluate.Window.run 과 같은 계산) ══════════
def window_range(d, lo, hi):
    """Window.__init__ 과 같은 (a, b)"""
    a, b = decision_range(d, lo, hi)
    while b > a and (d.ts[b] + BAR_SEC > hi or (b + 1 < d.T and d.ts[b + 1] > hi)):
        b -= 1
    return a, b


def window_run(d, lo, hi, tg, frac, cost=COST):
    """Window.run 과 같은 계산 (잠금 구간 감사 기록 없이). 반환 None(구간 없음) 또는 dict"""
    a, b = window_range(d, lo, hi)
    if b <= a:
        return None
    sim = simulate_weights(tg, d.o, d.c, cost, a, b) if frac else \
        simulate(tg, d.o, d.c, cost, a, b, forced_hold=d.forced_hold)
    days, eq = daily_marks(d.ts, sim["mark"], a, b, lo=lo, hi=hi)
    pos = sim["pos"][a:b]
    prev = np.concatenate([[0.0], pos[:-1]])
    g = d.o[a + 1:b + 1] / d.o[a:b]                       # 직전 체결 뒤 가격 변동으로 흘러간 비중
    den = prev * g + (1.0 - prev)
    w_pre = np.where(den > 0, prev * g / np.where(den > 0, den, 1.0), 0.0)
    traded = np.abs(pos - w_pre)
    return dict(days=days, eq=eq, r=eq[1:] / eq[:-1] - 1.0, pos=pos, w_pre=w_pre, traded=traded, a=a, b=b)


def rep_targets(d0, decs):
    """판단 기록 → phase 0 목표 배열 (판단봉 밖은 NaN). 봉 시각이 데이터와 안 맞으면 오류"""
    tg = np.full(d0.T, np.nan)
    close = d0.ts + BAR_SEC
    decs = [x for x in decs if int(x["ts_close"]) <= int(close[-1])]     # 더 이른 now로 보는 보고서
    ts = np.array([int(x["ts_close"]) for x in decs], dtype=np.int64)
    if len(ts):
        i = np.searchsorted(close, ts)
        if np.any(i >= d0.T) or np.any(close[np.minimum(i, d0.T - 1)] != ts):
            raise PaperError("판단 기록의 봉 시각이 데이터에 없습니다 (15분봉 파일이 바뀌었나?)")
        tg[i] = [_f(x["target"]) for x in decs]
    return tg


def rep_hi(d0, decs, lo, now, mask):
    """장부를 계산할 수 있는 끝(자정): 데이터·now·아직 판단 못 한 첫 판단봉 가운데 가장 이른 것"""
    close = d0.ts + BAR_SEC
    hi = (min(int(now), int(close[-1])) // DAY) * DAY
    done = set(int(x["ts_close"]) for x in decs)
    exp = close[mask & (close >= lo) & (close < hi)]
    miss = [c for c in exp if int(c) not in done]
    if miss:
        hi = (int(min(miss)) // DAY) * DAY
    return hi


def metrics(res):
    if res is None or len(res["r"]) == 0:
        return dict(days=0)
    s = S.summary(res["r"])
    years = len(res["pos"]) / (6 * 365.0)
    return dict(days=int(len(res["r"])), sharpe=s["sharpe"], cagr=s["cagr"], max_dd=s["max_dd"], vol=s["vol"],
                twm=s["twm"], exposure=float(res["pos"].mean()), turnover_per_year=float(res["traded"].sum() / years),
                n_trades=int(np.sum(res["traded"] > 1e-12)))


def bh_targets(d, lo, hi):
    """매수·보유 (evaluate.baseline_targets 의 B0: 구간 [a, b)에서 1)"""
    a, b = window_range(d, lo, hi)
    tg = np.full(d.T, np.nan)
    if b > a:
        tg[a:b] = 1.0
    return tg


def write_ledgers(ent, d0, now, paper_dir=None):
    """반복별 일별 자산·체결 + 매수·보유 장부를 판단 기록에서 다시 계산해 씀"""
    cfg = _cfg_from_json(ent["cfg"])
    vdir = os.path.join(paper_dir or PAPER_DIR, ent["name"])
    lo = _ts(ent["start"])
    mask = walk.decision_mask(d0, int(cfg.get("stride", 1)))
    frac = _is_frac(cfg)
    out, his = {}, []
    for r in range(ent["reps"]):
        decs = _read_csv(_dec_path(vdir, r))
        hi = rep_hi(d0, decs, lo, now, mask)
        his.append(hi)
        res = window_run(d0, lo, hi, rep_targets(d0, decs), frac) if hi > lo else None
        rows, fills = [], []
        if res is not None:
            rows = [dict(date=_date(t), equity=_num(e)) for t, e in zip(res["days"], res["eq"])]
            for j in np.nonzero(res["traded"] > 1e-12)[0]:
                t = res["a"] + j
                fills.append(dict(close_utc=_iso(d0.ts[t] + BAR_SEC), fill_utc=_iso(d0.ts[t + 1]),
                                  fill_open=_num(d0.o[t + 1]), w_before=_num(res["w_pre"][j]),
                                  w_after=_num(res["pos"][j])))
        _write_csv(os.path.join(vdir, f"ledger_rep{r:02d}.csv"), ["date", "equity"], rows)
        _write_csv(os.path.join(vdir, f"fills_rep{r:02d}.csv"),
                   ["close_utc", "fill_utc", "fill_open", "w_before", "w_after"], fills)
        out[r] = dict(ledger_through=_date(hi) if res is not None else None,
                      equity=float(res["eq"][-1]) if res is not None else None)
    hi = min(his) if his else lo
    bh = window_run(d0, lo, hi, bh_targets(d0, lo, hi), False) if hi > lo else None
    _write_csv(os.path.join(vdir, "bh_ledger.csv"), ["date", "equity"],
               [dict(date=_date(t), equity=_num(e)) for t, e in zip(bh["days"], bh["eq"])] if bh is not None else [])
    return out


# ══════════ step ══════════
def _rep_job(args):
    """--workers 용 (프로세스마다 데이터를 다시 읽음)"""
    ent, r, now, paper_dir, base_path, deadline = args
    df = truncate(load_15m_all(paper_dir, base_path), now)
    return process_rep(ent, r, df, now, paper_dir, deadline)


def step(now=None, offline=False, paper_dir=None, base_path=BASE_15M, df15=None, max_minutes=None, workers=1,
         fetch=None, verbose=False):
    """
    모의매매 한 걸음. now(초 또는 ISO, 기본 지금) 시각에 있었을 데이터만으로 처리합니다.
    df15: 테스트용 — 15분봉 전체를 직접 넘김 (연구용 파일·덧붙인 파일·거래소 조회 대신)
    반환 dict (status.json 과 같은 내용)
    """
    paper_dir = paper_dir or PAPER_DIR
    now = int(time.time()) if now is None else (int(now) if isinstance(now, (int, float, np.integer)) else _ts(now))
    t_start = time.time()
    deadline = t_start + 60.0 * max_minutes if max_minutes else None
    status_p = os.path.join(paper_dir, "status.json")
    prev = _read_json(status_p, {})
    if prev.get("now_ts") is not None and now < int(prev["now_ts"]):
        raise PaperError(f"now({_iso(now)})가 지난 실행({prev.get('now')})보다 이릅니다")
    alarms = []
    ext_info = None
    if not offline and df15 is None:
        try:
            ext_info = extend_cache(paper_dir, base_path, now, fetch=fetch)
        except Exception as e:                                         # 조회 실패: 있는 데이터로 계속
            alarms.append(dict(alarm="fetch_failed", err=str(e)[:300]))
    df = truncate(df15 if df15 is not None else load_15m_all(paper_dir, base_path), now)
    data_end = int(df["ts"].iloc[-1]) + Q15
    if now - data_end > 8 * 3600:
        alarms.append(dict(alarm="stale_data", data_end=_iso(data_end), age_h=round((now - data_end) / 3600, 1)))
    key = _df_key(df)
    reg = load_registry(paper_dir)
    out = dict(now=_iso(now), now_ts=now, data_end=_iso(data_end), ext=ext_info, variants={}, alarms=alarms)
    jobs = []
    for name, ent in reg["variants"].items():
        bad, warn = check_entry(ent, base_path, check_base=df15 is None)
        for w in warn:
            alarms.append(dict(alarm="changed_since_registration", variant=name, what=w))
        if bad:
            alarms.append(dict(alarm="frozen_mismatch", variant=name, what=bad,
                               note="고정한 후보와 코드·설정·데이터가 다릅니다 — 이 변형은 처리하지 않음"))
            out["variants"][name] = dict(skipped=bad)
            continue
        jobs += [(ent, r) for r in range(ent["reps"])]
    if workers > 1 and len(jobs) > 1 and df15 is None:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(workers) as pool:
            res = pool.map(_rep_job, [(e, r, now, paper_dir, base_path, deadline) for e, r in jobs])
    else:
        res = [process_rep(e, r, df, now, paper_dir, deadline, key) for e, r in jobs]
    d0 = phases_for(df, 1, key)[0]
    for (ent, r), x in zip(jobs, res):
        alarms += x.pop("alarms")
        out["variants"].setdefault(ent["name"], dict(reps={}))["reps"][r] = x
        if verbose:
            print(f"  {ent['name']} rep{r}: 학습 {x['trained_now']}달 (→{x['trained_through']}), "
                  f"새 판단 {x['new_decisions']}개, 미판단 {x['undecided']}", flush=True)
    for name, ent in reg["variants"].items():
        if "skipped" in out["variants"].get(name, {}):
            continue
        led = write_ledgers(ent, d0, now, paper_dir)
        for r, v in led.items():
            out["variants"][name]["reps"][r].update(v)
    # status.json 에는 저장된 상태로 정해지는 값만 (같은 now로 다시 돌려도 바이트 단위로 같게)
    per_run = ("trained_now", "new_decisions")
    ep = ext_path(paper_dir)
    status = dict(now=out["now"], now_ts=now, data_end=out["data_end"],
                  ext_until=(_iso(int(pd.read_csv(ep, usecols=["ts"])["ts"].iloc[-1]) + Q15)
                             if os.path.exists(ep) else None),
                  variants={n: (v if "reps" not in v else
                                dict(reps={r: {k: x for k, x in rv.items() if k not in per_run}
                                           for r, rv in v["reps"].items()}))
                            for n, v in out["variants"].items()})
    _write_json(status_p, status)
    _log_alarms(paper_dir, now, alarms)
    return out


def _log_alarms(paper_dir, now, alarms):
    """경고는 alarms.jsonl 에 덧붙임 (같은 now의 같은 경고는 한 번만)"""
    if not alarms:
        return
    p = os.path.join(paper_dir, "alarms.jsonl")
    seen = set()
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            seen = {l.strip() for l in f if l.strip()}
    new = []
    for al in alarms:
        line = json.dumps(dict(now=_iso(now), **{k: v for k, v in al.items() if k != "tb"}),
                          ensure_ascii=False, sort_keys=True, default=_json_default)
        if line not in seen:
            seen.add(line)
            new.append(line)
    if new:
        os.makedirs(paper_dir, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write("\n".join(new) + "\n")


# ══════════ 보고서 ══════════
def _lower_median(xs):
    from ..evaluate import lower_median
    return lower_median(xs)


def evaluate_variant(ent, d0, now, paper_dir=None):
    """
    시작일부터: 반복별 성과·매수·보유·헤드라인(ΔSharpe 작은 쪽 중앙값)·점검일 판정.
    모든 반복을 같은 구간(판단이 다 끝난 가장 이른 끝)에서 잽니다.
    """
    cfg = _cfg_from_json(ent["cfg"])
    vdir = os.path.join(paper_dir or PAPER_DIR, ent["name"])
    lo = _ts(ent["start"])
    mask = walk.decision_mask(d0, int(cfg.get("stride", 1)))
    frac = _is_frac(cfg)
    decs = [_read_csv(_dec_path(vdir, r)) for r in range(ent["reps"])]
    tgs = [rep_targets(d0, x) for x in decs]
    hi_all = min(rep_hi(d0, x, lo, now, mask) for x in decs)

    def block(hi):
        if hi <= lo:
            return None
        reps = [metrics(window_run(d0, lo, hi, tg, frac)) for tg in tgs]
        bh = metrics(window_run(d0, lo, hi, bh_targets(d0, lo, hi), False))
        if not bh.get("days"):
            return None
        ds = [x["sharpe"] - bh["sharpe"] for x in reps]
        lm = _lower_median(ds)
        h = reps[lm]
        return dict(start=_date(lo), end=_date(hi), reps=reps, bh=bh, d_sharpe=ds, headline_rep=int(lm),
                    headline=dict(sharpe=h["sharpe"], max_dd=h["max_dd"], cagr=h["cagr"], d_sharpe=ds[lm],
                                  sharpe_gt_bh=bool(h["sharpe"] > bh["sharpe"]),
                                  mdd_shallower=bool(h["max_dd"] > bh["max_dd"])))

    since = block(hi_all)
    cps = {}
    final = ent["checkpoints"][-1]
    verdict = "pending"
    for c in ent["checkpoints"]:
        c_ts = _ts(c)
        if hi_all < c_ts or now < c_ts:
            cps[c] = dict(status="not_reached")
            continue
        b = block(c_ts)
        if c == final:
            ok = b is not None and b["headline"]["sharpe_gt_bh"] and b["headline"]["mdd_shallower"]
            verdict = "pass" if ok else "fail"
            cps[c] = dict(status="final", verdict=verdict, **(b or {}))
        else:
            cps[c] = dict(status="interim", note="중간 점검 — 판정하지 않음", **(b or {}))
    return dict(start=ent["start"], reps=ent["reps"], cfg_hash=ent["cfg_hash"], code_hash=ent["code_hash"],
                since_start=since, checkpoints=cps, final_checkpoint=final, verdict=verdict,
                verdict_rule=ent["verdict_rule"])


def report(now=None, paper_dir=None, base_path=BASE_15M, df15=None, write=True):
    paper_dir = paper_dir or PAPER_DIR
    now = int(time.time()) if now is None else (int(now) if isinstance(now, (int, float, np.integer)) else _ts(now))
    df = truncate(df15 if df15 is not None else load_15m_all(paper_dir, base_path), now)
    d0 = phases_for(df, 1)[0]
    reg = load_registry(paper_dir)
    out = dict(now=_iso(now), data_end=_iso(int(df["ts"].iloc[-1]) + Q15), variants={})
    for name, ent in reg["variants"].items():
        out["variants"][name] = evaluate_variant(ent, d0, now, paper_dir)
    if write:
        from ..evaluate import _clean
        _write_json(os.path.join(paper_dir, "report.json"), _clean(out))
    return out


# ══════════ CLI ══════════
def main(argv=None):
    ap = argparse.ArgumentParser(description="BTC 3단계 — 앞으로의 모의매매")
    ap.add_argument("--dir", default=PAPER_DIR, help="모의매매 폴더 (기본 data/btc/paper)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("register", help="후보 고정 등록")
    g.add_argument("variant")
    g.add_argument("--reps", type=int, default=5)
    g.add_argument("--start", default=START)
    s = sub.add_parser("step", help="데이터 덧붙이기 → 재학습 사슬 → 판단 → 장부")
    s.add_argument("--now", default=None, help="ISO 시각 (기본: 지금)")
    s.add_argument("--offline", action="store_true", help="거래소 조회 없이 있는 데이터만")
    s.add_argument("--max-minutes", type=float, default=None,
                   help="이 시간이 지나면 새 달 학습을 시작하지 않음 (남은 달은 다음 실행에서 같은 결과로 이어 감)")
    s.add_argument("--workers", type=int, default=1, help="반복을 프로세스로 나눠 돌림 (결과는 같음)")
    p = sub.add_parser("report", help="시작일 이후 성과·점검일 판정 → report.json")
    p.add_argument("--now", default=None)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "register":
            ent = register(a.variant, reps=a.reps, start=a.start, paper_dir=a.dir)
            print(json.dumps({k: ent[k] for k in ("name", "cfg_hash", "code_hash", "algo_hash", "reps", "start",
                                                  "checkpoints")}, ensure_ascii=False))
        elif a.cmd == "step":
            out = step(a.now, offline=a.offline, paper_dir=a.dir, max_minutes=a.max_minutes, workers=a.workers,
                       verbose=True)
            print(f"now {out['now']}  데이터 끝 {out['data_end']}  덧붙임 {out['ext']}")
            for al in out["alarms"]:
                print("경고:", json.dumps(al, ensure_ascii=False, default=str)[:400])
            if any(al["alarm"] == "frozen_mismatch" for al in out["alarms"]):
                return 2
        else:
            out = report(a.now, paper_dir=a.dir)
            for n, v in out["variants"].items():
                ss = v["since_start"]
                if ss is None:
                    print(f"{n}: 아직 장부 없음 (판정 {v['verdict']})")
                    continue
                print(f"{n}: {ss['start']}~{ss['end']} {ss['bh']['days']}일  헤드라인 rep{ss['headline_rep']} "
                      f"샤프 {ss['headline']['sharpe']:.2f} (매수·보유 {ss['bh']['sharpe']:.2f})  "
                      f"MDD {ss['headline']['max_dd'] * 100:.1f}% (매수·보유 {ss['bh']['max_dd'] * 100:.1f}%)  "
                      f"판정 {v['verdict']}")
    except PaperError as e:
        print("거부:", e)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
