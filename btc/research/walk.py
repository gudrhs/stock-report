# -*- coding: utf-8 -*-
"""
연구용 워크포워드 — 핵심 walkforward.py와 같은 흐름(매월 1일 재학습, 1월은 처음부터, 나머지는
L2-SP 이어학습, 결정론적 시드)을 일반화 에이전트(KTrainer)로 돌립니다.

  · 판단 주기: stride=6 이면 하루 한 번 (phase 0은 00:00 UTC 마감 봉에서만 판단)
  · 점검(gate): 'p0'(원래 규칙) | 'switch'(노출도 상한 없이 전환 횟수·유한성만) | 'none'(유한성만)
  · 결과: data/btc/research_runs/<이름>/repNN.npz  (phase 0 판단봉의 평균 U, 판단비용별)
  · 시험 기록: data/btc/research_runs/trials.jsonl  (기존 24개에 이어서 N을 셈)

  python -m btc.research.walk <변형이름> --reps 5
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import hashlib
import json
import multiprocessing as mp
import time

import numpy as np
import pandas as pd

from .. import config as C
from ..env import BAR_SEC
from ..features import WARMUP
from ..walkforward import load_phases, months, decision_range, FIRST_TRAIN, OOS_END, TRIALS as CORE_TRIALS
from ..data import DATA_DIR
from .rl import KTrainer, KEnsemble, k_policy, trend_filter, allowed_matrix
from ..nn import StackedMLP

RUNS = os.path.join(DATA_DIR, "research_runs")
TRIALS = os.path.join(RUNS, "trials.jsonl")
C_DECS = (0.001, 0.0015, 0.002, 0.003)          # 판단용 비용 (업비트·바이낸스 × 배수 1·2)


def decision_mask(d, stride):
    """판단하는 봉: stride=1이면 전부, 6이면 phase 0 기준 00:00 UTC에 마감하는 봉 (다른 phase는 그날 첫 마감 봉)"""
    if stride == 1:
        return np.ones(d.T, bool)
    close = d.ts + BAR_SEC
    off = int(d.ts[0] % BAR_SEC)                            # phase k의 15분 단위 밀림 (phase 0 = 0 → 00:00 UTC)
    return (close % (stride * BAR_SEC)) == off


# ══════════ 점검 ══════════
_B2 = {}


def b2_of(d):
    if id(d) not in _B2:
        _B2[id(d)] = (d, trend_filter(d.c))
    return _B2[id(d)][1]


def _positions(ens, d, a, b, cost, stride, veto=False):
    mask = decision_mask(d, stride)[a:b]
    U, umax = ens.values(d.X[a:b][mask], cost)
    allowed = allowed_matrix(b2_of(d)[a:b][mask], ens.acts) if veto else None
    idx = k_policy(U, ens.acts, cost, d.forced_hold[a:b][mask], allowed=allowed)
    return ens.acts[idx], umax, U


def gate_cold(cfg, ens, d0, T_k, cost=0.003):
    a, b = decision_range(d0, T_k - 2 * 365 * 86400, T_k)
    if b - a < 2 * 365 * 6 * 0.9:
        return False, dict(reason="short_window")
    w, umax, U = _positions(ens, d0, a, b, cost, cfg.get("stride", 1), cfg.get("veto_b2", False))
    finite = bool(np.all(np.isfinite(U))) and float(umax.max()) < 200
    years = (b - a) / (6 * 365.0)
    per_year = float(np.abs(np.diff(np.concatenate([[0.0], w]))).sum() / max(years, 1e-9))   # 바꾼 비중 합 / 연수
    exp_ = float(w.mean())
    kind = cfg.get("gate", "p0")
    if kind == "none":
        ok = finite
    elif kind == "switch":
        ok = finite and per_year <= 60
    else:
        ok = finite and per_year <= 60 and 0.10 <= exp_ <= 0.95
    return bool(ok), dict(exposure=exp_, switches_per_year=per_year, umax=float(umax.max()))


def gate_finetune(cfg, new, old, d0, T_k, cost=0.003):
    if cfg.get("gate", "p0") == "none":
        return True, {}
    b = int(np.searchsorted(d0.ts + BAR_SEC, T_k, side="left"))
    a = b - 180
    if a < 0 or T_k - (d0.ts[b - 1] + BAR_SEC) > 7 * 86400:
        return False, dict(reason="short_window")
    wn, un, Un = _positions(new, d0, a, b, cost, cfg.get("stride", 1), cfg.get("veto_b2", False))
    wo, _, _ = _positions(old, d0, a, b, cost, cfg.get("stride", 1), cfg.get("veto_b2", False))
    if not np.all(np.isfinite(Un)) or float(un.max()) >= 200:
        return False, dict(reason="nonfinite")
    dis = float((np.abs(wn - wo) > 1e-9).mean())
    return dis <= 0.40, dict(disagree=dis)


class _DirectModel:
    """샤프 직접 최적화 정책(비교용)을 워크포워드에서 앙상블처럼 다루기 위한 얇은 포장"""

    def __init__(self, tr):
        from ..nn import StackedMLP as _SM
        self.tr = tr
        self.net = _SM.__new__(_SM)
        self.net.__dict__.update(tr.net.__dict__)
        self.net.params = tr.net.copy_params()
        self.net._cache = None
        self.fi = tr.fi

    def weights(self, X):
        z = self.net.forward(X[:, self.fi].astype(np.float32), cache=False)[..., 0]
        return (1.0 / (1.0 + np.exp(-z))).mean(axis=0)

    def state(self):
        return self.net.state("p")


def monthly_update_direct(cfg, datas, T_k, seed, ens, anchor):
    from .direct import DirectTrainer
    Tk = int(T_k.timestamp())
    entry = dict(month=str(T_k.date()))
    tr = DirectTrainer(cfg, seed)
    n = tr.make_pool(datas, Tk, FIRST_TRAIN, WARMUP)
    if T_k.month == 1 or ens is None:
        tr.init_fresh()
        losses = tr.fit_cold()
        entry.update(kind="cold", pool=n, loss=float(np.mean(losses[-200:])), accepted=True)
        new = _DirectModel(tr)
        return new, new.net.copy_params(), entry
    tr.init_from(ens.net.params)
    losses = tr.fit_finetune(anchor)
    entry.update(kind="finetune", loss=float(np.mean(losses[-50:])), accepted=True)
    return _DirectModel(tr), anchor, entry


def monthly_update(cfg, datas, T_k, seed, ens, anchor):
    if cfg.get("algo") == "direct":
        return monthly_update_direct(cfg, datas, T_k, seed, ens, anchor)
    Tk = int(T_k.timestamp())
    entry = dict(month=str(T_k.date()))
    tr = KTrainer(cfg, seed)
    n_pool = tr.make_pool(datas, Tk, FIRST_TRAIN, WARMUP)
    if T_k.month == 1 or ens is None:
        tr.init_fresh()
        losses = tr.fit_cold()
        new = tr.ensemble()
        ok, info = gate_cold(cfg, new, datas[0], Tk)
        entry.update(kind="cold", pool=n_pool, td=float(np.mean(losses[-200:])), accepted=ok, **info)
        if ok:
            ens, anchor = new, new.net.copy_params()
        elif ens is None:
            entry["note"] = "gate_failed_no_model_hold_cash"
    elif cfg["fine_tune"]:
        tr.init_from(ens.net.params)
        losses = tr.fit_finetune(anchor)
        new = tr.ensemble()
        ok, info = gate_finetune(cfg, new, ens, datas[0], Tk)
        entry.update(kind="finetune", td=float(np.mean(losses[-50:])), accepted=ok, **info)
        if ok:
            ens = new
    return ens, anchor, entry


def run_replication(cfg, r, datas=None, end=OOS_END):
    datas = datas or load_phases()
    d0 = datas[0]
    K = len(cfg.get("acts", (0.0, 1.0)))
    direct = cfg.get("algo") == "direct"
    U = {cd: np.full((d0.T, K), np.nan, np.float32) for cd in C_DECS} if not direct else \
        {0.0: np.full((d0.T, 1), np.nan, np.float32)}
    log, ens, anchor = [], None, None
    Ms = [m for m in months() if m < end]
    mask = decision_mask(d0, cfg.get("stride", 1))
    for i, T_k in enumerate(Ms):
        Tk = int(T_k.timestamp())
        nxt = int(Ms[i + 1].timestamp()) if i + 1 < len(Ms) else int(end.timestamp())
        t0 = time.time()
        ens, anchor, entry = monthly_update(cfg, datas, T_k, (int(r), T_k.year, T_k.month, 0), ens, anchor)
        entry["secs"] = round(time.time() - t0, 2)
        log.append(entry)
        if ens is None:
            continue
        a, b = decision_range(d0, Tk, nxt)
        sel = np.arange(a, b)[mask[a:b]]
        if len(sel) == 0:
            continue
        if direct:
            U[0.0][sel, 0] = ens.weights(d0.X[sel])
            continue
        for cd in C_DECS:
            u, _ = ens.values(d0.X[sel], cd)
            U[cd][sel] = u
    st = ens.state() if ens is not None else {}
    return dict(U=U, log=log, last_state=st)


# ══════════ 저장·해시·시험 기록 ══════════
def code_hash():
    here = os.path.dirname(os.path.abspath(__file__))
    core = os.path.dirname(here)
    h = hashlib.sha1()
    for f in [os.path.join(here, "rl.py"), os.path.join(here, "walk.py")] + \
             [os.path.join(core, n) for n in ("nn.py", "env.py", "data.py", "features.py", "agent.py")]:
        with open(f, "rb") as fp:
            h.update(fp.read())
    return h.hexdigest()[:12]


def cfg_hash(cfg):
    keys = sorted(k for k in cfg if k != "name")
    return hashlib.sha1(json.dumps({k: cfg[k] for k in keys}, sort_keys=True, default=list).encode()).hexdigest()[:12]


def run_path(name, r):
    return os.path.join(RUNS, name, f"rep{r:02d}.npz")


def save_run(cfg, r, res, ch=None):
    path = run_path(cfg["name"], r)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    flat = {f"U_{cd:.4f}": v for cd, v in res["U"].items()}
    for k, v in res["last_state"].items():
        flat[f"s_{k}"] = v
    flat["log"] = np.frombuffer(json.dumps(res["log"]).encode(), dtype=np.uint8)
    flat["meta"] = np.frombuffer(json.dumps(dict(cfg=cfg, cfg_hash=cfg_hash(cfg), code_hash=ch or code_hash()),
                                            default=list).encode(), dtype=np.uint8)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **flat)
    os.replace(tmp, path)


def load_run(name, r, check=False):
    """
    연구 단계에서는 코드에 기능을 '추가'만 하는 경우가 많아, 해시가 달라도 막지 않고 기록(meta.code_hash)만 합니다.
    기존 변형의 동작을 바꾸는 수정(버그 수정)을 하면 그 변형은 반드시 다시 돌립니다. check=True면 엄격 모드.
    """
    z = np.load(run_path(name, r))
    meta = json.loads(bytes(z["meta"]).decode())
    if check and meta["code_hash"] != code_hash():
        raise RuntimeError(f"{name} rep{r}: 코드가 바뀌었습니다 — 다시 실행하세요")
    U = {float(k[2:]): z[k] for k in z.files if k.startswith("U_")}
    return dict(U=U, log=json.loads(bytes(z["log"]).decode()), meta=meta)


def log_trial(cfg, note=""):
    os.makedirs(RUNS, exist_ok=True)
    seen = set()
    if os.path.exists(TRIALS):
        with open(TRIALS, encoding="utf-8") as f:
            seen = {json.loads(l)["cfg_hash"] for l in f if l.strip()}
    h = cfg_hash(cfg)
    if h in seen:
        return
    with open(TRIALS, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(name=cfg["name"], cfg_hash=h, note=note,
                                time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))) + "\n")


def n_trials_total():
    """본 연구 24개 + 연구 단계에서 시험한 설정 수"""
    from ..walkforward import n_trials
    n = n_trials()
    if os.path.exists(TRIALS):
        with open(TRIALS, encoding="utf-8") as f:
            n += len({json.loads(l)["cfg_hash"] for l in f if l.strip()})
    return n


def _job(args):
    cfg, r = args
    path = run_path(cfg["name"], r)
    if os.path.exists(path):
        return cfg["name"], r, "cached"
    t = time.time()
    ch = code_hash()                                  # 실행을 시작한 시점의 코드 (출처 기록)
    try:
        res = run_replication(cfg, r)
    except Exception:
        import traceback
        return cfg["name"], r, "FAILED\n" + traceback.format_exc()
    save_run(cfg, r, res, ch)
    return cfg["name"], r, round(time.time() - t, 1)


def run_many(cfgs, reps, workers=4):
    load_phases()
    for c in cfgs:
        if not c["name"].startswith("R0_"):          # 재현 확인용은 시험으로 세지 않음
            log_trial(c)
    jobs = [(c, r) for c in cfgs for r in range(reps)]
    with mp.get_context("spawn").Pool(workers) as pool:
        for out in pool.imap_unordered(_job, jobs):
            print("  완료", out, flush=True)


def main():
    from .variants import VARIANTS
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    run_many([VARIANTS[n] for n in a.names], a.reps, a.workers)


if __name__ == "__main__":
    main()
