# -*- coding: utf-8 -*-
"""
워크포워드 — 실제로 운용했다면 어땠을지를 시간 순서대로 재현합니다.

  매월 1일 00:00 UTC (2017-01 ~ 2026-09, 117번)
    · 1월: 새 가중치로 처음부터 학습 (2014-01부터 그 시점까지 전부, 최근 2년에 가중)
    · 나머지 달: 지난달 모델에서 300스텝만 이어 학습 (1월 모델에서 멀어지지 않게 L2-SP)
    · 학습 데이터는 그 시점보다 43봉(약 7일) 앞에서 끊습니다 — 보조 목표가 앞으로 43봉을 보기 때문
    · 그 달 동안은 이 모델로만 판단합니다 (달 중간에 미래 정보로 고치지 않음)

  안전장치(백테스트와 실운용에서 똑같이):
    · 1월 재학습 점검: 최근 2년을 돌려 보고 연 60회 넘게 갈아타거나 보유비중이 10~95% 밖이면
      새 모델을 버리고 이전 모델 유지
    · 월간 이어학습 거부: 최근 180봉에서 이전 모델과 판단이 40% 넘게 다르면 이전 모델 유지

  python -m btc.walkforward budget          ← 학습 스텝 수 결정 (2014~2016, 손익 안 봄)
  python -m btc.walkforward run P0 --reps 10
  python -m btc.walkforward grid --reps 5
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")      # 반복 실험을 프로세스로 병렬 실행하므로 각자 1스레드

import argparse
import json
import math
import multiprocessing as mp
import time

import numpy as np
import pandas as pd

from . import config as C
from .agent import Trainer, Ensemble, policy_from_delta, threshold
from .data import DATA_DIR, load_15m, bars_4h, N_PHASES
from .env import PhaseData, simulate, BAR_SEC
from .features import compute, WARMUP

CACHE_DIR = os.path.join(DATA_DIR, "cache")
RUNS_DIR = os.path.join(DATA_DIR, "runs")
TRIALS = os.path.join(RUNS_DIR, "trials.jsonl")

FIRST_TRAIN = int(pd.Timestamp("2014-01-01", tz="UTC").timestamp())
OOS_START = pd.Timestamp("2017-01-01", tz="UTC")
LAST_RETRAIN = pd.Timestamp("2026-09-01", tz="UTC")
OOS_END = pd.Timestamp("2026-09-25", tz="UTC")      # 마지막 결정봉 종가 < 이 시각
N2_END = pd.Timestamp("2025-01-01", tz="UTC")        # 무작위 대조(N2)는 2017~2024만


# ══════════ 데이터 준비 (16 phase) ══════════
def load_phases(n=N_PHASES):
    """phase별 PhaseData 목록. 특징 계산은 캐시해 둡니다 (data/btc/cache, git 제외)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    src = os.path.join(DATA_DIR, "btcusd_15m.csv.gz")
    # 캐시 키: 데이터 파일 + 봉·지표 계산 코드. 어느 하나라도 바뀌면 새로 계산합니다.
    import hashlib
    from . import data as _d, features as _f, env as _e
    h = hashlib.sha1()
    with open(src, "rb") as fp:
        h.update(fp.read())
    for mod in (_d, _f, _e):
        with open(mod.__file__, "rb") as fp:
            h.update(fp.read())
    stamp = h.hexdigest()[:16]
    path = os.path.join(CACHE_DIR, f"phases_{stamp}.npz")
    df15 = None
    out = []
    if os.path.exists(path):
        z = np.load(path)
        for k in range(n):
            bars = pd.DataFrame({c: z[f"{k}_{c}"] for c in
                                 ("ts", "open", "high", "low", "close", "volume", "gap_before", "forced_hold")})
            out.append(PhaseData(bars, z[f"{k}_X"], z[f"{k}_sig"]))
        return out
    df15 = load_15m()
    save = {}
    for k in range(N_PHASES):
        b = bars_4h(df15, k)
        X, sig = compute(b)
        for c in ("ts", "open", "high", "low", "close", "volume", "gap_before", "forced_hold"):
            save[f"{k}_{c}"] = b[c].to_numpy()
        save[f"{k}_X"], save[f"{k}_sig"] = X, sig
    tmp = f"{path}.{os.getpid()}.tmp.npz"            # 여러 프로세스가 동시에 만들어도 깨지지 않게
    np.savez(tmp, **save)
    os.replace(tmp, path)
    return load_phases(n)


def months(start=OOS_START, end=LAST_RETRAIN):
    return list(pd.date_range(start, end, freq="MS", tz="UTC"))


def decision_range(d, lo_ts, hi_ts):
    """종가 시각이 [lo, hi)인 결정봉 번호 범위 (다음 봉 시가가 있어야 함)"""
    close = d.ts + BAR_SEC
    a = int(np.searchsorted(close, lo_ts, side="left"))
    b = int(np.searchsorted(close, hi_ts, side="left"))
    return a, min(b, d.T - 1)


# ══════════ 점검 ══════════
def sanity_ok(ens, d0, T_k, cost=0.003):
    """1월 재학습 점검 — 최근 2년 phase 0에서 탐욕 정책을 돌려 봄 (손익은 안 봄)"""
    a, b = decision_range(d0, T_k - 2 * 365 * 86400, T_k)
    if b - a < 2 * 365 * 6 * 0.9:                  # 최근 2년 데이터가 없으면 점검 불가 — 조용히 넘기지 않음
        return False, dict(reason="short_window", bars=int(b - a), umax=float("nan"))
    delta, um = ens.delta(d0.X[a:b], cost)
    umax = float(um.max())
    if not np.all(np.isfinite(delta)) or umax >= 200:
        return False, dict(reason="nonfinite_or_big", umax=umax)
    pos = policy_from_delta(delta, cost, d0.forced_hold[a:b])
    years = (b - a) / (6 * 365.0)
    per_year = float(np.abs(np.diff(np.concatenate([[0], pos]))).sum() / max(years, 1e-9))
    exp_ = float(pos.mean())
    ok = per_year <= 60 and 0.10 <= exp_ <= 0.95
    return bool(ok), dict(switches_per_year=per_year, exposure=exp_, umax=umax)


def finetune_ok(new, old, d0, T_k, cost=0.003):
    """월간 이어학습 거부 — 최근 180봉에서 판단이 40% 넘게 다르면 거부"""
    a, b = decision_range(d0, T_k - 180 * BAR_SEC - 1, T_k)
    a = max(a, b - 180)
    if b - a < 180:                                  # 최근 180봉이 없으면(데이터가 끊김) 명시적으로 거부
        return False, dict(reason="short_window", bars=int(b - a), umax=float("nan"))
    dn, un = new.delta(d0.X[a:b], cost)
    un = float(un.max())
    do, _ = old.delta(d0.X[a:b], cost)
    if not np.all(np.isfinite(dn)) or un >= 200:
        return False, dict(reason="nonfinite_or_big", umax=un)
    pn = policy_from_delta(dn, cost, d0.forced_hold[a:b])
    po = policy_from_delta(do, cost, d0.forced_hold[a:b])
    dis = float((pn != po).mean())
    return dis <= 0.40, dict(disagree=dis, umax=un)


# ══════════ 매월 갱신 (백테스트·실운용 공용) ══════════
def monthly_update(cfg, datas, T_k, seed, ens, anchor):
    """
    T_k(매월 1일 00:00 UTC) 시점의 모델 갱신 한 번.
      1월이거나 모델이 없으면: 새 가중치로 처음부터 학습 → 점검 통과 시 채택, 1월 모델을 L2-SP 기준으로
      그 밖의 달: 이전 모델에서 이어 학습 → 거부 점검 통과 시 채택
    반환 (채택된 앙상블, 기준 가중치, 기록)
    """
    Tk = int(T_k.timestamp())
    entry = dict(month=str(T_k.date()))
    if T_k.month == 1 or ens is None:
        tr = Trainer(cfg, seed)
        n_pool = tr.make_pool(datas, Tk, FIRST_TRAIN, WARMUP)
        tr.init_fresh()
        losses = tr.fit_cold()
        new = tr.ensemble()
        ok, info = sanity_ok(new, datas[0], Tk)
        entry.update(kind="cold", pool=n_pool, td=float(np.mean(losses[-200:])), accepted=ok, **info)
        if ok:
            ens = new
            anchor = new.net.copy_params()
        elif ens is None:
            # 사전 등록 규칙: 점검을 통과한 모델이 아직 하나도 없으면 매매하지 않고(현금 유지)
            # 다음 달에 다시 처음부터 학습합니다. 점검에 떨어진 모델로는 절대 매매하지 않습니다.
            entry["note"] = "gate_failed_no_model_hold_cash"
    elif cfg["fine_tune"]:
        tr = Trainer(cfg, seed)
        n_pool = tr.make_pool(datas, Tk, FIRST_TRAIN, WARMUP)
        tr.init_from(ens.net.params)
        losses = tr.fit_finetune(anchor)
        new = tr.ensemble()
        ok, info = finetune_ok(new, ens, datas[0], Tk)
        entry.update(kind="finetune", pool=n_pool, td=float(np.mean(losses[-50:])), accepted=ok, **info)
        if ok:
            ens = new
    else:
        entry.update(kind="frozen")
    return ens, anchor, entry


# ══════════ 한 번의 워크포워드 ══════════
def run_replication(cfg, r, eval_phases=None, c_dec_list=None, end=OOS_END, datas=None, verbose=False):
    """
    반복 r 하나. 매월 학습 → 그 달 결정봉의 Δ(=평균 U1−U0)를 여러 판단비용으로 계산해 저장.
    반환 dict: delta[phase][c_dec] = (T,) 배열 (그 phase의 결정봉 위치에만 값, 나머지 NaN), log
    """
    datas = datas or load_phases()
    eval_phases = list(range(N_PHASES)) if eval_phases is None else eval_phases
    c_dec_list = c_dec_list or C.all_c_dec()
    prim = sorted({round(C.c_dec(C.PRIMARY_COST, 1.0), 6), round(C.c_dec(C.PRIMARY_COST, 2.0), 6)})
    delta = {k: {cd: np.full(datas[k].T, np.nan, np.float32)
                 for cd in (c_dec_list if k == 0 else prim)} for k in eval_phases}
    umax = {k: np.full(datas[k].T, np.nan, np.float32) for k in eval_phases}
    log = []
    ens, anchor = None, None
    Ms = [m for m in months() if m < end]
    for i, T_k in enumerate(Ms):
        Tk = int(T_k.timestamp())
        nxt = int(Ms[i + 1].timestamp()) if i + 1 < len(Ms) else int(end.timestamp())
        seed = (int(r), T_k.year, T_k.month, 7 if cfg["permute"] else 0)
        t0 = time.time()
        ens, anchor, entry = monthly_update(cfg, datas, T_k, seed, ens, anchor)
        entry["secs"] = round(time.time() - t0, 2)
        if ens is None:                              # 아직 채택된 모델 없음 → 이달 Δ는 NaN (판단 보류 = 현금)
            log.append(entry)
            continue
        # 이 모델이 담당하는 결정봉: 종가 ∈ [T_k, 다음 재학습)
        for k in eval_phases:
            d = datas[k]
            a, b = decision_range(d, Tk, nxt)
            if b <= a:
                continue
            if cfg["online"] and k == 0:
                info = online_month(cfg, ens, datas, Tk, a, b, delta[0], seed)
                entry["online"] = info
                continue
            for cd in delta[k]:
                dl, um = ens.delta(d.X[a:b], cd)
                delta[k][cd][a:b] = dl
                umax[k][a:b] = np.fmax(umax[k][a:b], um)
        log.append(entry)
        if verbose:
            print(f"  r{r} {entry['month']} {entry.get('kind')} acc={entry.get('accepted')} "
                  f"td={entry.get('td', 0):.3f} {entry['secs']}s", flush=True)
    last_state = ens.state() if ens is not None else {}
    for i, a in enumerate(anchor or []):
        last_state[f"anchor{i}"] = a
    return dict(delta=delta, umax=umax, log=log, last_state=last_state)


# ══════════ 온라인(봉마다) 학습 — 보조 트랙 X2 ══════════
ONLINE = dict(K=8, lr=1e-4, n_recent=64, window=540, lam_sp=1e-3, drift=0.25, gate_cost=0.003)


def online_month(cfg, month_ens, datas, Tk, a, b, delta0, seed):
    """
    한 달 동안 phase 0 결정봉마다 온라인 학습 (X2).
      t봉 판단 직전: 이미 끝난 거래(번호 ≤ t−2, O[t]까지만 필요)로 Adam 8스텝
        배치 = 최근 540봉에서 64개 + 이달 학습 표본에서 192개, 이달 모델 쪽으로 L2-SP 1e-3
        보조 목표는 t 이후 가격이 필요하면 가림 (6봉: u+7 > t, 42봉: u+43 > t)
      이탈 점검: 최근 540봉에서 이달 모델과 판단이 25% 넘게 다르면 다음 재학습까지 이달 모델로 복귀
    """
    from .nn import Adam
    O = ONLINE
    d0 = datas[0]
    tr = Trainer(cfg, seed + (99,))
    tr.make_pool(datas, Tk, FIRST_TRAIN, WARMUP)
    tr.init_from(month_ens.net.params)
    anchor = month_ens.net.copy_params()
    opt = Adam(tr.net.params, lr=O["lr"], clip=cfg["clip"])
    M, B = cfg["members"], cfg["batch"]
    nr = O["n_recent"]
    F = tr.flat                                   # phase 0 은 오프셋 0
    reverted_at = None
    steps = 0
    for t in range(a, b):
        if reverted_at is None:
            hi = t - 2
            u = np.arange(max(0, hi - O["window"] + 1), hi + 1)
            u = u[d0.contig[u] & ~d0.forced_hold[u] & (u >= WARMUP)]
            if len(u) > 10:
                for _ in range(O["K"]):
                    X0, X1, m, cost, z6, z42, ok = tr._batch()
                    ui = u[tr.rng.integers(0, len(u), size=(M, nr))]
                    X0[:, :nr, :-1] = tr.Xsel[ui]
                    X1[:, :nr, :-1] = tr.Xsel[ui + 1]
                    if cfg["noise"]:
                        X0[:, :nr, :-1] += cfg["noise"] * tr.rng.standard_normal(X0[:, :nr, :-1].shape, dtype=np.float32)
                        X1[:, :nr, :-1] += cfg["noise"] * tr.rng.standard_normal(X1[:, :nr, :-1].shape, dtype=np.float32)
                    m[:, :nr] = F["m"][ui]
                    z6[:, :nr] = F["z6"][ui]
                    z42[:, :nr] = F["z42"][ui]
                    # 보조 목표 두 개를 한 마스크로: 42봉 목표가 t까지 확정된 표본만
                    ok[:, :nr] = F["ok"][ui] * ((ui + 43) <= t)
                    z6[:, :nr] = np.where((ui + 7) <= t, z6[:, :nr], 0.0)
                    y = tr.targets(X1, m, cost)
                    _, grads, _ = tr.loss_grads(X0, y, z6, z42, ok, anchor, O["lam_sp"])
                    opt.step(grads)
                    for pt, p in zip(tr.tgt.params, tr.net.params):
                        pt *= (1 - cfg["tau"])
                        pt += cfg["tau"] * p
                    steps += 1
            on = tr.ensemble()
            w0 = max(a - O["window"], t - O["window"])
            if t - w0 >= 20:
                dn, _ = on.delta(d0.X[w0:t], O["gate_cost"])
                dm, _ = month_ens.delta(d0.X[w0:t], O["gate_cost"])
                pn = policy_from_delta(dn, O["gate_cost"], d0.forced_hold[w0:t])
                pm = policy_from_delta(dm, O["gate_cost"], d0.forced_hold[w0:t])
                if float((pn != pm).mean()) > O["drift"]:
                    reverted_at = int(t)
            model = on if reverted_at is None else month_ens
        else:
            model = month_ens
        for cd in delta0:
            dl, _ = model.delta(d0.X[t:t + 1], cd)
            delta0[cd][t] = dl[0]
    return dict(steps=steps, reverted_at=reverted_at)


# ══════════ 실행·저장 ══════════
TRAIN_CODE = ("agent.py", "nn.py", "env.py", "data.py", "features.py", "config.py", "walkforward.py")


def code_hash():
    """학습 결과에 영향을 주는 코드의 해시 — 코드가 바뀌면 저장된 실행 결과를 재사용하지 않습니다"""
    import hashlib
    h = hashlib.sha1()
    here = os.path.dirname(os.path.abspath(__file__))
    for f in TRAIN_CODE:
        with open(os.path.join(here, f), "rb") as fp:
            h.update(fp.read())
    return h.hexdigest()[:12]


def run_path(cfg, r):
    return os.path.join(RUNS_DIR, cfg["name"], f"rep{r:02d}.npz")


def run_is_current(cfg, r):
    path = run_path(cfg, r)
    if not os.path.exists(path):
        return False
    z = np.load(path)
    if "cfg_hash" not in z.files or "code_hash" not in z.files:
        return False
    return bytes(z["cfg_hash"]).decode() == C.train_hash(cfg) and bytes(z["code_hash"]).decode() == code_hash()


def save_run(cfg, r, res):
    path = run_path(cfg, r)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    flat = {}
    for k, per in res["delta"].items():
        for cd, arr in per.items():
            flat[f"d_{k}_{cd:.6f}"] = arr
    for k, arr in res["umax"].items():
        flat[f"u_{k}"] = arr
    for key, arr in res["last_state"].items():
        flat[f"s_{key}"] = arr
    flat["log"] = np.frombuffer(json.dumps(res["log"]).encode(), dtype=np.uint8)
    flat["cfg_hash"] = np.frombuffer(C.train_hash(cfg).encode(), dtype=np.uint8)
    flat["code_hash"] = np.frombuffer(code_hash().encode(), dtype=np.uint8)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **flat)
    os.replace(tmp, path)


def load_run(cfg_name, r):
    z = np.load(os.path.join(RUNS_DIR, cfg_name, f"rep{r:02d}.npz"))
    delta, umax, st = {}, {}, {}
    for key in z.files:
        if key.startswith("d_"):
            _, k, cd = key.split("_")
            delta.setdefault(int(k), {})[float(cd)] = z[key]
        elif key in ("cfg_hash", "code_hash", "log"):
            continue
        elif key.startswith("u_"):
            umax[int(key[2:])] = z[key]
        elif key.startswith("s_"):
            st[key[2:]] = z[key]
    out = dict(delta=delta, umax=umax, last_state=st, log=json.loads(bytes(z["log"]).decode()))
    out["code_hash"] = bytes(z["code_hash"]).decode() if "code_hash" in z.files else None
    out["cfg_hash"] = bytes(z["cfg_hash"]).decode() if "cfg_hash" in z.files else None
    return out


def log_trial(cfg, kind, extra=None):
    os.makedirs(RUNS_DIR, exist_ok=True)
    row = dict(hash=C.full_hash(cfg), train_hash=C.train_hash(cfg), name=cfg["name"], kind=kind,
               time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **(extra or {}))
    with open(TRIALS, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _trial_key(row):
    # 예산 연구는 시드별 실행을 따로 셉니다(설계서: 9회). 그 밖에는 같은 설정 해시를 한 번만.
    return ("budget", row["name"]) if row.get("kind") == "budget" else ("cfg", row["hash"])


def n_trials():
    """DSR의 N — 지금까지 시험한 서로 다른 설정 수 (보수적으로 크게 셈)"""
    if not os.path.exists(TRIALS):
        return 0
    with open(TRIALS, encoding="utf-8") as f:
        return len({_trial_key(json.loads(l)) for l in f if l.strip()})


def _job(args):
    cfg, r, eval_phases, end_s = args
    if run_is_current(cfg, r):
        return cfg["name"], r, "cached"
    t = time.time()
    try:
        res = run_replication(cfg, r, eval_phases=eval_phases, end=pd.Timestamp(end_s))
    except Exception:                               # 한 작업이 실패해도 나머지는 계속
        import traceback
        return cfg["name"], r, "FAILED\n" + traceback.format_exc()
    save_run(cfg, r, res)
    return cfg["name"], r, round(time.time() - t, 1)


def run_many(jobs, workers=4):
    load_phases()                                   # 캐시를 먼저 만들어 둠
    if workers <= 1:
        return [print(_job(j), flush=True) for j in jobs]
    with mp.get_context("spawn").Pool(workers) as pool:
        for out in pool.imap_unordered(_job, jobs):
            print("  완료", out, flush=True)


# ══════════ 학습 예산 연구 (2014~2016, 손익은 보지 않음) ══════════
def budget_study(workers=4):
    datas = load_phases()
    T_k = int(pd.Timestamp("2016-07-01", tz="UTC").timestamp())
    T_v = int(pd.Timestamp("2017-01-01", tz="UTC").timestamp())
    jobs = [(b, s) for b in (2000, 4000, 8000) for s in range(3)]
    with mp.get_context("spawn").Pool(workers) as pool:
        res = pool.map(_budget_job, [(b, s, T_k, T_v) for b, s in jobs])
    for (b, s), out in zip(jobs, res):
        cfg = C.make(f"budget{b}_s{s}", cold_steps=b, cold_split=int(b * 0.75))
        log_trial(cfg, "budget", dict(val=out["val"], slope=out["slope"]))
    by_b = {}
    for (b, s), out in zip(jobs, res):
        by_b.setdefault(b, []).append(out)
    summ = {b: dict(val=float(np.mean([o["val"] for o in v])), slope=float(np.mean([o["slope"] for o in v])))
            for b, v in by_b.items()}
    best = min(v["val"] for v in summ.values())
    ok = [b for b in sorted(summ) if summ[b]["val"] <= best * 1.01 and summ[b]["slope"] > 0]
    choice = ok[0] if ok else 4000
    out = dict(summary=summ, choice=choice, runs=[dict(budget=b, seed=s, **o) for (b, s), o in zip(jobs, res)])
    os.makedirs(RUNS_DIR, exist_ok=True)
    with open(os.path.join(RUNS_DIR, "budget.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


def _budget_job(args):
    b, s, T_k, T_v = args
    datas = load_phases()
    cfg = C.make("budget", cold_steps=b, cold_split=int(b * 0.75))
    tr = Trainer(cfg, (1000 + s, 2016, 7, 0))
    tr.make_pool(datas, T_k, FIRST_TRAIN, WARMUP)
    tr.init_fresh()
    tr.fit_cold()
    # 검증: 2016-07 ~ 2016-12 (t+43 종가가 2017-01-01 이전인 봉)
    idx = {}
    for k, d in enumerate(datas):
        e = d.eligible(T_v, T_k, 0)
        idx[k] = e[d.ts[e] >= T_k]
    val = tr.td_loss(datas, idx)
    # Δ 보정 기울기: 예측 Δ 10분위 vs 실현 κ·m (phase 0)
    d0 = datas[0]
    ens = tr.ensemble()
    dl, _ = ens.delta(d0.X[idx[0]], 0.003)
    km = 100.0 * d0.m[idx[0]]
    q = np.quantile(dl, np.linspace(0, 1, 11))
    bx, by = [], []
    for j in range(10):
        sel = (dl >= q[j]) & (dl <= q[j + 1])
        if sel.sum() > 5:
            bx.append(dl[sel].mean())
            by.append(km[sel].mean())
    slope = float(np.polyfit(bx, by, 1)[0]) if len(bx) >= 3 else 0.0
    return dict(val=float(val), slope=slope)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("budget")
    a1 = sub.add_parser("run")
    a1.add_argument("name")
    a1.add_argument("--reps", type=int, default=10)
    a1.add_argument("--workers", type=int, default=4)
    a2 = sub.add_parser("all", help="P0×10, 격자×5, 절제×3, 무작위대조 N2×8 전부")
    a2.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "all":
        jobs = all_jobs()
        run_many(jobs, a.workers)
        return
    if a.cmd == "budget":
        out = budget_study()
        print(json.dumps(out["summary"], indent=1), "→ 선택:", out["choice"])
    elif a.cmd == "run":
        cfg = named_cfg(a.name)
        jobs = [(cfg, r, None if a.name == "P0" else [0], str(OOS_END)) for r in range(a.reps)]
        run_many(jobs, a.workers)


def named_cfg(name):
    if name == "P0":
        return C.P0
    if name in C.ABLATIONS:
        return C.make(name, **C.ABLATIONS[name])
    if name == "N2":
        return C.make("N2", permute=True)
    g, f = name[1:].split("_", 1)
    return C.make(name, gamma=float(g), features=f)


def all_jobs():
    """사전 등록한 실험 전부. 느린 것(X2 온라인)을 먼저 넣어 병렬 효율을 높입니다."""
    jobs = []
    for r in range(3):
        jobs.append((named_cfg("X2"), r, [0], str(OOS_END)))
    for r in range(10):
        jobs.append((C.P0, r, None, str(OOS_END)))
    for g in C.grid():
        if g["c_mult"] != 2.0:
            continue
        name = f"g{g['gamma']:g}_{g['features']}"
        if name == f"g{C.P0['gamma']:g}_{C.P0['features']}":
            continue
        for r in range(5):
            jobs.append((named_cfg(name), r, [0], str(OOS_END)))
    for key in ("X1", "X3"):
        for r in range(3):
            jobs.append((named_cfg(key), r, [0], str(OOS_END)))
    for r in range(8):
        jobs.append((named_cfg("N2"), r, [0], str(N2_END)))
    # 시험 기록 (DSR의 N): 격자 12 + 절제 3. 같은 설정 해시는 한 번만 셉니다.
    seen = set()
    if os.path.exists(TRIALS):
        with open(TRIALS, encoding="utf-8") as f:
            seen = {_trial_key(json.loads(l)) for l in f if l.strip()}
    for kind, cfgs in (("grid", C.grid()), ("ablation", [named_cfg(k) for k in C.ABLATIONS])):
        for g in cfgs:
            row = dict(hash=C.full_hash(g), name=g["name"], kind=kind)
            if _trial_key(row) not in seen:
                log_trial(g, kind)
                seen.add(_trial_key(row))
    return jobs


if __name__ == "__main__":
    main()
