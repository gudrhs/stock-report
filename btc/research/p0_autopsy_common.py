# -*- coding: utf-8 -*-
"""
P0 부검(autopsy) 공용 도구 — 사후(post-hoc) 연구용. btc/ 본체 코드는 건드리지 않고 가져다 씁니다.

· 잠금 구간 감사 로그(data/btc/lockbox_audit.log)에 쓰지 않도록 evaluate.Window 대신
  같은 회계를 여기서 다시 구현합니다(guard 호출 없음). 잠금 구간은 이미 열렸고, 이 연구는 사후 분석입니다.
· replay(): walkforward.monthly_update 와 한 줄씩 같은 월간 갱신을, 점검(gate)을 켜고 끄며 다시 돌립니다.
  시드는 원래와 같은 (r, 연, 월, 0) 이라 게이트를 켠 재현은 저장된 Δ와 비트 단위로 같아야 합니다.
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import math
import time

import numpy as np
import pandas as pd

from btc import config as C
from btc import stats as S
from btc.agent import Trainer, policy_from_delta, threshold
from btc.env import simulate, daily_marks, BAR_SEC
from btc.features import NAMES, WARMUP
from btc.walkforward import (load_phases, load_run, decision_range, months, sanity_ok, finetune_ok,
                             FIRST_TRAIN, OOS_END)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "p0_autopsy")
COST = C.PRIMARY_COST                       # 0.0015 편도
CD = round(C.c_dec(COST, C.P0["c_mult"]), 6)  # 0.003 판단용
TH = threshold(CD)                          # 0.3005
LO, HI = "2017-01-01", "2026-09-25"
REPS = 10
FI = {n: i for i, n in enumerate(NAMES)}


def ts(s):
    return int(pd.Timestamp(s, tz="UTC").timestamp())


# ══════════ 회계 (evaluate.Window.run 과 같은 계산, 감사 로그 없음) ══════════
def window_range(d, lo, hi):
    lo, hi = ts(lo), ts(hi)
    a, b = decision_range(d, lo, hi)
    while b > a and (d.ts[b] + BAR_SEC > hi or (b + 1 < d.T and d.ts[b + 1] > hi)):
        b -= 1
    return a, b


def run_targets(d, targets, cost, lo=LO, hi=HI):
    a, b = window_range(d, lo, hi)
    sim = simulate(targets, d.o, d.c, cost, a, b, forced_hold=d.forced_hold)
    days, eq = daily_marks(d.ts, sim["mark"], a, b, lo=ts(lo), hi=ts(hi))
    r = eq[1:] / eq[:-1] - 1.0
    return dict(days=days[1:], r=r, pos=sim["pos"][a:b], a=a, b=b, trades=sim["trades"])


def agent_targets(d, delta, cd=CD, lo=LO, hi=HI):
    a, b = window_range(d, lo, hi)
    tg = np.full(d.T, np.nan)
    tg[a:b] = policy_from_delta(delta[a:b], cd, d.forced_hold[a:b])
    return tg


def bh_targets(d, lo=LO, hi=HI):
    a, b = window_range(d, lo, hi)
    tg = np.full(d.T, np.nan)
    tg[a:b] = 1.0
    return tg


def b2_targets(d, lo=LO, hi=HI):
    a, b = window_range(d, lo, hi)
    sma = pd.Series(d.c).rolling(1200).mean().to_numpy()
    tg = np.full(d.T, np.nan)
    tg[a:b] = np.where(d.c[a:b] > sma[a:b], 1.0, 0.0)
    return tg


def sub_stats(days, r, lo, hi):
    """evaluate.subperiods 와 같은 자르기: lo < 날짜 ≤ hi 에 끝나는 일별 수익"""
    sel = (days > ts(lo)) & (days <= ts(hi))
    rr = r[sel]
    s = S.summary(rr)
    return dict(ret=float(np.prod(1 + rr) - 1), cagr=s["cagr"], sharpe=s["sharpe"], max_dd=s["max_dd"])


# ══════════ 월간 갱신 재현 (게이트 켜고 끄기) ══════════
def _behavioural(info):
    """게이트 탈락 사유가 '행동'(노출도·전환·불일치)인지 — 수치 이상(nonfinite/short_window)은 항상 거부"""
    return "reason" not in info


def replay(r, datas, gate_cold=True, gate_ft=True, start="2017-01-01", end=OOS_END, cds=(CD,),
           keep=(), verbose=False):
    """
    walkforward.monthly_update 를 그대로 따라 하되, 행동 점검을 끌 수 있습니다.
      gate_cold=False: 1월 새 모델을 노출도·전환 점검과 무관하게 채택
      gate_ft=False  : 월간 이어학습을 불일치율과 무관하게 채택
    반환: delta[cd] (phase 0, T), shadow[cd] (거부된 후보가 그 달에 냈을 Δ), log, models{month: ens}
    """
    cfg = C.P0
    d0 = datas[0]
    delta = {cd: np.full(d0.T, np.nan, np.float32) for cd in cds}
    shadow = {cd: np.full(d0.T, np.nan, np.float32) for cd in cds}
    log, models = [], {}
    ens, anchor = None, None
    Ms = [m for m in months() if pd.Timestamp(start, tz="UTC") <= m < end]
    for i, T_k in enumerate(Ms):
        Tk = int(T_k.timestamp())
        nxt = int(Ms[i + 1].timestamp()) if i + 1 < len(Ms) else int(end.timestamp())
        seed = (int(r), T_k.year, T_k.month, 0)
        t0 = time.time()
        entry = dict(month=str(T_k.date()))
        if T_k.month == 1 or ens is None:
            tr = Trainer(cfg, seed)
            n_pool = tr.make_pool(datas, Tk, FIRST_TRAIN, WARMUP)
            tr.init_fresh()
            losses = tr.fit_cold()
            new = tr.ensemble()
            ok, info = sanity_ok(new, d0, Tk)
            acc = ok or (not gate_cold and _behavioural(info))
            entry.update(kind="cold", pool=n_pool, td=float(np.mean(losses[-200:])), gate_ok=ok, accepted=acc, **info)
            if acc:
                ens = new
                anchor = new.net.copy_params()
        else:
            tr = Trainer(cfg, seed)
            n_pool = tr.make_pool(datas, Tk, FIRST_TRAIN, WARMUP)
            tr.init_from(ens.net.params)
            losses = tr.fit_finetune(anchor)
            new = tr.ensemble()
            ok, info = finetune_ok(new, ens, d0, Tk)
            acc = ok or (not gate_ft and _behavioural(info))
            entry.update(kind="finetune", pool=n_pool, td=float(np.mean(losses[-50:])), gate_ok=ok, accepted=acc, **info)
            if acc:
                ens = new
        if entry["month"] in keep:
            models[entry["month"]] = new
        entry["secs"] = round(time.time() - t0, 2)
        if ens is not None:
            a, b = decision_range(d0, Tk, nxt)
            if b > a:
                for cd in cds:
                    delta[cd][a:b] = ens.delta(d0.X[a:b], cd)[0]
                    if not entry["accepted"]:
                        shadow[cd][a:b] = new.delta(d0.X[a:b], cd)[0]
        log.append(entry)
        if verbose:
            print(f"  r{r} {entry['month']} {entry['kind']} ok={entry['gate_ok']} acc={entry['accepted']} "
                  f"{entry['secs']}s", flush=True)
    return dict(delta=delta, shadow=shadow, log=log, models=models)


def fwd_discounted(m, gamma=0.97, kappa=100.0, horizon=200):
    """G_t = Σ_{k<H} γ^k κ m[t+k] — '계속 보유 vs 계속 현금'의 실현 가치 차이 (Δ의 몬테카를로 짝)"""
    m = np.nan_to_num(m)
    T = len(m)
    G = np.zeros(T)
    w = gamma ** np.arange(horizon)
    mm = np.concatenate([m, np.zeros(horizon)])
    # 직접 합 (T·H = 20k×200 수준이라 충분히 빠름)
    for k in range(horizon):
        G += w[k] * mm[k:k + T]
    G *= kappa
    valid = np.arange(T) + horizon < T
    G[~valid] = np.nan
    return G
