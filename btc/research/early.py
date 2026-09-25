# -*- coding: utf-8 -*-
"""
과거 미사용 구간 2015~2016 시험 — btc/research/success_criteria.md 의 같은 이름 조항(결과 보기 전 등록) 그대로.

"2015-01부터 이 방식으로 운용했다면": 고정된 변형 설정·코드로 walk.run_replication 을 그대로 부르되,
달력(walk.months)만 이 프로세스 안에서 2015-01부터 시작하게 바꿉니다. 첫 모델은 2014년 1년치로(1월 처음부터) 학습하고
이후 매월 다시 학습합니다. 시드는 달마다 (반복, 연, 월)로 정해지므로 2017년 이후 실행과 같은 규칙입니다.

  python -m btc.research.early run R6_daily_trend8_uniform R3_daily_trend8_log5 W2_dp_band --reps 5 --workers 4
  python -m btc.research.early evaluate R6_daily_trend8_uniform R3_daily_trend8_log5 W2_dp_band
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import json
import multiprocessing as mp
import time

import numpy as np
import pandas as pd

from .. import stats as S
from ..evaluate import Window, baseline_targets, lower_median
from ..walkforward import load_phases
from .rl import k_policy, trend_filter, allowed_matrix
from .walk import decision_mask, load_run, save_run, run_path, code_hash, RUNS
from .evaluate import daily_hold
from .variants import VARIANTS

LO, HI = pd.Timestamp("2015-01-01", tz="UTC"), pd.Timestamp("2017-01-01", tz="UTC")
TAG = "__early"
COST, C_DEC = 0.0015, 0.003
OUT = os.path.join(RUNS, "early.json")


def run_replication(cfg, r, datas=None):
    """walk.run_replication 을 그대로 부르고 달력만 2015-01 ~ 2016-12 로 (이 프로세스 안에서만)"""
    from . import walk
    orig = walk.months
    walk.months = lambda *a, **k: list(pd.date_range(LO, HI - pd.DateOffset(months=1), freq="MS", tz="UTC"))
    try:
        return walk.run_replication(cfg, r, datas=datas, end=HI)
    finally:
        walk.months = orig


def _job(args):
    cfg, r = args
    name = cfg["name"] + TAG
    if os.path.exists(run_path(name, r)):
        return name, r, "cached"
    t = time.time()
    ch = code_hash()
    try:
        res = run_replication(cfg, r)
    except Exception:
        import traceback
        return name, r, "FAILED\n" + traceback.format_exc()
    save_run(cfg, r, res, ch, name=name)
    return name, r, round(time.time() - t, 1)


def run_many(names, reps, workers):
    load_phases()
    jobs = [(VARIANTS[n], r) for n in names for r in range(reps)]      # 같은 설정의 다른 기간 — 새 시험으로 세지 않음
    with mp.get_context("spawn").Pool(workers) as pool:
        for out in pool.imap_unordered(_job, jobs):
            print("  완료", out, flush=True)


def targets(W, name, r):
    """btc.research.screen.targets 와 같은 계산 (판단비용 0.3%)"""
    cfg = VARIANTS[name]
    d = W.datas[0]
    a, b = W.rng[0]
    sel = np.arange(a, b)[decision_mask(d, cfg.get("stride", 1))[a:b]]
    run = load_run(name + TAG, r)
    tg = np.full(d.T, np.nan)
    if cfg.get("algo") == "direct" or cfg.get("output") == "weights":
        w = run["U"][0.0][sel, 0]
        tg[sel] = np.round(w * 4) / 4
        return tg, True, float(np.isnan(w).mean())
    acts = np.asarray(cfg.get("acts", (0.0, 1.0)), dtype=float)
    U = run["U"][C_DEC][sel]
    allowed = allowed_matrix(trend_filter(d.c)[sel], acts) if cfg.get("veto_b2") else None
    w = acts[k_policy(U, acts, C_DEC, d.forced_hold[sel], allowed=allowed)].astype(float)
    nan = np.isnan(U).any(axis=1)
    w[nan] = np.nan
    tg[sel] = w
    frac = bool(np.any((acts > 0) & (acts < 1)))
    return tg, frac, float(nan.mean())


def evaluate(names, reps=5):
    datas = load_phases()
    W = Window(datas, str(LO.date()), str(HI.date()), "research-early:2015-2016")
    d = W.datas[0]
    a, b = W.rng[0]
    bt = baseline_targets(W)
    m0 = decision_mask(d, 6)
    bh = W.run(bt["B0"], COST)
    rules = {"B2_200일선": W.run(daily_hold(bt["B2"], m0, a, b), COST),
             "B5_일봉MACD": W.run(daily_hold(bt["B5"], m0, a, b), COST)}
    sb = S.summary(bh["r"])
    out = dict(window=[str(LO.date()), str(HI.date())], cost=COST,
               bh=dict(sharpe=sb["sharpe"], cagr=sb["cagr"], max_dd=sb["max_dd"]),
               rules={k: {kk: S.summary(v["r"])[kk] for kk in ("sharpe", "cagr", "max_dd")} for k, v in rules.items()},
               variants={})
    for name in names:
        res, nan_shares = [], []
        for r in range(reps):
            tg, frac, ns = targets(W, name, r)
            res.append(W.run(tg, COST, weights=True) if frac else W.run(tg, COST))
            nan_shares.append(ns)
        srs = [S.sharpe(x["r"]) for x in res]
        lm = lower_median([s - sb["sharpe"] for s in srs])
        h = res[lm]
        sh = S.summary(h["r"])
        boot = S.stationary_bootstrap_diff(h["r"], bh["r"], n_boot=4000, seed=1)
        flagged = max(nan_shares) > 0.05
        v = dict(sharpe_reps=srs, headline_rep=lm, sharpe=sh["sharpe"], cagr=sh["cagr"], max_dd=sh["max_dd"],
                 exposure=float(np.nanmean(h["pos"])), d_vs_bh=boot["obs"], p_vs_bh=boot["p"], ci90=boot["ci90"],
                 d_vs_rules={k: sh["sharpe"] - S.sharpe(x["r"]) for k, x in rules.items()},
                 no_model_share=nan_shares, no_model_flag=flagged,
                 years={}, verdict=None)
        days = h["days"]
        for y in (2015, 2016):
            sel = (days >= int(pd.Timestamp(f"{y}-01-01", tz="UTC").timestamp())) & \
                  (days < int(pd.Timestamp(f"{y + 1}-01-01", tz="UTC").timestamp()))
            v["years"][y] = dict(strategy=S.summary(h["r"][sel])["cagr"], bh=S.summary(bh["r"][sel])["cagr"])
        if flagged:
            v["verdict"] = "판정 불가 (모델 없음 비율 5% 초과)"
        else:
            ok = sh["sharpe"] > sb["sharpe"] and sh["max_dd"] > sb["max_dd"]
            v["verdict"] = "유지" if ok else "유지 안 됨"
        out["variants"][name] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("names", nargs="+")
    r.add_argument("--reps", type=int, default=5)
    r.add_argument("--workers", type=int, default=4)
    e = sub.add_parser("evaluate")
    e.add_argument("names", nargs="+")
    e.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    if a.cmd == "run":
        run_many(a.names, a.reps, a.workers)
        return
    from ..evaluate import _clean
    out = _clean(evaluate(a.names, a.reps))
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    b = out["bh"]
    print(f"판정 기간 {out['window'][0]} ~ {out['window'][1]} (편도 {COST*100:.2f}%)")
    print(f"  매수·보유: 샤프 {b['sharpe']:.2f} CAGR {b['cagr']*100:.1f}% 최대낙폭 {b['max_dd']*100:.1f}%")
    for k, v in out["rules"].items():
        print(f"  {k}: 샤프 {v['sharpe']:.2f} CAGR {v['cagr']*100:.1f}% 최대낙폭 {v['max_dd']*100:.1f}%")
    for n, v in out["variants"].items():
        print(f"  {n}: {v['verdict']} — 샤프 {v['sharpe']:.2f} (반복 {min(v['sharpe_reps']):.2f}~{max(v['sharpe_reps']):.2f}) "
              f"CAGR {v['cagr']*100:.1f}% 최대낙폭 {v['max_dd']*100:.1f}% 노출 {v['exposure']*100:.0f}% | "
              f"매수·보유 대비 {v['d_vs_bh']:+.2f} (p={v['p_vs_bh']:.2f}) | "
              + ", ".join(f"{k} 대비 {x:+.2f}" for k, x in v["d_vs_rules"].items())
              + f" | 모델 없음 최대 {max(v['no_model_share'])*100:.1f}% | 연도 CAGR "
              + ", ".join(f"{y}: {x['strategy']*100:+.0f}% (보유 {x['bh']*100:+.0f}%)" for y, x in v["years"].items()))


if __name__ == "__main__":
    main()
