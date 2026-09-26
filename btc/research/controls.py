# -*- coding: utf-8 -*-
"""
연구 변형용 합성 대조 실험 — 새 보상·드리프트 제거·비율 보유가 '진짜 신호는 잡고, 없는 신호는 지어내지 않는지'.

  양성 대조 : 추세 국면(평균 33일 지속)을 90% 맞히는 '심은' 지표를 변형의 첫 입력 자리에 넣은 경로
             → 매수·보유보다 뚜렷이 나아야 함 (p < 0.01)
  음성 대조 : 예측력 없는 GARCH(1,1) 무추세 경로 여러 개
             → 평균 ΔSharpe가 2 표준오차 안, 시험 수로 걷어낸 DSR ≥ 0.95인 경로 없음

기간: 2017-01 ~ 2025-01 (합성 데이터라 잠금 구간과 무관), phase 1개, 하루 판단은 변형 설정대로.
합성 경로 생성은 tests/test_btc_controls.py 와 같은 식입니다.

  python -m btc.research.controls R3_daily_trend8_log5 R7_daily_trend8_drift0 --garch 6 --workers 4
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import json
import math
import multiprocessing as mp

import numpy as np
import pandas as pd

LO, HI = "2017-01-01", "2025-01-01"
COST = 0.0015
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "controls.json")


def _series(kind, seed):
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests"))
    from test_btc_controls import _series as s
    return s(kind, seed)


def _job(args):
    name, kind, seed = args
    from .. import features as Fe, stats as S
    from ..env import PhaseData
    from ..evaluate import Window
    from .rl import feat_index, FEATURE_SETS, k_policy, trend_filter, allowed_matrix
    from .walk import run_replication, decision_mask
    from .variants import VARIANTS
    cfg = dict(VARIANTS[name], phases=1)
    df, reg = _series(kind, seed)
    X, sig = Fe.compute(df)
    planted = None
    if kind == "regime":
        rng = np.random.default_rng(seed + 100)
        nxt = np.concatenate([reg[1:], reg[-1:]])
        noisy = np.where(rng.random(len(reg)) < 0.9, nxt, 1 - nxt)
        planted = (cfg.get("feat") or FEATURE_SETS["full22"])[0]
        X[:, feat_index([planted])[0]] = np.where(noisy == 1, 1.0, -1.0)
    d = PhaseData(df, X, sig)
    end = pd.Timestamp(HI, tz="UTC")
    res = run_replication(cfg, seed, datas=[d], end=end)
    W = Window([d], LO, HI, f"control:{name}")
    a, b = W.rng[0]
    sel = np.arange(a, b)[decision_mask(d, cfg.get("stride", 1))[a:b]]
    tg = np.full(d.T, np.nan)
    if cfg.get("algo") == "direct":
        tg[sel] = np.round(res["U"][0.0][sel, 0] * 4) / 4
        ag = W.run(tg, COST, weights=True)
    else:
        acts = np.asarray(cfg.get("acts", (0.0, 1.0)))
        cd = round(min(0.02, max(0.0005, COST * cfg.get("c_mult", 2.0))), 4)
        U = res["U"][cd][sel]
        allowed = allowed_matrix(trend_filter(d.c)[sel], acts) if cfg.get("veto_b2") else None
        w = acts[k_policy(U, acts, cd, d.forced_hold[sel], allowed=allowed)]
        w[np.isnan(U).any(axis=1)] = np.nan
        tg[sel] = w
        frac = bool(np.any((acts > 0) & (acts < 1)))
        ag = W.run(tg, COST, weights=frac)
    bh_t = np.full(d.T, np.nan)
    bh_t[a:b] = 1.0
    bh = W.run(bh_t, COST)
    boot = S.stationary_bootstrap_diff(ag["r"], bh["r"], n_boot=4000, seed=seed)
    rej = sum(1 for e in res["log"] if e.get("accepted") is False)
    return dict(name=name, kind=kind, seed=seed, planted=planted, d_sharpe=boot["obs"], p=boot["p"],
                sharpe=S.sharpe(ag["r"]), bh_sharpe=S.sharpe(bh["r"]),
                exposure=float(np.nanmean(ag["pos"])), rejected_gates=rej, r=ag["r"].tolist())


def verdict(runs):
    from .. import stats as S
    pos = [r for r in runs if r["kind"] == "regime"]
    neg = [r for r in runs if r["kind"] == "garch"]
    out = dict(positive=[dict(seed=r["seed"], d_sharpe=r["d_sharpe"], p=r["p"], exposure=r["exposure"])
                         for r in pos])
    out["positive_pass"] = bool(pos) and all(r["d_sharpe"] > 0 and r["p"] < 0.01 for r in pos)
    if len(neg) >= 2:
        dd = np.array([r["d_sharpe"] for r in neg])
        se = dd.std(ddof=1) / math.sqrt(len(dd))
        M = np.column_stack([np.array(r["r"]) for r in neg])
        srv = S.trials_sr_var(M)
        dsr = [S.dsr(np.array(r["r"]), len(neg), srv) for r in neg]
        out.update(neg_mean_d=float(dd.mean()), neg_se=float(se), neg_dsr=dsr,
                   neg_exposure=[r["exposure"] for r in neg],
                   negative_pass=bool(dd.mean() <= 2 * se + 1e-9 and all(x < 0.95 for x in dsr)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--garch", type=int, default=6)
    ap.add_argument("--regime", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    jobs = [(n, "regime", 1 + s) for n in a.names for s in range(a.regime)] + \
           [(n, "garch", s) for n in a.names for s in range(a.garch)]
    with mp.get_context("spawn").Pool(a.workers) as pool:
        runs = []
        for r in pool.imap_unordered(_job, jobs):
            print(f"  {r['name']:26s} {r['kind']:6s} s{r['seed']} ΔSharpe {r['d_sharpe']:+.2f} p={r['p']:.3f} "
                  f"노출 {r['exposure']*100:.0f}%", flush=True)
            runs.append(r)
    prev = {}
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            prev = json.load(f)
    for n in a.names:
        rr = [r for r in runs if r["name"] == n]
        prev[n] = dict(verdict(rr), runs=[{k: v for k, v in r.items() if k != "r"} for r in rr])
        v = prev[n]
        print(f"{n}: 양성 {'통과' if v['positive_pass'] else '실패'} / 음성 "
              f"{'통과' if v.get('negative_pass') else '실패'} (평균 Δ {v.get('neg_mean_d', float('nan')):+.3f})")
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(prev, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
