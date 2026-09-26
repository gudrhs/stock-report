# -*- coding: utf-8 -*-
"""
1단계 선별 — btc/research/success_criteria.md 의 S1~S4를 변형마다 그대로 확인합니다.

  BTC_LOCKBOX_OPEN=1 python -m btc.research.screen R6_daily_trend8_uniform ...
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import json

import numpy as np

from .. import stats as S
from ..evaluate import Window, baseline_targets, lower_median
from ..walkforward import load_phases
from .rl import k_policy, trend_filter, allowed_matrix
from .walk import load_run, decision_mask, n_trials_total, RUNS, C_DECS
from .evaluate import era_stats
from .variants import VARIANTS

OUT = os.path.join(RUNS, "screen.json")


def targets(W, name, r):
    cfg = VARIANTS[name]
    d = W.datas[0]
    a, b = W.rng[0]
    sel = np.arange(a, b)[decision_mask(d, cfg.get("stride", 1))[a:b]]
    run = load_run(name, r)
    tg = np.full(d.T, np.nan)
    if cfg.get("algo") == "direct" or cfg.get("output") == "weights":
        tg[sel] = np.round(run["U"][0.0][sel, 0] * 4) / 4
        return tg, True, run
    acts = np.asarray(cfg.get("acts", (0.0, 1.0)), dtype=float)
    out = {}
    for cd in C_DECS:
        U = run["U"][cd][sel]
        allowed = allowed_matrix(trend_filter(d.c)[sel], acts) if cfg.get("veto_b2") else None
        w = acts[k_policy(U, acts, cd, d.forced_hold[sel], allowed=allowed)].astype(float)
        w[np.isnan(U).any(axis=1)] = np.nan
        t = tg.copy()
        t[sel] = w
        out[cd] = t
    frac = bool(np.any((acts > 0) & (acts < 1))) or bool(np.any(acts < 0))
    return out, frac, run


def run_tg(W, tg, cost, frac):
    return W.run(tg, cost, weights=True) if frac else W.run(tg, cost)


def screen(names, reps=5):
    datas = load_phases()
    W = Window(datas, "2017-01-01", "2026-09-25", "research-screen:" + ",".join(names))
    d = W.datas[0]
    a, b = W.rng[0]
    bh = W.run(baseline_targets(W)["B0"], 0.0015)
    bh3 = W.run(baseline_targets(W)["B0"], 0.003)
    days = bh["days"]
    sb = S.summary(bh["r"])
    bh_eras = era_stats(days, bh["r"])
    out = dict(n_trials=n_trials_total(), bh=dict(sharpe=sb["sharpe"], cagr=sb["cagr"], max_dd=sb["max_dd"], eras=bh_eras),
               variants={})
    for name in names:
        res15, res30 = [], []
        for r in range(reps):
            tg, frac, run = targets(W, name, r)
            if isinstance(tg, dict):
                res15.append(run_tg(W, tg[0.003], 0.0015, frac))      # 판단비용 = 2 × 0.15%
                res30.append(run_tg(W, tg[0.003], 0.003, frac))       # 0.30%에서는 2배(0.6%) 판단비용이 없어 1배
            else:
                res15.append(run_tg(W, tg, 0.0015, frac))
                res30.append(run_tg(W, tg, 0.003, frac))
        srs = [S.sharpe(x["r"]) for x in res15]
        lm = lower_median([s - sb["sharpe"] for s in srs])
        h = S.summary(res15[lm]["r"])
        er = era_stats(days, res15[lm]["r"])
        s30 = S.sharpe(res30[lm]["r"])
        c = dict(
            S1=all(s > sb["sharpe"] for s in srs),
            S2=bool(h["max_dd"] > sb["max_dd"] and h["cagr"] >= sb["cagr"] - 0.10),
            S3=all(er[k]["sharpe"] >= bh_eras[k]["sharpe"] - 0.15 for k in bh_eras),
            S4=bool(s30 > S.sharpe(bh3["r"])),
        )
        out["variants"][name] = dict(sharpe_reps=srs, headline=lm, sharpe=h["sharpe"], cagr=h["cagr"], max_dd=h["max_dd"],
                                     eras=er, sharpe_030=s30, checks=c, passed=all(c.values()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    from ..evaluate import _clean
    out = _clean(screen(a.names, a.reps))
    prev = {}
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            prev = json.load(f)
    prev.setdefault("variants", {}).update(out["variants"])
    prev["bh"], prev["n_trials"] = out["bh"], out["n_trials"]
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(prev, f, ensure_ascii=False, indent=1)
    bh = out["bh"]
    fmt = lambda eras: ", ".join("%s: %.2f" % (k, e["sharpe"]) for k, e in eras.items())
    print("매수·보유: 샤프 %.2f CAGR %.1f%% MDD %.1f%% 시기 {%s}  N=%d"
          % (bh["sharpe"], bh["cagr"] * 100, bh["max_dd"] * 100, fmt(bh["eras"]), out["n_trials"]))
    for n, v in out["variants"].items():
        ck = " ".join("%s%s" % (k, "○" if ok else "×") for k, ok in v["checks"].items())
        print("%-30s %s [%s] 샤프 %.2f (반복 %.2f~%.2f) CAGR %.1f%% MDD %.1f%% 0.30%%: %.2f 시기 {%s}"
              % (n, "통과" if v["passed"] else "탈락", ck, v["sharpe"], min(v["sharpe_reps"]), max(v["sharpe_reps"]),
                 v["cagr"] * 100, v["max_dd"] * 100, v["sharpe_030"], fmt(v["eras"])))


if __name__ == "__main__":
    main()
