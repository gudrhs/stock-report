# -*- coding: utf-8 -*-
"""
W1(softmin_direct)용 합성 대조 실험 실행기 — btc/research/controls.py 와 같은 경로·같은 판정.

controls.py 의 _job 은 비중 출력 모델을 cfg["algo"] == "direct" 로만 알아보므로,
cfg["output"] == "weights" 인 플러그인 기법(W1 등)은 여기서 돌립니다. 판정(verdict)·합성 경로·기간·비용은
controls.py 것을 그대로 가져다 씁니다. 비중은 evaluate.py 와 같이 25% 단위로 반올림합니다.

  python -m btc.research.softmin_direct_controls W1_softmin_direct --garch 6 --workers 4
  (variants.py 에 아직 없으면 btc/research/algos/softmin_direct.py 의 proposed_cfg() 를 씀)
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import json
import multiprocessing as mp

import numpy as np
import pandas as pd

from . import controls as CT

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "controls_weights.json")


def _cfg(name):
    from .variants import VARIANTS
    if name in VARIANTS:
        return dict(VARIANTS[name])
    from .algos.softmin_direct import proposed_cfg
    cfg = proposed_cfg()
    if cfg["name"] != name:
        raise KeyError(name)
    return cfg


def _job(args):
    name, kind, seed = args
    from .. import features as Fe, stats as S
    from ..env import PhaseData
    from ..evaluate import Window
    from .rl import feat_index, FEATURE_SETS
    from .walk import run_replication, decision_mask
    cfg = dict(_cfg(name), phases=1)
    assert cfg.get("output") == "weights", "비중 출력 모델만"
    df, reg = CT._series(kind, seed)
    X, sig = Fe.compute(df)
    planted = None
    if kind == "regime":
        rng = np.random.default_rng(seed + 100)
        nxt = np.concatenate([reg[1:], reg[-1:]])
        noisy = np.where(rng.random(len(reg)) < 0.9, nxt, 1 - nxt)
        planted = (cfg.get("feat") or FEATURE_SETS["full22"])[0]
        X[:, feat_index([planted])[0]] = np.where(noisy == 1, 1.0, -1.0)
    d = PhaseData(df, X, sig)
    end = pd.Timestamp(CT.HI, tz="UTC")
    res = run_replication(cfg, seed, datas=[d], end=end)
    W = Window([d], CT.LO, CT.HI, f"control:{name}")
    a, b = W.rng[0]
    sel = np.arange(a, b)[decision_mask(d, cfg.get("stride", 1))[a:b]]
    tg = np.full(d.T, np.nan)
    tg[sel] = np.round(res["U"][0.0][sel, 0] * 4) / 4
    ag = W.run(tg, CT.COST, weights=True)
    bh_t = np.full(d.T, np.nan)
    bh_t[a:b] = 1.0
    bh = W.run(bh_t, CT.COST)
    boot = S.stationary_bootstrap_diff(ag["r"], bh["r"], n_boot=4000, seed=seed)
    lam = [e.get("lam") for e in res["log"] if e.get("lam") is not None]
    return dict(name=name, kind=kind, seed=seed, planted=planted, d_sharpe=boot["obs"], p=boot["p"],
                sharpe=S.sharpe(ag["r"]), bh_sharpe=S.sharpe(bh["r"]),
                exposure=float(np.nanmean(ag["pos"])), rejected_gates=0,
                lam_mean=float(np.mean(lam)) if lam else None, r=ag["r"].tolist())


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
                  f"노출 {r['exposure']*100:.0f}% λ평균 {r['lam_mean']}", flush=True)
            runs.append(r)
    prev = {}
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            prev = json.load(f)
    for n in a.names:
        rr = [r for r in runs if r["name"] == n]
        prev[n] = dict(CT.verdict(rr), runs=[{k: v for k, v in r.items() if k != "r"} for r in rr])
        v = prev[n]
        print(f"{n}: 양성 {'통과' if v['positive_pass'] else '실패'} / 음성 "
              f"{'통과' if v.get('negative_pass') else '실패'} (평균 Δ {v.get('neg_mean_d', float('nan')):+.3f})")
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(prev, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
