# -*- coding: utf-8 -*-
"""
연구 변형 평가 — 사후(post-hoc) 탐색입니다. 2017~2026 전체를 이미 본 뒤의 연구라서
여기 숫자는 '증거'로만 쓰고, 유망한 변형은 새 모의매매 트랙으로 등록해 앞으로의 기록으로 판정합니다.
잠금 구간(2025~) 계산은 여전히 감사 기록에 'research:<이름>'으로 남깁니다.

  BTC_LOCKBOX_OPEN=1 python -m btc.research.evaluate <변형이름> ...
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import json
import math

import numpy as np
import pandas as pd

from .. import stats as S
from ..evaluate import Window, baseline_targets, lower_median, FI
from ..walkforward import load_phases
from ..data import DATA_DIR
from .rl import k_policy, trend_filter, allowed_matrix
from .walk import load_run, decision_mask, n_trials_total, RUNS
from .variants import VARIANTS

ERAS = {"2017-20": ("2017-01-01", "2021-01-01"), "2021-24": ("2021-01-01", "2025-01-01"),
        "2025-26": ("2025-01-01", "2026-09-25")}
COST = 0.0015


def _ts(s):
    return int(pd.Timestamp(s, tz="UTC").timestamp())


def daily_hold(tg, mask, a, b):
    """판단봉에서만 목표를 두고 나머지는 NaN(유지)"""
    out = np.full(len(tg), np.nan)
    sel = np.arange(a, b)[mask[a:b]]
    out[sel] = tg[sel]
    return out


def baselines_daily(W, stride):
    d = W.datas[0]
    a, b = W.rng[0]
    bt = baseline_targets(W)
    mask = decision_mask(d, stride)
    out = {}
    for k in ("B0", "B2", "B5", "B6"):
        tg = bt[k] if k == "B0" else daily_hold(bt[k], mask, a, b)
        if k == "B0":
            tg = tg.copy()
        out[k] = W.run(tg, COST)
    return out


def variant_results(W, name, reps, cost_dec):
    cfg = VARIANTS[name]
    d = W.datas[0]
    a, b = W.rng[0]
    stride = cfg.get("stride", 1)
    mask = decision_mask(d, stride)
    acts = np.asarray(cfg.get("acts", (0.0, 1.0)))
    frac = bool(np.any((acts > 0) & (acts < 1)))
    out = []
    for r in range(reps):
        run = load_run(name, r)
        sel = np.arange(a, b)[mask[a:b]]
        if cfg.get("algo") == "direct":                          # 샤프 직접 최적화: 비중을 그대로 사용
            tg = np.full(d.T, np.nan)
            tg[sel] = np.round(run["U"][0.0][sel, 0] * 4) / 4       # 25% 단위로 (잔거래 억제)
            res = W.run(tg, COST, weights=True)
            res["log"], res["code_hash"] = run["log"], run["meta"].get("code_hash")
            out.append(res)
            continue
        U = run["U"][cost_dec]
        allowed = allowed_matrix(trend_filter(d.c)[sel], acts) if cfg.get("veto_b2") else None
        idx = k_policy(U[sel], acts, cost_dec, d.forced_hold[sel], allowed=allowed)
        tg = np.full(d.T, np.nan)
        w = acts[idx]
        w[np.isnan(U[sel]).any(axis=1)] = np.nan                  # 모델 없는 달(현금 유지)은 판단 보류
        tg[sel] = w
        if not frac:
            res = W.run(np.where(np.isnan(tg), np.nan, tg), COST)
        else:
            res = W.run(tg, COST, weights=True)
        res["log"] = run["log"]
        res["code_hash"] = run["meta"].get("code_hash")
        out.append(res)
    return out


def era_stats(days, r):
    out = {}
    for k, (lo, hi) in ERAS.items():
        sel = (days > _ts(lo)) & (days <= _ts(hi))
        if sel.sum() > 30:
            s = S.summary(r[sel])
            out[k] = dict(cagr=s["cagr"], sharpe=s["sharpe"], max_dd=s["max_dd"])
    return out


def evaluate(names, reps=5, window=("2017-01-01", "2026-09-25")):
    datas = load_phases()
    W = Window(datas, window[0], window[1], "research:" + ",".join(names))
    base = baselines_daily(W, 6)
    days = base["B0"]["days"]
    out = dict(window=window, cost=COST, n_trials=n_trials_total(), baselines={}, variants={})
    for k, v in base.items():
        s = S.summary(v["r"])
        out["baselines"][k] = dict(sharpe=s["sharpe"], cagr=s["cagr"], max_dd=s["max_dd"],
                                   exposure=float(np.nanmean(v["pos"])), eras=era_stats(days, v["r"]))
    allr = []
    for name in names:
        cfg = VARIANTS[name]
        cd = round(min(0.02, max(0.0005, COST * cfg.get("c_mult", 2.0))), 4)
        res = variant_results(W, name, reps, cd)
        srs = [S.sharpe(x["r"]) for x in res]
        d_bh = [s - S.sharpe(base["B0"]["r"]) for s in srs]
        d_b2 = [s - S.sharpe(base["B2"]["r"]) for s in srs]
        lm = lower_median(d_bh)
        head = res[lm]
        sm = S.summary(head["r"])
        boot_bh = S.stationary_bootstrap_diff(head["r"], base["B0"]["r"], n_boot=4000, seed=1)
        boot_b2 = S.stationary_bootstrap_diff(head["r"], base["B2"]["r"], n_boot=4000, seed=1)
        n_rb = None
        if head.get("rebal") is not None:
            n_rb = float(head["rebal"].sum() / (len(head["pos"]) / (6 * 365.0)))
        else:
            n_rb = float(np.abs(np.diff(np.concatenate([[0], head["pos"]]))).sum() / (len(head["pos"]) / (6 * 365.0)))
        gates = [(e["month"], e.get("kind"), e.get("accepted")) for e in head["log"] if e.get("accepted") is False]
        out["variants"][name] = dict(
            cfg={k: v for k, v in cfg.items()}, reps=len(res), c_dec=cd, headline_rep=lm,
            code_hashes=sorted({x.get("code_hash") for x in res if x.get("code_hash")}),
            sharpe_reps=srs, d_sharpe_bh_reps=d_bh, d_sharpe_b2_reps=d_b2,
            sharpe=sm["sharpe"], cagr=sm["cagr"], max_dd=sm["max_dd"], calmar=sm["calmar"],
            exposure=float(np.nanmean(head["pos"])), turnover_per_year=n_rb,
            vs_bh=dict(d=boot_bh["obs"], p=boot_bh["p"], ci90=boot_bh["ci90"]),
            vs_b2=dict(d=boot_b2["obs"], p=boot_b2["p"], ci90=boot_b2["ci90"]),
            eras=era_stats(days, head["r"]), rejected_gates=gates)
        allr.append((name, head["r"]))
    # 시험 수로 부풀림 걷어내기 (매수·보유 대비 초과수익)
    if allr:
        M = np.column_stack([r - base["B0"]["r"] for _, r in allr])
        srv = S.trials_sr_var(M) if M.shape[1] > 1 else 1e-4
        N = out["n_trials"]
        for name, r in allr:
            out["variants"][name]["dsr_excess"] = S.dsr(r - base["B0"]["r"], N, max(srv, 1e-6))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(RUNS, "report.json"))
    a = ap.parse_args()
    from ..evaluate import _clean
    out = _clean(evaluate(a.names, a.reps))
    prev = {}
    if os.path.exists(a.out):
        with open(a.out, encoding="utf-8") as f:
            prev = json.load(f)
    prev.setdefault("variants", {}).update(out["variants"])
    prev["baselines"], prev["n_trials"], prev["window"] = out["baselines"], out["n_trials"], out["window"]
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(prev, f, ensure_ascii=False, indent=1)
    b = out["baselines"]
    print("기준:", {k: (round(v["sharpe"], 2), f"{v['max_dd']*100:.0f}%") for k, v in b.items()})
    for n, v in out["variants"].items():
        print(f"{n:28s} 샤프 {v['sharpe']:.2f} (반복 {min(v['sharpe_reps']):.2f}~{max(v['sharpe_reps']):.2f}) "
              f"CAGR {v['cagr']*100:5.1f}% MDD {v['max_dd']*100:5.1f}% 노출 {v['exposure']*100:3.0f}% "
              f"회전 {v['turnover_per_year']:.1f}/년 | vs보유 {v['vs_bh']['d']:+.2f} p={v['vs_bh']['p']:.2f} "
              f"| vs200일 {v['vs_b2']['d']:+.2f} p={v['vs_b2']['p']:.2f} | DSR {v.get('dsr_excess', float('nan')):.2f} "
              f"| 2025-26 샤프 {v['eras'].get('2025-26', {}).get('sharpe', float('nan')):.2f}")


if __name__ == "__main__":
    main()
