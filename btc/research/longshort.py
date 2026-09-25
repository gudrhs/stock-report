# -*- coding: utf-8 -*-
"""
롱숏(공매도 포함) 평가 — 수수료와 숏 유지비(대여이자·펀딩비)를 넣은 사후(post-hoc) 연구입니다.

회계 (하루 한 번 판단, 다음 봉 시가 체결)
  · 자산 E = 현금 + 코인수량 × 가격  (숏이면 코인수량 < 0, 판 돈은 현금에 들어감)
  · 목표 비중 w ∈ [−1, 1]로 맞출 때 수수료 = cost × 거래금액. 수수료를 낸 뒤 비중이 정확히 w가 되게 풉니다.
  · 판단봉 사이에는 수량 고정 (가격이 오르면 숏 비중이 커짐) — 다음 판단봉에서 목표와 2%p 넘게 벌어지면 다시 맞춤
  · 숏 유지비: 숏 명목금액 × 연율 ÷ (1년 봉 수)를 봉마다 현금에서 뺌 (+면 숏이 냄, −면 숏이 받음)
  · 자산이 0 이하가 되면 청산(이후 0) — 1배 숏은 가격이 두 배가 되면 전부 잃음
  · 마지막 날에는 청산 비용(|수량| × 종가 × cost)까지 뺌

로그성장 보상(reward='log')이 이 회계와 같은 식이라 숏의 변동성 손실(≈ 하루 분산만큼)이 학습에도 들어갑니다.

  BTC_LOCKBOX_OPEN=1 python -m btc.research.longshort L1_daily_trend8_ls3 L2_daily_trend8_ls5 ...
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import json
import math

import numpy as np

from .. import stats as S
from ..env import BAR_SEC, daily_marks
from ..evaluate import Window, baseline_targets, lower_median
from ..walkforward import load_phases
from .rl import k_policy
from .walk import load_run, decision_mask, n_trials_total, log_trial, RUNS, C_DECS
from .evaluate import daily_hold, era_stats
from .variants import VARIANTS

COSTS = (0.0005, 0.0015, 0.003)           # 편도 수수료+슬리피지 (선물 테이커 · 기본 · 보수적)
CARRY = (-0.10, 0.0, 0.10)                 # 숏 유지비 연율 (−: 펀딩을 받음, +: 대여이자·펀딩을 냄)
PRIMARY = (0.0015, 0.0)
BARS_PER_YEAR = 365.25 * 86400 / BAR_SEC
OUT = os.path.join(RUNS, "longshort.json")

RULES = {   # 기준 신호 (1 = 강세) — 교과서 기본값, 조정하지 않음
    "B1": "골든크로스(≈일봉 50/200)", "B2": "200일선", "B5": "일봉 MACD", "B6": "1개월 모멘텀",
}


def simulate_ls(weights, o, c, cost, start, end, carry=0.0, band=0.02, forced_hold=None, exit_cost=True):
    """
    weights: 봉별 목표 비중 (NaN = 유지), −1 ~ 1.  carry: 숏 유지비 연율 (실수 또는 봉별 배열)
    반환: pos(체결 후 비중), logr(봉별 로그 자산변화), mark(종가 기준 자산), rebal, liquidated,
          fees·carried(봉별 수수료·유지비, 그때 자산 대비 비율 — 합하면 대략 로그수익 손실)
    """
    T = len(o)
    if end >= T:
        raise ValueError("end 봉이 있어야 합니다")
    carry_bar = np.broadcast_to(np.asarray(carry, dtype=float), (T,)) / BARS_PER_YEAR
    pos = np.zeros(T)
    logr = np.zeros(T)
    mark = np.full(T, np.nan)
    rebal = np.zeros(T, bool)
    fees = np.zeros(T)
    carried = np.zeros(T)
    cash, units = 1.0, 0.0
    mark[start] = 1.0
    E_prev = 1.0
    dead = False
    for t in range(start, end):
        j = t + 1
        if dead:
            mark[j] = 0.0
            continue
        E = cash + units * o[j]
        tg = weights[t]
        if not np.isnan(tg) and not (forced_hold is not None and forced_hold[t]):
            w_cur = units * o[j] / E
            if abs(tg - w_cur) > band:
                gap = tg * E - units * o[j]
                V = gap / (1.0 + tg * cost * math.copysign(1.0, gap))    # 수수료 낸 뒤 비중이 정확히 tg
                fee = cost * abs(V)
                units += V / o[j]
                cash -= V + fee
                fees[t] = fee / E                                             # 자산 대비 비율
                rebal[t] = True
        if units < 0:                                                     # j봉 동안의 숏 유지비
            ch = carry_bar[j] * (-units) * o[j]
            carried[t] = ch / (cash + units * o[j])                          # 자산 대비 비율
            cash -= ch
        E_after = cash + units * o[j]
        pos[t] = units * o[j] / E_after if E_after > 0 else 0.0
        E_next = cash + units * o[j + 1] if j + 1 < T else E_after
        mark[j] = cash + units * c[j]
        if E_after <= 0 or mark[j] <= 0:                                  # 이 봉 안에서 청산
            dead, mark[j], logr[t] = True, 0.0, -np.inf
            continue
        if E_next <= 0:                                                   # 다음 봉 시가 전에 청산
            dead, logr[t] = True, -np.inf
            continue
        logr[t] = math.log(E_next / E_prev)
        E_prev = E_next
    if not dead:
        if exit_cost and units != 0:
            mark[end] = cash + units * c[end] - cost * abs(units) * c[end]
    return dict(pos=pos, logr=logr, mark=mark, rebal=rebal, fees=fees, carried=carried, liquidated=dead)


def run_ls(W, targets, cost, carry=0.0, phase=0):
    """Window.run과 같은 일별 수익 (잠금 구간 이후 가격 미사용)"""
    d = W.datas[phase]
    a, b = W.rng[phase]
    sim = simulate_ls(targets, d.o, d.c, cost, a, b, carry=carry, forced_hold=d.forced_hold)
    days, eq = daily_marks(d.ts, sim["mark"], a, b, lo=W.lo, hi=W.hi)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = eq[1:] / eq[:-1] - 1.0
    r = np.where(np.isfinite(r), r, 0.0)                                  # 청산 뒤(0/0)는 0
    pos = sim["pos"][a:b]
    lr = sim["logr"][a:b]
    return dict(days=days[1:], r=r, pos=pos, liquidated=sim["liquidated"],
                long_logr=float(lr[pos > 1e-9].sum()), short_logr=float(lr[pos < -1e-9].sum()),
                fees=float(sim["fees"][a:b].sum()), carried=float(sim["carried"][a:b].sum()),
                rebal_per_year=float(sim["rebal"][a:b].sum() / ((b - a) / (6 * 365.0))))


def describe(res, days):
    s = S.summary(res["r"])
    pos = res["pos"]
    return dict(sharpe=s["sharpe"], cagr=s["cagr"], max_dd=s["max_dd"], calmar=s["calmar"],
                net_exposure=float(pos.mean()), gross_exposure=float(np.abs(pos).mean()),
                time_short=float((pos < -1e-9).mean()), time_long=float((pos > 1e-9).mean()),
                long_logr=res["long_logr"], short_logr=res["short_logr"], fees=res["fees"],
                carried=res["carried"], rebal_per_year=res["rebal_per_year"],
                liquidated=bool(res["liquidated"]), eras=era_stats(days, res["r"]))


def rule_targets(W):
    """기준 규칙의 롱·현금(LF)과 롱·숏(LS) 목표 — 하루 한 번(phase 0 00:00 UTC 판단봉)"""
    d = W.datas[0]
    a, b = W.rng[0]
    bt = baseline_targets(W)
    mask = decision_mask(d, 6)
    out = {"B0": bt["B0"].copy()}
    short = np.full(d.T, np.nan)
    short[a:b] = -1.0
    out["S0_always_short"] = short
    for k in RULES:
        s = daily_hold(bt[k], mask, a, b)
        out[f"{k}_LF"] = s
        out[f"{k}_LS"] = np.where(np.isnan(s), np.nan, 2.0 * s - 1.0)
    return out


def variant_targets(W, name, r, cost):
    cfg = VARIANTS[name]
    d = W.datas[0]
    a, b = W.rng[0]
    acts = np.asarray(cfg["acts"], dtype=float)
    cd = min(C_DECS, key=lambda x: abs(x - 2.0 * cost))
    run = load_run(name, r)
    sel = np.arange(a, b)[decision_mask(d, cfg.get("stride", 1))[a:b]]
    U = run["U"][cd][sel]
    w = acts[k_policy(U, acts, cd, d.forced_hold[sel])].astype(float)
    w[np.isnan(U).any(axis=1)] = np.nan
    tg = np.full(d.T, np.nan)
    tg[sel] = w
    return tg, run["log"]


def evaluate(names, reps=5, window=("2017-01-01", "2026-09-25"), compare=None):
    datas = load_phases()
    W = Window(datas, window[0], window[1], "research-ls:" + ",".join(names or ["rules"]))
    rules = rule_targets(W)
    for k in rules:                                                       # 규칙도 시험으로 셈 (B0 제외)
        if k != "B0":
            log_trial(dict(name=f"LSRULE_{k}", rule=k, daily=True))
    out = dict(window=window, costs=COSTS, carry=CARRY, primary=PRIMARY, rules={}, variants={})
    prim = {}
    for k, tg in rules.items():
        out["rules"][k] = {}
        for cost in COSTS:
            for carry in CARRY:
                res = run_ls(W, tg, cost, carry)
                out["rules"][k][f"{cost}|{carry}"] = describe(res, res["days"])
                if (cost, carry) == PRIMARY:
                    prim[k] = res
    days = prim["B0"]["days"]
    heads = {}
    for name in names:
        per_rep = []
        for r in range(reps):
            tg, log = variant_targets(W, name, r, PRIMARY[0])
            per_rep.append((tg, log))
        prim_res = [run_ls(W, tg, *PRIMARY) for tg, _ in per_rep]
        srs = [S.sharpe(x["r"]) for x in prim_res]
        lm = lower_median([s - S.sharpe(prim["B0"]["r"]) for s in srs])
        heads[name] = prim_res[lm]
        v = dict(cfg=VARIANTS[name], headline_rep=lm, sharpe_reps=srs, scen={},
                 rejected_gates=[(e["month"], e.get("kind")) for e in per_rep[lm][1] if e.get("accepted") is False])
        for cost in COSTS:
            tg_c, _ = variant_targets(W, name, lm, cost)
            for carry in CARRY:
                res = run_ls(W, tg_c, cost, carry)
                v["scen"][f"{cost}|{carry}"] = describe(res, res["days"])
        out["variants"][name] = v
    # 비교: 매수·보유, 같은 규칙의 롱·현금판, (변형이면) 짝이 되는 롱 전용 변형
    pairs = [(f"{k}_LS", "B0") for k in RULES] + [(f"{k}_LS", f"{k}_LF") for k in RULES] + \
            [(f"{k}_LF", "B0") for k in RULES] + [(n, "B0") for n in names] + \
            [(n, "B2_LF") for n in names] + [(n, "B2_LS") for n in names]
    allr = {**{k: v["r"] for k, v in prim.items()}, **{n: h["r"] for n, h in heads.items()}}
    out["tests"] = []
    for x, y in pairs:
        bt = S.stationary_bootstrap_diff(allr[x], allr[y], n_boot=4000, seed=1)
        out["tests"].append(dict(a=x, b=y, d_sharpe=bt["obs"], p=bt["p"], ci90=bt["ci90"]))
    if compare:
        for n, base in compare.items():
            from .walk import run_path
            if base in VARIANTS and all(os.path.exists(run_path(base, r)) for r in range(reps)):
                from .evaluate import variant_results
                cd = min(C_DECS, key=lambda x: abs(x - 2.0 * PRIMARY[0]))
                rr = variant_results(W, base, reps, cd)
                lmb = lower_median([S.sharpe(x["r"]) - S.sharpe(prim["B0"]["r"]) for x in rr])
                bt = S.stationary_bootstrap_diff(allr[n], rr[lmb]["r"], n_boot=4000, seed=1)
                out["tests"].append(dict(a=n, b=base, d_sharpe=bt["obs"], p=bt["p"], ci90=bt["ci90"]))
    # 시험 수로 부풀림 걷어내기 (매수·보유 대비 초과수익)
    N = n_trials_total()
    keys = [k for k in allr if k not in ("B0",)]
    M = np.column_stack([allr[k] - allr["B0"] for k in keys])
    srv = max(S.trials_sr_var(M), 1e-6)
    out["n_trials"] = N
    out["dsr_excess"] = {k: S.dsr(allr[k] - allr["B0"], N, srv) for k in keys}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    from ..evaluate import _clean
    comp = {"L1_daily_trend8_ls3": "R2_daily_trend8", "L2_daily_trend8_ls5": "R3_daily_trend8_log5",
            "L3_daily_trend8_ls3_drift0": "R7_daily_trend8_drift0"}
    out = _clean(evaluate(a.names, a.reps, compare={k: v for k, v in comp.items() if k in a.names}))
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    key = f"{PRIMARY[0]}|{PRIMARY[1]}"

    def line(n, s):
        return (f"{n:28s} 샤프 {s['sharpe']:5.2f} CAGR {s['cagr']*100:6.1f}% MDD {s['max_dd']*100:6.1f}% "
                f"순노출 {s['net_exposure']*100:4.0f}% 숏시간 {s['time_short']*100:3.0f}% "
                f"롱손익 {s['long_logr']:+.2f} 숏손익 {s['short_logr']:+.2f} 수수료 {s['fees']:.2f} "
                f"2025-26 {s['eras'].get('2025-26', {}).get('sharpe', float('nan')):+.2f}"
                + ("  [청산]" if s["liquidated"] else ""))
    print(f"— 기본 시나리오: 편도 {PRIMARY[0]*100:.2f}%, 숏 유지비 {PRIMARY[1]*100:.0f}%/년 —")
    for k, v in out["rules"].items():
        print(line(k, v[key]))
    for n, v in out["variants"].items():
        print(line(n, v["scen"][key]), f"(반복 {min(v['sharpe_reps']):.2f}~{max(v['sharpe_reps']):.2f})")
    print("— 숏 유지비·수수료 민감도 (샤프) —")
    for k in list(out["rules"]) + list(out["variants"]):
        src = out["rules"].get(k) or out["variants"][k]["scen"]
        print(f"{k:28s} " + " ".join(f"{c*100:.2f}%/{cr*100:+.0f}%:{src[f'{c}|{cr}']['sharpe']:5.2f}"
                                       for c in COSTS for cr in CARRY))
    print("— 검정 (정상 부트스트랩, ΔSharpe) —")
    for t in out["tests"]:
        print(f"{t['a']:28s} vs {t['b']:24s} {t['d_sharpe']:+.2f} p={t['p']:.2f}")
    print("N =", out["n_trials"], "DSR:", {k: round(v, 2) for k, v in out["dsr_excess"].items()})


if __name__ == "__main__":
    main()
