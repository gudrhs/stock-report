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
from ..data import DATA_DIR
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


FUNDING = os.path.join(DATA_DIR, "funding", "btc_perp_funding_8h.csv")


def funding_annual(ts, first="binance_BTCUSDT"):
    """
    봉(시작 시각 ts)마다 무기한 선물 펀딩비의 연율 (+ = 롱이 숏에게 냄). 그 봉이 속한 8시간 정산 구간의 비율을 씁니다.
    first 거래소를 먼저 쓰고 없는 구간은 다른 거래소로 채움, 둘 다 없으면 0 (2026-08-06 이후).
    출처·검증: data/btc/funding/README.md
    """
    import pandas as pd
    f = pd.read_csv(FUNDING)
    out = np.full(len(ts), np.nan)
    for src in [first] + [x for x in f["source"].unique() if x != first]:
        g = f[f["source"] == src].sort_values("ts_utc_seconds")
        ft, fr = g["ts_utc_seconds"].to_numpy(np.int64), g["funding_rate_per_8h"].to_numpy(float)
        k = np.searchsorted(ft, ts, side="right")                        # 봉 시작 이후 첫 정산
        ok = (k < len(ft))
        kk = np.minimum(k, len(ft) - 1)
        ok &= (ft[kk] - ts) <= 8 * 3600
        val = np.where(ok, fr[kk] * 3 * 365.25, np.nan)
        out = np.where(np.isnan(out), val, out)
    return np.nan_to_num(out, nan=0.0)


def hold_changes(tg):
    """다른 보유 방식: 목표가 이전 판단과 같으면 NaN(수량 그대로) — 숏을 매일 −1배로 다시 맞추지 않음"""
    out = tg.copy()
    prev = np.nan
    for i in np.where(~np.isnan(tg))[0]:
        if tg[i] == prev:
            out[i] = np.nan
        prev = tg[i]
    return out


def simulate_ls(weights, o, c, cost, start, end, carry=0.0, band=0.02, forced_hold=None, exit_cost=True,
                carry_long=0.0):
    """
    weights: 봉별 목표 비중 (NaN = 유지), −1 ~ 1.  carry: 숏 유지비 연율 (실수 또는 봉별 배열)
    carry_long: 롱 유지비 연율 (롱도 무기한 선물로 할 때의 펀딩비; 현물 롱이면 0)
    반환: pos(체결 후 비중), logr(봉별 로그 자산변화), mark(종가 기준 자산), rebal, liquidated,
          fees·carried(봉별 수수료·유지비, 그때 자산 대비 비율 — 합하면 대략 로그수익 손실)
    """
    T = len(o)
    if end >= T:
        raise ValueError("end 봉이 있어야 합니다")
    carry_bar = np.broadcast_to(np.asarray(carry, dtype=float), (T,)) / BARS_PER_YEAR
    carry_long_bar = np.broadcast_to(np.asarray(carry_long, dtype=float), (T,)) / BARS_PER_YEAR
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
        elif units > 0 and carry_long_bar[j] != 0.0:                      # 롱을 선물로 할 때의 펀딩비
            ch = carry_long_bar[j] * units * o[j]
            carried[t] = ch / (cash + units * o[j])
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


def run_ls(W, targets, cost, carry=0.0, phase=0, carry_long=0.0):
    """Window.run과 같은 일별 수익 (잠금 구간 이후 가격 미사용)"""
    d = W.datas[phase]
    a, b = W.rng[phase]
    sim = simulate_ls(targets, d.o, d.c, cost, a, b, carry=carry, forced_hold=d.forced_hold, carry_long=carry_long)
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
    out["S0_always_short"] = daily_hold(short, mask, a, b)              # 다른 규칙처럼 하루 한 번만 다시 맞춤
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


def scenarios(d):
    """
    이름 → (편도 비용, 숏 유지비 연율, 롱 유지비 연율, 보유 방식). 보유 방식: 'retarget'(매일 목표 비중으로 다시 맞춤)
    또는 'hold'(목표가 바뀔 때까지 수량 그대로). 기본(헤드라인)은 'base' — 둘 다 보고합니다.
    """
    fb = funding_annual(d.ts, "binance_BTCUSDT")
    fm = funding_annual(d.ts, "bitmex_XBTUSD")
    sc = {
        "base": (0.0015, 0.0, 0.0, "retarget"),
        "base_hold": (0.0015, 0.0, 0.0, "hold"),
        "fund": (0.0015, -fb, 0.0, "retarget"),                  # 실제 펀딩비(바이낸스 우선): 숏이 받음/냄
        "fund_hold": (0.0015, -fb, 0.0, "hold"),
        "fund_bitmex_hold": (0.0015, -fm, 0.0, "hold"),
        "perp_both_hold": (0.0005, -fb, fb, "hold"),             # 롱·숏 모두 선물(테이커 0.05%), 롱은 펀딩을 냄
        "korea_lend_hold": (0.0015, 0.11, 0.0, "hold"),          # 국내 코인 빌리기(업비트 연 약 11%)
    }
    for cost in COSTS:
        for carry in CARRY:
            sc[f"{cost}|{carry}"] = (cost, carry, 0.0, "retarget")
    return sc


def evaluate(names, reps=5, window=("2017-01-01", "2026-09-25"), compare=None):
    datas = load_phases()
    W = Window(datas, window[0], window[1], "research-ls:" + ",".join(names or ["rules"]))
    d = W.datas[0]
    rules = rule_targets(W)
    for k in rules:                                                       # 규칙도 시험으로 셈 (B0 제외)
        if k != "B0":
            log_trial(dict(name=f"LSRULE_{k}", rule=k, daily=True))
    SC = scenarios(d)
    out = dict(window=window, scenarios={k: dict(cost=v[0], carry=(v[1] if np.isscalar(v[1]) else "funding"),
                                                 carry_long=(v[2] if np.isscalar(v[2]) else "funding"), conv=v[3])
                                         for k, v in SC.items()}, rules={}, variants={})

    def run_sc(tg, key):
        cost, cs, cl, conv = SC[key]
        return run_ls(W, hold_changes(tg) if conv == "hold" else tg, cost, cs, carry_long=cl)

    prim, prim_hold = {}, {}
    for k, tg in rules.items():
        out["rules"][k] = {}
        for key in SC:
            res = run_sc(tg, key)
            out["rules"][k][key] = describe(res, res["days"])
            if key == "base":
                prim[k] = res
            if key == "base_hold":
                prim_hold[k] = res
    heads, heads_hold = {}, {}
    for name in names:
        tgs = [variant_targets(W, name, r, PRIMARY[0]) for r in range(reps)]
        prim_res = [run_sc(tg, "base") for tg, _ in tgs]
        srs = [S.sharpe(x["r"]) for x in prim_res]
        lm = lower_median([s - S.sharpe(prim["B0"]["r"]) for s in srs])
        heads[name] = prim_res[lm]
        heads_hold[name] = run_sc(tgs[lm][0], "base_hold")
        v = dict(cfg=VARIANTS[name], headline_rep=lm, sharpe_reps=srs,
                 sharpe_reps_hold=[S.sharpe(run_sc(tg, "base_hold")["r"]) for tg, _ in tgs], scen={},
                 rejected_gates=[(e["month"], e.get("kind")) for e in tgs[lm][1] if e.get("accepted") is False])
        tg_by_cost = {}
        for key, (cost, cs, cl, conv) in SC.items():
            if cost not in tg_by_cost:
                tg_by_cost[cost] = variant_targets(W, name, lm, cost)[0]      # 판단 비용은 실제 비용에 맞춤
            tg = tg_by_cost[cost]
            res = run_ls(W, hold_changes(tg) if conv == "hold" else tg, cost, cs, carry_long=cl)
            v["scen"][key] = describe(res, res["days"])
        out["variants"][name] = v
    # 검정: 두 보유 방식 모두 — 매수·보유, 같은 규칙의 롱·현금판, (변형이면) 200일선 롱·현금
    out["tests"] = []
    for conv, P, H in (("retarget", prim, heads), ("hold", prim_hold, heads_hold)):
        allr = {**{k: v["r"] for k, v in P.items()}, **{n: h["r"] for n, h in H.items()}}
        pairs = [(f"{k}_LS", "B0") for k in RULES] + [(f"{k}_LS", f"{k}_LF") for k in RULES] + \
                [(n, "B0") for n in names] + [(n, "B2_LF") for n in names] + [(n, "B5_LF") for n in names]
        for x, y in pairs:
            bt = S.stationary_bootstrap_diff(allr[x], allr[y], n_boot=4000, seed=1)
            out["tests"].append(dict(conv=conv, a=x, b=y, d_sharpe=bt["obs"], p=bt["p"], ci90=bt["ci90"]))
    if compare:
        from .walk import run_path
        from .evaluate import variant_results
        for n, base in compare.items():
            if base in VARIANTS and all(os.path.exists(run_path(base, r)) for r in range(reps)):
                cd = min(C_DECS, key=lambda x: abs(x - 2.0 * PRIMARY[0]))
                rr = variant_results(W, base, reps, cd)
                lmb = lower_median([S.sharpe(x["r"]) - S.sharpe(prim["B0"]["r"]) for x in rr])
                for conv, H in (("retarget", heads), ("hold", heads_hold)):
                    bt = S.stationary_bootstrap_diff(H[n]["r"], rr[lmb]["r"], n_boot=4000, seed=1)
                    out["tests"].append(dict(conv=conv, a=n, b=base, d_sharpe=bt["obs"], p=bt["p"], ci90=bt["ci90"]))
    # 시험 수로 부풀림 걷어내기 (매수·보유 대비 초과수익). 시험 간 분산은 핵심 평가의 실제 강화학습 시험 24개에서 온
    # 값(report_full.json)을 씁니다 — 규칙 롱·현금/롱·숏 쌍은 서로 거의 같은 신호라 여기서 추정하면 너무 작아집니다.
    N = n_trials_total()
    srv = None
    try:
        with open(os.path.join(DATA_DIR, "report_full.json"), encoding="utf-8") as f:
            srv = float(json.load(f)["dsr"]["sr_var_excess"])
    except Exception:
        pass
    allr = {**{k: v["r"] for k, v in prim.items()}, **{n: h["r"] for n, h in heads.items()}}
    keys = [k for k in allr if k not in ("B0", "S0_always_short")]
    if srv is None:
        srv = max(S.trials_sr_var(np.column_stack([allr[k] - allr["B0"] for k in keys])), 1e-6)
    out["n_trials"], out["sr_var_excess"] = N, srv
    out["dsr_excess"] = {k: S.dsr(allr[k] - allr["B0"], N, srv) for k in keys}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--print-only", action="store_true", help="저장된 결과만 다시 출력")
    a = ap.parse_args()
    from ..evaluate import _clean
    comp = {"L1_daily_trend8_ls3": "R2_daily_trend8", "L2_daily_trend8_ls5": "R3_daily_trend8_log5",
            "L3_daily_trend8_ls3_drift0": "R7_daily_trend8_drift0"}
    if a.print_only:
        with open(a.out, encoding="utf-8") as f:
            out = json.load(f)
    else:
        out = _clean(evaluate(a.names, a.reps, compare={k: v for k, v in comp.items() if k in a.names}))
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)

    def line(n, s):
        s = {k: (float("nan") if v is None else v) for k, v in s.items()}       # 청산(−∞) 등은 NaN으로 표시
        return (f"{n:28s} 샤프 {s['sharpe']:5.2f} CAGR {s['cagr']*100:6.1f}% MDD {s['max_dd']*100:6.1f}% "
                f"순노출 {s['net_exposure']*100:4.0f}% 숏시간 {s['time_short']*100:3.0f}% "
                f"롱손익 {s['long_logr']:+.2f} 숏손익 {s['short_logr']:+.2f} 수수료 {s['fees']:.2f} 유지비 {s['carried']:+.2f} "
                f"2025-26 {s['eras'].get('2025-26', {}).get('sharpe', float('nan')):+.2f}"
                + ("  [청산]" if s["liquidated"] else ""))
    allk = list(out["rules"]) + list(out["variants"])
    src = lambda k: out["rules"].get(k) or out["variants"][k]["scen"]
    for key, title in (("base", "매일 다시 맞춤, 편도 0.15%, 숏 유지비 0"), ("base_hold", "신호가 바뀔 때까지 수량 유지, 편도 0.15%, 숏 유지비 0"),
                       ("fund_hold", "수량 유지 + 실제 펀딩비(바이낸스 우선)")):
        print(f"— {title} —")
        for k in allk:
            extra = ""
            if k in out["variants"]:
                rr = out["variants"][k]["sharpe_reps" if key == "base" else "sharpe_reps_hold"]
                extra = f" (반복 {min(rr):.2f}~{max(rr):.2f})"
            print(line(k, src(k)[key]) + extra)
    print("— 시나리오별 샤프 —")
    keys = ["base", "base_hold", "fund", "fund_hold", "fund_bitmex_hold", "perp_both_hold", "korea_lend_hold"]
    print(f"{'':28s} " + " ".join(f"{k:>16s}" for k in keys))
    for k in allk:
        print(f"{k:28s} " + " ".join(f"{src(k)[x]['sharpe']:16.2f}" if src(k)[x]['sharpe'] is not None else f"{'청산':>16s}"
                                       for x in keys))
    print("— 검정 (정상 부트스트랩, ΔSharpe) —")
    for t in out["tests"]:
        print(f"[{t['conv']:8s}] {t['a']:28s} vs {t['b']:24s} {t['d_sharpe']:+.2f} p={t['p']:.2f}")
    print("N =", out["n_trials"], "시험 간 분산", f"{out['sr_var_excess']:.2e}", "DSR:",
          {k: round(v, 2) for k, v in out["dsr_excess"].items()})


if __name__ == "__main__":
    main()
