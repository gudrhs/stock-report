# -*- coding: utf-8 -*-
"""
비트코인 전용 2단계 — 견고성 검사 B1~B4 (btc/research/success_criteria.md, 2026-09-25 11:06 UTC 조항 그대로).

  B1 시드 견고성   : 반복 10개(0~9)의 작은 쪽 중앙값(5번째) 샤프 > 매수·보유, 10개 중 9개 이상 > 매수·보유
  B2 판단 시각     : 같은 모델로 00·04·08·12·16·20시 UTC 마감 봉에서 각각 판단 (반복 0~4, <이름>__allbars 실행).
                     시각마다 작은 쪽 중앙값 샤프 → 6개 평균 > 매수·보유, 6개 중 5개 이상 > 매수·보유.
                     00시 판단값은 기존 실행과 비트 단위로 같아야 함 (다르면 구현 오류로 중단)
  B3 기간 견고성   : 2017-01부터 한 달씩 옮기는 2년 창에서 헤드라인 반복 샤프 > 매수·보유 인 창 ≥ 60%
  B4 합성 대조     : btc/research/controls.json · controls_weights.json (양성 통과, 음성 통과)
통과·실패와 무관하게 p값(매수·보유 대비), 200일선·일봉 MACD 대비 차이, 전체 N으로 계산한 DSR을 함께 보고합니다.

  BTC_LOCKBOX_OPEN=1 python -m btc.research.robust R6_daily_trend8_uniform R3_daily_trend8_log5
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import json

import numpy as np

from .. import stats as S
from ..env import BAR_SEC
from ..evaluate import Window, baseline_targets, lower_median
from ..walkforward import load_phases
from ..data import DATA_DIR
from .rl import k_policy, trend_filter, allowed_matrix
from .walk import load_run, decision_mask, n_trials_total, RUNS, run_path
from .evaluate import daily_hold
from .variants import VARIANTS

COST = 0.0015
C_DEC = 0.003
OUT = os.path.join(RUNS, "robust.json")
CONTROLS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "controls.json")
HOURS = (0, 4, 8, 12, 16, 20)


def clock_mask(d, hour):
    """phase 0에서 hour시 UTC에 마감하는 봉 (hour=0이면 walk.decision_mask(d, 6)과 같음)"""
    close = d.ts + BAR_SEC
    off = int(d.ts[0] % BAR_SEC)
    return (close % 86400) == (off + hour * 3600) % 86400


def targets(W, cfg, run, mask):
    """판단봉 mask에서의 목표 비중 (btc.research.screen.targets 와 같은 계산, 판단비용 0.3%)"""
    d = W.datas[0]
    a, b = W.rng[0]
    sel = np.arange(a, b)[mask[a:b]]
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
    frac = bool(np.any((acts > 0) & (acts < 1))) or bool(np.any(acts < 0))
    return tg, frac, float(nan.mean())


def dp_clock_targets(W, log, hour):
    """
    상태가 있는 동적계획 띠 모델(W2)은 '모든 봉' 출력이 4시간마다 판단하는 다른 전략이 되므로,
    월별 기록(예측 계수·띠 경계)으로 hour시 판단봉의 p를 다시 계산하고 그 시각만의 이전 포지션을 이어 갑니다.
    hour=0이면 기본 실행과 같아야 합니다 (robust()에서 확인).
    """
    import pandas as pd
    from ..walkforward import decision_range, OOS_END
    from .algos.dp_band import band_path, parse_q
    d = W.datas[0]
    a0, b0 = W.rng[0]
    mask = clock_mask(d, hour)
    tg = np.full(d.T, np.nan)
    pos = 0.0
    for i, e in enumerate(log):
        if "ridge_beta" not in e:
            continue
        Tk = int(pd.Timestamp(e["month"], tz="UTC").timestamp())
        nxt = int(pd.Timestamp(log[i + 1]["month"], tz="UTC").timestamp()) if i + 1 < len(log) else int(OOS_END.timestamp())
        a, b = decision_range(d, Tk, nxt)
        sel = np.arange(a, b)[mask[a:b]]
        if len(sel) == 0:
            continue
        Z = (np.asarray(d.X[sel], np.float64)[:, np.asarray(e["fi"])] - np.asarray(e["feat_mu"])) / np.asarray(e["feat_sd"])
        pv = float(e["c0"]) + Z @ np.asarray(e["ridge_beta"])
        q_in, q_out = parse_q(e["q_in"]), parse_q(e["q_out"])
        if e.get("policy_monotone", True):
            w = band_path(pv, q_in, q_out, pos)
        else:
            g = np.linspace(e["grid_lo"], e["grid_hi"], e["grid_n"])
            pi = np.array([[int(ch) for ch in col] for col in e["pi_str"]], np.int8).T
            w = band_path(pv, q_in, q_out, pos, g, pi)
        pos = float(w[-1])
        tg[sel] = w
    out = np.full(d.T, np.nan)
    out[a0:b0] = tg[a0:b0]
    return out


def run_tg(W, tg, frac):
    return W.run(tg, COST, weights=True) if frac else W.run(tg, COST)


def rolling_windows(days, r, rb, years=2, step_days=30):
    """2년 창을 한 달씩 옮기며 (시작일, 전략 샤프, 매수·보유 샤프)"""
    n = int(round(365.25 * years))
    out = []
    for s in range(0, len(r) - n + 1, step_days):
        out.append((int(days[s]), S.sharpe(r[s:s + n]), S.sharpe(rb[s:s + n])))
    return out


def robust(names):
    datas = load_phases()
    W = Window(datas, "2017-01-01", "2026-09-25", "research-robust:" + ",".join(names))
    d = W.datas[0]
    a, b = W.rng[0]
    bt = baseline_targets(W)
    bh = W.run(bt["B0"], COST)
    sb = S.sharpe(bh["r"])
    m0 = decision_mask(d, 6)
    rules = {k: W.run(daily_hold(bt[k], m0, a, b), COST) for k in ("B2", "B5")}
    srv = None
    try:
        with open(os.path.join(DATA_DIR, "report_full.json"), encoding="utf-8") as f:
            srv = float(json.load(f)["dsr"]["sr_var_excess"])
    except Exception:
        srv = 1.5e-4
    N = n_trials_total()
    controls = {}
    for path in (CONTROLS, CONTROLS.replace("controls.json", "controls_weights.json")):   # DQN 계열 / 비중 출력 기법
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                controls.update(json.load(f))
    out = dict(bh_sharpe=sb, n_trials=N, sr_var_excess=srv, rules={k: S.sharpe(v["r"]) for k, v in rules.items()},
               variants={})
    for name in names:
        cfg = VARIANTS[name]
        v = dict()
        # ── B1: 반복 10개 ──
        reps = [r for r in range(10) if os.path.exists(run_path(name, r))]
        res10, runs10 = [], {}
        for r in reps:
            runs10[r] = load_run(name, r)
            tg, frac, _ = targets(W, cfg, runs10[r], m0)
            res10.append(run_tg(W, tg, frac))
        s10 = [S.sharpe(x["r"]) for x in res10]
        lm10 = lower_median([s - sb for s in s10]) if s10 else None
        v["B1"] = dict(reps=reps, sharpe_reps=s10, lower_median=s10[lm10] if s10 else None,
                       n_above=int(sum(s > sb for s in s10)),
                       passed=bool(len(s10) == 10 and s10[lm10] > sb and sum(s > sb for s in s10) >= 9))
        # ── 헤드라인(반복 0~4, 1단계와 같은 정의) ──
        s5 = s10[:5] if reps[:5] == list(range(5)) else []
        lm5 = lower_median([s - sb for s in s5]) if len(s5) == 5 else None
        head = res10[lm5] if lm5 is not None else None
        # ── B2: 판단 시각 6가지 ──
        allname = name + "__allbars"
        have_all = all(os.path.exists(run_path(allname, r)) for r in range(5))
        stateful = cfg.get("algo") == "dp_band"
        if have_all or stateful:
            per_hour = {}
            sel0 = np.arange(a, b)[m0[a:b]]
            for r in range(5):
                if stateful:
                    # 00시 재구성이 저장된 기본 실행과 같아야 함
                    t0 = dp_clock_targets(W, runs10[r]["log"], 0)
                    base = runs10[r]["U"][0.0][:, 0]
                    if not np.allclose(t0[sel0], base[sel0], equal_nan=True, atol=0):
                        raise SystemExit(f"{name} rep{r}: 00시 재구성이 기본 실행과 다릅니다 — 구현 오류")
                    for h in HOURS:
                        tg = dp_clock_targets(W, runs10[r]["log"], h)
                        per_hour.setdefault(h, []).append((S.sharpe(run_tg(W, tg, True)["r"]),
                                                           float(np.isnan(tg[a:b][clock_mask(d, h)[a:b]]).mean())))
                    continue
                ra = load_run(allname, r)
                # 00시 판단: 포지션이 기존 실행과 같아야 하고 값 차이 < 1e-5 (행 묶음 크기에 따른 부동소수 차이만 허용)
                key = 0.0 if (cfg.get("algo") == "direct" or cfg.get("output") == "weights") else C_DEC
                ua, ub = ra["U"][key][sel0], runs10[r]["U"][key][sel0]
                if not np.array_equal(np.isnan(ua), np.isnan(ub)) or np.nanmax(np.abs(ua - ub)) >= 1e-5:
                    raise SystemExit(f"{name} rep{r}: 00시 판단값이 기존 실행과 다릅니다 — 구현 오류")
                ta, _, _ = targets(W, cfg, ra, m0)
                tb, _, _ = targets(W, cfg, runs10[r], m0)
                if not np.array_equal(ta, tb, equal_nan=True):
                    raise SystemExit(f"{name} rep{r}: 00시 포지션이 기존 실행과 다릅니다 — 구현 오류")
                for h in HOURS:
                    tg, frac, nan_share = targets(W, cfg, ra, clock_mask(d, h))
                    per_hour.setdefault(h, []).append((S.sharpe(run_tg(W, tg, frac)["r"]), nan_share))
            hs = {}
            for h, lst in per_hour.items():
                srs = [x[0] for x in lst]
                hs[h] = dict(sharpe_reps=srs, lower_median=srs[lower_median([s - sb for s in srs])],
                             no_model_share=max(x[1] for x in lst))
            lmv = [hs[h]["lower_median"] for h in HOURS]
            v["B2"] = dict(hours=hs, mean=float(np.mean(lmv)), n_above=int(sum(x > sb for x in lmv)),
                           passed=bool(np.mean(lmv) > sb and sum(x > sb for x in lmv) >= 5))
        else:
            v["B2"] = dict(passed=None, note=f"{allname} 실행(반복 0~4)이 아직 없습니다")
        # ── B3: 2년 이동 창 ──
        if head is not None:
            rw = rolling_windows(head["days"], head["r"], bh["r"])
            share = float(np.mean([x[1] > x[2] for x in rw]))
            v["B3"] = dict(n_windows=len(rw), share_above=share, passed=bool(share >= 0.60),
                           worst=min((x[1] - x[2], x[0]) for x in rw))
            boot = S.stationary_bootstrap_diff(head["r"], bh["r"], n_boot=4000, seed=1)
            v["headline"] = dict(rep=lm5, sharpe=S.sharpe(head["r"]), p_vs_bh=boot["p"], d_vs_bh=boot["obs"],
                                 d_vs_B2=S.sharpe(head["r"]) - S.sharpe(rules["B2"]["r"]),
                                 d_vs_B5=S.sharpe(head["r"]) - S.sharpe(rules["B5"]["r"]),
                                 dsr_excess=S.dsr(head["r"] - bh["r"], N, srv))
        # ── B4: 합성 대조 ──
        c = controls.get(name)
        v["B4"] = dict(passed=(bool(c.get("positive_pass")) and bool(c.get("negative_pass"))) if c else None,
                       positive=c.get("positive_pass") if c else None, negative=c.get("negative_pass") if c else None,
                       neg_mean_d=c.get("neg_mean_d") if c else None)
        checks = [v[k]["passed"] for k in ("B1", "B2", "B3", "B4")]
        v["passed"] = None if any(x is None for x in checks) else all(checks)
        out["variants"][name] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    a = ap.parse_args()
    from ..evaluate import _clean
    out = _clean(robust(a.names))
    prev = {}
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            prev = json.load(f)
    prev.setdefault("variants", {}).update(out["variants"])
    for k in ("bh_sharpe", "n_trials", "sr_var_excess", "rules"):
        prev[k] = out[k]
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(prev, f, ensure_ascii=False, indent=1)
    sb = out["bh_sharpe"]
    ok = lambda x: "통과" if x is True else ("탈락" if x is False else "미완")
    print(f"매수·보유 샤프 {sb:.2f} | 200일선 {out['rules']['B2']:.2f} | 일봉 MACD {out['rules']['B5']:.2f} | N={out['n_trials']}")
    for n, v in out["variants"].items():
        print(f"\n{n}: 종합 {ok(v['passed'])}")
        b1 = v["B1"]
        print(f"  B1 시드 {ok(b1['passed'])}: 반복 {len(b1['sharpe_reps'])}개, 작은 쪽 중앙값 "
              f"{(b1['lower_median'] or float('nan')):.2f}, 매수·보유보다 높은 반복 {b1['n_above']}개 "
              f"[{', '.join(f'{s:.2f}' for s in b1['sharpe_reps'])}]")
        b2 = v["B2"]
        if b2.get("hours"):
            hs = ", ".join(f"{h}시 {b2['hours'][str(h)]['lower_median']:.2f}" if str(h) in b2["hours"]
                           else f"{h}시 {b2['hours'][h]['lower_median']:.2f}" for h in HOURS)
            print(f"  B2 판단 시각 {ok(b2['passed'])}: 평균 {b2['mean']:.2f}, 매수·보유보다 높은 시각 {b2['n_above']}/6 ({hs})")
        else:
            print(f"  B2 판단 시각 미완: {b2.get('note')}")
        if v.get("B3"):
            print(f"  B3 2년 창 {ok(v['B3']['passed'])}: {v['B3']['n_windows']}개 중 {v['B3']['share_above']*100:.0f}%에서 매수·보유보다 높음")
        b4 = v["B4"]
        print(f"  B4 합성 대조 {ok(b4['passed'])}: 양성 {b4['positive']}, 음성 {b4['negative']}")
        if v.get("headline"):
            h = v["headline"]
            print(f"  참고: 헤드라인 샤프 {h['sharpe']:.2f}, 매수·보유 대비 {h['d_vs_bh']:+.2f} (p={h['p_vs_bh']:.2f}), "
                  f"200일선 대비 {h['d_vs_B2']:+.2f}, 일봉 MACD 대비 {h['d_vs_B5']:+.2f}, DSR {h['dsr_excess']:.2f}")


if __name__ == "__main__":
    main()
