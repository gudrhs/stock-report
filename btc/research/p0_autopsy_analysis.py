# -*- coding: utf-8 -*-
"""
P0 부검 — 저장된 10개 반복(data/btc/runs/P0/rep*.npz)과 게이트 반사실 재현(p0_autopsy_replay) 분석.

  (1) Δ = U1−U0 (c_dec 0.3%) 대 ±0.30 문턱 — 연도별 분포, 문턱 근처 빈도, 보정(실현 할인수익 G와 비교)
  (2) 게이트 탈락 전부와 그 결과 (V0 shadow Δ, V1~V3 반사실)
  (3) 2025~26: 거부된 1월 새 모델(2025·2026) — 그대로 썼다면 / 채택 후 이어학습했다면
  (4) Δ가 무엇과 상관있나 — 연도별 22지표 회귀, C>SMA1200 등 규칙과의 일치
  (5) 낙폭 감소는 어느 구간에서 왔나 — 매수·보유 낙폭 구간별 분해

  python -m btc.research.p0_autopsy_analysis            (재현 결과가 있으면 (2)(3)도 계산)

사후 연구(잠금 구간 포함)입니다. 결과: btc/research/out/p0_autopsy/analysis.json
"""
import json
import math
import os

import numpy as np
import pandas as pd

from btc.research.p0_autopsy_common import (
    OUT, COST, CD, TH, LO, HI, REPS, FI, NAMES, S, ts, load_phases, load_run, window_range, run_targets,
    agent_targets, bh_targets, b2_targets, sub_stats, fwd_discounted, policy_from_delta, decision_range, BAR_SEC)

GAMMA = 0.97
KAPPA = 100.0
NEAR = 0.05          # |Δ ∓ 0.30| < 0.05 → '문턱 근처'


def year_of(d, idx):
    return pd.to_datetime(d.ts[idx] + BAR_SEC, unit="s", utc=True).year.to_numpy()


def med(xs):
    return float(np.nanmedian(xs))


# ══════════ (1) Δ 대 문턱 ══════════
def part1(d0, runs, a, b):
    yrs = year_of(d0, np.arange(a, b))
    G = fwd_discounted(d0.m, GAMMA, KAPPA, 200)[a:b]
    km = KAPPA * np.nan_to_num(d0.m[a:b])
    rows = []
    per_rep = {}
    for y in sorted(set(yrs)):
        sel = yrs == y
        stats = []
        for r, run in enumerate(runs):
            dl = run["delta"][0][CD][a:b].astype(float)
            pos = policy_from_delta(dl, CD, d0.forced_hold[a:b])
            x = dl[sel]
            p = pos[sel]
            prev = np.concatenate([[pos[np.argmax(sel) - 1] if np.argmax(sel) > 0 else 0], p[:-1]])
            ok = ~np.isnan(G[sel])
            stats.append(dict(
                mean=np.nanmean(x), p05=np.nanpercentile(x, 5), p50=np.nanmedian(x), p95=np.nanpercentile(x, 95),
                min=np.nanmin(x), above=np.mean(x > TH), below=np.mean(x < -TH),
                band=np.mean(np.abs(x) <= TH),
                near_buy=np.mean(np.abs(x - TH) < NEAR), near_sell=np.mean(np.abs(x + TH) < NEAR),
                near_any=np.mean((np.abs(x - TH) < NEAR) | (np.abs(x + TH) < NEAR)),
                exposure=p.mean(), sells=int(((prev == 1) & (p == 0)).sum()), buys=int(((prev == 0) & (p == 1)).sum()),
                margin_long=np.nanmedian((x + TH)[p == 1]) if (p == 1).any() else np.nan,
                ic_G=np.corrcoef(x[ok], G[sel][ok])[0, 1] if ok.sum() > 50 else np.nan,
                ic_m=np.corrcoef(x, km[sel])[0, 1]))
            per_rep.setdefault(r, {})[int(y)] = stats[-1]
        row = dict(year=int(y), n=int(sel.sum()))
        for k in stats[0]:
            v = [s[k] for s in stats]
            row[k] = med(v)
            if k in ("mean", "exposure", "below", "near_sell"):
                row[k + "_lo"] = float(np.nanmin(v))
                row[k + "_hi"] = float(np.nanmax(v))
        row["G_mean"] = float(np.nanmean(G[sel]))           # 실현: 계속 보유 − 계속 현금 (γ 할인, κ=100)
        row["G_p05"] = float(np.nanpercentile(G[sel], 5)) if np.isfinite(G[sel]).any() else None
        row["km_mean_x33"] = float(km[sel].mean() / (1 - GAMMA))
        rows.append(row)
    return rows


def pool_drift(d0, years):
    """1월 학습 표본의 (최근 가중 70% + 균등 30%) 평균 κ·m — '무조건 기울기'만 배웠을 때의 Δ 크기를 가늠"""
    out = {}
    for y in years:
        T_k = ts(f"{y}-01-01")
        t = d0.eligible(T_k, ts("2014-01-01"), 2400)
        age = (T_k - d0.ts[t]) / (2.0 * 365.25 * 86400)
        wr = np.power(2.0, -age)
        wr /= wr.sum()
        w = 0.7 * wr + 0.3 / len(t)
        km = KAPPA * d0.m[t]
        mu = float((w * km).sum() / w.sum())
        # 최근 2년만
        r2 = t[d0.ts[t] >= T_k - 2 * 365 * 86400]
        out[int(y)] = dict(pool_km=mu, pool_km_x33=mu / (1 - GAMMA),
                           trail2y_km_x33=float(KAPPA * d0.m[r2].mean() / (1 - GAMMA)))
    return out


# ══════════ (2)(3) 게이트 ══════════
def gate_table(runs):
    rows = []
    for r, run in enumerate(runs):
        for e in run["log"]:
            if not e.get("accepted", True):
                rows.append(dict(rep=r, month=e["month"], kind=e["kind"],
                                 exposure=e.get("exposure"), switches=e.get("switches_per_year"),
                                 disagree=e.get("disagree")))
    return rows


def have_replays(vs=("V0", "V1", "V2", "V3")):
    from btc.research.p0_autopsy_replay import path
    return {v: [r for r in range(REPS) if os.path.exists(path(v, r))] for v in vs}


def perf_block(d0, delta, lo=LO, hi=HI, subs=None):
    res = run_targets(d0, agent_targets(d0, delta), COST, lo, hi)
    out = dict(full=sub_stats(res["days"], res["r"], lo, hi), exposure=float(res["pos"].mean()))
    for name, (l, h) in (subs or {}).items():
        out[name] = sub_stats(res["days"], res["r"], l, h)
        sel = (d0.ts[res["a"]:res["b"]] + BAR_SEC >= ts(l)) & (d0.ts[res["a"]:res["b"]] + BAR_SEC < ts(h))
        out[name]["exposure"] = float(res["pos"][sel].mean())
    return out, res


SUBS = {"2017-2024": ("2017-01-01", "2025-01-01"), "2018": ("2018-01-01", "2019-01-01"),
        "2022": ("2022-01-01", "2023-01-01"), "lockbox": ("2025-01-01", "2026-09-25")}


def part2_3(d0, runs):
    from btc.research.p0_autopsy_replay import load_replay
    avail = have_replays()
    out = dict(avail=avail)
    # 재현성
    out["repro"] = {r: bool(load_replay("V0", r)["repro_exact"][0]) for r in avail["V0"]}
    # 거부된 후보가 '그 달'에 냈을 판단 (V0 shadow): 실제 보유와 후보 보유 비교
    one_month = []
    for r in avail["V0"]:
        z = load_replay("V0", r)
        act = z["delta"].astype(float)
        a, b = window_range(d0, LO, HI)
        pos_act = policy_from_delta(act[a:b], CD, d0.forced_hold[a:b])
        for e in z["log"]:
            if e["accepted"]:
                continue
            T0 = ts(e["month"])
            T1 = ts((pd.Timestamp(e["month"]) + pd.offsets.MonthBegin(1)).strftime("%Y-%m-%d"))
            if e["kind"] == "cold":
                T1 = ts(f"{int(e['month'][:4]) + 1}-01-01")    # 1월 모델은 1년을 맡음 → 그해 전체를 frozen으로
                key = f"frozen_{e['month'][:4]}"
                sh = z[key].astype(float) if key in z else None
            else:
                sh = z["shadow"].astype(float)
            if sh is None:
                continue
            i0, i1 = decision_range(d0, T0, min(T1, ts(HI)))
            i1 = min(i1, b)
            # 후보로 갈아탔다면: 진입 직전 실제 보유에서 시작해 후보 Δ로 판단
            p0 = int(pos_act[i0 - a - 1]) if i0 - a - 1 >= 0 else 0
            pc = policy_from_delta(sh[i0:i1], CD, d0.forced_hold[i0:i1], p0=p0)
            pa = pos_act[i0 - a:i1 - a]
            m = np.nan_to_num(d0.m[i0:i1])
            lnc = math.log(1 - COST)
            def lr(p, pprev):
                return float((p * m).sum() + np.abs(np.diff(np.concatenate([[pprev], p]))).sum() * lnc)
            one_month.append(dict(rep=r, month=e["month"], kind=e["kind"],
                                  exp_actual=float(pa.mean()), exp_candidate=float(pc.mean()),
                                  ret_actual=math.expm1(lr(pa, p0)), ret_candidate=math.expm1(lr(pc, p0)),
                                  ret_bh=math.expm1(float(m.sum())), bars=int(i1 - i0),
                                  min_delta_candidate=float(np.nanmin(sh[i0:i1])),
                                  min_delta_actual=float(np.nanmin(act[i0:i1]))))
    out["one_period"] = one_month
    # 반사실 워크포워드 성과
    bh = run_targets(d0, bh_targets(d0), COST)
    b2 = run_targets(d0, b2_targets(d0), COST)
    ref = dict(B0={k: sub_stats(bh["days"], bh["r"], *v) for k, v in SUBS.items()},
               B2={k: sub_stats(b2["days"], b2["r"], *v) for k, v in SUBS.items()})
    ref["B0"]["full"] = sub_stats(bh["days"], bh["r"], LO, HI)
    ref["B2"]["full"] = sub_stats(b2["days"], b2["r"], LO, HI)
    out["ref"] = ref
    var = {}
    for v, reps in avail.items():
        rows = []
        for r in reps:
            z = load_replay(v, r)
            pb, _ = perf_block(d0, z["delta"].astype(float), subs=SUBS)
            n_rej = sum(1 for e in z["log"] if not e["gate_ok"])
            rows.append(dict(rep=r, n_gate_fail=n_rej, **pb))
        var[v] = rows
    # 저장된 원본 (비교 기준)
    rows = []
    for r, run in enumerate(runs):
        pb, _ = perf_block(d0, run["delta"][0][CD].astype(float), subs=SUBS)
        rows.append(dict(rep=r, **pb))
    var["saved"] = rows
    out["variants"] = var
    # (3) 2025·2026 1월 새 모델: 그대로(frozen) 썼다면 — 2024년까지는 실제 경로, 2025·2026은 frozen Δ
    fz = []
    for r in avail["V0"]:
        z = load_replay("V0", r)
        act = runs[r]["delta"][0][CD].astype(float).copy()
        mix = act.copy()
        for y in (2025, 2026):
            i0, i1 = decision_range(d0, ts(f"{y}-01-01"), ts(f"{y + 1}-01-01"))
            mix[i0:i1] = z[f"frozen_{y}"][i0:i1]
        pb, res = perf_block(d0, mix, subs=SUBS)
        i0, _ = decision_range(d0, ts("2025-01-01"), ts(HI))
        a, b = window_range(d0, LO, HI)
        lb = mix[i0:b]
        fz.append(dict(rep=r, lockbox=pb["lockbox"], min_delta=float(np.nanmin(lb)),
                       frac_below=float(np.mean(lb < -TH)), p05=float(np.nanpercentile(lb, 5)),
                       mean_delta=float(np.nanmean(lb)),
                       sanity=[e for e in z["log"] if e["month"] in ("2025-01-01", "2026-01-01")]))
    out["frozen_2025_26"] = fz
    return out


# ══════════ (4) Δ 회귀 ══════════
def ols_r2(X, y):
    Xc = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    res = y - Xc @ beta
    return beta, 1 - res.var() / y.var()


GROUPS = {
    "long_trend": ["ma_1200", "x_300_1200", "rsi_84", "macdh_1d", "ret_180"],
    "mid_trend": ["ma_200", "ma_50", "x_50_200", "ret_42", "dd_180"],
    "short": ["ret_1", "ret_6", "ma_20", "rsi_14", "macd_4h", "macdh_4h", "clv", "range"],
    "volatility": ["vol_regime"],
    "volume": ["rel_vol", "vol_trend", "flow_20"],
}


def group_r2(Z, y):
    full = ols_r2(Z, y)[1]
    out = dict(full=float(full))
    for g, names in GROUPS.items():
        idx = [FI[n] for n in names]
        rest = [j for j in range(Z.shape[1]) if j not in idx]
        out[g] = dict(alone=float(ols_r2(Z[:, idx], y)[1]), drop=float(full - ols_r2(Z[:, rest], y)[1]))
    return out


def part4(d0, runs, a, b):
    yrs = year_of(d0, np.arange(a, b))
    X = d0.X[a:b].astype(float)
    sma = pd.Series(d0.c).rolling(1200).mean().to_numpy()[a:b]
    above = d0.c[a:b] > sma
    D = np.array([run["delta"][0][CD][a:b] for run in runs], float)     # (R, T)
    Dm = np.nanmean(D, axis=0)
    out = dict(years=[], global_={})
    # 전체: 연도 평균(절편)과 연도 내 변동 분해
    ok = ~np.isnan(Dm)
    ym = {y: np.nanmean(Dm[yrs == y]) for y in set(yrs)}
    between = np.array([ym[y] for y in yrs])
    out["global_"]["var_total"] = float(np.nanvar(Dm))
    out["global_"]["var_between_year"] = float(np.nanvar(between[ok]))
    out["global_"]["share_between_year"] = float(np.nanvar(between[ok]) / np.nanvar(Dm[ok]))
    # 반복 간 일치: 반복별 Δ의 상관
    C_ = np.corrcoef(D[:, ok])
    out["global_"]["rep_corr_median"] = float(np.median(C_[np.triu_indices(len(runs), 1)]))
    Z = (X - X.mean(0)) / X.std(0)
    beta, r2 = ols_r2(Z[ok], Dm[ok])
    out["global_"]["r2_all22"] = float(r2)
    out["global_"]["beta"] = {n: float(beta[1 + i]) for i, n in enumerate(NAMES)}
    _, r2m = ols_r2(Z[ok][:, [FI["ma_1200"]]], Dm[ok])
    out["global_"]["r2_ma1200"] = float(r2m)
    # ma_1200 − x_300_1200 = ln(C/SMA300)/(20σ) (클립 제외) — '종가 vs 50일선'
    z300 = X[:, FI["ma_1200"]] - X[:, FI["x_300_1200"]]
    out["global_"]["r2_c_vs_sma300"] = float(ols_r2(z300[ok][:, None], Dm[ok])[1])
    out["global_"]["r2_c_vs_sma300_plus_ma1200"] = float(ols_r2(np.column_stack([z300, X[:, FI["ma_1200"]]])[ok], Dm[ok])[1])
    # 연도 더미 + 22지표
    Ys = sorted(set(yrs))
    dum = np.column_stack([(yrs == y).astype(float) for y in Ys[1:]])
    _, r2d = ols_r2(dum[ok], Dm[ok])
    _, r2dz = ols_r2(np.column_stack([dum, Z])[ok], Dm[ok])
    out["global_"]["r2_year_dummies"] = float(r2d)
    out["global_"]["r2_year_dummies_plus22"] = float(r2dz)
    # 지표 묶음별 설명력 (ma_200 = 0.5·ma_50 + x_50_200 처럼 정확한 공선성이 있어 개별 β보다 묶음이 안전)
    out["global_"]["groups"] = group_r2(Z[ok], Dm[ok])
    Zq = np.column_stack([Z, Z ** 2])
    out["global_"]["r2_quad44"] = float(ols_r2(Zq[ok], Dm[ok])[1])
    keep = [j for j, n in enumerate(NAMES) if n != "x_50_200"]
    for y in Ys:
        sel = (yrs == y) & ok
        Zy = (X[sel] - X[sel].mean(0)) / (X[sel].std(0) + 1e-12)
        betas, r2s, r2_1, r2_2, r2_5 = [], [], [], [], []
        for r in range(len(runs)):
            dy = D[r, sel]
            rr = ols_r2(Zy, dy)[1]
            be = np.full(len(NAMES) + 1, np.nan)
            bk, _ = ols_r2(Zy[:, keep], dy)
            be[0] = bk[0]
            be[1 + np.array(keep)] = bk[1:]
            betas.append(be[1:])
            r2s.append(rr)
            r2_1.append(ols_r2(Zy[:, [FI["ma_1200"]]], dy)[1])
            r2_2.append(ols_r2(Zy[:, [FI["ma_1200"], FI["rsi_84"]]], dy)[1])
            r2_5.append(ols_r2(Zy[:, [FI[n] for n in ("ma_1200", "rsi_84", "x_300_1200", "ret_180", "vol_regime")]], dy)[1])
        bm = np.median(betas, axis=0)
        corr = [float(np.corrcoef(X[sel][:, j], Dm[sel])[0, 1]) for j in range(len(NAMES))]
        top = np.argsort(-np.nan_to_num(np.abs(bm)))[:5]
        grp = group_r2(Zy, Dm[sel])
        # 규칙 일치 (반복별 보유 vs 규칙, 중앙값)
        agree_b2, p_long_above, p_long_below = [], [], []
        for r, run in enumerate(runs):
            pos = policy_from_delta(run["delta"][0][CD][a:b].astype(float), CD, d0.forced_hold[a:b])
            p = pos[yrs == y]
            ab = above[yrs == y]
            agree_b2.append(np.mean(p == ab))
            p_long_above.append(p[ab].mean() if ab.any() else np.nan)
            p_long_below.append(p[~ab].mean() if (~ab).any() else np.nan)
        out["years"].append(dict(
            year=int(y), r2_22=med(r2s), r2_ma1200=med(r2_1), r2_ma1200_rsi84=med(r2_2), r2_top5=med(r2_5),
            top=[(NAMES[j], float(bm[j])) for j in top], beta={n: float(bm[i]) for i, n in enumerate(NAMES)},
            corr=dict(zip(NAMES, corr)), frac_above_sma1200=float(above[yrs == y].mean()),
            agree_b2=med(agree_b2), p_long_above=med(p_long_above), p_long_below=med(p_long_below),
            sd_delta=float(np.nanstd(Dm[sel])), mean_delta=float(np.nanmean(Dm[sel])), groups=grp))
    # 전체 기간 규칙 일치
    agree = {}
    c = pd.Series(d0.c)
    rules = {
        "B2 C>SMA1200": above,
        "B1 SMA300>SMA1200": (c.rolling(300).mean() > c.rolling(1200).mean()).to_numpy()[a:b],
        "C>SMA300 (50d)": (c > c.rolling(300).mean()).to_numpy()[a:b],
        "B5 MACDh_1d>0": X[:, FI["macdh_1d"]] > 0,
        "B6 ret_180>0": X[:, FI["ret_180"]] > 0,
        "RSI84>50": X[:, FI["rsi_84"]] > 0,
    }
    for name, rule in rules.items():
        ag, pa, pb = [], [], []
        for run in runs:
            pos = policy_from_delta(run["delta"][0][CD][a:b].astype(float), CD, d0.forced_hold[a:b])
            ag.append(np.mean(pos == rule))
            pa.append(pos[rule].mean())
            pb.append(pos[~rule].mean())
        agree[name] = dict(agree=med(ag), p_long_if_rule_long=med(pa), p_long_if_rule_flat=med(pb),
                           rule_exposure=float(np.mean(rule)))
    out["rules"] = agree
    # 매도(보유→현금)가 일어난 봉의 지표 모습
    sells = []
    for r, run in enumerate(runs):
        pos = policy_from_delta(run["delta"][0][CD][a:b].astype(float), CD, d0.forced_hold[a:b])
        idx = np.nonzero((pos[1:] == 0) & (pos[:-1] == 1))[0] + 1
        sells.extend(idx.tolist())
    sells = np.array(sells)
    out["at_sell"] = dict(n=int(len(sells)), frac_below_sma1200=float(np.mean(~above[sells])),
                          median={n: float(np.median(X[sells, FI[n]])) for n in ("ma_1200", "rsi_84", "ret_180", "ret_42", "dd_180", "x_300_1200")},
                          overall_median={n: float(np.median(X[:, FI[n]])) for n in ("ma_1200", "rsi_84", "ret_180", "ret_42", "dd_180", "x_300_1200")})
    return out


# ══════════ (5) 낙폭 구간 ══════════
def dd_episodes(days, eq, min_dd=0.25):
    """매수·보유 자산에서 고점→저점 낙폭 구간 (회복 전 최저점까지). min_dd 이상만"""
    eps = []
    peak_i = 0
    i = 1
    n = len(eq)
    while i < n:
        if eq[i] >= eq[peak_i]:
            peak_i = i
            i += 1
            continue
        # 낙폭 시작: 다음 신고가까지
        j = i
        while j < n and eq[j] < eq[peak_i]:
            j += 1
        trough = peak_i + int(np.argmin(eq[peak_i:j]))
        dd = eq[trough] / eq[peak_i] - 1
        if dd <= -min_dd:
            eps.append(dict(peak=peak_i, trough=trough, recover=j if j < n else None, dd=float(dd)))
        peak_i = j if j < n else peak_i
        i = j + 1 if j < n else n
    return eps


def zigzag(x, th):
    """로그 자산 x에서 th 이상 되돌림으로 확정되는 고점·저점 번호 (처음·마지막 극값·끝 포함)"""
    piv = [0]
    trend, ext, mx, mn = 0, 0, 0, 0
    for i in range(1, len(x)):
        if trend == 0:
            mx = i if x[i] > x[mx] else mx
            mn = i if x[i] < x[mn] else mn
            if x[i] - x[mn] >= th:
                if mn != 0:
                    piv.append(mn)
                trend, ext = 1, i
            elif x[mx] - x[i] >= th:
                if mx != 0:
                    piv.append(mx)
                trend, ext = -1, i
            continue
        if trend == 1:
            if x[i] >= x[ext]:
                ext = i
            elif x[ext] - x[i] >= th:
                piv.append(ext)
                trend, ext = -1, i
        else:
            if x[i] <= x[ext]:
                ext = i
            elif x[i] - x[ext] >= th:
                piv.append(ext)
                trend, ext = 1, i
    if ext != piv[-1] and ext != len(x) - 1:
        piv.append(ext)
    if piv[-1] != len(x) - 1:
        piv.append(len(x) - 1)
    return piv


def daily_pos(d0, res, n_days):
    day_id = ((d0.ts[res["a"]:res["b"]] + BAR_SEC - ts(LO)) // 86400).astype(int)
    pos = np.zeros(n_days + 1)
    cnt = np.zeros(n_days + 1)
    dd = np.clip(day_id, 0, n_days)
    np.add.at(pos, dd, res["pos"])
    np.add.at(cnt, dd, 1)
    return np.where(cnt > 0, pos / np.maximum(cnt, 1), 0)[:n_days + 1]


def part5(d0, runs):
    bh = run_targets(d0, bh_targets(d0), COST)
    b2 = run_targets(d0, b2_targets(d0), COST)
    days = bh["days"]
    eqb = np.concatenate([[1.0], np.cumprod(1 + bh["r"])])     # eq[k] = k일째 자정 자산 (eq[0]=시작)
    day0 = np.concatenate([[days[0] - 86400], days])
    ag = []
    for run in runs:
        res = run_targets(d0, agent_targets(d0, run["delta"][0][CD].astype(float)), COST)
        ag.append(np.concatenate([[1.0], np.cumprod(1 + res["r"])]))
    ag = np.array(ag)
    eq2 = np.concatenate([[1.0], np.cumprod(1 + b2["r"])])
    # 낙폭 구간 (매수·보유 기준 25% 이상, 회복 전 최저점 기준 — 2017~2026)
    eps = dd_episodes(day0, eqb, 0.25)
    rows = []
    dstr = lambda k: str(pd.to_datetime(day0[k], unit="s").date())
    for e in eps:
        p, t = e["peak"], e["trough"]
        rv = e["recover"] if e["recover"] is not None else len(eqb) - 1
        ar = ag[:, t] / ag[:, p] - 1
        a_rec = ag[:, rv] / ag[:, t] - 1
        # 에이전트 자신의 구간 내 최대낙폭
        own = [float(np.min(x[p:t + 1] / np.maximum.accumulate(x[p:t + 1])) - 1) for x in ag]
        rows.append(dict(peak=dstr(p), trough=dstr(t), recover=dstr(rv) if e["recover"] is not None else None,
                         bh=e["dd"], agent_med=med(ar), agent_min=float(ar.min()), agent_max=float(ar.max()),
                         agent_rep6=float(ar[6]), agent_own_dd_med=med(own),
                         b2=float(eq2[t] / eq2[p] - 1),
                         bh_rebound=float(eqb[rv] / eqb[t] - 1), agent_rebound_med=med(a_rec),
                         b2_rebound=float(eq2[rv] / eq2[t] - 1),
                         log_excess_dd_med=med(np.log(ag[:, t] / ag[:, p]) - math.log(eqb[t] / eqb[p])),
                         log_excess_rebound_med=med(np.log(ag[:, rv] / ag[:, t]) - math.log(eqb[rv] / eqb[t]))))
    # 각 반복의 최대낙폭 구간
    mdd = []
    for r in range(len(runs)):
        x = ag[r]
        dd = x / np.maximum.accumulate(x) - 1
        t = int(np.argmin(dd))
        p = int(np.argmax(x[:t + 1]))
        mdd.append(dict(rep=r, mdd=float(dd[t]), peak=dstr(p), trough=dstr(t)))
    ddb = eqb / np.maximum.accumulate(eqb) - 1
    tb = int(np.argmin(ddb))
    pb = int(np.argmax(eqb[:tb + 1]))
    # 연도별 로그 초과수익 (에이전트 − 매수·보유) 분해
    yrs = pd.to_datetime(day0, unit="s", utc=True).year.to_numpy()
    lx = []
    for y in sorted(set(yrs[1:])):
        idx = np.nonzero(yrs == y)[0]
        k0, k1 = max(idx[0] - 1, 0), idx[-1]
        v = np.log(ag[:, k1] / ag[:, k0]) - math.log(eqb[k1] / eqb[k0])
        lx.append(dict(year=int(y), log_excess_med=med(v), log_excess_rep6=float(v[6]),
                       b2_log_excess=float(math.log(eq2[k1] / eq2[k0]) - math.log(eqb[k1] / eqb[k0]))))
    total = np.log(ag[:, -1]) - math.log(eqb[-1])
    # 지그재그 구간(매수·보유 25% 이상 되돌림 기준): 하락 구간·상승 구간마다 로그 초과수익 → 합이 전체 초과수익
    legs = []
    piv = zigzag(np.log(eqb), math.log(1 / 0.75))
    pos_days = []
    for run in runs:
        res = run_targets(d0, agent_targets(d0, run["delta"][0][CD].astype(float)), COST)
        pos_days.append(daily_pos(d0, res, len(days)))
    pos_days = np.array(pos_days)
    for (i0, i1) in zip(piv[:-1], piv[1:]):
        v = np.log(ag[:, i1] / ag[:, i0]) - math.log(eqb[i1] / eqb[i0])
        legs.append(dict(start=dstr(i0), end=dstr(i1), kind="down" if eqb[i1] < eqb[i0] else "up",
                         bh=float(eqb[i1] / eqb[i0] - 1), agent_med=med(ag[:, i1] / ag[:, i0] - 1),
                         agent_rep6=float(ag[6, i1] / ag[6, i0] - 1), b2=float(eq2[i1] / eq2[i0] - 1),
                         exposure_med=med(pos_days[:, i0:i1].mean(axis=1)),
                         log_excess_med=med(v), log_excess_min=float(v.min()), log_excess_max=float(v.max()),
                         log_excess_rep6=float(v[6]),
                         b2_log_excess=float(math.log(eq2[i1] / eq2[i0]) - math.log(eqb[i1] / eqb[i0]))))
    return dict(episodes=rows, legs=legs, rep_mdd=mdd, bh_mdd=dict(mdd=float(ddb[tb]), peak=dstr(pb), trough=dstr(tb)),
                year_log_excess=lx, total_log_excess_med=med(total), total_log_excess_rep6=float(total[6]),
                b2_total_log_excess=float(math.log(eq2[-1]) - math.log(eqb[-1])))


# ══════════ (6) 모델이 '팔 수 있나' + Huber 손실의 낙관 편향 ══════════
def huber_loc(x, delta, w=None, it=50):
    x = np.asarray(x, float)
    w = np.ones_like(x) if w is None else np.asarray(w, float)
    mu = float(np.median(x))
    for _ in range(it):
        r = np.abs(x - mu)
        ww = w * np.minimum(1.0, delta / np.maximum(r, 1e-12))
        mu_new = float((ww * x).sum() / ww.sum())
        if abs(mu_new - mu) < 1e-10:
            break
        mu = mu_new
    return mu


def ens_from_params(ps):
    from btc.nn import StackedMLP
    from btc.agent import Ensemble
    net = StackedMLP.from_state({f"p{i}": p for i, p in enumerate(ps)})
    return Ensemble(net, "full22")


def part6(d0, runs, a, b):
    from btc.research.p0_autopsy_replay import load_replay, path
    yrs = year_of(d0, np.arange(a, b))
    X = d0.X[a:b]
    models = {}
    for r, run in enumerate(runs):
        st = run["last_state"]
        models[("final_2026-09", r)] = ens_from_params([st[f"p{i}"] for i in range(6)])
        if os.path.exists(path("V0", r)):
            z = load_replay("V0", r)
            for y in (2022, 2025, 2026):
                if f"model_{y}_p0" in z:
                    models[(f"cold_{y}", r)] = ens_from_params([z[f"model_{y}_p{i}"] for i in range(6)])
    out = {}
    for name in sorted({k[0] for k in models}):
        per_year = {}
        for (nm, r), ens in models.items():
            if nm != name:
                continue
            dl = ens.delta(X, CD)[0].astype(float)
            pos = policy_from_delta(dl, CD, d0.forced_hold[a:b], p0=1)
            for y in sorted(set(yrs)):
                sel = yrs == y
                per_year.setdefault(int(y), []).append((np.mean(dl[sel] < -TH), pos[sel].mean(), np.mean(dl[sel]),
                                                        np.min(dl[sel])))
        out[name] = {y: dict(frac_sell_zone=med([v[0] for v in vs]), exposure=med([v[1] for v in vs]),
                             mean_delta=med([v[2] for v in vs]), min_delta=med([v[3] for v in vs]), n_models=len(vs))
                     for y, vs in per_year.items()}
    # Huber(δ) 위치 대 평균: U1·U0 는 각각 ±½κm 을 Huber(δ=1)로 맞춤 → κm 기준 δ=2
    t_all = np.nonzero(d0.contig & ~d0.forced_hold & (np.arange(d0.T) >= 2400))[0]
    t_all = t_all[t_all < d0.T - 2]
    km = KAPPA * d0.m[t_all]
    yy = year_of(d0, t_all)
    hub = []
    for y in sorted(set(yy)):
        x = km[yy == y]
        hub.append(dict(year=int(y), n=int(len(x)), mean=float(x.mean()), median=float(np.median(x)),
                        huber2=huber_loc(x, 2.0), sd=float(x.std()),
                        frac_abs_gt2=float(np.mean(np.abs(x) > 2.0)),
                        bias_x33=float((huber_loc(x, 2.0) - x.mean()) / (1 - GAMMA))))
    # 1월 학습 표본 가중(최근 70% + 균등 30%)으로
    pool = []
    for y in range(2017, 2027):
        T_k = ts(f"{y}-01-01")
        t = d0.eligible(T_k, ts("2014-01-01"), 2400)
        age = (T_k - d0.ts[t]) / (2.0 * 365.25 * 86400)
        wr = np.power(2.0, -age)
        wr /= wr.sum()
        w = 0.7 * wr + 0.3 / len(t)
        x = KAPPA * d0.m[t]
        mu = float((w * x).sum() / w.sum())
        hl = huber_loc(x, 2.0, w)
        pool.append(dict(year=y, mean=mu, huber2=hl, bias=hl - mu, mean_x33=mu / (1 - GAMMA),
                         huber2_x33=hl / (1 - GAMMA), bias_x33=(hl - mu) / (1 - GAMMA)))
    return dict(models=out, huber_by_year=hub, huber_pool=pool)


def main():
    datas = load_phases()
    d0 = datas[0]
    runs = [load_run("P0", r) for r in range(REPS)]
    a, b = window_range(d0, LO, HI)
    out = dict(threshold=TH, c_dec=CD, cost=COST)
    out["part1"] = part1(d0, runs, a, b)
    # 잠금 구간에서 매도 문턱에 가장 가까이 간 때 (반복별)
    i0, _ = decision_range(d0, ts("2025-01-01"), ts(HI))
    close = []
    for r, run in enumerate(runs):
        x = run["delta"][0][CD][i0:b].astype(float)
        j = int(np.nanargmin(x))
        close.append(dict(rep=r, min_delta=float(x[j]), when=str(pd.to_datetime(d0.ts[i0 + j] + BAR_SEC, unit="s")),
                          price=float(d0.c[i0 + j]), gap_to_sell=float(x[j] + TH),
                          frac_within_0_1_of_sell=float(np.mean(x < -TH + 0.10))))
    out["lockbox_closest"] = close
    out["pool_drift"] = pool_drift(d0, range(2017, 2027))
    out["gates"] = gate_table(runs)
    av = have_replays()
    if av["V0"]:
        out["part23"] = part2_3(d0, runs)
    out["part4"] = part4(d0, runs, a, b)
    out["part5"] = part5(d0, runs)
    out["part6"] = part6(d0, runs, a, b)
    os.makedirs(OUT, exist_ok=True)

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            f = float(o)
            return None if (math.isnan(f) or math.isinf(f)) else round(f, 6)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o
    with open(os.path.join(OUT, "analysis.json"), "w", encoding="utf-8") as f:
        json.dump(clean(out), f, ensure_ascii=False, indent=1)
    print("saved", os.path.join(OUT, "analysis.json"))


if __name__ == "__main__":
    main()
