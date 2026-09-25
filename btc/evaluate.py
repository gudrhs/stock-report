# -*- coding: utf-8 -*-
"""
평가 — 저장된 워크포워드 결과(Δ 계열)를 성과표·통계검정으로 바꿉니다.

잠금 구간(2025-01-01 이후)은 BTC_LOCKBOX_OPEN=1 을 줘야만 계산하고, 열 때마다
data/btc/lockbox_audit.log 에 기록합니다. 개발하면서 잠금 구간 성적을 보고 설정을
고치는 일을 막기 위한 장치입니다.

  python -m btc.evaluate --window dev          2017~2024 (표본외 A·B)
  BTC_LOCKBOX_OPEN=1 python -m btc.evaluate --window full   2017~2026-09 전체 (한 번만)
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import json
import math
import time

import numpy as np
import pandas as pd

from . import config as C
from . import stats as S
from .agent import policy_from_delta
from .data import DATA_DIR
from .env import simulate, simulate_weights, daily_marks, BAR_SEC
from .features import NAMES
from .walkforward import load_phases, load_run as _load_run, decision_range, RUNS_DIR, n_trials, code_hash

_SEEN_CODE = set()


def load_run(name, r):
    """저장된 실행 결과 — 지금 코드와 다른 코드로 만든 결과면 거부 (고친 버그가 조용히 섞이지 않게)"""
    run = _load_run(name, r)
    if run.get("code_hash") != code_hash() and os.environ.get("BTC_ALLOW_STALE") != "1":
        raise RuntimeError(f"{name} rep{r}: 코드 해시 불일치 ({run.get('code_hash')} ≠ {code_hash()}) — 다시 실행하세요")
    _SEEN_CODE.add(run.get("code_hash"))
    return run

LOCK_TS = int(pd.Timestamp("2025-01-01", tz="UTC").timestamp())
AUDIT = os.path.join(DATA_DIR, "lockbox_audit.log")
WINDOWS = {
    "dev": ("2017-01-01", "2025-01-01"),
    "full": ("2017-01-01", "2026-09-25"),
}
SUBPERIODS = {"OOS-A": ("2017-01-01", "2023-01-01"), "OOS-B": ("2023-01-01", "2025-01-01"),
              "Lockbox": ("2025-01-01", "2026-09-25")}
FI = {n: i for i, n in enumerate(NAMES)}


class Sealed(RuntimeError):
    pass


def _ts(s):
    return int(pd.Timestamp(s, tz="UTC").timestamp())


def guard(hi_ts, what):
    """잠금 구간을 건드리는 계산은 플래그 없이는 거부"""
    if hi_ts > LOCK_TS:
        if os.environ.get("BTC_LOCKBOX_OPEN") != "1":
            raise Sealed(f"잠금 구간(2025-01-01 이후)입니다: {what}. BTC_LOCKBOX_OPEN=1 필요")
        os.makedirs(os.path.dirname(AUDIT), exist_ok=True)
        with open(AUDIT, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\topened\t{what}\n")


# ══════════ 한 전략 → 일별 수익 ══════════
class Window:
    def __init__(self, datas, lo, hi, what):
        guard(_ts(hi), what)
        self.datas, self.lo, self.hi = datas, _ts(lo), _ts(hi)
        self.rng = {}
        for k, d in enumerate(datas):
            a, b = decision_range(d, self.lo, self.hi)
            # simulate()는 체결 봉 b의 종가와 b+1 시가까지 읽습니다 → 둘 다 hi 이전이어야 함
            # (15분 밀린 phase는 마지막 한 봉을 줄여 잠금 구간 가격을 한 틱도 읽지 않게)
            while b > a and (d.ts[b] + BAR_SEC > self.hi or (b + 1 < d.T and d.ts[b + 1] > self.hi)):
                b -= 1
            self.rng[k] = (a, b)

    def run(self, targets, cost, phase=0, weights=False):
        d = self.datas[phase]
        a, b = self.rng[phase]
        if weights:
            sim = simulate_weights(targets, d.o, d.c, cost, a, b)
        else:
            sim = simulate(targets, d.o, d.c, cost, a, b, forced_hold=d.forced_hold)
        days, eq = daily_marks(d.ts, sim["mark"], a, b, lo=self.lo, hi=self.hi)
        r = eq[1:] / eq[:-1] - 1.0
        logr = sim["logr"][a:b].copy()
        if b + 1 < d.T and d.ts[b + 1] >= self.hi and b > a:
            # 마지막 결정의 가격 항은 다음 시가(= hi 이후 가격) 대신 체결 봉 종가로 — 잠금 구간 가격 미사용
            p_last = sim["pos"][b - 1]
            p_prev = sim["pos"][b - 2] if b - 2 >= a else 0.0
            logr[-1] = p_last * math.log(d.c[b] / d.o[b]) + abs(p_last - p_prev) * math.log(1.0 - cost)
        return dict(days=days[1:], r=r, eq=eq, pos=sim["pos"][a:b], logr=logr,
                    trades=sim["trades"], a=a, b=b,
                    rebal=sim["rebal"][a:b] if "rebal" in sim else None)

    def m(self, phase=0):
        """결정봉별 시가→시가 로그수익 m[t] (구간 끝에서 hi 이후 가격이 필요하면 체결 봉 종가로 대신)"""
        d = self.datas[phase]
        a, b = self.rng[phase]
        m = d.m[a:b].copy()
        if b + 1 < d.T and d.ts[b + 1] >= self.hi and b > a:
            m[-1] = math.log(d.c[b] / d.o[b])
        return np.nan_to_num(m)

    def agent_targets(self, delta, c_dec, phase=0):
        d = self.datas[phase]
        a, b = self.rng[phase]
        tg = np.full(d.T, np.nan)
        if np.isnan(delta[a:b]).all():
            raise ValueError("이 구간의 Δ가 없습니다 (워크포워드 미실행?)")
        tg[a:b] = policy_from_delta(delta[a:b], c_dec, d.forced_hold[a:b])
        return tg


def baseline_targets(W, phase=0):
    """B0~B7 (B8은 에이전트 노출도를 알아야 해서 따로). 교과서 기본값, 조정하지 않음"""
    d = W.datas[phase]
    a, b = W.rng[phase]
    X = d.X
    c = pd.Series(d.c)
    sma300, sma1200 = c.rolling(300).mean().to_numpy(), c.rolling(1200).mean().to_numpy()
    out = {}
    ones = np.full(d.T, np.nan)
    ones[a:b] = 1.0
    out["B0"] = ones
    out["B1"] = np.where(sma300 > sma1200, 1.0, 0.0)
    out["B2"] = np.where(d.c > sma1200, 1.0, 0.0)
    rsi = 50.0 + 25.0 * X[:, FI["rsi_14"]]
    b3 = np.full(d.T, np.nan)
    p = 0.0
    for t in range(a, b):
        if rsi[t] < 30:
            p = 1.0
        elif rsi[t] > 70:
            p = 0.0
        b3[t] = p
    out["B3"] = b3
    out["B4"] = np.where(X[:, FI["macdh_4h"]] > 0, 1.0, 0.0)
    out["B5"] = np.where(X[:, FI["macdh_1d"]] > 0, 1.0, 0.0)
    out["B6"] = np.where(X[:, FI["ret_180"]] > 0, 1.0, 0.0)
    out["B7"] = np.minimum(1.0, 0.5 / (d.sigma * math.sqrt(2190)))
    for k in out:
        v = np.full(d.T, np.nan)
        v[a:b] = out[k][a:b]
        out[k] = v
    return out


BASE_LABEL = {
    "B0": "매수·보유", "B1": "골든크로스 (≈일봉 50/200)", "B2": "200일선 위에서만 보유",
    "B3": "RSI14 30 매수·70 매도", "B4": "MACD(12,26,9) 4시간봉", "B5": "MACD 일봉 환산(72,156,54)",
    "B6": "1개월 모멘텀", "B7": "변동성 목표 매수·보유", "B8": "노출도 맞춘 고정비율 (사후정보)",
}


def perf(res, cost=None):
    s = S.summary(res["r"])
    if res.get("rebal") is not None:
        # 비율 보유 전략(B7·B8): 가격 변동에 따른 비중 흔들림은 전환이 아니므로 '재조정 횟수'로 셈
        years = len(res["pos"]) / (6 * 365.0)
        n_rb = int(res["rebal"].sum())
        s.update(exposure=float(res["pos"].mean()), switches_per_year=n_rb / years, avg_hold_days=None,
                 hit_rate=None, profit_factor=None, n_trades=n_rb)
        return s
    t = S.trade_stats(res["pos"], res["logr"])
    s.update({k: t[k] for k in ("exposure", "switches_per_year", "avg_hold_days", "hit_rate",
                                "profit_factor", "n_trades")})
    return s


def lower_median(xs):
    """10개면 5번째(작은 쪽 중앙값)"""
    order = np.argsort(xs, kind="stable")
    return int(order[(len(xs) - 1) // 2])


# ══════════ 전체 평가 ══════════
def evaluate(window="dev", reps_p0=10, reps_grid=5, reps_abl=3, n_null2=8, quick=False):
    lo, hi = WINDOWS[window]
    datas = load_phases()
    W = Window(datas, lo, hi, f"evaluate:{window}")
    cost = C.PRIMARY_COST
    cd_p0 = round(C.c_dec(cost, C.P0["c_mult"]), 6)
    out = dict(window=window, lo=lo, hi=hi, cost=cost, c_dec=cd_p0,
               generated=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    # ── 기준전략 ──
    bt = baseline_targets(W)
    base = {}
    for k, tg in bt.items():
        base[k] = W.run(tg, cost, weights=(k == "B7"))
    bh = base["B0"]
    days = bh["days"]

    # ── P0 반복 10개 ──
    runs = [load_run("P0", r) for r in range(reps_p0)]            # 없으면 예외 — 조용히 건너뛰지 않음
    p0 = []
    for r, run in enumerate(runs):
        res = W.run(W.agent_targets(run["delta"][0][cd_p0], cd_p0), cost)
        assert np.array_equal(res["days"], days)
        p0.append(res)
    dsr_vs_bh = [S.sharpe(x["r"]) - S.sharpe(bh["r"]) for x in p0]
    hi_idx = lower_median(dsr_vs_bh)
    head = p0[hi_idx]
    out["headline_rep"] = hi_idx
    out["reps"] = [dict(rep=r, d_sharpe=dsr_vs_bh[r], **perf(x)) for r, x in enumerate(p0)]
    # B8: 헤드라인 반복의 평균 노출도로 고정비율 (사후 정보라 표시)
    e = float(head["pos"].mean())
    w8 = np.full(datas[0].T, np.nan)
    a0, b0 = W.rng[0]
    w8[a0:b0] = e
    base["B8"] = W.run(w8, cost, weights=True)
    out["baselines"] = {k: dict(label=BASE_LABEL[k], **perf(v)) for k, v in base.items()}
    out["agent"] = perf(head)
    # 비용 전(gross)과 비용 부담
    gross = W.run(W.agent_targets(runs[hi_idx]["delta"][0][cd_p0], cd_p0), 0.0)
    out["agent"]["gross_cagr"] = S.summary(gross["r"])["cagr"]
    out["agent"]["cost_drag"] = out["agent"]["gross_cagr"] - out["agent"]["cagr"]
    nw = S.newey_west_alpha_beta(head["r"], bh["r"], lags=10)
    out["agent"].update(alpha=nw["alpha_annual"], beta=nw["beta"], t_alpha=nw["t_alpha"])

    # ── 비용 시나리오·손익분기 ──
    cost_rows = []
    for name, c in list(C.COSTS.items()):
        for m in (1.0, 2.0):
            cdx = round(C.c_dec(c, m), 6)
            bh_c = W.run(bt["B0"], c)
            ag = W.run(W.agent_targets(runs[hi_idx]["delta"][0][cdx], cdx), c)
            cost_rows.append(dict(scenario=name, cost=c, mult=m, sharpe=S.sharpe(ag["r"]),
                                  bh_sharpe=S.sharpe(bh_c["r"]), cagr=S.summary(ag["r"])["cagr"],
                                  bh_cagr=S.summary(bh_c["r"])["cagr"],
                                  d_sharpe_all_reps=[S.sharpe(W.run(W.agent_targets(rr["delta"][0][cdx], cdx), c)["r"])
                                                     - S.sharpe(bh_c["r"]) for rr in runs] if not quick else None))
    out["costs"] = cost_rows
    be = []
    for c in C.BREAKEVEN:
        cdx = round(C.c_dec(c, C.P0["c_mult"]), 6)
        ag = W.run(W.agent_targets(runs[hi_idx]["delta"][0][cdx], cdx), c)
        bh_c = W.run(bt["B0"], c)
        be.append(dict(cost=c, sharpe=S.sharpe(ag["r"]), cagr=S.summary(ag["r"])["cagr"],
                       bh_sharpe=S.sharpe(bh_c["r"]), bh_cagr=S.summary(bh_c["r"])["cagr"]))
    out["breakeven"] = be

    # ── phase 16개 (헤드라인 반복) ──
    ph = []
    for k in range(len(datas)):
        dk = runs[hi_idx]["delta"].get(k, {}).get(cd_p0)
        if dk is None:
            continue
        ag = W.run(W.agent_targets(dk, cd_p0, phase=k), cost, phase=k)
        bk = W.run(baseline_targets(W, k)["B0"], cost, phase=k)
        ph.append(dict(phase=k, sharpe=S.sharpe(ag["r"]), bh_sharpe=S.sharpe(bk["r"]),
                       d_sharpe=S.sharpe(ag["r"]) - S.sharpe(bk["r"])))
    out["phases"] = ph

    # ── 검정 ──
    BOOT_SEED = 1                                   # H1과 H2 모든 비교에 같은 재표본 (같은 검정은 같은 p)
    boot = S.stationary_bootstrap_diff(head["r"], bh["r"], mean_block=20, n_boot=10000, seed=BOOT_SEED)
    lw = S.lw_hac_test(head["r"], bh["r"])
    out["H1"] = dict(bootstrap=boot, lw=lw)
    h2 = []
    for k in ("B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7"):
        bb = boot if k == "B0" else S.stationary_bootstrap_diff(head["r"], base[k]["r"], mean_block=20,
                                                                 n_boot=10000, seed=BOOT_SEED)
        h2.append(dict(base=k, d_sharpe=bb["obs"], p=bb["p"], ci90=bb["ci90"]))
    adj = S.holm([x["p"] for x in h2])
    for x, pa in zip(h2, adj):
        x["p_holm"] = pa
    out["H2"] = h2
    out["psr"] = dict(sr0=S.psr(head["r"]), excess=S.psr(head["r"] - bh["r"]))

    # ── 격자 12개·절제·선택기 ──
    grid_r, grid_names = {}, []
    for g in C.grid():
        tname = train_name(g)
        cdx = round(C.c_dec(cost, g["c_mult"]), 6)
        rs = []
        for r in range(reps_grid):
            run = load_run(tname, r)                                  # 반복 번호로 짝을 맞춤 — 빠지면 예외
            rs.append(W.run(W.agent_targets(run["delta"][0][cdx], cdx), cost))
        grid_r[g["name"]] = rs
        grid_names.append(g["name"])
    out["grid"] = [dict(name=n, reps=len(v), sharpe_mean=float(np.mean([S.sharpe(x["r"]) for x in v])),
                        sharpe_reps=[S.sharpe(x["r"]) for x in v],
                        cagr_mean=float(np.mean([S.summary(x["r"])["cagr"] for x in v])),
                        maxdd_mean=float(np.mean([S.summary(x["r"])["max_dd"] for x in v])),
                        exposure_mean=float(np.mean([x["pos"].mean() for x in v])))
                   for n, v in grid_r.items()]
    abl = {}
    for key in C.ABLATIONS:
        rs = []
        for r in range(reps_abl):
            run = load_run(key, r)
            rs.append(W.run(W.agent_targets(run["delta"][0][cd_p0], cd_p0), cost))
        abl[key] = rs
    out["ablations"] = [dict(name=k, reps=len(v), sharpe_reps=[S.sharpe(x["r"]) for x in v],
                             sharpe_mean=float(np.mean([S.sharpe(x["r"]) for x in v])),
                             cagr_mean=float(np.mean([S.summary(x["r"])["cagr"] for x in v])),
                             exposure_mean=float(np.mean([x["pos"].mean() for x in v])))
                        for k, v in abl.items()]
    # 선택기 S
    sel = selector(W, grid_r, datas, cost, days) if len(grid_r) == 12 else None
    out["selector"] = sel["summary"] if sel else None

    # ── PBO·DSR (2017~2024) ──
    if len(grid_r) == 12:
        cut = np.searchsorted(days, _ts("2025-01-01"), side="right")   # 2024-12-31을 덮는 수익까지
        M = np.column_stack([np.mean([x["r"][:cut] for x in grid_r[n]], axis=0) for n in grid_names])
        pb = S.pbo_cscv(M, S=16)
        out["pbo"] = {k: pb[k] for k in ("pbo", "slope", "intercept", "p_oos_loss", "n_splits", "block_len")}
        out["pbo"]["logit_hist"] = np.histogram(pb["logits"], bins=20, range=(-4, 4))[0].tolist()
        allM = np.column_stack([M] + [np.mean([x["r"][:cut] for x in v], axis=0)[:, None] for v in abl.values()])
        srv = S.trials_sr_var(allM)
        N = max(n_trials(), allM.shape[1])
        best = grid_names[int(np.argmax([S.sharpe(M[:, j]) for j in range(M.shape[1])]))]
        bh_cut = bh["r"][:cut]
        srv_x = S.trials_sr_var(allM - bh_cut[:, None])          # 초과수익(전략 − 매수·보유) 기준
        # 사후 최고 변형도 P0처럼 '작은 쪽 중앙값 반복 하나'로, 2017~2024에서만 비교
        bi = grid_r[best]
        bi_lm = bi[lower_median([S.sharpe(x["r"][:cut]) - S.sharpe(bh_cut) for x in bi])]["r"][:cut]
        i19 = int(np.searchsorted(days, _ts("2019-01-01"), side="right"))
        out["dsr"] = dict(
            N=N, sr_var=srv, sr_var_excess=srv_x,
            # 판정에 쓰는 것은 '매수·보유 대비 초과수익'의 DSR — 절대 샤프(0 대비)는 장기 상승장에선 누구나 높음
            p0_excess=S.dsr(head["r"] - bh["r"], N, srv_x),
            p0_abs=S.dsr(head["r"], N, srv),
            best_in_hindsight=dict(name=best, excess=S.dsr(bi_lm - bh_cut, N, srv_x), abs=S.dsr(bi_lm, N, srv)),
            selector=dict(excess=S.dsr(sel["r"] - bh["r"][i19:], N, srv_x), abs=S.dsr(sel["r"], N, srv)) if sel else None)

    # ── 귀무모형 ──
    d0 = datas[0]
    ex, sw = S.exposure_switch_rate(head["pos"])
    day_id = ((d0.ts[a0:b0] + BAR_SEC - W.lo) // 86400).astype(int)
    m = W.m(0)
    if 0 < ex < 1 and sw > 0:
        n1 = S.null_markov(m, cost, ex, min(sw, 2 * min(ex, 1 - ex) * 0.999), n=1000, seed=3, day_id=day_id)
        out["N1"] = dict(percentile=S.null_percentile(S.sharpe(head["r"]), n1), median=float(np.median(n1)),
                         p95=float(np.percentile(n1, 95)))
    pos_d = daily_position(head, day_id, len(days))
    out["N3"] = null_shift_bar(head["pos"], m, cost, day_id, len(days), n=2000, min_days=30, seed=4)
    n2 = []
    Wd = Window(datas, lo, "2025-01-01", "N2 compare")
    for r in range(n_null2):
        run = load_run("N2", r)
        n2.append(S.sharpe(Wd.run(Wd.agent_targets(run["delta"][0][cd_p0], cd_p0), cost)["r"]))
    hd = Wd.run(Wd.agent_targets(runs[hi_idx]["delta"][0][cd_p0], cd_p0), cost)
    out["N2"] = dict(sharpes=n2, n=len(n2), p0_dev_sharpe=S.sharpe(hd["r"]),
                     beats_all=bool(len(n2) == n_null2 and S.sharpe(hd["r"]) > max(n2)))

    # ── 연도별·국면별·해석 ──
    out["years"] = by_year(days, {"agent": head["r"], "B0": bh["r"], "B2": base["B2"]["r"]},
                           exposure=pos_d)
    out["regimes"] = by_regime(W, head, bh, datas)
    out["interpret"] = interpret(W, head, runs[hi_idx], cd_p0, datas)

    # ── 판정 (claim rule) ──
    reps_pos = int(sum(x > 0 for x in dsr_vs_bh))
    ph_pos = int(sum(x["d_sharpe"] > 0 for x in ph))
    claim = dict(
        holm_p=h2[0]["p_holm"], dsr=out.get("dsr", {}).get("p0_excess"), reps_positive=reps_pos, reps=len(p0),
        phases_positive=ph_pos, phases=len(ph), n1_percentile=out.get("N1", {}).get("percentile"),
        n2_beats_all=out.get("N2", {}).get("beats_all"))
    claim["n2_runs"] = out["N2"]["n"]
    claim["passed"] = bool(claim["holm_p"] is not None and claim["holm_p"] < 0.05
                           and (claim["dsr"] or 0) >= 0.95 and reps_pos >= 8 and ph_pos >= 12
                           and (claim["n1_percentile"] or 0) >= 0.95 and claim["n2_beats_all"])
    out["claim"] = claim
    out["b2_ge_agent"] = bool(S.sharpe(base["B2"]["r"]) >= S.sharpe(head["r"]))
    out["bh_ge_agent"] = bool(S.sharpe(bh["r"]) >= S.sharpe(head["r"]))
    # 연도별 부호 검정: 에이전트가 매수·보유보다 나은 해의 수 (이항, 단측)
    yrs = [y for y in out["years"] if y["n_days"] >= 300]
    k = sum(1 for y in yrs if y["agent"] > y["B0"])
    n = len(yrs)
    out["year_sign_test"] = dict(wins=k, years=n, p=float(sum(math.comb(n, j) for j in range(k, n + 1)) / 2 ** n) if n else None)
    # 손익분기 비용: 샤프 차이(에이전트 − 매수·보유)가 0을 지나는 편도 비용 (선형 보간)
    be_x = [(b["cost"], b["sharpe"] - b["bh_sharpe"]) for b in out["breakeven"]]
    out["breakeven_cost"] = None
    for (c0, d0_), (c1, d1_) in zip(be_x, be_x[1:]):
        if d0_ >= 0 > d1_:
            out["breakeven_cost"] = c0 + (c1 - c0) * d0_ / (d0_ - d1_)
            break
    if be_x and be_x[0][1] < 0:
        out["breakeven_cost"] = 0.0                  # 비용 0에서도 매수·보유보다 못함

    # ── 차트용 계열 (일별 자산) ──
    out["series"] = dict(days=[int(x) for x in days],
                         agent=np.cumprod(1 + head["r"]).round(5).tolist(),
                         B0=np.cumprod(1 + bh["r"]).round(5).tolist(),
                         B2=np.cumprod(1 + base["B2"]["r"]).round(5).tolist(),
                         B7=np.cumprod(1 + base["B7"]["r"]).round(5).tolist(),
                         exposure=pos_d.round(3).tolist(),
                         reps=[np.cumprod(1 + x["r"])[::7].round(4).tolist() for x in p0])
    s_full = None
    if sel:
        i19 = int(np.searchsorted(days, _ts("2019-01-01"), side="right"))
        s_full = np.full(len(days), np.nan)
        s_full[i19:] = sel["r"]
    out["subperiods"] = subperiods(days, {"agent": head["r"], "B0": bh["r"], "B2": base["B2"]["r"],
                                          "B6": base["B6"]["r"], "S": s_full})
    # 잠금 구간은 하락장이라 보유를 줄인 전략이 구조적으로 매수·보유를 이김 → 200일선·모멘텀·무작위정책과 비교
    lb = [p for p in out["subperiods"] if p["name"] == "Lockbox"]
    if lb and "N1" in out:
        sel_lb = days > _ts(SUBPERIODS["Lockbox"][0])
        la, lb_ = int(np.argmax(sel_lb)), len(days)
        dmask = (day_id >= la) & (day_id < lb_)
        if dmask.sum() > 100:
            e2, s2 = S.exposure_switch_rate(head["pos"][dmask])
            if 0 < e2 < 1 and s2 > 0:
                n1_lb = S.null_markov(m[dmask], cost, e2, min(s2, 2 * min(e2, 1 - e2) * 0.999), n=1000, seed=5,
                                      day_id=day_id[dmask] - la)
                lb[0]["N1_percentile"] = S.null_percentile(S.sharpe(head["r"][sel_lb]), n1_lb)
    out["n_trials"] = n_trials()
    out["code_hash"] = sorted(x for x in _SEEN_CODE if x)
    out["counts"] = dict(p0=len(p0), grid={k: len(v) for k, v in grid_r.items()},
                         ablations={k: len(v) for k, v in abl.items()}, n2=out["N2"]["n"])
    out["phase_d_sharpe"] = [x["d_sharpe"] for x in ph]
    try:
        with open(AUDIT, encoding="utf-8") as f:
            out["lockbox_openings"] = sum(1 for l in f if "\topened\t" in l)
    except FileNotFoundError:
        out["lockbox_openings"] = 0
    return out


def train_name(g):
    """격자 변형의 학습 설정 이름 (비용배수는 판단 때만 달라서 학습은 공유)"""
    if g["gamma"] == C.P0["gamma"] and g["features"] == C.P0["features"]:
        return "P0"
    return f"g{g['gamma']:g}_{g['features']}"


def daily_position(res, day_id, n_days):
    """
    4시간봉 보유 → 일별 평균 보유 (일별 수익과 같은 길이).
    결정봉 t의 보유는 t+1봉(종가 시각부터 4시간) 동안 수익을 내므로, 종가가 i번째 날
    [D_i, D_i+1) 안에 있는 결정들이 일별 수익 r[i] (D_i → D_i+1)을 만듭니다.
    """
    pos = np.zeros(n_days + 1)
    cnt = np.zeros(n_days + 1)
    dd = np.clip(day_id, 0, n_days)
    np.add.at(pos, dd, res["pos"])
    np.add.at(cnt, dd, 1)
    p = np.where(cnt > 0, pos / np.maximum(cnt, 1), 0)
    return p[:n_days]


def null_shift_bar(pos, m, cost, day_id, n_days, n=2000, min_days=30, seed=4):
    """
    N3 귀무: 4시간봉 보유 계열을 하루 단위(≥30일)로 원형 이동해 수익과 어긋나게 한 뒤,
    관측값과 똑같은 회계(a·m + |Δa|·ln(1−c), 일별 합산)로 샤프를 비교합니다.
    """
    lnc = math.log(1.0 - cost)
    dd = np.clip(day_id, 0, n_days - 1)

    def sh(p):
        g = p * m + np.abs(np.diff(np.concatenate([[0.0], p]))) * lnc
        r = np.expm1(np.bincount(dd, weights=g, minlength=n_days)[:n_days])
        return S.sharpe(r)

    obs = sh(pos)
    rng = np.random.default_rng(seed)
    ks = rng.integers(min_days, max(min_days + 1, n_days - min_days), size=n)
    null = np.array([sh(np.roll(pos, 6 * int(k))) for k in ks])
    return dict(obs=obs, p=S.null_pvalue(obs, null), median=float(np.median(null)),
                p95=float(np.percentile(null, 95)))


def selector(W, grid_r, datas, cost, days):
    """
    선택기 S: 매년 1월 1일(2019~2026), 반복마다 12개 변형의 '직전 24개월' 일별 샤프를 비교해
    가장 높은 변형으로 그해를 매매. 노출도 10~95%, 연 60회 이하 전환만 자격. 동점이면 P0.
    """
    names = list(grid_r)
    reps = min(len(v) for v in grid_r.values())
    p0name = f"g{C.P0['gamma']:g}_{C.P0['features']}_m{C.P0['c_mult']:g}"
    d0 = datas[0]
    a, b = W.rng[0]
    close_t = d0.ts[a:b] + BAR_SEC
    years = [y for y in range(2019, 2027) if _ts(f"{y}-01-01") < W.hi]
    rs, picks = [], []
    for r in range(reps):
        tg = np.full(d0.T, np.nan)
        pick_r = {}
        for y in years:
            t_y = _ts(f"{y}-01-01")
            t_n = _ts(f"{y + 1}-01-01")
            i0 = np.searchsorted(days, t_y - 730 * 86400, side="right")   # (t_y−2년, t_y] 에 끝나는 일별 수익
            i1 = np.searchsorted(days, t_y, side="right")
            j0 = np.searchsorted(close_t, t_y - 730 * 86400)
            j1 = np.searchsorted(close_t, t_y)
            best, best_sr = p0name, -np.inf
            for n in names:
                x = grid_r[n][r]
                pos = x["pos"][j0:j1]
                exp_ = pos.mean()
                per_year = np.abs(np.diff(pos)).sum() / 2.0          # 직전 2년 전환 횟수 / 2
                if not (0.10 <= exp_ <= 0.95 and per_year <= 60):
                    continue
                sr = S.sharpe(x["r"][i0:i1])
                if sr > best_sr + 1e-12 or (abs(sr - best_sr) <= 1e-12 and n == p0name):
                    best, best_sr = n, sr
            pick_r[y] = best
            k0 = np.searchsorted(close_t, t_y)
            k1 = np.searchsorted(close_t, t_n)
            tg[a + k0:a + k1] = grid_r[best][r]["pos"][k0:k1]
        # 2017~2018은 P0으로 (선택 이력이 쌓이기 전)
        k_first = np.searchsorted(close_t, _ts("2019-01-01"))
        tg[a:a + k_first] = grid_r[p0name][r]["pos"][:k_first]
        res = W.run(tg, cost)
        rs.append(res)
        picks.append(pick_r)
    # 2019-01-01 이후만 보고 (그 전은 P0과 같으므로 선택기 성적에 섞지 않음)
    i19 = np.searchsorted(days, _ts("2019-01-01"), side="right")
    srs = [S.sharpe(x["r"][i19:]) for x in rs]
    p0s = [S.sharpe(grid_r[p0name][r]["r"][i19:]) for r in range(reps)]
    lm = lower_median(srs)
    return dict(r=rs[lm]["r"][i19:], summary=dict(
        picks=picks, sharpe_reps=srs, p0_sharpe_reps=p0s, headline_rep=lm,
        sharpe=srs[lm], p0_sharpe_same_window=float(np.median(p0s)),
        cagr=S.summary(rs[lm]["r"][i19:])["cagr"], max_dd=S.summary(rs[lm]["r"][i19:])["max_dd"]))


def by_year(days, series, exposure):
    # days[i]는 r[i]가 끝나는 자정 → 그 수익이 속한 날은 하루 전 (기간 시작 기준으로 연도 배정)
    yrs = pd.to_datetime(days - 86400, unit="s", utc=True).year
    out = []
    for y in sorted(set(yrs)):
        sel = yrs == y
        row = dict(year=int(y), n_days=int(sel.sum()), exposure=float(exposure[sel].mean()))
        for k, r in series.items():
            row[k] = float(np.prod(1 + r[sel]) - 1)
            row[k + "_sharpe"] = S.sharpe(r[sel]) if sel.sum() > 2 else None
        out.append(row)
    return out


def subperiods(days, series):
    out = []
    for name, (lo, hi) in SUBPERIODS.items():
        sel = (days > _ts(lo)) & (days <= _ts(hi))
        if sel.sum() < 30:
            continue
        row = dict(name=name, lo=lo, hi=hi)
        for k, r in series.items():
            if r is None:
                continue
            rr = r[sel]
            rr = rr[~np.isnan(rr)]                       # 선택기 S는 2019년부터만
            if len(rr) < 30:
                continue
            s = S.summary(rr)
            row[k] = dict(cagr=s["cagr"], sharpe=s["sharpe"], max_dd=s["max_dd"])
        out.append(row)
    return out


def by_regime(W, head, bh, datas):
    """추세(직전 180일 수익) 3분위·변동성 국면 3분위별 노출도와 초과수익"""
    d0 = datas[0]
    a, b = W.rng[0]
    # 추세 = 직전 180일(1080봉) 로그수익 (지표 ret_180은 180봉=30일이라 따로 계산)
    lc = np.log(d0.c)
    ret180 = np.full(b - a, np.nan)
    idx = np.arange(a, b)
    okk = idx >= 1080
    ret180[okk] = lc[idx[okk]] - lc[idx[okk] - 1080]
    vr = d0.X[a:b, FI["vol_regime"]]
    m = W.m(0)
    out = {}
    for name, x in (("trend", ret180), ("vol", vr)):
        q = np.nanquantile(x, [1 / 3, 2 / 3])
        rows = []
        for j, (lo, hi) in enumerate(((-np.inf, q[0]), (q[0], q[1]), (q[1], np.inf))):
            sel = (x > lo) & (x <= hi)
            rows.append(dict(tercile=j, exposure=float(head["pos"][sel].mean()),
                             agent_bar_ret=float((head["pos"][sel] * m[sel]).mean() * 6 * 365),
                             bh_bar_ret=float(m[sel].mean() * 6 * 365)))
        out[name] = rows
    return out


def interpret(W, head, run, cd, datas):
    """지표 10분위별 보유 비율, Δ 보정(예측 Δ 10분위 vs 실현 κ·m)"""
    d0 = datas[0]
    a, b = W.rng[0]
    out = {}
    for n in ("ma_1200", "rsi_84", "macdh_1d", "vol_regime", "rsi_14", "ret_42"):
        x = d0.X[a:b, FI[n]]
        q = np.quantile(x, np.linspace(0, 1, 11))
        out[n] = [float(head["pos"][(x >= q[j]) & (x <= q[j + 1])].mean()) for j in range(10)]
    dl = run["delta"][0][cd][a:b]
    km = 100 * W.m(0)
    ok = ~np.isnan(dl)
    q = np.quantile(dl[ok], np.linspace(0, 1, 11))
    cal = []
    for j in range(10):
        sel = ok & (dl >= q[j]) & (dl <= q[j + 1])
        cal.append(dict(pred=float(dl[sel].mean()), real=float(km[sel].mean()), n=int(sel.sum())))
    out["calibration"] = cal
    return out


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="dev", choices=list(WINDOWS))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = _clean(evaluate(a.window))
    path = a.out or os.path.join(DATA_DIR, f"report_{a.window}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    ag, bh = out["agent"], out["baselines"]["B0"]
    print(f"[{a.window}] 헤드라인 반복 r{out['headline_rep']}: CAGR {ag['cagr']*100:.1f}% 샤프 {ag['sharpe']:.2f} "
          f"MDD {ag['max_dd']*100:.1f}% 노출 {ag['exposure']*100:.0f}%  |  매수·보유: CAGR {bh['cagr']*100:.1f}% "
          f"샤프 {bh['sharpe']:.2f} MDD {bh['max_dd']*100:.1f}%")
    print("판정:", out["claim"])
    print("저장:", path)


if __name__ == "__main__":
    main()
