# -*- coding: utf-8 -*-
"""
Cheap pilot checks that calibrate the predictions in btc/research/rl_literature_review.md.

POST-HOC RESEARCH. The 2025-01..2026-09 lockbox was opened on 2026-09-25 by the P0 study, so this
script reads all of 2012-2026. None of it was pre-registered; every number is exploratory. It writes
only btc/research/litreview_pilots.json and never calls btc.evaluate (no lockbox audit-log write).
It is NOT an RL experiment: it tests two non-RL building blocks the literature review proposes to
put inside the next agent, so the proposals' predicted effects are anchored in BTC data.

  A. Volatility scaling (Zhang/Zohren/Roberts 2019; Lim et al. 2019; Harvey et al. 2018 via those
     papers). Report B7 (4h, 50% vol target, 2% band) gross and net, the same decided once a day with
     a 10% band, and a DOWNSIDE-deviation target (Shu/Yu/Mulvey 2024 use downside deviation as the
     risk feature). Question: is B7's loss vs. buy & hold (Sharpe 0.89 vs 1.01) a cost problem or a
     BTC "upside volatility" problem?
  B. Statistical jump model (Shu, Yu & Mulvey 2024, arXiv 2402.05272) 0/1 strategy on daily bars:
     features = EWM downside deviation (halflife 10d) and EWM Sortino (halflives 20d, 60d) of daily
     log returns; 2 states; jump penalty lambda; refit every 6 months on an expanding window;
     online inference with the exact forward (Viterbi) recursion; bull = state with the higher
     in-sample cumulative return (the paper's rule). Decide at the 00:00 UTC close, fill at the next 4h open, cost 0.15%
     (same accounting as report_full.json). Every lambda in the grid is reported (no selection), plus
     the paper's rule: pick lambda monthly by trailing validation Sharpe. A post-hoc BTC adaptation
     (K=3 states, flat only in the state with the lowest in-sample cumulative return) is reported too.

  OMP_NUM_THREADS=1 python btc/research/litreview_pilots.py      (~2.5 min, 1 process)
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import json
import math
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from btc import stats as S                                          # noqa: E402
from btc.env import BAR_SEC, daily_marks, simulate_weights          # noqa: E402
from btc.research.predictability import (DAY, END, OOS_ERAS, Ser, Win, H4,  # noqa: E402
                                         _ts, daily_only, daily_series)
from btc.walkforward import load_phases                             # noqa: E402

OUT_JSON = os.path.join(HERE, "litreview_pilots.json")
COST = 0.0015
LAMS = [5.0, 15.0, 50.0, 100.0]
REFIT_MONTHS = (1, 7)
FIRST_REFIT = "2015-01-01"
VAL_YEARS = 4.0            # trailing validation window for the lambda-selection rule
VAL_MIN_YEARS = 1.0


# ══════════ accounting helpers ══════════
def run_w(W, weights, cost, band):
    """Fractional-weight twin of Win.run (same window, fills and daily marks)."""
    s = W.s
    sim = simulate_weights(weights, s.o, s.c, cost, W.a, W.b, band=band)
    days, eq = daily_marks(s.ts, sim["mark"], W.a, W.b, bar_sec=s.bar_sec, lo=W.lo, hi=W.hi)
    pos = sim["pos"][W.a:W.b]
    n_reb = int(sim["rebal"][W.a:W.b].sum())
    return dict(days=days[1:], r=eq[1:] / eq[:-1] - 1.0, pos=pos, rebal=n_reb)


def perf(res, bh_r=None, boot=False):
    r, days = res["r"], res["days"]
    s = S.summary(r)
    yrs = len(r) / 365.0
    out = dict(sharpe=s["sharpe"], cagr=s["cagr"], max_dd=s["max_dd"],
               exposure=float(np.mean(res["pos"])))
    if "rebal" in res:
        out["rebal_per_year"] = res["rebal"] / yrs
    else:
        p = res["pos"]
        out["switches_per_year"] = float(np.sum(np.abs(np.diff(p)) > 0) / yrs)
    eras = {}
    for era, lo, hi in OOS_ERAS[:-1]:
        m = (days > _ts(lo)) & (days <= _ts(hi))
        se = S.summary(r[m])
        eras[era] = dict(sharpe=se["sharpe"], cagr=se["cagr"], max_dd=se["max_dd"])
    out["eras"] = eras
    if bh_r is not None:
        out["d_sharpe"] = out["sharpe"] - S.sharpe(bh_r)
        if boot:
            bd = S.stationary_bootstrap_diff(r, bh_r, mean_block=20, n_boot=4000, seed=1)
            out["d_sharpe_ci90"], out["p"] = bd["ci90"], bd["p"]
    return out


def ewm(x, hl):
    return pd.Series(x).ewm(halflife=hl, adjust=False).mean().to_numpy()


# ══════════ A. volatility scaling ══════════
def part_a(s4, sd, W4, bh_r):
    out = {}
    sig_ann = s4.sigma * math.sqrt(2190.0)
    w7 = np.minimum(1.0, 0.5 / sig_ann)
    out["B7_4h_band2_net"] = perf(run_w(W4, w7, COST, 0.02), bh_r)
    out["B7_4h_band2_gross"] = perf(run_w(W4, w7, 0.0, 0.02), bh_r)
    w7d = daily_only(s4, w7)
    out["B7_daily_band10_net"] = perf(run_w(W4, w7d, COST, 0.10), bh_r)
    out["B7_daily_band10_gross"] = perf(run_w(W4, w7d, 0.0, 0.10), bh_r)
    # downside deviation of daily close-to-close log returns, EWM halflife 10d, annualized
    rd = np.zeros(sd.T)
    rd[1:] = np.log(sd.c[1:] / sd.c[:-1])
    dd = np.sqrt(ewm(np.minimum(rd, 0.0) ** 2, 10.0) * 365.0)
    vol = np.sqrt(ewm(rd ** 2, 10.0) * 365.0)
    for name, risk, tgt in (("DD35", dd, 0.35), ("DD50", dd, 0.50), ("VOL50_daily_ewm10", vol, 0.50)):
        wd = np.minimum(1.0, tgt / np.maximum(risk, 1e-6))
        w = map_daily_to_4h(s4, sd, wd)
        out[f"{name}_daily_band10_net"] = perf(run_w(W4, w, COST, 0.10), bh_r)
    # which vol states carry BTC's return? forward 30d log return by tercile of trailing vol / DD
    fwd = np.full(sd.T, np.nan)
    fwd[:-31] = np.log(sd.o[31:] / sd.o[1:-30])
    m = (sd.tau >= _ts("2014-01-01")) & np.isfinite(fwd)
    tab = {}
    for name, x in (("vol_ewm10", vol), ("downside_dev_ewm10", dd), ("upside_minus_downside", vol - dd)):
        q = np.nanquantile(x[m], [1 / 3, 2 / 3])
        k = np.digitize(x, q)
        tab[name] = [float(np.nanmean(fwd[m & (k == j)]) * 365.0 / 30.0) for j in range(3)]
    out["fwd30_ann_by_tercile_2014_2026"] = tab
    return out


def map_daily_to_4h(s4, sd, val_daily):
    """Daily value decided at the day's 00:00 UTC close -> 4h bar closing at that instant; NaN elsewhere."""
    out = np.full(s4.T, np.nan)
    pos = {int(t): i for i, t in enumerate(s4.tau)}
    for j in range(sd.T):
        i = pos.get(int(sd.tau[j]))
        if i is not None and np.isfinite(val_daily[j]):
            out[i] = val_daily[j]
    return out


# ══════════ B. statistical jump model ══════════
def jm_features(sd):
    r = np.zeros(sd.T)
    r[1:] = np.log(sd.c[1:] / sd.c[:-1])
    neg2 = np.minimum(r, 0.0) ** 2
    dd10 = np.sqrt(ewm(neg2, 10.0))
    so20 = ewm(r, 20.0) / np.sqrt(ewm(neg2, 20.0) + 1e-12)
    so60 = ewm(r, 60.0) / np.sqrt(ewm(neg2, 60.0) + 1e-12)
    F = np.column_stack([dd10, so20, so60])
    F[:180] = np.nan                                  # warm-up (3 x the longest halflife)
    return F, r


def viterbi(L, lam):
    """Exact K-state jump-penalized path: argmin sum L[t, s_t] + lam * #switches (Viterbi)."""
    T, K = L.shape
    V = L[0].copy()
    back = np.zeros((T, K), np.int64)
    for t in range(1, T):
        j = int(np.argmin(V))
        stay = V
        jump = V[j] + lam
        take_jump = jump < stay
        back[t] = np.where(take_jump, j, np.arange(K))
        V = np.where(take_jump, jump, stay) + L[t]
    s = np.empty(T, np.int64)
    s[-1] = int(np.argmin(V))
    for t in range(T - 1, 0, -1):
        s[t - 1] = back[t, s[t]]
    return s


def online_states(L, lam):
    """Online inference: argmin of the forward recursion = last state of the optimal path so far."""
    V = L[0].copy()
    out = np.empty(len(L), np.int64)
    out[0] = int(np.argmin(V))
    for t in range(1, len(L)):
        V = np.minimum(V, V.min() + lam) + L[t]
        V -= V.min()                                   # keep numbers small (argmin unchanged)
        out[t] = int(np.argmin(V))
    return out


def loss_mat(Z, theta):
    return 0.5 * ((Z[:, None, :] - theta[None, :, :]) ** 2).sum(-1)


def fit_jm(Z, lam, K, rng, n_init=6, iters=30):
    best = None
    for _ in range(n_init):
        idx = [int(rng.integers(len(Z)))]                 # k-means++ initial centroids
        for _ in range(K - 1):
            d2 = np.min(((Z[:, None, :] - Z[idx][None]) ** 2).sum(-1), axis=1)
            idx.append(int(rng.choice(len(Z), p=d2 / d2.sum())))
        theta = Z[idx].copy()
        s_old = None
        for _ in range(iters):
            s = viterbi(loss_mat(Z, theta), lam)
            for k in range(K):
                if np.any(s == k):
                    theta[k] = Z[s == k].mean(0)
            if s_old is not None and np.array_equal(s, s_old):
                break
            s_old = s
        L = loss_mat(Z, theta)
        s = viterbi(L, lam)
        obj = L[np.arange(len(Z)), s].sum() + lam * np.sum(s[1:] != s[:-1])
        if best is None or obj < best[0]:
            best = (obj, theta.copy(), s.copy())
    return best[1], best[2]


def jm_signals(sd, F, r, lam, K=2, seed=0):
    """
    Daily long(1)/flat(0) signal decided at each day's close, from 2015-01 on (NaN before).
    K=2: long only in the state with the HIGHER in-sample cumulative return (Shu et al.'s rule).
    K=3: flat only in the state with the LOWEST in-sample cumulative return (BTC adaptation, post-hoc).
    """
    rng = np.random.default_rng(seed)
    ok = np.all(np.isfinite(F), axis=1)
    t0 = int(np.argmax(ok))
    refits = [d for d in pd.date_range(FIRST_REFIT, END, freq="MS", tz="UTC") if d.month in REFIT_MONTHS]
    sig = np.full(sd.T, np.nan)
    info = []
    for k, R in enumerate(refits):
        Rts = int(R.timestamp())
        nxt = int(refits[k + 1].timestamp()) if k + 1 < len(refits) else 10 ** 12
        tr = np.nonzero(ok & (sd.tau <= Rts))[0]          # features known by the refit instant
        tr = tr[tr >= t0]
        mu, sdv = F[tr].mean(0), F[tr].std(0) + 1e-12
        Z = (F[tr] - mu) / sdv
        theta, s_in = fit_jm(Z, lam, K, rng)
        # same-day return r[t] is inside the feature window that defines the state at t
        cum = np.array([np.sum(r[tr][s_in == j]) for j in range(K)])
        long_states = [int(np.argmax(cum))] if K == 2 else [j for j in range(K) if j != int(np.argmin(cum))]
        # online forward recursion restarted at the training-window start with the new centroids
        span = np.nonzero(ok & (sd.tau < nxt))[0]
        span = span[span >= tr[0]]
        Zs = (F[span] - mu) / sdv
        st = online_states(loss_mat(Zs, theta), lam)
        live = (sd.tau[span] > Rts) & (sd.tau[span] <= nxt)
        sig[span[live]] = np.isin(st[live], long_states).astype(float)
        frac = [float(np.mean(s_in == j)) for j in range(K)]
        mean_ann = [float(np.mean(r[tr][s_in == j]) * 365) if np.any(s_in == j) else None for j in range(K)]
        info.append(dict(refit=R.strftime("%Y-%m-%d"), n_train=int(len(tr)),
                         flat_frac_in=float(np.mean(~np.isin(s_in, long_states))),
                         state_frac=frac, state_mean_ann=mean_ann,
                         switches_in_per_year=float(np.sum(s_in[1:] != s_in[:-1]) / (len(tr) / 365.0))))
    return sig, info


def select_lambda(sd, sigs, s4, W4full):
    """Paper's rule (Shu et al. 3.4.3): each month pick the lambda with the best trailing validation
    Sharpe of its own online 0/1 strategy (cost 0.15%), use it for the next month."""
    rets = {}
    for lam, sg in sigs.items():
        tg = map_daily_to_4h(s4, sd, sg)
        res = W4full.run(tg, COST)
        rets[lam] = (res["days"], res["r"])
    days = rets[LAMS[0]][0]
    months = pd.date_range("2016-01-01", END, freq="MS", tz="UTC")
    chosen = np.full(sd.T, np.nan)
    picks = []
    for k, M in enumerate(months):
        Mts = int(M.timestamp())
        nxt = int(months[k + 1].timestamp()) if k + 1 < len(months) else 10 ** 12
        lo = Mts - int(VAL_YEARS * 365 * DAY)
        m = (days > lo) & (days <= Mts)
        have = [lam for lam in LAMS if np.sum(m & np.isfinite(sigs_valid_mask(days, sigs[lam], sd))) >=
                VAL_MIN_YEARS * 365]
        if not have:
            continue
        best = max(have, key=lambda lam: S.sharpe(rets[lam][1][m]) if np.std(rets[lam][1][m]) > 0 else -9)
        picks.append((M.strftime("%Y-%m"), best))
        live = (sd.tau > Mts) & (sd.tau <= nxt)
        chosen[live] = sigs[best][live]
    return chosen, picks


def sigs_valid_mask(days, sg, sd):
    """1.0 on days whose signal existed (decided at or before that day's midnight), NaN otherwise."""
    have = np.isfinite(sg)
    first = sd.tau[np.argmax(have)] if have.any() else 10 ** 12
    return np.where(days >= first, 1.0, np.nan)


# ══════════ main ══════════
def main():
    t0 = time.time()
    d = load_phases(1)[0]
    s4 = Ser(d.ts, d.o, d.h, d.l, d.c, d.v, d.X, d.sigma, d.forced_hold, BAR_SEC, H4)
    sd = daily_series(d)
    W4 = Win(s4, "2017-01-01", END)
    bh = W4.run(np.ones(s4.T), COST)
    bh_r = bh["r"]
    res = dict(meta=dict(note="post-hoc pilot; lockbox already opened; exploratory; not pre-registered",
                         cost=COST, window=["2017-01-01", END], lambdas=LAMS,
                         generated=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
    res["B0"] = perf(dict(r=bh_r, days=bh["days"], pos=bh["pos"]))
    # reference trend rules, re-run here (4h decisions, like report_full) and once a day
    s1200 = pd.Series(s4.c).rolling(1200).mean().to_numpy()
    b2 = np.where(np.isfinite(s1200), (s4.c > s1200).astype(float), np.nan)
    macd = s4.X[:, list(__import__("btc.features", fromlist=["NAMES"]).NAMES).index("macdh_1d")]
    b5 = np.where(np.isfinite(macd), (macd > 0).astype(float), np.nan)
    for name, tg in (("B2_4h", b2), ("B5_4h", b5), ("B2_daily", daily_only(s4, b2)),
                     ("B5_daily", daily_only(s4, b5))):
        res[name] = perf(W4.run(tg, COST), bh_r, boot=True)
    print(f"refs {time.time() - t0:.0f}s  B0 {res['B0']['sharpe']:.4f}  B2_4h {res['B2_4h']['sharpe']:.4f}",
          flush=True)

    res["A_vol_scaling"] = part_a(s4, sd, W4, bh_r)
    print(f"A {time.time() - t0:.0f}s", flush=True)

    F, r = jm_features(sd)
    W4full = Win(s4, "2015-01-01", END)
    jm = {}
    for K in (2, 3):
        sigs, infos = {}, {}
        for lam in LAMS:
            sigs[lam], infos[lam] = jm_signals(sd, F, r, lam, K=K)
            print(f"  JM K={K} lambda={lam:g} {time.time() - t0:.0f}s", flush=True)
        for lam in LAMS:
            key = f"K{K}_lam{lam:g}"
            tg = map_daily_to_4h(s4, sd, sigs[lam])
            jm[key] = perf(W4.run(tg, COST), bh_r, boot=True)
            jm[key]["refits"] = infos[lam]
            # robustness: one extra day of delay (paper's "trading delay")
            sg1 = np.full(sd.T, np.nan)
            sg1[1:] = sigs[lam][:-1]
            jm[key + "_delay1d"] = perf(W4.run(map_daily_to_4h(s4, sd, sg1), COST), bh_r)
        chosen, picks = select_lambda(sd, sigs, s4, W4full)
        jm[f"K{K}_selected"] = perf(W4.run(map_daily_to_4h(s4, sd, chosen), COST), bh_r, boot=True)
        jm[f"K{K}_selected_picks"] = picks
    res["B_jump_model"] = jm
    print(f"B {time.time() - t0:.0f}s", flush=True)

    def _clean(o):
        if isinstance(o, dict):
            return {str(k): _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else round(float(o), 6)
        if isinstance(o, np.integer):
            return int(o)
        return o

    with open(OUT_JSON, "w") as f:
        json.dump(_clean(res), f, indent=1)

    # ── short table on stdout
    def row(name, p):
        e = p.get("eras", {})
        g = lambda k, f: e.get(k, {}).get(f, float("nan"))
        sw = p.get("switches_per_year", p.get("rebal_per_year", float("nan")))
        print(f"{name:32s} SR {p['sharpe']:.3f}  CAGR {p['cagr']*100:6.1f}%  MDD {p['max_dd']*100:6.1f}%  "
              f"expo {p['exposure']:.2f}  sw/yr {sw:6.1f}  | SR 17-20 {g('2017-2020','sharpe'):.2f} "
              f"21-24 {g('2021-2024','sharpe'):.2f} 25-26 {g('2025-2026','sharpe'):.2f} "
              f"MDD25-26 {g('2025-2026','max_dd')*100:.0f}%" + (f"  p {p['p']:.2f}" if "p" in p else ""))
    row("B0 buy&hold", res["B0"])
    for k in ("B2_4h", "B5_4h", "B2_daily", "B5_daily"):
        row(k, res[k])
    for k, v in res["A_vol_scaling"].items():
        if isinstance(v, dict) and "sharpe" in v:
            row(k, v)
    print("fwd 30d return (ann.) by tercile:", res["A_vol_scaling"]["fwd30_ann_by_tercile_2014_2026"])
    for k, v in jm.items():
        if isinstance(v, dict) and "sharpe" in v:
            row("JM " + k, v)
    for K in (2, 3):
        pk = jm[f"K{K}_selected_picks"]
        print(f"K={K} lambda picks:", pk[:2], "...", pk[-2:], " counts:",
              {lam: sum(1 for _, x in pk if x == lam) for lam in LAMS})
    print(f"done {time.time() - t0:.0f}s -> {OUT_JSON}")


if __name__ == "__main__":
    main()
