"""느린 합성 대조 실험 — BTC_SLOW_TESTS=1 일 때만 (4코어 기준 약 10분).

  positive_control : 추세 국면(마르코프 전환)을 90% 맞히는 '심은' 지표가 있으면
                     전체 워크포워드가 매수·보유보다 유의하게 나아야 함 (p < 0.01)
  negative_control : 예측력이 전혀 없는 GARCH(1,1) 무추세 경로 10개에서는
                     평균 ΔSharpe가 2 표준오차 안이고 DSR ≥ 0.95인 경로가 없어야 함
  dqn_oracle       : 2국면 시장에서 DDQN-AA의 판단이 값반복 최적해와 95% 이상 일치

결과는 data/btc/controls.json 에 저장해 리포트에 싣습니다.
"""
import json
import math
import multiprocessing as mp
import os
import unittest

import numpy as np
import pandas as pd

SLOW = os.environ.get("BTC_SLOW_TESTS") == "1"


def _series(kind, seed, n=None):
    """합성 4시간봉 (2012-01 ~ 2026-09). kind: 'regime' 또는 'garch'"""
    from btc.env import BAR_SEC
    t0 = int(pd.Timestamp("2012-01-01", tz="UTC").timestamp())
    t1 = int(pd.Timestamp("2026-09-25", tz="UTC").timestamp())
    n = n or (t1 - t0) // BAR_SEC - 1
    rng = np.random.default_rng(seed)
    if kind == "regime":
        # 국면 지속 확률 0.995 (평균 200봉 ≈ 33일), 상승 +0.25σ, 하락 −0.25σ
        sig = 0.012
        reg = np.zeros(n, int)
        for i in range(1, n):
            reg[i] = reg[i - 1] if rng.random() < 0.995 else 1 - reg[i - 1]
        mu = np.where(reg == 1, 0.25 * sig, -0.25 * sig)
        r = mu + sig * rng.standard_normal(n)
    else:
        w, a, b = 1.5e-6, 0.08, 0.90                   # GARCH(1,1), 장기 σ ≈ 0.0122
        h = w / (1 - a - b)
        r = np.empty(n)
        for i in range(n):
            e = math.sqrt(h) * rng.standard_normal()
            r[i] = e
            h = w + a * e * e + b * h
        reg = None
    o = 1000 * np.exp(np.concatenate([[0], np.cumsum(r)[:-1]]))    # 시가→시가 수익이 r
    c = o * np.exp(rng.normal(0, 0.002, n))
    hi = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.003, n)))
    lo = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.003, n)))
    v = rng.lognormal(3, 0.4, n)
    ts = t0 + BAR_SEC * np.arange(n)
    df = pd.DataFrame(dict(ts=ts, open=o, high=hi, low=lo, close=c, volume=v, gap_before=0, forced_hold=False))
    return df, reg


def _walk(args):
    kind, seed = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from btc import config as C, features as Fe, stats as S
    from btc.env import PhaseData, simulate, daily_returns
    from btc.agent import policy_from_delta
    from btc.walkforward import run_replication, decision_range
    df, reg = _series(kind, seed)
    X, sig = Fe.compute(df)
    if kind == "regime":
        # 심은 지표: t봉 종가 시점의 '다음 결정 구간' 국면을 90% 맞힘 (ret_1 자리에 넣음)
        rng = np.random.default_rng(seed + 100)
        nxt = np.concatenate([reg[1:], reg[-1:]])
        noisy = np.where(rng.random(len(reg)) < 0.9, nxt, 1 - nxt)
        X[:, 0] = np.where(noisy == 1, 1.0, -1.0)
    d = PhaseData(df, X, sig)
    cfg = C.make(f"ctl_{kind}", phases=1)
    end = pd.Timestamp("2025-01-01", tz="UTC")
    res = run_replication(cfg, seed, eval_phases=[0], datas=[d], end=end)
    cd = round(C.c_dec(C.PRIMARY_COST, 2.0), 6)
    lo = int(pd.Timestamp("2017-01-01", tz="UTC").timestamp())
    a, b = decision_range(d, lo, int(end.timestamp()))
    tg = np.full(d.T, np.nan)
    tg[a:b] = policy_from_delta(res["delta"][0][cd][a:b], cd)
    ag = simulate(tg, d.o, d.c, C.PRIMARY_COST, a, b)
    bh_t = np.full(d.T, np.nan)
    bh_t[a:b] = 1.0
    bh = simulate(bh_t, d.o, d.c, C.PRIMARY_COST, a, b)
    _, ra = daily_returns(d.ts, ag["mark"], a, b)
    _, rb = daily_returns(d.ts, bh["mark"], a, b)
    boot = S.stationary_bootstrap_diff(ra, rb, n_boot=4000, seed=seed)
    return dict(kind=kind, seed=seed, d_sharpe=boot["obs"], p=boot["p"], sharpe=S.sharpe(ra),
                bh_sharpe=S.sharpe(rb), exposure=float(ag["pos"][a:b].mean()), r=ra.tolist())


@unittest.skipUnless(SLOW, "BTC_SLOW_TESTS=1 일 때만")
class Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        jobs = [("regime", 1)] + [("garch", s) for s in range(10)]
        with mp.get_context("spawn").Pool(4) as pool:
            cls.res = pool.map(_walk, jobs)
        out = [{k: v for k, v in r.items() if k != "r"} for r in cls.res]
        from btc import stats as S
        neg = [r for r in cls.res if r["kind"] == "garch"]
        M = np.column_stack([np.array(r["r"]) for r in neg])
        srv = S.trials_sr_var(M)
        cls.neg_dsr = [S.dsr(np.array(r["r"]), len(neg), srv) for r in neg]
        from btc.data import DATA_DIR
        with open(os.path.join(DATA_DIR, "controls.json"), "w") as f:
            json.dump(dict(runs=out, neg_dsr=cls.neg_dsr), f, indent=1)

    def test_positive_control(self):
        pos = [r for r in self.res if r["kind"] == "regime"][0]
        self.assertGreater(pos["d_sharpe"], 0)
        self.assertLess(pos["p"], 0.01)

    def test_negative_control(self):
        neg = [r["d_sharpe"] for r in self.res if r["kind"] == "garch"]
        se = np.std(neg, ddof=1) / math.sqrt(len(neg))
        self.assertLessEqual(np.mean(neg), 2 * se + 1e-9)
        self.assertTrue(all(x < 0.95 for x in self.neg_dsr), self.neg_dsr)


@unittest.skipUnless(SLOW, "BTC_SLOW_TESTS=1 일 때만")
class Oracle(unittest.TestCase):
    def test_dqn_recovers_value_iteration(self):
        from btc import config as C, features as Fe
        from btc.env import PhaseData
        from btc.agent import Trainer, policy_from_delta
        df, reg = _series("regime", 7, n=30000)
        X, sig = Fe.compute(df)
        nxt = np.concatenate([reg[1:], reg[-1:]])
        X[:, 0] = np.where(nxt == 1, 1.0, -1.0)            # 국면을 정확히 관측
        d = PhaseData(df, X, sig)
        cfg = C.make("oracle", phases=1, cold_steps=4000, cold_split=3000)
        tr = Trainer(cfg, (0, 1, 1, 0))
        tr.make_pool([d], int(df["ts"].iloc[-1]), 0, 2400)
        tr.init_fresh()
        tr.fit_cold()
        ens = tr.ensemble()
        # 값반복: 상태 = 국면(2) — 기대 κ·m, 전이확률은 생성 과정 그대로
        sig_m = 0.012
        mu = 100 * np.array([-0.25 * sig_m, 0.25 * sig_m])
        P = np.array([[0.995, 0.005], [0.005, 0.995]])
        cost = 0.003
        lnc, g = 100 * math.log(1 - cost), cfg["gamma"]
        U = np.zeros((2, 2))
        for _ in range(5000):
            U = np.array([[a * mu[s] + g * P[s] @ np.array([max(abs(b - a) * lnc + U[s2, b] for b in (0, 1))
                                                          for s2 in (0, 1)]) for a in (0, 1)] for s in (0, 1)])
        idx = np.arange(3000, 29000, 7)
        dl, _ = ens.delta(X[idx], cost)
        agree = 0
        for p in (0, 1):
            got = policy_from_delta(dl, cost, p0=p) if False else np.array(
                [policy_from_delta(np.array([x]), cost, p0=p)[0] for x in dl])
            want = np.array([policy_from_delta(np.array([U[s, 1] - U[s, 0]]), cost, p0=p)[0]
                             for s in (X[idx, 0] > 0).astype(int)])
            agree += (got == want).mean() / 2
        self.assertGreaterEqual(agree, 0.95)
