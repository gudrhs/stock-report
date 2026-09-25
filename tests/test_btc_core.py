"""BTC 강화학습 핵심 회귀 테스트 (네트워크 없음, 약 1분).

python -m unittest tests.test_btc_core -v
느린 합성 대조 실험은 BTC_SLOW_TESTS=1 일 때만 (tests/test_btc_controls.py).
"""
import json
import math
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from btc import config as C
from btc import data as D
from btc import env as E
from btc import features as Fe
from btc.agent import Trainer, Ensemble, policy_from_delta, threshold
from btc.nn import StackedMLP, Adam

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = D.CACHE_15M


def synth_bars(n=3000, seed=0, start="2014-01-01"):
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.012, n)
    c = 1000 * np.exp(np.cumsum(r))
    o = np.concatenate([[1000], c[:-1]]) * np.exp(rng.normal(0, 0.001, n))
    h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.004, n)))
    l = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.004, n)))
    v = rng.lognormal(3, 0.5, n)
    ts = int(pd.Timestamp(start, tz="UTC").timestamp()) + np.arange(n) * E.BAR_SEC
    df = pd.DataFrame(dict(ts=ts, open=o, high=h, low=l, close=c, volume=v,
                           gap_before=0, forced_hold=False))
    return df


def minutes_frame(start, n, price=100.0, vol=1.0, seed=0):
    rng = np.random.default_rng(seed)
    ts = int(pd.Timestamp(start, tz="UTC").timestamp()) + 60 * np.arange(n)
    c = price * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    o = np.concatenate([[price], c[:-1]])
    return pd.DataFrame(dict(timestamp=ts, open=o, high=np.maximum(o, c) * 1.0005,
                             low=np.minimum(o, c) * 0.9995, close=c, volume=np.full(n, vol)))


# ──────────────────────── 데이터 ────────────────────────
class DataTests(unittest.TestCase):
    def test_resample_alignment_active_minutes_and_phase(self):
        m = minutes_frame("2020-01-01 00:00", 60 * 12)
        # 00:00~00:04 거래 없음(평봉 0거래량), 첫 거래 분은 00:05
        m.loc[:4, "volume"] = 0.0
        q = D.minutes_to_15m(m, cutoff=pd.Timestamp("2030-01-01", tz="UTC"))
        self.assertEqual(q.index[0], pd.Timestamp("2020-01-01 00:00", tz="UTC"))
        self.assertEqual(int(q["active_min"].iloc[0]), 10)
        self.assertAlmostEqual(q["open"].iloc[0], m["open"].iloc[5])            # 첫 '거래' 분의 시가
        self.assertAlmostEqual(q["close"].iloc[0], m["close"].iloc[14])         # 00:14 종가
        b0 = D.bars_4h(q, 0)
        self.assertEqual(b0.index[0], pd.Timestamp("2020-01-01 00:00", tz="UTC"))
        self.assertAlmostEqual(b0["close"].iloc[0], m["close"].iloc[239])       # [00:00, 04:00) 마지막 분
        self.assertEqual(len(b0), 3)                                            # 12시간 = 4h봉 3개
        b3 = D.bars_4h(q, 3)                                                    # 45분 밀린 격자
        self.assertEqual(b3.index[0], pd.Timestamp("2020-01-01 00:45", tz="UTC"))
        self.assertEqual(len(b3), 2)                                            # 덜 찬 앞·뒤 봉 제외
        self.assertAlmostEqual(b3["open"].iloc[0], m["open"].iloc[45])

    def test_stale_bars_dropped_and_forced_hold(self):
        m = minutes_frame("2020-01-01 00:00", 60 * 4 * 12)
        m.loc[240 * 2:240 * 9 - 1, "volume"] = 0.0      # 4h봉 7개 연속 거래 없음
        q = D.minutes_to_15m(m, cutoff=pd.Timestamp("2030-01-01", tz="UTC"))
        b = D.bars_4h(q, 0)
        self.assertEqual(len(b), 5)
        row = b.loc[pd.Timestamp("2020-01-02 12:00", tz="UTC")]
        self.assertEqual(int(row["gap_before"]), 7)
        self.assertTrue(bool(row["forced_hold"]))
        self.assertFalse(bool(b["forced_hold"].iloc[0]))

    @unittest.skipUnless(os.path.exists(CACHE), "15분봉 캐시 없음")
    def test_data_integrity(self):
        q = D.load_15m()
        ts = q["ts"].to_numpy()
        self.assertTrue(np.all(np.diff(ts) == 900))
        act = q[q["active_min"] > 0]
        self.assertFalse(act[["open", "high", "low", "close"]].isna().any().any())
        self.assertTrue(np.all(act["high"] >= np.maximum(act["open"], act["close"]) - 1e-9))
        self.assertTrue(np.all(act["low"] <= np.minimum(act["open"], act["close"]) + 1e-9))
        self.assertLess(ts[-1], int(D.CUTOFF.timestamp()))
        b = D.bars_4h(q, 0)
        self.assertEqual(int(b[b.index >= "2014-01-01"]["gap_before"].sum()), 27)   # 2015-01 해킹 26봉 + 2020-04 1봉

    @unittest.skipUnless(os.path.exists(CACHE), "15분봉 캐시 없음")
    def test_metric_consistency_buy_hold(self):
        """매수·보유 2017~2024 (일별 종가) — 설계 단계에서 따로 잰 값 CAGR 76.4%와 맞아야 함"""
        b = D.bars_4h(D.load_15m(), 0)
        s = b[(b.index >= "2016-12-31 20:00") & (b.index < "2024-12-31 20:00")]
        yrs = (len(s)) / (6 * 365.25)
        cagr = (s["close"].iloc[-1] / s["close"].iloc[0]) ** (1 / yrs) - 1
        self.assertAlmostEqual(cagr, 0.764, delta=0.015)


# ──────────────────────── 특징 ────────────────────────
class FeatureTests(unittest.TestCase):
    def test_no_lookahead_features(self):
        df = synth_bars(2600, seed=1)
        X, sig = Fe.compute(df)
        rng = np.random.default_rng(0)
        for cut in rng.integers(100, 2500, size=25):
            bad = df.copy()
            tail = bad.index > cut
            bad.loc[tail, ["open", "high", "low", "close"]] *= rng.uniform(0.2, 5, size=(tail.sum(), 1))
            bad.loc[tail, "volume"] = 0.0
            Xb, sb = Fe.compute(bad)
            np.testing.assert_array_equal(X[:cut + 1], Xb[:cut + 1])
            np.testing.assert_array_equal(sig[:cut + 1], sb[:cut + 1])

    def test_warmup_parity(self):
        """최근 2400봉만으로 시작한 엔진이 전체 이력 엔진과 1e-8 안에서 같아야 함 (실시간 백필 근거)"""
        df = synth_bars(6000, seed=2)
        X, _ = Fe.compute(df)
        Xs, _ = Fe.compute(df.iloc[6000 - 2400 - 200:].reset_index(drop=True))
        np.testing.assert_allclose(X[-200:], Xs[-200:], atol=1e-8, rtol=0)

    def test_scale_invariance(self):
        df = synth_bars(1500, seed=3)
        X, _ = Fe.compute(df)
        df2 = df.copy()
        df2[["open", "high", "low", "close"]] *= 1234.5
        df2["volume"] *= 0.01
        X2, _ = Fe.compute(df2)
        np.testing.assert_allclose(X[1300:], X2[1300:], atol=1e-9)


# ──────────────────────── 회계 ────────────────────────
class AccountingTests(unittest.TestCase):
    def setUp(self):
        self.o = np.array([100, 110, 121, 100, 90, 99, 108, 120.])
        self.c = self.o * 1.01

    def test_handcalc_two_round_trips(self):
        cost = 0.01
        tg = np.array([1, 0, 1, 1, 0, np.nan, 0, 0.])
        s = E.simulate(tg, self.o, self.c, cost, 0, 6)
        hand = (1 - cost) / 110 * 121 * (1 - cost) * (1 - cost) / 100 * 99 * (1 - cost)
        self.assertAlmostEqual(s["mark"][6], hand, places=12)
        np.testing.assert_allclose(s["trades"], [(1 - cost) ** 2 * 121 / 110 - 1, (1 - cost) ** 2 * 99 / 100 - 1])

    def test_trivial_policies(self):
        cost = 0.003
        o, c = self.o, self.c
        flat = E.simulate(np.zeros(8), o, c, cost, 0, 6)
        self.assertEqual(flat["mark"][6], 1.0)
        long_ = E.simulate(np.ones(8), o, c, cost, 0, 6)
        self.assertAlmostEqual(long_["mark"][6], (1 - cost) ** 2 * c[6] / o[1], places=12)
        flat_px = np.full(10, 50.0)
        flip = E.simulate(np.array([1, 0] * 5, float), flat_px, flat_px, cost, 0, 8)
        self.assertAlmostEqual(flip["mark"][8], (1 - cost) ** 8, places=12)

    def test_reward_equals_equity(self):
        rng = np.random.default_rng(5)
        o = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400)))
        tg = (rng.random(400) > 0.5).astype(float)
        cost = 0.0015
        s = E.simulate(tg, o, o, cost, 10, 390, exit_cost=False)
        # ΣR_t = ln(최종 자산 at O[end+1]) — 마지막 결정 이후 보유로 O[end+1]까지 평가
        pos_last = s["pos"][389]
        units_eq = s["mark"][390] / o[390] * o[391] if pos_last == 1 else s["mark"][390]
        self.assertAlmostEqual(s["logr"][10:390].sum(), math.log(units_eq), places=9)
        # 보상 공식 a·m + |a−p|·ln(1−c) 와도 같아야 함
        m = E.m_next(o)
        p = np.concatenate([[0], s["pos"][10:389]])
        R = s["pos"][10:390] * m[10:390] + np.abs(s["pos"][10:390] - p) * math.log(1 - cost)
        np.testing.assert_allclose(R, s["logr"][10:390], atol=1e-12)

    def test_fill_timing_jump_earns_nothing(self):
        """C_t와 O_{t+1} 사이 +10% 급등이 신호와 동시에 오면 그 급등은 못 먹어야 함"""
        o = np.array([100, 100, 110, 110, 110, 110.])
        c = np.array([100, 100, 100, 110, 110, 110.])   # t=1 종가 100, t=2 시가 110
        tg = np.array([0, 1, 1, 1, 1, 1.])
        s = E.simulate(tg, o, c, 0.0, 0, 4)
        self.assertAlmostEqual(s["mark"][4], 1.0, places=12)

    def test_daily_marks_phase0(self):
        ts = int(pd.Timestamp("2020-01-01", tz="UTC").timestamp()) - 4 * 3600 + E.BAR_SEC * np.arange(20)
        mark = np.linspace(1, 2, 20)
        days, eq = E.daily_marks(ts, mark, 0, 19)
        self.assertEqual(days[0], int(pd.Timestamp("2020-01-01", tz="UTC").timestamp()))
        self.assertAlmostEqual(eq[0], mark[0])
        self.assertAlmostEqual(eq[1], mark[6])


# ──────────────────────── 에이전트 ────────────────────────
class AgentTests(unittest.TestCase):
    def test_gradients_stacked_aux_l2_sp(self):
        cfg = dict(C.P0, members=2, hidden=(5, 4), batch=6)
        tr = Trainer(cfg, (1, 2, 3))
        rng = np.random.default_rng(0)
        tr.net = StackedMLP(2, [4, 5, 4, 4], rng, last_scale=0.5, dtype=np.float64)
        for i in range(1, len(tr.net.params), 2):       # ReLU 꺾임을 피하려고 편향을 무작위로
            tr.net.params[i] = rng.normal(0, 0.3, tr.net.params[i].shape)
        anchor = [p + rng.normal(0, 0.1, p.shape) for p in tr.net.params]
        X0 = rng.normal(size=(2, 6, 4))
        y = rng.normal(size=(2, 6, 2)) * 2
        z6, z42 = rng.normal(size=(2, 6)), rng.normal(size=(2, 6))
        ok = (rng.random((2, 6)) > 0.3).astype(float)
        f = lambda: tr.loss_grads(X0, y, z6, z42, ok, anchor, 0.01)[0]
        _, g, _ = tr.loss_grads(X0, y, z6, z42, ok, anchor, 0.01)
        for pi, p in enumerate(tr.net.params):
            it = np.nditer(p, flags=["multi_index"])
            for _ in it:
                i = it.multi_index
                o = p[i]
                p[i] = o + 1e-6
                a = f()
                p[i] = o - 1e-6
                b = f()
                p[i] = o
                fd = (a - b) / 2e-6
                self.assertLess(abs(fd - g[pi][i]), 1e-6 * max(1.0, abs(fd)), (pi, i))

    def test_adam_matches_hand_step(self):
        p = [np.array([[[1.0, -2.0]]])]
        opt = Adam(p, lr=0.1, clip=0)
        g = [np.array([[[0.5, -0.25]]])]
        opt.step(g)
        # 첫 스텝: m̂ = g, v̂ = g² → Δ = lr·sign(g)
        np.testing.assert_allclose(p[0], [[[0.9, -1.9]]], atol=1e-7)

    def test_decision_rule_is_argmax_of_cost_decomposed_q(self):
        rng = np.random.default_rng(1)
        for _ in range(200):
            cost = float(np.exp(rng.uniform(np.log(5e-4), np.log(0.02))))
            U = rng.normal(0, 1, 2)
            for p in (0, 1):
                q = [100 * abs(a - p) * math.log(1 - cost) + U[a] for a in (0, 1)]
                want = p if q[0] == q[1] else int(np.argmax(q))
                got = int(policy_from_delta(np.array([U[1] - U[0]]), cost, p0=p)[0])
                self.assertEqual(got, want)

    def test_cost_decomposition_matches_tabular_value_iteration(self):
        """3상태 마르코프 시장: Q(s,p,a) 값반복의 탐욕 행동 == U + 분석적 비용"""
        P = np.array([[0.8, 0.15, 0.05], [0.2, 0.6, 0.2], [0.05, 0.25, 0.7]])
        mu = np.array([-0.8, 0.1, 0.9])          # 상태별 기대 κ·m
        g, lnc = 0.9, 100 * math.log(1 - 0.004)
        Q = np.zeros((3, 2, 2))
        U = np.zeros((3, 2))
        for _ in range(2000):
            V = Q.max(axis=2)                     # V(s, p)
            for s in range(3):
                for p in (0, 1):
                    for a in (0, 1):
                        Q[s, p, a] = abs(a - p) * lnc + a * mu[s] + g * P[s] @ V[:, a]
            for s in range(3):
                for a in (0, 1):
                    U[s, a] = a * mu[s] + g * P[s] @ np.array([max(abs(b - a) * lnc + U[s2, b] for b in (0, 1))
                                                             for s2 in range(3)])
        for s in range(3):
            for p in (0, 1):
                self.assertEqual(int(Q[s, p].argmax()),
                                 int(policy_from_delta(np.array([U[s, 1] - U[s, 0]]), 0.004, p0=p)[0]))

    def test_cost_monotonicity(self):
        rng = np.random.default_rng(2)
        delta = np.cumsum(rng.normal(0, 0.3, 3000)) * 0.1
        prev = None
        for c in (0.0005, 0.001, 0.002, 0.004, 0.008, 0.016):
            sw = np.abs(np.diff(policy_from_delta(delta, c))).sum()
            if prev is not None:
                self.assertLessEqual(sw, prev)
            prev = sw

    def test_determinism_and_member_diversity(self):
        df = synth_bars(3200, seed=4)
        X, sig = Fe.compute(df)
        d = E.PhaseData(df, X, sig)
        cfg = dict(C.P0, phases=1, cold_steps=30, cold_split=20)
        Tk = int(df["ts"].iloc[-1])
        outs = []
        for _ in range(2):
            tr = Trainer(cfg, (0, 2020, 1, 0))
            tr.make_pool([d], Tk, 0, 2400)
            tr.init_fresh()
            tr.fit_cold()
            outs.append([p.copy() for p in tr.net.params])
        for a, b in zip(*outs):
            np.testing.assert_array_equal(a, b)
        self.assertFalse(np.allclose(outs[0][0][0], outs[0][0][1]))
        tr2 = Trainer(cfg, (1, 2020, 1, 0))
        tr2.make_pool([d], Tk, 0, 2400)
        tr2.init_fresh()
        tr2.fit_cold()
        self.assertFalse(np.allclose(tr2.net.params[0], outs[0][0]))

    def test_training_pool_respects_purge(self):
        """T_k에 학습하는 표본은 모두 t+43봉 종가 ≤ T_k (보조 목표·다음 상태·보상 전부 T_k 이전)"""
        df = synth_bars(4000, seed=6)
        X, sig = Fe.compute(df)
        d = E.PhaseData(df, X, sig)
        Tk = int(df["ts"].iloc[3500])
        e = d.eligible(Tk, 0, 2400)
        self.assertTrue(np.all(d.ts[e + 43] + E.BAR_SEC <= Tk))
        self.assertTrue(np.all(d.ts[e + 2] < Tk))
        self.assertGreater(len(e), 900)


# ──────────────────────── 잠금 구간·시험 기록 ────────────────────────
class GuardTests(unittest.TestCase):
    def test_lockbox_guard(self):
        from btc import evaluate as Ev
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(Ev, "AUDIT", os.path.join(tmp, "audit.log")), \
                    patch.dict(os.environ, {"BTC_LOCKBOX_OPEN": ""}):
                with self.assertRaises(Ev.Sealed):
                    Ev.guard(Ev.LOCK_TS + 1, "test")
                Ev.guard(Ev.LOCK_TS, "dev ok")
                self.assertFalse(os.path.exists(os.path.join(tmp, "audit.log")))
            with patch.object(Ev, "AUDIT", os.path.join(tmp, "audit.log")), \
                    patch.dict(os.environ, {"BTC_LOCKBOX_OPEN": "1"}):
                Ev.guard(Ev.LOCK_TS + 1, "opened for test")
                with open(os.path.join(tmp, "audit.log")) as f:
                    self.assertIn("opened for test", f.read())

    def test_trial_logging_counts_distinct_hashes(self):
        from btc import walkforward as W
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(W, "TRIALS", os.path.join(tmp, "t.jsonl")), patch.object(W, "RUNS_DIR", tmp):
                W.log_trial(C.P0, "x")
                W.log_trial(C.P0, "x")
                self.assertEqual(W.n_trials(), 1)
                W.log_trial(C.make("v", gamma=0.9), "x")
                self.assertEqual(W.n_trials(), 2)


# ──────────────────────── 실시간 ────────────────────────
class LiveTests(unittest.TestCase):
    def test_hourly_to_4h_and_in_progress_guard(self):
        from btc import live as L
        t0 = int(pd.Timestamp("2023-01-01", tz="UTC").timestamp())
        h = pd.DataFrame(dict(ts=t0 + 3600 * np.arange(14), open=1.0, high=2.0, low=0.5, close=1.5, volume=1.0))

        class Fake(L.Venue):
            def fetch_hourly(self, n, now=None):
                return self.closed_only(h, now)
        now = t0 + 3600 * 13 + 1800                  # 13시 봉은 아직 형성 중
        got = Fake().fetch_hourly(20, now=now)
        self.assertEqual(int(got["ts"].max()), t0 + 3600 * 12)
        bars = L.hourly_to_4h(got)
        self.assertEqual(list(bars["ts"]), [t0, t0 + 4 * 3600, t0 + 8 * 3600])   # 12~16시 봉은 미완성
        h2 = h.drop(index=5)                          # 05시 봉 누락 → 04~08시 봉은 죽은 봉
        bars2 = L.hourly_to_4h(h2)
        self.assertNotIn(t0 + 4 * 3600, list(bars2["ts"]))
        self.assertEqual(int(bars2.set_index("ts").loc[t0 + 8 * 3600, "gap_before"]), 1)


class LiveReplayTests(unittest.TestCase):
    """과거 1시간봉을 실시간처럼 흘려 넣은 모의매매 == 같은 봉의 백테스트 (판단·체결·자산 100% 일치)"""

    @unittest.skipUnless(os.path.exists(CACHE), "15분봉 캐시 없음")
    def test_live_replay_matches_backtest(self):
        import pickle
        from btc import live as L
        q = D.load_15m()
        q = q[(q.index >= "2021-01-01") & (q.index < "2023-07-01")]
        g = q.resample("1h", label="left", closed="left")
        hr = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                           "close": g["close"].last(), "volume": g["volume"].sum()}).dropna()
        hr.insert(0, "ts", D.to_unix(hr.index))
        hr = hr.reset_index(drop=True)
        hole = int(pd.Timestamp("2023-03-10 09:00", tz="UTC").timestamp())
        hr = hr[hr["ts"] != hole].reset_index(drop=True)           # 시간봉 하나 누락 → 그 4시간봉은 죽은 봉
        late = int(pd.Timestamp("2023-02-01 03:00", tz="UTC").timestamp())

        class Mock(L.Venue):
            calls = 0

            def fetch_hourly(self, n, now=None):
                Mock.calls += 1
                ok = hr["ts"] + 3600 <= now
                ok &= ~((hr["ts"] == late) & (now < late + 3600 + 95))  # 95초 늦게 도착
                df = hr[ok].tail(n)
                if Mock.calls % 7 == 0:                              # 중복 행 섞기
                    df = pd.concat([df, df.tail(3)])
                # 형성 중인 봉도 섞어서 보냄 → 버려야 함
                forming = hr[(hr["ts"] <= now) & (hr["ts"] + 3600 > now)]
                return self.closed_only(pd.concat([df, forming]), now)

        rng = np.random.default_rng(0)
        net = StackedMLP(3, [len(Fe.NAMES) + 1, 16, 16, 4], rng, last_scale=3.0)
        ens = Ensemble(net, "full22")
        trader = L.PaperTrader(ens, cost=0.0015, model_tag="test")
        t_start = int(pd.Timestamp("2023-01-01", tz="UTC").timestamp())
        t_end = int(pd.Timestamp("2023-06-30", tz="UTC").timestamp())
        with tempfile.TemporaryDirectory() as tmp:
            sp = os.path.join(tmp, "state.pkl")
            for i, bnd in enumerate(range(t_start, t_end, 4 * 3600)):
                L.process(trader, Mock(), now=bnd + 20, backfill_hours=24 * 900, sleep=lambda s: None)
                if i % 50 == 25:                                     # 강제 종료·재시작
                    L.save_state(trader, sp)
                    trader = L.load_state(sp)
        # 백테스트: 같은 시간봉 → 같은 4시간봉 → 같은 특징 → 같은 규칙
        bars = L.hourly_to_4h(hr[hr["ts"] + 3600 <= t_end])
        X, _ = Fe.compute(bars)
        ts = bars["ts"].to_numpy()
        logs = [r for r in trader.log if r.get("decision") is not None]
        a = int(np.searchsorted(ts, logs[0]["ts"]))
        b = int(np.searchsorted(ts, logs[-1]["ts"])) + 1
        self.assertEqual(b - a, len(logs))                              # 한 봉도 빠지거나 겹치지 않음
        dl, _ = ens.delta(X[a:b], C.c_dec(0.0015, 2.0))
        pos = policy_from_delta(dl, C.c_dec(0.0015, 2.0), bars["forced_hold"].to_numpy()[a:b])
        np.testing.assert_array_equal(pos, [r["decision"] for r in logs])
        self.assertGreater(np.abs(np.diff(pos)).sum(), 4)               # 실제로 매매가 일어났어야 의미 있음
        tg = np.full(len(bars), np.nan)
        tg[a:b] = pos
        sim = E.simulate(tg, bars["open"].to_numpy(), bars["close"].to_numpy(), 0.0015, a, b - 1,
                         forced_hold=bars["forced_hold"].to_numpy(), exit_cost=False)
        eq_live = np.array([r["equity"] for r in logs])
        np.testing.assert_allclose(eq_live[1:], sim["mark"][a + 1:b], rtol=1e-12)


class ReviewRegressionTests(unittest.TestCase):
    """코드 리뷰에서 나온 문제들이 다시 생기지 않게"""

    def test_missing_minute_rows_count_as_dead_bars(self):
        m = minutes_frame("2020-01-01 00:00", 60 * 4 * 10)
        m = m.drop(index=range(240 * 3, 240 * 3 + 30)).reset_index(drop=True)   # 네 번째 4h봉 30분 행 누락
        q = D.minutes_to_15m(m, cutoff=pd.Timestamp("2030-01-01", tz="UTC"))
        b = D.bars_4h(q, 0)
        self.assertNotIn(pd.Timestamp("2020-01-01 12:00", tz="UTC"), b.index)
        self.assertEqual(int(b.loc[pd.Timestamp("2020-01-01 16:00", tz="UTC"), "gap_before"]), 1)
        X, sig = Fe.compute(b)
        d = E.PhaseData(b, X, sig)
        i = int(np.searchsorted(d.ts, int(pd.Timestamp("2020-01-01 08:00", tz="UTC").timestamp())))
        self.assertFalse(d.contig[i - 1])            # t..t+2가 구멍을 건너면 학습 표본에서 제외

    def test_flatten_cache_never_serves_other_objects(self):
        from btc import agent as A
        mk = lambda seed: E.PhaseData(synth_bars(300, seed), *Fe.compute(synth_bars(300, seed)))
        d1 = mk(1)
        f1 = A._flatten([d1])["X"].copy()
        d2 = mk(2)
        f2 = A._flatten([d2])["X"]
        self.assertFalse(np.array_equal(f1, f2))
        np.testing.assert_array_equal(f2, d2.X)

    def test_last_decision_cost_in_logr(self):
        o = np.array([100., 101, 102, 103])
        s = E.simulate(np.array([0, 0, 1, np.nan]), o, o, 0.001, 0, 3, exit_cost=False)
        self.assertAlmostEqual(s["logr"][2], math.log(1 - 0.001), places=12)
        self.assertAlmostEqual(s["logr"][:3].sum(), math.log(s["mark"][3]), places=12)

    def test_offset_phase_daily_marks_include_entry_and_exit_cost(self):
        cost = 0.0015
        lo = int(pd.Timestamp("2020-01-01", tz="UTC").timestamp())
        hi = lo + 5 * 86400
        for k in (0, 1, 8, 15):
            ts = lo - 4 * 3600 + 900 * k + E.BAR_SEC * np.arange(40)
            px = np.full(40, 100.0)
            a = int(np.searchsorted(ts + E.BAR_SEC, lo))
            b = int(np.searchsorted(ts + E.BAR_SEC, hi))
            while ts[b] + E.BAR_SEC > hi:
                b -= 1
            s = E.simulate(np.ones(40), px, px, cost, a, b)
            days, eq = E.daily_marks(ts, s["mark"], a, b, lo=lo, hi=hi)
            self.assertEqual(days[0], lo)
            self.assertEqual(days[-1], hi)
            self.assertAlmostEqual(eq[-1] / eq[0], (1 - cost) ** 2, places=12, msg=f"phase {k}")

    def test_weights_simulator_counts_rebalances_not_drift(self):
        rng = np.random.default_rng(0)
        o = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 600)))
        s = E.simulate_weights(np.full(600, 0.5), o, o, 0.001, 0, 598)
        self.assertLess(int(s["rebal"].sum()), 60)
        self.assertGreater(len(np.unique(np.round(s["pos"][:598], 6))), 100)   # 비중은 흔들리지만 거래는 드묾

    def test_n3_null_does_not_flag_reactive_rule_without_edge(self):
        from btc import evaluate as Ev
        rng = np.random.default_rng(3)
        n_days = 1500
        m = rng.normal(0, 0.012, n_days * 6)
        mom = np.convolve(m, np.ones(6), "full")[:len(m)]           # 직전 1일 수익 (t봉까지)
        pos = (np.concatenate([[0], mom[:-1]]) > 0).astype(float)   # 인과적 1일 모멘텀
        day_id = np.arange(len(m)) // 6
        res = Ev.null_shift_bar(pos, m, 0.0015, day_id, n_days, n=400)
        self.assertGreater(res["p"], 0.02)

    def test_failed_first_gate_never_trades(self):
        from btc import walkforward as W
        df = synth_bars(21000, seed=9, start="2012-06-01")
        X, sig = Fe.compute(df)
        d = E.PhaseData(df, X, sig)
        cfg = C.make("t", phases=1, cold_steps=20, cold_split=10, ft_steps=5)
        calls = []

        def fake_gate(ens, d0, T_k, cost=0.003):
            calls.append(T_k)
            return (len(calls) >= 3), dict(exposure=0.99)
        with patch.object(W, "sanity_ok", fake_gate):
            res = W.run_replication(cfg, 0, eval_phases=[0], datas=[d], c_dec_list=[0.003],
                                    end=pd.Timestamp("2017-05-01", tz="UTC"))
        kinds = [(e["month"], e.get("kind"), e.get("accepted")) for e in res["log"]]
        self.assertEqual([k[1] for k in kinds[:3]], ["cold", "cold", "cold"])  # 채택 전까지 매달 다시 처음부터
        a, b = W.decision_range(d, int(pd.Timestamp("2017-01-01", tz="UTC").timestamp()),
                                int(pd.Timestamp("2017-03-01", tz="UTC").timestamp()))
        self.assertTrue(np.isnan(res["delta"][0][0.003][a:b]).all())         # 점검 탈락 모델은 매매 안 함
        a2, b2 = W.decision_range(d, int(pd.Timestamp("2017-03-01", tz="UTC").timestamp()),
                                  int(pd.Timestamp("2017-04-01", tz="UTC").timestamp()))
        self.assertFalse(np.isnan(res["delta"][0][0.003][a2:b2]).any())

    def test_live_model_hot_swap_and_kill_switch_without_fallback(self):
        from btc import live as L
        rng = np.random.default_rng(1)
        ens_a = Ensemble(StackedMLP(2, [23, 8, 8, 4], rng, last_scale=0.1), "full22")
        ens_b = Ensemble(StackedMLP(2, [23, 8, 8, 4], rng, last_scale=0.1), "full22")
        tr = L.PaperTrader(ens_a, model_tag="p0_r0_2026-09-01")
        with tempfile.TemporaryDirectory() as tmp:
            st = ens_b.state()
            np.savez(os.path.join(tmp, "p0_latest.npz"), **st,
                     tag=np.frombuffer(b"p0_r0_2026-10-01", dtype=np.uint8))
            self.assertTrue(L.maybe_swap_model(tr, tmp))
            self.assertIs(tr.fallback, ens_a)
            self.assertEqual(tr.model_tag, "p0_r0_2026-10-01")
            self.assertFalse(L.maybe_swap_model(tr, tmp))              # 같은 모델이면 교체 안 함
        # |U| ≥ 200 이면 대체 모델이 없어도 관망
        big = Ensemble(StackedMLP(2, [23, 8, 8, 4], rng, last_scale=1e4), "full22")
        tr2 = L.PaperTrader(big, model_tag="x")
        df = synth_bars(2500, seed=3)
        for i in range(2500):
            rec = tr2.on_bar(df.iloc[i], trade=True)
        self.assertEqual(rec["hold"], "model_bad")
        self.assertEqual(rec["decision"], rec["pos"])
