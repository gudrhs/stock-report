"""2단계 확증 시험 데이터 적재기 테스트 (네트워크 없음, 학습 없음, 수 초).

python -m unittest tests.test_btc_holdout -v

맹검 규칙: 새 코인 파일은 구조(정렬·중복·가격>0·고저 모순·거래량≥0)만 확인하고 수익률 통계는 보지 않습니다.
"""
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from btc import data as D
from btc import stats as S
from btc.env import BAR_SEC
from btc.features import compute, NAMES
from btc.research import holdout as H
from btc.research import holdout_data as HD


def synth_15m(days=70, seed=0, start="2020-03-01", dead=None, missing=None):
    """합성 15분봉 (btc/data.py 캐시와 같은 컬럼). dead=(시작 번호, 길이): 거래 없는 구간, missing: 빠진 15분 행 번호"""
    rng = np.random.default_rng(seed)
    n = days * 96
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    o = np.concatenate([[100.0], c[:-1]])
    h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.001, n)))
    l = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.001, n)))
    v = rng.lognormal(2, 0.5, n)
    am = rng.integers(8, 16, n)
    ts = int(pd.Timestamp(start, tz="UTC").timestamp()) + np.arange(n) * 900
    df = pd.DataFrame(dict(ts=ts, open=o, high=h, low=l, close=c, volume=v, active_min=am))
    if dead is not None:
        s, k = dead
        df.loc[s:s + k - 1, ["open", "high", "low", "close"]] = np.nan
        df.loc[s:s + k - 1, "volume"] = 0.0
        df.loc[s:s + k - 1, "active_min"] = 0
    if missing is not None:
        df = df.drop(index=missing).reset_index(drop=True)
    return df


def agg(df, rule_sec):
    """하위 봉 → rule_sec 봉 (first/max/min/last/sum), active_min 합"""
    x = df.copy()
    x.index = pd.to_datetime(x["ts"], unit="s", utc=True)
    g = x.resample(f"{rule_sec}s", label="left", closed="left")
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                        "close": g["close"].last(), "volume": g["volume"].sum(), "active_min": g["active_min"].sum()})
    out.insert(0, "ts", ((out.index - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)).to_numpy(np.int64))
    return out.reset_index(drop=True)


class TestLoader(unittest.TestCase):
    def test_15m_phases_aligned_and_equal_to_core(self):
        df = synth_15m(dead=(3000, 30 * 4), missing=[5000])      # 30시간 거래 없음(죽은 봉 7개 이상) + 15분 행 하나 빠짐
        pds = H.phases_from_frame(df)
        self.assertEqual(len(pds), 16)
        sub, res, est = H.prepare_sub(df)
        self.assertEqual((res, est), (900, False))
        for k, d in enumerate(pds):
            self.assertTrue(np.all(d.ts % BAR_SEC == 900 * k), k)
            self.assertTrue(np.all(np.diff(d.ts) > 0))
            self.assertEqual(d.X.shape, (d.T, len(NAMES)))
            self.assertTrue(np.all(np.isfinite(d.X)))
            self.assertEqual(len(d.sigma), d.T)
            b = D.bars_4h(sub, k)                                   # 핵심 코드와 봉·특징이 같음
            np.testing.assert_array_equal(d.ts, b["ts"].to_numpy())
            np.testing.assert_array_equal(d.o, b["open"].to_numpy())
            np.testing.assert_array_equal(d.gap, b["gap_before"].to_numpy())
            X, _ = compute(b)
            np.testing.assert_array_equal(d.X, X.astype(np.float32))
        d0 = pds[0]
        self.assertGreaterEqual(int(d0.gap.max()), D.FORCED_HOLD_GAP)
        self.assertTrue(d0.forced_hold.any())
        # 빠진 15분 행이 든 4시간봉은 죽은 봉으로 빠짐 (phase 0)
        t_miss = int(df["ts"].iloc[4999]) + 900
        self.assertFalse(np.any((d0.ts <= t_miss) & (d0.ts + BAR_SEC > t_miss)))
        # 4시간봉 값 = 그 안 15분봉 16개의 first/max/min/last/sum
        i = 10
        w = df[(df["ts"] >= d0.ts[i]) & (df["ts"] < d0.ts[i] + BAR_SEC)]
        self.assertEqual(len(w), 16)
        self.assertEqual(d0.o[i], w["open"].iloc[0])
        self.assertEqual(d0.h[i], w["high"].max())
        self.assertEqual(d0.l[i], w["low"].min())
        self.assertEqual(d0.c[i], w["close"].iloc[-1])
        self.assertAlmostEqual(d0.v[i], w["volume"].sum(), places=9)

    def test_1h_generic_matches_15m_grid(self):
        """1시간봉 일반화 경로: 4개 격자(1시간씩), 15분봉에서 만든 phase 4k 와 봉 값이 같음"""
        df = synth_15m(days=40, seed=1, dead=(1500, 40))
        h1 = agg(df, 3600)
        sub15, _, _ = H.prepare_sub(df)
        sub1h, res, est = H.prepare_sub(h1)
        self.assertEqual((res, est), (3600, False))
        self.assertEqual(H.n_phases_for(res), 4)
        for k in range(4):
            a = H.bars_4h_any(sub1h, k, res)
            b = D.bars_4h(sub15, 4 * k)
            self.assertTrue(np.all(a["ts"].to_numpy() % BAR_SEC == 3600 * k))
            pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        pds = H.phases_from_frame(h1)
        self.assertEqual(len(pds), 4)

    def test_missing_active_min_uses_bar_count(self):
        df = synth_15m(days=20, seed=2).drop(columns="active_min")
        df.loc[100:130, "volume"] = 0.0                      # 거래 없는 15분봉 → 가격 비우고 활동 0
        sub, res, est = H.prepare_sub(df)
        self.assertTrue(est)
        self.assertTrue((sub["active_min"].iloc[100:131] == 0).all())
        self.assertTrue(sub["close"].iloc[100:131].isna().all())
        self.assertTrue((sub["active_min"].iloc[:100] == 15).all())

    def test_fivemin_to_15m_matches_minutes(self):
        """5분봉 경로(holdout_data)는 1분봉 경로(btc.data.minutes_to_15m)와 가격·거래량이 같고 활동 분은 상한"""
        rng = np.random.default_rng(3)
        n = 3 * 24 * 60
        c = 50 * np.exp(np.cumsum(rng.normal(0, 0.0005, n)))
        o = np.concatenate([[50.0], c[:-1]])
        h, l = np.maximum(o, c) * 1.0002, np.minimum(o, c) * 0.9998
        v = rng.lognormal(0, 1, n) * (rng.random(n) > 0.3)          # 30% 분은 거래 없음
        t0 = int(pd.Timestamp("2021-01-01", tz="UTC").timestamp())
        m = pd.DataFrame(dict(timestamp=t0 + 60 * np.arange(n), open=o, high=h, low=l, close=c, volume=v))
        q1 = D.minutes_to_15m(m)
        a = m[m["volume"] > 0].copy()
        a["b"] = a["timestamp"] // 300 * 300
        g = a.groupby("b")
        m5 = pd.DataFrame({"timestamp": g["timestamp"].first().index, "open": g["open"].first().to_numpy(),
                           "high": g["high"].max().to_numpy(), "low": g["low"].min().to_numpy(),
                           "close": g["close"].last().to_numpy(), "volume": g["volume"].sum().to_numpy(),
                           "trades": g["open"].count().to_numpy()})
        q5 = HD.fivemin_to_15m(m5, "omit")
        j = q1.join(q5, rsuffix="_5", how="inner")
        self.assertGreater(len(j), 250)
        for k in ("open", "high", "low", "close", "volume"):
            np.testing.assert_allclose(j[k].to_numpy(), j[k + "_5"].to_numpy(), rtol=1e-12)
        self.assertTrue((j["active_min_5"] >= j["active_min"]).all())

    def test_eval_window_rule(self):
        """판정 시작 = max(데이터 시작, 예열 끝) + 2년 이후 첫 1월 1일 (2026-09-25 사전 등록 규칙)"""
        T = lambda x: pd.Timestamp(x, tz="UTC")
        end = T("2026-09-24 23:45")
        for start, warm, want in (("2015-08-07 14:00", "2017-04-12", "2020-01-01"),
                                  ("2017-12-13 03:45", "2019-01-17", "2022-01-01"),
                                  ("2016-01-01 00:00", "2016-01-01", "2018-01-01"),
                                  ("2016-01-01 00:00", "2017-01-01 00:00", "2019-01-01"),
                                  ("2018-05-04 08:15", "2018-05-04 08:15", "2021-01-01")):
            lo, hi = H.window_rule(T(start), end, T(warm))
            self.assertEqual(lo, T(want), (start, warm))
            self.assertEqual(hi, T("2026-09-24"))

    def test_pooled_bootstrap_single_asset_equals_stats(self):
        rng = np.random.default_rng(5)
        n = 800
        a, b = rng.normal(0.001, 0.02, n), rng.normal(0.0005, 0.03, n)
        days = 1_600_000_000 // 86400 * 86400 + 86400 * np.arange(n)
        pb = H.pooled_bootstrap([(days, a, b)], n_boot=600, seed=1)
        st = S.stationary_bootstrap_diff(a, b, mean_block=20, n_boot=600, seed=1)
        self.assertAlmostEqual(pb["obs"], st["obs"], places=12)
        self.assertEqual(pb["p"], st["p"])
        # 두 자산(기간 다름): 관측값은 자산별 ΔSR의 평균
        c, d = rng.normal(0.0, 0.02, 500), rng.normal(0.0, 0.02, 500)
        pb2 = H.pooled_bootstrap([(days, a, b), (days[300:], c, d)], n_boot=300, seed=1)
        self.assertAlmostEqual(pb2["obs"], ((S.sharpe(a) - S.sharpe(b)) + (S.sharpe(c) - S.sharpe(d))) / 2, places=12)
        self.assertTrue(0.0 <= pb2["p"] <= 1.0)


@unittest.skipUnless(os.path.exists(D.CACHE_15M), "BTC 15분봉 캐시 없음")
class TestBtcEquivalence(unittest.TestCase):
    """비트코인 15분봉 캐시를 holdout 적재기로 읽으면 load_phases()[0] 과 같아야 함"""

    @classmethod
    def setUpClass(cls):
        from btc.walkforward import load_phases
        cls.ref = load_phases(1)[0]
        cls.df = pd.read_csv(D.CACHE_15M)

    def test_phase0_identical(self):
        mine = H.phases_from_frame(self.df, phases=[0])[0]
        ref = self.ref
        np.testing.assert_array_equal(mine.ts, ref.ts)
        diff = float(np.abs(mine.X - ref.X).max())
        print(f"\n  BTC phase 0: {ref.T}봉, 특징 최대 절대차 {diff:g}, σ 최대차 {np.abs(mine.sigma - ref.sigma).max():g}")
        self.assertLessEqual(diff, 1e-7)
        np.testing.assert_array_equal(mine.o, ref.o)
        np.testing.assert_array_equal(mine.c, ref.c)
        np.testing.assert_array_equal(mine.gap, ref.gap)
        np.testing.assert_array_equal(mine.forced_hold, ref.forced_hold)

    def test_phase0_without_active_min(self):
        """활동 분 수가 없는 파일용 대체 규칙(거래 있던 15분봉 × 15분)의 차이 — 비트스탬프 초기(2012~13) 한산한 봉만 달라짐"""
        fb = H.phases_from_frame(self.df.drop(columns="active_min"), phases=[0])[0]
        ref = self.ref
        common, ia, ib = np.intersect1d(ref.ts, fb.ts, return_indices=True)
        self.assertEqual(len(common), ref.T)                         # 대체 규칙은 봉을 덜 거르기만 함
        d = np.abs(ref.X[ia] - fb.X[ib]).max(axis=1)
        extra_years = sorted(set(pd.to_datetime(np.setdiff1d(fb.ts, ref.ts), unit="s").year))
        late = common >= int(pd.Timestamp("2016-01-01", tz="UTC").timestamp())
        print(f"\n  대체 규칙: 더 살아남은 봉 {fb.T - ref.T}개 (연도 {extra_years}), 겹치는 봉 특징 최대차 {d.max():.4g}, "
              f"2016년 이후 최대차 {d[late].max():g}")
        self.assertTrue(set(extra_years) <= {2012, 2013})
        self.assertEqual(float(d[late].max()), 0.0)


class TestHoldoutFiles(unittest.TestCase):
    """저장된 새 코인 파일의 구조만 확인 (수익률 통계 없음)"""

    def test_files(self):
        files = sorted(glob_files())
        if not files:
            self.skipTest("data/btc/holdout 파일 없음")
        for p in files:
            q = pd.read_csv(p)
            self.assertEqual(list(q.columns[:6]), ["ts", "open", "high", "low", "close", "volume"], p)
            ts = q["ts"].to_numpy(np.int64)
            self.assertTrue(np.all(np.diff(ts) > 0), p)
            self.assertTrue(np.all(ts % 900 == 0), p)
            ok = q["close"].notna()
            o, h, l, c = (q.loc[ok, k].to_numpy(float) for k in ("open", "high", "low", "close"))
            self.assertTrue(np.all((o > 0) & (h > 0) & (l > 0) & (c > 0)), p)
            self.assertTrue(np.all(h >= np.maximum(o, c) - 1e-9 * np.maximum(o, c)), p)
            self.assertTrue(np.all(l <= np.minimum(o, c) + 1e-9 * np.minimum(o, c)), p)
            self.assertTrue(np.all(q["volume"].to_numpy(float) >= 0), p)
            self.assertLess(os.path.getsize(p), 15e6, p)


class TestPlumbing(unittest.TestCase):
    """실행기·판정기 배관 — 합성 코인과 가짜 U(학습 없음)로만. 실제 새 코인은 건드리지 않음."""

    def setUp(self):
        from unittest.mock import patch
        self.td = tempfile.TemporaryDirectory()
        root = self.td.name
        self.patches = [patch.object(H, "HOLDOUT_DIR", os.path.join(root, "holdout")),
                        patch.object(H, "CACHE_DIR", os.path.join(root, "cache")),
                        patch.object(H, "RUNS_H", os.path.join(root, "runs")),
                        patch.object(H, "LEDGER", os.path.join(root, "runs", "confirmatory.jsonl"))]
        for p in self.patches:
            p.start()
        os.makedirs(H.HOLDOUT_DIR)
        df = synth_15m(days=int(365 * 5.2), seed=7, start="2019-12-01")
        df.to_csv(os.path.join(H.HOLDOUT_DIR, "SYN_15m.csv.gz"), index=False)

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.td.cleanup()

    def test_months_patch_and_evaluate(self):
        from unittest.mock import patch
        from btc.research import walk
        from btc.research.variants import VARIANTS
        cfg = VARIANTS["R6_daily_trend8_uniform"]
        datas = H.load_asset("SYN")
        self.assertEqual(len(datas), 16)
        lo, hi = H.eval_window("SYN")
        self.assertEqual(lo, pd.Timestamp("2024-01-01", tz="UTC"))      # 예열 400일 + 2년 뒤 첫 1월 1일
        fm = H.first_month(cfg, datas, hi)
        # 첫 학습 달: 학습 표본이 있고 최근 2년 봉이 90% 이상인 첫 달
        Tk = int(fm.timestamp())
        self.assertTrue(H._pool_nonempty(cfg, datas, Tk) and not H._gate_short(cfg, datas[0], Tk))
        prev = fm - pd.DateOffset(months=1)
        self.assertTrue(H._gate_short(cfg, datas[0], int(prev.timestamp()))
                        or not H._pool_nonempty(cfg, datas, int(prev.timestamp())))
        seen = {}

        def fake_run(c, r, datas=None, end=None):             # 학습 대신 달력만 기록하고 가짜 U
            seen["months"] = walk.months()
            seen["end"] = end
            d0 = datas[0]
            rng = np.random.default_rng(r)
            U = {cd: rng.normal(0, 1, (d0.T, 2)).astype(np.float32) for cd in walk.C_DECS}
            return dict(U=U, log=[dict(month=str(m.date()), kind="fake") for m in walk.months() if m < end],
                        last_state={})

        orig = walk.months
        with patch.object(walk, "run_replication", fake_run):
            for r in range(3):
                res = H.run_replication("SYN", cfg, r, datas=datas)
                H.save_run("SYN", cfg, r, res, "test")
        self.assertIs(walk.months, orig)                        # 바꾼 달력은 원래대로 돌아옴
        self.assertEqual(seen["months"][0], fm)
        self.assertEqual(seen["end"], hi)
        run = H.load_run("SYN", cfg["name"], 0)
        self.assertEqual(run["meta"]["first_month"], str(fm.date()))
        self.assertEqual(run["log"][0]["kind"], "skipped")
        with self.assertRaises(SystemExit):                     # 확증 판정은 사전 등록한 자산 4개·반복 5만
            H.evaluate(["SYN"], [cfg["name"]], reps=3)
        out = H.evaluate(["SYN"], [cfg["name"]], reps=3, exploratory=True)
        res = out["results"][cfg["name"]]
        self.assertEqual(set(res), {"per_asset", "pooled", "H1", "H2", "H3", "passed"})
        self.assertEqual(len(res["per_asset"]["SYN"]["sharpe_reps"]), 3)
        self.assertFalse(out["asset_set_matches_predeclared"])
        self.assertEqual(H.candidates_so_far(), [cfg["name"]])
        import contextlib, io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            H.print_report(out)
        self.assertIn("H2", buf.getvalue())
        # 후보는 최대 3개 — 4번째 새 후보는 거부
        with self.assertRaises(SystemExit):
            H.evaluate(["SYN"], ["R1_daily", "R2_daily_trend8", "R4_daily_trend8_mv2"], reps=3, exploratory=True)
        # 실행 결과가 모자라면 판정하지 않음
        with self.assertRaises(SystemExit):
            H.evaluate(["SYN"], [cfg["name"]], reps=4, record=False)


def glob_files():
    import glob
    return glob.glob(os.path.join(H.HOLDOUT_DIR, "*_15m.csv.gz")) + glob.glob(os.path.join(H.HOLDOUT_DIR, "*_1h.csv.gz"))


if __name__ == "__main__":
    unittest.main()
