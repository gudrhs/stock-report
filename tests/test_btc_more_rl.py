"""새 강화학습 4가지 묶음 판정기(btc/research/more_rl.py) 단위 테스트 — 합성 데이터만, 실제 실행 결과 불필요.

python -m unittest tests.test_btc_more_rl -v
  · C1 합의: ≥6 보유, ≤4 현금, 5:5 직전 합의 유지(시작 현금), 0/1 아닌 표는 거부
  · 강제 유지 봉에서는 합의도 바뀌지 않음 → 합의 = 실제로 시뮬레이션된 포지션 (들지 않은 보유를 5:5로 유지하지 않음)
  · 문턱은 반올림 전 R6 값, 등록 문구의 반올림 값이면 판정이 달라지는 경우 표시
  · 갱신한 시험 수 N: 기록에 없는 C1 등을 더하고, 그 N에서의 디플레이티드 검정도 계산
  · 반복별 표 = 그 반복의 k_policy 포지션 (U가 NaN이면 직전 포지션 유지 — 목표는 NaN)
  · 작은 쪽 중앙값(5개면 3번째, 10개면 5번째)·위쪽 중앙값(10개면 6번째)
  · 판정: 두 구간 모두 R6 기준·매수·보유보다 '높아야'(같으면 탈락), 합성 대조 없음 → 미완, 탈락 → 후보 아님
  · 실행 결과가 빠지면 무엇이 없는지 적은 MissingRuns (데이터를 읽기 전에)
  · 디플레이티드 검정 문턱 = robust.py 와 같은 식
  · 합성 가격·가짜 실행으로 한 구간(2015~2016, 잠금 구간 아님) 전체 계산: 같은 U면 P1 = R6, 10개가 같으면 C1 = R6
"""
import math
import os
import sys
import types
import unittest

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btc import stats as S
from btc.env import BAR_SEC
from btc.evaluate import Window
from btc.research import more_rl as M
from btc.research.rl import k_policy


class TestCommittee(unittest.TestCase):
    def votes(self, counts, m=10):
        return np.array([[1.0] * k + [0.0] * (m - k) for k in counts])

    def test_thresholds_and_hysteresis(self):
        counts = [5, 6, 5, 5, 4, 5, 7, 10, 5, 0, 5, 3, 6]
        want = [0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1]
        np.testing.assert_array_equal(M.committee(self.votes(counts)), np.array(want, float))

    def test_start_flat_and_tie_keeps(self):
        np.testing.assert_array_equal(M.committee(self.votes([5, 5, 5])), [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(M.committee(self.votes([6, 5, 5])), [1.0, 1.0, 1.0])

    def test_vote_order_irrelevant(self):
        rng = np.random.default_rng(0)
        v = (rng.random((200, 10)) < 0.5).astype(float)
        a = M.committee(v)
        b = M.committee(v[:, rng.permutation(10)])
        np.testing.assert_array_equal(a, b)
        # 반복문으로 따로 짠 규칙
        p, ref = 0.0, []
        for row in v:
            k = int(row.sum())
            p = 1.0 if k >= 6 else (0.0 if k <= 4 else p)
            ref.append(p)
        np.testing.assert_array_equal(a, ref)

    def test_forced_hold_keeps_actual_position(self):
        # 검토에서 찾은 경우: 첫 봉이 강제 유지면 6표여도 진입하지 않았으므로, 다음 5:5는 '현금 유지'
        v = self.votes([6, 5, 5])
        fh = np.array([True, False, False])
        np.testing.assert_array_equal(M.committee(v, fh), [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(M.committee(v, fh * False), [1.0, 1.0, 1.0])
        # 강제 유지 중에는 보유도 유지 (4표여도 청산하지 않음)
        np.testing.assert_array_equal(M.committee(self.votes([6, 4, 5, 4]), np.array([False, True, False, False])),
                                      [1.0, 1.0, 1.0, 0.0])
        with self.assertRaises(ValueError):
            M.committee(v, np.array([True, False]))

    def test_forced_hold_matches_simulate(self):
        # 합의 포지션을 목표로 env.simulate 에 넣으면 판단봉의 실제 포지션이 합의와 정확히 같아야 함
        from btc.env import simulate
        rng = np.random.default_rng(5)
        T = 400
        v = (rng.random((T, 10)) < rng.random((T, 1))).astype(float)
        v[rng.random(T) < 0.3] = np.repeat([[1.0] * 5 + [0.0] * 5], 1, 0)   # 5:5 를 자주
        fh = rng.random(T) < 0.1
        c = 100 * np.exp(np.cumsum(0.01 * rng.standard_normal(T)))
        o = np.concatenate([[100.0], c[:-1]])
        pos = M.committee(v, fh)
        sim = simulate(pos, o, c, 0.0015, 0, T - 1, forced_hold=fh)
        held = np.asarray(sim["pos"], dtype=float)
        # simulate 의 pos[t] = t 봉 목표를 t+1 시가에 체결한 뒤의 실제 보유 (강제 유지 봉은 목표 무시)
        np.testing.assert_array_equal(held[:T - 1], pos[:T - 1])
        # 고치기 전 규칙(강제 유지 무시)은 이 자료에서 실제 보유와 어긋남 — 시험이 그 경우를 담고 있는지 확인
        self.assertFalse(np.array_equal(M.committee(v)[:T - 1], held[:T - 1]))

    def test_rejects_nonbinary(self):
        v = self.votes([6, 5])
        v[0, 0] = np.nan
        with self.assertRaises(ValueError):
            M.committee(v)
        v = self.votes([6, 5])
        v[1, 3] = 0.5
        with self.assertRaises(ValueError):
            M.committee(v)


class TestRepTargets(unittest.TestCase):
    def fake(self, T=60):
        ts = 1420070400 - BAR_SEC + BAR_SEC * np.arange(T, dtype=np.int64)   # 첫 봉이 2015-01-01 00:00에 마감
        return types.SimpleNamespace(ts=ts, T=T, c=np.ones(T), forced_hold=np.zeros(T, bool))

    def test_dqn_held_vs_target(self):
        d = self.fake()
        U = np.zeros((d.T, 2), np.float32)
        U[:, 1] = -1.0
        U[6, 1] = 1.0            # 두 번째 판단봉에서 보유로
        U[12] = np.nan           # 세 번째 판단봉은 모델 없음 → 유지
        U[18, 1] = -1.0          # 네 번째는 현금으로
        run = dict(U={M.C_DEC: U}, meta=dict(cfg=dict(stride=6, acts=[0.0, 1.0])))
        tg, frac, held, sel, ns = M.rep_targets(d, run["meta"]["cfg"], run, 0, d.T - 1)
        self.assertFalse(frac)
        np.testing.assert_array_equal(sel[:4], [0, 6, 12, 18])
        np.testing.assert_array_equal(held[:4], [0.0, 1.0, 1.0, 0.0])
        self.assertTrue(np.isnan(tg[12]))
        self.assertEqual(tg[6], 1.0)
        self.assertTrue(np.isnan(tg[1]))          # 판단봉 밖은 NaN
        self.assertAlmostEqual(ns, 1.0 / len(sel))
        np.testing.assert_array_equal(held, np.array([0.0, 1.0])[k_policy(U[sel], (0.0, 1.0), M.C_DEC)])

    def test_weights_rounding(self):
        d = self.fake()
        w = np.full((d.T, 1), np.nan, np.float32)
        w[0, 0], w[6, 0], w[12, 0] = 0.12, 0.13, 0.9
        run = dict(U={0.0: w}, meta=dict(cfg=dict(stride=6, output="weights", algo="a2c")))
        tg, frac, held, sel, ns = M.rep_targets(d, run["meta"]["cfg"], run, 0, d.T - 1)
        self.assertTrue(frac)
        self.assertIsNone(held)
        self.assertEqual(tg[0], 0.0)
        self.assertEqual(tg[6], 0.25)
        self.assertEqual(tg[12], 1.0)
        self.assertTrue(np.isnan(tg[18]))


class TestSelection(unittest.TestCase):
    def test_lower_median(self):
        s = [1.0, 0.5, 0.9, 1.2, 0.7]
        self.assertEqual(M.headline_index(s, 0.3), 2)                  # 0.5 0.7 [0.9] 1.0 1.2
        self.assertEqual(M.headline_index(s, 5.0), 2)                  # 기준을 빼도 순서는 같음
        s10 = [0.1 * k for k in (7, 3, 9, 1, 5, 2, 8, 4, 6, 10)]
        self.assertAlmostEqual(s10[M.headline_index(s10, 0.0)], 0.5)  # 10개면 5번째
        self.assertAlmostEqual(s10[M.upper_median_index(s10)], 0.6)   # 위쪽 중앙값 = 6번째
        self.assertAlmostEqual(float(np.median(s10)), 0.55)


class TestCriteria(unittest.TestCase):
    def pw(self, sm, rm, bm, se, re, be, p=0.3):
        return {"main": dict(sharpe=sm, ref_sharpe=rm, bh_sharpe=bm, p_vs_r6=p),
                "early": dict(sharpe=se, ref_sharpe=re, bh_sharpe=be, p_vs_r6=p)}

    def test_candidate(self):
        c = M.criteria(self.pw(1.2, 1.149, 1.0, 2.0, 1.90, 1.22), True)
        self.assertTrue(c["performance_ok"])
        self.assertIs(c["candidate"], True)
        self.assertFalse(c["p_vs_r6_below_005_both"])

    def test_one_window_fails(self):
        c = M.criteria(self.pw(1.2, 1.149, 1.0, 1.85, 1.90, 1.22), True)
        self.assertIs(c["candidate"], False)
        self.assertEqual(c["beats_r6"], {"main": True, "early": False})

    def test_equal_is_not_better(self):
        c = M.criteria(self.pw(1.149, 1.149, 1.0, 2.0, 1.90, 1.22), True)
        self.assertIs(c["candidate"], False)

    def test_must_beat_bh(self):
        # R6 기준보다 높아도 매수·보유보다 낮으면 탈락
        c = M.criteria(self.pw(1.2, 1.149, 1.3, 2.0, 1.90, 1.22), True)
        self.assertIs(c["candidate"], False)
        self.assertEqual(c["beats_bh"], {"main": False, "early": True})

    def test_no_model_flag(self):
        pw = self.pw(1.2, 1.149, 1.0, 2.0, 1.90, 1.22)
        self.assertIs(M.criteria(pw, True, no_model_max=0.05)["candidate"], True)       # 5% 는 허용 (early.py 와 같음)
        c = M.criteria(pw, True, no_model_max=0.051)
        self.assertIsNone(c["candidate"])
        self.assertTrue(c["no_model_flag"])
        self.assertIn("판정 불가", c["verdict"])

    def test_controls(self):
        pw = self.pw(1.2, 1.149, 1.0, 2.0, 1.90, 1.22, p=0.01)
        self.assertIsNone(M.criteria(pw, None)["candidate"])
        self.assertIs(M.criteria(pw, False)["candidate"], False)
        self.assertTrue(M.criteria(pw, True)["p_vs_r6_below_005_both"])

    def test_threshold_is_unrounded_and_flagged(self):
        # 2015~2016 헤드라인 1.902: 등록 문구(1.90)로는 통과, 반올림 전 R6(1.9044)로는 탈락 → 탈락 + 표시
        pw = self.pw(1.2, 1.148687, 1.0, 1.902, 1.904419, 1.22)
        c = M.criteria(pw, True, registered={"main": 1.149, "early": 1.90}, ref_basis="x")
        self.assertIs(c["candidate"], False)
        self.assertTrue(c["verdict_differs_registered"])
        self.assertEqual(c["beats_r6_registered"], {"main": True, "early": True})
        self.assertIn("반올림", c["verdict"])
        self.assertEqual(c["r6_threshold"], {"main": 1.148687, "early": 1.904419})
        self.assertEqual(c["r6_threshold_basis"], "x")
        # 둘 다 같은 결론이면 표시 없음
        c = M.criteria(self.pw(1.2, 1.148687, 1.0, 2.0, 1.904419, 1.22), True, registered={"main": 1.149, "early": 1.90})
        self.assertFalse(c["verdict_differs_registered"])
        self.assertIs(c["candidate"], True)
        # C1은 등록 숫자가 없음
        self.assertIsNone(M.registered_refs("C1_r6_committee10"))
        self.assertEqual(M.registered_refs("A1_actor_critic"), {"main": 1.149, "early": 1.90})
        c = M.criteria(pw, True)
        self.assertIsNone(c["beats_r6_registered"])
        self.assertFalse(c["verdict_differs_registered"])

    def test_n_trials_updated(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "trials.jsonl")
            with open(p, "w", encoding="utf-8") as f:
                for n in ("R6_daily_trend8_uniform", "Q1_qrdqn_cvar", "A1_actor_critic", "P1_pessimistic_dqn"):
                    f.write(json.dumps(dict(name=n, cfg_hash=n[:3])) + "\n")
            self.assertEqual(M.n_trials_updated(base=60, trials=p), (61, ["C1_r6_committee10"]))
            self.assertEqual(M.n_trials_updated(base=60, trials=os.path.join(td, "없음.jsonl"))[0], 64)
        self.assertEqual(M.dsr_ns(61), (50, 200, 1000, 61))
        self.assertEqual(M.dsr_ns(200), (50, 200, 1000))
        self.assertEqual(M.dsr_ns(None), (50, 200, 1000))
        g = M.dsr_gain(dict(obs=0.5, se=0.25), M.dsr_ns(61))
        self.assertIn("N61", g)
        self.assertLess(g["N50"]["threshold"], g["N61"]["threshold"])

    def test_needs_both_windows(self):
        with self.assertRaises(ValueError):
            M.criteria({"main": dict(sharpe=1, ref_sharpe=0, bh_sharpe=0, p_vs_r6=0.1)}, True)

    def test_controls_source(self):
        ctl = {"R6_daily_trend8_uniform": dict(positive_pass=True, negative_pass=True),
               "A1_actor_critic": dict(positive_pass=True, negative_pass=False)}
        self.assertTrue(M.controls_status("C1_r6_committee10", ctl)["passed"])
        self.assertTrue(M.controls_status("P1_pessimistic_dqn", ctl)["passed"])
        self.assertFalse(M.controls_status("A1_actor_critic", ctl)["passed"])
        self.assertIsNone(M.controls_status("Q1_qrdqn_cvar", ctl)["passed"])

    def test_dsr_threshold_same_as_robust(self):
        bt = dict(obs=0.5, se=0.25)
        g = M.dsr_gain(bt)
        for n in (50, 200, 1000):
            thr = 0.109 * math.sqrt(365.0) * S.expected_max_sharpe(1.0 / 365.0, n)
            self.assertAlmostEqual(g[f"N{n}"]["threshold"], thr, places=12)
            self.assertAlmostEqual(g[f"N{n}"]["dsr"], S.norm_cdf((0.5 - thr) / 0.25), places=12)
        self.assertLess(g["N50"]["threshold"], g["N1000"]["threshold"])

    def test_ref_check(self):
        self.assertTrue(M.ref_check("main", 1.1494)["matches"])
        self.assertFalse(M.ref_check("main", 1.1506)["matches"])
        self.assertTrue(M.ref_check("early", 1.8987)["matches"])


class TestMissingRuns(unittest.TestCase):
    def test_required(self):
        req = M.required_runs()
        self.assertEqual(len(req), 2 * (3 * 5 + 10))
        self.assertIn(("R6_daily_trend8_uniform__early", 9), req)
        self.assertIn(("A1_actor_critic", 4), req)
        self.assertNotIn(("A1_actor_critic", 5), req)

    def test_error_lists_missing_before_loading_data(self):
        have = set(M.required_runs()) - {("Q1_qrdqn_cvar__early", 2), ("R6_daily_trend8_uniform__early", 7),
                                         ("R6_daily_trend8_uniform__early", 9)}

        def boom(*a, **k):
            raise AssertionError("실행이 빠졌는데 데이터를 읽음")

        with self.assertRaises(M.MissingRuns) as cm:
            M.evaluate(loader=boom, exists=lambda n, r: (n, r) in have, datas=None)
        msg = str(cm.exception)
        self.assertIn("3개", msg)
        self.assertIn("Q1_qrdqn_cvar__early: 반복 2", msg)
        self.assertIn("R6_daily_trend8_uniform__early: 반복 7, 9", msg)
        self.assertNotIn("A1_actor_critic", msg)
        M.check_runs(exists=lambda n, r: True)             # 다 있으면 조용히 통과


class TestWindowSynthetic(unittest.TestCase):
    """합성 가격 + 가짜 실행으로 2015~2016 구간 전체 계산 (잠금 구간 아님)"""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(3)
        t0 = int(pd.Timestamp("2014-12-01", tz="UTC").timestamp())
        T = 6 * 780
        ts = t0 + BAR_SEC * np.arange(T, dtype=np.int64)
        lr = 0.0004 + 0.012 * rng.standard_normal(T)
        c = 300.0 * np.exp(np.cumsum(lr))
        o = np.concatenate([[300.0], c[:-1]])
        fh = np.zeros(T, bool)
        fh[rng.integers(0, T, 20)] = True
        cls.d = types.SimpleNamespace(ts=ts, T=T, o=o, c=c, forced_hold=fh)
        cls.W = Window([cls.d], "2015-01-01", "2017-01-01", "test-more-rl")
        # 반복마다 다른 신호 (다음 하루 수익 + 잡음) — 미래를 보는 가짜지만 계산 경로 확인용
        fwd = np.concatenate([np.log(o[7:] / o[1:-6]), np.zeros(7)])
        cls.runs = {}
        dq = dict(stride=6, acts=[0.0, 1.0])
        for r in range(10):
            U = np.zeros((T, 2), np.float32)
            U[:, 1] = 100 * fwd + 2.0 * np.random.default_rng(100 + r).standard_normal(T)
            U[:30] = np.nan                                  # 처음엔 모델 없음
            cls.runs[("R6_daily_trend8_uniform__early", r)] = dict(U={M.C_DEC: U}, meta=dict(cfg=dq))
        for r in range(5):
            cls.runs[("P1_pessimistic_dqn__early", r)] = cls.runs[("R6_daily_trend8_uniform__early", r)]
            U = np.zeros((T, 2), np.float32)
            U[:, 1] = 100 * fwd + 4.0 * np.random.default_rng(200 + r).standard_normal(T)
            cls.runs[("Q1_qrdqn_cvar__early", r)] = dict(U={M.C_DEC: U}, meta=dict(cfg=dict(dq, algo="qrdqn")))
            w = np.clip(0.5 + 20 * fwd + 0.1 * np.random.default_rng(300 + r).standard_normal(T), 0, 1)[:, None]
            cls.runs[("A1_actor_critic__early", r)] = dict(U={0.0: w.astype(np.float32)},
                                                           meta=dict(cfg=dict(stride=6, algo="a2c", output="weights",
                                                                              acts=[0.0, 0.25, 0.5, 0.75, 1.0])))
        cls.out = M.evaluate_window(cls.W, "__early", lambda n, r: cls.runs[(n, r)], M.dsr_ns(61))

    def test_bh_and_reference(self):
        o = self.out
        self.assertGreater(o["n_days"], 700)
        r6 = o["r6"]
        self.assertEqual(len(r6["sharpe_reps"]), 10)
        s5 = r6["sharpe_reps"][:5]
        self.assertEqual(r6["lower_median5"], sorted(s5)[2])
        self.assertAlmostEqual(r6["median10"], float(np.median(r6["sharpe_reps"])))
        self.assertEqual(len(set(r6["sharpe_reps"])), 10)                # 반복마다 다른 전략 (전부 현금이 아님)
        self.assertTrue(all(0.0 < s for s in r6["sharpe_reps"]))
        self.assertTrue(0.0 < self.out["methods"]["C1_r6_committee10"]["tie_share"] < 1.0)

    def test_p1_same_model_same_result(self):
        v = self.out["methods"]["P1_pessimistic_dqn"]
        self.assertEqual(v["sharpe"], self.out["r6"]["lower_median5"])
        self.assertEqual(v["d_vs_r6"], 0.0)
        self.assertEqual(v["headline_rep"], self.out["r6"]["headline_rep"])

    def test_weights_fractional(self):
        v = self.out["methods"]["A1_actor_critic"]
        self.assertTrue(v["fractional"])
        self.assertTrue(0.0 < v["exposure"] < 1.0)
        self.assertGreater(v["turnover"], 0.0)

    def test_committee_matches_manual(self):
        d, W = self.d, self.W
        a, b = W.rng[0]
        held = []
        for r in range(10):
            tg, frac, h, sel, _ = M.rep_targets(d, dict(stride=6, acts=[0.0, 1.0]),
                                                self.runs[("R6_daily_trend8_uniform__early", r)], a, b)
            held.append(h)
        v = np.stack(held, 1)
        p, pos = 0.0, []
        for row, f in zip(v, d.forced_hold[sel]):
            k = int(row.sum())
            if not f:                                            # 강제 유지 봉은 합의도 유지
                p = 1.0 if k >= 6 else (0.0 if k <= 4 else p)
            pos.append(p)
        tg = np.full(d.T, np.nan)
        tg[sel] = pos
        want = S.sharpe(W.run(tg, M.COST)["r"])
        c1 = self.out["methods"]["C1_r6_committee10"]
        self.assertAlmostEqual(c1["sharpe"], want, places=12)
        self.assertAlmostEqual(c1["ref_sharpe"], self.out["r6"]["median10"])
        self.assertGreater(c1["unanimous_share"], 0.0)

    def test_identical_reps_committee_equals_rep(self):
        runs = dict(self.runs)
        for r in range(10):
            runs[("R6_daily_trend8_uniform__early", r)] = self.runs[("R6_daily_trend8_uniform__early", 0)]
        out = M.evaluate_window(self.W, "__early", lambda n, r: runs[(n, r)])
        self.assertAlmostEqual(out["methods"]["C1_r6_committee10"]["sharpe"], out["r6"]["sharpe_reps"][0], places=12)
        self.assertEqual(out["methods"]["C1_r6_committee10"]["tie_share"], 0.0)

    def test_report_prints(self):
        from btc.evaluate import _clean
        wins = {k: self.out for k in M.WINDOWS}
        methods = {}
        for name in M.ORDER:
            pw = {k: wins[k]["methods"][name] for k in M.WINDOWS}
            ctl = M.controls_status(name, {})
            methods[name] = dict(windows=pw, controls=ctl,
                                 criteria=M.criteria(pw, ctl["passed"], M.registered_refs(name), M.REF_BASIS[name]))
        for k in wins:
            wins[k]["r6"]["ref_check"] = M.ref_check(k, wins[k]["r6"]["lower_median5"])
        out = _clean(dict(cost=M.COST, n_boot=M.N_BOOT, boot_seed=M.BOOT_SEED, n_trials=61,
                          n_trials_added=["C1_r6_committee10"], dsr_ns=[50, 200, 1000, 61],
                          windows={k: {kk: vv for kk, vv in w.items() if kk != "methods"} for k, w in wins.items()},
                          methods=methods))
        txt = M.report(out)
        for name in M.ORDER:
            self.assertIn(name, txt)
        self.assertIn("판정", txt)
        self.assertIn("N=50/200/1000/61", txt)
        self.assertIn("반올림 전", txt)
        self.assertIn("C1_r6_committee10 포함", txt)


if __name__ == "__main__":
    unittest.main()
