"""현실 점검(btc/research/recost.py)과 모의매매 보조 기록(btc/research/paper_bench.py) 테스트 — 네트워크 없음, 1스레드.

python -m unittest tests.test_btc_reality -v
  · 규칙(Rules): 기준 규칙 목표가 evaluate.baseline_targets + daily_hold 와 같고, run_one 이 Window.run 과 같음
  · 체결 시각(Timing): 하루·이틀 지연 밀기, 실행 가능 시각(00:30) — 규칙은 그날, 학습 모델의 월초 판단은 c(T) 뒤,
    같은 체결 봉이면 늦은 판단이 이김, 00:00 봉이 죽었으면 그 뒤 첫 봉, 00:30 봉이 비었으면 그 뒤 첫 15분봉
  · 원화(Krw): 00:00 체결가가 원화 시가와 같음, 원화 자료 앞은 NaN
  · 재현(Recost): 2015~2016 등록 계산(USD·00:00·0.15%)이 기록된 값과 같음 (연구 실행 파일이 있을 때만)
  · 보조 기록(Bench): 가짜 판단 기록으로 — B0 등록 체결 장부가 고정 코드의 bh_ledger.csv 와 바이트 단위로 같고,
    규칙 장부가 Window.run 과 같고, 실행 가능 장부가 독립 구현과 같고, benchmarks/ 밖은 한 바이트도 안 바뀌며,
    고정 report 결과가 그대로이고, 두 번 돌려도 같고, 미래 가격을 망가뜨려도 그 전 장부가 같음
"""
import json
import os
import shutil
import tempfile
import unittest

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from btc.data import load_15m
from btc.env import BAR_SEC
from btc.evaluate import Window, baseline_targets
from btc.walkforward import load_phases
from btc.research import paper as P
from btc.research import paper_bench as PB
from btc.research import recost as R
from btc.research.evaluate import daily_hold
from btc.research.variants import VARIANTS
from btc.research.walk import decision_mask, run_path

DAY = 86400
_C = {}


def datas():
    if "d" not in _C:
        _C["d"] = load_phases()
    return _C["d"]


def df15():
    if "f" not in _C:
        _C["f"] = load_15m()
    return _C["f"]


def ts(s):
    return int(pd.Timestamp(s, tz="UTC").timestamp())


class FakeD:
    """exec_targets 가 쓰는 것만 (ts, T)"""

    def __init__(self, ts_arr):
        self.ts = np.asarray(ts_arr, dtype=np.int64)
        self.T = len(self.ts)


def four_hour_grid(start, days, drop=()):
    t = np.arange(ts(start), ts(start) + days * DAY, 4 * 3600, dtype=np.int64)
    return np.array([x for x in t if x not in set(drop)], dtype=np.int64)


class Rules(unittest.TestCase):
    LO, HI = "2019-01-01", "2021-01-01"

    @classmethod
    def setUpClass(cls):
        cls.W = Window(datas(), cls.LO, cls.HI, "test_btc_reality")
        cls.d = cls.W.datas[0]
        cls.a, cls.b = cls.W.rng[0]
        cls.mask = decision_mask(cls.d, 6)

    def test_rule_targets_match_baseline(self):
        bt = baseline_targets(self.W)
        rt = R.rule_targets(self.d, self.a, self.b, self.mask)
        for k in ("B2", "B5", "B6"):
            ref = daily_hold(bt[k], self.mask, self.a, self.b)
            self.assertTrue(np.array_equal(rt[k][0], ref, equal_nan=True), k)
            self.assertFalse(rt[k][1])
        eq_b0 = self.W.run(bt["B0"], 0.0015)["eq"]                      # B0 는 판단봉에만 둬도 같은 결과
        res = R.run_one(self.d, self.a, self.b, self.W.lo, self.W.hi, rt["B0"][0], False, 0.0015, self.d.o, self.d.c)
        self.assertTrue(np.array_equal(res["eq"], eq_b0))
        self.assertTrue(rt["B80"][1])
        sel = np.nonzero(np.isfinite(rt["B80"][0]))[0]
        self.assertTrue(np.all(rt["B80"][0][sel] == 0.8))
        self.assertTrue(np.array_equal(sel, np.arange(self.a, self.b)[self.mask[self.a:self.b]]))

    def test_run_one_matches_window_run(self):
        rt = R.rule_targets(self.d, self.a, self.b, self.mask)
        for k in ("B2", "B5", "B80", "SMA120"):
            tg, frac = rt[k]
            ref = self.W.run(tg, 0.0015, weights=frac)
            res = R.run_one(self.d, self.a, self.b, self.W.lo, self.W.hi, tg, frac, 0.0015, self.d.o, self.d.c)
            self.assertTrue(np.array_equal(res["eq"], ref["eq"]), k)
            self.assertTrue(np.array_equal(res["pos"], ref["pos"]), k)
            self.assertGreater(len(np.unique(res["pos"].round(6))), 1, k)

    def test_exec_prices_are_0030_opens(self):
        o_ex, info = R.exec_open(self.d, df15())
        f = df15()
        op = pd.Series(f["open"].to_numpy(), index=f["ts"].to_numpy())
        j = np.nonzero((self.d.ts % DAY == 0) & (self.d.ts >= self.W.lo) & (self.d.ts < self.W.hi))[0]
        n_eq = 0
        for jj in j:
            p = op.get(int(self.d.ts[jj]) + 1800, np.nan)
            if np.isfinite(p):
                self.assertEqual(o_ex[jj], p)
                n_eq += 1
        self.assertGreater(n_eq, 0.99 * len(j))
        other = np.nonzero(self.d.ts % DAY != 0)[0]
        self.assertTrue(np.array_equal(o_ex[other], self.d.o[other]))     # 00:00 봉 밖은 그대로


class Timing(unittest.TestCase):
    def test_lag_shift(self):
        tg = np.full(12, np.nan)
        sel = np.array([1, 3, 5, 7, 9])
        tg[sel] = [1, 0, 1, 1, 0]
        l1 = R.lag_targets(tg, 1)
        self.assertTrue(np.array_equal(l1[sel], [0, 1, 0, 1, 1]))
        self.assertTrue(np.all(np.isnan(np.delete(l1, sel))))
        l2 = R.lag_targets(tg, 2)
        self.assertTrue(np.array_equal(l2[sel], [0, 0, 1, 0, 1]))
        self.assertIs(R.lag_targets(tg, 0), tg)

    def _dec(self, grid, days_vals):
        """판단봉(마감 00:00) = 20:00 시작 봉. {날짜: 목표}"""
        tg = np.full(len(grid), np.nan)
        for day, v in days_vals.items():
            i = int(np.nonzero(grid + BAR_SEC == ts(day))[0][0])
            tg[i] = v
        return tg

    def _at(self, grid, out, day):
        """day 00:00 에 시작하는 체결 봉의 바로 앞 봉 목표"""
        j = int(np.searchsorted(grid, ts(day)))
        return out[j - 1]

    def test_exec_rules_same_day(self):
        g = four_hour_grid("2024-02-27", 6)
        d = FakeD(g)
        vals = {"2024-02-28": 1.0, "2024-02-29": 0.0, "2024-03-01": 1.0, "2024-03-02": 0.0}
        tg = self._dec(g, vals)
        out = R.exec_targets(d, tg, None)
        self.assertTrue(np.array_equal(out, tg, equal_nan=True))           # 규칙: 그날 00:30 (같은 봉 목표)

    def test_exec_month_start_waits_for_cut(self):
        g = four_hour_grid("2024-02-27", 6)
        d = FakeD(g)
        vals = {"2024-02-28": 1.0, "2024-02-29": 0.0, "2024-03-01": 1.0, "2024-03-02": 0.0}
        tg = self._dec(g, vals)
        cuts = {ts("2024-02-01"): ts("2024-02-02"), ts("2024-03-01"): ts("2024-03-02")}
        out = R.exec_targets(d, tg, cuts)
        self.assertEqual(self._at(g, out, "2024-02-28"), 1.0)
        self.assertEqual(self._at(g, out, "2024-02-29"), 0.0)
        self.assertTrue(np.isnan(self._at(g, out, "2024-03-01")))          # 3월 1일 판단은 2일 00:20에야 → 2일 판단이 이김
        self.assertEqual(self._at(g, out, "2024-03-02"), 0.0)
        self.assertEqual(int(np.isfinite(out).sum()), 3)
        # c(T) = T + 2일: 1·2일 판단 모두 3일 00:30 체결 봉으로 → 3일 판단이 없으면 2일 판단이 남음
        cuts2 = {ts("2024-02-01"): ts("2024-02-02"), ts("2024-03-01"): ts("2024-03-03")}
        tg2 = self._dec(g, {"2024-03-01": 1.0, "2024-03-02": 0.0})
        out2 = R.exec_targets(d, tg2, cuts2)
        self.assertEqual(self._at(g, out2, "2024-03-03"), 0.0)
        self.assertEqual(int(np.isfinite(out2).sum()), 1)
        with self.assertRaises(ValueError):
            R.exec_targets(d, tg, {ts("2024-02-01"): ts("2024-02-02")})     # 3월 c(T) 없음

    def test_exec_dead_midnight_bar(self):
        g = four_hour_grid("2024-02-20", 5, drop=(ts("2024-02-22"),))       # 22일 00:00 시작 봉이 죽음
        d = FakeD(g)
        tg = np.full(len(g), np.nan)
        i = int(np.nonzero(g + BAR_SEC == ts("2024-02-22"))[0][0])          # 22일 00:00 마감 판단봉
        tg[i] = 1.0
        out = R.exec_targets(d, tg, None)
        j = int(np.searchsorted(g, ts("2024-02-22")))                       # 그 뒤 첫 봉 (04:00 시작)
        self.assertEqual(g[j], ts("2024-02-22") + 4 * 3600)
        self.assertEqual(out[j - 1], 1.0)
        self.assertEqual(int(np.isfinite(out).sum()), 1)

    def test_open_at_fallback(self):
        t0 = ts("2024-05-01")
        tt = t0 + 900 * np.arange(0, 20)
        op = np.arange(20, dtype=float) + 100.0
        op[2] = np.nan                                                      # 00:30 봉이 빔 → 00:45 봉
        f = pd.DataFrame({"ts": tt, "open": op})
        px, n_sub = R.open_at(f, [t0])
        self.assertEqual(px[0], 103.0)
        self.assertEqual(n_sub, 1)
        op2 = op.copy()
        op2[2:] = np.nan
        px2, _ = R.open_at(pd.DataFrame({"ts": tt, "open": op2}), [t0])
        self.assertTrue(np.isnan(px2[0]))


class Krw(unittest.TestCase):
    def test_booked_fill_equals_krw_open(self):
        d = datas()[0]
        fac, info = R.krw_factor(d, df15(), R.load_krw())
        krw = R.load_krw()
        j = np.nonzero((d.ts % DAY == 0) & (d.ts >= ts("2018-01-01")) & (d.ts < ts("2026-09-01")))[0]
        days = pd.to_datetime(d.ts[j], unit="s", utc=True)
        ref = krw["krw_open"].reindex(days).to_numpy()
        np.testing.assert_allclose(d.o[j] * fac[j], ref, rtol=1e-12)
        self.assertTrue(np.all(np.isnan(fac[d.ts < ts("2014-01-01")])))
        self.assertTrue(np.all(np.isfinite(fac[(d.ts >= ts("2015-01-01")) & (d.ts < ts("2026-09-25"))])))


@unittest.skipUnless(all(os.path.exists(run_path(n + "__early", r)) for n, _, k in R.MODELS.values() for r in range(k)),
                     "연구 실행 파일(research_runs/*__early) 없음 — 로컬에서만")
class Recost(unittest.TestCase):
    def test_early_registered_reproduced(self):
        lo, hi, tag, what = R.WINDOWS["early"]
        W = Window(datas(), lo, hi, what)
        w = R.evaluate_window(W, datas(), df15(), R.load_krw(), tag, do_boot=False)
        rc = R.ref_check("early", w["cells"])
        self.assertTrue(all(v["matches"] for v in rc.values()), rc)
        self.assertEqual(w["month_cut_not_T_plus_1"], {})
        self.assertEqual(len(w["cells"]), len(R.CURRENCIES) * len(R.TIMINGS) * len(R.COSTS))
        c0 = w["cells"]["USD|booked|0.0015"]
        c1 = w["cells"]["USD|booked|0.0005"]
        for n in R.ORDER:                                                   # 비용이 낮으면 샤프가 같거나 높음
            self.assertGreaterEqual(c1[n]["sharpe"] + 1e-12, c0[n]["sharpe"], n)


# ══════════ 보조 기록 ══════════
CFG = dict(VARIANTS["R6_daily_trend8_uniform"])


def write_variant(tmp, name, start, reps, checkpoints, targets_by_rep, d0, cuts_iso=None, committee=False):
    """가짜 등록 + 판단 기록 + train_log (paper_bench 가 읽는 것만)"""
    reg = P.load_registry(tmp)
    cfg = dict(CFG, committee=reps) if committee else dict(CFG)
    reg["variants"][name] = dict(name=name, cfg=cfg, reps=reps, start=start, checkpoints=list(checkpoints),
                                 registered_at="2024-01-01T00:00:00Z", cfg_hash="test", code_hash="test",
                                 git=dict(commit="test"),
                                 verdict_rule=P.COMMITTEE_VERDICT_RULE if committee else P.VERDICT_RULE)
    P._write_json(P.reg_path(tmp), reg)
    vdir = os.path.join(tmp, name)
    close = d0.ts + BAR_SEC
    for r in range(reps):
        tg = targets_by_rep[r]
        rows = [dict(close_utc=P._iso(close[i]), ts_close=str(int(close[i])), model_month="", target=P._num(tg[i]),
                     forced_hold="0", reason="", decided_at="", lag_h="") for i in np.nonzero(np.isfinite(tg))[0]]
        P._write_decs(vdir, r, cfg, rows)
        with open(os.path.join(vdir, f"train_log_rep{r:02d}.jsonl"), "w", encoding="utf-8") as f:
            for m, c in (cuts_iso or {}).items():
                f.write(json.dumps(dict(month=m, data_cut=c)) + "\n")


class Bench(unittest.TestCase):
    LO, NOW = "2024-01-01", "2024-04-10T00:20Z"
    HI = "2024-04-10"
    CPS = ("2024-02-15", "2024-03-15")

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="paper_bench_")
        now = P._ts(cls.NOW)
        cls.df = P.truncate(load_15m(), now)
        cls.d0 = P.phases_for(cls.df, 1)[0]
        d0 = cls.d0
        mask = decision_mask(d0, 6)
        lo, hi = P._ts(cls.LO), P._ts(cls.HI)
        a, b = P.window_range(d0, lo, hi)
        close = d0.ts + BAR_SEC
        sel = np.nonzero(mask & (close >= lo) & (close < hi))[0]
        sig = R.rule_signals(d0)
        t0 = np.full(d0.T, np.nan)
        t0[sel] = sig["B5"][sel]
        t1 = t0.copy()
        t1[sel[::7]] = 1.0 - t1[sel[::7]]                                  # 다른 반복은 일부 뒤집음
        t2 = t0.copy()
        t2[sel[::5]] = 0.0
        cls.tgs = [t0, t1, t2]
        # 3월 c(T) = 3월 3일 (이틀 늦음), 4월 = 4월 2일 — 로그에서 읽는지 확인
        cls.cuts_iso = {"2024-03-01": "2024-03-03T00:00:00Z", "2024-04-01": "2024-04-02T00:00:00Z",
                        "2024-02-01": "hist"}
        write_variant(cls.tmp, "V_A", cls.LO, 3, cls.CPS, cls.tgs, d0, cls.cuts_iso, committee=True)
        P._write_json(os.path.join(cls.tmp, "status.json"), dict(now=cls.NOW, now_ts=now))
        ent = P.load_registry(cls.tmp)["variants"]["V_A"]
        P.write_ledgers(ent, d0, now, paper_dir=cls.tmp)                  # 고정 코드와 같은 장부 (bh_ledger.csv 포함)
        cls.before = cls.snap(exclude_bench=True)
        cls.rep_before = P.report(cls.NOW, paper_dir=cls.tmp, df15=cls.df, write=False)
        cls.out = PB.build(cls.tmp, now=None, df15=cls.df)
        cls.bench1 = cls.snap(only_bench=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    @classmethod
    def snap(cls, exclude_bench=False, only_bench=False):
        out = {}
        for dp, _, fs in os.walk(cls.tmp):
            for f in fs:
                rel = os.path.relpath(os.path.join(dp, f), cls.tmp)
                inb = rel.startswith(PB.SUB + os.sep)
                if (exclude_bench and inb) or (only_bench and not inb):
                    continue
                with open(os.path.join(dp, f), "rb") as fh:
                    out[rel] = fh.read()
        return out

    def _eq(self, rel):
        rows = P._read_csv(os.path.join(self.tmp, rel))
        return np.array([float(x["equity"]) for x in rows]), [x["date"] for x in rows]

    def test_b0_booked_equals_frozen_bh_ledger(self):
        a = self.bench1[os.path.join(PB.SUB, "rules", "ledger_B0_booked.csv")]
        b = self.snap()[os.path.join("V_A", "bh_ledger.csv")]
        self.assertEqual(a, b)
        self.assertGreater(len(a.splitlines()), 90)

    def test_rules_booked_match_window_run(self):
        W = Window(datas(), self.LO, self.HI, "test_btc_reality")
        bt = baseline_targets(W)
        d = W.datas[0]
        a, b = W.rng[0]
        mask = decision_mask(d, 6)
        for k in ("B2", "B5", "B6"):
            ref = W.run(daily_hold(bt[k], mask, a, b), 0.0015)["eq"]
            eq, dates = self._eq(os.path.join(PB.SUB, "rules", f"ledger_{k}_booked.csv"))
            np.testing.assert_allclose(eq, ref, rtol=0, atol=1e-12)
            self.assertEqual(dates[0], self.LO)
            self.assertEqual(dates[-1], self.HI)

    def test_exec_ledgers_match_independent(self):
        """독립 구현: 판단 '유효일' = max(판단일, c(T)) → 그날 00:30 15분봉 시가에 체결, 같은 날이면 늦은 판단"""
        d0, f = self.d0, self.df
        close = d0.ts + BAR_SEC
        op = pd.Series(f["open"].to_numpy(), index=f["ts"].to_numpy())
        cut = {P._ts(m): (P._ts(c) if c != "hist" else P._ts(m) + DAY) for m, c in self.cuts_iso.items()}
        cut.setdefault(P._ts("2024-01-01"), P._ts("2024-01-02"))
        o = d0.o.copy()
        for jj in np.nonzero(d0.ts % DAY == 0)[0]:
            p = op.get(int(d0.ts[jj]) + 1800, np.nan)
            if np.isfinite(p):
                o[jj] = p
        lo, hi = P._ts(self.LO), P._ts(self.HI)
        a, b = P.window_range(d0, lo, hi)
        for r, tg in enumerate(self.tgs):
            exp = np.full(d0.T, np.nan)
            for i in np.nonzero(np.isfinite(tg))[0]:
                X = int(close[i])
                t = pd.Timestamp(X, unit="s", tz="UTC")
                T = int(pd.Timestamp(t.year, t.month, 1, tz="UTC").timestamp())
                eff = max(X, cut[T])
                j = int(np.searchsorted(d0.ts, eff))
                exp[j - 1] = tg[i]
            res = R.run_one(d0, a, b, lo, hi, exp, False, 0.0015, o, d0.c)
            eq, _ = self._eq(os.path.join(PB.SUB, "exec", "V_A", f"ledger_rep{r:02d}.csv"))
            np.testing.assert_allclose(eq, res["eq"], rtol=0, atol=1e-12)
        # 3월 1·2일 판단은 3월 3일 00:30 전에는 체결되지 않음 → 3월 1·2일 체결 없음
        fills = P._read_csv(os.path.join(self.tmp, PB.SUB, "exec", "V_A", "fills_rep00.csv"))
        fdays = {x["fill_utc"][:10] for x in fills}
        self.assertFalse({"2024-03-01", "2024-03-02"} & fdays)
        self.assertTrue(all(x["fill_utc"].endswith("T00:30:00Z") for x in fills))
        self.assertGreater(len(fills), 5)

    def test_committee_exec(self):
        d0 = self.d0
        lo, hi = P._ts(self.LO), P._ts(self.HI)
        cut = {P._ts(m): (P._ts(c) if c != "hist" else P._ts(m) + DAY) for m, c in self.cuts_iso.items()}
        cut.setdefault(P._ts("2024-01-01"), P._ts("2024-01-02"))
        ctg = P.committee_targets(d0, self.tgs)
        o_ex, _ = R.exec_open(d0, self.df)
        a, b = P.window_range(d0, lo, hi)
        ref = R.run_one(d0, a, b, lo, hi, R.exec_targets(d0, ctg, cut), False, 0.0015, o_ex, d0.c)
        eq, _ = self._eq(os.path.join(PB.SUB, "exec", "V_A", "ledger_committee.csv"))
        np.testing.assert_allclose(eq, ref["eq"], rtol=0, atol=1e-12)

    def test_nothing_outside_benchmarks_changes(self):
        self.assertEqual(self.snap(exclude_bench=True), self.before)
        rep_after = P.report(self.NOW, paper_dir=self.tmp, df15=self.df, write=False)
        self.assertEqual(json.dumps(rep_after, sort_keys=True, default=str),
                         json.dumps(self.rep_before, sort_keys=True, default=str))

    def test_idempotent(self):
        PB.build(self.tmp, now=None, df15=self.df)
        self.assertEqual(self.snap(only_bench=True), self.bench1)

    def test_report_checkpoints(self):
        rep = json.loads(self.bench1[os.path.join(PB.SUB, "report.json")].decode())
        v = rep["variants"]["V_A"]
        self.assertEqual(v["checkpoints"]["2024-02-15"]["status"], "computed")
        self.assertIn("committee", v["checkpoints"]["2024-02-15"])
        self.assertEqual(set(v["checkpoints"]["2024-02-15"]["rules"]), set(R.RULES))
        self.assertEqual(rep["ledger_through"], self.HI)

    def test_no_lookahead(self):
        """2024-03-10 00:00 이후 가격을 모두 망가뜨려도 그날까지의 장부(등록·실행 가능)가 같음"""
        cut_ts = P._ts("2024-03-10")
        bad = self.df.copy()
        m = bad["ts"].to_numpy() >= cut_ts
        for c in ("open", "high", "low", "close"):
            bad.loc[m, c] = bad.loc[m, c] * 3.0
        tmp2 = tempfile.mkdtemp(prefix="paper_bench_la_")
        try:
            write_variant(tmp2, "V_A", self.LO, 3, self.CPS, self.tgs, self.d0, self.cuts_iso, committee=True)
            P._write_json(os.path.join(tmp2, "status.json"), dict(now=self.NOW, now_ts=P._ts(self.NOW)))
            PB.build(tmp2, now=None, df15=bad)
            for rel in (os.path.join("rules", "ledger_B5_booked.csv"), os.path.join("rules", "ledger_B5_exec.csv"),
                        os.path.join("rules", "ledger_B2_exec.csv"), os.path.join("exec", "V_A", "ledger_rep00.csv")):
                ok = {x["date"]: x["equity"] for x in P._read_csv(os.path.join(self.tmp, PB.SUB, rel))}
                got = {x["date"]: x["equity"] for x in P._read_csv(os.path.join(tmp2, PB.SUB, rel))}
                early = [k for k in ok if k <= "2024-03-10"]
                self.assertGreater(len(early), 60)
                self.assertEqual([ok[k] for k in early], [got[k] for k in early], rel)
                self.assertNotEqual(ok[self.HI], got[self.HI], rel)
        finally:
            shutil.rmtree(tmp2, ignore_errors=True)

    def test_no_registry_no_output(self):
        tmp2 = tempfile.mkdtemp(prefix="paper_bench_empty_")
        try:
            self.assertIsNone(PB.build(tmp2, df15=self.df))
            self.assertEqual(os.listdir(tmp2), [])
        finally:
            shutil.rmtree(tmp2, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
