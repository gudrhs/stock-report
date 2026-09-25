"""3단계 모의매매(btc/research/paper.py) 테스트 — 네트워크 없음, 연구용 15분봉 캐시(data/btc) 사용, 1스레드.

python -m unittest tests.test_btc_paper -v
  · 재현(Parity): 과거 시작일로 등록 → step --offline → 판단값(U·비중)이 walk.run_replication 과 비트 단위로 같고,
    목표 비중이 screen.targets 와 같고, 장부가 evaluate.Window.run 과 같음 (0/1·비율 보유 DQN, 플러그인 W1, 상태 있는 dp_band)
  · 따라잡기(CatchUp): 하루씩 여러 번 step == 한 번에 step (달 경계·체결 대기 포함, 달 중간 묶음 채우기 확인)
  · 멱등(Idempotent): 같은 now로 두 번 돌려도 파일이 바이트 단위로 같음
  · 등록 거부(Register): 같은 이름에 다른 설정·반복 수 거부, 코드 지문이 바뀌면 step이 처리하지 않음
  · 미래 정보 없음(NoLookAhead): 판단봉 다음 시가 이후 가격을 모두 망가뜨려도 그 판단·체결가가 같음
  · 데이터 덧붙이기(Extend): 가짜 거래소 응답으로 15분봉 덧붙이기(빠진 분·늦은 분·덧붙이기만), 조회 실패 시 계속
학습 스텝·멤버·phase 수는 줄인 설정을 쓰되, 기준(run_replication)과 모의매매가 같은 설정을 씁니다.
"""
import gzip
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from btc.data import load_15m, minutes_to_15m, COLS
from btc.env import BAR_SEC
from btc.evaluate import Window, baseline_targets
from btc.walkforward import load_phases
from btc.research import paper as P
from btc.research import screen, walk
from btc.research.variants import VARIANTS

TINY = dict(members=2, batch=32, cold_steps=6, cold_split=4, cold_lr1=0.03, cold_lr2=0.01, ft_steps=2, ft_lr=0.01,
            phases=2)


def tiny(base, **kw):
    cfg = dict(VARIANTS[base])
    cfg.update(TINY)
    cfg.update(kw)
    return cfg


CFGS = {
    "T_R6": tiny("R6_daily_trend8_uniform"),                              # 0/1 보유 → simulate
    "T_R3": tiny("R3_daily_trend8_log5"),                                 # 5단계 비중 → simulate_weights
    "T_W1": tiny("W1_softmin_direct", seq_batch=4, groups=2),             # 플러그인, 비중 출력
}
DP = dict(VARIANTS["W2_dp_band"], phases=2, dp_grid_n=51)                 # 플러그인, 판단에 상태 있음
REPS = {"T_R6": 2, "T_R3": 1, "T_W1": 1}
_DATAS = {}


def datas_full():
    if "d" not in _DATAS:
        _DATAS["d"] = load_phases()
    return _DATAS["d"]


def read_bytes(p):
    with open(p, "rb") as f:
        return f.read()


def snapshot(root):
    out = {}
    for dp, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(dp, f)
            with open(p, "rb") as fh:
                out[os.path.relpath(p, root)] = fh.read()
    return out


def reference(cfgs, reps, lo, hi):
    """기준: run_replication(끝 = hi) → screen.targets·run_tg (저장 파일 대신 메모리의 결과를 넘김)"""
    datas = datas_full()
    runs = {(n, r): walk.run_replication(dict(c, name=n), r, datas=datas, end=pd.Timestamp(hi, tz="UTC"))
            for n, c in cfgs.items() for r in range(reps[n])}
    W = Window(datas, lo, hi, "test_btc_paper")
    out = {}
    with mock.patch.object(screen, "load_run", lambda n, r: runs[(n, r)]), \
            mock.patch.object(screen, "VARIANTS", {n: dict(c, name=n) for n, c in cfgs.items()}):
        for (n, r), run in runs.items():
            tg, frac, _ = screen.targets(W, n, r)
            tg = tg[0.003] if isinstance(tg, dict) else tg
            out[(n, r)] = dict(run=run, tg=tg, frac=frac, res=screen.run_tg(W, tg, 0.0015, frac))
    return W, out


def paper_arrays(tmp, name, r, cfg):
    decs = P._read_csv(P._dec_path(os.path.join(tmp, name), r))
    ts = np.array([int(x["ts_close"]) for x in decs])
    if P._is_weights(cfg):
        U = np.array([[P._f(x["w_raw"])] for x in decs], np.float32)
    else:
        K = len(cfg.get("acts", (0.0, 1.0)))
        U = np.array([[P._f(x[f"u{k}"]) for k in range(K)] for x in decs], np.float32)
    tg = np.array([P._f(x["target"]) for x in decs])
    return ts, U, tg, decs


def ledger(tmp, name, fname):
    rows = P._read_csv(os.path.join(tmp, name, fname))
    return np.array([float(x["equity"]) for x in rows]), [x["date"] for x in rows]


class Parity(unittest.TestCase):
    """과거 시작일(2024-01-01)로 등록한 모의매매 == 같은 설정의 run_replication + screen + Window.run"""

    LO, HI = "2024-01-01", "2024-04-01"

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="paper_parity_")
        for n, c in CFGS.items():
            P.register(n, reps=REPS[n], start=cls.LO, cfg=c, paper_dir=cls.tmp, checkpoints=("2024-02-15", "2024-03-15"))
        cls.out = P.step("2024-04-01T00:00Z", offline=True, paper_dir=cls.tmp)
        cls.W, cls.ref = reference(CFGS, REPS, cls.LO, cls.HI)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_decisions_match_backtest(self):
        d = self.W.datas[0]
        a, b = self.W.rng[0]
        close = d.ts + BAR_SEC
        for (n, r), x in self.ref.items():
            cfg = CFGS[n]
            sel = np.arange(a, b)[walk.decision_mask(d, 6)[a:b]]
            ts, U, tg, _ = paper_arrays(self.tmp, n, r, cfg)
            m = ts < P._ts(self.HI)
            self.assertTrue(np.array_equal(ts[m], close[sel]), n)
            Uref = x["run"]["U"][0.0 if P._is_weights(cfg) else 0.003][sel]
            self.assertTrue(np.array_equal(U[m], Uref, equal_nan=True), f"{n} rep{r}: U가 비트 단위로 다름")
            self.assertTrue(np.array_equal(tg[m], x["tg"][sel], equal_nan=True), f"{n} rep{r}: 목표 비중이 다름")
            self.assertGreater(len(np.unique(tg[m][~np.isnan(tg[m])])), 0)

    def test_ledger_matches_window_run(self):
        pd0 = P.phases_for(P.truncate(P.load_15m_all(self.tmp), P._ts(self.HI)), 1)[0]
        for (n, r), x in self.ref.items():
            eq, dates = ledger(self.tmp, n, f"ledger_rep{r:02d}.csv")
            self.assertEqual(dates[0], self.LO)
            self.assertEqual(dates[-1], self.HI)
            np.testing.assert_allclose(eq, x["res"]["eq"], rtol=0, atol=1e-12)
            _, _, _, decs = paper_arrays(self.tmp, n, r, CFGS[n])
            res = P.window_run(pd0, P._ts(self.LO), P._ts(self.HI), P.rep_targets(pd0, decs), P._is_frac(CFGS[n]))
            self.assertTrue(np.array_equal(res["pos"], x["res"]["pos"]), f"{n} rep{r}: 보유 경로가 다름")
            self.assertTrue(np.array_equal(res["eq"], x["res"]["eq"]))
        bh = self.W.run(baseline_targets(self.W)["B0"], 0.0015)
        for n in CFGS:
            eq, _ = ledger(self.tmp, n, "bh_ledger.csv")
            np.testing.assert_allclose(eq, bh["eq"], rtol=0, atol=1e-12)
        # 판단이 실제로 바뀌는 구간이어야 의미 있는 비교
        for key in (("T_R6", 0), ("T_R3", 0)):
            self.assertGreater(len(np.unique(self.ref[key]["res"]["pos"].round(6))), 1, key)

    def test_chain_log_matches(self):
        for (n, r), x in self.ref.items():
            path = os.path.join(self.tmp, n, f"train_log_rep{r:02d}.jsonl")
            with open(path, encoding="utf-8") as f:
                log = [json.loads(l) for l in f if l.strip()]
            ref = x["run"]["log"]
            self.assertEqual([e["month"] for e in log[:len(ref)]], [e["month"] for e in ref])
            strip = lambda e: json.dumps({k: v for k, v in e.items() if k != "secs"}, sort_keys=True, default=str)
            self.assertEqual([strip(e) for e in log[:len(ref)]], [strip(e) for e in ref])
            self.assertEqual(log[-1]["month"], "2024-04-01")         # now = 4월 1일 00:00 → 4월 모델까지 학습

    def test_report_and_verdict(self):
        rep = P.report("2024-04-01T00:00Z", paper_dir=self.tmp)
        W2 = Window(self.W.datas, self.LO, "2024-03-15", "test_btc_paper")
        bh = self.W.run(baseline_targets(self.W)["B0"], 0.0015)
        for n in CFGS:
            v = rep["variants"][n]
            self.assertEqual(v["checkpoints"]["2024-02-15"]["status"], "interim")
            fin = v["checkpoints"]["2024-03-15"]
            self.assertEqual(fin["status"], "final")
            from btc import stats as S
            from btc.evaluate import lower_median
            bh2 = W2.run(baseline_targets(W2)["B0"], 0.0015)
            srs, dds = [], []
            for r in range(REPS[n]):
                x = self.ref[(n, r)]
                t = x["tg"].copy()
                res = W2.run(t, 0.0015, weights=True) if x["frac"] else W2.run(t, 0.0015)
                s = S.summary(res["r"])
                srs.append(s["sharpe"])
                dds.append(s["max_dd"])
            sb = S.summary(bh2["r"])
            lm = lower_median([s - sb["sharpe"] for s in srs])
            self.assertEqual(fin["headline_rep"], lm)
            self.assertAlmostEqual(fin["headline"]["sharpe"], srs[lm], places=12)
            want = "pass" if (srs[lm] > sb["sharpe"] and dds[lm] > sb["max_dd"]) else "fail"
            self.assertEqual(v["verdict"], want)
            self.assertEqual(v["since_start"]["end"], self.HI)
            self.assertAlmostEqual(v["since_start"]["bh"]["sharpe"], S.summary(bh["r"])["sharpe"], places=12)
        early = P.report("2024-03-01T00:00Z", paper_dir=self.tmp, write=False)
        for n in CFGS:
            self.assertEqual(early["variants"][n]["verdict"], "pending")
            self.assertEqual(early["variants"][n]["checkpoints"]["2024-03-15"]["status"], "not_reached")

    def test_idempotent(self):
        before = snapshot(self.tmp)
        out = P.step("2024-04-01T00:00Z", offline=True, paper_dir=self.tmp)
        self.assertEqual(snapshot(self.tmp), before)
        self.assertTrue(all(x["new_decisions"] == 0 and x["trained_now"] == 0
                            for v in out["variants"].values() for x in v["reps"].values()))
        with self.assertRaises(P.PaperError):                          # 시간을 거꾸로 돌리면 거부
            P.step("2024-03-30T00:00Z", offline=True, paper_dir=self.tmp)

    def test_register_refuses_changes(self):
        reg = read_bytes(P.reg_path(self.tmp))
        P.register("T_R6", reps=2, start=self.LO, cfg=CFGS["T_R6"], paper_dir=self.tmp,
                   checkpoints=("2024-02-15", "2024-03-15"))                 # 같으면 그대로
        self.assertEqual(read_bytes(P.reg_path(self.tmp)), reg)
        with self.assertRaises(P.PaperError):
            P.register("T_R6", reps=2, start=self.LO, cfg=dict(CFGS["T_R6"], ft_steps=3), paper_dir=self.tmp,
                       checkpoints=("2024-02-15", "2024-03-15"))
        with self.assertRaises(P.PaperError):
            P.register("T_R6", reps=3, start=self.LO, cfg=CFGS["T_R6"], paper_dir=self.tmp,
                       checkpoints=("2024-02-15", "2024-03-15"))
        self.assertEqual(read_bytes(P.reg_path(self.tmp)), reg)


class DPBandParity(unittest.TestCase):
    """판단 호출에 상태(지난달 포지션)가 있는 dp_band — 달마다 이어지는 포지션까지 백테스트와 같음"""

    def test_dp_band(self):
        tmp = tempfile.mkdtemp(prefix="paper_dp_")
        try:
            P.register("T_DP", reps=1, start="2017-02-15", cfg=DP, paper_dir=tmp)
            P.step("2017-05-01T00:00Z", offline=True, paper_dir=tmp)
            W, ref = reference({"T_DP": DP}, {"T_DP": 1}, "2017-02-15", "2017-05-01")
            x = ref[("T_DP", 0)]
            d = W.datas[0]
            a, b = W.rng[0]
            sel = np.arange(a, b)[walk.decision_mask(d, 6)[a:b]]
            ts, U, tg, _ = paper_arrays(tmp, "T_DP", 0, DP)
            m = ts < P._ts("2017-05-01")
            self.assertTrue(np.array_equal(U[m], x["run"]["U"][0.0][sel]))
            self.assertTrue(np.array_equal(tg[m], x["tg"][sel], equal_nan=True))
            eq, _ = ledger(tmp, "T_DP", "ledger_rep00.csv")
            np.testing.assert_allclose(eq, x["res"]["eq"], rtol=0, atol=1e-12)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class CatchUp(unittest.TestCase):
    """하루씩 step (달 경계, 판단은 됐지만 체결가가 아직 없는 시각 포함) == 한 번에 step"""

    NOWS = ["2017-03-28T00:10Z", "2017-03-29T13:00Z", "2017-03-31T02:00Z", "2017-04-01T00:05Z",
            "2017-04-01T05:00Z", "2017-04-02T00:20Z", "2017-04-04T00:20Z"]
    CF = {"T_R6": CFGS["T_R6"], "T_DP": DP}

    def _reg(self, tmp):
        for n, c in self.CF.items():
            P.register(n, reps=1, start="2017-03-27", cfg=c, paper_dir=tmp)

    def test_daily_equals_single(self):
        a, b = tempfile.mkdtemp(prefix="paper_a_"), tempfile.mkdtemp(prefix="paper_b_")
        try:
            self._reg(a)
            self._reg(b)
            alarms = []
            for now in self.NOWS:
                alarms += P.step(now, offline=True, paper_dir=a)["alarms"]
            self.assertFalse([x for x in alarms if x["alarm"] == "recheck_mismatch"], alarms)
            P.step(self.NOWS[-1], offline=True, paper_dir=b)
            sa, sb = snapshot(a), snapshot(b)
            self.assertEqual(sorted(sa), sorted(sb))
            for f in sa:
                if f.startswith(("T_R6/decisions", "T_DP/decisions")):
                    ra = [{k: v for k, v in x.items() if k not in ("decided_at", "lag_h")}
                          for x in P._read_csv(os.path.join(a, f))]
                    rb = [{k: v for k, v in x.items() if k not in ("decided_at", "lag_h")}
                          for x in P._read_csv(os.path.join(b, f))]
                    self.assertEqual(ra, rb, f)
                    self.assertGreaterEqual(len(ra), 8)
                elif "train_log" in f:
                    strip = lambda s: [{k: v for k, v in json.loads(l).items() if k != "secs"}
                                       for l in s.decode().splitlines() if l.strip()]
                    self.assertEqual(strip(sa[f]), strip(sb[f]), f)
                elif f.endswith(".csv"):
                    self.assertEqual(sa[f], sb[f], f)
            # 하루씩 돌린 쪽의 판단 시각은 실제로 그 날 (판단봉 마감 후 1일 미만)
            for x in P._read_csv(os.path.join(a, "T_R6", "decisions_rep00.csv")):
                self.assertLess(float(x["lag_h"]), 48.0)
            # 같은 now로 다시 돌려도 그대로
            before = snapshot(a)
            P.step(self.NOWS[-1], offline=True, paper_dir=a)
            self.assertEqual(snapshot(a), before)
        finally:
            shutil.rmtree(a, ignore_errors=True)
            shutil.rmtree(b, ignore_errors=True)

    def test_recheck_detects_unpadded(self):
        """묶음 채우기를 끄면(달 중간 묶음 크기가 달라짐) 달 마감 때 recheck_mismatch 경고가 나야 함 — 검사 자체의 민감도"""
        tmp = tempfile.mkdtemp(prefix="paper_rc_")
        try:
            P.register("T_R6", reps=1, start="2017-03-27", cfg=CFGS["T_R6"], paper_dir=tmp)
            with mock.patch.object(P, "_future_slots", lambda *a: 0):
                P.step("2017-04-01T00:05Z", offline=True, paper_dir=tmp)       # 4월 1일 판단: 묶음 1개 (백테스트는 30개)
                out = P.step("2017-05-01T00:30Z", offline=True, paper_dir=tmp)  # 4월 마감 호출 → 비교
            self.assertIn("recheck_mismatch", [x["alarm"] for x in out["alarms"]])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_frozen_mismatch_skips(self):
        tmp = tempfile.mkdtemp(prefix="paper_fz_")
        try:
            P.register("T_R6", reps=1, start="2017-03-27", cfg=CFGS["T_R6"], paper_dir=tmp)
            reg = P.load_registry(tmp)
            reg["variants"]["T_R6"]["code_hash"] = "000000000000"
            P._write_json(P.reg_path(tmp), reg)
            out = P.step("2017-03-28T00:10Z", offline=True, paper_dir=tmp)
            self.assertIn("frozen_mismatch", [x["alarm"] for x in out["alarms"]])
            self.assertFalse(os.path.exists(os.path.join(tmp, "T_R6", "decisions_rep00.csv")))
            self.assertEqual(P.main(["--dir", tmp, "step", "--offline", "--now", "2017-03-28T00:10Z"]), 2)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class NoLookAhead(unittest.TestCase):
    """판단봉 다음 시가 이후 가격을 모두 바꿔도 그 판단과 체결가, 그날까지의 장부가 같음"""

    def test_corrupt_future(self):
        now = "2017-04-12T00:20Z"
        D = P._ts("2017-04-06")                                  # 판단봉 종가 = 체결 봉(00:00~04:00) 시작
        base = load_15m()
        bad = base.copy()
        rng = np.random.default_rng(0)
        after = bad["ts"].to_numpy() >= D + 900
        f = rng.uniform(0.5, 2.0, size=after.sum())
        for k in ("open", "high", "low", "close"):
            bad.loc[after, k] = bad.loc[after, k].to_numpy() * f
        first = bad["ts"].to_numpy() == D                         # 체결 봉 첫 15분: 시가만 그대로
        for k in ("high", "low", "close"):
            bad.loc[first, k] = bad.loc[first, k].to_numpy() * 1.37
        dirs = [tempfile.mkdtemp(prefix="paper_la_"), tempfile.mkdtemp(prefix="paper_lb_")]
        try:
            outs = []
            for tmp, df in zip(dirs, (base, bad)):
                P.register("T_R6", reps=1, start="2017-03-27", cfg=CFGS["T_R6"], paper_dir=tmp)
                P.step(now, offline=True, paper_dir=tmp, df15=df)
                decs = P._read_csv(os.path.join(tmp, "T_R6", "decisions_rep00.csv"))
                d0 = P.phases_for(P.truncate(df, P._ts(now)), 1)[0]
                i = int(np.searchsorted(d0.ts + BAR_SEC, D))
                self.assertEqual(int(d0.ts[i] + BAR_SEC), D)
                eq, dates = ledger(tmp, "T_R6", "ledger_rep00.csv")
                outs.append(dict(decs=decs, fill=d0.o[i + 1], eq=dict(zip(dates, eq))))
            a, b = outs
            upto = lambda ds: [x for x in ds if int(x["ts_close"]) <= D]
            self.assertEqual(upto(a["decs"]), upto(b["decs"]))
            self.assertTrue(any(int(x["ts_close"]) == D for x in a["decs"]))
            self.assertEqual(a["fill"], b["fill"])
            for day in a["eq"]:
                if P._ts(day) <= D:
                    self.assertEqual(a["eq"][day], b["eq"][day], day)
            # 망가뜨린 효과는 실제로 있어야 함 (그 뒤 판단값이 달라짐)
            later = lambda ds: [(x["u0"], x["u1"]) for x in ds if int(x["ts_close"]) > D + 86400]
            self.assertNotEqual(later(a["decs"]), later(b["decs"]))
        finally:
            for t in dirs:
                shutil.rmtree(t, ignore_errors=True)


class Extend(unittest.TestCase):
    """15분봉 덧붙이기 — 가짜 거래소 응답 (btc.live.fetch_bitstamp_minutes 자리)"""

    T0 = P._ts("2030-01-01")

    def _minutes(self, a, b, rng):
        t = np.arange(a, b, 60, dtype=np.int64)
        c = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, len(t))))
        o = np.concatenate([[100.0], c[:-1]])
        return pd.DataFrame(dict(timestamp=t, open=o, high=np.maximum(o, c) * 1.0005, low=np.minimum(o, c) * 0.9995,
                                 close=c, volume=rng.uniform(0.1, 2.0, len(t))))

    def test_extend_append_only(self):
        tmp = tempfile.mkdtemp(prefix="paper_ext_")
        try:
            rng = np.random.default_rng(1)
            allm = self._minutes(self.T0 - 4 * 3600, self.T0 + 6 * 3600, rng)
            base = minutes_to_15m(allm[allm["timestamp"] < self.T0], cutoff=pd.Timestamp(self.T0, unit="s", tz="UTC"))
            bpath = os.path.join(tmp, "base.csv.gz")
            base[COLS].to_csv(bpath, index=False, compression={"method": "gzip", "mtime": 0})
            gap = (allm["timestamp"] >= self.T0 + 1800) & (allm["timestamp"] < self.T0 + 1800 + 300)   # 5분 빠짐
            lag = allm["timestamp"] < self.T0 + 2 * 3600 + 7 * 60                                    # 늦게 온 분
            calls = []

            def fetch1(s, e):
                calls.append((s, e))
                m = allm[(allm["timestamp"] >= s) & (allm["timestamp"] < e) & ~gap & lag]
                return m.reset_index(drop=True)

            info = P.extend_cache(tmp, bpath, now=self.T0 + 3 * 3600 + 30, fetch=fetch1)
            self.assertEqual(calls[0], (self.T0, self.T0 + 3 * 3600 - 900))       # 진행 중인 15분 구간은 안 받음
            self.assertEqual(info["added"], 8)                                    # 받은 분이 끝나는 2시간까지만
            ext = pd.read_csv(P.ext_path(tmp))
            self.assertTrue(np.array_equal(ext["ts"], self.T0 + 900 * np.arange(8)))
            self.assertEqual(int(ext.loc[ext["ts"] == self.T0 + 1800, "active_min"].iloc[0]), 10)
            filled = allm[(allm["timestamp"] < self.T0 + 7200) & (allm["timestamp"] >= self.T0)].copy()
            filled.loc[gap, "volume"] = 0.0                                   # 빠진 분 = 거래 없는 분
            want = minutes_to_15m(filled, cutoff=pd.Timestamp(self.T0 + 7200, unit="s", tz="UTC"))
            for k in ("open", "high", "low", "close"):
                np.testing.assert_allclose(ext[k].to_numpy(), want[k].round(2).to_numpy(), rtol=0, atol=0)

            def fetch2(s, e):
                return allm[(allm["timestamp"] >= s) & (allm["timestamp"] < e)].reset_index(drop=True)

            info = P.extend_cache(tmp, bpath, now=self.T0 + 5 * 3600 + 30, fetch=fetch2)
            self.assertEqual(info["added"], 11)
            ext2 = pd.read_csv(P.ext_path(tmp))
            pd.testing.assert_frame_equal(ext2.iloc[:8].reset_index(drop=True), ext)       # 덧붙이기만
            self.assertEqual(P.extend_cache(tmp, bpath, now=self.T0 + 5 * 3600 + 30, fetch=fetch2)["added"], 0)
            both = P.load_15m_all(tmp, bpath)
            self.assertTrue(np.all(np.diff(both["ts"].to_numpy()) == 900))
            self.assertEqual(int(both["ts"].iloc[0]), int(base["ts"].iloc[0]))
            self.assertEqual(len(gzip.decompress(read_bytes(P.ext_path(tmp))).splitlines()), 20)

            def boom(s, e):
                raise RuntimeError("조회 실패")

            out = P.step(self.T0 + 6 * 3600, offline=False, paper_dir=tmp, base_path=bpath, fetch=boom)
            self.assertIn("fetch_failed", [x["alarm"] for x in out["alarms"]])
            self.assertTrue(os.path.exists(os.path.join(tmp, "status.json")))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
