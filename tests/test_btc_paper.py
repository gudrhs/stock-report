"""3단계 모의매매(btc/research/paper.py) 테스트 — 네트워크 없음, 연구용 15분봉 캐시(data/btc) 사용, 1스레드.

python -m unittest tests.test_btc_paper -v
  · 재현(Parity): 과거 시작일로 등록 → step --offline → 판단값(U·비중)이 walk.run_replication 과 비트 단위로 같고,
    목표 비중이 screen.targets 와 같고, 장부가 evaluate.Window.run 과 같음 (0/1·비율 보유 DQN, 플러그인 W1, 상태 있는 dp_band)
  · 학습 자르기(TrainCut): 월 T 학습이 실행 시각과 무관 — 마지막 표본을 꼭 뽑는 설정으로 T+20분 데이터 학습은
    백테스트와 다르고(검사 민감도), 모의매매(하루씩·한 번에)는 c(T)=T+1일에서 잘라 백테스트와 비트 단위로 같음
  · 따라잡기(CatchUp): 하루씩 여러 번 step == 한 번에 step (달 경계·체결 대기 포함, 달 중간 묶음 채우기 확인)
  · 멱등(Idempotent): 같은 now로 두 번 돌려도 파일이 바이트 단위로 같음
  · 중단 복구(Crash): 새 달 상태 저장 뒤 판단 중 죽어도 판단이 사라지지 않고, 복구 결과가 한 번에 돌린 것과 같음
  · 운영 중단 규칙(StopRule): 학습 실패·고정 불일치·중간에 빠진 판단봉 → 종가 + 2일 뒤 현금 기록, 장부·판정 계속
  · 등록 거부(Register): 같은 이름에 다른 설정·반복 수 거부, 코드·패키지 지문이 바뀌면 step이 처리하지 않음
  · 미래 정보 없음(NoLookAhead): 판단봉 다음 시가 이후 가격을 모두 망가뜨려도 그 판단·체결가가 같음
  · 데이터 덧붙이기(Extend): 가짜 거래소 응답으로 15분봉 덧붙이기(빠진 분·늦은 분·덧붙이기만·달 파일), 조회 실패 시 계속
학습 스텝·멤버·phase 수는 줄인 설정을 쓰되, 기준(run_replication)과 모의매매가 같은 설정을 씁니다.
"""
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
# 마지막 학습 표본(t+43봉이 T에 마감)을 반드시 여러 번 뽑는 설정: 최근 가중만, 반감기 약 1일, 이어학습 스텝 늘림
TC = tiny("R6_daily_trend8_uniform", recency_frac=1.0, half_life_years=0.003, ft_steps=40, cold_steps=40)
_DATAS = {}
NOKEYS = ("decided_at", "lag_h")


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


def reference(cfgs, reps, lo, hi, run_end=None):
    """기준: run_replication(끝 = run_end 또는 hi) → screen.targets·run_tg (저장 파일 대신 메모리의 결과를 넘김)"""
    datas = datas_full()
    end = pd.Timestamp(run_end or hi, tz="UTC")
    runs = {(n, r): walk.run_replication(dict(c, name=n), r, datas=datas, end=end)
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
    decs = P._read_decs(os.path.join(tmp, name), r)
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


def strip_decs(rows):
    return [{k: v for k, v in x.items() if k not in NOKEYS} for x in rows]


def strip_log(path):
    with open(path, encoding="utf-8") as f:
        return [{k: v for k, v in json.loads(l).items() if k not in ("secs", "data_cut")} for l in f if l.strip()]


def strip_ref_log(log):
    return [json.loads(json.dumps({k: v for k, v in e.items() if k != "secs"}, default=str)) for e in log]


class Parity(unittest.TestCase):
    """과거 시작일(2024-01-01)로 등록한 모의매매 == 같은 설정의 run_replication + screen + Window.run"""

    LO, HI = "2024-01-01", "2024-04-01"
    NOW2 = "2024-04-02T00:20Z"            # 4월 모델은 c(T) = 4월 2일 00:00 이후에야 학습 → 4월 1·2일 봉을 이때 판단

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="paper_parity_")
        for n, c in CFGS.items():
            P.register(n, reps=REPS[n], start=cls.LO, cfg=c, paper_dir=cls.tmp, checkpoints=("2024-02-15", "2024-03-15"))
        cls.out1 = P.step("2024-04-01T00:00Z", offline=True, paper_dir=cls.tmp)
        cls.led1 = {f: read_bytes(os.path.join(cls.tmp, f)) for n in CFGS for f in
                    [os.path.join(n, f"ledger_rep{r:02d}.csv") for r in range(REPS[n])] + [os.path.join(n, "bh_ledger.csv")]}
        cls.log1 = {(n, r): strip_log(os.path.join(cls.tmp, n, f"train_log_rep{r:02d}.jsonl"))
                    for n in CFGS for r in range(REPS[n])}
        cls.out = P.step(cls.NOW2, offline=True, paper_dir=cls.tmp)
        cls.W, cls.ref = reference(CFGS, REPS, cls.LO, cls.HI, run_end="2024-05-01")
        cls.Wt = Window(cls.W.datas, cls.LO, "2024-04-03", "test_btc_paper")      # 4월 1·2일 판단까지 목표 비중 기준
        with mock.patch.object(screen, "load_run", lambda n, r: cls.ref[(n, r)]["run"]), \
                mock.patch.object(screen, "VARIANTS", {n: dict(c, name=n) for n, c in CFGS.items()}):
            for (n, r), x in cls.ref.items():
                tg, _, _ = screen.targets(cls.Wt, n, r)
                x["tg_t"] = tg[0.003] if isinstance(tg, dict) else tg

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_decisions_match_backtest(self):
        d = self.Wt.datas[0]
        a, b = self.Wt.rng[0]
        close = d.ts + BAR_SEC
        sel = np.arange(a, b)[walk.decision_mask(d, 6)[a:b]]
        self.assertEqual(int(close[sel[-1]]), P._ts("2024-04-02"))
        for (n, r), x in self.ref.items():
            cfg = CFGS[n]
            ts, U, tg, decs = paper_arrays(self.tmp, n, r, cfg)
            self.assertTrue(np.array_equal(ts, close[sel]), n)
            Uref = x["run"]["U"][0.0 if P._is_weights(cfg) else 0.003][sel]
            self.assertTrue(np.array_equal(U, Uref, equal_nan=True), f"{n} rep{r}: U가 비트 단위로 다름")
            self.assertTrue(np.array_equal(tg, x["tg_t"][sel], equal_nan=True), f"{n} rep{r}: 목표 비중이 다름")
            self.assertFalse(any(z["reason"] for z in decs), f"{n}: 현금 처리된 봉이 없어야 함")
            self.assertGreater(len(np.unique(tg[~np.isnan(tg)])), 0)
            # 4월 1일 봉은 4월 2일 00:20에 판단 (c(T) = T + 1일) — 판단 지연이 기록됨
            apr1 = [z for z in decs if z["close_utc"] == "2024-04-01T00:00:00Z"][0]
            self.assertEqual(apr1["model_month"], "2024-04-01")
            self.assertEqual(apr1["decided_at"], "2024-04-02T00:20:00Z")

    def test_ledger_matches_window_run(self):
        pd0 = P.phases_for(P.truncate(P.load_15m_all(self.tmp), P._ts(self.HI)), 1)[0]
        for (n, r), x in self.ref.items():
            rows = self.led1[os.path.join(n, f"ledger_rep{r:02d}.csv")].decode().split()[1:]   # 4월 1일 00:00 실행의 장부
            dates1 = [z.split(",")[0] for z in rows]
            eq = np.array([float(z.split(",")[1]) for z in rows])
            self.assertEqual(dates1[0], self.LO)
            self.assertEqual(dates1[-1], self.HI)                      # 4월 1일 봉 미판단 → 장부 끝
            np.testing.assert_allclose(eq, x["res"]["eq"], rtol=0, atol=1e-12)
            _, _, _, decs = paper_arrays(self.tmp, n, r, CFGS[n])
            res = P.window_run(pd0, P._ts(self.LO), P._ts(self.HI), P.rep_targets(pd0, decs), P._is_frac(CFGS[n]))
            self.assertTrue(np.array_equal(res["pos"], x["res"]["pos"]), f"{n} rep{r}: 보유 경로가 다름")
            self.assertTrue(np.array_equal(res["eq"], x["res"]["eq"]))
        bh = self.W.run(baseline_targets(self.W)["B0"], 0.0015)
        for n in CFGS:
            rows = self.led1[os.path.join(n, "bh_ledger.csv")].decode().split()[1:]
            np.testing.assert_allclose([float(z.split(",")[1]) for z in rows], bh["eq"], rtol=0, atol=1e-12)
        # 판단이 실제로 바뀌는 구간이어야 의미 있는 비교
        for key in (("T_R6", 0), ("T_R3", 0)):
            self.assertGreater(len(np.unique(self.ref[key]["res"]["pos"].round(6))), 1, key)

    def test_chain_log_matches(self):
        for (n, r), x in self.ref.items():
            log = strip_log(os.path.join(self.tmp, n, f"train_log_rep{r:02d}.jsonl"))
            ref = strip_ref_log(x["run"]["log"])
            self.assertEqual([e["month"] for e in log], [e["month"] for e in ref])
            self.assertEqual(log, ref)
            self.assertEqual(log[-1]["month"], "2024-04-01")          # 4월 2일 00:20 → 4월 모델까지
            self.assertEqual(self.log1[(n, r)][-1]["month"], "2024-03-01")   # 4월 1일 00:00엔 아직 3월 모델

    def test_report_and_verdict(self):
        rep = P.report("2024-04-01T00:00Z", paper_dir=self.tmp)
        W2 = Window(self.W.datas, self.LO, "2024-03-15", "test_btc_paper")
        bh = self.W.run(baseline_targets(self.W)["B0"], 0.0015)
        from btc import stats as S
        from btc.evaluate import lower_median
        for n in CFGS:
            v = rep["variants"][n]
            self.assertEqual(v["checkpoints"]["2024-02-15"]["status"], "interim")
            fin = v["checkpoints"]["2024-03-15"]
            self.assertEqual(fin["status"], "final")
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
        out = P.step(self.NOW2, offline=True, paper_dir=self.tmp)
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
        with mock.patch.object(P, "frozen_hash", lambda: "ffffffffffff"):      # 판단·장부 코드가 바뀜
            with self.assertRaises(P.PaperError):
                P.register("T_R6", reps=2, start=self.LO, cfg=CFGS["T_R6"], paper_dir=self.tmp,
                           checkpoints=("2024-02-15", "2024-03-15"))
        self.assertEqual(read_bytes(P.reg_path(self.tmp)), reg)
        ent = P.load_registry(self.tmp)["variants"]["T_W1"]
        self.assertEqual(ent["pins"], dict(numpy=np.__version__, pandas=pd.__version__))
        self.assertIn("btc/research/direct.py", ent["algo_files"])            # 플러그인이 가져다 쓰는 연구 모듈도 고정
        self.assertIn("btc/research/paper.py", ent["frozen_files"])
        self.assertIn("btc/evaluate.py", ent["frozen_files"])
        self.assertIn("commit", ent["git"])
        # CLI 등록은 고정 대상 파일이 커밋과 다르면 거부 (CI가 그 커밋을 꺼내 돌리므로)
        with mock.patch.object(P, "git_info", lambda cfg: dict(commit="abc", dirty=["btc/research/rl.py"])):
            with self.assertRaises(P.PaperError):
                P.register("T_new", reps=1, start=self.LO, cfg=CFGS["T_R6"], paper_dir=self.tmp, require_clean=True)
        self.assertNotIn("T_new", P.load_registry(self.tmp)["variants"])


class TrainCut(unittest.TestCase):
    """월 T 학습은 실행 시각과 무관 (학습 자르기 c(T) = T + 1일). 전에는 T 직후 실행이 백테스트와 다른 모델을 만들었음"""

    NOWS = ["2017-01-01T00:20Z", "2017-01-02T00:20Z", "2017-01-31T00:20Z", "2017-02-01T00:20Z",
            "2017-02-02T00:20Z", "2017-02-04T00:20Z"]
    CF = {"T_TC": TC, "T_DP": DP}

    def test_sensitivity_tail_sample(self):
        """검사 민감도: 이 설정에서는 T+20분 데이터로 학습하면 백테스트(전체 데이터)와 실제로 달라짐"""
        base = load_15m()
        full = datas_full()
        J, F = pd.Timestamp("2017-01-01", tz="UTC"), pd.Timestamp("2017-02-01", tz="UTC")
        ens, anc, _ = walk.monthly_update(TC, full, J, (0, 2017, 1, 0), None, None)
        e1, _, l1 = walk.monthly_update(TC, full, F, (0, 2017, 2, 0), ens, anc)
        early = P.phases_for(P.truncate(base, P._ts(F) + 1200), 2)
        e2, _, l2 = walk.monthly_update(TC, early, F, (0, 2017, 2, 0), ens, anc)
        self.assertNotEqual(l1["td"], l2["td"])
        cut = P.phases_for(P.truncate(base, P._ts(F) + P.DAY), 2)          # c(T) = T + 1일 → 백테스트와 같음
        e3, _, l3 = walk.monthly_update(TC, cut, F, (0, 2017, 2, 0), ens, anc)
        self.assertEqual(l1["td"], l3["td"])
        X = full[0].X[:4000][-200:]
        self.assertTrue(np.array_equal(e1.values(X, 0.003)[0], e3.values(X, 0.003)[0]))

    def test_daily_single_and_backtest(self):
        base = load_15m()
        a, b = tempfile.mkdtemp(prefix="paper_tca_"), tempfile.mkdtemp(prefix="paper_tcb_")
        try:
            for d in (a, b):
                for n, c in self.CF.items():
                    P.register(n, reps=1, start="2017-01-01", cfg=c, paper_dir=d)
            outs = [P.step(now, offline=True, paper_dir=a, df15=base) for now in self.NOWS]   # 과거 달도 c(T)에서 자름
            P.step(self.NOWS[-1], offline=True, paper_dir=b, df15=base)
            # 1일 00:20에는 그 달을 학습하지 않음 (c(T) = 2일 00:00)
            self.assertIsNone(outs[0]["variants"]["T_TC"]["reps"][0]["trained_through"])
            self.assertEqual(outs[3]["variants"]["T_TC"]["reps"][0]["trained_through"], "2017-01-01")
            self.assertEqual(outs[4]["variants"]["T_TC"]["reps"][0]["trained_through"], "2017-02-01")
            _, ref = reference(self.CF, {n: 1 for n in self.CF}, "2017-01-01", "2017-02-05", run_end="2017-03-01")
            for n, c in self.CF.items():
                da, db = P._read_decs(os.path.join(a, n), 0), P._read_decs(os.path.join(b, n), 0)
                self.assertEqual(strip_decs(da), strip_decs(db), n)
                la = strip_log(os.path.join(a, n, "train_log_rep00.jsonl"))
                self.assertEqual(la, strip_log(os.path.join(b, n, "train_log_rep00.jsonl")), n)
                self.assertEqual(la, strip_ref_log(ref[(n, 0)]["run"]["log"]), n)
                self.assertEqual([e["month"] for e in la], ["2017-01-01", "2017-02-01"])
                with open(os.path.join(a, n, "train_log_rep00.jsonl"), encoding="utf-8") as f:
                    cuts = [json.loads(l)["data_cut"] for l in f if l.strip()]
                self.assertEqual(cuts, ["2017-01-02T00:00:00Z", "2017-02-02T00:00:00Z"])
                ts, U, _, _ = paper_arrays(a, n, 0, c)
                d0 = datas_full()[0]
                i = np.searchsorted(d0.ts + BAR_SEC, ts)
                Uref = ref[(n, 0)]["run"]["U"][0.0 if P._is_weights(c) else 0.003][i]
                self.assertTrue(np.array_equal(U, Uref, equal_nan=True), f"{n}: 학습 자르기 경로가 백테스트와 다름")
                self.assertEqual(ts[-1], P._ts("2017-02-04"))
                for f in ("ledger_rep00.csv", "fills_rep00.csv", "bh_ledger.csv"):
                    self.assertEqual(read_bytes(os.path.join(a, n, f)), read_bytes(os.path.join(b, n, f)), f)
        finally:
            shutil.rmtree(a, ignore_errors=True)
            shutil.rmtree(b, ignore_errors=True)


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
            self.assertFalse([x for x in alarms if x["alarm"] in ("recheck_mismatch", "decision_gap",
                                                                  "decisions_overdue", "cash_by_rule")], alarms)
            P.step(self.NOWS[-1], offline=True, paper_dir=b)
            sa, sb = snapshot(a), snapshot(b)
            self.assertEqual(sorted(sa), sorted(sb))
            for f in sa:
                if f.startswith(("T_R6/decisions", "T_DP/decisions")):
                    ra, rb = P._read_csv(os.path.join(a, f)), P._read_csv(os.path.join(b, f))
                    self.assertEqual(strip_decs(ra), strip_decs(rb), f)
                    self.assertGreaterEqual(len(ra), 8)
                elif "train_log" in f:
                    self.assertEqual(strip_log(os.path.join(a, f)), strip_log(os.path.join(b, f)), f)
                elif f.endswith(".csv"):
                    self.assertEqual(sa[f], sb[f], f)
            # 하루씩 돌린 쪽의 판단 시각은 실제로 그 날 (판단봉 마감 후 2일 미만)
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
        """달 중간 판단을 한 행씩(묶음 크기 1 — 행렬곱 대신 행렬·벡터 곱) 계산하면 달 마감 때 recheck_mismatch 경고가
        나야 함 — 검사 자체의 민감도"""
        tmp = tempfile.mkdtemp(prefix="paper_rc_")
        try:
            P.register("T_R6", reps=1, start="2017-03-27", cfg=CFGS["T_R6"], paper_dir=tmp)
            orig = P._outputs
            one_by_one = lambda ens, X, direct: np.concatenate([orig(ens, X[i:i + 1], direct) for i in range(len(X))])
            with mock.patch.object(P, "_outputs", one_by_one):
                P.step("2017-04-02T00:20Z", offline=True, paper_dir=tmp)
            out = P.step("2017-05-02T00:30Z", offline=True, paper_dir=tmp)  # 4월 마감 호출 → 비교
            self.assertIn("recheck_mismatch", [x["alarm"] for x in out["alarms"]])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class Crash(unittest.TestCase):
    """새 달 상태를 저장한 뒤 판단 중에 죽어도(예외·kill·시간 초과) 판단이 사라지지 않음"""

    def test_crash_after_state_save(self):
        a, b = tempfile.mkdtemp(prefix="paper_cr_"), tempfile.mkdtemp(prefix="paper_crb_")
        try:
            for d in (a, b):
                P.register("T_R6", reps=1, start="2017-03-27", cfg=CFGS["T_R6"], paper_dir=d)
            P.step("2017-03-28T00:10Z", offline=True, paper_dir=a)          # 3-27, 3-28 판단. 3-29..3-31 실행은 놓침
            orig = P._decide

            def boom(cfg, ens, d0, mask, Tk, Tn, *x, **k):
                if P._utc(Tk).month == 4:
                    raise MemoryError("4월 상태 저장 뒤 죽음")
                return orig(cfg, ens, d0, mask, Tk, Tn, *x, **k)

            with mock.patch.object(P, "_decide", boom):
                with self.assertRaises(MemoryError):
                    P.step("2017-04-02T00:20Z", offline=True, paper_dir=a)
            self.assertEqual(P._load_state(os.path.join(a, "T_R6"), 0)["month"], "2017-04-01")
            got = [x["close_utc"][:10] for x in P._read_decs(os.path.join(a, "T_R6"), 0)]
            self.assertEqual(got, ["2017-03-27", "2017-03-28", "2017-03-29", "2017-03-30", "2017-03-31"])
            out = P.step("2017-04-10T00:20Z", offline=True, paper_dir=a)
            rep = out["variants"]["T_R6"]["reps"][0]
            self.assertEqual(rep["undecided"], 0)
            self.assertEqual(rep["ledger_through"], "2017-04-10")
            self.assertFalse([x for x in out["alarms"] if x["alarm"] in ("decision_gap", "decisions_overdue")])
            P.step("2017-04-10T00:20Z", offline=True, paper_dir=b)
            # 복구 결과 == 한 번에 돌린 결과 (4월 k_policy 는 3-31 포지션에서 이어받음)
            self.assertEqual(strip_decs(P._read_decs(os.path.join(a, "T_R6"), 0)),
                             strip_decs(P._read_decs(os.path.join(b, "T_R6"), 0)))
            for f in ("ledger_rep00.csv", "fills_rep00.csv"):
                self.assertEqual(read_bytes(os.path.join(a, "T_R6", f)), read_bytes(os.path.join(b, "T_R6", f)))
        finally:
            shutil.rmtree(a, ignore_errors=True)
            shutil.rmtree(b, ignore_errors=True)


class StopRule(unittest.TestCase):
    """운영 중단 규칙 — 모델이 없는 판단봉은 종가 + 2일 뒤 현금(목표 0)으로 기록, 장부·판정은 계속"""

    def test_retrain_failure_goes_cash_then_recovers(self):
        tmp = tempfile.mkdtemp(prefix="paper_rf_")
        try:
            P.register("T_R6", reps=1, start="2017-03-27", cfg=CFGS["T_R6"], paper_dir=tmp,
                       checkpoints=("2017-04-03", "2017-04-08"))
            orig = walk.monthly_update

            def boom(cfg, datas, T, *a):
                if T.month == 4:
                    raise RuntimeError("학습 실패")
                return orig(cfg, datas, T, *a)

            alarms = []
            with mock.patch.object(walk, "monthly_update", boom):
                for now in ("2017-04-02T00:20Z", "2017-04-03T00:20Z", "2017-04-05T00:20Z"):
                    alarms += P.step(now, offline=True, paper_dir=tmp)["alarms"]
            kinds = [x["alarm"] for x in alarms]
            self.assertIn("retrain_failed", kinds)
            self.assertIn("cash_by_rule", kinds)
            decs = P._read_decs(os.path.join(tmp, "T_R6"), 0)
            cash = [x for x in decs if x["reason"]]
            self.assertEqual([x["close_utc"][:10] for x in cash], ["2017-04-01", "2017-04-02", "2017-04-03"])
            self.assertTrue(all(x["reason"] == "retrain_failed" and float(x["target"]) == 0.0 and x["model_month"] == ""
                                for x in cash))
            with open(os.path.join(tmp, "status.json"), encoding="utf-8") as f:
                st = json.load(f)
            self.assertEqual(st["variants"]["T_R6"]["reps"]["0"]["ledger_through"], "2017-04-04")
            out = P.step("2017-04-10T00:20Z", offline=True, paper_dir=tmp)       # 학습 복구 → 그 뒤 봉은 모델로
            self.assertEqual(out["variants"]["T_R6"]["reps"][0]["trained_through"], "2017-04-01")
            decs2 = P._read_decs(os.path.join(tmp, "T_R6"), 0)
            self.assertEqual(decs2[:len(decs)], decs)                           # 기록한 판단은 그대로
            later = decs2[len(decs):]
            self.assertEqual(later[0]["close_utc"][:10], "2017-04-04")
            self.assertTrue(all(x["model_month"] == "2017-04-01" and not x["reason"] for x in later))
            self.assertEqual(later[-1]["close_utc"][:10], "2017-04-10")
            rep = P.report("2017-04-10T00:20Z", paper_dir=tmp, write=False)["variants"]["T_R6"]
            self.assertIn(rep["verdict"], ("pass", "fail"))
            self.assertEqual(rep["cash_by_rule"], [{"retrain_failed": 3}])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_hole_is_flagged_and_filled(self):
        tmp = tempfile.mkdtemp(prefix="paper_hole_")
        try:
            P.register("T_R6", reps=1, start="2017-03-27", cfg=CFGS["T_R6"], paper_dir=tmp)
            P.step("2017-04-04T00:20Z", offline=True, paper_dir=tmp)
            vdir = os.path.join(tmp, "T_R6")
            decs = P._read_decs(vdir, 0)
            hole = [x for x in decs if x["close_utc"][:10] == "2017-03-30"]
            P._write_decs(vdir, 0, CFGS["T_R6"], [x for x in decs if x not in hole])
            out = P.step("2017-04-04T00:20Z", offline=True, paper_dir=tmp)
            self.assertIn("decision_gap", [x["alarm"] for x in out["alarms"]])
            new = P._read_decs(vdir, 0)
            filled = [x for x in new if x["close_utc"][:10] == "2017-03-30"][0]
            self.assertEqual((filled["reason"], filled["target"]), ("decision_gap", "0.0"))
            self.assertEqual(strip_decs([x for x in new if x is not filled]), strip_decs([x for x in decs if x not in hole]))
            self.assertEqual(out["variants"]["T_R6"]["reps"][0]["ledger_through"], "2017-04-04")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_frozen_mismatch_skips_then_cash(self):
        tmp = tempfile.mkdtemp(prefix="paper_fz_")
        try:
            P.register("T_R6", reps=1, start="2017-03-27", cfg=CFGS["T_R6"], paper_dir=tmp)
            P.register("T_R6b", reps=1, start="2017-03-27", cfg=CFGS["T_R6"], paper_dir=tmp)
            reg = P.load_registry(tmp)
            reg["variants"]["T_R6"]["code_hash"] = "000000000000"
            reg["variants"]["T_R6b"]["pins"] = dict(numpy="0.0.0", pandas=pd.__version__)   # 다른 numpy로 등록됐다면
            P._write_json(P.reg_path(tmp), reg)
            out = P.step("2017-03-28T00:10Z", offline=True, paper_dir=tmp)
            what = {x["variant"]: x["what"] for x in out["alarms"] if x["alarm"] == "frozen_mismatch"}
            self.assertEqual(what, {"T_R6": ["code_hash"], "T_R6b": ["pins"]})
            self.assertFalse(os.path.exists(os.path.join(tmp, "T_R6", "decisions_rep00.csv")))   # 기한 전: 기다림
            self.assertEqual(P.main(["--dir", tmp, "step", "--offline", "--now", "2017-03-28T00:10Z"]), 2)
            P.step("2017-04-02T00:20Z", offline=True, paper_dir=tmp)
            decs = P._read_decs(os.path.join(tmp, "T_R6"), 0)
            self.assertEqual([x["close_utc"][:10] for x in decs],
                             ["2017-03-27", "2017-03-28", "2017-03-29", "2017-03-30", "2017-03-31"])
            self.assertTrue(all(x["reason"] == "frozen_mismatch" and x["target"] == "0.0" for x in decs))
            self.assertFalse(os.path.exists(os.path.join(tmp, "T_R6", "state")))
            eq, dates = ledger(tmp, "T_R6", "ledger_rep00.csv")
            self.assertEqual(dates[-1], "2017-04-01")
            self.assertTrue(np.all(eq == 1.0))                                  # 현금 → 자산 그대로
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class NoLookAhead(unittest.TestCase):
    """판단봉 다음 시가 이후 가격을 모두 바꿔도 그 판단과 체결가, 그날까지의 장부가 같음 (학습도 c(T)에서 자름)"""

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
                decs = P._read_decs(os.path.join(tmp, "T_R6"), 0)
                d0 = P.phases_for(P.truncate(df, P._ts(now)), 1)[0]
                i = int(np.searchsorted(d0.ts + BAR_SEC, D))
                self.assertEqual(int(d0.ts[i] + BAR_SEC), D)
                eq, dates = ledger(tmp, "T_R6", "ledger_rep00.csv")
                outs.append(dict(decs=decs, fill=d0.o[i + 1], eq=dict(zip(dates, eq))))
            a, b = outs
            upto = lambda ds: strip_decs([x for x in ds if int(x["ts_close"]) <= D])
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
    """15분봉 덧붙이기 — 가짜 거래소 응답 (btc.live.fetch_bitstamp_minutes 자리). 달이 바뀌는 경계 포함"""

    T0 = P._ts("2030-01-31T22:00Z")

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
            jan = os.path.join(P.ext_dir(tmp), "btcusd_15m_2030-01.csv")
            self.assertEqual([os.path.basename(p) for p in P._ext_files(tmp)], ["btcusd_15m_2030-01.csv"])
            jan_bytes = read_bytes(jan)
            ext = P.read_ext(tmp)
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
            self.assertEqual([os.path.basename(p) for p in P._ext_files(tmp)],
                             ["btcusd_15m_2030-01.csv", "btcusd_15m_2030-02.csv"])
            self.assertEqual(read_bytes(jan), jan_bytes)                       # 지난달 파일은 그대로 (덧붙이기만)
            ext2 = P.read_ext(tmp)
            pd.testing.assert_frame_equal(ext2.iloc[:8].reset_index(drop=True), ext)
            self.assertEqual(P.extend_cache(tmp, bpath, now=self.T0 + 5 * 3600 + 30, fetch=fetch2)["added"], 0)
            both = P.load_15m_all(tmp, bpath)
            self.assertTrue(np.all(np.diff(both["ts"].to_numpy()) == 900))
            self.assertEqual(int(both["ts"].iloc[0]), int(base["ts"].iloc[0]))
            n_lines = sum(len(read_bytes(p).splitlines()) for p in P._ext_files(tmp))
            self.assertEqual(n_lines, 2 + 19)                                # 머리줄 2 + 15분봉 19개 (압축 없는 CSV)
            reread = pd.read_csv(os.path.join(P.ext_dir(tmp), "btcusd_15m_2030-02.csv"))
            self.assertTrue(np.array_equal(reread[COLS].to_numpy(), ext2.iloc[8:][COLS].to_numpy()))

            def boom(s, e):
                raise RuntimeError("조회 실패")

            out = P.step(self.T0 + 6 * 3600, offline=False, paper_dir=tmp, base_path=bpath, fetch=boom)
            self.assertIn("fetch_failed", [x["alarm"] for x in out["alarms"]])
            self.assertTrue(os.path.exists(os.path.join(tmp, "status.json")))
            self.assertFalse([f for f in snapshot(tmp) if f.endswith(".tmp")])     # 쓰다 만 파일 없음
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
