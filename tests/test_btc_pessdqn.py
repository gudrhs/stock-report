"""P1 pessdqn(비관적 앙상블 DQN: 평균 − 1.0×표본 표준편차 판단) 단위 테스트 — 합성 데이터만, 1스레드.

python -m unittest tests.test_btc_pessdqn -v
  · proposed_cfg = R6 + P1 키 (algo, pess_kappa 1.0, pess_ddof 1)
  · 같은 시드로 2달(처음부터 + 이어서) 돌리면 속 KEnsemble 의 파라미터·앵커·월 기록이 R6(walk DQN 경로)와 비트 단위로 같음
    (점검 'p0' 그대로, 그리고 이어학습 경로를 확실히 타도록 점검 'none')
  · values() = 멤버 출력에서 따로 계산한 평균 − 1.0 × 표본 표준편차(ddof 1), |U| 최댓값은 R6와 같음
  · κ_p = 0 이면 R6(KEnsemble.values)와 비트 단위로 같음
  · walk.run_replication 전체(달력만 2달로 바꿈): 월 기록·마지막 상태가 R6와 같고, 저장되는 U 는 R6 이하,
    κ_p = 0 이면 저장되는 U 가 R6와 같음
"""
import os
import sys
import unittest

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from btc.research.rl import KEnsemble
from btc.research.variants import VARIANTS
from btc.research.algos import pessdqn as P

SMALL = dict(phases=2, cold_steps=30, cold_split=20, ft_steps=10)


def _ts(s):
    return pd.Timestamp(s, tz="UTC")


def r6_cfg(**kw):
    cfg = dict(VARIANTS["R6_daily_trend8_uniform"])
    cfg.update(SMALL, **kw)
    return cfg


def p1_cfg(**kw):
    cfg = P.proposed_cfg()
    cfg.update(SMALL, **kw)
    return cfg


_PHASE = {}


def _phase(seed=3, n=7000):
    if seed not in _PHASE:
        from test_btc_controls import _series
        from btc import features as Fe
        from btc.env import PhaseData
        df, _ = _series("regime", seed, n=n)
        X, sig = Fe.compute(df)
        _PHASE[seed] = PhaseData(df, X, sig)
    return _PHASE[seed]


def _run(cfg, d, months):
    from btc.research.walk import monthly_update
    ens, anc, log = None, None, []
    for T in months:
        ens, anc, e = monthly_update(cfg, [d, d], T, (0, T.year, T.month, 0), ens, anc)
        log.append(e)
    return ens, anc, log


def _same_params(a, b):
    return len(a) == len(b) and all(p.dtype == q.dtype and np.array_equal(p, q) for p, q in zip(a, b))


MONTHS = (_ts("2015-01-01"), _ts("2015-02-01"))


class Cfg(unittest.TestCase):
    def test_proposed_cfg_is_r6_plus_p1_keys(self):
        cfg = P.proposed_cfg()
        r6 = VARIANTS["R6_daily_trend8_uniform"]
        extra = {"algo": "pessdqn", "pess_kappa": 1.0, "pess_ddof": 1}
        self.assertEqual({k: v for k, v in cfg.items() if k not in extra and k != "name"},
                         {k: v for k, v in r6.items() if k != "name"})
        for k, v in extra.items():
            self.assertEqual(cfg[k], v)
        self.assertEqual(cfg["name"], "P1_pessimistic_dqn")


class SameTrainingAsR6(unittest.TestCase):
    def _compare(self, gate):
        d = _phase()
        e6, a6, l6 = _run(r6_cfg(gate=gate), d, MONTHS)
        e1, a1, l1 = _run(p1_cfg(gate=gate), d, MONTHS)
        self.assertEqual(l1, l6)                                      # 점검 결과(노출도·전환·불일치율)까지 같음 → 평균 U 로 점검
        if e6 is None:
            self.assertIsNone(e1)
            return l1
        self.assertIsInstance(e1, P.PessEnsemble)
        self.assertIsInstance(e1.inner, KEnsemble)
        self.assertNotIsInstance(e1.inner, P.PessEnsemble)
        self.assertTrue(_same_params(e1.inner.net.params, e6.net.params))
        self.assertTrue(_same_params(a1, a6))
        np.testing.assert_array_equal(e1.acts, e6.acts)
        return l1

    def test_gate_p0_bit_identical(self):
        self._compare("p0")

    def test_gate_none_finetune_bit_identical(self):
        log = self._compare("none")
        self.assertEqual([e["kind"] for e in log], ["cold", "finetune"])
        self.assertTrue(all(e["accepted"] for e in log))

    def test_previous_wrapper_is_unwrapped(self):
        """이어학습 달에 walk 로 넘어가는 모델은 속 KEnsemble (점검 거부로 옛 모델이 돌아와도 다시 포장)"""
        from btc.research import walk
        d = _phase()
        cfg = p1_cfg(gate="none")
        e1, a1, _ = _run(cfg, d, MONTHS[:1])
        seen = {}
        orig = walk.monthly_update

        def spy(c, datas, T_k, seed, ens, anchor):
            seen.update(algo=c.get("algo"), ens=ens, anchor=anchor)
            return ens, anchor, dict(month=str(T_k.date()), kind="finetune", accepted=False)   # 거부 흉내

        walk.monthly_update = spy
        try:
            out, anc, _ = P.monthly_update(cfg, [d, d], MONTHS[1], (0, 2015, 2, 0), e1, a1)
        finally:
            walk.monthly_update = orig
        self.assertIsNone(seen["algo"])
        self.assertIs(seen["ens"], e1.inner)
        self.assertIs(seen["anchor"], a1)
        self.assertIsInstance(out, P.PessEnsemble)
        self.assertIs(out.inner, e1.inner)
        self.assertEqual(out.kappa, 1.0)
        self.assertEqual(out.ddof, 1)


class Values(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = _phase()
        cls.ens, _, _ = _run(p1_cfg(gate="none"), cls.d, MONTHS)
        cls.sel = np.arange(cls.d.T - 400, cls.d.T - 10)

    def test_mean_minus_sample_std(self):
        inner = self.ens.inner
        K = len(inner.acts)
        for cost in (0.001, 0.003):
            U, umax = self.ens.values(self.d.X[self.sel], cost)
            # 멤버 출력에서 따로: 반복문으로 평균·표본 표준편차
            out = inner.net.forward(inner.inputs(self.d.X[self.sel], cost), cache=False)[..., :K].astype(np.float64)
            M = out.shape[0]
            self.assertEqual(M, 5)
            mu = sum(out[m] for m in range(M)) / M
            sd = np.sqrt(sum((out[m] - mu) ** 2 for m in range(M)) / (M - 1))
            self.assertEqual(U.shape, (len(self.sel), K))
            np.testing.assert_allclose(U, mu - 1.0 * sd, rtol=0, atol=1e-12)
            np.testing.assert_array_equal(umax, inner.values(self.d.X[self.sel], cost)[1])
            self.assertTrue(np.all(sd > 0))
            self.assertTrue(np.all(U < inner.values(self.d.X[self.sel], cost)[0]))

    def test_kappa0_is_r6_exactly(self):
        inner = self.ens.inner
        e0 = P.PessEnsemble(inner, kappa=0.0)
        for cost in (0.001, 0.0015, 0.002, 0.003):
            u0, m0 = e0.values(self.d.X[self.sel], cost)
            u6, m6 = inner.values(self.d.X[self.sel], cost)
            np.testing.assert_array_equal(u0, u6)
            np.testing.assert_array_equal(m0, m6)

    def test_kappa0_cfg_path_is_r6_exactly(self):
        """cfg pess_kappa=0 으로 walk 플러그인 경로를 돌려도 판단 가치가 R6 모델과 비트 단위로 같음"""
        e6, _, _ = _run(r6_cfg(gate="none"), self.d, MONTHS)
        e0, _, _ = _run(p1_cfg(gate="none", pess_kappa=0.0), self.d, MONTHS)
        u0, m0 = e0.values(self.d.X[self.sel], 0.003)
        u6, m6 = e6.values(self.d.X[self.sel], 0.003)
        np.testing.assert_array_equal(u0, u6)
        np.testing.assert_array_equal(m0, m6)

    def test_state_delegates(self):
        st, s6 = self.ens.state(), self.ens.inner.state()
        for k, v in s6.items():
            np.testing.assert_array_equal(st[k], v)
        self.assertEqual(set(st) - set(s6), {"pess_kappa", "pess_ddof"})


class Replication(unittest.TestCase):
    def test_run_replication_two_months(self):
        """walk.run_replication 을 그대로 (달력만 2달): 기록·마지막 상태 = R6, 저장 U ≤ R6, κ_p = 0 이면 저장 U = R6"""
        from btc.research import walk
        d = _phase()
        orig = walk.months
        walk.months = lambda *a, **k: list(MONTHS)
        try:
            res = {name: walk.run_replication(cfg, 0, datas=[d, d], end=_ts("2015-03-01"))
                   for name, cfg in (("r6", r6_cfg(gate="none")), ("p1", p1_cfg(gate="none")),
                                     ("p1k0", p1_cfg(gate="none", pess_kappa=0.0)))}
        finally:
            walk.months = orig
        strip = lambda log: [{k: v for k, v in e.items() if k != "secs"} for e in log]
        self.assertEqual(strip(res["p1"]["log"]), strip(res["r6"]["log"]))
        for k, v in res["r6"]["last_state"].items():
            np.testing.assert_array_equal(res["p1"]["last_state"][k], v)
        for cd in walk.C_DECS:
            u6, u1, u0 = res["r6"]["U"][cd], res["p1"]["U"][cd], res["p1k0"]["U"][cd]
            fin = np.isfinite(u6)
            self.assertGreater(fin.sum(), 50)
            np.testing.assert_array_equal(np.isfinite(u1), fin)
            self.assertTrue(np.all(u1[fin] < u6[fin]))
            np.testing.assert_array_equal(u0, u6)                    # NaN 위치까지 같음


if __name__ == "__main__":
    unittest.main()
