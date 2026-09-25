# -*- coding: utf-8 -*-
"""
P1 — 비관적 앙상블 DQN: 학습은 R6와 완전히 같고, 판단 가치만 '멤버 평균 − κ_p × 멤버 표준편차'.
사전 등록: btc/research/success_criteria.md 마지막 조항 '새 강화학습 방식 4가지 묶음' (커밋 441be61) 의 P1_pessimistic_dqn.

  "학습은 R6와 완전히 같습니다(같은 시드이므로 같은 모델). 판단만 멤버 5개의 U에서 평균 − 1.0×표준편차를 씁니다.
   점검(게이트)은 R6처럼 평균 U로 합니다."

어떻게 R6와 '같은 모델'을 보장하나
  · 이 파일은 학습 코드를 한 줄도 갖고 있지 않습니다. 매달 walk.monthly_update 를 algo=None 으로 그대로 부릅니다
    (KTrainer·같은 시드 (반복, 연, 월, 0)·처음부터/이어학습·L2-SP 앵커·점검 walk.gate_cold / walk.gate_finetune).
  · 그 안의 점검은 속(inner) KEnsemble 을 받으므로 R6처럼 '평균 U'로 한 포지션을 봅니다.
  · 이어학습 달에는 지난달 포장(PessEnsemble)을 벗겨 속 KEnsemble 을 넘깁니다 (init_from(ens.net.params),
    gate_finetune 의 옛 모델 포지션도 평균 U 로).
  · 돌아온 속 KEnsemble(점검 거부로 지난달 모델이 그대로 돌아온 경우 포함)을 PessEnsemble 로 다시 포장합니다.
    점검이 처음부터 실패해 모델이 없으면(None) 그대로 None (walk 가 그달은 현금).
  → 같은 시드면 속 모델의 파라미터·앵커·월 기록이 R6 실행과 비트 단위로 같습니다 (tests/test_btc_pessdqn.py).

판단 가치 (values 가 돌려주는 U, 판단비용 c 마다)
  U_m(s, c, a) : 멤버 m 의 U (KEnsemble.values 와 같은 입력·같은 float64 변환)
  U_pess(s, c, a) = 평균_m U_m − κ_p · sd_m U_m,   κ_p = 1.0 (cfg["pess_kappa"])
  sd 는 '표본 표준편차'(ddof = 1, 멤버 M = 5 이면 제곱편차 합 / 4 의 제곱근) — cfg["pess_ddof"] = 1 로 고정.
    (사전 등록 문구는 ddof 를 적지 않았으므로 여기서 결과를 보기 전에 1 로 고정. 모집단 표준편차(ddof 0)보다
     √(5/4) ≈ 1.118 배 큼)
  판단은 R6와 같은 k_policy(U_pess, 전환 비용 κ|a−p|ln(1−c)) — 비용 분해 식은 그대로, U 만 비관적 값.
  κ_p = 0 이면 KEnsemble.values 와 비트 단위로 같은 값 (R6로 돌아감).
  두 번째 반환값(봉별 |U| 최댓값)은 R6와 같은 정의 — 멤버·행동별 U 의 |값| 최댓값 (점검은 속 앙상블로 하므로
  여기 값은 기록용일 뿐).

  (walk.py 가 cfg["algo"] == "pessdqn" 이면 이 파일의 monthly_update 를 부름. cfg["output"] 없음 → values(X, cost))
"""
import numpy as np

PESS_KAPPA = 1.0
PESS_DDOF = 1


class PessEnsemble:
    """속 KEnsemble 을 감싸 판단 가치만 비관적으로. 학습·점검용 망은 inner 에 그대로 둠"""

    def __init__(self, inner, kappa=PESS_KAPPA, ddof=PESS_DDOF):
        if isinstance(inner, PessEnsemble):
            inner = inner.inner
        self.inner = inner
        self.kappa = float(kappa)
        self.ddof = int(ddof)
        self.acts = inner.acts
        self.feat_names = inner.feat_names

    def member_values(self, X, cost):
        """(M, B, K) 멤버별 U (KEnsemble.values 와 같은 계산, 평균 전)"""
        K = len(self.inner.acts)
        return self.inner.net.forward(self.inner.inputs(X, cost), cache=False)[..., :K].astype(np.float64)

    def values(self, X, cost):
        """(B, K) 비관적 판단 가치 평균 − κ_p·표준편차(ddof), (B,) 봉별 멤버·행동 |U| 최댓값 (R6와 같은 정의)"""
        U = self.member_values(X, cost)
        mean = U.mean(axis=0)
        if self.kappa != 0.0:
            if U.shape[0] <= self.ddof:
                raise ValueError(f"멤버 {U.shape[0]}개로는 ddof={self.ddof} 표준편차를 계산할 수 없습니다")
            mean = mean - self.kappa * U.std(axis=0, ddof=self.ddof)
        return mean, np.abs(U).max(axis=(0, 2))

    def mean_values(self, X, cost):
        """(B, K) 멤버 평균 U (R6 판단 가치) — 진단용"""
        return self.inner.values(X, cost)[0]

    def state(self):
        st = self.inner.state()
        st["pess_kappa"] = np.array(self.kappa)
        st["pess_ddof"] = np.array(self.ddof)
        return st


def unwrap(ens):
    """포장을 벗겨 속 KEnsemble (없으면 None)"""
    return ens.inner if isinstance(ens, PessEnsemble) else ens


def proposed_cfg():
    """variants.py 에 등록할 설정 (R6_daily_trend8_uniform + P1 키)"""
    from ..variants import v, TREND8
    return v("P1_pessimistic_dqn", algo="pessdqn", stride=6, gamma=0.967, feat=TREND8, recency_frac=0.0,
             pess_kappa=1.0, pess_ddof=1)


def monthly_update(cfg, datas, T_k, seed, ens, anchor):
    """학습·점검은 walk.monthly_update 의 DQN 경로 그대로(속 KEnsemble 로), 결과만 PessEnsemble 로 포장"""
    from .. import walk
    inner, anchor, entry = walk.monthly_update(dict(cfg, algo=None), datas, T_k, seed, unwrap(ens), anchor)
    if inner is None:
        return None, anchor, entry
    return PessEnsemble(inner, cfg.get("pess_kappa", PESS_KAPPA), cfg.get("pess_ddof", PESS_DDOF)), anchor, entry
