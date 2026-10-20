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

  주의 — 등록된 P1 은 '매수 쪽으로' 거의 비관적이지 않음 (검토에서 확인, 사양은 바꾸지 않음)
  · rl.targets 는 U 를 가운데 행동(mid = acts 평균 = 0.5) 기준으로 잽니다: base = κ(R(a,m) − R(mid,m)).
    그래서 U(s,0) 에는 −0.5m, U(s,1) 에는 +0.5m 이 들어가고, 멤버 간 불일치(sd)는 두 행동에서 거의 같습니다.
  · 판단은 U(s,1) − U(s,0) 과 전환 비용에만 달렸으므로, 행동마다 따로 빼는 κ_p·sd 는 대부분 상쇄됩니다:
      U_pess(s,1) − U_pess(s,0) = [평균 U1 − 평균 U0] − κ_p · [sd U1 − sd U0]
    (합성 점검: 평균 sd(U0) 0.154 대 sd(U1) 0.147 → 벌점이 오히려 '유지' 쪽으로 기운 봉이 57%, R6와 판단이
     다른 봉 0.5%, 노출도 0.644 대 R6 0.641. 초기 24개월 달력(작은 학습)에서도 점검 방식마다 0.5~0.6% 만 뒤집힘)
  · 사전 등록 문구("U에서 평균 − 1.0×표준편차")를 글자 그대로 구현한 것이며, 결과를 보기 전 등록이라
    여기서 방향성 있게 고치면 등록 후 조정(튜닝)이 됩니다 → 고치지 않습니다.
  · 보고서에는: P1 은 행동별 U 에 따로 벌점을 주므로 R6와 거의 같은 결과가 예상되고, 차이가 없다는 결과를
    '비관주의 일반'에 대한 반증으로 읽지 말 것. 방향성 비관(예: 매수 쪽에만 sd_m(U_m(s,1) − U_m(s,0)) 벌점, 또는
    U − U(s,mid) 에 평균 − κ·sd)이 필요하면 결과를 보기 전에 별도 시도로 새로 등록하고 N 에 셉니다.
  · 이를 보고서에서 수치로 적을 수 있게 진단용 member_sd(행동별 sd)와 decision_tilt(sd U1 − sd U0) 를 둡니다
    (판단에는 쓰지 않음).

★ 해석 정정 (2026-09-25, 결과 보기 전 — success_criteria.md 같은 날짜 조항): 벌점은 '현금 기준'으로 잽니다
  위 주의대로 글자 그대로의 P1(ref="action")은 R6와 사실상 같으므로 돌리지 않습니다.
  cfg["pess_ref"] = "cash" 이면 멤버마다 현금 대비 이득 D_m(a) = U_m(a) − U_m(현금) 을 만들고
      U_pess(a) = 평균_m U_m(a) − κ_p · sd_m D_m(a)          (현금 행동은 D ≡ 0 → 벌점 0)
  즉 '평균 − 1.0×표준편차' 를 무위험 현금 기준으로 잰 값에 적용합니다 (Q1 의 qr_base="cash" 와 같은 기준).
  판단 차이 U_pess(1) − U_pess(0) = [평균 U1 − 평균 U0] − κ_p·sd_m(U1 − U0) — 멤버들이 매수 이득에 대해
  엇갈릴수록 매수를 덜 합니다. 학습·점검은 그대로 R6.

  (walk.py 가 cfg["algo"] == "pessdqn" 이면 이 파일의 monthly_update 를 부름. cfg["output"] 없음 → values(X, cost))
"""
import numpy as np

PESS_KAPPA = 1.0
PESS_DDOF = 1


class PessEnsemble:
    """속 KEnsemble 을 감싸 판단 가치만 비관적으로. 학습·점검용 망은 inner 에 그대로 둠"""

    def __init__(self, inner, kappa=PESS_KAPPA, ddof=PESS_DDOF, ref="action"):
        if isinstance(inner, PessEnsemble):
            inner = inner.inner
        self.inner = inner
        self.kappa = float(kappa)
        self.ddof = int(ddof)
        if ref not in ("action", "cash"):
            raise ValueError(f"pess_ref {ref!r}")
        self.ref = ref                                   # "action" = 행동별 U 의 sd (글자 그대로), "cash" = U(a) − U(현금) 의 sd
        if ref == "cash":
            hit = np.flatnonzero(np.asarray(inner.acts, dtype=float) == 0.0)
            if len(hit) != 1:
                raise ValueError("pess_ref='cash' 에는 행동 0.0(현금)이 정확히 하나 있어야 합니다")
            self.i_cash = int(hit[0])
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
            mean = mean - self.kappa * self._sd(U)
        return mean, np.abs(U).max(axis=(0, 2))

    def _sd(self, U):
        """(M,B,K) 멤버 U → (B,K) 벌점에 쓰는 표준편차 (ref 에 따라 행동별 U 또는 현금 대비 이득)"""
        if self.ref == "cash":
            U = U - U[..., self.i_cash:self.i_cash + 1]
        return U.std(axis=0, ddof=self.ddof)

    def member_sd(self, X, cost):
        """(B, K) 벌점에 쓰는 멤버 표준편차(ddof, ref 기준) — 진단용, 판단에는 values 만 씀"""
        return self._sd(self.member_values(X, cost))

    def decision_tilt(self, X, cost):
        """(B,) 벌점이 판단 차 U(s,마지막 행동) − U(s,첫 행동) 에 주는 몫 = −κ_p·(sd U_last − sd U_first).
        음수면 벌점이 현금 쪽, 양수면 매수 쪽으로 기움 (행동별 벌점이 대부분 상쇄됨을 보고서에 적기 위한 진단)"""
        sd = self.member_sd(X, cost)
        return -self.kappa * (sd[:, -1] - sd[:, 0])

    def mean_values(self, X, cost):
        """(B, K) 멤버 평균 U (R6 판단 가치) — 진단용"""
        return self.inner.values(X, cost)[0]

    def state(self):
        st = self.inner.state()
        st["pess_kappa"] = np.array(self.kappa)
        st["pess_ddof"] = np.array(self.ddof)
        st["pess_ref"] = np.frombuffer(self.ref.encode(), dtype=np.uint8)
        return st


def unwrap(ens):
    """포장을 벗겨 속 KEnsemble (없으면 None)"""
    return ens.inner if isinstance(ens, PessEnsemble) else ens


def proposed_cfg():
    """variants.py 에 등록할 설정 (R6_daily_trend8_uniform + P1 키, 해석 정정대로 현금 기준 벌점)"""
    from ..variants import v, TREND8
    return v("P1_pessimistic_dqn", algo="pessdqn", stride=6, gamma=0.967, feat=TREND8, recency_frac=0.0,
             pess_kappa=1.0, pess_ddof=1, pess_ref="cash")


def monthly_update(cfg, datas, T_k, seed, ens, anchor):
    """학습·점검은 walk.monthly_update 의 DQN 경로 그대로(속 KEnsemble 로), 결과만 PessEnsemble 로 포장"""
    from .. import walk
    inner, anchor, entry = walk.monthly_update(dict(cfg, algo=None), datas, T_k, seed, unwrap(ens), anchor)
    if inner is None:
        return None, anchor, entry
    return PessEnsemble(inner, cfg.get("pess_kappa", PESS_KAPPA), cfg.get("pess_ddof", PESS_DDOF),
                        cfg.get("pess_ref", "action")), anchor, entry
