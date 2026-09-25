# -*- coding: utf-8 -*-
"""
연구 변형 목록 — 사후 탐색(post-hoc). 하나씩 바꿔 효과를 분리해 보는 사다리 구조입니다.
모두 data/btc/research_runs/trials.jsonl 에 기록되어 DSR의 시험 수 N에 더해집니다.

근거 (btc/research/predictability.md, p0_autopsy.md, literature.md)
  · 4시간 신호(되돌림)는 비용보다 작음 → 하루 한 번 판단
  · 쓸 만한 예측력은 1주~3달 추세 → 약 1달 앞을 보는 할인율 (일 단위 γ≈0.967)
  · 입력은 느린 추세 지표 + 변동성 하나로 줄임
  · 변동성은 방향보다 훨씬 예측 가능 → 비율 보유 + 로그 성장 보상이어야 활용 가능
"""
from .. import config as C

TREND8 = ["ret_42", "ret_180", "ma_50", "ma_200", "rsi_84", "macdh_1d", "dd_180", "vol_regime"]


def v(name, **kw):
    cfg = dict(C.P0)
    cfg.update(acts=(0.0, 1.0), reward="lin", lam=0.0, stride=1, gate="p0", feat=None)
    cfg.update(kw)
    cfg["name"] = name
    return cfg


VARIANTS = {c["name"]: c for c in [
    # 0. 기준: P0를 연구 코드로 다시 (연구 코드가 P0를 재현하는지 확인용 — 시험으로 세지 않음)
    v("R0_p0_repro"),
    # 1. 하루 한 번 판단, 약 1달 앞 (γ=0.967/일)
    v("R1_daily", stride=6, gamma=0.967),
    # 2. + 느린 추세 지표 8개만
    v("R2_daily_trend8", stride=6, gamma=0.967, feat=TREND8),
    # 3. + 비율 보유(0·25·50·75·100%) + 로그 성장 보상
    v("R3_daily_trend8_log5", stride=6, gamma=0.967, feat=TREND8,
      acts=(0.0, 0.25, 0.5, 0.75, 1.0), reward="log"),
    # 4. R2 + 평균-분산 보상 (위험회피 λ=2)
    v("R4_daily_trend8_mv2", stride=6, gamma=0.967, feat=TREND8, reward="mv", lam=2.0),
    # 5. R2 + 점검에서 노출도 상한 제거 (2025·26 1월 재학습이 거부된 원인)
    v("R5_daily_trend8_gateSw", stride=6, gamma=0.967, feat=TREND8, gate="switch"),
    # 6. R2 + 최근 가중 없이 균등 추출 (상승장 편향 완화)
    v("R6_daily_trend8_uniform", stride=6, gamma=0.967, feat=TREND8, recency_frac=0.0),
    # ── 문헌 검토에서 나온 것 (btc/research/rl_literature_review.md) ──
    # 7·8. 드리프트 제거: '오른다'는 사전 믿음을 빼고 시점 선택만 배우게 (α=0 전부 제거, 0.5 절반)
    v("R7_daily_trend8_drift0", stride=6, gamma=0.967, feat=TREND8, drift_alpha=0.0),
    v("R8_daily_trend8_drift05", stride=6, gamma=0.967, feat=TREND8, drift_alpha=0.5),
    # 9. 하락만 벌점 (R4의 전체 분산 벌점과 대조)
    v("R9_daily_trend8_down2", stride=6, gamma=0.967, feat=TREND8, reward="down", lam=2.0),
    # 10. 거부권 방식: 200일선 규칙이 보유일 때만 보유 가능, 에이전트는 그걸 거부만 할 수 있음
    v("R10_daily_veto_b2", stride=6, gamma=0.967, feat=TREND8 + ["ma_1200"], veto_b2=True),
    # 11. 비교용(강화학습 아님): 비용 차감 로그성장을 직접 최대화하는 정책
    v("R11_direct_loggrowth", algo="direct", stride=6, feat=TREND8, obj="log", cost_train=0.003,
      turnover_pen=0.0, seq_len=60, seq_batch=32),
    # ── 롱숏 (사후 추가, 사용자 요청) — 숏은 로그성장 보상으로 정확히 회계(변동성 손실 포함) ──
    # 행동 순서의 첫 번째(0.0)가 시작 포지션(현금)입니다. 노출도 상한 점검은 롱숏에 맞지 않아 'switch' 점검.
    v("L1_daily_trend8_ls3", stride=6, gamma=0.967, feat=TREND8,
      acts=(0.0, -1.0, 1.0), reward="log", gate="switch"),
    v("L2_daily_trend8_ls5", stride=6, gamma=0.967, feat=TREND8,
      acts=(0.0, -1.0, -0.5, 0.5, 1.0), reward="log", gate="switch"),
    v("L3_daily_trend8_ls3_drift0", stride=6, gamma=0.967, feat=TREND8,
      acts=(0.0, -1.0, 1.0), reward="log", gate="switch", drift_alpha=0.0),
    # ── 1단계 선별 통과 요소 결합 (사후, 사용자 요청 "매수·보유를 이길 때까지") ──
    # 12. R6(균등 추출) + R3(비율 보유 0~100%·로그성장 보상)
    v("R12_daily_trend8_uniform_log5", stride=6, gamma=0.967, feat=TREND8, recency_frac=0.0,
      acts=(0.0, 0.25, 0.5, 0.75, 1.0), reward="log"),
    # ── 인터넷 조사로 고른 기법 (btc/research/rl_effective_methods.md; 사후, 시험 수에 포함) ──
    # W1. R11 직접 정책 + DeePM식 최악 구간 가중(SoftMin, τ=0.2, λ≤0.2). 검토 지적으로 손실을 척도 k로 나눠
    #     λ=0에서 R11과 정확히 같게(loss_div_k) — 실행 전에 확정.
    v("W1_softmin_direct", algo="softmin_direct", output="weights", stride=6, feat=TREND8, obj="log",
      cost_train=0.003, turnover_pen=0.0, seq_len=63, seq_batch=32, groups=8, softmin_tau=0.2,
      lambda_max=0.2, loss_div_k=True),
    # W2. 능형회귀 추세 예측 + 정확한 동적계획 무거래 띠 (결정론적 진단; 같은 예측의 부호 규칙은 후보 아님)
    v("W2_dp_band", algo="dp_band", output="weights", stride=6, feat=TREND8, dp_horizon_days=20, dp_ridge_alpha=1.0,
      dp_mu0_frac=0.5, dp_grid_n=201, dp_grid_sd=4.0, dp_cost=0.0015, dp_gamma=0.998, dp_tol=1e-12),
]}
