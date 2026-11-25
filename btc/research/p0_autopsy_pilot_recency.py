# -*- coding: utf-8 -*-
"""
P0 부검 — 작은 예비 실험(pilot) 2: 학습 표본의 '최근 가중'이 Δ 절편(= 늘 보유)을 만드는가?

가설 (부검 뒤 사후 가설): 1월 학습 표본은 70%를 반감기 2년 최근 가중으로 뽑아, 큰 상승장 직후(2018·2022·2025·2026년 1월)
  표본 평균 수익이 크게 양(+)이 되고, 그 평균이 Δ 절편이 되어 '팔 수 없는' 모델이 나옵니다.
  예측: 균등 표본(recency_frac=0)이면 1월 점검 노출도가 낮아지고 게이트 통과가 늘며, 2022·2025~26 노출도가 낮아집니다.
설계: p0_autopsy_pilot_huber 와 같은 frozen 1월 모델 워크포워드, δ=1, 시드 쌍 맞춤 (r ∈ {0, 6, 9}).
사후 연구입니다(잠금 구간 포함).

  python -m btc.research.p0_autopsy_pilot_recency
결과: btc/research/out/p0_autopsy/pilot_recency.json
"""
from btc import config as C
from btc.research.p0_autopsy_pilot_huber import run_pilot


def main():
    run_pilot([(C.make("P0_uniform", recency_frac=0.0), 1.0)], "pilot_recency.json")


if __name__ == "__main__":
    main()
