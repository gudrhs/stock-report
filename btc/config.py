# -*- coding: utf-8 -*-
"""
설정 — P0(사전 등록한 주 전략), 비교용 격자 12개, 절제 실험 3개.

P0는 결과를 보기 전에 고정했습니다. 격자에서 사후에 가장 좋았던 변형을 주 전략으로
올리지 않습니다(그게 바로 '과거에 맞추기'입니다). 격자는 참고용이고, 격자 중에서 고르는
선택기 S는 매년 1월 1일에 '그때까지의' 성적만 보고 고릅니다.
"""
import hashlib
import json

CODE_VERSION = "btc-p0-v1"

P0 = dict(
    name="P0",
    gamma=0.97,              # 할인율 — 약 33봉(5.5일) 앞까지 봄
    features="full22",       # 22개 지표 전부
    c_mult=2.0,              # 판단용 비용 = 실제 비용 × 2 (잦은 매매 억제)
    hidden=(32, 32),
    members=5,               # 앙상블 멤버 수
    batch=256,
    cold_steps=2000,         # 1월 처음부터 학습: 1500스텝 lr 1e-3 + 500스텝 lr 3e-4
    cold_lr1=1e-3, cold_lr2=3e-4, cold_split=1500,
    # ↑ 2014~2016 검증 손실로 {2000, 4000, 8000} 중 한 번 고른 값 (data/btc/runs/budget.json).
    #   손익은 보지 않았고, 이후 이 값을 바꾸지 않습니다.
    ft_steps=300, ft_lr=3e-4,  # 매월 이어서 학습
    lambda_sp=1e-2,          # 이어 학습 때 1월 모델에서 너무 멀어지지 않게 (L2-SP)
    wd=1e-4, tau=0.01, noise=0.1, aux_w=0.25, clip=5.0,
    recency_frac=0.7, half_life_years=2.0,
    phases=16,
    fine_tune=True,
    online=False,
    permute=False,
)

TRAIN_KEYS = ("gamma", "features", "hidden", "members", "batch", "cold_steps", "cold_lr1", "cold_lr2",
              "cold_split", "ft_steps", "ft_lr", "lambda_sp", "wd", "tau", "noise", "aux_w", "clip",
              "recency_frac", "half_life_years", "phases", "fine_tune", "online", "permute")

# 비용 시나리오 (편도, 수수료+슬리피지)
COSTS = {"upbit": 0.0010, "binance": 0.0015, "stress": 0.0025, "bitstamp": 0.0045}
PRIMARY_COST = COSTS["binance"]
BREAKEVEN = (0.0, 0.0005, 0.0010, 0.0015, 0.0025, 0.0045, 0.0060, 0.0100)
C_DEC_MIN, C_DEC_MAX = 0.0005, 0.02


def c_dec(cost, mult):
    return min(C_DEC_MAX, max(C_DEC_MIN, cost * mult))


def all_c_dec():
    """백테스트에서 필요한 판단용 비용 전부 (phase 0에서 미리 계산해 둠)"""
    s = set()
    for c in list(BREAKEVEN) + list(COSTS.values()):
        for m in (1.0, 2.0):
            s.add(round(c_dec(c, m), 6))
    return sorted(s)


def make(name, **over):
    cfg = dict(P0)
    cfg.update(over)
    cfg["name"] = name
    return cfg


def grid():
    """γ × 지표 × 비용배수 = 12개. 학습이 필요한 건 γ×지표 6개 (비용배수는 판단 때만 다름)"""
    out = []
    for g in (0.0, 0.9, 0.97):
        for f in ("full22", "min8"):
            for m in (1.0, 2.0):
                out.append(make(f"g{g:g}_{f}_m{m:g}", gamma=g, features=f, c_mult=m))
    return out


ABLATIONS = {
    "X1": dict(fine_tune=False),     # 1월에만 학습, 월간 이어학습 없음
    "X2": dict(online=True),         # 봉마다 온라인 학습 (보조 트랙)
    "X3": dict(phases=1),            # 15분 밀린 격자 없이 phase 0만
}


def train_hash(cfg):
    d = {k: cfg[k] for k in TRAIN_KEYS}
    d["code"] = CODE_VERSION
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=list).encode()).hexdigest()[:12]


def full_hash(cfg):
    d = {k: cfg[k] for k in TRAIN_KEYS}
    d["c_mult"] = cfg["c_mult"]
    d["code"] = CODE_VERSION
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=list).encode()).hexdigest()[:12]
