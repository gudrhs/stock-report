# -*- coding: utf-8 -*-
"""
Q1 — 분포형 강화학습 (QR-DQN) + 판단은 CVaR 0.5 (비관적 진입).
사전 등록: btc/research/success_criteria.md 마지막 조항 '새 강화학습 방식 4가지 묶음' (커밋 441be61) 의 Q1_qrdqn_cvar.

R6(R6_daily_trend8_uniform)와 같은 것 — 조정하지 않음
  입력 TREND8 + 비용 입력, 보상 'lin'(κ = KAPPA 배), γ 0.967, stride 6 (하루 한 번 판단용 표본),
  균등 추출(recency_frac 0), 5개 멤버를 쌓은 MLP(은닉 32·32), 보조 출력 2개(z6·z42)와 그 손실,
  L2 감쇠, 이어학습의 L2-SP, 걸음 수·학습률·목표망 갱신률 τ·입력 잡음·기울기 클리핑,
  시드 사용법(SeedSequence(seed) → 초기화·표본 두 난수열), 1월(또는 첫 달) 처음부터·나머지 달 이어학습,
  점검(walk.gate_cold / walk.gate_finetune 그대로).
  → 학습 표본 추출(_batch)·표본 풀(make_pool)·학습 일정(step/fit_cold/fit_finetune)은 KTrainer 것을 그대로 물려받습니다.

다른 것
  · U 머리 대신 행동마다 분위수 N = 11개 Z(s, c, a)_i, τ_i = (i + 0.5)/11  (i = 0..10)
    출력 배치: out[..., :K·N] 을 (K, N) 으로 (행동 먼저), 그 뒤에 보조 z6, z42.
  · 분포형 목표 (표본 하나·행동 보강의 출발 행동 a 마다, 비용 분해는 KTrainer.targets 와 똑같이):
        z_target(a)_j = κ·(R(a, m) − R(mid, m)) + γ·[ κ·|a* − a|·ln(1 − c) + Z_tgt(s', a*)_j ]
        a* = argmax_a' [ κ·|a' − a|·ln(1 − c) + 평균_i Z_online(s', a')_i ]
    (이중 DQN 방식: 다음 행동은 온라인 망의 '평균'으로 고르고, 분위수는 목표망에서 가져옴)
  · 손실: 분위수 후버 손실 (κ_huber = 1)
        ρ_τ(u) = |τ − 1{u < 0}| · H(u),  H(u) = ½u² (|u| ≤ 1), |u| − ½ (그 밖),  u_ij = y_j − θ_i
    한 (멤버, 표본, 행동) 에서 분위수 쌍 N×N 개의 '평균', 이를 멤버마다 표본·행동 평균(KTrainer 와 같은 /(K·B))한 뒤
    멤버끼리는 더함 (KTrainer 의 U 손실과 같은 정규화). 보조 손실·L2·L2-SP 는 KTrainer 와 똑같이 더함.
    기울기: ∂ρ/∂θ_i = −|τ_i − 1{u < 0}| · clip(u, −1, 1)   (u = 0 에서는 0)
    주의: |u| ≤ 1 안에서는 제곱 손실이라 고정점이 기대분위수(expectile) 쪽으로 눌림. 분포 폭 ≫ 1 (κ = 100 배 단위,
    하루 ½κm 표준편차 ≈ 1.7, γ 0.967 로 쌓인 분포는 훨씬 넓음) 이면 참 분위수에 가깝고, 폭 ≈ 1 이면 약 30% 좁게 학습됨
    (tests/test_btc_qrdqn.py 2상태 MDP). 사전 등록대로 κ_huber = 1 고정.
  · 판단 가치 (values 가 돌려주는 U): 멤버마다 그 멤버의 분위수 11개를 '정렬'한 뒤
        CVaR_0.5 = ( θ_(0) + θ_(1) + θ_(2) + θ_(3) + θ_(4) + ½·θ_(5) ) / 5.5
    (분위수 i 는 확률 [i/11, (i+1)/11] 구간을 대표 → 하위 50% = 5칸 반. 정렬은 분위수 교차에 대비한 고정 규칙)
    그리고 멤버 5개의 CVaR 를 평균 (KEnsemble 이 멤버 U 를 평균하는 것과 같은 순서).
    정렬 후 하위 절반 평균이라 멤버마다 CVaR_0.5 ≤ 평균이고, 멤버 평균도 마찬가지.
    판단: k_policy(U=CVaR, 전환 비용 κ|a−p|ln(1−c)) — 사전 등록의 '하위 50% 평균에 전환 비용을 더해'.
    점검(게이트)도 이 판단 가치로 한 포지션을 씀. |U| 상한(200) 검사도 멤버별 판단 가치에서.
  · 부트스트랩의 a* 선택은 평균(위험 중립)으로 — 학습하는 분포는 '평균 기준 탐욕 정책'의 수익 분포이고,
    CVaR 는 실행 때 진입 판단에만 씁니다 (사전 등록 그대로).

★ 알려진 한계 (2026-09-25 검토에서 확인, 결과 보기 전) — 기준선 R(mid, m) 때문에 CVaR 가 사실상 평균과 같아짐
  위 목표식(사전 등록 과제에 적힌 그대로)은 KTrainer 처럼 무작위 기준선 R(mid, m) 을 뺍니다.
  평균(DQN)에서는 모든 행동에 같은 값이라 무해하지만, 분포에서는 무해하지 않습니다.
  acts (0, 1), mid ½, 'lin' 이면 현금은 −½κm, 보유는 +½κm 를 배우므로 현금도 보유만큼 '위험'하고
  현금의 하위 꼬리는 BTC 가 오른 날입니다. α = 0.5 이면 즉시 보상 부분에서 정확히
      CVaR_0.5(+½κm) − CVaR_0.5(−½κm) = (하위 절반 평균 + 상위 절반 평균)·½κ = κ·E[m] = 평균 차이
  (분포 모양과 무관, 11분위수 규칙 [1,1,1,1,1,½,0,…]/5.5 에서도 같음). γ > 0 의 부트스트랩 잡음이 있어도
  CVaR 차이는 평균 차이 근처에 머뭅니다. 즉 기본값(qr_base = "mid")의 Q1 은 '위험 중립 QR-DQN + R6 식 판단'에 가깝고,
  사전 등록 문구 '하위 50% 평균 … (비관적 진입)'의 의도를 시험하지 못합니다.
  대안: cfg["qr_base"] = "cash" 이면 기준선으로 현금의 확정 수익 R(0, m) (= 0, 모든 보상 종류) 을 빼서
  '행동의 절대 수익' 분포를 배웁니다. 평균 기준 a* 와 평균 판단은 바뀌지 않고(모든 행동에 같은 이동),
  CVaR 가 무위험 현금 대비 보유의 하방 꼬리를 벌점으로 줍니다.
  코드 기본값은 "mid"(적힌 식)로 두되, 등록 설정은 "cash" 입니다: 2026-09-25 Q1 을 한 번도 돌리기 전에
  success_criteria.md 에 해석 정정('비관적 진입' 의도대로 무위험 현금 기준)을 기록했습니다. "mid" 판은 돌리지 않습니다.
  tests/test_btc_qrdqn.py 의 BaselineEffect 가 두 경우를 모두 확인합니다.

누출 방지: 표본 풀·보상·다음 상태는 KTrainer.make_pool 그대로 (다음 날 시가가 T_k 이전인 표본만).

  (walk.py 가 cfg["algo"] == "qrdqn" 이면 이 파일의 monthly_update 를 부름. cfg["output"] 없음 → values(X, cost))
"""
import numpy as np

from ...env import KAPPA
from ...features import WARMUP
from ...nn import StackedMLP
from ...walkforward import FIRST_TRAIN
from ..rl import KTrainer, KEnsemble, reward

N_QUANT = 11
CVAR_ALPHA = 0.5


def taus(n):
    """분위수 중간점 τ_i = (i + 0.5)/n"""
    return (np.arange(n) + 0.5) / n


def cvar_weights(n, alpha):
    """정렬한 분위수 n개에 곱할 CVaR_α 가중치 (합 1): 분위수 i 는 확률 [i/n, (i+1)/n] 구간을 대표하므로
    하위 α 에 들어가는 비율 clip(α·n − i, 0, 1) 만큼 넣고 α·n 으로 나눔.  n = 11, α = 0.5 → [1,1,1,1,1,½,0,…]/5.5"""
    w = np.clip(alpha * n - np.arange(n), 0.0, 1.0)
    return w / w.sum()


def quantile_huber(theta, y, tau, kh=1.0):
    """
    분위수 후버 손실과 θ 기울기.
      theta: (..., N) 예측 분위수,  y: (..., N') 목표 표본,  tau: (N,)
    반환: 원소별 손실 (..., ) = 쌍 N×N' 평균,  dθ (..., N)  (그 평균에 대한 기울기)
    """
    u = y[..., None, :] - theta[..., :, None]                        # (..., N, N')  u_ij = y_j − θ_i
    au = np.abs(u)
    H = np.where(au <= kh, 0.5 * u * u, kh * (au - 0.5 * kh)) / kh
    wt = np.abs(tau[:, None] - (u < 0))
    npair = u.shape[-1] * u.shape[-2]
    loss = (wt * H).sum(axis=(-1, -2)) / npair
    g = -(wt * np.clip(u, -kh, kh) / kh).sum(axis=-1) / npair        # ∂/∂θ_i
    return loss, g


class QREnsemble(KEnsemble):
    """판단 가치 = 멤버별 CVaR_α(정렬한 분위수) 의 멤버 평균. inputs·state 는 KEnsemble 과 같음"""

    def __init__(self, net, feat_names, acts, n_quant=N_QUANT, alpha=CVAR_ALPHA, base_kind="mid"):
        super().__init__(net, feat_names, acts)
        self.base_kind = base_kind
        self.N = int(n_quant)
        self.alpha = float(alpha)
        self.wc = cvar_weights(self.N, self.alpha)

    def quantiles(self, X, cost):
        """(M, B, K, N) 멤버별 분위수 (정렬 전, 망 출력 그대로)"""
        K = len(self.acts)
        out = self.net.forward(self.inputs(X, cost), cache=False)[..., :K * self.N].astype(np.float64)
        return out.reshape(out.shape[:-1] + (K, self.N))

    def values(self, X, cost):
        """(B, K) 판단 가치(멤버 평균 CVaR_α), (B,) 봉별 멤버·행동 판단 가치 |값| 최댓값"""
        Z = np.sort(self.quantiles(X, cost), axis=-1)
        cv = Z @ self.wc                                                  # (M, B, K)
        return cv.mean(axis=0), np.abs(cv).max(axis=(0, 2))

    def mean_values(self, X, cost):
        """(B, K) 분포 평균(위험 중립 U)의 멤버 평균 — 진단용"""
        return self.quantiles(X, cost).mean(axis=-1).mean(axis=0)

    def state(self):
        st = super().state()
        st["n_quant"] = np.array(self.N)
        st["cvar_alpha"] = np.array(self.alpha)
        st["qr_base"] = np.frombuffer(self.base_kind.encode(), dtype=np.uint8)
        return st


class QRTrainer(KTrainer):
    """KTrainer 에서 U 머리·목표·손실만 분위수판으로 바꿈 (표본·학습 일정·잡음·보조·L2·L2-SP 는 물려받음)"""

    def __init__(self, cfg, seed_seq):
        super().__init__(cfg, seed_seq)
        self.N = int(cfg.get("n_quantiles", N_QUANT))
        self.alpha = float(cfg.get("cvar_alpha", CVAR_ALPHA))
        self.kh = float(cfg.get("huber_k", 1.0))
        self.tau_q = taus(self.N).astype(np.float32)
        self.base_kind = cfg.get("qr_base", "mid")                 # "mid" = 사전 등록 식 그대로, "cash" = 절대 수익
        if self.base_kind not in ("mid", "cash"):
            raise ValueError(f"qr_base {self.base_kind!r}")

    # ── 망 ──
    def _new_net(self):
        c = self.cfg
        return StackedMLP(c["members"], [self.n_in, *c["hidden"], self.K * self.N + 2], self.rng_init, last_scale=0.01)

    def ensemble(self):
        net = StackedMLP.__new__(StackedMLP)
        net.__dict__.update(self.net.__dict__)
        net.params = self.net.copy_params()
        net._cache = None
        return QREnsemble(net, self.feat_names, self.acts, self.N, self.alpha, self.base_kind)

    def _Z(self, out):
        K, N = self.K, self.N
        return out[..., :K * N].reshape(out.shape[:-1] + (K, N))

    # ── 목표·손실 ──
    def targets(self, X1, m, cost):
        """(M, B, K_from, N) 분포형 목표 — KTrainer.targets 의 비용 분해·행동 보강 그대로, 부트스트랩만 분위수"""
        c = self.cfg
        g = c["gamma"]
        A = self.acts
        K, N = self.K, self.N
        lnc = (KAPPA * np.log1p(-cost))[..., None, None]                     # (M,B,1,1)
        pen = np.abs(A[None, :] - A[:, None])[None, None] * lnc               # (M,B,K_from,K_to)
        R = reward(c.get("reward", "lin"), A[None, None, :], m[..., None], c.get("lam", 0.0))
        # 기준선: "mid" 는 KTrainer 와 같은 R(mid, m) (사전 등록 식), "cash" 는 현금의 확정 수익 R(0, m) (= 0)
        a0 = np.float32(self._mid) if self.base_kind == "mid" else np.float32(0.0)
        R0 = reward(c.get("reward", "lin"), a0, m, c.get("lam", 0.0))[..., None]
        base = KAPPA * (R - R0)                                                # (M,B,K)
        if g > 0:
            Zon = self._Z(self.net.forward(X1, cache=False))                  # (M,B,K,N)
            Ztg = self._Z(self.tgt.forward(X1, cache=False))
            q = Zon.mean(axis=-1)[..., None, :] + pen                          # (M,B,from,to)
            if getattr(self, "_b2_next", None) is not None:
                allow = np.where(self._b2_next[..., None] > 0.5, True, (A == 0.0)[None, None, :])
                q = np.where(allow[..., None, :], q, -np.inf)
            astar = q.argmax(axis=-1)                                           # (M,B,K_from)
            pen_sel = np.take_along_axis(pen, astar[..., None], axis=-1)       # (M,B,K_from,1)
            Zsel = np.take_along_axis(Ztg, astar[..., None], axis=2)           # (M,B,K_from,N)
            y = base[..., None] + g * (pen_sel + Zsel)
        else:
            y = np.broadcast_to(base[..., None], base.shape + (N,))
        return np.ascontiguousarray(y, dtype=np.float32)

    def loss_grads(self, X0, y, z6, z42, ok, anchor=None, lam_sp=0.0):
        c = self.cfg
        K, N = self.K, self.N
        hub = lambda d: np.where(np.abs(d) <= 1, 0.5 * d * d, np.abs(d) - 0.5)
        out = self.net.forward(X0, cache=True)
        B = y.shape[1]
        theta = self._Z(out)
        ql, gq = quantile_huber(theta, y, self.tau_q, self.kh)                # (M,B,K), (M,B,K,N)
        a6 = out[..., K * N] - z6
        a42 = out[..., K * N + 1] - z42
        w = c["aux_w"]
        loss = ql.sum() / (K * B) + w * (ok * hub(a6)).sum() / B + w * (ok * hub(a42)).sum() / B
        gU = gq.reshape(gq.shape[:-2] + (K * N,)) / (K * B)
        g6 = w * np.clip(a6, -1.0, 1.0) * ok / B
        g42 = w * np.clip(a42, -1.0, 1.0) * ok / B
        gout = np.concatenate([gU, g6[..., None], g42[..., None]], axis=2).astype(self.net.params[0].dtype)
        grads = self.net.backward(gout)
        for i, p in enumerate(self.net.params):
            if p.ndim == 3 and p.shape[1] > 1:
                grads[i] = grads[i] + 2.0 * c["wd"] * p
                loss += c["wd"] * float((p.astype(np.float64) ** 2).sum())
            if anchor is not None and lam_sp:
                diff = p - anchor[i]
                grads[i] = grads[i] + 2.0 * lam_sp * diff
                loss += lam_sp * float((diff.astype(np.float64) ** 2).sum())
        return float(loss), grads, float(ql.mean())


def proposed_cfg():
    """variants.py 에 등록할 설정 (R6_daily_trend8_uniform + Q1 키). 기준선은 해석 정정대로 현금 (qr_base="cash")"""
    from ..variants import v, TREND8
    return v("Q1_qrdqn_cvar", algo="qrdqn", stride=6, gamma=0.967, feat=TREND8, recency_frac=0.0,
             n_quantiles=11, cvar_alpha=0.5, huber_k=1.0, qr_base="cash")


def monthly_update(cfg, datas, T_k, seed, ens, anchor):
    """walk.monthly_update 의 DQN 경로와 같은 흐름 (처음부터·이어학습·점검·앵커), 트레이너만 QRTrainer"""
    from ..walk import gate_cold, gate_finetune
    Tk = int(T_k.timestamp())
    entry = dict(month=str(T_k.date()))
    tr = QRTrainer(cfg, seed)
    n_pool = tr.make_pool(datas, Tk, FIRST_TRAIN, WARMUP)
    if T_k.month == 1 or ens is None:
        tr.init_fresh()
        losses = tr.fit_cold()
        new = tr.ensemble()
        ok, info = gate_cold(cfg, new, datas[0], Tk)
        entry.update(kind="cold", pool=n_pool, td=float(np.mean(losses[-200:])), accepted=ok, **info)
        if ok:
            ens, anchor = new, new.net.copy_params()
        elif ens is None:
            entry["note"] = "gate_failed_no_model_hold_cash"
    elif cfg["fine_tune"]:
        tr.init_from(ens.net.params)
        losses = tr.fit_finetune(anchor)
        new = tr.ensemble()
        ok, info = gate_finetune(cfg, new, ens, datas[0], Tk)
        entry.update(kind="finetune", td=float(np.mean(losses[-50:])), accepted=ok, **info)
        if ok:
            ens = new
    return ens, anchor, entry
