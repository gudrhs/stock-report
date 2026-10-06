# -*- coding: utf-8 -*-
"""
W1 — 구간 강건(SoftMin) 직접 정책: R11(비용 차감 로그성장 직접 최대화)에 '가장 나빴던 묶음' 항 하나를 더함.

근거: DeePM(Wood·Roberts·Zohren 2026, arXiv 2601.05975) 5.1절·부록 C — 전체 성적만 최대화하면
'느리게 움직이는 베타'(우리 경우 매수·보유)로 물러나는데, 가장 나쁜 구간들에 가중치를 몰아주는
가상의 적(KL-DRO/EVaR 쌍대형)과 싸우게 하면 조건부 신호를 쓰도록 강제된다는 결과.
(자세한 논의: btc/research/rl_effective_methods.md 의 W1)

사전 등록한 명세 (조정하지 않음)
  정책·학습은 R11과 같음: w = sigmoid(MLP(TREND8)), 5개 멤버, 같은 은닉층, L2 감쇠, L2-SP 이어학습,
  1월 처음부터·나머지 달 이어서, 같은 걸음 수·학습률, cost_train 0.003, 회전율 벌점 0, 입력 잡음 R11과 같음.
  다른 점:
    · 구간 길이 L = 63일(하루 = 6봉), 배치 32개 구간을 G = 8묶음 × K = 4구간으로 나눔
      (구간은 direct.py 표본 풀에서 서로 독립으로, 최근 가중도 설정대로 뽑음 — 묶음은 단순히 연속 4개씩)
    · 하루 순로그성장 g_t = ln(1 + w_t(e^{m_t} − 1)) + ln(1 − c·|w_t − w_{t−1}|)   (구간 첫날은 비용 0)
    · 규모를 가진 점수 S(집합) = 평균(g) / σ_ref · √365
        σ_ref = 학습 풀의 하루 시가→시가 로그수익 m의 표준편차 (그 달 상수, 기울기 없음)
    · P = S(배치 전체 32×63), z_g = S(묶음 g), SoftMin_τ(z) = −τ·ln(평균_g exp(−z_g/τ)),  τ = 0.2
    · λ = min(0.2, 0.5·P_BH / |SoftMin_BH|)  — 매월 재학습 때 한 번, 항상 보유(w=1) 정책을 같은 풀에서
      고정된 큰 표본으로 평가해 계산. P_BH ≤ 0 이거나 SoftMin_BH ≥ 0 이면 λ = 0.2.
      뜻: '항상 현금'(손실 0)이 '항상 보유'보다 목적상 유리해지지 않도록 여유를 둔 상한.
      - P_BH 는 표본 없이 정확히 계산: w=1 이면 P 는 m 의 선형식이라 기댓값 = √365/σ_ref · Σ_j p_j·평균(m_j)
        (p_j = 풀의 최근 가중 뽑기 확률, 평균(m_j) = 구간 j 의 63일 평균 수익)
      - SoftMin_BH 는 비선형이라 1024배치 × 32구간 표본의 배치별 SoftMin 평균. 난수는 반복 씨앗과 무관하게
        SeedSequence((T_k 연, 월)) 에서만 뽑아, 5번의 반복 실행이 같은 달에 같은 λ 를 씀
        (64배치일 때 씨앗 따라 λ 가 0.06~0.2 로 흔들렸던 것을 고침 — 검토 지적)
    · 손실 = −P − λ·SoftMin_τ(z)
      해석적 기울기: ∂L/∂z_g = −λ·softmax(−z/τ)_g,  ∂L/∂P = −1 → g → w(당일 항과 t·t+1의 비용 항) → 로짓 → nn.py
    · L2·L2-SP 기울기는 direct.py와 똑같이 더함.
  실행 비중 = 멤버별 sigmoid(로짓)의 평균 (walk.py 의 _DirectModel과 같음).
  진단: 매월 마지막 50걸음의 적 가중치 softmax(−z/τ) 상위 3개(정렬값)와 최댓값, 묶음 점수 평균을 기록.
    (τ = 0.2 는 묶음 점수 폭 1~4 에 비해 작아 적이 8묶음 중 최악 하나에 ~90% 를 거는 거의 '딱딱한 최솟값'이 됨
     — 명세대로 두고, 결과 해석 때 adv_max 를 함께 보고할 것)

척도에 관한 주의 (검토 지적, 명세는 바꾸지 않음)
  S 에 √365/σ_ref (≈ 480~590) 가 곱해져 손실·기울기가 R11 의 k 배. Adam(클리핑 포함)은 기울기 척도에 거의
  무관하므로, 명세대로 L2·L2-SP 기울기를 척도 없이 더하면 정규화의 상대 세기가 R11 보다 약 k 배 약해짐
  (λ=0 으로 고정한 이어학습 300걸음에서 앵커와의 거리 R11 0.17 대 W1 3.06). 따라서 W1 대 R11 비교는
  'SoftMin 항'과 '약해진 정규화'가 섞인 결과임. 사전 등록 명세를 지키려고 기본값은 그대로 두고,
  cfg["loss_div_k"] = True 로 켜면 손실·기울기를 k 로 나눠(z·τ·λ 규칙은 S 척도 그대로) λ = 0 에서 R11 의
  기울기와 같아지는 대안을 쓸 수 있게만 해 둠 (채택 여부는 조율자 결정).

누출 방지: 표본 풀은 direct.py 의 make_pool 그대로 — 구간 마지막 수익의 끝 시가 봉이 T_k 이전에 마감한
구간만 씁니다. σ_ref와 λ도 같은 풀에서만 계산합니다. 판단은 판단봉 종가 시점의 지표만 씁니다.

  (walk.py 가 cfg["algo"] == "softmin_direct" 이면 이 파일의 monthly_update 를 부름, cfg["output"] == "weights")
"""
import math

import numpy as np
import pandas as pd

from ...features import WARMUP
from ...walkforward import FIRST_TRAIN
from ..direct import DirectTrainer

ANN = math.sqrt(365.0)          # 하루 점수 → 연 환산
N_BH_BATCHES = 1024             # λ 계산용 항상 보유 표본: 1024배치 × 32구간 (SoftMin_BH 용, 고정)
DIAG_LAST = 50                  # 진단은 마지막 50걸음 평균


def softmin(z, tau, axis=-1):
    """SoftMin_τ(z) = −τ·ln(평균 exp(−z/τ))  — 수치 안정형 (최솟값을 빼고 계산)"""
    z = np.asarray(z, dtype=np.float64)
    zmin = z.min(axis=axis, keepdims=True)
    e = np.exp(-(z - zmin) / tau)
    return np.squeeze(zmin, axis=axis) - tau * np.log(e.mean(axis=axis))


def lambda_rule(P_bh, sm_bh, lam_max=0.2):
    """λ = min(λ_max, 0.5·P_BH/|SoftMin_BH|);  P_BH ≤ 0 또는 SoftMin_BH ≥ 0 이면 λ_max"""
    if not (P_bh > 0) or not (sm_bh < 0):
        return float(lam_max)
    return float(min(lam_max, 0.5 * P_bh / abs(sm_bh)))


def softmin_objective(W, m, cost, sigma_ref, lam, tau, G):
    """
    멤버별 손실과 비중에 대한 해석적 기울기.
      W: (M, n, L) 비중,  m: (n, L) 하루 로그수익,  n = G·K (연속 K개 구간이 한 묶음)
    반환: loss (M,), dL/dW (M, n, L), 정보 dict(P, z, q, sm — 멤버별)
    """
    W = np.asarray(W, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    M, n, L = W.shape
    if n % G:
        raise ValueError(f"배치 {n}개가 묶음 {G}개로 나누어지지 않음")
    K = n // G
    k = ANN / sigma_ref                                              # 점수 척도 (상수)
    em = np.expm1(m)[None]
    dW = np.diff(np.concatenate([W[..., :1], W], axis=2), axis=2)    # 첫날은 Δw = 0 → 비용 0
    ad = np.abs(dW)
    g = np.log1p(W * em) + np.log1p(-cost * ad)                      # (M, n, L) 하루 순로그성장
    P = k * g.mean(axis=(1, 2))                                      # (M,)
    z = k * g.reshape(M, G, K * L).mean(axis=2)                      # (M, G) 묶음 점수
    zmin = z.min(axis=1, keepdims=True)
    e = np.exp(-(z - zmin) / tau)
    q = e / e.sum(axis=1, keepdims=True)                             # 적의 가중치 softmax(−z/τ)
    sm = zmin[:, 0] - tau * np.log(e.mean(axis=1))                   # SoftMin_τ(z)
    loss = -P - lam * sm
    # ∂L/∂g_t = ∂L/∂P·∂P/∂g + Σ_g ∂L/∂z_g·∂z_g/∂g = −k/N − λ·q_g(t)·k/(K·L)
    a = -k / (n * L) - lam * np.repeat(q, K, axis=1)[..., None] * (k / (K * L))   # (M, n, 1)
    a = np.broadcast_to(a, W.shape)
    sgn = np.sign(dW)
    g_dW = a * (-cost * sgn / (1.0 - cost * ad))                     # 비용 항: ∂/∂Δw_t
    g_W = a * em / (1.0 + W * em) + g_dW                             # Δw_t = w_t − w_{t−1} 의 w_t 쪽
    g_W[..., :-1] -= g_dW[..., 1:]                                   # w_{t−1} 쪽 (t+1의 비용 항)
    return loss, g_W, dict(P=P, z=z, q=q, sm=sm)


class SoftminTrainer(DirectTrainer):
    """DirectTrainer(R11)를 상속 — 표본 풀·초기화·Adam 일정은 그대로, 손실과 월별 상수(σ_ref, λ)만 다름"""

    def __init__(self, cfg, seed_seq):
        super().__init__(cfg, seed_seq)
        # 부모의 두 난수열(초기화·표본)은 그대로. 항상 보유 평가용 난수열은 make_pool 에서 T_k 로만 만듦
        self.G = int(cfg.get("groups", 8))
        self.tau = float(cfg.get("softmin_tau", 0.2))
        self.lam_max = float(cfg.get("lambda_max", 0.2))
        self.sigma_ref = self.lam = self.P_bh = self.sm_bh = None
        self.diag = []

    # ── 월별 상수 ──
    def make_pool(self, datas, T_k, first_ts, warmup):
        n = super().make_pool(datas, T_k, first_ts, warmup)
        if n == 0:
            raise RuntimeError("학습 표본 풀이 비었습니다")
        # 국면별 하루 수익 r[t] = ln(O[t+S+1]/O[t+1]) — 판단봉 t 의 종가 직후 시가부터 다음 날 같은 시각 시가까지.
        # 풀의 구간이 쓰는 t 는 모두 구간 끝 시가(t0 + S·L + 1)가 T_k 이전이므로 여기서 읽는 시가도 모두 T_k 이전
        S = self.S
        self._r = [np.log(d.o[S + 1:] / d.o[1:-S]) for d in self.datas]
        self.sigma_ref = self._sigma_ref()
        self.P_bh = self._p_bh_exact()
        T = pd.Timestamp(int(T_k), unit="s", tz="UTC")
        rng_bh = np.random.default_rng(np.random.SeedSequence((T.year, T.month)))   # 반복 씨앗과 무관
        self.sm_bh = self._softmin_bh(int(self.cfg.get("bh_batches", N_BH_BATCHES)), rng_bh)
        self.lam = lambda_rule(self.P_bh, self.sm_bh, self.lam_max)
        return n

    def _sigma_ref(self):
        """풀의 구간들이 덮는 모든 하루(판단 봉)의 m = ln(O[t+S+1]/O[t+1]) 표준편차 (가중 없음, 중복 없음)"""
        S, L = self.S, self.L
        vals = []
        for k, d in enumerate(self.datas):
            t0 = self.starts[self.starts[:, 0] == k, 1]
            if not len(t0):
                continue
            cov = np.zeros(d.T, bool)
            for j in range(L):
                cov[t0 + S * j] = True
            vals.append(self._r[k][np.flatnonzero(cov)])
        return float(np.concatenate(vals).std())

    def _seq_mean_m(self):
        """풀의 구간마다 L일 평균 수익 (시작점 순서 그대로) — 보폭 S 별 누적합으로 한 번에"""
        S, L = self.S, self.L
        out = np.empty(len(self.starts))
        for k in range(len(self.datas)):
            sel = self.starts[:, 0] == k
            if not sel.any():
                continue
            r = self._r[k]
            C = np.empty_like(r)
            for rho in range(S):
                C[rho::S] = np.cumsum(r[rho::S])                 # C[t] = r[t] + r[t−S] + r[t−2S] + …
            t0 = self.starts[sel, 1]
            last = C[t0 + S * (L - 1)]
            prev = np.where(t0 >= S, C[np.maximum(t0 - S, 0)], 0.0)
            out[sel] = (last - prev) / L
        return out

    def _p_bh_exact(self):
        """항상 보유의 P 기댓값 (정확): w=1 이면 g = m 이라 P 는 표본 평균의 선형식 → 뽑기 확률로 가중한 구간 평균"""
        p = np.diff(np.concatenate([[0.0], self.cdf]))
        p = p / p.sum()
        return float(ANN / self.sigma_ref * (p * self._seq_mean_m()).sum())

    def _m_seqs(self, n, rng):
        """풀에서 (최근 가중대로) n개 구간을 뽑아 하루 수익 m만 (항상 보유 평가용 — 지표 불필요)"""
        j = np.minimum(np.searchsorted(self.cdf, rng.random(n)), len(self.cdf) - 1)
        S, L = self.S, self.L
        st = self.starts[j]
        m = np.empty((n, L))
        steps = S * np.arange(L)
        for k in range(len(self.datas)):
            sel = st[:, 0] == k
            if sel.any():
                m[sel] = self._r[k][st[sel, 1][:, None] + steps]
        return m

    def _softmin_bh(self, nb, rng):
        """항상 보유(w=1)의 SoftMin — 학습과 같은 배치 모양(32구간 = 8묶음 × 4)으로 nb개 배치를 뽑아 배치별 값의 평균
        (w=1 이면 g = m, 비용 0)"""
        n = int(self.cfg.get("seq_batch", 32))
        m = self._m_seqs(nb * n, rng).reshape(nb, n, self.L)
        z = (ANN / self.sigma_ref) * m.reshape(nb, self.G, -1).mean(axis=2)          # (nb, G)
        return float(softmin(z, self.tau, axis=1).mean())

    # ── 학습 ──
    def loss_grads(self, X, m):
        """한 배치의 멤버별 손실, 파라미터 기울기(L2·L2-SP 포함), 진단 정보"""
        c = self.cfg
        M = c["members"]
        n, L = m.shape
        Xf = X.reshape(n * L, -1)
        zl = self.net.forward(np.broadcast_to(Xf, (M,) + Xf.shape).copy(), cache=True)[..., 0]   # (M, B)
        w = 1.0 / (1.0 + np.exp(-zl))
        loss, g_W, info = softmin_objective(w.reshape(M, n, L), m, c.get("cost_train", 0.003),
                                            self.sigma_ref, self.lam, self.tau, self.G)
        if c.get("loss_div_k", False):
            # 대안(기본 꺼짐, 명세 밖): 손실·기울기를 k = √365/σ_ref 로 나눠 정규화의 상대 세기를 R11 과 맞춤
            kk = ANN / self.sigma_ref
            loss, g_W = loss / kk, g_W / kk
        g_z = (g_W.reshape(M, n * L) * w * (1 - w))[..., None].astype(self.net.dtype)
        grads = self.net.backward(g_z)
        anchor = getattr(self, "anchor", None)
        for i, p in enumerate(self.net.params):
            if p.ndim == 3 and p.shape[1] > 1:
                grads[i] = grads[i] + 2.0 * c["wd"] * p
            if anchor is not None:
                grads[i] = grads[i] + 2.0 * c["lambda_sp"] * (p - anchor[i])
        return loss, grads, info

    def step(self, opt):
        X, m = self._seqs(int(self.cfg.get("seq_batch", 32)))
        loss, grads, info = self.loss_grads(X, m)
        opt.step(grads)
        q = np.sort(info["q"], axis=1)[:, ::-1]
        self.diag.append((q[:, :3].mean(axis=0), float(info["z"].mean()), float(info["P"].mean()),
                          float(info["sm"].mean())))
        return float(loss.mean())

    def month_info(self):
        return dict(sigma_ref=self.sigma_ref, lam=self.lam, P_BH=self.P_bh, SoftMin_BH=self.sm_bh,
                    lam_capped=bool(self.lam >= self.lam_max),
                    bh_batches=int(self.cfg.get("bh_batches", N_BH_BATCHES)),
                    loss_div_k=bool(self.cfg.get("loss_div_k", False)))

    def diag_info(self, last=DIAG_LAST):
        d = self.diag[-last:]
        top3 = np.mean([x[0] for x in d], axis=0)
        return dict(adv_top3=[float(v) for v in top3], adv_max=float(top3[0]),
                    z_mean=float(np.mean([x[1] for x in d])), P=float(np.mean([x[2] for x in d])),
                    softmin=float(np.mean([x[3] for x in d])))


class SoftminModel:
    """워크포워드용 얇은 포장 — 비중 = 멤버별 sigmoid(로짓)의 평균 (_DirectModel과 같음)"""

    def __init__(self, tr):
        from ...nn import StackedMLP
        self.net = StackedMLP.__new__(StackedMLP)
        self.net.__dict__.update(tr.net.__dict__)
        self.net.params = tr.net.copy_params()
        self.net._cache = None
        self.fi = tr.fi

    def weights(self, X):
        z = self.net.forward(X[:, self.fi].astype(np.float32), cache=False)[..., 0]
        return (1.0 / (1.0 + np.exp(-z))).mean(axis=0)

    def state(self):
        return self.net.state("p")


def proposed_cfg():
    """variants.py 에 등록할 설정 (R11_direct_loggrowth + W1 키). 'tau'는 P0의 DQN 목표망 갱신률(0.01)과
    이름이 겹쳐 SoftMin 온도는 'softmin_tau'로 둡니다."""
    from ..variants import v, TREND8
    return v("W1_softmin_direct", algo="softmin_direct", output="weights", stride=6, feat=TREND8, obj="log",
             cost_train=0.003, turnover_pen=0.0, seq_len=63, seq_batch=32, groups=8, softmin_tau=0.2,
             lambda_max=0.2)


def monthly_update(cfg, datas, T_k, seed, ens, anchor):
    """1월(또는 첫 달)은 처음부터, 나머지 달은 지난달 모델에서 L2-SP 이어학습 — R11(monthly_update_direct)과 같은 흐름"""
    Tk = int(T_k.timestamp())
    entry = dict(month=str(T_k.date()))
    tr = SoftminTrainer(cfg, seed)
    n = tr.make_pool(datas, Tk, FIRST_TRAIN, WARMUP)
    entry.update(pool=n, **tr.month_info())
    if T_k.month == 1 or ens is None:
        tr.init_fresh()
        losses = tr.fit_cold()
        entry.update(kind="cold", loss=float(np.mean(losses[-200:])), accepted=True, **tr.diag_info())
        new = SoftminModel(tr)
        return new, new.net.copy_params(), entry
    tr.init_from(ens.net.params)
    losses = tr.fit_finetune(anchor)
    entry.update(kind="finetune", loss=float(np.mean(losses[-50:])), accepted=True, **tr.diag_info())
    return SoftminModel(tr), anchor, entry
