# -*- coding: utf-8 -*-
"""
A1_actor_critic — 정책경사 액터-크리틱(A2C, 온-정책). btc/research/success_criteria.md 마지막 절
'새 강화학습 방식 4가지 묶음'(결과 보기 전 등록, 커밋 441be61)의 A1 행을 그대로 구현합니다. 값을 조정하지 않습니다.

사전 등록 명세 (조정하지 않음)
  · 상태 s_t = [TREND8 8개(판단봉 종가 시점), 직전 포지션 w_{t−1} ∈ [0, 1]]  → 입력 9개 (w_{t−1}은 그대로, 표준화 없음)
  · 행동: 목표 비중 {0, 0.25, 0.5, 0.75, 1.0} 위의 소프트맥스 정책 π(a | s)
  · 비평가: V(s) = V(TREND8, w_{t−1})
  · 하루 보상 g_t = ln(1 + a_t·(e^{m_t} − 1)) + ln(1 − c·|a_t − w_{t−1}|),  c = cost_train = 0.003 (R11과 같음)
      m_t = 판단봉 t 다음 봉 시가부터 하루(6봉) 뒤 시가까지의 로그수익 = ln(O[t+7] / O[t+1])  (stride 6)
  · γ = 0.967(하루), GAE λ = 0.95, 엔트로피 계수 0.01, 가치 손실 계수 0.5, 배치마다 이점(advantage) 표준화
  · 에피소드: 60일 구간. 표본 풀·구간 뽑기는 direct.py DirectTrainer 그대로(16격자 × 6시작, 구간 끝 시가 봉이
    T_k 전에 마감한 구간만) — 단 recency_frac = 0 이라 균등 추출(R6와 같음). 한 걸음에 32구간(seq_batch, direct와 같음).
    구간마다 첫 w_{t−1}은 5개 행동에서 균등하게 뽑음. 정책을 확률적으로 굴려(행동 표본) w_{t−1} = 직전 표본 행동,
    구간 끝에서 V(s_60)로 부트스트랩 (s_60 = 61번째 날의 지표 + 마지막 행동; 그 봉의 종가도 T_k 이전 — 누출 없음).
  · 5개 멤버를 쌓은 MLP(nn.StackedMLP), 은닉층 P0 (32, 32), Adam·학습률 일정 P0
    (1월 처음부터 2000걸음: 1500걸음 lr 1e-3 + 500걸음 lr 3e-4 / 나머지 달 300걸음 lr 3e-4, 1월 모델로 L2-SP λ 1e-2),
    L2 감쇠 1e-4(가중치 행렬만, direct.py와 같음), 입력 잡음 0.1(P0), 멤버별 기울기 노름 클리핑 5(P0).

구현상 선택 (명세가 열어 둔 것 — 여기서 고정)
  · 액터와 비평가는 **몸통을 공유**합니다: 멤버마다 MLP 하나 [9 → 32 → 32 → 6], 출력 = 로짓 5개 + V 1개.
    (그래서 가치 손실 계수 0.5가 몸통에서 정책 기울기와의 상대 세기로 실제 의미를 가짐.) 마지막 층 초기 크기 0.01
    (KTrainer와 같음 → 처음 정책은 거의 균등, V ≈ 0).
  · 멤버별 손실 (N = 32구간 × 60일):
        L = −(1/N)Σ log π(a_t|s_t)·Â_t + 0.5·(1/N)Σ (V(s_t) − R_t)² − 0.01·(1/N)Σ H(π(·|s_t)) + L2 + L2-SP
    Â = GAE 이점을 멤버별 배치(N개)의 평균·표준편차로 표준화한 값, R_t = GAE 이점(표준화 전) + V(s_t) (λ-수익).
    Â·R은 굴린 시점의 값으로 고정(기울기 없음). 가치 손실은 제곱오차(후버 아님).
  · 멤버들은 같은 32구간(같은 잡음)을 보지만 첫 w_{t−1}·행동 표본은 멤버마다 따로 뽑습니다 (초기값도 다름).
  · 첫날도 비용을 셉니다: w_{t−1}이 상태의 일부이므로 g_0 = … + ln(1 − c·|a_0 − w_{−1}|). (direct.py는 첫날 비용 0)
  · 점검(gate)은 없습니다 — 등록 명세에 없고, 같은 직접 정책 계열(R11·W1)처럼 매달 항상 채택합니다.

실행 (상태가 있는 판단 — dp_band.py와 같은 방식)
  · 판단봉마다 멤버 m의 기대 비중 E_m = Σ_a π_m(a | s, w_prev)·a, 멤버 평균 Ē, 실행 비중 w = round(4·Ē)/4
    (np.round — screen.py가 다시 반올림해도 그대로). 다음 판단의 w_prev = 이 실행 비중(25% 격자 위).
  · walk.run_replication은 한 달치 phase 0 판단봉을 시간 순서대로 한 번에 weights(X)로 넘깁니다.
    weights(X)는 행을 순서대로 처리합니다. 시작 w_prev = pos0 = 지난달 모델의 마지막 실행 비중(ens.last_pos),
    첫 달은 0(현금). 매 호출 pos0에서 다시 시작(멱등)하고, 끝나면 last_pos를 마지막 비중으로 둡니다.
    지표가 NaN인 행은 판단을 보류하고 이전 비중을 유지합니다.
  · w_prev는 늘 5개 격자값 중 하나이므로 (행, w_prev) 5가지를 한 번에 순전파해 표로 만든 뒤 순서대로 조회합니다
    (행마다 순전파하는 것과 비트 단위로 같음 — 테스트).
  주의: 행이 시간 순서가 아니거나 여러 달을 섞어 넘기면 틀립니다. run_replication(all_bars=True)처럼 4시간봉을 모두
  넘기면 4시간마다 이어지는 다른 경로가 됩니다. 강제 유지(forced_hold)로 실제 체결이 목표와 달라져도 내부 w_prev는
  목표 기준으로 이어집니다 (dp_band와 같음). 학습은 확률적 정책, 실행은 기대 비중 반올림 — 명세대로의 차이입니다.

기록 (월별 entry): kind, pool, pos0, loss(마지막 200/50걸음 평균), pg·vf·ent(엔트로피)·reward(하루 평균 g)·
  exposure(굴린 행동 평균)·turnover(하루 평균 |Δw|)·v_mean·adv_sd(표준화 전) — 모두 마지막 걸음들 평균.

  (walk.py 가 cfg["algo"] == "a2c" 이면 이 파일의 monthly_update 를 부름, cfg["output"] == "weights")
"""
import numpy as np

from ...features import WARMUP
from ...nn import StackedMLP
from ...walkforward import FIRST_TRAIN
from ..direct import DirectTrainer

ACTS = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULTS = dict(            # 사전 등록값 — 조정하지 않음
    acts=ACTS,
    cost_train=0.003,
    gamma=0.967,
    gae_lambda=0.95,
    ent_coef=0.01,
    vf_coef=0.5,
    adv_norm=True,
    seq_len=60,
    seq_batch=32,
)
DIAG_COLD, DIAG_FT = 200, 50


def hparams(cfg):
    return {k: cfg.get(k, v) for k, v in DEFAULTS.items()}


# ══════════ 순수 함수 (테스트 대상) ══════════
def log_softmax(z):
    z = np.asarray(z)
    zm = z.max(axis=-1, keepdims=True)
    return z - zm - np.log(np.exp(z - zm).sum(axis=-1, keepdims=True))


def step_reward(a, w_prev, m, cost):
    """하루 순로그성장 g = ln(1 + a(e^m − 1)) + ln(1 − c|a − w_prev|)"""
    return np.log1p(a * np.expm1(m)) + np.log1p(-cost * np.abs(a - w_prev))


def sample_actions(pi, u):
    """확률 pi (..., K)에서 균등난수 u (...)로 행동 번호 — 역CDF (u < cdf 인 첫 칸)"""
    cdf = np.cumsum(pi, axis=-1)
    return np.minimum((u[..., None] >= cdf).sum(axis=-1), pi.shape[-1] - 1)


def gae(r, v, v_boot, gamma, lam):
    """
    일반화 이점 추정 (마지막 축이 시간). r, v: (..., L), v_boot: (...) = V(s_L).
      δ_t = r_t + γ·V(s_{t+1}) − V(s_t),   Â_t = Σ_k (γλ)^k δ_{t+k}  (구간 끝에서 자름, 끝은 V(s_L)로 부트스트랩)
    반환 (이점, λ-수익 = 이점 + V)
    """
    r = np.asarray(r, np.float64)
    v = np.asarray(v, np.float64)
    L = r.shape[-1]
    vn = np.concatenate([v[..., 1:], np.asarray(v_boot, np.float64)[..., None]], axis=-1)
    delta = r + gamma * vn - v
    adv = np.empty_like(delta)
    acc = np.zeros(delta.shape[:-1])
    for t in range(L - 1, -1, -1):
        acc = delta[..., t] + gamma * lam * acc
        adv[..., t] = acc
    return adv, adv + v


def normalize_adv(adv, axis):
    """배치(멤버별) 평균 0·표준편차 1로"""
    mu = adv.mean(axis=axis, keepdims=True)
    sd = adv.std(axis=axis, keepdims=True)
    return (adv - mu) / (sd + 1e-8)


def logit_grads(z, a_idx, adv, ent_coef, N, pg_coef=1.0):
    """
    정책 손실 −(1/N)Σ log π_a·Â − ent_coef·(1/N)Σ H 의 값과 로짓 기울기.
      ∂(−log π_a)/∂z_k = π_k − 1[k=a],   ∂H/∂z_k = −π_k(log π_k + H)
    z: (..., K), a_idx: (...), adv: (...). 반환 (정책경사 항 합 (앞축들 중 마지막 축 제외 모양의 합), 엔트로피 합, dL/dz)
    """
    lp = log_softmax(np.asarray(z, np.float64))
    pi = np.exp(lp)
    H = -(pi * lp).sum(axis=-1)
    onehot = np.zeros_like(pi)
    np.put_along_axis(onehot, np.asarray(a_idx)[..., None], 1.0, axis=-1)
    lpa = np.take_along_axis(lp, np.asarray(a_idx)[..., None], axis=-1)[..., 0]
    g = pg_coef * (-np.asarray(adv, np.float64)[..., None]) * (onehot - pi) + ent_coef * pi * (lp + H[..., None])
    return -(lpa * adv), H, g / N


# ══════════ 학습기 ══════════
class A2CTrainer(DirectTrainer):
    """DirectTrainer(R11)의 표본 풀·구간 뽑기·Adam 일정(fit_cold/fit_finetune)을 그대로 쓰고, 망·손실·굴리기만 다름"""

    def __init__(self, cfg, seed_seq):
        super().__init__(cfg, seed_seq)
        hp = hparams(cfg)
        self.hp = hp
        self.acts = np.asarray(hp["acts"], np.float32)
        self.K = len(self.acts)
        self.n_in = len(self.fi) + 1
        # 행동 표본·첫 w_prev 전용 난수열 (부모의 초기화·구간 뽑기 난수열과 분리)
        self.rng_act = np.random.default_rng(np.random.SeedSequence(seed_seq).spawn(3)[2])
        self.diag = []

    def init_fresh(self):
        c = self.cfg
        self.net = StackedMLP(c["members"], [self.n_in, *c["hidden"], self.K + 1], self.rng_init, last_scale=0.01)

    # ── 표본: 60일 구간 + 부트스트랩용 61번째 날 지표 ──
    def _seqs(self, n):
        j = np.minimum(np.searchsorted(self.cdf, self.rng.random(n)), len(self.cdf) - 1)
        S, L = self.S, self.L
        X = np.empty((n, L + 1, len(self.fi)), np.float32)
        m = np.empty((n, L), np.float32)
        for i, (k, t) in enumerate(self.starts[j]):
            d = self.datas[k]
            idx = t + S * np.arange(L + 1)                     # 마지막 idx = t + S·L: 종가 ≤ 시가 t+S·L+1 ≤ T_k
            X[i] = d.X[idx][:, self.fi]
            m[i] = np.log(d.o[idx[:-1] + S + 1] / d.o[idx[:-1] + 1])
        if self.cfg["noise"]:
            X += (self.cfg["noise"] * self.rng.standard_normal(X.shape)).astype(np.float32)
        return X, m

    def _inp(self, x, w_prev):
        """x: (n, F) 지표, w_prev: (M, n) → (M, n, F+1)"""
        M = w_prev.shape[0]
        out = np.empty((M,) + x.shape[:-1] + (x.shape[-1] + 1,), self.net.dtype)
        out[..., :-1] = x
        out[..., -1] = w_prev
        return out

    def rollout(self, X, m):
        """
        확률적 정책으로 구간을 굴림. X: (n, L+1, F), m: (n, L).
        반환 dict: S (M, n, L, F+1) 상태, a (M, n, L) 행동 번호, r (M, n, L) 보상, v (M, n, L) V(s_t), v_boot (M, n) V(s_L)
        """
        M = self.cfg["members"]
        n, L = m.shape
        K, A = self.K, self.acts
        cost = self.hp["cost_train"]
        S = np.empty((M, n, L, self.n_in), self.net.dtype)
        a = np.empty((M, n, L), np.int64)
        r = np.empty((M, n, L), np.float64)
        v = np.empty((M, n, L), np.float64)
        w = A[self.rng_act.integers(0, K, size=(M, n))].astype(np.float64)       # 첫 w_prev: 5개 행동에서 균등
        for t in range(L):
            inp = self._inp(X[:, t], w)
            out = self.net.forward(inp, cache=False)
            pi = np.exp(log_softmax(out[..., :K].astype(np.float64)))
            at = sample_actions(pi, self.rng_act.random((M, n)))
            wt = A[at].astype(np.float64)
            S[:, :, t] = inp
            a[:, :, t] = at
            v[:, :, t] = out[..., K]
            r[:, :, t] = step_reward(wt, w, m[None, :, t].astype(np.float64), cost)
            w = wt
        v_boot = self.net.forward(self._inp(X[:, L], w), cache=False)[..., K].astype(np.float64)
        return dict(S=S, a=a, r=r, v=v, v_boot=v_boot)

    def advantages(self, ro):
        hp = self.hp
        adv, ret = gae(ro["r"], ro["v"], ro["v_boot"], self.cfg["gamma"], hp["gae_lambda"])
        M = adv.shape[0]
        advn = normalize_adv(adv.reshape(M, -1), axis=1).reshape(adv.shape) if hp["adv_norm"] else adv
        return adv, advn, ret

    # ── 손실·기울기 ──
    def loss_grads(self, S, a, adv, ret, coefs=None):
        """
        멤버별 손실 (M,)과 파라미터 기울기(L2·L2-SP 포함). S: (M, …, F+1), a·adv·ret: (M, …) — adv는 표준화된 값.
        coefs = (정책경사 계수, 가치 계수, 엔트로피 계수) — 기본 (1, vf_coef, ent_coef). 테스트에서 항별 검사용.
        """
        c, hp = self.cfg, self.hp
        pg_c, vf_c, ent_c = coefs if coefs is not None else (1.0, hp["vf_coef"], hp["ent_coef"])
        M = S.shape[0]
        K = self.K
        Sf = S.reshape(M, -1, S.shape[-1])
        af, advf, retf = a.reshape(M, -1), np.asarray(adv).reshape(M, -1), np.asarray(ret).reshape(M, -1)
        N = Sf.shape[1]
        out = self.net.forward(Sf, cache=True)
        z = out[..., :K].astype(np.float64)
        V = out[..., K].astype(np.float64)
        pgl, H, gz = logit_grads(z, af, advf, ent_c, N, pg_c)
        dv = V - retf
        pg = pgl.mean(axis=1)
        vf = (dv ** 2).mean(axis=1)
        ent = H.mean(axis=1)
        loss = pg_c * pg + vf_c * vf - ent_c * ent
        gout = np.concatenate([gz, (2.0 * vf_c * dv / N)[..., None]], axis=2).astype(self.net.dtype)
        grads = self.net.backward(gout)
        anchor = getattr(self, "anchor", None)
        loss = loss.astype(np.float64)
        for i, p in enumerate(self.net.params):
            if p.ndim == 3 and p.shape[1] > 1:
                grads[i] = grads[i] + 2.0 * c["wd"] * p
                loss += c["wd"] * (p.astype(np.float64) ** 2).reshape(M, -1).sum(axis=1)
            if anchor is not None:
                diff = p - anchor[i]
                grads[i] = grads[i] + 2.0 * c["lambda_sp"] * diff
                loss += c["lambda_sp"] * (diff.astype(np.float64) ** 2).reshape(M, -1).sum(axis=1)
        return loss, grads, dict(pg=pg, vf=vf, ent=ent)

    def step(self, opt):
        X, m = self._seqs(int(self.hp["seq_batch"]))
        ro = self.rollout(X, m)
        adv, advn, ret = self.advantages(ro)
        loss, grads, info = self.loss_grads(ro["S"], ro["a"], advn, ret)
        opt.step(grads)
        A = self.acts[ro["a"]]
        self.diag.append(dict(pg=float(info["pg"].mean()), vf=float(info["vf"].mean()), ent=float(info["ent"].mean()),
                              reward=float(ro["r"].mean()), exposure=float(A.mean()),
                              turnover=float(np.abs(np.diff(A, axis=2)).mean()),
                              v_mean=float(ro["v"].mean()), adv_sd=float(adv.std())))
        return float(loss.mean())

    def diag_info(self, last):
        d = self.diag[-last:]
        return {k: float(np.mean([x[k] for x in d])) for k in d[0]} if d else {}


# ══════════ 실행 모델 (상태 있음) ══════════
class A2CModel:
    """월별 모델: 멤버 평균 기대 비중을 25%로 반올림해 실행하고, 그 비중을 다음 판단의 w_prev로 이어감"""

    def __init__(self, net, fi, acts, pos0=0.0):
        self.net = StackedMLP.__new__(StackedMLP)
        self.net.__dict__.update(net.__dict__)
        self.net.params = net.copy_params()
        self.net._cache = None
        self.fi = np.asarray(fi)
        self.acts = np.asarray(acts, np.float64)
        if not np.allclose(self.acts, np.round(self.acts * 4) / 4) or len(np.unique(self.acts)) != len(self.acts):
            raise ValueError("a2c: 실행 비중(25% 격자)이 행동 목록 안에 있어야 합니다")
        self.K = len(self.acts)
        self.pos0 = float(pos0)
        self.last_pos = float(pos0)
        self.last_w = np.zeros(0)

    def _ew(self, x, w_prev):
        """고른 지표 x (B, F)와 w_prev (B,) → 멤버 평균 기대 비중 Ē (B,)"""
        inp = np.concatenate([np.asarray(x, self.net.dtype), np.asarray(w_prev, self.net.dtype)[:, None]], axis=1)
        z = self.net.forward(inp, cache=False)[..., :self.K].astype(np.float64)
        pi = np.exp(log_softmax(z))
        return (pi * self.acts).sum(axis=-1).mean(axis=0)

    def expected_weight(self, X, w_prev):
        """행마다 멤버 평균 기대 비중 Ē (B,) — X: (B, 전체 지표), w_prev: (B,)"""
        return self._ew(np.asarray(X)[:, self.fi], w_prev)

    def _table(self, X):
        """(B, K): 행 b에서 w_prev = acts[j]일 때의 실행 비중 번호 — 5가지를 한 번에 순전파. NaN 행은 fin=False"""
        B = len(X)
        x = np.asarray(X)[:, self.fi].astype(np.float64)
        fin = np.isfinite(x).all(axis=1)
        x = np.where(fin[:, None], x, 0.0)
        E = self._ew(np.repeat(x, self.K, axis=0), np.tile(self.acts, B)).reshape(B, self.K)
        w = np.round(4.0 * E) / 4.0
        idx = np.abs(w[..., None] - self.acts).argmin(axis=-1)
        return idx, fin, E

    def weights(self, X):
        """한 달치 판단봉을 시간 순서대로 → 실행 비중 (B,) ∈ 25% 격자. 매번 pos0에서 시작(멱등), last_pos 갱신"""
        B = len(X)
        out = np.empty(B)
        p = int(np.abs(self.acts - self.pos0).argmin())
        if B:
            idx, fin, _ = self._table(X)
            for i in range(B):
                if fin[i]:
                    p = int(idx[i, p])
                out[i] = self.acts[p]
        self.last_pos = float(out[-1]) if B else self.pos0
        self.last_w = out.copy()
        return out

    def state(self):
        st = self.net.state("p")
        st["acts"] = self.acts
        st["fi"] = np.asarray(self.fi)
        st["pos"] = np.array([self.pos0, self.last_pos])
        st["last_w"] = np.asarray(self.last_w, np.float64)
        return st


def proposed_cfg():
    """variants.py 에 등록할 설정 — R6_daily_trend8_uniform(stride 6, γ 0.967, TREND8, 균등 추출)에 A1 키만 더함.
    reward='log'는 문서용(이 기법은 KTrainer 보상을 쓰지 않고 step_reward를 씀). 'tau'(P0의 목표망 갱신률)는 쓰지 않음."""
    from ..variants import v, TREND8
    return v("A1_actor_critic", algo="a2c", output="weights", stride=6, gamma=0.967, feat=TREND8, recency_frac=0.0,
             acts=ACTS, reward="log", cost_train=0.003, seq_len=60, seq_batch=32, gae_lambda=0.95, ent_coef=0.01,
             vf_coef=0.5, adv_norm=True)


def monthly_update(cfg, datas, T_k, seed, ens, anchor):
    """
    walk.monthly_update 훅. 1월(또는 첫 달)은 처음부터 2000걸음, 나머지 달은 지난달 모델에서 300걸음 L2-SP 이어학습
    (앵커 = 그해 1월 모델). 점검 없이 항상 채택. 실행 시작 비중 pos0 = 지난달 모델의 마지막 실행 비중.
    """
    Tk = int(T_k.timestamp())
    pos0 = float(getattr(ens, "last_pos", 0.0)) if ens is not None else 0.0
    tr = A2CTrainer(cfg, seed)
    n = tr.make_pool(datas, Tk, FIRST_TRAIN, WARMUP)
    if n == 0:
        raise RuntimeError(f"a2c: 학습 표본 풀이 비었습니다 (T_k={T_k})")
    entry = dict(month=str(T_k.date()), pool=int(n), pos0=pos0)
    if T_k.month == 1 or ens is None:
        tr.init_fresh()
        losses = tr.fit_cold()
        entry.update(kind="cold", loss=float(np.mean(losses[-DIAG_COLD:])), accepted=True, **tr.diag_info(DIAG_COLD))
        model = A2CModel(tr.net, tr.fi, tr.acts, pos0)
        return model, model.net.copy_params(), entry
    tr.init_from(ens.net.params)
    losses = tr.fit_finetune(anchor)
    entry.update(kind="finetune", loss=float(np.mean(losses[-DIAG_FT:])), accepted=True, **tr.diag_info(DIAG_FT))
    return A2CModel(tr.net, tr.fi, tr.acts, pos0), anchor, entry
