# -*- coding: utf-8 -*-
"""
차트 지표 → 신경망 입력 (22개 + 비용 1개).

사람이 차트에서 보는 이동평균·거래량·RSI·MACD 등을 계산하되, 가격 수준과 무관한
숫자로 바꿔 넣습니다. 비트코인이 1천 달러일 때와 10만 달러일 때 같은 모양의 차트는
같은 숫자가 되어야 과거 가격대를 외우지 않습니다.

  · 가격 차이는 로그비율 ÷ 최근 변동성 σ (σ = 4시간 수익률 제곱의 지수평균, span 42 ≈ 1주)
  · 오실레이터는 50 중심으로 옮기고 고정 상수로 나눔
  · 거래량은 자기 평균 대비 로그비율 (거래소마다 단위가 달라도 같은 값)
  · 학습 데이터로 맞춘 정규화 상수(평균·표준편차 등)는 하나도 없습니다 → 미래 정보가 샐 틈이 없음

4시간봉 기준 창 길이: 6봉=1일, 42봉=1주, 180봉=1달, 1200봉≈200일.
"일봉 지표"(200일선, 일봉 RSI14, 일봉 MACD)는 같은 4시간봉 계열에서 창을 6배로 늘려 근사합니다.

계산은 FeatureEngine.update(봉 하나)로만 합니다. 백테스트용 compute()도 과거 봉을 이
함수에 순서대로 넣을 뿐이라 실시간과 값이 같고, 아직 오지 않은 봉을 볼 방법이 없습니다.
tests/btc_features_ref.py 가 pandas로 따로 짠 구현과 1e-9까지 맞는지 확인합니다.
"""
import math
from collections import deque

import numpy as np

NAMES = ["ret_1", "ret_6", "ret_42", "ret_180", "ma_20", "ma_50", "ma_200", "ma_1200",
         "x_50_200", "x_300_1200", "rsi_14", "rsi_84", "macd_4h", "macdh_4h", "macdh_1d",
         "vol_regime", "rel_vol", "vol_trend", "flow_20", "range", "clv", "dd_180"]
MIN8 = ["ret_6", "ret_42", "vol_regime", "ma_50", "ma_200", "rsi_14", "macdh_4h", "rel_vol"]
WARMUP = 2400          # 이만큼 봉이 쌓여야 가장 긴 지표(1200봉)와 느린 지수평균이 안정

SIG_FLOOR = 0.002
C_REF = 0.00316        # 비용 입력 정규화: c ∈ [0.05%, 2%] → [−1, 1]
C_SCALE = 1.846
C_MIN, C_MAX = 0.0005, 0.02


def c_in(c):
    return (math.log(c) - math.log(C_REF)) / C_SCALE


def _u(x):
    return max(-4.0, min(4.0, x)) / 2.0


def _clip(x, a, b):
    return a if x < a else (b if x > b else x)


class _Win:
    """최근 n개 합 (n번마다 정확히 다시 더해 오차 누적 방지)"""
    __slots__ = ("n", "q", "s", "k")

    def __init__(self, n):
        self.n, self.q, self.s, self.k = n, deque(), 0.0, 0

    def push(self, x):
        self.q.append(x)
        self.s += x
        if len(self.q) > self.n:
            self.s -= self.q.popleft()
        self.k += 1
        if self.k >= self.n:
            self.k = 0
            self.s = math.fsum(self.q)

    def mean(self):
        return self.s / len(self.q) if self.q else float("nan")


class _Ema:
    __slots__ = ("a", "v")

    def __init__(self, n=None, alpha=None):
        self.a = alpha if alpha is not None else 2.0 / (n + 1.0)
        self.v = None

    def push(self, x):
        self.v = x if self.v is None else self.v + self.a * (x - self.v)
        return self.v


class _Rsi:
    """와일더 RSI — 처음 n개 변화의 단순평균으로 시작"""
    __slots__ = ("n", "g", "l", "seed_g", "seed_l", "cnt")

    def __init__(self, n):
        self.n, self.g, self.l, self.seed_g, self.seed_l, self.cnt = n, None, None, 0.0, 0.0, 0

    def push(self, d):
        gain, loss = (d if d > 0 else 0.0), (-d if d < 0 else 0.0)
        self.cnt += 1
        if self.g is None:
            self.seed_g += gain
            self.seed_l += loss
            if self.cnt == self.n:
                self.g, self.l = self.seed_g / self.n, self.seed_l / self.n
        else:
            self.g = (self.g * (self.n - 1) + gain) / self.n
            self.l = (self.l * (self.n - 1) + loss) / self.n
        return self.value()

    def value(self):
        if self.g is None:
            return 50.0
        if self.l == 0.0:
            return 50.0 if self.g == 0.0 else 100.0
        return 100.0 - 100.0 / (1.0 + self.g / self.l)


class _Max:
    """최근 n개 최댓값 (단조 덱)"""
    __slots__ = ("n", "i", "q")

    def __init__(self, n):
        self.n, self.i, self.q = n, -1, deque()

    def push(self, x):
        self.i += 1
        q = self.q
        while q and q[-1][1] <= x:
            q.pop()
        q.append((self.i, x))
        while q[0][0] <= self.i - self.n:
            q.popleft()
        return q[0][1]


class FeatureEngine:
    def __init__(self):
        self.n = 0
        self.prev_c = None
        self.E = _Ema(42)
        self.r2L = _Win(540)
        self.lnc = deque(maxlen=181)           # ln C 최근 181개 (180봉 전까지)
        self.sma = {k: _Win(k) for k in (20, 50, 200, 300, 1200)}
        self.e12, self.e26, self.s9 = _Ema(12), _Ema(26), _Ema(9)
        self.e72, self.e156, self.s54 = _Ema(72), _Ema(156), _Ema(54)
        self.rsi14, self.rsi84 = _Rsi(14), _Rsi(84)
        self.vprev = _Win(180)                  # 현재 봉 제외 직전 180봉 거래량
        self.v20, self.v180 = _Win(20), _Win(180)
        self.flow_num, self.flow_den = _Win(20), _Win(20)
        self.hmax = _Max(180)
        self.sigma = SIG_FLOOR
        self.last = None

    @property
    def ready(self):
        return self.n >= WARMUP

    def update(self, o, h, l, c, v):
        """봉 하나(시·고·저·종·거래량)를 넣고 22개 특징(np.ndarray)을 돌려줍니다."""
        self.n += 1
        pc = self.prev_c
        lnc = math.log(c)
        if pc is None:
            r = float("nan")
            d = 0.0
            sgn = 0.0
        else:
            r = lnc - math.log(pc)
            d = c - pc
            sgn = 1.0 if d > 0 else (-1.0 if d < 0 else 0.0)
            self.E.push(r * r)
            self.r2L.push(r * r)
            self.rsi14.push(d)
            self.rsi84.push(d)
        E = self.E.v if self.E.v is not None else 0.0
        sig = max(SIG_FLOOR, math.sqrt(E))
        sigL = max(SIG_FLOOR, math.sqrt(self.r2L.mean())) if self.r2L.q else SIG_FLOOR
        self.sigma = sig

        self.lnc.append(lnc)
        L = len(self.lnc)
        f = [0.0] * 22
        for j, k in enumerate((1, 6, 42, 180)):
            base = self.lnc[L - 1 - k] if L > k else self.lnc[0]
            f[j] = _u((lnc - base) / (sig * math.sqrt(k)))

        for k, w in self.sma.items():
            w.push(c)
        m20, m50, m200 = self.sma[20].mean(), self.sma[50].mean(), self.sma[200].mean()
        m300, m1200 = self.sma[300].mean(), self.sma[1200].mean()
        f[4] = _u(math.log(c / m20) / (sig * math.sqrt(20 / 3)))
        f[5] = _u(math.log(c / m50) / (sig * math.sqrt(50 / 3)))
        f[6] = _u(math.log(c / m200) / (sig * math.sqrt(200 / 3)))
        f[7] = _u(math.log(c / m1200) / (sig * math.sqrt(400)))
        f[8] = _u(math.log(m50 / m200) / (sig * math.sqrt(200 / 3)))
        f[9] = _u(math.log(m300 / m1200) / (sig * math.sqrt(400)))

        f[10] = _clip((self.rsi14.value() - 50.0) / 25.0, -2.0, 2.0)
        f[11] = _clip((self.rsi84.value() - 50.0) / 10.0, -2.0, 2.0)

        macd = self.e12.push(c) - self.e26.push(c)
        sig9 = self.s9.push(macd)
        f[12] = _clip(macd / (c * sig), -8.0, 8.0) / 4.0
        f[13] = _u((macd - sig9) / (c * sig))
        macd_d = self.e72.push(c) - self.e156.push(c)
        sig54 = self.s54.push(macd_d)
        f[14] = _u((macd_d - sig54) / (c * sig * math.sqrt(6)))

        f[15] = _clip(math.log(sig / sigL), -2.0, 2.0)

        B = self.vprev.mean() if self.vprev.q else 0.0
        f[16] = (_clip(math.log((v + 0.01 * B) / B), -3.0, 3.0) / 1.5) if B > 0 else 0.0
        self.vprev.push(v)
        self.v20.push(v)
        self.v180.push(v)
        vb20, vb180 = self.v20.mean(), self.v180.mean()
        if vb180 > 0:
            eps = 0.01 * vb180
            f[17] = _clip(math.log((vb20 + eps) / (vb180 + eps)), -3.0, 3.0) / 1.5
        self.flow_num.push(sgn * v)
        self.flow_den.push(v)
        f[18] = self.flow_num.s / (self.flow_den.s + 1e-12)

        f[19] = _clip(math.log(h / l) / sig, 0.0, 6.0) / 2.0 - 1.0 if l > 0 else -1.0
        f[20] = (2.0 * c - h - l) / (h - l) if h > l else 0.0
        hm = self.hmax.push(h)
        f[21] = _clip(math.log(c / hm) / (sig * math.sqrt(180)), -4.0, 0.0) / 2.0

        self.prev_c = c
        self.last = np.array(f)
        return self.last


def compute(df):
    """
    DataFrame(open, high, low, close, volume) → (X (T,22), sigma (T,)).
    X[t]는 t봉 종가까지만 보고 계산한 값입니다.
    """
    eng = FeatureEngine()
    o, h, l, c, v = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close", "volume"))
    T = len(df)
    X = np.empty((T, len(NAMES)))
    sig = np.empty(T)
    for i in range(T):
        X[i] = eng.update(o[i], h[i], l[i], c[i], v[i])
        sig[i] = eng.sigma
    return X, sig
