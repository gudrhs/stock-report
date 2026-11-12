# -*- coding: utf-8 -*-
"""
numpy만으로 만든 작은 신경망 — torch 없이 돌아가도록 역전파를 직접 씁니다.

앙상블 멤버 M개를 맨 앞 축에 쌓아 한 번의 행렬곱으로 같이 계산합니다.
  입력 (M, B, I) → ReLU 32 → ReLU 32 → 출력 4 (U0, U1, 보조6, 보조42)
멤버마다 초기값·미니배치가 달라서 서로 다른 모델이 됩니다.
tests/test_btc_core.py 가 유한차분으로 기울기를 검증합니다.
"""
import numpy as np


class StackedMLP:
    def __init__(self, n_members, sizes, rng, last_scale=0.01, dtype=np.float32):
        self.M = n_members
        self.sizes = list(sizes)
        self.dtype = dtype
        self.params = []
        L = len(sizes) - 1
        for k in range(L):
            a, b = sizes[k], sizes[k + 1]
            W = rng.standard_normal((n_members, a, b)) * np.sqrt(2.0 / a)     # He 초기화
            if k == L - 1:
                W *= last_scale
            self.params += [W.astype(dtype), np.zeros((n_members, 1, b), dtype=dtype)]
        self._cache = None

    @property
    def n_layers(self):
        return len(self.params) // 2

    def forward(self, x, cache=True):
        """x: (M, B, I) 또는 (B, I) — 후자는 모든 멤버에 같은 입력"""
        if x.ndim == 2:
            x = np.broadcast_to(x, (self.M,) + x.shape)
        hs = [x]
        h = x
        L = self.n_layers
        for k in range(L):
            z = np.matmul(h, self.params[2 * k]) + self.params[2 * k + 1]
            h = np.maximum(z, 0) if k < L - 1 else z
            hs.append(h)
        if cache:
            self._cache = hs
        return h

    def backward(self, gout):
        hs = self._cache
        L = self.n_layers
        grads = [None] * len(self.params)
        g = gout
        for k in range(L - 1, -1, -1):
            hin = hs[k]
            grads[2 * k] = np.matmul(np.swapaxes(hin, 1, 2), g)
            grads[2 * k + 1] = g.sum(axis=1, keepdims=True)
            if k > 0:
                g = np.matmul(g, np.swapaxes(self.params[2 * k], 1, 2)) * (hs[k] > 0)
        return grads

    def copy_params(self):
        return [p.copy() for p in self.params]

    def set_params(self, ps):
        self.params[:] = [np.array(p, dtype=self.dtype) for p in ps]

    def state(self, prefix="p"):
        return {f"{prefix}{i}": p for i, p in enumerate(self.params)}

    @classmethod
    def from_state(cls, st, prefix="p", dtype=np.float32):
        ps = []
        i = 0
        while f"{prefix}{i}" in st:
            ps.append(np.array(st[f"{prefix}{i}"], dtype=dtype))
            i += 1
        sizes = [ps[0].shape[1]] + [ps[2 * k].shape[2] for k in range(len(ps) // 2)]
        net = cls.__new__(cls)
        net.M, net.sizes, net.dtype, net.params, net._cache = ps[0].shape[0], sizes, dtype, ps, None
        return net


class Adam:
    """멤버별 전역 노름 클리핑을 하는 Adam"""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, clip=5.0):
        self.params = params
        self.lr, self.b1, self.b2, self.eps, self.clip = lr, betas[0], betas[1], eps, clip
        self.reset()

    def reset(self):
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0

    def step(self, grads):
        if self.clip:
            sq = sum((g.astype(np.float64) ** 2).reshape(g.shape[0], -1).sum(1) for g in grads)
            norm = np.sqrt(sq)                                   # (M,)
            scale = np.minimum(1.0, self.clip / (norm + 1e-12)).astype(grads[0].dtype)
            grads = [g * scale.reshape((-1,) + (1,) * (g.ndim - 1)) for g in grads]
        self.t += 1
        c1 = 1 - self.b1 ** self.t
        c2 = 1 - self.b2 ** self.t
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            p -= (self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)).astype(p.dtype)
