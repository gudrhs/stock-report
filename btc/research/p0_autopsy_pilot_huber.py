# -*- coding: utf-8 -*-
"""
P0 부검 — 작은 예비 실험(pilot): Huber 손실 δ가 Δ의 '낙관 편향'과 매도 능력을 만드는가?

가설 (부검 결과를 본 뒤 세운 사후 가설, 이 실험 전 기록):
  P0는 U0·U1을 각각 Huber(δ=1)로 맞춥니다. 목표 ±½κm 의 4시간 수익(κ=100 → % 단위) 분포는 꼬리가 두껍고
  음의 왜도라, Huber 위치는 평균보다 높습니다(큰 하락 봉을 덜 반영). 이 편향이 γ-할인으로 ~33배 누적되면
  Δ의 절편이 ±0.30 문턱과 같은 크기만큼 위로 밀려, 모델이 '팔 수 없는' 상태가 됩니다.
  예측: δ를 키우면(→ 제곱 손실) (a) 1월 새 모델의 최근 2년 노출도가 낮아지고, (b) 2018·2022·2025~26에서
  Δ<−0.30 비율이 늘고, (c) 노출도가 낮아집니다. 성과(샤프)가 좋아진다는 예측은 하지 않습니다.

설계: 매년 1월(2017~2026) P0 설정 그대로 처음부터 학습(같은 시드 (r, 연, 1, 0) → δ=1은 P0의 실제 1월 모델과 동일),
  이어학습·게이트 없이 그해 내내 그대로 사용(frozen). δ ∈ {1, 3, 10}, r ∈ {0, 6, 9}.
  사후 연구입니다(잠금 구간 포함, 결과를 본 뒤의 실험). 판정용이 아니라 다음 설계의 방향을 보는 용도입니다.

  python -m btc.research.p0_autopsy_pilot_huber
결과: btc/research/out/p0_autopsy/pilot_huber.json
"""
import json
import os
import time

import numpy as np
import pandas as pd

from btc import config as C
from btc.agent import Trainer
from btc.research.p0_autopsy_common import (OUT, CD, TH, COST, LO, HI, S, ts, load_phases, decision_range,
                                            sanity_ok, run_targets, agent_targets, bh_targets, sub_stats,
                                            policy_from_delta, window_range, BAR_SEC)
from btc.walkforward import FIRST_TRAIN
from btc.features import WARMUP

DELTAS = (1.0, 3.0, 10.0)
REPS = (0, 6, 9)
YEARS = range(2017, 2027)


class HuberTrainer(Trainer):
    """Trainer 와 같되 U 손실의 Huber 문턱만 δ (보조 손실은 그대로 δ=1)"""

    def __init__(self, cfg, seed_seq, delta=1.0):
        super().__init__(cfg, seed_seq)
        self.hd = float(delta)

    def loss_grads(self, X0, y, z6, z42, ok, anchor=None, lam_sp=0.0):
        c = self.cfg
        k = self.hd
        hubk = lambda d: np.where(np.abs(d) <= k, 0.5 * d * d, k * (np.abs(d) - 0.5 * k))
        hub1 = lambda d: np.where(np.abs(d) <= 1, 0.5 * d * d, np.abs(d) - 0.5)
        out = self.net.forward(X0, cache=True)
        B = y.shape[1]
        d = out[..., :2] - y
        a6 = out[..., 2] - z6
        a42 = out[..., 3] - z42
        w = c["aux_w"]
        loss = (hubk(d).sum() / (2.0 * B) + w * (ok * hub1(a6)).sum() / B + w * (ok * hub1(a42)).sum() / B)
        gU = np.clip(d, -k, k) / (2.0 * B)
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
        return float(loss), grads, float(hubk(d).mean())


def frozen_cold_run(datas, cfg, hd, r, yrs_bar, a, b):
    """매년 1월 새 모델(시드 (r, 연, 1, 0))을 그해 내내 그대로 사용 — 이어학습·게이트 없음"""
    d0 = datas[0]
    t0 = time.time()
    dl = np.full(d0.T, np.nan, np.float32)
    gates = {}
    for y in YEARS:
        Tk = ts(f"{y}-01-01")
        tr = HuberTrainer(cfg, (r, y, 1, 0), delta=hd)
        tr.make_pool(datas, Tk, FIRST_TRAIN, WARMUP)
        tr.init_fresh()
        tr.fit_cold()
        ens = tr.ensemble()
        ok, info = sanity_ok(ens, d0, Tk)
        gates[y] = dict(ok=ok, exposure=info.get("exposure"), switches=info.get("switches_per_year"))
        i0, i1 = decision_range(d0, Tk, ts(f"{y + 1}-01-01"))
        dl[i0:i1] = ens.delta(d0.X[i0:i1], CD)[0]
    res = run_targets(d0, agent_targets(d0, dl.astype(float)), COST)
    x = dl[a:b].astype(float)
    pos = res["pos"]
    per_year = {}
    for y in YEARS:
        sel = yrs_bar == y
        per_year[y] = dict(mean_delta=float(np.nanmean(x[sel])), sell_zone=float(np.mean(x[sel] < -TH)),
                           exposure=float(pos[sel].mean()),
                           ret=sub_stats(res["days"], res["r"], f"{y}-01-01", f"{y + 1}-01-01" if y < 2026 else HI)["ret"])
    return dict(delta=hd, rep=r, cfg=cfg["name"], gates=gates, per_year=per_year,
                full=sub_stats(res["days"], res["r"], LO, HI),
                dev=sub_stats(res["days"], res["r"], LO, "2025-01-01"),
                lockbox=sub_stats(res["days"], res["r"], "2025-01-01", HI),
                exposure=float(pos.mean()), secs=round(time.time() - t0, 1))


def run_pilot(configs, fname):
    datas = load_phases()
    d0 = datas[0]
    a, b = window_range(d0, LO, HI)
    yrs_bar = pd.to_datetime(d0.ts[a:b] + BAR_SEC, unit="s", utc=True).year.to_numpy()
    bh = run_targets(d0, bh_targets(d0), COST)
    out = dict(configs=[(c["name"], hd) for c, hd in configs], reps=REPS, runs=[])
    for cfg, hd in configs:
        for r in REPS:
            row = frozen_cold_run(datas, cfg, hd, r, yrs_bar, a, b)
            out["runs"].append(row)
            print(f"{cfg['name']} δ={hd:g} r{r}: Sharpe {row['full']['sharpe']:.3f} MDD {row['full']['max_dd']:.3f} "
                  f"exp {row['exposure']:.2f} gates_ok {sum(g['ok'] for g in row['gates'].values())}/10 {row['secs']}s",
                  flush=True)
    out["bh"] = dict(full=sub_stats(bh["days"], bh["r"], LO, HI), dev=sub_stats(bh["days"], bh["r"], LO, "2025-01-01"),
                     lockbox=sub_stats(bh["days"], bh["r"], "2025-01-01", HI))
    with open(os.path.join(OUT, fname), "w") as f:
        json.dump(out, f, indent=1, default=float)


def main():
    run_pilot([(C.P0, hd) for hd in DELTAS], "pilot_huber.json")


if __name__ == "__main__":
    main()
