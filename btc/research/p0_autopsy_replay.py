# -*- coding: utf-8 -*-
"""
P0 부검 — 게이트 반사실(counterfactual) 워크포워드 재현.

  V0  원래대로 (두 게이트 켬)  → 저장된 data/btc/runs/P0/rep*.npz 의 Δ와 비트 단위 일치 확인 + 거부된 후보의 그달 Δ(shadow)
  V1  게이트 둘 다 끔         (행동 점검만 끔; 수치 이상 점검은 유지)
  V2  1월 새 모델 점검만 끔     (노출도 10~95%·연 60회 전환)
  V3  월간 이어학습 불일치 점검만 끔 (180봉 불일치 40%)

  python -m btc.research.p0_autopsy_replay --reps 0 6 --variants V0 V1 V2 V3

사후 연구입니다(잠금 구간 2025-01 ~ 2026-09 포함). 결과: btc/research/out/p0_autopsy/replay_{V}_r{rr}.npz
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from btc.research.p0_autopsy_common import (OUT, CD, replay, load_phases, load_run, decision_range, ts)

VARIANTS = {"V0": (True, True), "V1": (False, False), "V2": (False, True), "V3": (True, False)}
KEEP = ("2018-01-01", "2022-01-01", "2025-01-01", "2026-01-01")


def path(v, r):
    return os.path.join(OUT, f"replay_{v}_r{r:02d}.npz")


def run_one(v, r, datas):
    gc, gf = VARIANTS[v]
    t0 = time.time()
    res = replay(r, datas, gate_cold=gc, gate_ft=gf, keep=KEEP if v == "V0" else ())
    out = dict(delta=res["delta"][CD], shadow=res["shadow"][CD],
               log=np.frombuffer(json.dumps(res["log"]).encode(), dtype=np.uint8))
    d0 = datas[0]
    if v == "V0":
        # 저장된 실행과 비교 (재현성)
        sd = load_run("P0", r)["delta"][0][CD]
        mine = res["delta"][CD]
        same_nan = np.array_equal(np.isnan(sd), np.isnan(mine))
        ok = ~np.isnan(mine)
        out["repro_exact"] = np.array([same_nan and np.array_equal(sd[ok], mine[ok])])
        out["repro_maxabs"] = np.array([float(np.nanmax(np.abs(sd[ok] - mine[ok])))])
        # 거부된 1월 모델을 '그해 내내 그대로' 썼다면의 Δ (frozen) — 그해 1월 ~ 이듬해 1월 (2026은 9/25까지)
        for mon, ens in res["models"].items():
            y = int(mon[:4])
            a, b = decision_range(d0, ts(f"{y}-01-01"), ts(f"{y + 1}-01-01"))
            fz = np.full(d0.T, np.nan, np.float32)
            fz[a:b] = ens.delta(d0.X[a:b], CD)[0]
            out[f"frozen_{y}"] = fz
            for i, p in enumerate(ens.net.params):
                out[f"model_{y}_p{i}"] = p
    os.makedirs(OUT, exist_ok=True)
    tmp = path(v, r) + ".tmp.npz"
    np.savez_compressed(tmp, **out)
    os.replace(tmp, path(v, r))
    msg = f"{v} r{r} {time.time() - t0:.0f}s"
    if v == "V0":
        msg += f" repro_exact={bool(out['repro_exact'][0])} maxabs={out['repro_maxabs'][0]:.3g}"
    print(msg, flush=True)


def load_replay(v, r):
    z = np.load(path(v, r))
    out = {k: z[k] for k in z.files if k != "log"}
    out["log"] = json.loads(bytes(z["log"]).decode())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    a = ap.parse_args()
    datas = load_phases()
    for r in a.reps:
        for v in a.variants:
            if os.path.exists(path(v, r)):
                print(f"{v} r{r} cached", flush=True)
                continue
            run_one(v, r, datas)


if __name__ == "__main__":
    main()
