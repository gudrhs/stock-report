# -*- coding: utf-8 -*-
"""
P0 부검 — 월간 이어학습 거부 점검(불일치 > 40%)이 실제로 무엇을 재는가.

walkforward.finetune_ok 는 최근 180봉에서 새·옛 모델의 보유 계열을 '현금(p0=0)에서 시작'해 비교합니다.
Δ가 ±0.30 사이(무거래 띠)에 오래 머무는 해에는, 두 모델이 띠 위(+0.30)를 처음 넘는 시점만 달라도
불일치가 크게 나옵니다 — 실제 보유(대개 코인)에서 시작하면 둘 다 계속 보유라 불일치가 0일 수 있습니다.
여기서는 2024-01부터 다시 돌려(비트 단위 재현) 2025~26년의 거부된 이어학습마다
  · 기록된 불일치(p0=0), · 실제 보유에서 시작한 불일치, · 두 모델이 띠 안에 있던 비율
을 계산합니다. 사후 연구입니다.

  python -m btc.research.p0_autopsy_ftgate --reps 0 6
결과: btc/research/out/p0_autopsy/ftgate.json
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from btc.research.p0_autopsy_common import (OUT, CD, TH, ts, load_phases, load_run, replay, policy_from_delta,
                                            decision_range, BAR_SEC, window_range, LO, HI)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, nargs="+", default=[0, 6])
    ap.add_argument("--start", default="2024-01-01")
    a_ = ap.parse_args()
    datas = load_phases()
    d0 = datas[0]
    A, B = window_range(d0, LO, HI)
    Ms = [str(m.date()) for m in pd.date_range(a_.start, "2026-09-01", freq="MS")]
    out = []
    for r in a_.reps:
        res = replay(r, datas, start=a_.start, end=pd.Timestamp("2026-09-25", tz="UTC"), keep=tuple(Ms))
        saved = load_run("P0", r)
        pos_act = policy_from_delta(saved["delta"][0][CD][A:B].astype(float), CD, d0.forced_hold[A:B])
        cur = None
        for e in res["log"]:
            new = res["models"][e["month"]]
            if e["kind"] == "finetune" and not e["accepted"] and cur is not None:
                Tk = ts(e["month"])
                b = int(np.searchsorted(d0.ts + BAR_SEC, Tk, side="left"))
                a = b - 180
                dn = new.delta(d0.X[a:b], 0.003)[0]
                do = cur.delta(d0.X[a:b], 0.003)[0]
                fh = d0.forced_hold[a:b]
                p_start = int(pos_act[a - A - 1]) if a - A - 1 >= 0 else 0
                dis0 = float((policy_from_delta(dn, 0.003, fh) != policy_from_delta(do, 0.003, fh)).mean())
                pn1 = policy_from_delta(dn, 0.003, fh, p0=p_start)
                po1 = policy_from_delta(do, 0.003, fh, p0=p_start)
                out.append(dict(rep=r, month=e["month"], logged_disagree=e.get("disagree"), disagree_p0_0=dis0,
                                actual_pos_at_window_start=p_start,
                                disagree_from_actual=float((pn1 != po1).mean()),
                                new_exposure_from_actual=float(pn1.mean()), old_exposure_from_actual=float(po1.mean()),
                                in_band_new=float(np.mean(np.abs(dn) <= TH)), in_band_old=float(np.mean(np.abs(do) <= TH)),
                                min_new=float(dn.min()), min_old=float(do.min()),
                                mean_new=float(dn.mean()), mean_old=float(do.mean())))
                print(out[-1], flush=True)
            if e["accepted"]:
                cur = new
    with open(os.path.join(OUT, "ftgate.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
