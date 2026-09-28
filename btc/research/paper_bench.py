# -*- coding: utf-8 -*-
"""
모의매매 보조 기록 — btc/research/success_criteria.md '모의매매 보조 기록 보완 (2026-09-28 02:10 UTC 등록)'.

등록 커밋의 고정 코드(paper.py step·report)와 따로, HEAD 코드로 같은 데이터(기본 15분봉 + 덧붙인 봉)를 읽어 씁니다.
  · 기준 규칙 장부 — B0 매수·보유, B2 200일선, B5 일봉 MACD, B6 1개월 모멘텀, B80 늘 80%, SMA120(사후 선택)
      booked: 등록 계산과 같은 체결(판단봉 다음 봉 시가 = 00:00). B0 booked 는 변형 폴더의 bh_ledger.csv 와 같은 값
      exec  : 실행 가능 시각(00:30 15분봉 시가) 체결
  · 등록 변형의 실행 가능 장부 — 판단 기록(decisions_repNN.csv)을 읽기만 하고, 판단을 00:20 실행 + 00:30 체결로 옮김.
      매달 첫 판단봉은 그 달 학습 자르기 c(T)(train_log 의 data_cut, 없으면 T + 1일) + 20분 뒤 첫 00:30 에 체결
      (c(T) = T + 1일이면 2일 판단과 같은 시각 → 2일 판단이 이김)
판정에 쓰지 않습니다 (등록 문구: 보조 기록). 판정은 고정 코드의 report.json 그대로입니다.

안전 규칙: 쓰는 곳은 <모의매매 폴더>/benchmarks/ 뿐입니다. 등록부·status.json·report.json·alarms.jsonl·
변형 폴더·ext/(덧붙인 봉)는 읽기만 하고, 데이터를 새로 받지도 않습니다 (고정 step 이 받은 봉까지만 씀).
기준 시각 now = status.json 의 now_ts (마지막 고정 step 과 같은 데이터 범위).

  python -m btc.research.paper_bench [--dir data/btc/paper] [--base data/btc/btcusd_15m.csv.gz] [--now ISO]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from ..env import BAR_SEC
from . import paper as P
from .recost import rule_signals, exec_open, exec_targets, run_one, RULES, RULE_KO, FRAC_RULES
from .evaluate import daily_hold
from .walk import decision_mask

DAY = 86400
SUB = "benchmarks"
STRIDE = 6
NOTE = ("보조 기록 — 판정에 쓰지 않음 (success_criteria.md '모의매매 보조 기록 보완'). booked = 등록 계산과 같은 00:00 체결, "
        "exec = 00:20 판단 뒤 00:30 체결 (학습 모델의 월초 판단은 c(T) + 20분 뒤)")


def bench_dir(paper_dir):
    return os.path.join(paper_dir or P.PAPER_DIR, SUB)


def _hi(d0, now):
    close = d0.ts + BAR_SEC
    return (min(int(now), int(close[-1])) // DAY) * DAY


def _ledger_rows(res):
    return [dict(date=P._date(t), equity=P._num(e)) for t, e in zip(res["days"], res["eq"])] if res is not None else []


def _fill_rows(d0, res, fill_px, fill_at):
    if res is None:
        return []
    out = []
    a = res["a"]
    for j in np.nonzero(res["traded"] > 1e-12)[0]:
        t = a + j
        out.append(dict(close_utc=P._iso(d0.ts[t] + BAR_SEC), fill_utc=P._iso(d0.ts[t + 1] + fill_at),
                        fill_open=P._num(fill_px[t + 1]), w_before=P._num(res["w_pre"][j]),
                        w_after=P._num(res["pos"][j])))
    return out


def run(d0, lo, hi, tg, frac, o, c):
    """(lo, hi) 구간 장부 — paper.window_range 와 같은 (a, b). 반환 None(구간 없음) 또는 dict"""
    a, b = P.window_range(d0, lo, hi)
    if b <= a:
        return None
    res = run_one(d0, a, b, lo, hi, tg, frac, P.COST, o, c)
    pos = res["pos"]
    prev = np.concatenate([[0.0], pos[:-1]])
    g = o[a + 1:b + 1] / o[a:b]                         # paper.window_run 과 같은 체결 전 비중
    den = prev * g + (1.0 - prev)
    res["w_pre"] = np.where(den > 0, prev * g / np.where(den > 0, den, 1.0), 0.0)
    res["traded"] = np.abs(pos - res["w_pre"])
    res["a"], res["b"] = a, b
    return res


def _write_pair(folder, tag, d0, res, fill_px, fill_at):
    P._write_csv(os.path.join(folder, f"ledger_{tag}.csv"), ["date", "equity"], _ledger_rows(res))
    P._write_csv(os.path.join(folder, f"fills_{tag}.csv"), ["close_utc", "fill_utc", "fill_open", "w_before", "w_after"],
                 _fill_rows(d0, res, fill_px, fill_at))


def month_cuts_from_logs(vdir, reps, lo, hi):
    """월 시작 T → c(T): train_log 의 data_cut(ISO)이 있으면 그것, 없거나 'hist' 면 T + 1일 (반복마다 같아야 함)"""
    cuts = {}
    T = pd.Timestamp(P._utc(lo).year, P._utc(lo).month, 1, tz="UTC")
    while int(T.timestamp()) < hi:
        cuts[int(T.timestamp())] = int(T.timestamp()) + DAY
        T = T + pd.offsets.MonthBegin(1)
    for r in range(reps):
        p = os.path.join(vdir, f"train_log_rep{r:02d}.jsonl")
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                e = json.loads(line)
                cut = e.get("data_cut")
                if not cut or cut == "hist":
                    continue
                Tm = P._ts(e["month"])
                if Tm in cuts:
                    c = P._ts(cut)
                    if cuts[Tm] != Tm + DAY and cuts[Tm] != c:
                        raise P.PaperError(f"{os.path.basename(vdir)}: 반복마다 c(T)가 다름 ({e['month']})")
                    cuts[Tm] = c
    return cuts


def _block(res):
    return P.metrics(res) if res is not None else dict(days=0)


def build(paper_dir=None, base_path=None, now=None, df15=None):
    """보조 기록 계산 + 쓰기. 반환 report dict (또는 등록·상태가 없으면 None). df15: 시험용 15분봉 (주면 파일 대신)"""
    paper_dir = paper_dir or P.PAPER_DIR
    base_path = base_path or P.BASE_15M
    reg = P.load_registry(paper_dir)["variants"]
    if not reg:
        return None
    if now is None:
        st = P._read_json(os.path.join(paper_dir, "status.json"))
        if not st or "now_ts" not in st:
            return None
        now = int(st["now_ts"])
    now = P._now_arg(now)
    df = P.truncate(P.load_15m_all(paper_dir, base_path) if df15 is None else df15, now)
    d0 = P.phases_for(df, 1)[0]
    mask = decision_mask(d0, STRIDE)
    o_ex, ex_info = exec_open(d0, df)
    hi = _hi(d0, now)
    lo = min(P._ts(e["start"]) for e in reg.values())
    out_dir = bench_dir(paper_dir)
    report = dict(now=P._iso(now), data_end=P._iso(int(df["ts"].iloc[-1]) + 900), start=P._date(lo),
                  ledger_through=P._date(hi), note=NOTE, exec_price=ex_info, rules={}, variants={})

    # ── 기준 규칙 ──
    sig = rule_signals(d0)
    a, b = P.window_range(d0, lo, hi)
    rule_tg = {}
    for k in RULES:
        tg = daily_hold(sig[k], mask, a, b) if b > a else np.full(d0.T, np.nan)
        frac = k in FRAC_RULES
        rule_tg[k] = (tg, frac)
        booked = run(d0, lo, hi, tg, frac, d0.o, d0.c)
        ex = run(d0, lo, hi, exec_targets(d0, tg, None), frac, o_ex, d0.c)
        folder = os.path.join(out_dir, "rules")
        _write_pair(folder, f"{k}_booked", d0, booked, d0.o, 0)
        _write_pair(folder, f"{k}_exec", d0, ex, o_ex, 1800)
        report["rules"][k] = dict(label=RULE_KO[k], booked=_block(booked), exec=_block(ex))

    # ── 등록 변형의 실행 가능 장부 ──
    for name, ent in sorted(reg.items()):
        cfg = P._cfg_from_json(ent["cfg"])
        vdir = os.path.join(paper_dir, name)
        vlo = P._ts(ent["start"])
        vmask = decision_mask(d0, int(cfg.get("stride", 1)))
        frac = P._is_frac(cfg)
        decs = [P._read_decs(vdir, r) for r in range(ent["reps"])]
        tgs = [P.rep_targets(d0, x) for x in decs]
        his = [P.rep_hi(d0, x, vlo, now, vmask) for x in decs]
        vhi = min(his) if his else vlo
        cuts = month_cuts_from_logs(vdir, ent["reps"], vlo, int(d0.ts[-1]) + 2 * DAY)
        folder = os.path.join(out_dir, "exec", name)
        reps_out = []
        ex_res = []
        for r in range(ent["reps"]):
            res = run(d0, vlo, his[r], exec_targets(d0, tgs[r], cuts), frac, o_ex, d0.c) if his[r] > vlo else None
            _write_pair(folder, f"rep{r:02d}", d0, res, o_ex, 1800)
            ex_res.append(res)
            reps_out.append(dict(ledger_through=P._date(his[r]) if res is not None else None, exec=_block(res)))
        v = dict(start=ent["start"], reps=reps_out, ledger_through=P._date(vhi) if vhi > vlo else None)
        if cfg.get("committee"):
            ctg = P.committee_targets(d0, tgs)
            res = run(d0, vlo, vhi, exec_targets(d0, ctg, cuts), False, o_ex, d0.c) if vhi > vlo else None
            _write_pair(folder, "committee", d0, res, o_ex, 1800)
            v["committee"] = dict(exec=_block(res))
        # 같은 구간(변형 시작 ~ 반복 공통 끝)의 규칙 성과 — 나란히 보기용
        v["rules_same_window"] = {}
        for k, (tg, fr) in rule_tg.items():
            bk = run(d0, vlo, vhi, tg, fr, d0.o, d0.c) if vhi > vlo else None
            ex = run(d0, vlo, vhi, exec_targets(d0, tg, None), fr, o_ex, d0.c) if vhi > vlo else None
            v["rules_same_window"][k] = dict(booked=_block(bk), exec=_block(ex))
        # 점검일: 고정 판정과 같은 구간 [시작, 점검일) — 끝이 점검일에 닿은 뒤에만
        cps = {}
        for cp in ent.get("checkpoints", []):
            T = P._ts(cp)
            if now < T:
                cps[cp] = dict(status="not_reached")
                continue
            if vhi < T:
                cps[cp] = dict(status="waiting_for_ledger", ledger_through=P._date(vhi))
                continue
            blk = dict(status="computed", reps=[], rules={})
            for r in range(ent["reps"]):
                res = run(d0, vlo, T, exec_targets(d0, tgs[r], cuts), frac, o_ex, d0.c)
                blk["reps"].append(dict(exec=_block(res)))
            if cfg.get("committee"):
                res = run(d0, vlo, T, exec_targets(d0, P.committee_targets(d0, tgs), cuts), False, o_ex, d0.c)
                blk["committee"] = dict(exec=_block(res))
            for k, (tg, fr) in rule_tg.items():
                blk["rules"][k] = dict(booked=_block(run(d0, vlo, T, tg, fr, d0.o, d0.c)),
                                       exec=_block(run(d0, vlo, T, exec_targets(d0, tg, None), fr, o_ex, d0.c)))
            cps[cp] = blk
        v["checkpoints"] = cps
        report["variants"][name] = v
    P._write_json(os.path.join(out_dir, "report.json"), report)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description="모의매매 보조 기록 (기준 규칙·실행 가능 장부, 판정에 안 씀)")
    ap.add_argument("--dir", default=None, help="모의매매 폴더 (기본 data/btc/paper)")
    ap.add_argument("--base", default=None, help="기본 15분봉 파일")
    ap.add_argument("--now", default=None, help="기준 시각 (기본: status.json 의 now_ts)")
    args = ap.parse_args(argv)
    rep = build(args.dir, args.base, args.now)
    if rep is None:
        print("등록된 변형이나 status.json 이 없어 건너뜀")
        return 0
    print(f"보조 기록: now {rep['now']}  장부 끝 {rep['ledger_through']}  시작 {rep['start']}")
    for k, v in rep["rules"].items():
        bk, ex = v["booked"], v["exec"]
        f = lambda m: "—" if not m.get("days") else f"샤프 {m['sharpe']:.2f} 배수 {m['twm']:.3f}"
        print(f"  {v['label']:<18} 등록 체결 {f(bk)} | 00:30 체결 {f(ex)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
