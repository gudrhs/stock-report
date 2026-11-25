# -*- coding: utf-8 -*-
"""analysis.json → 마크다운 표 (노트 작성용). python -m btc.research.p0_autopsy_tables"""
import json
import os

import numpy as np

from btc.research.p0_autopsy_common import OUT


def f(x, d=2, pct=False, sign=False):
    if x is None:
        return "–"
    if pct:
        return f"{x * 100:+.{d}f}%" if sign else f"{x * 100:.{d}f}%"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def main():
    o = json.load(open(os.path.join(OUT, "analysis.json")))
    print("## (1) Δ by year (median of 10 reps; threshold ±%.3f)" % o["threshold"])
    print("| year | mean Δ [min–max rep] | p5 / p50 / p95 | min Δ | in band | >+0.30 | <−0.30 | near ±0.30 | exposure | sells | IC(Δ,G) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in o["part1"]:
        print(f"| {r['year']} | {f(r['mean'])} [{f(r['mean_lo'])}–{f(r['mean_hi'])}] | {f(r['p05'])} / {f(r['p50'])} / {f(r['p95'])} | "
              f"{f(r['min'])} | {f(r['band'], 0, True)} | {f(r['above'], 0, True)} | {f(r['below'], 1, True)} | "
              f"{f(r['near_any'], 0, True)} (sell side {f(r['near_sell'], 1, True)}) | {f(r['exposure'], 0, True)} | {r['sells']:.0f} | {f(r['ic_G'])} |")
    print()
    print("pool drift:", {k: round(v["pool_km_x33"], 2) for k, v in o["pool_drift"].items()})
    p6 = o.get("part6")
    if p6:
        print("\n## Huber bias (per calendar year, κm units; bias×33)")
        print("| year | mean κm | median | Huber δ=2 | bias×33 | P(|κm|>2) |")
        print("|---|---|---|---|---|---|")
        for h in p6["huber_by_year"]:
            print(f"| {h['year']} | {f(h['mean'], 3, sign=True)} | {f(h['median'], 3, sign=True)} | {f(h['huber2'], 3, sign=True)} | "
                  f"{f(h['bias_x33'], 2, sign=True)} | {f(h['frac_abs_gt2'], 0, True)} |")
        print("\npool (January training weights):")
        print("| Jan | mean×33 | Huber×33 | bias×33 |")
        print("|---|---|---|---|")
        for h in p6["huber_pool"]:
            print(f"| {h['year']} | {f(h['mean_x33'], 2, sign=True)} | {f(h['huber2_x33'], 2, sign=True)} | {f(h['bias_x33'], 2, sign=True)} |")
        print("\n## Sell capacity: model applied to all 2017-2026 bars (median over models)")
        names = sorted(p6["models"])
        yrs = sorted({y for n in names for y in p6["models"][n]})
        print("| year | " + " | ".join(f"{n}: Δ<−0.30 / exposure" for n in names) + " |")
        print("|---|" + "---|" * len(names))
        for y in yrs:
            cells = []
            for n in names:
                v = p6["models"][n].get(y)
                cells.append(f"{f(v['frac_sell_zone'], 1, True)} / {f(v['exposure'], 0, True)} (n={v['n_models']})" if v else "–")
            print(f"| {y} | " + " | ".join(cells) + " |")
    print("\n## (2) gate rejections")
    rows = o["gates"]
    by = {}
    for g in rows:
        by.setdefault((g["month"], g["kind"]), []).append(g)
    print("| month | kind | reps | exposure / disagree |")
    print("|---|---|---|---|")
    for (m, k), gs in sorted(by.items()):
        vals = [g["exposure"] if k == "cold" else g["disagree"] for g in gs]
        print(f"| {m[:7]} | {k} | {','.join(str(g['rep']) for g in gs)} | {min(vals):.3f}–{max(vals):.3f} |")
    p23 = o.get("part23")
    if p23:
        print("\nrepro:", p23["repro"])
        print("\n### one-period consequence (candidate vs actual, same period)")
        print("| rep | month | kind | exp act→cand | ret act / cand / B&H | min Δ act / cand |")
        print("|---|---|---|---|---|---|")
        for x in p23["one_period"]:
            print(f"| {x['rep']} | {x['month'][:7]} | {x['kind']} | {f(x['exp_actual'], 0, True)}→{f(x['exp_candidate'], 0, True)} | "
                  f"{f(x['ret_actual'], 1, True)} / {f(x['ret_candidate'], 1, True)} / {f(x['ret_bh'], 1, True)} | "
                  f"{f(x['min_delta_actual'])} / {f(x['min_delta_candidate'])} |")
        print("\n### variants (median over available reps)")
        ref = p23["ref"]
        print("| variant | n | full Sharpe | full CAGR | full MDD | 2017-24 Sharpe | 2018 ret | 2022 ret | lockbox ret | lockbox MDD | lockbox exp |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for v, rows in p23["variants"].items():
            if not rows:
                continue
            md = lambda get: float(np.median([get(r) for r in rows]))
            print(f"| {v} | {len(rows)} | {f(md(lambda r: r['full']['sharpe']))} | {f(md(lambda r: r['full']['cagr']), 1, True)} | "
                  f"{f(md(lambda r: r['full']['max_dd']), 1, True)} | {f(md(lambda r: r['2017-2024']['sharpe']))} | "
                  f"{f(md(lambda r: r['2018']['ret']), 1, True)} | {f(md(lambda r: r['2022']['ret']), 1, True)} | "
                  f"{f(md(lambda r: r['lockbox']['ret']), 1, True)} | {f(md(lambda r: r['lockbox']['max_dd']), 1, True)} | "
                  f"{f(md(lambda r: r['lockbox']['exposure']), 0, True)} |")
        for k in ("B0", "B2"):
            x = ref[k]
            print(f"| {k} | – | {f(x['full']['sharpe'])} | {f(x['full']['cagr'], 1, True)} | {f(x['full']['max_dd'], 1, True)} | "
                  f"{f(x['2017-2024']['sharpe'])} | {f(x['2018']['ret'], 1, True)} | {f(x['2022']['ret'], 1, True)} | "
                  f"{f(x['lockbox']['ret'], 1, True)} | {f(x['lockbox']['max_dd'], 1, True)} | – |")
        print("\n### per-rep full Sharpe by variant")
        reps = sorted({r["rep"] for rows in p23["variants"].values() for r in rows})
        print("| rep | " + " | ".join(p23["variants"]) + " |")
        print("|---|" + "---|" * len(p23["variants"]))
        for rp in reps:
            cells = []
            for v, rows in p23["variants"].items():
                x = [r for r in rows if r["rep"] == rp]
                cells.append(f"{f(x[0]['full']['sharpe'])} / {f(x[0]['lockbox']['ret'], 0, True)}" if x else "–")
            print(f"| {rp} | " + " | ".join(cells) + " |")
        print("\n### frozen Jan-2025 / Jan-2026 cold models in 2025-01..2026-09")
        print("| rep | lockbox ret | Sharpe | MDD | exposure | mean Δ | min Δ | Δ<−0.30 |")
        print("|---|---|---|---|---|---|---|---|")
        for x in p23["frozen_2025_26"]:
            lb = x["lockbox"]
            print(f"| {x['rep']} | {f(lb['ret'], 1, True)} | {f(lb['sharpe'])} | {f(lb['max_dd'], 1, True)} | {f(lb['exposure'], 0, True)} | "
                  f"{f(x['mean_delta'])} | {f(x['min_delta'])} | {f(x['frac_below'], 1, True)} |")


if __name__ == "__main__":
    main()
