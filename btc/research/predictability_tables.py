# -*- coding: utf-8 -*-
"""
Markdown tables for btc/research/predictability.md, built from btc/research/predictability.json.

  python btc/research/predictability_tables.py > /some/scratch/tables.md
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from btc.features import NAMES          # noqa: E402
from btc.stats import holm              # noqa: E402

R = json.load(open(os.path.join(HERE, "predictability.json"), encoding="utf-8"))
ERAS = ["2014-2016", "2017-2020", "2021-2024", "2025-2026", "pooled"]
H4 = ["1", "6", "42", "180", "540"]
HLAB = {"1": "4h", "6": "1d", "42": "1w", "180": "1m", "540": "3m"}
HD = ["1", "7", "30", "90"]
DLAB = {"1": "1d", "7": "1w", "30": "1m", "90": "3m"}
OOS = ["2017-2020", "2021-2024", "2025-2026"]


def f1(x, k=1):
    return "–" if x is None else f"{x:+.{k}f}"


def pct(x, k=0):
    return "–" if x is None else f"{100 * x:+.{k}f}%"


def ic_cell(c, dagger=False):
    s = f"{100 * c['ic']:+.1f} ({c['t']:+.1f})"
    if dagger and c.get("p_shift") is not None and c["p_shift"] < 0.05:
        s += "†"
    return s


def section_summary():
    ic = R["ic_4h"]
    sd = R["y_sd_4h"]
    # Holm over all 110 pooled shift-null p-values
    keys = [(h, f) for h in H4 for f in NAMES]
    adj = holm([ic["pooled"][h][f]["p_shift"] for h, f in keys])
    holm_ok = {k: a < 0.05 for k, a in zip(keys, adj)}
    print("| horizon | σ(y_h) pooled / 2025-26 | indep. obs pooled | features \\|t\\|≥2 pooled | shift-null p<0.05 | Holm (110 tests) | sign-stable* | strongest pooled (IC×100, t) | edge ≈ max\\|IC\\|·σ vs 0.30% round trip |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for h in H4:
        P = ic["pooled"][h]
        n_t = sum(abs(P[f]["t"]) >= 2 for f in NAMES)
        n_p = sum(P[f]["p_shift"] < 0.05 for f in NAMES)
        n_h = sum(holm_ok[(h, f)] for f in NAMES)
        stable = 0
        for f in NAMES:
            sg = 1 if P[f]["ic"] > 0 else -1
            agree = sum(1 for e in ERAS[:4] if ic[e][h][f]["ic"] * sg > 0)
            if agree == 4 and abs(P[f]["t"]) >= 2:
                stable += 1
        top = sorted(NAMES, key=lambda f: -abs(P[f]["t"]))[:3]
        tops = ", ".join(f"{f} {100 * P[f]['ic']:+.1f} ({P[f]['t']:+.1f})" for f in top)
        best = max(abs(P[f]["ic"]) for f in NAMES)
        edge = best * sd["pooled"][h]
        print(f"| {HLAB[h]} (h={h}) | {100 * sd['pooled'][h]:.1f}% / {100 * sd['2025-2026'][h]:.1f}% | "
              f"{P['_n_indep']:.0f} | {n_t}/22 | {n_p}/22 | {n_h}/22 | {stable}/22 | {tops} | "
              f"{100 * best:.1f}% × {100 * sd['pooled'][h]:.1f}% ≈ {100 * edge:.2f}% |")
    print("\n*sign-stable = pooled \\|t\\| ≥ 2 and the same sign in all four eras.\n")


def section_pooled_ic():
    ic = R["ic_4h"]["pooled"]
    print("| feature | " + " | ".join(f"{HLAB[h]}" for h in H4) + " |")
    print("|---|" + "---|" * len(H4))
    for f in NAMES:
        print(f"| {f} | " + " | ".join(ic_cell(ic[h][f], True) for h in H4) + " |")
    print("\nIC×100 (Newey-West t, lags = h). † = circular-shift null p < 0.05 (shifts ≥ max(2h, 1 month); "
          "p = max(empirical, normal approx. from the null sd)).\n")


def section_era_tables(key="ic_4h", hs=H4, lab=HLAB):
    ic = R[key]
    for h in hs:
        print(f"\n**{lab[h]} (h = {h})** — n_indep per era: " +
              ", ".join(f"{e} {ic[e][h]['_n_indep']:.0f}" for e in ERAS))
        print("\n| feature | " + " | ".join(ERAS) + " |")
        print("|---|" + "---|" * len(ERAS))
        for f in NAMES:
            print(f"| {f} | " + " | ".join(ic_cell(ic[e][h][f], e == "pooled") for e in ERAS) + " |")


def section_family(hs=("1", "6", "42", "180", "540")):
    """Compact era view of a few representative features."""
    ic = R["ic_4h"]
    feats = ["ret_1", "clv", "rel_vol", "ret_180", "ma_200", "x_50_200", "rsi_84", "macdh_1d", "dd_180", "ma_1200"]
    for h in ("1", "42", "180"):
        print(f"\n**{HLAB[h]}** — IC×100 by era (t)\n")
        print("| feature | " + " | ".join(ERAS) + " |")
        print("|---|" + "---|" * len(ERAS))
        for f in feats:
            print(f"| {f} | " + " | ".join(ic_cell(ic[e][h][f], e == "pooled") for e in ERAS) + " |")


def section_daily_pooled():
    ic = R["ic_daily"]["pooled"]
    icd = R["ic_daily"]
    print("| feature (daily bars) | " + " | ".join(DLAB[h] for h in HD) + " | eras same sign as pooled (1m) |")
    print("|---|" + "---|" * (len(HD) + 1))
    for f in NAMES:
        sg = 1 if ic["30"][f]["ic"] > 0 else -1
        agree = sum(1 for e in ERAS[:4] if icd[e]["30"][f]["ic"] * sg > 0)
        print(f"| {f} | " + " | ".join(ic_cell(ic[h][f], True) for h in HD) + f" | {agree}/4 |")


def section_signals_ic():
    ic = R["ic_signals_4h"]
    sp = R["signal_spread_4h"]
    print("| signal | h | IC×100 pooled (t) | long-state minus flat-state fwd return, ann. (NW t): " + " / ".join(ERAS) + " |")
    print("|---|---|---|---|")
    for k in sp:
        for h in ("6", "42", "180"):
            cells = []
            for e in ERAS:
                v = sp[k][h][e]
                cells.append("–" if v is None else f"{100 * v['spread']:+.0f}% ({v['t']:+.1f})")
            print(f"| {k} | {HLAB[h]} | {ic_cell(ic['pooled'][h][k], True)} | " + " / ".join(cells) + " |")


def section_wf():
    bh = R["bh_4h"]
    print(f"Buy & hold 2017-01..2026-09: Sharpe {bh['sharpe']:.2f}, CAGR {100 * bh['cagr']:.1f}%, MDD {100 * bh['max_dd']:.0f}%. "
          "Eras (Sharpe): " + ", ".join(f"{e} {bh['eras'][e]['sharpe']:.2f}" for e in OOS) + "\n")
    print("| model | h | OOS R² 2017-26 | R² by era (17-20 / 21-24 / 25-26) | CW t | f > hurdle | 4h: net SR | gross SR | CAGR | MDD | expo | sw/yr | daily: net SR | MDD | sw/yr | ΔSR vs B&H p (daily) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k, e in R["wf_4h"].items():
        o = e["oos"]
        a, d = e["econ_4h"], e["econ_daily_decision"]
        print(f"| {e['model']} | {HLAB[str(e['h'])]} | {pct(o['2017-2026']['r2_oos'], 2)} | "
              + " / ".join(pct(o[x]['r2_oos'], 1) for x in OOS)
              + f" | {o['2017-2026']['cw_t']:+.1f} | {100 * o['2017-2026']['above_hurdle']:.0f}% | "
              f"{a['sharpe']:.2f} | {a['gross_sharpe']:.2f} | {100 * a['cagr']:.0f}% | {100 * a['max_dd']:.0f}% | "
              f"{a['exposure']:.2f} | {a['switches_per_year']:.0f} | {d['sharpe']:.2f} | {100 * d['max_dd']:.0f}% | "
              f"{d['switches_per_year']:.0f} | {d['p']:.2f} |")


def section_wf_eras():
    bh = R["bh_4h"]["eras"]
    print("| rule (daily decisions) | " + " | ".join(f"{e} SR" for e in OOS) + " |")
    print("|---|---|---|---|")
    print("| buy & hold | " + " | ".join(f"{bh[e]['sharpe']:.2f}" for e in OOS) + " |")
    for k, e in R["wf_4h"].items():
        d = e["econ_daily_decision"]["eras"]
        print(f"| {k} | " + " | ".join(f"{d[x]['sharpe']:.2f}" for x in OOS) + " |")


def section_signals_econ():
    print("| signal | decisions | net SR | gross SR | CAGR | MDD | expo | sw/yr | ΔSR (90% CI) | p | SR 17-20 / 21-24 / 25-26 | MDD 17-20 / 21-24 / 25-26 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    bh = R["bh_4h"]
    print(f"| buy & hold | – | {bh['sharpe']:.2f} | {bh['gross_sharpe']:.2f} | {100 * bh['cagr']:.0f}% | {100 * bh['max_dd']:.0f}% | 1.00 | 0 | – | – | "
          + " / ".join(f"{bh['eras'][e]['sharpe']:.2f}" for e in OOS) + " | "
          + " / ".join(f"{100 * bh['eras'][e]['max_dd']:.0f}%" for e in OOS) + " |")
    for k, e in R["signals"].items():
        for v, lab in (("econ_4h", "4h"), ("econ_daily_decision", "daily")):
            a = e[v]
            ci = a["d_sharpe_ci90"]
            print(f"| {e['label']} | {lab} | {a['sharpe']:.2f} | {a['gross_sharpe']:.2f} | {100 * a['cagr']:.0f}% | "
                  f"{100 * a['max_dd']:.0f}% | {a['exposure']:.2f} | {a['switches_per_year']:.0f} | "
                  f"{a['d_sharpe']:+.2f} ({ci[0]:+.2f}, {ci[1]:+.2f}) | {a['p']:.2f} | "
                  + " / ".join(f"{a['eras'][x]['sharpe']:.2f}" for x in OOS) + " | "
                  + " / ".join(f"{100 * a['eras'][x]['max_dd']:.0f}%" for x in OOS) + " |")
    print("\nOOS R² (%) of a walk-forward univariate regression y_h ~ 1 + signal, 2017-2026:\n")
    print("| signal | " + " | ".join(HLAB[h] for h in H4) + " |")
    print("|---|" + "---|" * len(H4))
    for k, e in R["signals"].items():
        print(f"| {k} | " + " | ".join(pct(e['oos'][h]['2017-2026']['r2_oos'], 2) for h in H4) + " |")


def section_cost():
    grid = ["0.0", "0.0005", "0.001", "0.0015", "0.0025"]
    print("| rule | " + " | ".join(f"c={100 * float(c):.2f}%" for c in grid) + " |")
    print("|---|" + "---|" * len(grid))
    print("| buy & hold | " + " | ".join(f"{R['bh_cost_curve'][c]:.2f}" for c in grid) + " |")
    for k in ("ols_h1", "ols_h6", "ridge_vs_h6", "ridge_vs_h42", "ridge_h180", "ridge_h540"):
        e = R["wf_4h"][k]
        print(f"| {k}, 4h decisions | " + " | ".join(f"{e['cost_curve_4h'][c]['sharpe']:.2f}" for c in grid) + " |")
        print(f"| {k}, daily decisions | " + " | ".join(f"{e['cost_curve_daily'][c]['sharpe']:.2f}" for c in grid) + " |")
    for k, e in R["signals"].items():
        print(f"| {k}, 4h decisions | " + " | ".join(f"{e['cost_curve_4h'][c]:.2f}" for c in grid) + " |")
    print("\n(Forecast rules: hurdle = round trip at that cost, so the rule itself changes with c.)")


def section_band():
    bands = ["0.001", "0.002", "0.003", "0.004", "0.005", "0.0075", "0.01"]
    print("| forecast | " + " | ".join(f"±{100 * float(b):.2f}%" for b in bands) + " |")
    print("|---|" + "---|" * len(bands))
    for k, row in R["band_sensitivity_4h"].items():
        print(f"| {k} | " + " | ".join(f"{row[b]['sharpe']:.2f} ({row[b]['switches_per_year']:.0f})" for b in bands) + " |")
    print("\nnet Sharpe (switches/yr), 4h decisions, 0.15% cost; enter above +band, exit below −band.")


def section_rv():
    rv = R["rv_ic_4h"]
    names = sorted(rv["42"], key=lambda f: -abs(rv["42"][f]["ic"]))[:8]
    print("| predictor | fwd realized vol 1d: IC×100 (t) | 1w: IC×100 (t) |")
    print("|---|---|---|")
    for f in names:
        print(f"| {f} | {ic_cell(rv['6'][f])} | {ic_cell(rv['42'][f])} |")


def section_daily_wf():
    bh = R["bh_daily_bars"]
    print(f"Daily bars, buy & hold Sharpe {bh['sharpe']:.2f}.\n")
    print("| model | h | OOS R² | CW t | net SR | gross SR | MDD | sw/yr |")
    print("|---|---|---|---|---|---|---|---|")
    for k, e in R["wf_daily_bars"].items():
        o, a = e["oos"]["2017-2026"], e["econ"]
        print(f"| {e['model']} | {DLAB[str(e['h_days'])]} | {pct(o['r2_oos'], 2)} | {o['cw_t']:+.1f} | {a['sharpe']:.2f} | "
              f"{a['gross_sharpe']:.2f} | {100 * a['max_dd']:.0f}% | {a['switches_per_year']:.0f} |")


def section_quint():
    q = R["quintiles_4h"]
    feats = ["ret_180", "ma_200", "x_50_200", "rsi_84", "dd_180", "macdh_1d", "vol_regime", "rel_vol"]
    for h in ("42",):
        print(f"\nMean forward {HLAB[h]} log return by feature quintile, annualized (Q1 low … Q5 high)\n")
        print("| feature | pooled 2014-26 | 2017-2020 | 2021-2024 | 2025-2026 |")
        print("|---|---|---|---|---|")
        for f in feats:
            print(f"| {f} | " + " | ".join(" ".join(f"{100 * x:+.0f}" for x in q[e][h][f])
                                          for e in ("pooled", "2017-2020", "2021-2024", "2025-2026")) + " |")


def section_stability():
    """Era view of representative feature/horizon pairs."""
    ic = R["ic_4h"]
    pairs = [("clv", "1"), ("ret_1", "1"), ("rel_vol", "1"), ("x_50_200", "6"), ("rel_vol", "6"),
             ("ma_200", "42"), ("rsi_84", "42"), ("ret_180", "42"), ("ma_50", "180"), ("macdh_1d", "180"),
             ("ma_200", "180"), ("ma_1200", "180"), ("x_300_1200", "180"), ("ma_200", "540"), ("dd_180", "540")]
    print("| feature @ horizon | " + " | ".join(ERAS) + " |")
    print("|---|" + "---|" * len(ERAS))
    for f, h in pairs:
        print(f"| {f} @ {HLAB[h]} | " + " | ".join(ic_cell(ic[e][h][f], e == "pooled") for e in ERAS) + " |")


SECTIONS = {"stability": section_stability, "summary": section_summary, "pooled": section_pooled_ic, "family": section_family,
            "daily": section_daily_pooled, "sig_ic": section_signals_ic, "wf": section_wf,
            "wf_eras": section_wf_eras, "sig_econ": section_signals_econ, "cost": section_cost,
            "band": section_band, "rv": section_rv, "daily_wf": section_daily_wf, "quint": section_quint,
            "era_4h": section_era_tables,
            "era_daily": lambda: section_era_tables("ic_daily", HD, DLAB)}

if __name__ == "__main__":
    for name in (sys.argv[1:] or SECTIONS):
        print(f"\n<!-- {name} -->\n")
        SECTIONS[name]()
