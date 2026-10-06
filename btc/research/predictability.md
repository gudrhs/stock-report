# Predictability in the 22 BTC chart features: where it is, and whether it beats costs

**Status: post-hoc research.** The 2025-01..2026-09 lockbox was opened on 2026-09-25 by the P0 study. This note
uses all of 2014-2026 and none of it was pre-registered, so treat every number as exploratory. It covers 110
feature×horizon IC tests per era plus about 100 trading-rule variants. Nothing here changes the P0 verdict.
Code: `btc/research/predictability.py` (about 40 s, 1 process) and `btc/research/predictability_tables.py`.
All numbers: `btc/research/predictability.json`.

## Summary

1. **The strongest and most stable effect is a 4-hour reversal, and it is too small to trade.** `clv` and
   `ret_1` have rank IC −6.7% / −5.8% with the next 4h return (NW t −11.2 / −9.4). The sign is the same in all
   four eras, and the effect survives Holm correction over all 110 pooled tests; nothing at 1d-3m does. But a 1-σ signal is worth only about
   0.10% per bar, against a 0.30% round trip. A 4h sign rule on the 22-feature 4h forecast has gross Sharpe 1.34 and
   net Sharpe **−0.53** (665 switches/yr).
2. **Trend/momentum at 1-week to 3-month horizons can clear costs (0.8-7% per trade), but it faded after 2020.**
   Pooled ICs are +5% to +17% (NW t 2.3-3.6). The effect was strong in 2014-2020, weak in 2021-2024, and
   around zero or negative for the slow features in 2025-26. Only the medium-speed trend features at the
   **1-month** horizon kept the same sign in all four eras, and their size shrank 2-3× after 2020. With 154
   independent months (51 quarters) in the sample, none of these survives a multiple-testing correction, and the slowest features fail even the uncorrected circular-shift null.
3. **Volatility is far more predictable than direction.** `ln σ` predicts forward realized volatility with
   IC ≈ 0.60-0.66, compared with |IC| ≤ 0.19 for returns.
4. **The 22-feature linear forecasts essentially never beat buy & hold after costs.** Out-of-sample R² was at most +1.1%,
   and negative in 2021-24 for every model. With decisions every 4h, 0/15 rules beat B&H's Sharpe of 1.01.
   With daily decisions, 1/15 did (1.07, p = 0.40). Most rules sit at 0.3-0.9, and the longer-horizon ones still
   take −80% to −90% drawdowns.
5. **Simple slow single signals beat the 22-feature models.** Their net Sharpe is 0.99-1.25, compared with
   B&H's 1.01, and they are barely hurt by costs. The best is the daily-MACD-histogram sign with daily
   decisions: Sharpe 1.25, max drawdown −52%, p = 0.13 against B&H. Their value comes from smaller drawdowns,
   mostly earned in 2017-2020.
6. **For the RL design:** decide once a day, target a horizon of about 1 month, feed the agent only a few
   trend features at 1-week-to-2-month speed plus a volatility state, and leave the 4h reversal and candle
   features out of the decision inputs. Judge the agent against the daily-MACD and 200-day-MA rules as well as
   B&H. The realistic ceiling is ΔSharpe of about +0.1-0.25, mostly from smaller drawdowns. That is below the
   roughly 0.6 needed to detect an edge statistically in 10 years.

## 1. Setup

- **Data.** Phase-0 4h Bitstamp bars and the existing 22 features (`load_phases()[0]`). Decisions run from
  2014-01 to 2026-09, 27,870 bars. For robustness there is also a **daily resampling** of the same bars
  (4,644 days), with all 22 features recomputed on daily bars by the same `FeatureEngine`. On daily bars,
  "ma_200" means the 200-day MA, "x_300_1200" means the 300-day/1200-day MA pair, and so on. The 1200-day
  windows are only complete from 2015-04, so the daily 2014-16 era uses partial windows for those features.
- **Target.** y_h[t] = ln(O[t+1+h]/O[t+1]), aligned to calendar time. It is NaN if the entry or exit bar
  falls inside a data gap longer than 1 day; there are only two gaps after 2014.
- **IC.** Spearman rank correlation within each era. Eras are assigned by decision time: 2014-16, 2017-20,
  2021-24, 2025-01..2026-09, and pooled. The t-stat is Newey-West (Bartlett kernel, lags = h) on the product
  of the standardized ranks. For the pooled sample only, a **circular-shift null** also absorbs feature
  persistence: every shift ≥ max(2h, 1 month) is tried, and † marks p < 0.05. "Indep. obs" = n/h.
- **Walk-forward forecasts.** An expanding window from 2014-01, refit every 1 January from 2017 to 2026.
  A training row is used only if its forward return was fully realized before the refit date. Three models,
  all using all 22 standardized features:
  - `ols`: plain OLS.
  - `ridge`: λ chosen by an inner time-ordered 70/30 split with an embargo. The grid is 0-30 on standardized
    features, and the chosen λ ranged from 0 to 30.
  - `ridge_vs`: ridge on the volatility-scaled target y/(σ√h), rescaled back to returns.
- **Forecast evaluation.** OOS R² is measured against the expanding historical mean
  (Campbell-Thompson). A Clark-West t-stat (NW, lags = h) tests the nested comparison.
- **Economics.** The rule is "long iff forecast > 0.30%" (the round trip at 0.15% one-way). Fills are at the
  next open. Accounting uses `btc.env.simulate` and `daily_marks`, through a copy of `evaluate.Window` that
  does not write to the lockbox audit log. This copy reproduces the report_full Sharpe ratios for B0/B1/B2/B5/B6
  exactly (1.0148 / 0.9947 / 1.1233 / 1.1523 / 1.0969). Decisions are made either every 4h bar or once a day,
  at the bar closing 00:00 UTC. The OOS window is 2017-01..2026-09, the same as P0.

## 2. Where the predictability is (4h bars)

| horizon | σ(y_h) pooled / 2025-26 | indep. obs pooled | features \|t\|≥2 pooled | shift-null p<0.05 | Holm (110 tests) | sign-stable* | strongest pooled (IC×100, t) | edge ≈ max\|IC\|·σ vs 0.30% round trip |
|---|---|---|---|---|---|---|---|---|---|
| 4h (h=1) | 1.4% / 0.9% | 27870 | 18/22 | 17/22 | 3/22 | 5/22 | clv -6.7 (-11.2), ret_1 -5.8 (-9.4), x_50_200 +2.6 (+4.4) | 6.7% × 1.4% ≈ 0.10% |
| 1d (h=6) | 3.5% / 2.3% | 4643 | 12/22 | 8/22 | 0/22 | 0/22 | x_50_200 +4.6 (+3.6), rsi_84 +4.4 (+3.5), ma_200 +4.4 (+3.5) | 4.6% × 3.5% ≈ 0.16% |
| 1w (h=42) | 9.3% / 5.8% | 662 | 10/22 | 7/22 | 0/22 | 4/22 | rsi_84 +9.1 (+3.0), ret_180 +8.6 (+2.9), ma_200 +8.7 (+2.8) | 9.1% × 9.3% ≈ 0.85% |
| 1m (h=180) | 20.5% / 12.6% | 154 | 12/22 | 10/22 | 0/22 | 8/22 | ret_1 +2.1 (+3.2), ma_20 +7.6 (+3.2), ret_6 +4.8 (+3.1) | 12.7% × 20.5% ≈ 2.61% |
| 3m (h=540) | 38.6% / 19.1% | 51 | 15/22 | 9/22 | 0/22 | 0/22 | ret_1 +2.3 (+3.0), ma_20 +8.2 (+3.0), clv +3.4 (+2.9) | 18.7% × 38.6% ≈ 7.23% |

*sign-stable = pooled \|t\| ≥ 2 and the same sign in all four eras.

Pooled rank IC for 2014-01..2026-09, all 22 features:

| feature | 4h | 1d | 1w | 1m | 3m |
|---|---|---|---|---|---|
| ret_1 | -5.8 (-9.4)† | -0.1 (-0.2) | +1.4 (+2.2)† | +2.1 (+3.2)† | +2.3 (+3.0)† |
| ret_6 | -1.8 (-3.0)† | -1.7 (-1.6) | +3.0 (+2.0)† | +4.8 (+3.1)† | +5.2 (+2.9)† |
| ret_42 | +1.7 (+2.7)† | +3.0 (+2.5)† | +4.2 (+1.5) | +10.3 (+2.9)† | +11.1 (+2.8)† |
| ret_180 | +2.2 (+3.8)† | +4.3 (+3.5)† | +8.6 (+2.9)† | +9.5 (+1.7) | +15.2 (+2.2) |
| ma_20 | -1.1 (-1.8) | +0.3 (+0.3) | +4.0 (+1.9) | +7.6 (+3.2)† | +8.2 (+3.0)† |
| ma_50 | +0.7 (+1.1) | +2.2 (+1.8) | +4.7 (+1.7) | +10.3 (+3.0)† | +11.2 (+2.9)† |
| ma_200 | +2.1 (+3.5)† | +4.4 (+3.5)† | +8.7 (+2.8)† | +12.0 (+2.3) | +16.2 (+2.4) |
| ma_1200 | +2.2 (+3.9)† | +4.2 (+3.4) | +5.3 (+1.6) | +7.7 (+1.2) | +18.7 (+1.7) |
| x_50_200 | +2.6 (+4.4)† | +4.6 (+3.6)† | +7.9 (+2.6) | +9.3 (+1.7) | +15.1 (+2.2) |
| x_300_1200 | +1.6 (+2.9) | +2.9 (+2.4) | +2.2 (+0.7) | +4.4 (+0.7) | +15.1 (+1.3) |
| rsi_14 | -0.4 (-0.6) | +1.3 (+1.1) | +4.7 (+1.9) | +9.2 (+3.0)† | +10.0 (+2.9)† |
| rsi_84 | +2.0 (+3.3)† | +4.4 (+3.5)† | +9.1 (+3.0)† | +12.7 (+2.5) | +16.8 (+2.5) |
| macd_4h | +1.3 (+2.1)† | +2.0 (+1.7) | +4.5 (+1.7) | +10.2 (+2.9)† | +11.3 (+2.9)† |
| macdh_4h | -1.4 (-2.3)† | -1.8 (-1.5) | +1.7 (+1.0) | +1.4 (+1.3) | +1.3 (+1.7) |
| macdh_1d | +1.3 (+2.2)† | +1.7 (+1.4) | +5.2 (+1.8) | +10.2 (+2.4)† | +6.4 (+1.8)† |
| vol_regime | +1.7 (+2.8)† | +3.1 (+2.5) | +6.1 (+2.1) | +7.6 (+1.4) | +4.8 (+0.7) |
| rel_vol | +2.2 (+3.8)† | +3.1 (+3.4)† | +3.9 (+2.6)† | +2.4 (+1.2) | +4.1 (+2.2) |
| vol_trend | +1.7 (+2.8)† | +3.6 (+3.0)† | +4.9 (+1.9) | +3.1 (+0.9) | +6.1 (+2.1) |
| flow_20 | +0.2 (+0.3) | +1.3 (+1.2) | +3.8 (+1.7) | +7.0 (+2.9)† | +5.9 (+2.3) |
| range | +1.9 (+3.3)† | +1.7 (+2.1) | +1.3 (+1.1) | -0.4 (-0.3) | +0.0 (+0.0) |
| clv | -6.7 (-11.2)† | -0.2 (-0.3) | +1.8 (+2.7)† | +2.1 (+2.5)† | +3.4 (+2.9)† |
| dd_180 | +1.6 (+2.7)† | +3.5 (+2.8)† | +7.6 (+2.5) | +9.3 (+1.8) | +12.3 (+1.8) |

IC×100 (Newey-West t, lags = h). † = circular-shift null p < 0.05 (shifts ≥ max(2h, 1 month); p = max(empirical, normal approx. from the null sd)).

Era stability for representative feature/horizon pairs (IC×100, NW t). The full per-era tables are in
Appendix A.

| feature @ horizon | 2014-2016 | 2017-2020 | 2021-2024 | 2025-2026 | pooled |
|---|---|---|---|---|---|
| clv @ 4h | -6.4 (-5.3) | -6.0 (-5.5) | -9.0 (-8.2) | -3.7 (-2.4) | -6.7 (-11.2)† |
| ret_1 @ 4h | -6.8 (-5.2) | -5.3 (-4.8) | -6.8 (-6.0) | -3.1 (-1.9) | -5.8 (-9.4)† |
| rel_vol @ 4h | +5.1 (+4.1) | +1.5 (+1.4) | +1.1 (+1.1) | +0.8 (+0.5) | +2.2 (+3.8)† |
| x_50_200 @ 1d | +6.8 (+2.6) | +7.2 (+3.2) | +1.0 (+0.5) | -0.1 (-0.0) | +4.6 (+3.6)† |
| rel_vol @ 1d | +4.0 (+2.0) | +3.3 (+2.1) | +3.9 (+2.5) | -2.3 (-0.9) | +3.1 (+3.4)† |
| ma_200 @ 1w | +13.2 (+2.0) | +11.2 (+2.0) | +2.6 (+0.5) | +2.4 (+0.3) | +8.7 (+2.8)† |
| rsi_84 @ 1w | +13.4 (+2.0) | +12.7 (+2.3) | +3.0 (+0.6) | +0.4 (+0.0) | +9.1 (+3.0)† |
| ret_180 @ 1w | +8.9 (+1.4) | +8.7 (+1.6) | +7.1 (+1.4) | +3.1 (+0.4) | +8.6 (+2.9)† |
| ma_50 @ 1m | +12.7 (+1.9) | +12.6 (+2.0) | +5.2 (+0.9) | +4.6 (+0.6) | +10.3 (+3.0)† |
| macdh_1d @ 1m | +12.5 (+1.4) | +8.6 (+1.1) | +6.1 (+0.8) | +16.3 (+1.6) | +10.2 (+2.4)† |
| ma_200 @ 1m | +15.7 (+1.6) | +14.6 (+1.6) | +9.5 (+1.1) | +1.4 (+0.1) | +12.0 (+2.3) |
| ma_1200 @ 1m | +0.6 (+0.0) | +21.6 (+1.9) | +2.8 (+0.2) | -19.2 (-1.3) | +7.7 (+1.2) |
| x_300_1200 @ 1m | -4.3 (-0.3) | +20.2 (+1.8) | -1.6 (-0.1) | -17.2 (-1.0) | +4.4 (+0.7) |
| ma_200 @ 3m | +9.2 (+0.6) | +30.8 (+3.4) | +4.6 (+0.5) | -12.2 (-0.6) | +16.2 (+2.4) |
| dd_180 @ 3m | +7.5 (+0.6) | +29.5 (+3.1) | -2.9 (-0.3) | -8.1 (-0.4) | +12.3 (+1.8) |

What the tables show:

- **4h horizon: microstructure-like reversal.** `clv` (−6.7) and `ret_1` (−5.8) dominate, and `macdh_4h`
  and `ret_6` point the same way. Every era has the same sign with |t| ≥ 1.9. Volume and range features
  (`rel_vol` +2.2, `range` +1.9, `vol_trend` +1.7) are positive: high-activity bars are followed by slightly
  higher returns. This is the most robust structure in the data. But σ(4h) is only 1.4% (0.9% in 2025-26),
  so the edge is about 0.1% per bar.
- **1d horizon.** Nothing is sign-stable. Trend features (x_50_200, rsi_84, ma_200: IC ≈ +4.5) are positive
  in 2014-2020 and ≈ 0 afterwards. The implied edge of about 0.16% per trade is still below the round trip.
- **1w-1m horizons: the trend premium.** Price-vs-MA, RSI84, MACD, 1-month return and drawdown-from-high have
  IC +8 to +13 (pooled t 2.3-3.0), 2-3× the IC at 1d. That is enough to clear costs on paper
  (0.85-2.6% per trade vs 0.30%). But:
  - The premium is concentrated in 2014-2020.
  - At 1w, the 2021-24 ICs drop to −2 to +7, and the 2025-26 ICs to −7 to +4 (about 0 on average).
  - At **1m**, 8 medium-speed features (ma_20, ma_50, ma_200, ret_42, rsi_14, macd_4h, macdh_1d, ret_1)
    stay positive in all four eras. This is the most era-consistent tradable-horizon signal, but its size in
    2021-26 is mostly +1 to +10 with |t| ≤ 1.6.
  - The slowest features (`ma_1200`, `x_300_1200`, `ret_180`, and `dd_180` at 3m) turned **negative** at
    1m-3m in 2025-26. For example, `ma_1200` @1m was −19.2 in 2025-26, against +21.6 in 2017-20.
- **3m horizon.** Pooled ICs reach +15 to +19, but they come from 51 independent quarters, mostly the
  2017-2020 cycle. No feature is sign-stable, and the shift-null p of the slow features is 0.05-0.5.
  Statistically this is roughly one bull/bear cycle.
- **Multiple testing.** With Holm correction across the 110 pooled shift-null tests, only 3 features survive,
  all at 4h: `clv`, `ret_1` and `x_50_200`. Nothing survives at 1d-3m. (The shift-null p is conservative: the larger of the empirical p and a normal approximation, because neighbouring shifts are highly correlated.)

Conditional (quintile) view:
Mean forward 1w log return by feature quintile, annualized (Q1 low … Q5 high)

| feature | pooled 2014-26 | 2017-2020 | 2021-2024 | 2025-2026 |
|---|---|---|---|---|
| ret_180 | +3 -35 +20 +102 +100 | +29 +0 +24 +258 +125 | -6 -28 +34 +82 +55 | -58 -19 +39 -15 +5 |
| ma_200 | -5 -20 +8 +75 +133 | +16 -61 +66 +261 +154 | +32 -13 +34 +23 +61 | -83 +15 +21 +7 -7 |
| x_50_200 | -19 +18 -1 +80 +112 | -15 -13 +97 +211 +156 | -2 +74 -6 +6 +65 | -66 -55 +75 +12 -13 |
| rsi_84 | +9 -22 +2 +36 +165 | +37 -71 +57 +203 +212 | +41 -13 +17 +6 +85 | -64 +6 +5 +20 -15 |
| dd_180 | +8 -15 +9 +49 +139 | -5 -27 +47 +155 +266 | +54 +11 +33 +16 +22 | -64 -29 +48 +5 -8 |
| macdh_1d | +30 -23 +8 +114 +61 | +31 -76 +123 +259 +99 | +74 +64 -32 -14 +45 | -58 +20 -33 +56 -33 |
| vol_regime | +5 +6 +37 +84 +57 | +27 +62 +71 +159 +117 | -11 -24 +40 +88 +44 | +43 -28 +32 -55 -40 |
| rel_vol | +20 +20 +34 +51 +64 | +59 +72 +79 +106 +120 | +8 +13 +18 +35 +63 | +33 -23 -18 -12 -27 |

The trend payoff is **asymmetric and concentrated in the top quintile**, i.e. strong up-trends continue. In
2017-2020 the low quintiles still had positive returns, so going flat mostly gave up upside. In 2025-26 the
ordering is flat or reversed.

## 3. Robustness: daily bars (features recomputed on daily bars)

| feature (daily bars) | 1d | 1w | 1m | 3m | eras same sign as pooled (1m) |
|---|---|---|---|---|---|
| ret_1 | -3.3 (-2.2)† | +1.8 (+1.1) | +4.2 (+2.6)† | +5.6 (+3.2)† | 3/4 |
| ret_6 | +1.0 (+0.7) | +4.3 (+1.5) | +9.7 (+2.9)† | +10.0 (+2.7)† | 4/4 |
| ret_42 | +4.9 (+3.2)† | +8.3 (+2.6) | +10.2 (+1.8) | +17.9 (+2.3) | 3/4 |
| ret_180 | +2.3 (+1.6) | +2.5 (+0.8) | +7.0 (+1.0) | +17.3 (+1.5) | 2/4 |
| ma_20 | +2.9 (+1.9) | +7.8 (+2.5)† | +12.7 (+2.7)† | +15.1 (+2.7)† | 4/4 |
| ma_50 | +4.4 (+2.9)† | +9.5 (+2.9)† | +12.4 (+2.2) | +18.2 (+2.4) | 3/4 |
| ma_200 | +3.9 (+2.6) | +6.1 (+1.8) | +9.2 (+1.4) | +19.1 (+1.7) | 3/4 |
| ma_1200 | -0.5 (-0.4) | -4.5 (-1.4) | -12.1 (-1.9) | -17.3 (-1.6) | 4/4 |
| x_50_200 | +2.8 (+1.9) | +2.2 (+0.6) | +4.4 (+0.6) | +13.5 (+1.2) | 2/4 |
| x_300_1200 | -4.3 (-3.1) | -11.2 (-3.6) | -24.4 (-3.8) | -39.4 (-4.0) | 4/4 |
| rsi_14 | +3.7 (+2.5)† | +9.1 (+2.9)† | +13.2 (+2.6)† | +17.3 (+2.5)† | 3/4 |
| rsi_84 | +4.5 (+2.9) | +7.9 (+2.4) | +11.6 (+1.9) | +18.8 (+1.7) | 3/4 |
| macd_4h | +4.9 (+3.2)† | +8.9 (+2.8)† | +11.6 (+2.1) | +17.7 (+2.3) | 3/4 |
| macdh_4h | +1.2 (+0.8) | +5.5 (+1.8) | +9.8 (+2.3)† | +6.3 (+1.8)† | 4/4 |
| macdh_1d | +3.5 (+2.3) | +4.5 (+1.4) | +4.8 (+0.8) | +12.5 (+1.4) | 2/4 |
| vol_regime | +2.7 (+1.9) | +5.1 (+1.6) | +9.4 (+1.4) | +6.8 (+0.6) | 3/4 |
| rel_vol | +4.7 (+3.3)† | +7.1 (+3.0)† | +8.6 (+2.2) | +11.7 (+1.8) | 3/4 |
| vol_trend | +3.6 (+2.5)† | +5.7 (+1.8) | +9.9 (+1.6) | +14.6 (+1.6) | 2/4 |
| flow_20 | +3.1 (+2.1)† | +7.5 (+2.4) | +15.5 (+2.9)† | +18.4 (+3.1)† | 4/4 |
| range | +1.9 (+1.3) | +2.4 (+1.2) | -2.6 (-0.8) | -4.3 (-1.1) | 3/4 |
| clv | -2.5 (-1.6) | +2.7 (+1.8) | +4.6 (+2.7)† | +7.6 (+3.2)† | 3/4 |
| dd_180 | +4.0 (+2.7) | +7.0 (+2.1) | +13.1 (+2.0) | +23.5 (+2.3) | 2/4 |

The daily resampling tells the same story. Trend/oscillator features (`ma_20`/`ma_50` daily ≈ 4h ma_120/ma_300,
`rsi_14` daily, `macd` daily) have IC +8 to +13 at 1w-1m. The effect is strong in 2014-20, weaker in 2021-24
and ≈ 0 or negative in 2025-26 (Appendix: `ic_daily` in the JSON).

Two things are new on daily bars:

1. **Daily `flow_20`** (20-day up-volume minus down-volume) is positive at 1m in all four eras and at 3m in
   three of four: pooled +15.5 (t 2.9) and +18.4 (t 3.1), shift p 0.014 and 0.020.
2. The ultra-slow **daily `x_300_1200`** (300-day vs 1200-day MA) is **negative in all four eras** at every
   horizon: pooled −24 (NW t −3.8) at 1m and −39 (NW t −4.0) at 3m. But once its persistence is accounted for, the shift-null p is only 0.06-0.07. This would be a
   multi-year mean-reversion signal ("far above the 3-year trend → lower forward returns"), resembling the
   halving cycle. The sample holds about 3 such cycles and the 2014-15 windows are partial, so it is a
   hypothesis only.

## 4. Is it enough to beat costs?

### 4.1 Walk-forward 22-feature forecasts (OOS 2017-01..2026-09, cost 0.15%, hurdle 0.30%)

Buy & hold 2017-01..2026-09: Sharpe 1.01, CAGR 58.2%, MDD -83%. Eras (Sharpe): 2017-2020 1.45, 2021-2024 0.78, 2025-2026 0.08

| model | h | OOS R² 2017-26 | R² by era (17-20 / 21-24 / 25-26) | CW t | f > hurdle | 4h: net SR | gross SR | CAGR | MDD | expo | sw/yr | daily: net SR | MDD | sw/yr | ΔSR vs B&H p (daily) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ols | 4h | -0.18% | -0.2% / -0.1% / -0.6% | +4.0 | 2% | 0.17 | 0.69 | 1% | -32% | 0.02 | 46 | -0.05 | -37% | 13 | 0.99 |
| ols | 1d | -0.35% | +0.0% / -0.8% / -1.0% | +3.5 | 31% | 0.48 | 1.53 | 12% | -63% | 0.31 | 303 | 0.81 | -67% | 88 | 0.75 |
| ols | 1w | -1.45% | -1.3% / -1.3% / -3.2% | +1.7 | 63% | 0.32 | 0.96 | 2% | -92% | 0.63 | 239 | 0.75 | -78% | 66 | 0.92 |
| ols | 1m | -2.62% | -3.0% / -1.4% / -3.4% | +0.8 | 77% | 0.60 | 0.90 | 19% | -81% | 0.77 | 119 | 0.65 | -82% | 35 | 1.00 |
| ols | 3m | -5.41% | -5.1% / -7.9% / +3.1% | +0.5 | 81% | 0.52 | 0.80 | 13% | -88% | 0.82 | 116 | 0.74 | -78% | 33 | 0.98 |
| ridge | 4h | -0.07% | -0.1% / -0.0% / -0.1% | +2.1 | 1% | -0.49 | -0.23 | -4% | -32% | 0.01 | 13 | -0.36 | -19% | 3 | 1.00 |
| ridge | 1d | +0.13% | +0.3% / -0.0% / -0.1% | +2.9 | 14% | 0.32 | 1.01 | 5% | -53% | 0.14 | 141 | 0.74 | -64% | 36 | 0.80 |
| ridge | 1w | +0.40% | +1.0% / -0.7% / +0.2% | +2.1 | 79% | 0.73 | 1.04 | 29% | -88% | 0.79 | 116 | 0.75 | -82% | 38 | 0.96 |
| ridge | 1m | +0.65% | +1.2% / -0.6% / +0.4% | +1.8 | 90% | 0.78 | 0.91 | 34% | -83% | 0.90 | 53 | 0.79 | -82% | 17 | 0.99 |
| ridge | 3m | -4.79% | -6.4% / -1.7% / +1.1% | +0.2 | 92% | 0.86 | 0.93 | 41% | -87% | 0.92 | 32 | 0.92 | -85% | 10 | 0.89 |
| ridge_vs | 4h | -0.53% | -0.8% / -0.2% / -0.1% | +0.9 | 3% | -0.19 | 0.16 | -5% | -58% | 0.03 | 42 | 0.10 | -42% | 13 | 0.99 |
| ridge_vs | 1d | -1.05% | -1.5% / -0.4% / -0.3% | +1.4 | 22% | 0.59 | 1.14 | 17% | -54% | 0.22 | 147 | 1.07 | -43% | 42 | 0.40 |
| ridge_vs | 1w | +1.11% | +2.3% / -0.8% / +0.1% | +2.5 | 81% | 0.78 | 1.10 | 34% | -91% | 0.81 | 125 | 0.98 | -79% | 42 | 0.58 |
| ridge_vs | 1m | -4.95% | -5.8% / -4.2% / +0.4% | +0.8 | 90% | 0.85 | 0.98 | 40% | -88% | 0.90 | 56 | 0.96 | -85% | 16 | 0.70 |
| ridge_vs | 3m | -8.62% | -5.6% / -17.8% / -1.8% | +0.6 | 92% | 0.89 | 0.95 | 45% | -90% | 0.93 | 27 | 0.90 | -89% | 10 | 0.93 |

"f > hurdle" is the share of OOS bars where the rule is long. The p value comes from a one-sided paired
stationary bootstrap of ΔSharpe against B&H, daily-decision version.

- **Forecast accuracy is essentially nil.** The best OOS R² is +1.1% (ridge_vs 1w, CW t 2.5), all of it from
  2017-20. **Every** model and horizon has negative OOS R² in 2021-24. OLS is negative at every horizon. At
  1m and 3m, most models are worse than the historical mean.
  - The Clark-West t is positive at 4h and 1d (up to +4.0). This is the reversal again: the forecasts carry
    information, but too little to lower the squared error.
- **Economics.**
  - With 4h decisions, no rule reaches B&H (best 0.89).
  - With daily decisions, one of 15 does: `ridge_vs` 1d, Sharpe 1.07, MDD −43%, p = 0.40. It is the best of
    ~100 variants, so read it as selection, not evidence.
  - Long-horizon rules stay 77-93% long but have drawdowns of −80% to −92%. They do not avoid the crashes,
    and they sit out part of the rallies.
- **Costs are the binding constraint at short horizons.** At 1d, the gross Sharpe is 1.0-1.5 but the net is
  0.3-0.6. The rule makes 140-300 switches a year, and at 0.15% each that costs 20-45% a year.
- **Daily decisions help 14 of the 15 rules.** Turnover drops up to 3-4×, and net Sharpe rises: ols 1d 0.48 → 0.81,
  ols 1w 0.32 → 0.75, ridge_vs 1d 0.59 → 1.07.

Net Sharpe by era, daily decisions:

| rule (daily decisions) | 2017-2020 SR | 2021-2024 SR | 2025-2026 SR |
|---|---|---|---|
| buy & hold | 1.45 | 0.78 | 0.08 |
| ols_h1 | -0.16 | 0.23 | 0.00 |
| ols_h6 | 1.16 | 0.69 | -0.28 |
| ols_h42 | 1.24 | 0.46 | -0.25 |
| ols_h180 | 1.00 | 0.55 | -0.30 |
| ols_h540 | 1.04 | 0.73 | -0.29 |
| ridge_h1 | -0.57 | 0.00 | 0.00 |
| ridge_h6 | 0.92 | 0.77 | -0.86 |
| ridge_h42 | 1.12 | 0.60 | -0.08 |
| ridge_h180 | 1.12 | 0.70 | -0.08 |
| ridge_h540 | 1.35 | 0.68 | 0.08 |
| ridge_vs_h1 | 0.08 | 0.56 | 0.00 |
| ridge_vs_h6 | 1.63 | 0.61 | -0.81 |
| ridge_vs_h42 | 1.55 | 0.62 | 0.10 |
| ridge_vs_h180 | 1.60 | 0.51 | 0.08 |
| ridge_vs_h540 | 1.24 | 0.76 | 0.08 |

### 4.2 Single well-known signals (no fitting)

| signal | decisions | net SR | gross SR | CAGR | MDD | expo | sw/yr | ΔSR (90% CI) | p | SR 17-20 / 21-24 / 25-26 | MDD 17-20 / 21-24 / 25-26 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| buy & hold | – | 1.01 | 1.02 | 58% | -83% | 1.00 | 0 | – | – | 1.45 / 0.78 / 0.08 | -83% / -77% / -53% |
| C > SMA1200 (~200d MA) | 4h | 1.12 | 1.16 | 58% | -67% | 0.58 | 14 | +0.11 (-0.24, +0.47) | 0.31 | 1.61 / 0.80 / 0.15 | -67% / -63% / -36% |
| C > SMA1200 (~200d MA) | daily | 1.10 | 1.12 | 56% | -67% | 0.58 | 7 | +0.08 (-0.26, +0.45) | 0.36 | 1.58 / 0.78 / 0.14 | -67% / -63% / -32% |
| 1-month momentum > 0 | 4h | 1.10 | 1.31 | 52% | -76% | 0.56 | 69 | +0.08 (-0.31, +0.48) | 0.36 | 1.39 / 1.10 / -0.04 | -76% / -37% / -32% |
| 1-month momentum > 0 | daily | 1.12 | 1.21 | 53% | -76% | 0.56 | 31 | +0.10 (-0.28, +0.49) | 0.32 | 1.48 / 1.01 / 0.05 | -76% / -49% / -33% |
| daily MACD hist > 0 | 4h | 1.15 | 1.23 | 54% | -64% | 0.52 | 25 | +0.14 (-0.23, +0.49) | 0.26 | 1.80 / 0.63 / 0.22 | -52% / -64% / -24% |
| daily MACD hist > 0 | daily | 1.25 | 1.33 | 61% | -52% | 0.53 | 24 | +0.24 (-0.11, +0.60) | 0.13 | 1.96 / 0.88 / -0.39 | -47% / -52% / -34% |
| SMA300 > SMA1200 (~50/200d cross) | 4h | 0.99 | 1.00 | 50% | -72% | 0.58 | 2 | -0.02 (-0.35, +0.33) | 0.55 | 1.48 / 0.77 / -0.44 | -72% / -59% / -41% |
| SMA300 > SMA1200 (~50/200d cross) | daily | 1.02 | 1.03 | 52% | -70% | 0.58 | 2 | +0.01 (-0.32, +0.35) | 0.50 | 1.52 / 0.76 / -0.37 | -70% / -56% / -39% |

OOS R² (%) of a walk-forward univariate regression y_h ~ 1 + signal, 2017-2026:

| signal | 4h | 1d | 1w | 1m | 3m |
|---|---|---|---|---|---|
| sma1200 | +0.02% | +0.09% | +0.35% | +0.30% | +1.79% |
| mom1m | +0.03% | +0.18% | +0.96% | +1.54% | +1.99% |
| macd1d | +0.01% | +0.09% | +0.23% | -0.22% | +0.33% |
| x300_1200 | -0.00% | -0.02% | -0.30% | -1.16% | +0.20% |

Conditional forward-return spread of each signal (long-state mean minus flat-state mean, annualized; NW t):

| signal | h | IC×100 pooled (t) | long-state minus flat-state fwd return, ann. (NW t): 2014-2016 / 2017-2020 / 2021-2024 / 2025-2026 / pooled |
|---|---|---|---|
| sma1200 | 1d | +4.4 (+3.7) | +71% (+1.1) / +133% (+1.9) / +49% (+0.9) / +21% (+0.4) / +80% (+2.5) |
| sma1200 | 1w | +5.9 (+1.9) | +25% (+0.5) / +136% (+2.1) / +39% (+0.8) / +9% (+0.2) / +66% (+2.2) |
| sma1200 | 1m | +7.6 (+1.2) | +14% (+0.3) / +122% (+1.8) / +25% (+0.5) / -17% (-0.4) / +53% (+1.7) |
| mom1m | 1d | +4.5 (+3.8)† | +129% (+2.0) / +124% (+1.8) / +100% (+2.0) / +43% (+0.8) / +109% (+3.4) |
| mom1m | 1w | +8.1 (+2.9)† | +99% (+2.0) / +124% (+1.9) / +70% (+1.6) / +20% (+0.5) / +90% (+3.2) |
| mom1m | 1m | +9.1 (+1.8) | +65% (+1.5) / +57% (+1.0) / +61% (+1.6) / -7% (-0.2) / +56% (+2.2) |
| macd1d | 1d | +3.2 (+2.7)† | +187% (+3.0) / +188% (+2.7) / +42% (+0.8) / +11% (+0.2) / +119% (+3.8) |
| macd1d | 1w | +8.3 (+3.0)† | +119% (+2.4) / +196% (+3.3) / -36% (-0.8) / +14% (+0.3) / +82% (+3.1) |
| macd1d | 1m | +10.1 (+2.5)† | +59% (+1.5) / +40% (+0.9) / +6% (+0.2) / +26% (+1.0) / +34% (+1.7) |
| x300_1200 | 1d | +2.6 (+2.2) | -14% (-0.2) / +120% (+1.8) / +36% (+0.7) / -51% (-0.9) / +42% (+1.3) |
| x300_1200 | 1w | +2.9 (+0.9) | -33% (-0.6) / +130% (+2.0) / +33% (+0.6) / -58% (-1.1) / +40% (+1.3) |
| x300_1200 | 1m | +3.5 (+0.5) | -28% (-0.5) / +148% (+2.2) / -3% (-0.0) / -44% (-0.9) / +38% (+1.2) |

- Three of the four simple rules beat B&H's Sharpe at 0.15% cost (**1.10-1.25 vs 1.01**), and the 50/200 cross
  matches it (0.99-1.02). None is significant (p 0.13-0.55). Their edge is **drawdown**: −52% to −76% against −83%.
- **1-month momentum** has the most era-stable spread: +70% to +130% annualized in 2014-16, 2017-20 and 2021-24,
  then +20% to +43% in 2025-26. It is the only rule that clearly beat B&H in 2021-24 (Sharpe 1.10 vs 0.78,
  MDD −37% vs −77%). It is also the most cost-sensitive (gross 1.31 → net 1.10 at 69 switches/yr).
  Deciding once a day halves its turnover.
- **Daily MACD** is best overall with daily decisions: 1.25, MDD −52%. But its 1w spread was negative in 2021-24
  (−36%, t −0.8), and it lost in 2025-26 (−0.39 vs 0.08). The 50/200-day cross (SMA300 > SMA1200) adds nothing (0.99).
- The single-signal OOS R² (+0.3 to +2.0% at 1w-3m for momentum and the 200-day MA) is **as good as or better
  than the 22-feature ridge**. With ~150 independent months, the error from estimating 22 coefficients
  outweighs the extra information.

### 4.3 Cost sensitivity and no-trade bands

Net Sharpe against one-way cost (forecast rules re-derive their hurdle at each cost):

| rule | c=0.00% | c=0.05% | c=0.10% | c=0.15% | c=0.25% |
|---|---|---|---|---|---|
| buy & hold | 1.02 | 1.02 | 1.01 | 1.01 | 1.01 |
| ols_h1, 4h decisions | 1.34 | 0.64 | 0.57 | 0.17 | -0.14 |
| ols_h1, daily decisions | 0.89 | 0.26 | 0.16 | -0.05 | 0.64 |
| ols_h6, 4h decisions | 1.23 | 0.83 | 0.59 | 0.48 | -0.10 |
| ols_h6, daily decisions | 0.95 | 0.92 | 0.73 | 0.81 | 0.68 |
| ridge_vs_h6, 4h decisions | 1.08 | 0.84 | 0.68 | 0.59 | -0.04 |
| ridge_vs_h6, daily decisions | 0.99 | 1.04 | 1.21 | 1.07 | 0.64 |
| ridge_vs_h42, 4h decisions | 1.13 | 1.00 | 0.88 | 0.78 | 0.30 |
| ridge_vs_h42, daily decisions | 1.05 | 1.06 | 0.98 | 0.98 | 0.63 |
| ridge_h180, 4h decisions | 0.97 | 0.90 | 0.84 | 0.78 | 0.66 |
| ridge_h180, daily decisions | 0.90 | 0.89 | 0.84 | 0.79 | 0.75 |
| ridge_h540, 4h decisions | 1.00 | 0.98 | 0.93 | 0.86 | 0.82 |
| ridge_h540, daily decisions | 0.98 | 0.98 | 0.95 | 0.92 | 0.92 |
| sma1200, 4h decisions | 1.16 | 1.15 | 1.14 | 1.12 | 1.10 |
| mom1m, 4h decisions | 1.31 | 1.24 | 1.17 | 1.10 | 0.96 |
| macd1d, 4h decisions | 1.23 | 1.21 | 1.18 | 1.15 | 1.10 |
| x300_1200, 4h decisions | 1.00 | 1.00 | 1.00 | 0.99 | 0.99 |

(Forecast rules: hurdle = round trip at that cost, so the rule itself changes with c.)

No-trade band (enter above +band, exit below −band) on the same forecasts, 4h decisions, 0.15% cost:

| forecast | ±0.10% | ±0.20% | ±0.30% | ±0.40% | ±0.50% | ±0.75% | ±1.00% |
|---|---|---|---|---|---|---|---|
| ols_h1 | 0.67 (136) | 1.16 (32) | 1.19 (7) | 1.00 (1) | 0.97 (0) | 1.01 (0) | 0.00 (0) |
| ols_h6 | 0.71 (225) | 1.04 (136) | 1.07 (83) | 1.01 (52) | 1.04 (34) | 1.15 (13) | 0.87 (5) |
| ridge_h6 | 0.89 (61) | 1.09 (33) | 1.15 (21) | 1.23 (14) | 1.21 (9) | 0.80 (5) | 0.01 (2) |
| ridge_vs_h42 | 0.97 (64) | 0.98 (46) | 1.00 (36) | 1.02 (29) | 0.91 (26) | 0.99 (17) | 1.05 (12) |
| ridge_h180 | 0.86 (27) | 0.83 (21) | 0.87 (16) | 0.87 (13) | 0.86 (11) | 0.86 (8) | 0.94 (6) |

net Sharpe (switches/yr), 4h decisions, 0.15% cost; enter above +band, exit below −band.

- The slow single signals lose 0.01-0.35 Sharpe going from zero cost to 0.25%. The short-horizon forecasts
  lose 1.1-1.5. Every forecast rule with more than 100 switches a year lost 0.3-1.0 Sharpe to costs at 0.15%.
  The rules that hold up make about 30 switches a year or fewer, i.e. an average holding of 1-2 weeks or more.
- A band of ±0.3% (the RL agent's optimal cost band works the same way) cuts turnover 3-7× relative to the
  hurdle rule. It brings the 4h and 1d forecasts to a net Sharpe of 1.07-1.19. But the result is **knife-edge in band width**: `ols_h1`
  gives 0.67 / 1.16 / 1.19 / 1.00 at ±0.1 / 0.2 / 0.3 / 0.4%. Its position is set by about 35 exits in 9.7
  years. This roughly matches P0's 1.03 and suggests a 4h cost-aware agent on these inputs cannot go much
  higher.

### 4.4 Models on daily bars (daily features, daily decisions)

Daily bars, buy & hold Sharpe 1.01.

| model | h | OOS R² | CW t | net SR | gross SR | MDD | sw/yr |
|---|---|---|---|---|---|---|---|
| ols | 1d | -1.13% | +1.2 | 0.47 | 0.73 | -72% | 75 |
| ols | 1w | -10.39% | -0.6 | 0.72 | 0.84 | -68% | 44 |
| ols | 1m | -41.41% | -1.5 | 0.65 | 0.72 | -75% | 27 |
| ols | 3m | -82.27% | -0.9 | 0.56 | 0.65 | -80% | 33 |
| ridge | 1d | +0.11% | +2.3 | 0.57 | 0.74 | -56% | 29 |
| ridge | 1w | +0.54% | +2.5 | 0.75 | 0.84 | -81% | 34 |
| ridge | 1m | -27.13% | -1.7 | 0.57 | 0.60 | -81% | 14 |
| ridge | 3m | -62.55% | -1.4 | 0.60 | 0.62 | -81% | 8 |
| ridge_vs | 1d | -1.52% | -0.7 | 0.74 | 0.89 | -54% | 35 |
| ridge_vs | 1w | -11.00% | -1.7 | 0.85 | 0.97 | -75% | 39 |
| ridge_vs | 1m | -66.21% | -1.6 | 0.69 | 0.75 | -81% | 20 |
| ridge_vs | 3m | -143.10% | -1.3 | 0.48 | 0.54 | -81% | 22 |

The daily-bar models are no better. Only ridge at 1d and 1w has a positive OOS R² (+0.1% / +0.5%); the net
Sharpe tops out at 0.85.

## 5. The second moment: volatility is predictable

| predictor | fwd realized vol 1d: IC×100 (t) | 1w: IC×100 (t) |
|---|---|---|
| ln_sigma | +59.7 (+43.1) | +65.6 (+19.2) |
| vol_regime | +34.5 (+23.9) | +33.3 (+9.5) |
| vol_trend | +16.4 (+11.8) | +17.7 (+6.3) |
| rel_vol | +21.1 (+20.2) | +15.2 (+9.5) |
| range | +15.1 (+15.5) | +10.7 (+8.1) |
| x_300_1200 | +6.6 (+4.6) | +8.5 (+2.3) |
| ma_1200 | +4.7 (+3.3) | +6.6 (+1.8) |
| rsi_84 | -5.3 (-3.7) | -4.7 (-1.4) |

Forward realized volatility is highly predictable, with IC ≈ 0.6 from current σ alone. Direction is not.
Volatility and returns are **positively** related in BTC (`vol_regime` has a positive return IC). That is
consistent with naive volatility targeting (B7: 0.89) losing to B&H. Volatility can only add value if sizing is learned
jointly with a risk-sensitive objective.

## 6. Conclusions and implications for the next RL design

**Which horizons and features carry signal.**

- *Statistically strongest:* the 4h reversal (`clv`, `ret_1`) and volume activity (`rel_vol`, `range`). These
  are real, stable across eras, and worth about 0.1% per bar, so they are uneconomic at 0.15% one-way, and
  still at Upbit's 0.10% (net Sharpe ≤ 0.6).
- *Economically relevant:* medium-speed trend features (price vs 50-200 bar MA, 1-week and 1-month returns,
  RSI84, MACD, daily MACD histogram, drawdown from the 1-month high) at 1w-1m horizons.
- *Volatility:* the level of σ is highly predictable.

**Is it stable across eras?** The reversal is stable. The trend premium is not: it was strong in 2014-2020,
roughly a third of that in 2021-2024, and about zero in 2025-26, where the slowest features even had a
negative IC. At the 1-month horizon, the medium-speed features at least kept a positive sign in every era.
Any model fitted on an expanding window mostly learns 2014-2020. P0 weighted the last two years more heavily
instead; after the 2023-24 bull market that produced "always long" models, which fits its lockbox behaviour.
In 2025-26 there was almost nothing positive left to act on, although the 200-day-MA rule and the 4h-decision MACD rule still cut B&H's drawdown (−36% / −24% vs −53%).

**What an RL agent should do.**

1. **Decide once a day, not every 4h.** Every piece of evidence points the same way. 4h-only information is
   below costs. Daily decisions cut turnover up to 3-4× at no loss of signal, and they raised net Sharpe for 14 of
   the 15 forecast rules and for the momentum and MACD signals. The UTC 00:00 bar can remain the execution
   bar.
2. **Target an objective horizon of about 1 month**, the horizon that is both economic and the most era-consistent. The edge is ≈ 2.6% per trade against a 0.3% round trip pooled, and still ≈ 0.6% at 2025-26 volatility and IC.
   On daily steps that means γ ≈ 0.967 (≈ 30 days), or an h-step reward. On 4h
   steps it would take γ ≈ 0.995. The current γ = 0.97 per 4h bar is about 5.5 days, the weak 1d-1w end.
   Expect about 25 switches/yr or fewer; a higher learned rate is a warning sign.
3. **Shrink the input set.** Drop the 4h microstructure/candle inputs (`ret_1`, `clv`, `range`, `macdh_4h`,
   `rsi_14`, `ret_6`) from the decision state. They carry real but untradable reversal and invite churn; if
   needed, use them only as an execution-timing layer. Keep about 6-8 medium-speed trend features plus one
   volatility-state feature. Only ~150 independent months support any fit, and the 22-feature ridge lost to
   every single signal.
4. **Consider fractional exposure** ({0, ½, 1}, or continuous with a band) together with a log-wealth reward.
   That lets the predictable second moment matter, which a 0/1 action cannot use. Evaluate it against the
   volatility-targeting and fixed-fraction baselines.
5. **Raise the benchmark bar.** Compare against the daily-MACD rule (1.25 daily, MDD −52%), the 200-day-MA
   rule (1.12) and 1-month momentum (1.10), not only B&H. Pre-register the success criterion as drawdown
   reduction at similar CAGR. A Sharpe gain of 0.1-0.25, which is what the data support, is statistically
   undetectable over 10 years.
6. **Non-stationarity needs explicit handling.** A per-era check of feature IC, like the tables above, should
   be part of the monthly gate. When the trend IC of the trailing 2-3 years is about zero, the prior should
   fall back to the simple rule or to B&H rather than to a freshly fitted model.

## 7. Caveats

- **Post-hoc.** The lockbox is open, and the horizon grid, eras and models were fixed before this run, but
  the no-trade band, the cost curves and the conservative shift-null p were added after seeing first results. About 100 rule variants were
  evaluated, so the best one (1.07 / 1.19 / 1.25) is expected to look good by selection alone.
- **Small samples.** NW t-stats with lags = h are unreliable when n/h is small. The 3m horizon has 12-16
  independent observations per era, and 2025-26 has only 6 at 3m. The circular-shift null is the better
  guide for the pooled numbers.
- **Data.** One venue (Bitstamp), and the phase-0 grid only. Part of the 4h reversal may be venue
  microstructure. There is no market impact, and cash earns 0%.

## Appendix A. Per-era rank IC, 4h bars, all horizons (IC×100, NW t with lags = h; † on pooled = shift-null p < 0.05)

**4h (h = 1)** — n_indep per era: 2014-2016 6548, 2017-2020 8765, 2021-2024 8766, 2025-2026 3791, pooled 27870

| feature | 2014-2016 | 2017-2020 | 2021-2024 | 2025-2026 | pooled |
|---|---|---|---|---|---|
| ret_1 | -6.8 (-5.2) | -5.3 (-4.8) | -6.8 (-6.0) | -3.1 (-1.9) | -5.8 (-9.4)† |
| ret_6 | +1.6 (+1.3) | -0.5 (-0.5) | -5.2 (-4.8) | -3.1 (-1.8) | -1.8 (-3.0)† |
| ret_42 | +4.1 (+3.2) | +3.2 (+3.0) | -1.0 (-1.0) | -1.4 (-0.8) | +1.7 (+2.7)† |
| ret_180 | +2.5 (+2.1) | +3.7 (+3.5) | +1.0 (+1.0) | -1.0 (-0.6) | +2.2 (+3.8)† |
| ma_20 | -0.4 (-0.3) | +0.8 (+0.7) | -2.9 (-2.7) | -3.8 (-2.3) | -1.1 (-1.8) |
| ma_50 | +2.1 (+1.7) | +2.5 (+2.2) | -1.5 (-1.4) | -2.9 (-1.7) | +0.7 (+1.1) |
| ma_200 | +2.9 (+2.3) | +4.1 (+3.7) | +0.1 (+0.1) | -1.7 (-1.0) | +2.1 (+3.5)† |
| ma_1200 | +1.6 (+1.4) | +4.3 (+4.3) | +1.0 (+0.9) | -1.0 (-0.6) | +2.2 (+3.9)† |
| x_50_200 | +3.4 (+2.8) | +4.2 (+4.0) | +0.9 (+0.9) | -0.3 (-0.2) | +2.6 (+4.4)† |
| x_300_1200 | +0.3 (+0.3) | +3.4 (+3.5) | +0.9 (+0.9) | +0.0 (+0.0) | +1.6 (+2.9) |
| rsi_14 | +0.7 (+0.5) | +1.4 (+1.3) | -2.4 (-2.2) | -3.5 (-2.1) | -0.4 (-0.6) |
| rsi_84 | +2.9 (+2.3) | +3.6 (+3.3) | +0.1 (+0.1) | -1.9 (-1.1) | +2.0 (+3.3)† |
| macd_4h | +2.5 (+2.0) | +3.1 (+2.8) | -0.6 (-0.6) | -2.4 (-1.4) | +1.3 (+2.1)† |
| macdh_4h | -1.1 (-0.9) | -0.3 (-0.3) | -2.5 (-2.4) | -2.0 (-1.2) | -1.4 (-2.3)† |
| macdh_1d | +2.9 (+2.4) | +2.2 (+2.0) | -0.4 (-0.4) | -0.4 (-0.2) | +1.3 (+2.2)† |
| vol_regime | +2.7 (+2.3) | +1.8 (+1.7) | +1.8 (+1.8) | -0.6 (-0.3) | +1.7 (+2.8)† |
| rel_vol | +5.1 (+4.1) | +1.5 (+1.4) | +1.1 (+1.1) | +0.8 (+0.5) | +2.2 (+3.8)† |
| vol_trend | +3.4 (+2.7) | +2.0 (+1.9) | +1.2 (+1.1) | -1.3 (-0.8) | +1.7 (+2.8)† |
| flow_20 | +1.4 (+1.1) | +1.7 (+1.6) | -1.6 (-1.6) | -2.2 (-1.3) | +0.2 (+0.3) |
| range | +5.9 (+4.7) | -0.7 (-0.7) | +2.4 (+2.3) | +0.8 (+0.5) | +1.9 (+3.3)† |
| clv | -6.4 (-5.3) | -6.0 (-5.5) | -9.0 (-8.2) | -3.7 (-2.4) | -6.7 (-11.2)† |
| dd_180 | +2.5 (+2.1) | +3.7 (+3.6) | -0.6 (-0.6) | -1.5 (-0.9) | +1.6 (+2.7)† |

**1d (h = 6)** — n_indep per era: 2014-2016 1090, 2017-2020 1461, 2021-2024 1461, 2025-2026 631, pooled 4643

| feature | 2014-2016 | 2017-2020 | 2021-2024 | 2025-2026 | pooled |
|---|---|---|---|---|---|
| ret_1 | +1.6 (+1.3) | +0.8 (+0.8) | -2.2 (-2.1) | -0.8 (-0.5) | -0.1 (-0.2) |
| ret_6 | +0.1 (+0.0) | -1.4 (-0.7) | -3.9 (-2.1) | -0.6 (-0.2) | -1.7 (-1.6) |
| ret_42 | +6.3 (+2.4) | +5.1 (+2.3) | -1.2 (-0.6) | -0.4 (-0.1) | +3.0 (+2.5)† |
| ret_180 | +4.9 (+1.8) | +6.1 (+2.7) | +2.1 (+1.0) | -0.0 (-0.0) | +4.3 (+3.5)† |
| ma_20 | +0.8 (+0.3) | +2.5 (+1.2) | -1.9 (-0.9) | -2.8 (-0.9) | +0.3 (+0.3) |
| ma_50 | +5.0 (+1.9) | +4.5 (+2.1) | -1.7 (-0.8) | -1.9 (-0.6) | +2.2 (+1.8) |
| ma_200 | +6.2 (+2.3) | +7.2 (+3.2) | +0.5 (+0.2) | -0.9 (-0.3) | +4.4 (+3.5)† |
| ma_1200 | +3.3 (+1.3) | +8.1 (+3.7) | +1.4 (+0.6) | -2.0 (-0.6) | +4.2 (+3.4) |
| x_50_200 | +6.8 (+2.6) | +7.2 (+3.2) | +1.0 (+0.5) | -0.1 (-0.0) | +4.6 (+3.6)† |
| x_300_1200 | +0.8 (+0.3) | +6.5 (+3.0) | +1.1 (+0.5) | -0.9 (-0.3) | +2.9 (+2.4) |
| rsi_14 | +2.7 (+1.1) | +3.4 (+1.6) | -1.8 (-0.9) | -2.2 (-0.7) | +1.3 (+1.1) |
| rsi_84 | +6.4 (+2.4) | +6.8 (+3.0) | +0.8 (+0.4) | -1.3 (-0.4) | +4.4 (+3.5)† |
| macd_4h | +3.9 (+1.5) | +4.3 (+2.0) | -1.0 (-0.5) | -2.9 (-0.9) | +2.0 (+1.7) |
| macdh_4h | -4.2 (-1.7) | -1.0 (-0.5) | -1.6 (-0.8) | -0.7 (-0.2) | -1.8 (-1.5) |
| macdh_1d | +4.4 (+1.7) | +3.0 (+1.4) | -1.0 (-0.5) | -2.0 (-0.6) | +1.7 (+1.4) |
| vol_regime | +4.1 (+1.6) | +3.9 (+1.9) | +3.4 (+1.6) | -1.9 (-0.6) | +3.1 (+2.5) |
| rel_vol | +4.0 (+2.0) | +3.3 (+2.1) | +3.9 (+2.5) | -2.3 (-0.9) | +3.1 (+3.4)† |
| vol_trend | +7.0 (+2.7) | +3.5 (+1.6) | +4.0 (+1.8) | -3.3 (-1.0) | +3.6 (+3.0)† |
| flow_20 | +5.5 (+2.2) | +2.6 (+1.3) | -2.1 (-1.0) | -2.4 (-0.8) | +1.3 (+1.2) |
| range | +2.1 (+1.1) | +0.8 (+0.6) | +3.3 (+2.5) | -0.9 (-0.4) | +1.7 (+2.1) |
| clv | +1.2 (+1.0) | +1.4 (+1.3) | -3.0 (-3.2) | +0.3 (+0.2) | -0.2 (-0.3) |
| dd_180 | +4.4 (+1.7) | +7.1 (+3.2) | -0.6 (-0.3) | -0.4 (-0.1) | +3.5 (+2.8)† |

**1w (h = 42)** — n_indep per era: 2014-2016 155, 2017-2020 209, 2021-2024 209, 2025-2026 89, pooled 662

| feature | 2014-2016 | 2017-2020 | 2021-2024 | 2025-2026 | pooled |
|---|---|---|---|---|---|
| ret_1 | +3.2 (+2.7) | +2.8 (+2.4) | -1.0 (-1.0) | -0.4 (-0.2) | +1.4 (+2.2)† |
| ret_6 | +7.0 (+2.4) | +4.6 (+1.7) | -0.8 (-0.3) | -1.1 (-0.3) | +3.0 (+2.0)† |
| ret_42 | +11.4 (+1.9) | +6.3 (+1.3) | -2.3 (-0.5) | -3.5 (-0.5) | +4.2 (+1.5) |
| ret_180 | +8.9 (+1.4) | +8.7 (+1.6) | +7.1 (+1.4) | +3.1 (+0.4) | +8.6 (+2.9)† |
| ma_20 | +10.0 (+2.4) | +6.6 (+1.7) | -1.6 (-0.4) | -2.2 (-0.4) | +4.0 (+1.9) |
| ma_50 | +12.6 (+2.2) | +7.0 (+1.4) | -2.1 (-0.4) | -3.4 (-0.5) | +4.7 (+1.7) |
| ma_200 | +13.2 (+2.0) | +11.2 (+2.0) | +2.6 (+0.5) | +2.4 (+0.3) | +8.7 (+2.8)† |
| ma_1200 | +2.9 (+0.4) | +11.9 (+2.1) | +1.2 (+0.2) | -6.8 (-0.8) | +5.3 (+1.6) |
| x_50_200 | +10.8 (+1.6) | +9.7 (+1.8) | +3.1 (+0.6) | +4.2 (+0.5) | +7.9 (+2.6) |
| x_300_1200 | -2.3 (-0.3) | +9.3 (+1.6) | -0.8 (-0.1) | -6.7 (-0.8) | +2.2 (+0.7) |
| rsi_14 | +11.3 (+2.3) | +7.3 (+1.7) | -1.4 (-0.3) | -2.7 (-0.5) | +4.7 (+1.9) |
| rsi_84 | +13.4 (+2.0) | +12.7 (+2.3) | +3.0 (+0.6) | +0.4 (+0.0) | +9.1 (+3.0)† |
| macd_4h | +11.9 (+2.1) | +6.6 (+1.4) | -2.0 (-0.4) | -3.0 (-0.5) | +4.5 (+1.7) |
| macdh_4h | +3.5 (+1.0) | +3.2 (+1.1) | -1.1 (-0.4) | +1.0 (+0.2) | +1.7 (+1.0) |
| macdh_1d | +9.2 (+1.5) | +9.0 (+1.8) | -1.9 (-0.4) | -0.1 (-0.0) | +5.2 (+1.8) |
| vol_regime | +7.3 (+1.1) | +8.1 (+1.5) | +8.4 (+1.7) | -7.8 (-1.0) | +6.1 (+2.1) |
| rel_vol | +6.7 (+1.9) | +3.8 (+1.4) | +4.4 (+1.9) | -4.7 (-1.2) | +3.9 (+2.6)† |
| vol_trend | +8.5 (+1.4) | +6.3 (+1.4) | +5.0 (+1.2) | -8.9 (-1.3) | +4.9 (+1.9) |
| flow_20 | +12.0 (+2.6) | +3.8 (+0.9) | -0.1 (-0.0) | -2.3 (-0.4) | +3.8 (+1.7) |
| range | +3.2 (+1.2) | -0.1 (-0.1) | +3.0 (+1.7) | -2.5 (-0.9) | +1.3 (+1.1) |
| clv | +4.7 (+3.2) | +3.4 (+2.8) | -1.3 (-1.5) | -0.3 (-0.2) | +1.8 (+2.7)† |
| dd_180 | +8.9 (+1.3) | +15.3 (+2.8) | -2.3 (-0.5) | +3.5 (+0.5) | +7.6 (+2.5) |

**1m (h = 180)** — n_indep per era: 2014-2016 36, 2017-2020 49, 2021-2024 49, 2025-2026 20, pooled 154

| feature | 2014-2016 | 2017-2020 | 2021-2024 | 2025-2026 | pooled |
|---|---|---|---|---|---|
| ret_1 | +2.6 (+2.1) | +3.2 (+2.3) | +0.7 (+0.9) | +0.4 (+0.3) | +2.1 (+3.2)† |
| ret_6 | +7.0 (+2.4) | +5.5 (+1.8) | +2.6 (+1.1) | -0.1 (-0.0) | +4.8 (+3.1)† |
| ret_42 | +11.8 (+1.7) | +12.7 (+2.0) | +5.4 (+1.0) | +4.6 (+0.6) | +10.3 (+2.9)† |
| ret_180 | +11.4 (+1.1) | +10.3 (+1.1) | +10.1 (+1.1) | -3.2 (-0.3) | +9.5 (+1.7) |
| ma_20 | +9.7 (+2.2) | +9.0 (+2.1) | +4.5 (+1.2) | +1.0 (+0.2) | +7.6 (+3.2)† |
| ma_50 | +12.7 (+1.9) | +12.6 (+2.0) | +5.2 (+0.9) | +4.6 (+0.6) | +10.3 (+3.0)† |
| ma_200 | +15.7 (+1.6) | +14.6 (+1.6) | +9.5 (+1.1) | +1.4 (+0.1) | +12.0 (+2.3) |
| ma_1200 | +0.6 (+0.0) | +21.6 (+1.9) | +2.8 (+0.2) | -19.2 (-1.3) | +7.7 (+1.2) |
| x_50_200 | +12.9 (+1.2) | +11.1 (+1.2) | +7.5 (+0.8) | -0.5 (-0.0) | +9.3 (+1.7) |
| x_300_1200 | -4.3 (-0.3) | +20.2 (+1.8) | -1.6 (-0.1) | -17.2 (-1.0) | +4.4 (+0.7) |
| rsi_14 | +11.1 (+1.9) | +11.0 (+2.0) | +4.9 (+1.0) | +3.4 (+0.5) | +9.2 (+3.0)† |
| rsi_84 | +14.8 (+1.5) | +15.8 (+1.8) | +10.5 (+1.2) | -0.2 (-0.0) | +12.7 (+2.5) |
| macd_4h | +12.3 (+1.8) | +12.4 (+2.0) | +5.4 (+1.0) | +4.8 (+0.6) | +10.2 (+2.9)† |
| macdh_4h | +3.3 (+1.8) | +1.1 (+0.6) | +0.9 (+0.5) | -3.0 (-1.0) | +1.4 (+1.3) |
| macdh_1d | +12.5 (+1.4) | +8.6 (+1.1) | +6.1 (+0.8) | +16.3 (+1.6) | +10.2 (+2.4)† |
| vol_regime | +6.7 (+0.7) | +9.7 (+1.0) | +12.9 (+1.4) | -13.1 (-0.9) | +7.6 (+1.4) |
| rel_vol | +4.0 (+0.8) | +1.8 (+0.5) | +3.3 (+1.1) | -7.4 (-1.6) | +2.4 (+1.2) |
| vol_trend | +6.6 (+0.8) | +1.4 (+0.2) | +6.4 (+1.1) | -13.9 (-1.6) | +3.1 (+0.9) |
| flow_20 | +12.6 (+2.3) | +9.6 (+2.2) | +3.3 (+0.9) | -6.6 (-1.2) | +7.0 (+2.9)† |
| range | +1.0 (+0.3) | -0.4 (-0.1) | -0.2 (-0.1) | -2.4 (-0.9) | -0.4 (-0.3) |
| clv | +4.6 (+2.6) | +3.5 (+2.1) | +0.5 (+0.5) | -2.6 (-1.8) | +2.1 (+2.5)† |
| dd_180 | +11.9 (+1.1) | +15.2 (+1.7) | +2.6 (+0.3) | +2.6 (+0.2) | +9.3 (+1.8) |

**3m (h = 540)** — n_indep per era: 2014-2016 12, 2017-2020 16, 2021-2024 16, 2025-2026 6, pooled 51

| feature | 2014-2016 | 2017-2020 | 2021-2024 | 2025-2026 | pooled |
|---|---|---|---|---|---|
| ret_1 | +1.9 (+1.2) | +4.1 (+3.9) | +0.2 (+0.3) | -1.7 (-1.1) | +2.3 (+3.0)† |
| ret_6 | +6.3 (+1.6) | +8.3 (+3.4) | +0.7 (+0.4) | -6.4 (-1.6) | +5.2 (+2.9)† |
| ret_42 | +8.2 (+1.0) | +19.5 (+3.5) | +2.7 (+0.5) | -7.7 (-0.7) | +11.1 (+2.8)† |
| ret_180 | +6.8 (+0.5) | +29.3 (+3.1) | +3.6 (+0.3) | -9.6 (-0.4) | +15.2 (+2.2) |
| ma_20 | +8.5 (+1.5) | +13.0 (+3.6) | +2.2 (+0.7) | -7.5 (-1.2) | +8.2 (+3.0)† |
| ma_50 | +9.5 (+1.1) | +19.1 (+3.5) | +2.6 (+0.6) | -8.2 (-0.8) | +11.2 (+2.9)† |
| ma_200 | +9.2 (+0.6) | +30.8 (+3.4) | +4.6 (+0.5) | -12.2 (-0.6) | +16.2 (+2.4) |
| ma_1200 | +23.7 (+1.2) | +33.5 (+1.9) | -4.9 (-0.3) | +1.6 (+0.1) | +18.7 (+1.7) |
| x_50_200 | +7.7 (+0.5) | +28.9 (+3.2) | +3.7 (+0.4) | -9.4 (-0.4) | +15.1 (+2.2) |
| x_300_1200 | +25.9 (+1.4) | +26.3 (+1.3) | -7.9 (-0.4) | +5.1 (+0.3) | +15.1 (+1.3) |
| rsi_14 | +8.8 (+1.2) | +16.8 (+3.5) | +2.4 (+0.6) | -7.9 (-0.9) | +10.0 (+2.9)† |
| rsi_84 | +8.6 (+0.6) | +31.4 (+3.3) | +5.8 (+0.6) | -11.2 (-0.5) | +16.8 (+2.5) |
| macd_4h | +9.4 (+1.1) | +19.3 (+3.5) | +3.0 (+0.6) | -8.0 (-0.7) | +11.3 (+2.9)† |
| macdh_4h | +3.7 (+2.2) | +0.5 (+0.5) | +0.4 (+0.3) | -5.3 (-2.9) | +1.3 (+1.7) |
| macdh_1d | +5.9 (+0.7) | +7.1 (+1.3) | +6.6 (+1.1) | -6.1 (-0.5) | +6.4 (+1.8)† |
| vol_regime | -5.4 (-0.4) | +13.7 (+1.2) | +7.3 (+0.7) | -9.4 (-0.5) | +4.8 (+0.7) |
| rel_vol | +5.2 (+1.6) | +4.4 (+1.6) | +2.9 (+1.2) | -2.7 (-0.4) | +4.1 (+2.2) |
| vol_trend | +8.1 (+1.3) | +7.2 (+1.8) | +7.0 (+1.4) | -3.1 (-0.4) | +6.1 (+2.1) |
| flow_20 | +10.6 (+1.6) | +10.8 (+2.9) | -0.0 (-0.0) | -14.6 (-2.2) | +5.9 (+2.3) |
| range | -0.3 (-0.1) | +1.4 (+0.5) | -2.4 (-0.9) | +1.9 (+0.6) | +0.0 (+0.0) |
| clv | +4.8 (+1.9) | +5.6 (+3.5) | +0.4 (+0.3) | -2.2 (-1.6) | +3.4 (+2.9)† |
| dd_180 | +7.5 (+0.6) | +29.5 (+3.1) | -2.9 (-0.3) | -8.1 (-0.4) | +12.3 (+1.8) |
