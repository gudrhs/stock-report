# RL for crypto and financial trading: what the papers show, and what to change in the BTC agent

**Status: post-hoc research.** The P0 study opened the 2025-01..2026-09 lockbox on 2026-09-25. So every
BTC number here uses data already seen, and nothing here was pre-registered. I read 16 papers through
alphaXiv (full text or targeted page queries). Two small non-RL pilot checks
(`btc/research/litreview_pilots.py` → `litreview_pilots.json`, about 2.5 min, 1 process) test whether two
ideas from the literature transfer to BTC. The accounting is the same as report_full: fill at the next
open, 0.15% one way, OOS 2017-01..2026-09. The script reproduces B0 = 1.0148 and B2 = 1.1233 exactly.

## Summary

1. **P0 is already more rigorous than almost all of the RL-trading literature.** It uses a monthly
   walk-forward, 10 replications, exact costs, action augmentation, phase-shift augmentation,
   PBO/DSR, and synthetic positive and negative controls. Only a few papers come close: Lim 2019,
   Wood 2021, Shu 2024 and Bysik 2026. P0's failure is a lack of signal at its horizon, not a lack of method.
2. **The strong positive RL results come from settings that don't match ours.** They assume costs
   ≤ 3 bp, test windows of 50 days to 2 months, or unreported seeds. When costs are close to our
   15 bp, the deep models' edge goes away. The DMN LSTM falls from a cost-free Sharpe of ≈2.8 to
   **−5.3 at 10 bp** unless a turnover penalty is added. The changepoint model's gain comes from fast
   reversion and dies above 2 bp. Hourly BTC XGBoost reaches **Sharpe 0.80 at 15 bp, MDD −81%**, which is not
   significant against buy & hold.
3. **When the signal is below costs, DQN-type agents collapse to buy & hold.** Théate & Ernst report
   this, and it is P0's failure. The P0 autopsy shows the decision margin Δ = U1−U0 has a yearly mean of +0.03 to
   +0.32 against a sell threshold of −0.30. Its 5th percentile only reaches −0.07 to −0.31, so the
   learned *drift* keeps the agent long. The Jan-2025 and Jan-2026 cold rebuilds were 99.6% and
   99.98% long (replayed rep 0).
4. **Where the literature agrees trading edge exists, it is slow.** Trend following on 1–3-month
   horizons (Sepp & Lucic 2026) delivers most of its value as smaller drawdowns. Short-memory
   autocorrelation can't pay realistic costs. This matches our predictability note: the 1-month IC is
   +8 to +13, and the 4h reversal can't be traded.
5. **Pilot A: total-volatility scaling hurts BTC.** B7 has gross Sharpe 0.95 against 1.01 for buy &
   hold, so costs are not the whole reason. The forward 30-day return is **highest in the high-vol
   tercile**: 44% / 4% / 61% annualized from low to high. Scaling by downside deviation only helps a
   little: Sharpe 1.06, MDD −68%, CAGR 49%.
6. **Pilot B: the equity regime model (statistical jump model, Shu 2024) does not transfer to BTC.** Its
   Sharpe ranges from 0.26 to 0.89 across all 10 variants, below buy & hold's 1.01. In-sample, BTC's
   whole return sits in a "rally" state covering about 16% of days, with an annualized mean log
   return of +4.8 against −0.2 for the rest. Detected online, that state arrives too late to trade.
7. **Implication for the next design.** Use a daily decision clock with a 1-month value horizon.
   Separate the drift from the learned conditional signal. Build the agent on top of a trend rule,
   not on top of buy & hold. Keep the state to a few slow features and penalize only downside risk.
   Judge the agent against B2 or B5, not B&H, and use IQM and bootstrap intervals over replications.
   The realistic target is ΔSharpe of 0 to +0.1 over the trend rule (+0.1 to +0.25 over B&H, per the
   predictability note), mostly from smaller drawdowns.
   Proving that statistically needs forward paper trading, not more backtests.

## 1. What has to be explained (2017-01..2026-09, 0.15%)

| | Sharpe | CAGR | MDD | exposure | switches/yr | 2025-26 Sharpe / MDD |
|---|---|---|---|---|---|---|
| B&H (B0) | 1.01 | 58% | −83% | 1.00 | 0 | 0.08 / −53% |
| **P0 agent** (median rep) | 1.03 | 58% | −75% | 0.91 | 5 | 0.08 / −53% (100% long) |
| B2 close > SMA1200 (4h) | 1.12 | 58% | −67% | 0.58 | 14 | 0.15 / −36% |
| B5 daily-MACD sign (4h) | 1.15 | 54% | −64% | 0.52 | 25 | 0.22 / −24% |
| B5 decided daily (post-hoc best rule) | 1.25 | 61% | −52% | 0.53 | 23 | −0.39 / −34% |
| B7 vol-target B&H | 0.89 | 37% | −72% | 0.83 | 489 rebal. | −0.07 / −53% |

The P0 autopsy (`btc/research/out/p0_autopsy/analysis.json`) explains the failure. The
decision margin Δ (κ = 100 units) averaged 0.03–0.32 per year, against switching thresholds of ±0.30
(c_dec = 0.3%). The training pool's drift alone, κ·E[m]·33 bars, was 0.31–1.71. With conditional ICs of
0.02–0.05, the learned conditional variation rarely overcame drift plus hysteresis. There were zero
sell-zone bars in 2021 and 2024–26. The 12-variant grid (γ ∈ {0, 0.9, 0.97} × 22 or 8 features ×
cost multiplier 1 or 2) stayed between Sharpe 0.88 and 1.06. The 8-feature variants averaged 1.04,
the 22-feature variants 0.96.

## 2. Papers read

Evidence grade: **A** = long OOS walk-forward, costs, and several assets or seeds. **B** = partial.
**C** = weak or cautionary.

| paper | setting | method / reward | state | decision freq. | costs | evaluation | honest OOS evidence | grade |
|---|---|---|---|---|---|---|---|---|
| Lim, Zohren, Roberts 2019, *Deep Momentum Networks* [1904.04912](https://www.alphaxiv.org/abs/1904.04912) | 88 futures | LSTM outputs position directly; **Sharpe loss** inside a vol-scaling (15%) TSMOM frame; turnover-regularized variant | vol-normalized returns (1d–1y), MACD at 3 speeds | daily | swept 0–5 bp, and 10 bp with regularizer | expanding window, refit every 5y, OOS 1995–2015, 50-trial random search | Sharpe-LSTM > 2× TSMOM **before costs**. Edge gone at 2–3 bp. At 10 bp: LSTM −5.31, **LSTM+turnover reg 0.91**, Sgn(12m) 0.86, long-only 0.63 | A |
| Wood, Roberts, Zohren 2021, *Slow momentum with fast reversion* [2105.13727](https://www.alphaxiv.org/abs/2105.13727) | 50 futures | DMN + online GP changepoint score/location as inputs; Sharpe loss | DMN inputs + CPD severity/location | daily | 0–5 bp (not in loss) | expanding, 5y blocks, 1995–2020; **5 repeated trials** | Sharpe 1.62 → 2.16 raw; +70% in 2015–20. Gain attributed to **fast mean reversion**; beats classical rules **only to ~2 bp**; longer CPD windows win as costs rise | A |
| Zhang, Zohren, Roberts 2019, *DRL for trading* [1911.10107](https://www.alphaxiv.org/abs/1911.10107) | 50 futures | DQN / PG / A2C (LSTM); **vol-scaled profit reward** with cost term; γ = 0.3; 20 bp training cost as turnover regularizer | 60 lags of normalized price, vol-normalized 1–12m returns, MACD, RSI | daily | 1–45 bp sweep | refit every 5y, OOS 2011–2019; no seeds reported | DQN best (vol-rescaled Sharpe 1.29 on all contracts); positive at 25 bp; long-only better on equity indices | B |
| Huang 2018, *Financial trading as a game* [1807.02787](https://www.alphaxiv.org/abs/1807.02787) | 12 FX pairs, 15 min | DRQN; **action augmentation** (reward for all actions, no ε-greedy); tiny replay (480) | returns, volume, one-hot position, time | 15 min | spread 0.08–0.2 bp (negligible) | online learning 2012–17, 5 runs | action augmentation +6.4%/yr over ε-greedy. **Costs unrealistically small**. P0 already uses action augmentation | C |
| Théate & Ernst 2020, *TDQN* [2004.06627](https://www.alphaxiv.org/abs/2004.06627) | 30 stocks | DDQN + Huber, Adam, data augmentation (shift, filter, noise); daily-return reward | OHLCV features, low-pass filtered | daily | 0–0.2% | train 2012–17, test 2018–19 | "only barely surpasses B&H". **Tends toward passive when uncertain; stops trading at high cost**. γ controls trade frequency; large run-to-run variance | B |
| Ishikawa & Nakata 2021 [2106.03035](https://www.alphaxiv.org/abs/2106.03035) | USD/JPY 1 min | Q redefined as discounted value of *holding a position* (close cousin of our U decomposition) | price diffs + position | 1 min | 1–10 bp | 6-month test | longer holds than LSTM or LightGBM, but Sharpe 2.04 at 1 bp → **−0.40 at 10 bp** | C |
| Jiang, Xu, Liang 2017, *EIIE* [1706.10059](https://www.alphaxiv.org/abs/1706.10059) | 12 coins, Poloniex, 30 min | deterministic policy gradient on average log return; portfolio-vector memory for costs | 50×OHLC window | 30 min | 0.25% | three **50-day** back-tests; zero slippage; volume-based coin preselection | "≥4× in 50 days". Short windows plus survivorship-prone universe: **cautionary** | C |
| Gort et al. 2022 [2209.05559](https://www.alphaxiv.org/abs/2209.05559) | 10 coins, 5 min | PPO/TD3/SAC; reject agents with **PBO > 10%** (CPCV N=5, k=2, 50 trials) | 6 decorrelated indicators (|ρ| < 0.6) | 5 min | 0.3% | train Feb–Apr 2022, **test 2 months** | PBO: PPO 8%, TD3 9.6%, SAC 21%, walk-forward PPO 17.5%. "Win" = −35% vs index −51%, with a CVIX kill-switch as confound | C |
| Hêche et al. 2025, *distributional RL* [2501.04421](https://www.alphaxiv.org/abs/2501.04421) | TTF gas futures | C51 / QR-DQN / IQN; **CVaR_α action selection**; Sharpe-type reward | 1141 features → PCA | daily | – | 4 experiments, 2017–22 | IQN_α gives tunable risk aversion (risky-state share 61% → 13% at α = 0.3) at lower P&L (666 → 266). Small sample, no tests | B− |
| Shu, Yu, Mulvey 2024, *statistical jump model* [2402.05272](https://www.alphaxiv.org/abs/2402.05272) | S&P, DAX, Nikkei | 2-state clustering with jump penalty; 0/1 strategy; λ picked monthly by 8y validation Sharpe | EWM downside deviation (hl 10d), Sortino (hl 20d, 60d) | daily, +1-day delay | 10 bp | 3000-day window refit every 6 months; OOS 1990–2023 | Sharpe 0.48 → 0.68 (S&P), 0.30 → 0.44, 0.12 → 0.31; MDD −55 → −27%; latency ≈ 2 weeks. **Does not transfer to BTC (pilot B)** | A |
| Bysik & Ślepaczuk 2026, *ML BTC under costs* [2606.00060](https://www.alphaxiv.org/abs/2606.00060) | BTC/USDT 1h, 2018–26 | XGBoost / LSTM / iTransformer forecasts + **cost-aware execution filter** \|r̂\| > λ·c·\|Δpos\| | OHLCV + TA + EGARCH | 1h | 0–25 bp | 27 rolling folds (12m/3m/3m), block bootstrap, Holm | sign rules fail at 10 bp; filter restores Sharpe 1.06 at 10 bp, **0.80 at 15 bp (MDD −81%)**; not significant vs B&H; long-short fragile | A |
| Sepp & Lucic 2026, *Science of trend following* [2607.19497](https://www.alphaxiv.org/abs/2607.19497) | 84 futures + theory | closed-form TF Sharpe = autocorrelation channel + drift channel; net Sharpe and cost-optimal span | vol-normalized returns | daily | analytic | in-sample attribution (ρ = 0.99) | AR(1) gross SR ≈ 2φ√(a/span); break-even cost ≈ 37–41 bp for φ = 0.05, **but futures φ fell from 0.04 to 0.01 after 2010**. Long memory supports 1–3 month spans; long spans converge to B&H | A (theory) |
| Song, Liu, Chen 2026, *Label Horizon Paradox* [2602.03395](https://www.alphaxiv.org/abs/2602.03395) | CSI 300/500/1000, S&P 500 | train on proxy horizon δ ≠ target Δ; bi-level choice of δ | stock features | intraday–daily | backtest appendix | 10 architectures, 5 seeds per horizon | the best supervision horizon trades signal realization against accumulated noise; consistent but modest IC and ICIR gains over training on the target horizon | B |
| Agarwal et al. 2021, *Statistical precipice* [2108.13264](https://www.alphaxiv.org/abs/2108.13264) | Atari, Procgen, DMC | evaluation method | – | – | – | – | with 3–10 runs, point estimates mislead; use **IQM + stratified bootstrap CIs**, performance profiles, P(improvement); medians need 50–100 runs | A (method) |
| Nguyen 2026, *AdaptiveTrend* [2602.11708](https://www.alphaxiv.org/abs/2602.11708) | 150 crypto perps, 6h | momentum entry + ATR trailing stop, monthly grid re-optimization, 70/30 long-short | momentum | 6h | 4 bp + slippage | one 36-month OOS | claims Sharpe 2.41. Monthly re-optimization on the prior month and a timeframe chosen from 6 → **selection-inflated; do not use as evidence** | C |
| Zhang 2025, *FR-LUX* [2510.02986](https://www.alphaxiv.org/abs/2510.02986) | regime × cost grid | PPO with cost in reward, trade-space trust region, regime conditioning | vol and liquidity regimes | – | 0–50 bp | 3 seeds per cell, Romano–Wolf | theory for an inaction band. Empirical detail thin | C |

Not on arXiv and **not read here**, only cited through the papers above: Moody & Saffell 2001 (*Learning to
trade via direct reinforcement*, differential Sharpe/downside ratio, IEEE TNN). López de Prado 2018 and
Joubert 2022 (meta-labeling). Bailey et al. 2016 (PBO). Harvey et al. 2018 (volatility targeting).
Proposals below that rest on these are marked (†).

## 3. Lessons most papers agree on

1. **Costs set the viable horizon.** Every model that relies on fast structure died at a few bp: DMN at
   2–3 bp, CPD at 2 bp, the FX DQN at 10 bp. What survives is either slow (TSMOM, 1–3m) or passed
   through an explicit cost filter. Sources: Lim, Wood, Ishikawa, Bysik, Sepp & Lucic.
2. **Optimize the objective you are judged on, and put turnover inside it.** Sharpe-loss direct policies
   beat return, MSE and classification targets (Lim). A turnover term turned −5.3 into 0.91 at 10 bp. A
   cost-aware execution filter mattered more than model choice (Bysik).
3. **Volatility scaling is central in futures TSMOM** (Lim, Zhang, Wood, and Harvey† via them). On BTC,
   total-vol scaling *hurts*, because the high-vol states carry the returns (pilot A).
4. **With a weak signal, value-based agents default to passive.** Théate, and P0. The discount factor
   also works as a turnover knob (Théate; Zhang uses γ = 0.3 plus a 20 bp training cost).
5. **Regime information helps most at turning points**, but regime detection lags by about 2 weeks to
   25 days (Shu; Wood). BTC turning points come fast, and its high-return regime is short.
6. **Small and slow beats big and fast** on daily financial data. WaveNet underperformed a linear model
   (Lim). XGBoost ≥ LSTM and iTransformer (Bysik). Decorrelated, few features (Gort).
7. **Evaluation.** Single short test windows (Jiang, Gort), selection across hyperparameters or
   timeframes (AdaptiveTrend), and point estimates from a few seeds (Agarwal) explain most of the
   impressive numbers in the field. PBO/CPCV (Gort, Bailey†) and IQM with stratified bootstrap
   (Agarwal) are the remedies.

## 4. Pilot checks (post-hoc, non-RL, 2017-01..2026-09, 0.15%)

**A. Volatility scaling.** Is B7's gap to B&H a cost problem or a BTC problem?

| strategy | Sharpe | CAGR | MDD | exposure | rebal/yr | Sharpe 17-20 / 21-24 / 25-26 |
|---|---|---|---|---|---|---|
| B&H | 1.01 | 58% | −83% | 1.00 | 0 | 1.45 / 0.78 / 0.08 |
| B7 (4h, 2% band) net | 0.89 | 37% | −72% | 0.83 | 489 | 1.47 / 0.60 / −0.07 |
| B7 (4h) **gross** | 0.95 | 41% | −70% | 0.83 | 489 | 1.54 / 0.66 / −0.04 |
| B7 daily, 10% band, net | 0.90 | 38% | −71% | 0.81 | 60 | 1.39 / 0.68 / −0.03 |
| total vol (EWM 10d) target 50%, daily | 0.95 | 41% | −70% | 0.78 | 23 | 1.44 / 0.77 / −0.00 |
| **downside dev. (EWM 10d) target 35%, daily** | **1.06** | 49% | −68% | 0.79 | 27 | 1.55 / 0.85 / 0.06 |
| downside dev. target 50%, daily | 1.03 | 54% | −74% | 0.90 | 15 | 1.52 / 0.82 / 0.01 |

Forward 30-day log return (annualized) by tercile of the trailing 10-day EWM (2014–2026, low → high):
total vol **44% / 4% / 61%**; downside deviation 57% / 12% / 40%. BTC's return is U-shaped in
volatility. Cutting exposure when total vol is high removes the rallies. Costs explain only about half
of B7's shortfall (0.06 of 0.13 Sharpe).

**B. Statistical jump model 0/1** (Shu 2024 features; refit every 6 months on an expanding window).
Every λ is shown, plus the paper's monthly λ-selection rule:

| variant | Sharpe | CAGR | MDD | exposure | sw/yr | Sharpe 17-20 / 21-24 / 25-26 | p vs B&H |
|---|---|---|---|---|---|---|---|
| K = 2, long in "bull", λ = 5 / 15 / 50 / 100 | 0.73 / 0.52 / 0.26 / 0.50 | 20 / 12 / 3 / 12% | −39 / −53 / −35 / −81% | 0.16 / 0.13 / 0.09 / 0.82 | 3 / 2 / 1 / 1 | e.g. λ = 5: 0.67 / 1.00 / −0.41 | 0.79–0.99 |
| K = 2, λ selected monthly | 0.68 | 18% | −39% | 0.14 | 3 | 0.55 / 0.96 / 0.64 | 0.84 |
| K = 3, flat only in the worst state, λ = 5 / 15 / 50 / 100 | 0.89 / 0.80 / 0.79 / 0.82 | 42 / 36 / 34 / 38% | −79 / −83 / −77 / −77% | 0.56 / 0.79 / 0.63 / 0.83 | 1–2 | e.g. λ = 50: 1.36 / 0.35 / −0.71 | 0.70–0.96 |
| K = 3, λ selected | 0.80 | 36% | −77% | 0.76 | 2 | 1.41 / 0.43 / −0.45 | 0.92 |

In-sample the 2-state fit splits BTC into a **16%-of-days rally state with an annualized mean log return of
+4.8** and an **84% state at −0.2** (2026-07 refit). That looks like a huge timing opportunity, but only with hindsight
smoothing. Online, with a latency of weeks, the rally state is caught late, and the strategy sits in
cash through most of the drift. In the equity version the rare state is the crash, which is why the
model works there. **Don't use a return-based regime gate on BTC.** At most, give it to the agent as a
feature (expected value ≈ 0).

## 5. Proposed modifications (testable, numpy-only, a few CPU-hours each)

All proposals keep P0's machinery: exact cost decomposition, action augmentation, monthly walk-forward,
10 replications, and the sanity gate. The P0 numbers above are the reference. "IQM" means the
interquartile mean over replications, with a stratified bootstrap 90% CI (Agarwal). Predictions are
mine, made before running the variants. They are the falsifiable part.

| # | change | motivated by | predicted effect (2017-01..2026-09, 0.15%) | kill criterion |
|---|---|---|---|---|
| **R1** | **Daily clock and 1-month value horizon.** Decide at the 00:00 UTC close only. Daily open-to-open m. γ_d ∈ {0.95, 0.97} (effective horizon 20–33 days vs 5.5 days now). Hysteresis at c_mult = 1. 24 hourly day-boundary offsets replace the 16 × 15-min phases. Aux heads at 1w and 1m | Théate (γ ↔ turnover); Sepp & Lucic (short-memory alpha < costs; 1–3m spans); Song (choose supervision horizon for SNR); Wood (longer windows win as costs rise); Bysik (execution discipline); predictability note (1m IC +8 to +13) | switches 2–8/yr; exposure 0.65–0.85; IQM Sharpe 1.05–1.20; MDD −60 to −72%; P(improve over P0) ≥ 0.6 | IQM ΔSharpe vs P0 ≤ 0 |
| **R2** | **Drift-neutral value plus an explicit drift prior.** Train U on m − μ̂_pool, so the network learns only the conditional excess. At decision time add κ·H·μ_prior with μ_prior ∈ {0, ½·trailing 4y mean, μ̂_pool}; the last is the P0-equivalent control | P0 autopsy (drift 0.31–1.71 vs threshold 0.30; rep 0's Jan-2025/26 cold rebuilds 99.6% and 99.98% long); Théate (collapse to passive); Sepp & Lucic (drift vs autocorrelation channels; long spans → B&H); pilot B (drift sits in 16% of days, so its estimate is unstable) | with μ_prior = 0: exposure 0.5–0.7, Sharpe 1.00–1.15, MDD −55 to −70%; **the Jan-2025/26 cold rebuilds pass the ≤95%-exposure gate in ≥8/10 reps (both were rejected in the headline rep)**; 2025-26 exposure < 0.8, MDD −35 to −45% (vs −53%); gives up 20–60 pp of CAGR in 2020 and 2023-24. μ_pool reproduces P0 within ±0.03 | μ_prior = 0 Sharpe < B&H − 0.05 |
| **R3** | **Trend backbone plus RL veto** (meta-labeling-style residual policy). Primary p_t = B2 (pre-registered earlier; B5-daily as a secondary check, since it was chosen post-hoc). Action a_t ∈ {0, p_t}: the agent can only stand aside when the rule is long. State adds the rule's signal age and its trailing trade P&L | Lim (put learning inside the TSMOM frame); Wood (add regime info to a momentum backbone); meta-labeling (López de Prado 2018 / Joubert 2022 †); Sepp & Lucic (structural TF alpha); Théate (an unconstrained agent collapses to B&H) | Sharpe within [B2 − 0.05, B2 + 0.10]; MDD no worse than B2's −67% (a veto only removes exposure); exposure 0.45–0.58; veto active on ≤ 25% of rule-long days. Can't fix B2's bad 2021 (−20% vs B&H +59%) | IQM ΔSharpe vs B2 < 0 |
| **R4** | **Direct-policy control (DMN/RRL).** π(x) = sigmoid(w·x), linear or 8 hidden units, on the R6 features. Gradient ascent on the annualized Sharpe (or Sortino) of *net* daily returns with c·\|π_t − π_{t−1}\| in the loss; executed with a 0.1 no-trade band; same walk-forward, 10 seeds | Lim (Sharpe loss; turnover term −5.31 → 0.91 at 10 bp); Moody & Saffell † (DSR); Jiang (explicit reward, full exploitation); Zhang (vol-scaled reward) | with turnover term: Sharpe 1.0–1.2, exposure 0.5–0.8, daily-return correlation with B2/B5 > 0.7. Without it: Sharpe < 0.8 and > 50 switches/yr. **If R4 ≥ R1 on IQM, the TD machinery isn't earning its complexity** | no kill; this is the control |
| **R5** | **Downside-risk reward.** r = κ[a·m − η·a·min(m, 0)²/σ̄], η ∈ {0, 1, 3}, with a total-variance-penalty twin as a negative control | Moody & Saffell † (downside deviation ratio); Lim (Sharpe > return loss); Hêche (CVaR policies); Shu (downside deviation feature); **pilot A** (total vol hurts, downside helps a little) | total-variance twin: Sharpe −0.05 to −0.10. Downside version: Sharpe −0.03 to +0.07, MDD 5–10 pp shallower, CAGR 5–10 pp lower, exposure −5 to −15 pp | downside Sharpe < η = 0 twin |
| **R6** | **Parsimonious multi-speed state (≤ 8 features).** Vol-normalized returns at 1w, 1m, 3m, 6m and 12m; MACD at (8, 24), (16, 48) and (32, 96) days, vol-normalized; daily flow_20; EWM downside deviation (hl 10d). Drop 4h reversal and candle features (clv, ret_1, range, macdh_4h) | Lim, Wood, Zhang (the same state family); Gort (drop \|ρ\| > 0.6); predictability note (4h reversal untradable, the slow block carries the tradable IC); P0 grid (the 8-feature variants averaged Sharpe 1.04 vs 0.96 for the 22-feature variants) | replication spread (sd of rep Sharpe) down ≥ 30%; fewer switches; mean Sharpe +0.00 to +0.08 vs the 22-feature twin. Use as the default state for R1–R5 | spread not reduced |
| **R7** | **Pessimistic entries.** Add 11 quantile heads to U (pinball loss, numpy). Enter on CVaR_α(Δ) with α ∈ {0.3, 0.5}, exit on the mean. Cheaper alternative: Δ − k·sd over the 5 ensemble members | Hêche (IQN_α tunable risk aversion); Huang, Zhang, Théate (distributional RL as future work); Agarwal (variance) | exposure −5 to −20 pp; fewer late-cycle entries; Sharpe ±0.05; MDD 3–8 pp shallower. Low priority | Sharpe < twin − 0.05 |

**Order and compute.** First build P1 = R1 + R6 (daily clock, 8 slow features). Then run R2, R3 and R5 as
single-factor changes on P1, R4 as the non-TD control, and R7 only if R2 or R5 help. That is
6–7 variants × 10 reps, about 70 walk-forward runs against P0's ~49. On the daily clock the training rows
shrink about 6×, so expect roughly 1–2 h on 2 processes. For comparison, P0's full sweep took ~1 h on 4 cores.

**What would change my mind.** If R1 + R6 does not beat P0 on IQM, the problem is the signal itself, not
its horizon; stop and keep B2. If R2 (μ_prior = 0) does not beat B&H, the conditional signal after
2020 is too weak to justify ever leaving the market. If R4 ≥ R1, drop Q-learning.

## 6. Protocol for the next study

- **Benchmark against the trend rule, not B&H.** The primary endpoint should be ΔSharpe vs B2
  (pre-registered in P0), with ΔMDD and ΔCVaR5% as co-primary. With 9.7 years, a ΔSharpe vs B&H needs
  about 0.6 to be significant, and nothing in the literature suggests we can reach that. Drawdown
  endpoints have more power for what these methods actually do.
- **Replications:** report IQM with stratified bootstrap CIs over reps × phases, performance profiles,
  and P(improvement) (Agarwal). Never report the best rep.
- **Selection:** count all variants (N so far = 24 + new). Compute PBO over the R-set with CSCV (Gort;
  Bailey †) and DSR with the updated N.
- **Controls:** re-run the synthetic positive/negative controls for every new reward or objective. A
  risk-sensitive objective must still lose to B&H on GARCH markets with no signal.
- **Clean test:** all of 2014–2026 has now been seen, so the only unbiased evaluation is forward paper
  trading from 2026-09-25 (checkpoints 2027-03-25 and 2027-09-25), with a single variant frozen
  before those dates.

## 7. Deprioritized, with reasons

| idea | why not (now) |
|---|---|
| Return-based regime gate (HMM / jump model) | pilot B: Sharpe 0.26–0.89 < B&H on BTC; the high-return regime is short and detected late |
| Total-volatility targeting of exposure | pilot A: gross Sharpe 0.95 < 1.01; BTC's returns are highest in high-vol states |
| Changepoint / fast-reversion features, 4h or hourly decisions | their gain dies at 2 bp (Wood; Lim); our 4h reversal is untradable at 15 bp |
| Bigger nets (LSTM, Transformer), PPO/SAC/TD3, hyperparameter search | no evidence they help at our costs (Lim: WaveNet < linear; Bysik: XGBoost ≥ neural; Gort: SAC PBO 21%); our PBO is already 0.74 |
| Long/short | spot account; in Bysik, long-short was the fragile leg |
| Trailing stops and monthly re-optimized trend parameters | evidence comes from selection-inflated backtests (AdaptiveTrend) |

## References (read via alphaXiv)

Lim et al. 2019 https://www.alphaxiv.org/abs/1904.04912 · Wood et al. 2021 https://www.alphaxiv.org/abs/2105.13727 ·
Zhang et al. 2019 https://www.alphaxiv.org/abs/1911.10107 · Huang 2018 https://www.alphaxiv.org/abs/1807.02787 ·
Théate & Ernst 2020 https://www.alphaxiv.org/abs/2004.06627 · Ishikawa & Nakata 2021 https://www.alphaxiv.org/abs/2106.03035 ·
Jiang et al. 2017 https://www.alphaxiv.org/abs/1706.10059 · Gort et al. 2022 https://www.alphaxiv.org/abs/2209.05559 ·
Hêche et al. 2025 https://www.alphaxiv.org/abs/2501.04421 · Shu et al. 2024 https://www.alphaxiv.org/abs/2402.05272 ·
Bysik & Ślepaczuk 2026 https://www.alphaxiv.org/abs/2606.00060 · Sepp & Lucic 2026 https://www.alphaxiv.org/abs/2607.19497 ·
Song et al. 2026 https://www.alphaxiv.org/abs/2602.03395 · Agarwal et al. 2021 https://www.alphaxiv.org/abs/2108.13264 ·
Nguyen 2026 https://www.alphaxiv.org/abs/2602.11708 · Zhang 2025 https://www.alphaxiv.org/abs/2510.02986

## Reproduce

```bash
OMP_NUM_THREADS=1 python btc/research/litreview_pilots.py    # ~2.5 min, 1 process -> btc/research/litreview_pilots.json
```
