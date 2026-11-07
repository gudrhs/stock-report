# -*- coding: utf-8 -*-
"""
새 강화학습 4가지 묶음 판정기 — btc/research/success_criteria.md 마지막 조항
'새 강화학습 방식 4가지 묶음'(커밋 441be61, 결과 보기 전 등록) 그대로. 값을 조정하지 않습니다.

대상
  · Q1_qrdqn_cvar, A1_actor_critic, P1_pessimistic_dqn — 반복 0~4 (walk.load_run(이름, r) / 이름 + '__early')
  · C1_r6_committee10 — 새 학습 없음. R6 반복 0~9의 판단을 합의
  · 기준 R6_daily_trend8_uniform — 반복 0~9 (두 구간 모두)

판단·체결 (btc.research.screen.targets / btc.research.early.targets 와 같은 계산)
  · 판단봉 = walk.decision_mask(d, stride) (stride 6 → phase 0의 00:00 UTC 마감 봉), 다음 봉 시가 체결, 편도 0.15%
  · 행동 가치(U) 모델: k_policy(U[판단비용 0.3%], acts, 0.003, forced_hold) → 목표 비중. U가 NaN인 봉은 목표 NaN(유지)
  · 비중 출력 모델(cfg['output'] == 'weights'): 비중을 25% 단위로 반올림
  · 행동에 0·1 말고 다른 값이 있거나 비중 출력이면 비율 회계(simulate_weights), 아니면 0/1 회계(simulate)
  · 설정은 실행 파일에 저장된 cfg(meta)를 쓰고, 없으면 variants.VARIANTS에서 찾습니다

C1 합의 (등록 문구: "6개 이상이 보유면 보유, 4개 이하면 현금, 5:5면 직전 포지션 유지")
  · 판단봉마다 R6 반복 0~9 각자의 0/1 포지션 = 그 반복의 U로 돌린 k_policy 출력 (반복마다 자기 이전 포지션을 이어 감)
    U가 NaN(모델 없음)인 봉은 k_policy가 그 반복의 직전 포지션을 유지하므로, 그 반복이 실제로 들고 있는 포지션이 표가 됨
    (첫 모델 전에는 현금 = 0표)
  · 보유 표 ≥ 6 → 보유, ≤ 4 → 현금, 정확히 5 → 합의의 직전 포지션 유지 (시작은 현금). 0/1 전략으로 시뮬레이션
  · '직전 포지션' = 합의가 실제로 든 포지션: 강제 유지 봉(forced_hold)에서는 체결이 없으므로(env.simulate) 합의도
    k_policy 처럼 바꾸지 않습니다 — 들지 않은 보유를 5:5로 '유지'하는 일이 없게

헤드라인과 R6 기준
  · Q1·A1·P1: 반복 5개 가운데 매수·보유 대비 샤프 차이의 작은 쪽 중앙값 반복 (evaluate.lower_median)
  · 이들의 R6 기준 = R6 반복 0~4의 작은 쪽 중앙값 (등록값: 2017~2026 1.149, 2015~2016 1.90 — 다시 계산해 대조)
  · C1의 R6 기준 = R6 반복 0~9 샤프의 중앙값 (np.median, 10개면 5·6번째 평균)
    C1의 R6 대비 p값은 한 반복의 일별 수익이 필요해, 두 가운데 반복 중 샤프가 높은 쪽(위쪽 중앙값, 6번째)과 비교합니다
    (보수적 선택 — 결과 보기 전 여기서 고정)
  · 문턱은 '다시 계산한 반올림 전 R6 값'입니다 (등록 문구의 1.149·1.90은 이 값을 반올림해 적은 것).
    반올림한 등록값으로 판정하면 결과가 달라지는 경우(예: 2015~2016 헤드라인이 1.90 초과 1.9044 이하)는
    criteria 의 verdict_differs_registered 로 표시하고 출력에 적습니다. 판정 자체는 반올림 전 값(더 엄격)으로 합니다

판정 (두 구간 모두)
  · 헤드라인 샤프 > R6 기준, 그리고 헤드라인 샤프 > 매수·보유 샤프 → 성적 조건 통과
  · 합성 대조 통과도 필요: controls.json · controls_weights.json 의 양성·음성 통과. C1·P1은 R6의 결과로 갈음.
    대조 결과가 아직 없으면 '미완'(candidate = None)으로 적습니다
  · 함께 보고: R6 대비 p, 매수·보유 대비 p (btc.stats.stationary_bootstrap_diff, 일별 수익, 시드 1, 4000번),
    샤프·CAGR·최대낙폭·노출·회전율, 샤프 차이의 디플레이티드 검정 (robust.py 와 같은 식, N 50·200·1000,
    시험 간 샤프 표준편차 0.109) — 매수·보유 대비(요청대로)와 R6 대비(등록 문구) 둘 다.
    등록 문구 '시험 수 N 갱신'에 따라 갱신한 N(아래)에서의 값도 함께 적습니다
  · 시험 수 N = walk.n_trials_total() + (trials.jsonl 에 이름이 없는 이 묶음의 방식 수).
    C1은 학습하지 않아 walk.run_many 가 기록하지 않으므로 여기서 1을 더합니다 ('네 가지 모두 시험 수에 더합니다')
  · 개선 후보는 보조 모의매매 트랙으로만. 주 후보는 R6 유지

  BTC_LOCKBOX_OPEN=1 python -m btc.research.more_rl evaluate
    → data/btc/research_runs/more_rl.json, 한국어 표 출력. 실행 결과가 빠져 있으면 무엇이 없는지 적고 멈춤.
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import json
import math

import numpy as np

from .. import stats as S
from ..evaluate import Window, lower_median
from .rl import k_policy, trend_filter, allowed_matrix
from .walk import load_run, run_path, decision_mask, n_trials_total, RUNS, TRIALS

COST, C_DEC = 0.0015, 0.003
REF = "R6_daily_trend8_uniform"
COMMITTEE = "C1_r6_committee10"
METHODS = ("Q1_qrdqn_cvar", "A1_actor_critic", "P1_pessimistic_dqn")
ORDER = (COMMITTEE,) + METHODS
REPS, REF_REPS = 5, 10
VOTE_LONG, VOTE_FLAT = 6, 4                     # 10표 가운데 ≥6 보유, ≤4 현금, 5는 유지
WINDOWS = {                                     # 이름: (시작, 끝, 실행 이름 꼬리표, 잠금 감사 기록용 설명)
    "main": ("2017-01-01", "2026-09-25", "", "research-morerl:main 2017-01-01..2026-09-25"),
    "early": ("2015-01-01", "2017-01-01", "__early", "research-morerl:early 2015-01-01..2017-01-01"),
}
WINDOW_KO = {"main": "2017-01~2026-09", "early": "2015~2016"}
REF_REGISTERED = {"main": (1.149, 3), "early": (1.90, 2)}      # (등록값, 등록에 적힌 소수 자리)
CTRL_SOURCE = {COMMITTEE: REF, "P1_pessimistic_dqn": REF,        # R6 모델을 그대로 쓰므로 R6 대조로 갈음
               "Q1_qrdqn_cvar": "Q1_qrdqn_cvar", "A1_actor_critic": "A1_actor_critic"}
_HERE = os.path.dirname(os.path.abspath(__file__))
CONTROL_FILES = (os.path.join(_HERE, "controls.json"), os.path.join(_HERE, "controls_weights.json"))
DSR_N = (50, 200, 1000)
DSR_SD = 0.109
N_BOOT, BOOT_SEED = 4000, 1
OUT = os.path.join(RUNS, "more_rl.json")


# ══════════ 필요한 실행 확인 ══════════
class MissingRuns(RuntimeError):
    """판정에 필요한 워크포워드 실행 결과(npz)가 없음"""


def required_runs():
    """(실행 이름, 반복) 목록 — 두 구간 × (Q1·A1·P1 반복 0~4, R6 반복 0~9)"""
    out = []
    for _, (_, _, tag, _) in WINDOWS.items():
        for n in METHODS:
            out += [(n + tag, r) for r in range(REPS)]
        out += [(REF + tag, r) for r in range(REF_REPS)]
    return out


def missing_runs(exists=None):
    exists = exists or (lambda n, r: os.path.exists(run_path(n, r)))
    return [(n, r) for n, r in required_runs() if not exists(n, r)]


def check_runs(exists=None):
    miss = missing_runs(exists)
    if not miss:
        return
    by = {}
    for n, r in miss:
        by.setdefault(n, []).append(r)
    lines = [f"  {n}: 반복 {', '.join(str(r) for r in rs)}  ({os.path.dirname(run_path(n, 0))})" for n, rs in by.items()]
    raise MissingRuns(f"판정에 필요한 실행 결과 {len(miss)}개가 없습니다 — 먼저 학습을 돌리세요:\n" + "\n".join(lines))


# ══════════ 포지션 ══════════
def cfg_of(name, run):
    """실행 파일에 저장된 설정(실제로 돌린 것). 없으면 variants.VARIANTS"""
    cfg = (run.get("meta") or {}).get("cfg")
    if cfg:
        return cfg
    from .variants import VARIANTS
    return VARIANTS[name]


def is_weights(cfg):
    return cfg.get("algo") == "direct" or cfg.get("output") == "weights"


def rep_targets(d, cfg, run, a, b):
    """
    screen.targets / early.targets 와 같은 계산 (판단비용 0.3%).
    반환 (목표 비중 tg[T], 비율 회계 여부, 반복이 실제로 든 포지션 held[판단봉] 또는 None, 판단봉, 모델 없음 비율)
    """
    sel = np.arange(a, b)[decision_mask(d, cfg.get("stride", 1))[a:b]]
    tg = np.full(d.T, np.nan)
    if is_weights(cfg):
        w = run["U"][0.0][sel, 0]
        tg[sel] = np.round(w * 4) / 4
        return tg, True, None, sel, float(np.isnan(w).mean())
    acts = np.asarray(cfg.get("acts", (0.0, 1.0)), dtype=float)
    U = run["U"][C_DEC][sel]
    allowed = allowed_matrix(trend_filter(d.c)[sel], acts) if cfg.get("veto_b2") else None
    held = acts[k_policy(U, acts, C_DEC, d.forced_hold[sel], allowed=allowed)].astype(float)
    nan = np.isnan(U).any(axis=1)
    w = held.copy()
    w[nan] = np.nan
    tg[sel] = w
    frac = bool(np.any((acts > 0) & (acts < 1))) or bool(np.any(acts < 0))
    return tg, frac, held, sel, float(nan.mean())


def committee(votes, forced_hold=None, long_at=VOTE_LONG, flat_at=VOTE_FLAT, start=0.0):
    """
    votes: (판단봉 수, 반복 수) 0/1 포지션 → 합의 포지션 (판단봉 수,).
    보유 표 ≥ long_at → 1, ≤ flat_at → 0, 그 사이(10표면 정확히 5) → 직전 합의 유지 (시작 start)
    forced_hold: (판단봉 수,) — 참인 봉은 체결이 없으므로(env.simulate) 합의도 바꾸지 않음 (k_policy 와 같은 처리).
      그래서 '직전 합의' = 합의가 실제로 들고 있는 포지션
    """
    votes = np.asarray(votes, dtype=float)
    if votes.ndim != 2:
        raise ValueError("votes 는 (판단봉, 반복) 2차원이어야 합니다")
    if forced_hold is not None:
        forced_hold = np.asarray(forced_hold, dtype=bool)
        if forced_hold.shape != (len(votes),):
            raise ValueError("forced_hold 길이가 판단봉 수와 다릅니다")
    if not np.all((votes == 0.0) | (votes == 1.0)):
        raise ValueError("합의에는 0/1 포지션만 씁니다 (NaN·비율 불가)")
    n_long = votes.sum(axis=1)
    out = np.empty(len(n_long))
    p = float(start)
    for t, k in enumerate(n_long):
        if forced_hold is not None and forced_hold[t]:
            pass
        elif k >= long_at:
            p = 1.0
        elif k <= flat_at:
            p = 0.0
        out[t] = p
    return out


# ══════════ 통계 ══════════
def headline_index(sharpes, bh_sharpe):
    """매수·보유 대비 샤프 차이의 작은 쪽 중앙값 반복 (5개면 3번째, 10개면 5번째)"""
    return lower_median([s - bh_sharpe for s in sharpes])


def upper_median_index(sharpes):
    """위쪽 중앙값 반복 (10개면 6번째로 작은 것)"""
    return int(np.argsort(sharpes, kind="stable")[len(sharpes) // 2])


def metrics(res):
    """샤프·CAGR·최대낙폭·노출(평균 보유 비중)·연 회전율(바꾼 비중 합 / 연수)"""
    sm = S.summary(res["r"])
    pos = np.asarray(res["pos"], dtype=float)
    dp = np.abs(np.diff(np.concatenate([[0.0], pos])))
    if res.get("rebal") is not None:
        # 비율 회계: 재조정한 봉의 |Δ비중|만 (그 밖의 변화는 가격 표류). 한 봉 사이 표류는 무시한 근사
        dp = dp * np.asarray(res["rebal"], dtype=bool)
    years = len(res["r"]) / 365.0
    return dict(sharpe=sm["sharpe"], cagr=sm["cagr"], max_dd=sm["max_dd"], exposure=float(pos.mean()),
                turnover=float(dp.sum() / years) if years > 0 else float("nan"))


def boot(ra, rb):
    return S.stationary_bootstrap_diff(ra, rb, n_boot=N_BOOT, seed=BOOT_SEED)


def dsr_ns(n_trials=None):
    """고정 격자 50·200·1000 + 갱신한 시험 수 N (격자에 없을 때만)"""
    ns = tuple(DSR_N)
    if n_trials and int(n_trials) not in ns:
        ns = ns + (int(n_trials),)
    return ns


def dsr_gain(bt, ns=DSR_N, sd=DSR_SD):
    """
    샤프 '차이'에 대한 디플레이티드 검정 (robust.py 와 같은 식): 필요한 차이 = sd·√365·E[max SR](분산 1/365, N)
    = sd × (N개 표준정규 최댓값의 기대값). DSR = Φ((관측 차이 − 필요한 차이) / 부트스트랩 표준오차)
    """
    out = {}
    for n in ns:
        thr = sd * math.sqrt(365.0) * S.expected_max_sharpe(1.0 / 365.0, n)
        se = bt["se"]
        z = (bt["obs"] - thr) / se if se > 0 else (math.inf if bt["obs"] > thr else -math.inf)
        out[f"N{n}"] = dict(threshold=thr, dsr=S.norm_cdf(z) if math.isfinite(z) else float(z > 0))
    return out


def compare(res, ref_res, ref_sharpe, bh_res, bh_sharpe, ns=DSR_N):
    """한 구간에서 헤드라인 결과를 R6 기준·매수·보유와 비교. ns: 디플레이티드 검정의 시험 수들"""
    m = metrics(res)
    bb, br = boot(res["r"], bh_res["r"]), boot(res["r"], ref_res["r"])
    m.update(ref_sharpe=float(ref_sharpe), bh_sharpe=float(bh_sharpe),
             d_vs_r6=m["sharpe"] - float(ref_sharpe), d_vs_bh=bb["obs"],
             p_vs_r6=br["p"], d_boot_vs_r6=br["obs"], se_vs_r6=br["se"],
             p_vs_bh=bb["p"], se_vs_bh=bb["se"], ci90_vs_bh=bb["ci90"], ci90_vs_r6=br["ci90"],
             dsr_gain_vs_bh=dsr_gain(bb, ns), dsr_gain_vs_r6=dsr_gain(br, ns))
    return m


# ══════════ 판정 ══════════
NO_MODEL_MAX = 0.05                              # early.py 와 같은 규칙: 모델 없는 판단봉 비율이 5% 넘는 반복이 있으면 판정 불가


def criteria(per_window, controls_passed, registered=None, ref_basis=None, no_model_max=None):
    """
    per_window: {"main": {sharpe, ref_sharpe, bh_sharpe, p_vs_r6}, "early": {...}} (두 구간 모두 있어야 함)
    controls_passed: True / False / None(대조 결과 없음)
    registered: {"main": 1.149, "early": 1.90} — 등록 문구에 반올림해 적힌 R6 값 (Q1·A1·P1). 판정은 ref_sharpe
      (다시 계산한 반올림 전 값)로 하고, 이 값으로 했다면 성적 조건이 달라지는지만 verdict_differs_registered 로 표시.
      C1처럼 등록값이 없으면 None
    ref_basis: 문턱이 무엇인지 적는 설명 (결과 파일·출력용)
    no_model_max: 두 구간 반복들 가운데 가장 큰 '모델 없음' 비율. NO_MODEL_MAX 초과면 판정 불가(candidate None)
      — 2026-09-25 해석 정정 조항(결과 보기 전)에서 early.py 규칙을 이 묶음에도 적용하기로 함
    """
    if set(per_window) != set(WINDOWS):
        raise ValueError(f"두 구간 {sorted(WINDOWS)} 이 모두 있어야 합니다: {sorted(per_window)}")
    beats_r6 = {k: bool(v["sharpe"] > v["ref_sharpe"]) for k, v in per_window.items()}
    beats_bh = {k: bool(v["sharpe"] > v["bh_sharpe"]) for k, v in per_window.items()}
    perf = all(beats_r6.values()) and all(beats_bh.values())
    if registered is not None:
        beats_reg = {k: bool(v["sharpe"] > float(registered[k])) for k, v in per_window.items()}
        perf_reg = all(beats_reg.values()) and all(beats_bh.values())
        differs = bool(perf_reg != perf)
    else:
        beats_reg, differs = None, False
    flagged = no_model_max is not None and no_model_max > NO_MODEL_MAX
    if flagged:
        cand, verdict = None, f"판정 불가 (모델 없음 비율 {no_model_max * 100:.1f}% > {NO_MODEL_MAX * 100:.0f}%)"
    elif not perf:
        cand, verdict = False, "개선 후보 아님"
    elif controls_passed is None:
        cand, verdict = None, "성적 조건 통과 — 합성 대조 결과 없음 (미완)"
    elif not controls_passed:
        cand, verdict = False, "성적 조건 통과, 합성 대조 탈락 → 개선 후보 아님"
    else:
        cand, verdict = True, "개선 후보 (보조 모의매매 트랙으로만, 주 후보는 R6 유지)"
    ps = [v.get("p_vs_r6") for v in per_window.values()]
    sig = None if any(p is None for p in ps) else bool(all(p < 0.05 for p in ps))
    if differs:
        verdict += (" (주의: 등록 문구의 반올림 값 기준이었다면 성적 조건 "
                    + ("통과" if not perf else "탈락") + " — 판정은 반올림 전 값 기준)")
    thresholds = {k: float(v["ref_sharpe"]) for k, v in per_window.items()}
    return dict(beats_r6=beats_r6, beats_bh=beats_bh, performance_ok=bool(perf), controls_passed=controls_passed,
                candidate=cand, verdict=verdict, p_vs_r6_below_005_both=sig,
                r6_threshold=thresholds, r6_threshold_basis=ref_basis, r6_registered=registered,
                beats_r6_registered=beats_reg, verdict_differs_registered=differs,
                no_model_max=no_model_max, no_model_flag=bool(flagged))


REF_BASIS = {COMMITTEE: "R6 반복 0~9 샤프의 중앙값 (다시 계산, 반올림 전)",
             **{n: "R6 반복 0~4 작은 쪽 중앙값 샤프 (다시 계산, 반올림 전 — 등록 문구 1.149·1.90은 이 값의 반올림)"
                for n in METHODS}}


def registered_refs(name):
    """등록 문구에 숫자로 적힌 R6 기준 (Q1·A1·P1만; C1의 중앙값은 등록 문구에 숫자가 없음)"""
    return None if name == COMMITTEE else {k: v[0] for k, v in REF_REGISTERED.items()}


def n_trials_updated(base=None, trials=TRIALS, names=ORDER):
    """
    갱신한 시험 수 N = walk.n_trials_total() + (trials.jsonl 에 이름이 없는 이 묶음 방식 수).
    C1은 학습하지 않아 run_many 가 기록하지 않음 → 여기서 더함. 반환 (N, 더한 이름 목록). base 계산 실패면 (None, [])
    """
    if base is None:
        try:
            base = n_trials_total()
        except Exception:
            return None, []
    logged = set()
    if trials and os.path.exists(trials):
        with open(trials, encoding="utf-8") as f:
            logged = {json.loads(l).get("name", "").split("__")[0] for l in f if l.strip()}
    added = [n for n in names if n not in logged]
    return int(base) + len(added), added


def load_controls(paths=CONTROL_FILES):
    out = {}
    for p in paths:
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                out.update(json.load(f))
    return out


def controls_status(name, controls):
    src = CTRL_SOURCE[name]
    c = controls.get(src)
    if not c:
        return dict(source=src, positive=None, negative=None, passed=None)
    pos, neg = c.get("positive_pass"), c.get("negative_pass")
    passed = None if (pos is None or neg is None) else bool(pos and neg)
    return dict(source=src, positive=pos, negative=neg, passed=passed)


# ══════════ 한 구간 ══════════
def evaluate_window(W, tag, loader=load_run, ns=DSR_N):
    """
    W: evaluate.Window. tag: 실행 이름 꼬리표('' 또는 '__early'). loader(이름, r) → walk.load_run 과 같은 dict.
    ns: 디플레이티드 검정의 시험 수들 (dsr_ns)
    """
    d = W.datas[0]
    a, b = W.rng[0]
    bh_tg = np.full(d.T, np.nan)
    bh_tg[a:b] = 1.0                                   # evaluate.baseline_targets(W)["B0"] 와 같음
    bh = W.run(bh_tg, COST)
    sb = S.sharpe(bh["r"])

    def run_rep(name, r):
        run = loader(name + tag, r)
        cfg = cfg_of(name, run)
        tg, frac, held, sel, ns = rep_targets(d, cfg, run, a, b)
        res = W.run(tg, COST, weights=True) if frac else W.run(tg, COST)
        return dict(res=res, held=held, sel=sel, frac=frac, nan_share=ns, sharpe=S.sharpe(res["r"]))

    # ── R6 기준 ──
    r6 = [run_rep(REF, r) for r in range(REF_REPS)]
    s6 = [x["sharpe"] for x in r6]
    lm6 = headline_index(s6[:REPS], sb)
    med10 = float(np.median(s6))
    up10 = upper_median_index(s6)
    out = dict(range=[W.lo, W.hi], n_days=int(len(bh["r"])), bh=metrics(bh),
               r6=dict(sharpe_reps=s6, headline_rep=int(lm6), lower_median5=s6[lm6], median10=med10,
                       upper_median_rep10=up10, headline=metrics(r6[lm6]["res"]),
                       no_model_share=[x["nan_share"] for x in r6]),
               methods={})

    # ── Q1·A1·P1 ──
    for name in METHODS:
        reps = [run_rep(name, r) for r in range(REPS)]
        srs = [x["sharpe"] for x in reps]
        lm = headline_index(srs, sb)
        v = compare(reps[lm]["res"], r6[lm6]["res"], s6[lm6], bh, sb, ns)
        v.update(sharpe_reps=srs, headline_rep=int(lm), fractional=bool(reps[lm]["frac"]),
                 no_model_share=[x["nan_share"] for x in reps])
        out["methods"][name] = v

    # ── C1 합의 ──
    sel = r6[0]["sel"]
    if any(not np.array_equal(x["sel"], sel) for x in r6) or any(x["held"] is None or x["frac"] for x in r6):
        raise ValueError("R6 반복들의 판단봉이 다르거나 0/1 포지션이 아닙니다 — 합의 불가")
    votes = np.stack([x["held"] for x in r6], axis=1)
    cpos = committee(votes, d.forced_hold[sel])
    tg = np.full(d.T, np.nan)
    tg[sel] = cpos
    res = W.run(tg, COST)
    v = compare(res, r6[up10]["res"], med10, bh, sb, ns)
    n_long = votes.sum(axis=1)
    v.update(ref_rule="R6 반복 0~9 샤프의 중앙값", p_ref_rep=up10,
             tie_share=float(np.mean(n_long == REF_REPS / 2)),
             forced_hold_bars=int(d.forced_hold[sel].sum()),
             unanimous_share=float(np.mean((n_long == 0) | (n_long == REF_REPS))),
             committee_long_share=float(cpos.mean()) if len(cpos) else float("nan"))
    out["methods"][COMMITTEE] = v
    return out


# ══════════ 전체 ══════════
def ref_check(key, value):
    reg, digits = REF_REGISTERED[key]
    return dict(registered=reg, computed=float(value), matches=bool(round(float(value), digits) == reg))


def evaluate(loader=load_run, exists=None, datas=None, controls=None):
    check_runs(exists)                                 # 데이터를 읽기 전에 먼저
    n_trials, added = n_trials_updated()
    ns = dsr_ns(n_trials)
    if datas is None:
        from ..walkforward import load_phases
        datas = load_phases()
    wins = {}
    for key, (lo, hi, tag, what) in WINDOWS.items():
        W = Window(datas, lo, hi, what)                # 2017~2026 은 BTC_LOCKBOX_OPEN=1 필요 (evaluate.guard)
        wins[key] = evaluate_window(W, tag, loader, ns)
        wins[key]["r6"]["ref_check"] = ref_check(key, wins[key]["r6"]["lower_median5"])
    controls = load_controls() if controls is None else controls
    methods = {}
    for name in ORDER:
        pw = {k: wins[k]["methods"][name] for k in WINDOWS}
        ctl = controls_status(name, controls)
        nm = [max(v["no_model_share"]) if name != COMMITTEE else max(wins[k]["r6"]["no_model_share"])
              for k, v in pw.items()]
        methods[name] = dict(windows=pw, controls=ctl,
                             criteria=criteria(pw, ctl["passed"], registered_refs(name), REF_BASIS[name],
                                               no_model_max=float(max(nm))))
    return dict(cost=COST, c_dec=C_DEC, n_boot=N_BOOT, boot_seed=BOOT_SEED, dsr_sd=DSR_SD, n_trials=n_trials,
                n_trials_added=added, dsr_ns=list(ns),
                windows={k: {kk: vv for kk, vv in w.items() if kk != "methods"} for k, w in wins.items()},
                methods=methods)


# ══════════ 출력 ══════════
def report(out):
    pct = lambda x: "—" if x is None else f"{x * 100:.1f}%"
    f2 = lambda x: "—" if x is None else f"{x:.2f}"
    lines = [f"새 강화학습 4가지 묶음 판정 (편도 {out['cost'] * 100:.2f}%, 하루 판단, 다음 봉 시가 체결, "
             f"부트스트랩 {out['n_boot']}번·시드 {out['boot_seed']}, 시험 수 N={out['n_trials']}"
             + (f" — 기록에 없던 {', '.join(out['n_trials_added'])} 포함" if out.get("n_trials_added") else "") + ")",
             "R6 기준 문턱 = 다시 계산한 반올림 전 값 (Q1·A1·P1: 반복 0~4 작은 쪽 중앙값, C1: 반복 0~9 중앙값). "
             "등록 문구의 1.149·1.90은 그 반올림"]
    ns = [int(n) for n in (out.get("dsr_ns") or DSR_N)]
    for key, w in out["windows"].items():
        r6, bh = w["r6"], w["bh"]
        rc = r6["ref_check"]
        lines.append(f"\n[{WINDOW_KO[key]}] 매수·보유 샤프 {f2(bh['sharpe'])} CAGR {pct(bh['cagr'])} 최대낙폭 {pct(bh['max_dd'])}")
        lines.append(f"  R6 반복 0~4 작은 쪽 중앙값 {r6['lower_median5']:.3f} (등록값 {rc['registered']}"
                     f"{'' if rc['matches'] else ' — 다름! 실행을 확인하세요'}), 반복 0~9 중앙값 {r6['median10']:.3f} "
                     f"[{', '.join(f'{s:.2f}' for s in r6['sharpe_reps'])}]")
        lines.append(f"  {'이름':<22}{'샤프':>6}{'R6 기준':>8}{'차이':>7}{'p(R6)':>7}{'p(보유)':>8}"
                     f"{'CAGR':>8}{'최대낙폭':>9}{'노출':>6}{'회전/년':>8}  DSR(보유 대비, N={'/'.join(str(n) for n in ns)})")
        for name in ORDER:
            v = out["methods"][name]["windows"][key]
            g = v["dsr_gain_vs_bh"]
            reps = v.get("sharpe_reps")
            lines.append(f"  {name:<22}{v['sharpe']:>6.2f}{v['ref_sharpe']:>8.3f}{v['d_vs_r6']:>+7.2f}{v['p_vs_r6']:>7.2f}"
                         f"{v['p_vs_bh']:>8.2f}{pct(v['cagr']):>8}{pct(v['max_dd']):>9}{pct(v['exposure']):>6}"
                         f"{v['turnover']:>8.1f}  " + "/".join(f"{g[f'N{n}']['dsr']:.2f}" for n in ns if f"N{n}" in g)
                         + (f"  반복 {min(reps):.2f}~{max(reps):.2f}" if reps else
                            f"  5:5 비율 {pct(v['tie_share'])}"))
    lines.append("\n판정 (두 구간 모두 R6 기준·매수·보유보다 높고, 합성 대조 통과)")
    for name in ORDER:
        m = out["methods"][name]
        c, ctl = m["criteria"], m["controls"]
        ok = lambda x: "○" if x else "×"
        lines.append(f"  {name:<22}{c['verdict']} | R6보다 높음: "
                     + ", ".join(f"{WINDOW_KO[k]} {ok(c['beats_r6'][k])}" for k in WINDOWS)
                     + " | 매수·보유보다 높음: " + ", ".join(f"{WINDOW_KO[k]} {ok(c['beats_bh'][k])}" for k in WINDOWS)
                     + f" | 합성 대조({ctl['source']}): {'통과' if ctl['passed'] else ('없음' if ctl['passed'] is None else '탈락')}"
                     + ("" if not c["performance_ok"] else
                        f" | R6 대비 p<0.05 두 구간 모두: {'예' if c['p_vs_r6_below_005_both'] else '아니오 (유의하지 않음)'}"))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("evaluate")
    ap.parse_args()
    from ..evaluate import _clean
    try:
        out = _clean(evaluate())
    except MissingRuns as e:
        raise SystemExit(str(e))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(report(out))
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
