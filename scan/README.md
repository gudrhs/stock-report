# 일일 스캔 엔진

매일 아침 7시(KST) GitHub Actions가 `daily.py`를 돌려 `../data/`를 갱신하고, 사이트는 그 파일을 읽습니다.

## 파일

| 파일 | 하는 일 |
|---|---|
| `universe.py` | 스캔 대상 350종목 산출 (코스피 시총 200 + 코스닥 시총 150) |
| `indi.py` | 지표 원시 계산 (G·T·B·U·그물망) — 실시간감시와 같은 코드 |
| `engine.py` | 지표를 종목당 한 번만 계산하고 날짜별로 조회 |
| `rules.py` | 매수·매도 규칙 — **자동매매로 옮길 때 이 파일만 보면 됩니다** |
| `daily.py` | 실행 진입점 |

## 실행

```bash
python daily.py                     # 최신 영업일 기준 1회
python daily.py --date 2026-08-11   # 특정 일자
python daily.py --backfill 120      # 최근 120영업일 소급
python daily.py --backfill 120 --reset   # 기록을 지우고 새로 생성
```

## 체결 방식 — 여기가 중요합니다

**종가로 낸 신호는 다음 거래일 시가에 체결**한 것으로 기록합니다. 종가를 보고 그 종가에 살 수는 없기 때문입니다. 이 구분을 하지 않으면 성과가 실제보다 좋게 나옵니다(이 저장소의 6개월 구간에서 약 0.8%p 차이).

**손절만 예외**입니다. 실제 자동매매는 손절주문을 미리 걸어두므로, 그날 저가가 손절선을 건드리면 장중 체결로 처리합니다. 갭하락으로 시가가 이미 손절선 아래면 시가 체결입니다.

하루 처리 순서:

1. 어제 낸 매도 주문 → 오늘 시가 체결
2. 어제 낸 매수 주문 → 오늘 시가 체결
3. 보유 종목 장중 손절 확인 (오늘 저가 기준)
4. 오늘 종가로 지표 계산 → 내일 낼 주문 결정

## 출력

| 파일 | 내용 |
|---|---|
| `data/signals.csv` | **매매 로그** — append-only, 30개 컬럼. 자동매매 재현용 |
| `data/d/YYYY-MM-DD.json` | 그날 스냅샷 (매수 후보 전체 + 체결 + 보유) |
| `data/latest.json` | 가장 최근 날짜 스냅샷 + 누적 성과 |
| `data/dates.json` | 기록이 있는 날짜 목록 |
| `data/positions.json` | 현재 보유 중인 가상 포지션 |
| `data/pending.json` | 다음 거래일 시가에 낼 주문 |
| `data/stats.json` | 누적 성과 요약 |

### signals.csv 컬럼

`date, code, name, market, action, severity, reason, price, score, rank, entry_date, entry_price, days_held, pnl_pct, gbuy, gsell, gs_hold, gs_mean, gs_std, gb_slope5, t, b, u, red, ma30, chg1, chg5, chg20, vol, amount`

`action`은 `BUY` / `SELL` / `HOLD`입니다. `HOLD`는 그날 보유를 유지한 기록이라, 이것만 이어 붙이면 포지션의 일별 궤적이 그대로 복원됩니다.

## 설정

`daily.py`의 `CFG`

```python
MAX_POS=10       # 동시 보유 종목 수
MIN_AMOUNT=3e8   # 최소 거래대금
TOP_SHOW=40      # 화면에 싣는 매수 후보 수
```

`rules.py`의 `SELL`

```python
STOP_LOSS=-7.0   # 손절선 (장중)
GS_EXIT=5.0      # 청선 이탈 기준
GS_HARD=8.0      # 즉시 청산
BIG_DOWN=-4.0    # 장대음봉
GB_DROP=-3.0     # 적선 기울기 꺾임
TRAIL=-8.0       # 고점 대비 되밀림
MAX_DAYS=40      # 최대 보유일
```

매수 조건은 `indi.py`의 `CFG`에 있습니다.

## 알아둘 한계

- **종목 선정** — KRX가 지수 구성종목을 공개 API로 주지 않아 시가총액 상위로 대신합니다. 실제 코스피200·코스닥150과 대부분 겹치지만 같지는 않습니다.
- **소급 기록의 편향** — 과거 날짜도 *현재* 시가총액 상위 종목으로 계산했습니다. 당시 지수에 없던 종목이 섞이고, 그동안 밀려난 종목은 빠져 있습니다(생존 편향). 앞으로 매일 쌓이는 기록에는 이 문제가 없습니다.
- **수수료·세금 미반영** — 실제 수익률은 이보다 낮습니다.
- **거래량 제약 미반영** — 시가 전량 체결을 가정합니다.
