# BTC 무기한 선물 펀딩비 (8시간) — 출처와 가공

파일: `btc_perp_funding_8h.csv` (14,964행, 약 0.5 MB)

| 열 | 뜻 |
|---|---|
| `ts_utc_seconds` | 펀딩 정산 시각 (UTC, 초). 이 시각에 포지션을 들고 있으면 정산 |
| `funding_rate_per_8h` | 8시간 펀딩률 (소수, 0.0001 = 0.01%). **+면 롱이 숏에게 지급** |
| `source` | `bitmex_XBTUSD` (인버스, 2016-06-04 20:00 ~ 2023-03-06 12:00, 정산 04/12/20시) · `binance_BTCUSDT` (USDT 선형, 2019-09-10 08:00 ~ 2026-08-06 00:00, 정산 00/08/16시) |

두 거래소가 겹치는 2019-09 ~ 2023-03에는 두 행이 모두 있습니다 (source로 구분).
2026-08-06 이후 ~ 2026-09-25는 **없음** (접근 가능한 공개 사본이 거기까지).

## 출처 (거래소 API 직접 접근은 이 머신에서 차단 → 공개 GitHub 사본 사용, 2026-09-25 수집)

| 구간 | 저장소 @커밋 | 파일 |
|---|---|---|
| Binance 2019-09-10 ~ 2026-06-30 | `VivekWar/Crypto-Funding-Rate-Arbitrage` @dcbe602 | `notebooks/BTCUSDT_historical_funding.csv` |
| Binance 2026-07-01 ~ 2026-08-06 | `kipopopo/autonomous-futures-bot` @3408be6 | `research/data/BTCUSDT-funding.csv` |
| BitMEX 2016-06 ~ 2023-03 | `hyxxsfwy/Get-Perpetual-Funding-Rate` @6b14d14 | `data/XBTUSD-funding-rate.csv` |

원자료는 각각 Binance `fapi/v1/fundingRate`, BitMEX `Funding` API를 받은 것 (저장소의 수집 스크립트로 확인).

## 가공
- 시각은 가장 가까운 정시로 반올림 (Binance fundingTime의 ms 오차 제거), 중복 제거, 8시간 간격 누락 없음 확인.
- BitMEX 파일은 수집 스크립트가 timestamp를 **−4시간** 옮긴 뒤 +08:00로 저장 → +4시간 되돌려 실제 정산 시각(04/12/20 UTC)으로 복원.

## 교차 확인
- Binance: 서로 다른 GitHub 사본 5개 (위 2개 + `hyxxsfwy`, `olaxbt/ai-market-maker`, `huiiisiii/IS424-btc-prediction`)가 겹치는 구간 전 행 **완전 일치** (차이 0).
- Hugging Face `Torch-Trade/btcusdt_perp_funding_8h_05_2021_to_02_2026` 카드의 통계 (2021-05-01 ~ 2026-02-28, 5,295행, 평균 0.00007416, 중앙값 0.00007826, 최소 −0.00119172, 최대 0.00098653, 음수 13.5%, 연율 8.12%)와 **소수 8자리까지 일치**.
  (Hugging Face 파일 자체는 이 머신의 프록시가 huggingface.co 접속을 막아 받지 못함. `linxy/USDT-M_Perpetual_Futures`의 `BTCUSDT/BTCUSDT_fundingRate.parquet`는 2026-09-23까지 있어 나중에 부족분을 채울 수 있음.)
- BitMEX: 독립 사본은 하나뿐. 겹치는 기간 Binance와 월평균 상관 0.87 (BitMEX가 평균 연 9.7%p 낮음 — 인버스 계약 특성). 다른 GitHub 사본 `horvaj6066/.../BitMEX_Funding_Data.csv`(2019-01~06, 500행)는 값이 서로 맞지 않고 (상관 −0.03, 2019-06 급등기에 −0.375% 연속) 믿기 어려워 쓰지 않음.

## 백테스트에서 쓰는 법 (longshort.simulate_ls의 carry 규약: +면 숏이 냄)
봉(4h)마다 그 봉이 속한 8시간 구간의 펀딩(봉 시작 뒤 첫 정산)을 찾아
`carry_annual[bar] = −funding_rate_per_8h × BARS_PER_YEAR / 2` 로 넣으면 봉당 −f/2가 됩니다.
