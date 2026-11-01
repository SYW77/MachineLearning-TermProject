# 전처리와 데이터 분할 상세

README의 "전처리와 데이터 분할" 섹션을 보충하는 문서입니다. 코드는 `src/preprocess.py`, `run_preprocess.py`, `run_eda.py`에 있습니다.

## 설정값 (`src/preprocess.py` 맨 위)
| 이름 | 현재 값 | 의미 |
|---|---|---|
| `MIN_USER_RATINGS` | 3 | 평점이 이 개수 이상인 사용자만 valid·test를 뗍니다. `N_VALID + N_TEST + 1` 이상이어야 합니다 |
| `KEEP_SPARSE_USERS_IN_TRAIN` | True | 평점이 `MIN_USER_RATINGS` 미만인 사용자를 train에 남깁니다(False면 제외) |
| `KCORE_K` (최소 평점 수 필터, minimum rating count filter) | None (꺼짐) | 정수 k면 평점이 k개 미만인 사용자·레시피를 반복해서 지웁니다. 현재는 쓰지 않습니다 |
| `N_VALID`, `N_TEST` | 1, 1 | 사용자별 valid / test 개수 |

최소 평점 수 필터를 쓰지 않는 이유: 기준을 5개로만 잡아도 사용자의 91%가 사라지고, test 정답 중 새 레시피 비율이 7.6%에서 0.02%로 줄어 평점이 적은 사용자와 새 레시피 추천을 평가할 수 없습니다(`eda/min_count_tradeoff.png`).

## test 사용자의 train 평점 수 분포
| train 평점 수 | 1~2개 | 3~4개 | 5~9개 | 10개 이상 |
|---|---|---|---|---|
| test 사용자 | 13,446 (37.9%) | 5,636 (15.9%) | 6,153 (17.3%) | 10,229 (28.8%) |

## 자동 검증
`run_preprocess.py`는 마지막에 저장된 파일을 다시 읽어 `validate_splits`로 아래 5개를 검사하고 결과 표를 출력합니다. 하나라도 실패하면 어떤 항목이 왜 실패했는지 메시지와 함께 에러가 납니다.
1. 평점 0 없음
2. 세 파일 간 `(user_id, recipe_id)` 겹침 없음
3. valid·test 사용자가 모두 train에 있음
4. 사용자별 train 최신 날짜 ≤ valid 날짜 ≤ test 날짜
5. train + valid + test 행 수 = 분할 전 행 수

## 결과 파일 SHA-256 해시
Windows, pandas 2.3.3에서 확인했습니다. 자기 PC 결과와 비교할 때 씁니다(`sha256sum data/processed/*` 또는 PowerShell `Get-FileHash data/processed/*`).
```
485ae753630c2a1393734bde2b149a175ee700a45d67f24870ed3afb440df59a  train.csv
545cefe3430a75fc501cc0ae64a2e2cfa82199391ca92d0c651a9416e6b21535  valid.csv
0359073745dc1a5a5c046bf9d4aca36274baf935b4ca9625bbbe6e7188247ef1  test.csv
36bb8a982ef7e0515a79b86aa450855839659c14a340a33ae7770359f02ac48d  split_stats.json
```

## EDA 그래프 (`eda/`)
| 파일 | 보여 주는 것 |
|---|---|
| `rating_distribution.png` | 원본 평점 0~5 개수. 5점이 72.1%, 0점(제거 대상)이 5.4% |
| `ratings_per_user.png` | 사용자당 평점 수(로그 축). 사용자의 71.6%가 평점 1개뿐 |
| `ratings_per_recipe.png` | 레시피당 평점 수(로그 축). 레시피의 40.5%가 평점 1개뿐 |
| `ratings_per_year.png` | 연도별 평점 수. 2008년이 가장 많음 |
| `min_count_tradeoff.png` | 최소 평점 수 기준(없음/2/3/5/10)별 남는 사용자 비율과 test 중 새 레시피 비율 |
| `test_users_train_count.png` | test 사용자의 train 평점 수 구간(1~2/3~4/5~9/10+)별 사용자 수 |
| `kcore_summary.csv` | 최소 평점 수 기준별 남는 평점·사용자·레시피 수(표) |

## 한계
- 같은 날짜 평가는 원본 파일 순서로 순서를 정합니다. 결과는 항상 같지만 실제 시간 순서라는 보장은 없습니다. test 사용자의 14%는 valid와 test가 같은 날짜입니다.
- 사용자별로 시간순 분할을 했기 때문에, 사용자 간에는 시점이 섞입니다. 어떤 사용자의 train이 다른 사용자의 test보다 나중일 수 있습니다.
- test 평점의 93%가 4점 이상입니다. RMSE는 전체 평균만으로 예측해도 낮게 나올 수 있으므로 기준선과 함께 비교해야 합니다.

담당: 이민지
