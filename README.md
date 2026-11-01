# Personalized Recipe Recommender (Team 6)

Food.com 레시피·평점 데이터로 개인 맞춤형 레시피 추천 시스템을 만드는 머신러닝(61375_004) 텀 프로젝트입니다.

- 콘텐츠 기반 필터링: TF-IDF + 코사인 유사도
- 모델 기반 협업 필터링: SGD 행렬 분해
- 평가: RMSE, Precision@K, Recall@K, 인기순 기준선과 비교

## 데이터 준비
1. Kaggle **"Food.com Recipes and Interactions"**(작성자 shuyangli94)를 내려받습니다.
   - 이름이 비슷한 "Food.com - Recipes and Reviews"(recipes.csv / reviews.csv)는 **다른 데이터셋**입니다.
2. 압축을 풀고 `RAW_recipes.csv`, `RAW_interactions.csv`를 `data/` 폴더에 넣습니다.
   - `data/`는 용량 때문에 Git에 올리지 않습니다(`.gitignore`).

## 폴더 구조
```
data/                    # 데이터 (Git 제외)
src/
  preprocess.py          # 전처리, 사용자별 시간순 train/valid/test 분할, 검증(validate_splits), load_splits
  recipe_parsing.py      # 재료·태그 문자열 파싱, TF-IDF 입력 텍스트 생성
  content_based.py       # ContentBasedRecommender
  temp_eval.py           # 임시 분할·Hit@K (공통 분할·평가 코드가 나오기 전까지 사용)
run_preprocess.py        # 전처리·분할 실행 + 자동 검증 → data/processed/
run_eda.py               # EDA 그래프 생성 → eda/
run_content_based.py     # 콘텐츠 기반 추천기 실행 (임시 분할로 동작 확인)
tune_content_based.py    # 콘텐츠 기반 튜닝 실험 (입력 필드, min_df, max_features)
tests/                   # pytest 테스트 (분할·검증 함수)
eda/                     # EDA 그래프, 최소 평점 수 기준별 요약표
results/                 # 실행 결과, plot
requirements.txt         # 필요한 라이브러리
Team6_Proposal.pdf       # 제안서
```

## 모델 공통 인터페이스
평가 코드가 두 모델을 같은 방식으로 호출할 수 있도록 아래 형식에 맞춥니다.
```python
model.fit(recipes_df, train_df)        # train_df 컬럼: user_id, recipe_id, rating (rating 0 제거)
model.recommend(user_id, k=10) -> list # 추천 recipe_id 리스트 (점수 높은 순)
```

## 전처리와 데이터 분할
모든 모델과 평가 코드는 같은 분할 파일을 씁니다. `data/`는 Git에 올리지 않으므로 각자 아래 명령으로 만듭니다. 원본과 설정이 같으면 결과 파일이 항상 똑같습니다.

### 실행
Python 3.10 기준입니다. 필요한 라이브러리는 `pip install -r requirements.txt`로 설치합니다.
```
python run_preprocess.py      # 분할 생성 + 자동 검증 → data/processed/
python run_eda.py             # EDA 그래프 → eda/
python -m pytest tests -v     # 작은 가짜 데이터로 분할·검증 함수 테스트
```
모두 프로젝트 루트에서 실행합니다.

### 처리 단계 (`src/preprocess.py`)
1. `RAW_interactions.csv`에서 `user_id, recipe_id, date, rating`만 읽습니다.
2. 평점 0(리뷰만 남기고 별점은 주지 않은 경우)을 제거합니다.
3. 같은 `(user_id, recipe_id)` 중복을 확인합니다. 있으면 가장 최근 것만 남깁니다(현재 데이터는 0건).
4. 사용자별로 날짜순 정렬한 뒤 가장 최근 1개를 test, 그 앞 1개를 valid, 나머지를 train으로 나눕니다. 같은 날짜면 원본 파일 순서를 따릅니다.
   - 평점이 3개 이상인 사용자만 평가 대상(valid·test)입니다.
   - 평점이 1~2개인 사용자는 train에만 들어갑니다.
5. 분할 결과를 자동으로 검증합니다(아래 "자동 검증").

### 출력 파일 (`data/processed/`)
| 파일 | 내용 |
|---|---|
| `train.csv`, `valid.csv`, `test.csv` | 컬럼 `user_id`(int), `recipe_id`(int), `date`(문자열 YYYY-MM-DD), `rating`(int, 1~5) |
| `split_stats.json` | 설정값, 단계별 행·사용자·레시피 수 |

불러오기:
```python
import sys; sys.path.insert(0, "src")      # src/ 폴더를 import 경로에 추가
from preprocess import load_splits
train, valid, test = load_splits("data/processed")
model.fit(recipes_df, train)                # 튜닝: train으로 학습, valid로 평가
```
`"data/processed"`는 실행 위치 기준 상대 경로입니다. 프로젝트 루트가 아닌 곳(예: 하위 폴더)에서 부를 때는 `load_splits("../data/processed")`처럼 경로를 직접 넘겨야 합니다.

### 설정값 (`src/preprocess.py` 맨 위)
| 이름 | 현재 값 | 의미 |
|---|---|---|
| `MIN_USER_RATINGS` | 3 | 평점이 이 개수 이상인 사용자만 valid·test를 뗍니다. `N_VALID + N_TEST + 1` 이상이어야 합니다 |
| `KEEP_SPARSE_USERS_IN_TRAIN` | True | 평점이 `MIN_USER_RATINGS` 미만인 사용자를 train에 남깁니다(False면 제외) |
| `KCORE_K` (최소 평점 수 필터, minimum rating count filter) | None (꺼짐) | 정수 k면 평점이 k개 미만인 사용자·레시피를 반복해서 지웁니다. 현재는 쓰지 않습니다 |
| `N_VALID`, `N_TEST` | 1, 1 | 사용자별 valid / test 개수 |

최소 평점 수 필터를 쓰지 않는 이유: 기준을 5개로만 잡아도 사용자의 91%가 사라지고, test 정답 중 새 레시피 비율이 7.6%에서 0.02%로 줄어 평점이 적은 사용자와 새 레시피 추천을 평가할 수 없습니다(`eda/min_count_tradeoff.png`).

### 분할 결과
| | 평점 | 사용자 | 레시피 |
|---|---|---|---|
| 원본 | 1,132,367 | 226,570 | 231,637 |
| 평점 0 제거 후 | 1,071,520 | 196,098 | 226,590 |
| train | 1,000,592 | 196,098 | 221,461 |
| valid | 35,464 | 35,464 | 22,195 |
| test | 35,464 | 35,464 | 21,975 |

- test 정답 레시피 중 7.6%(2,698개)는 train에 한 번도 나오지 않은 새 레시피입니다.
- test 사용자의 train 평점 수 분포:

| train 평점 수 | 1~2개 | 3~4개 | 5~9개 | 10개 이상 |
|---|---|---|---|---|
| test 사용자 | 13,446 (37.9%) | 5,636 (15.9%) | 6,153 (17.3%) | 10,229 (28.8%) |

### 자동 검증
`run_preprocess.py`는 마지막에 저장된 파일을 다시 읽어 `validate_splits`로 아래 5개를 검사하고 결과 표를 출력합니다. 하나라도 실패하면 어떤 항목이 왜 실패했는지 메시지와 함께 에러가 납니다.
1. 평점 0 없음
2. 세 파일 간 `(user_id, recipe_id)` 겹침 없음
3. valid·test 사용자가 모두 train에 있음
4. 사용자별 train 최신 날짜 ≤ valid 날짜 ≤ test 날짜
5. train + valid + test 행 수 = 분할 전 행 수

결과 파일 SHA-256 해시(Windows, pandas 2.3.3에서 확인). 자기 PC 결과와 비교할 때 씁니다(`sha256sum data/processed/*` 또는 PowerShell `Get-FileHash data/processed/*`).
```
485ae753630c2a1393734bde2b149a175ee700a45d67f24870ed3afb440df59a  train.csv
545cefe3430a75fc501cc0ae64a2e2cfa82199391ca92d0c651a9416e6b21535  valid.csv
0359073745dc1a5a5c046bf9d4aca36274baf935b4ca9625bbbe6e7188247ef1  test.csv
36bb8a982ef7e0515a79b86aa450855839659c14a340a33ae7770359f02ac48d  split_stats.json
```

### EDA 그래프 (`eda/`)
| 파일 | 보여 주는 것 |
|---|---|
| `rating_distribution.png` | 원본 평점 0~5 개수. 5점이 72.1%, 0점(제거 대상)이 5.4% |
| `ratings_per_user.png` | 사용자당 평점 수(로그 축). 사용자의 71.6%가 평점 1개뿐 |
| `ratings_per_recipe.png` | 레시피당 평점 수(로그 축). 레시피의 40.5%가 평점 1개뿐 |
| `ratings_per_year.png` | 연도별 평점 수. 2008년이 가장 많음 |
| `min_count_tradeoff.png` | 최소 평점 수 기준(없음/2/3/5/10)별 남는 사용자 비율과 test 중 새 레시피 비율 |
| `test_users_train_count.png` | test 사용자의 train 평점 수 구간(1~2/3~4/5~9/10+)별 사용자 수 |
| `kcore_summary.csv` | 최소 평점 수 기준별 남는 평점·사용자·레시피 수(표) |

### 한계
- 같은 날짜 평가는 원본 파일 순서로 순서를 정합니다. 결과는 항상 같지만 실제 시간 순서라는 보장은 없습니다. test 사용자의 14%는 valid와 test가 같은 날짜입니다.
- 사용자별로 시간순 분할을 했기 때문에, 사용자 간에는 시점이 섞입니다. 어떤 사용자의 train이 다른 사용자의 test보다 나중일 수 있습니다.
- test 평점의 93%가 4점 이상입니다. RMSE는 전체 평균만으로 예측해도 낮게 나올 수 있으므로 기준선과 함께 비교해야 합니다.

이 섹션 담당: 이민지

## 콘텐츠 기반 필터링
실행:
```
python run_content_based.py
```
1. `ingredients`, `tags` 문자열을 `ast.literal_eval`로 리스트로 바꿉니다. 여러 단어 재료는 `olive_oil`처럼 하나로 묶습니다.
2. 재료와 태그를 이어 붙인 텍스트를 `TfidfVectorizer`로 레시피 × 단어 희소 행렬로 만듭니다.
3. 학습 데이터에서 4점 이상 준 레시피 벡터의 평균을 사용자 프로필로 씁니다.
4. 프로필과 전체 레시피의 코사인 유사도를 구하고, 이미 평가한 레시피를 빼고 상위 K개를 추천합니다.
5. 4점 이상 레시피가 없는 사용자에게는 인기순으로 추천합니다.

아래는 임시 분할(사용자별 가장 최근 평점 1개를 테스트로 사용)로 낸 결과입니다. 동작 확인용이며, 최종 수치는 공통 분할과 평가 코드로 다시 냅니다.

| K | Hit@K 콘텐츠 기반 | Hit@K 인기순 |
|---|---|---|
| 5 | 0.40% | 1.35% |
| 10 | 0.55% | 2.40% |
| 20 | 0.70% | 3.65% |
| 50 | 1.25% | 6.30% |

### 튜닝 실험 (임시 검증셋)
```
python tune_content_based.py
```
사용자별 끝에서 두 번째 평점을 검증용으로 떼어 설정을 비교했습니다. 한 번에 하나씩만 바꿨고, 평가 사용자는 2,000명입니다.
결과: `results/content_based_tuning.csv`, `results/content_based_tuning.png`

| 실험 | 값 | Hit@10 | Hit@50 |
|---|---|---|---|
| 입력 필드 | 재료만 / +태그 / +설명 | 0.15% / 0.20% / 0.35% | 0.55% / 0.85% / 1.10% |
| min_df | 1 / 2 / 5 / 20 | 모두 0.20% | 0.85% / 0.85% / 0.85% / 0.95% |
| max_features | None / 1000 / 3000 / 5000 | 모두 0.20% | 0.85% / 0.65% / 0.85% / 0.95% |
| 인기순 (비교) | - | 1.70% | 5.65% |

- 설명(description)을 넣었을 때 가장 높았습니다. 다만 2,000명 중 hit 수가 Hit@10 기준 3~7명이라 차이가 우연일 수 있습니다.
- `min_df`, `max_features`는 결과에 거의 영향이 없었습니다. 단어 수를 4,805개(min_df=20)까지 줄여도 성능이 비슷하므로 행렬을 작게 만들 수 있습니다.
- 공통 분할이 나오면 그 검증셋으로 다시 돌려 최종 설정을 정합니다.

### 공통 분할이 나오면
`model.fit(recipes_df, train_df)`에 공통 분할의 학습 데이터를 넣고, 평가 코드에서 `model.recommend(user_id, k)`를 호출하면 됩니다.

## 참고 자료 및 출처
참고한 곳은 코드 주석에도 적었습니다.
- B. P. Majumder et al., "Generating Personalized Recipes from Historical User Preferences," EMNLP 2019 (데이터셋 원 논문)
- Kaggle, "Food.com Recipes and Interactions"
- scikit-learn 문서, `TfidfVectorizer`: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
- SciPy 문서, `scipy.sparse`: https://docs.scipy.org/doc/scipy/reference/sparse.html
- Python 문서, `ast.literal_eval`: https://docs.python.org/3/library/ast.html#ast.literal_eval
- Y. Koren, R. Bell, C. Volinsky, "Matrix Factorization Techniques for Recommender Systems," IEEE Computer, 2009
- pandas 문서, `GroupBy.cumcount`: https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.GroupBy.cumcount.html
- pandas 문서, `GroupBy.transform`: https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.DataFrameGroupBy.transform.html
- pandas 문서, `DataFrame.sort_values`: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_values.html
- pandas 문서, `DataFrame.to_csv`: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html
