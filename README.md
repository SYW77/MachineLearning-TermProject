# Personalized Recipe Recommender (Team 6)

Food.com 레시피·평점 데이터로 개인 맞춤형 레시피 추천 시스템을 만드는 머신러닝(61375_004) 텀 프로젝트입니다.
두 가지 추천 방법을 같은 데이터 분할과 같은 평가 코드로 비교합니다.

| 방법 | 핵심 아이디어 | 코드 |
| --- | --- | --- |
| 콘텐츠 기반 필터링 | 재료·태그·설명의 TF-IDF 벡터와 코사인 유사도 | `src/content_based.py` |
| 모델 기반 협업 필터링 | 평점 행렬을 SGD 행렬 분해(L2 정규화)로 학습 | `src/matrix_factorization.py` |
| 비교 기준선 | 학습 데이터에서 평점이 많은 레시피 순 | `src/evaluate.py` (`PopularityBaseline`) |

## 결과 요약

test 데이터, 평가 사용자 33,041명 전체, train+valid로 학습한 결과입니다.

| 모델        | Recall@10 | Recall@50 | Precision@10 | RMSE   |
| ----------- | --------- | --------- | ------------ | ------ |
| 인기순      | **2.15%** | **6.03%** | **0.215%**   | 0.7965 |
| 콘텐츠 기반 | 0.39%     | 0.94%     | 0.039%       | -      |
| 행렬 분해   | 0.31%     | 0.96%     | 0.031%       | **0.7721** |

- **추천 순위(Recall·Precision)는 인기순이 가장 좋았습니다.** 레시피가 22만 개인데 사용자당 정답이 1개뿐이고, 정답이 인기 레시피에 몰려 있기 때문으로 보입니다.
- **평점 예측(RMSE)은 행렬 분해가 가장 정확했습니다.** 평점 예측과 순위 추천은 서로 다른 문제라는 걸 보여 줍니다.
- **콘텐츠 기반은 평점이 하나도 없는 새 레시피도 추천할 수 있습니다.** test 정답의 7.6%가 학습에 없던 레시피입니다.

그래프: [`results/eval_test_topk.png`](results/eval_test_topk.png), [`results/eval_test_rmse.png`](results/eval_test_rmse.png)

## 실행 방법

Python 3.10 이상 기준입니다. 모든 명령은 프로젝트 루트에서 실행합니다.

**1. 라이브러리 설치**

```
pip install -r requirements.txt
```

**2. 데이터 준비**

Kaggle **"Food.com Recipes and Interactions"**(작성자 shuyangli94)를 내려받아 `RAW_recipes.csv`, `RAW_interactions.csv`를 `data/` 폴더에 넣습니다.

- 이름이 비슷한 "Food.com - Recipes and Reviews"(recipes.csv / reviews.csv)는 **다른 데이터셋**입니다.
- `data/`는 용량(약 640MB) 때문에 Git에 올리지 않았습니다.

**3. 전처리와 분할** (약 10초)

```
python run_preprocess.py      # data/processed/{train,valid,test}.csv 생성 + 자동 검증 5개 항목
```

**4. 최종 평가** (세 모델, test 전체 사용자)

```
python run_evaluate.py --split test --users 0
```

결과는 `results/eval_test_topk.csv/.png`, `results/eval_test_rmse.csv/.png`에 저장됩니다.

**그 밖의 스크립트**

```
python run_eda.py                 # EDA 그래프 → eda/
python tune_content_based.py      # 콘텐츠 기반 튜닝 (valid, 약 20분) → results/content_based_tuning.*
python run_content_based.py       # 콘텐츠 기반 추천 예시 → results/content_based_examples.txt
python run_evaluate.py            # valid로 세 모델 평가 (설정 확인용, 사용자 2,000명)
python -m pytest tests -v         # 분할·검증 함수 테스트
```

## 폴더 구조

```
data/                      # 원본·분할 데이터 (Git 제외)
src/
  preprocess.py            # 전처리, 사용자별 시간순 분할, 검증(validate_splits), load_splits
  recipe_parsing.py        # 재료·태그 문자열 파싱, TF-IDF 입력 텍스트 생성
  content_based.py         # ContentBasedRecommender
  matrix_factorization.py  # MatrixFactorizationRecommender (SGD, 평균 보정)
  evaluate.py              # RMSE, Precision@K, Recall@K, 인기순 기준선, 그래프 함수
run_preprocess.py          # 전처리·분할 실행 + 자동 검증
run_eda.py                 # EDA 그래프 생성
run_content_based.py       # 콘텐츠 기반 추천 예시
tune_content_based.py      # 콘텐츠 기반 튜닝 실험
run_evaluate.py            # 세 모델 공통 평가
tests/                     # pytest 테스트 (분할·검증 함수)
docs/preprocessing.md      # 전처리·분할 상세 (설정값, 검증, 해시, EDA, 한계)
eda/                       # EDA 그래프, 최소 평점 수 기준별 요약표
results/                   # 튜닝·평가 결과 (CSV, PNG)
```

모든 모델은 같은 인터페이스를 따르므로 평가 코드 하나로 비교할 수 있습니다.

```python
model.fit(recipes_df, train_df)             # train_df 컬럼: user_id, recipe_id, rating
model.recommend(user_id, k=10) -> list      # 추천 recipe_id 리스트 (점수 높은 순)
model.predict(user_id, recipe_id) -> float  # (선택) 예측 평점. 있으면 RMSE를 계산
```

## 방법

### 1. 전처리와 데이터 분할

- 평점 0(리뷰만 남긴 경우) 제거 → 중복 확인 → **사용자별 시간순 분할**(가장 최근 1개 test, 그 앞 1개 valid, 나머지 train)
- 평점이 3개 이상인 사용자만 평가 대상이고, 1~2개인 사용자는 train에만 넣습니다.
- 최소 평점 수 필터(k-core)는 쓰지 않았습니다. 5개 기준만 걸어도 사용자의 91%가 사라지고, 새 레시피를 평가할 수 없게 되기 때문입니다(`eda/min_count_tradeoff.png`).

|       | 평점      | 사용자  | 레시피  |
| ----- | --------- | ------- | ------- |
| train | 1,000,592 | 196,098 | 221,461 |
| valid | 35,464    | 35,464  | 22,195  |
| test  | 35,464    | 35,464  | 21,975  |

설정값, 자동 검증, SHA-256 해시, EDA 그래프, 한계: [docs/preprocessing.md](docs/preprocessing.md)

### 2. 콘텐츠 기반 필터링

1. `ingredients`, `tags` 문자열을 `ast.literal_eval`로 리스트로 바꿉니다. 여러 단어 재료는 `olive_oil`처럼 하나로 묶습니다.
2. 재료, 태그, 설명(description)을 이어 붙인 텍스트를 `TfidfVectorizer`로 레시피 × 단어 희소 행렬로 만듭니다.
3. 학습 데이터에서 4점 이상 준 레시피 벡터의 평균을 사용자 프로필로 씁니다.
4. 프로필과 전체 레시피의 코사인 유사도를 구하고, 이미 평가한 레시피를 빼고 상위 K개를 추천합니다. 레시피 × 레시피 유사도 행렬은 메모리에 들어가지 않으므로 만들지 않습니다.
5. 4점 이상 레시피가 없는 사용자에게는 인기순으로 추천합니다.

**튜닝** (valid, 무작위 10,000명, 한 번에 설정 하나만 변경)

| 실험          | 값                        | Recall@10                     | Recall@50                     |
| ------------- | ------------------------- | ----------------------------- | ----------------------------- |
| 입력 필드     | 재료만 / +태그 / +설명    | 0.21% / 0.26% / **0.31%**     | 0.56% / 0.77% / **0.99%**     |
| min_df        | 1 / 2 / 5 / 20            | 모두 0.26%                    | 0.78% / 0.77% / 0.78% / 0.76% |
| max_features  | None / 1000 / 3000 / 5000 | 0.26% / 0.23% / 0.28% / 0.26% | 0.77% / 0.69% / 0.77% / 0.76% |
| 인기순 (비교) | -                         | 1.83%                         | 5.45%                         |

최종 설정은 재료+태그+설명, `min_df=2`, `max_features=None`(단어 44,077개)입니다. `min_df`와 `max_features`는 결과에 거의 영향이 없었습니다.

### 3. 행렬 분해 (모델 기반 협업 필터링)

목적 함수: `min Σ (r − μ − pᵤᵀqᵢ)² + λ(‖pᵤ‖² + ‖qᵢ‖²)` (μ는 학습 평점 전체 평균, 학습하는 편향 항은 없음)

1. 평점에서 전체 평균 μ를 뺀 뒤 사용자 벡터 P와 레시피 벡터 Q를 SGD로 학습합니다.
2. 예측 평점은 `μ + pᵤᵀqᵢ`, 추천은 `pᵤᵀqᵢ`가 큰 순서(이미 평가한 레시피 제외)입니다.
3. 학습에 없던 사용자에게는 인기순으로 추천하고, 평점은 μ로 예측합니다.

- **평균을 빼는 이유:** 평균을 빼지 않으면 평점이 1~2개뿐인 사용자의 벡터가 0 근처에 머물러 예측값이 0에 가까워집니다. valid RMSE가 2.33에서 0.77로 개선되었습니다.
- **설정:** `n_factors=20`, `learning_rate=0.01`, `reg=0.02`, `n_epochs=20`
- **격자 탐색:** 81개 조합(요인 수 10/20/40 × 학습률 0.005/0.01/0.02 × λ 0.01/0.02/0.05 × epoch 10/20/40)을 valid에서 비교했습니다(`results/mf_grid_search_valid.csv`). RMSE는 모든 조합이 0.772~0.816으로 비슷했고 두드러지게 좋은 조합이 없어서, 기본 설정을 유지했습니다.

### 4. 평가

| 지표        | 정의                                                          |
| ----------- | ------------------------------------------------------------- |
| Precision@K | 추천한 K개 중 정답의 비율 (사용자별 계산 후 평균)             |
| Recall@K    | 정답 중 추천 K개에 들어간 비율 (사용자별 계산 후 평균)        |
| RMSE        | 평가 평점과 예측 평점의 제곱오차 평균의 제곱근. 낮을수록 좋음 |

- **정답:** 평가 데이터에서 사용자가 4점 이상 준 레시피입니다. 4점 이상이 없는 사용자는 평가에서 빠집니다(그래서 평가 사용자가 35,464명이 아니라 33,041명입니다).
- **추천 후보:** 이미 평가한 레시피는 추천에서 뺍니다.
- **RMSE:** `predict()`가 있는 모델(인기순, 행렬 분해)만 계산합니다. 콘텐츠 기반은 유사도 순위만 내므로 계산하지 않습니다.
- **튜닝과 최종 평가 분리:** 설정은 valid로만 고르고, test는 train+valid로 학습한 뒤 마지막에 한 번만 평가했습니다.

**해석할 때 주의할 점**

- test와 valid는 사용자당 평점이 1개뿐이라 Recall@K는 Hit@K와 같고, Precision@K는 Recall@K를 K로 나눈 값이라 매우 작게 나옵니다.
- 추천 후보가 22만 개가 넘는데 정답은 1개라서 모든 모델의 절댓값이 낮습니다. 모델끼리의 상대 비교로 읽어야 합니다.
- 콘텐츠 기반은 4점 이상 평가가 없는 사용자에게 인기순으로 추천하므로, 그런 사용자의 점수에는 인기순 결과가 섞여 있습니다.

## 참고 자료 및 출처

참고한 곳은 코드 주석에도 적었습니다.

- B. P. Majumder et al., "Generating Personalized Recipes from Historical User Preferences," EMNLP 2019 (데이터셋 원 논문)
- Kaggle, "Food.com Recipes and Interactions"
- Y. Koren, R. Bell, C. Volinsky, "Matrix Factorization Techniques for Recommender Systems," IEEE Computer, 2009
- scikit-learn 문서, `TfidfVectorizer`: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
- SciPy 문서, `scipy.sparse`: https://docs.scipy.org/doc/scipy/reference/sparse.html
- Python 문서, `ast.literal_eval`: https://docs.python.org/3/library/ast.html#ast.literal_eval
- pandas 문서, `GroupBy.cumcount`: https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.GroupBy.cumcount.html
- pandas 문서, `GroupBy.transform`: https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.DataFrameGroupBy.transform.html
- pandas 문서, `DataFrame.sort_values`: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_values.html
- pandas 문서, `DataFrame.to_csv`: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html
