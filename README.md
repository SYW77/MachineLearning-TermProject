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
  evaluate.py            # 평가 지표(RMSE, Precision@K, Recall@K), 인기순 기준선, 그래프 함수
run_preprocess.py        # 전처리·분할 실행 + 자동 검증 → data/processed/
run_eda.py               # EDA 그래프 생성 → eda/
run_content_based.py     # 콘텐츠 기반 추천 예시 (사용자별 좋아한 레시피 vs 추천 레시피)
tune_content_based.py    # 콘텐츠 기반 튜닝 실험 (입력 필드, min_df, max_features)
run_evaluate.py          # 공통 분할로 인기순·콘텐츠 기반·행렬 분해 평가 → results/
tests/                   # pytest 테스트 (분할·검증 함수)
docs/preprocessing.md    # 전처리·분할 상세 (설정값, 검증, 해시, EDA, 한계)
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
model.predict(user_id, recipe_id) -> float  # (선택) 예측 평점. 있으면 RMSE를 계산한다
```

## 전처리와 데이터 분할

모든 모델과 평가 코드는 같은 분할 파일을 씁니다. `data/`는 Git에 올리지 않으므로 각자 아래 명령으로 만듭니다.

```
python run_preprocess.py      # 분할 생성 + 자동 검증 → data/processed/
python run_eda.py             # EDA 그래프 → eda/
python -m pytest tests -v     # 분할·검증 함수 테스트
```

Python 3.10 기준, `pip install -r requirements.txt`. 모두 프로젝트 루트에서 실행합니다.

- 평점 0 제거 → 중복 확인 → 사용자별 시간순 분할(가장 최근 1개 test, 그 앞 1개 valid, 나머지 train)
- 평점 3개 이상 사용자만 평가 대상, 1~2개 사용자는 train에만
- 출력: `data/processed/{train,valid,test}.csv` (컬럼 `user_id`, `recipe_id`, `date`, `rating`)

```python
import sys; sys.path.insert(0, "src")
from preprocess import load_splits
train, valid, test = load_splits("data/processed")
```

루트가 아닌 곳에서 부를 때는 경로를 직접 넘깁니다.

|       | 평점      | 사용자  | 레시피  |
| ----- | --------- | ------- | ------- |
| train | 1,000,592 | 196,098 | 221,461 |
| valid | 35,464    | 35,464  | 22,195  |
| test  | 35,464    | 35,464  | 21,975  |

test 정답의 7.6%는 train에 없던 새 레시피입니다.
설정값, 자동 검증, SHA-256 해시, EDA 그래프, 한계: [docs/preprocessing.md](docs/preprocessing.md)

이 섹션 담당: 이민지

## 콘텐츠 기반 필터링

```
python tune_content_based.py      # 설정 비교 (공통 valid) → results/content_based_tuning.csv/.png
python run_content_based.py       # 추천 예시 → results/content_based_examples.txt
```

숫자 평가(Precision@K, Recall@K)는 아래 "평가" 섹션의 `run_evaluate.py`로 냅니다.

1. `ingredients`, `tags` 문자열을 `ast.literal_eval`로 리스트로 바꿉니다. 여러 단어 재료는 `olive_oil`처럼 하나로 묶습니다.
2. 재료, 태그, 설명(description)을 이어 붙인 텍스트를 `TfidfVectorizer`로 레시피 × 단어 희소 행렬로 만듭니다.
3. 학습 데이터에서 4점 이상 준 레시피 벡터의 평균을 사용자 프로필로 씁니다.
4. 프로필과 전체 레시피의 코사인 유사도를 구하고, 이미 평가한 레시피를 빼고 상위 K개를 추천합니다.
5. 4점 이상 레시피가 없는 사용자에게는 인기순으로 추천합니다.

최종 설정: 입력 필드 재료+태그+설명, `min_df=2`, `max_features=None` (단어 44,077개)

### 튜닝 실험 (공통 valid)

공통 분할의 valid로 설정을 비교했습니다. 한 번에 하나씩만 바꿨고, 평가 사용자는 무작위 10,000명입니다(공통 평가 코드 `evaluate_topk`, 같은 seed).

| 실험          | 값                        | Recall@10                   | Recall@50                     |
| ------------- | ------------------------- | --------------------------- | ----------------------------- |
| 입력 필드     | 재료만 / +태그 / +설명    | 0.21% / 0.26% / **0.31%**   | 0.56% / 0.77% / **0.99%**     |
| min_df        | 1 / 2 / 5 / 20            | 모두 0.26%                  | 0.78% / 0.77% / 0.78% / 0.76% |
| max_features  | None / 1000 / 3000 / 5000 | 0.26% / 0.23% / 0.28% / 0.26% | 0.77% / 0.69% / 0.77% / 0.76% |
| 인기순 (비교) | -                         | 1.83%                       | 5.45%                         |

- 입력 필드를 더할수록 K=10, K=50 모두에서 좋아졌습니다. 그래서 설명까지 넣은 설정을 골랐습니다.
- `min_df`, `max_features`는 결과에 거의 영향이 없어서 기본값을 그대로 둡니다. 단어 수를 4,805개(min_df=20)까지 줄여도 성능이 비슷합니다.
- 어떤 설정도 인기순 기준선을 넘지 못했습니다. 평가 데이터의 정답이 인기 레시피에 많이 몰려 있기 때문으로 보입니다. 반면 콘텐츠 기반은 평점이 하나도 없는 새 레시피도 추천할 수 있습니다.

## 평가

세 모델(인기순 기준선, 콘텐츠 기반, 행렬 분해)을 같은 분할, 같은 지표로 비교합니다.

실행 (프로젝트 루트에서):

```
python run_evaluate.py                          # valid로 세 모델 평가 (확인용)
python run_evaluate.py --split test --users 0   # test로 최종 평가 (최종 수치용, 한 번만)
python run_evaluate.py --models popularity content_based   # 일부 모델만
```

- `--split valid`: train으로 학습하고 valid로 평가합니다. 설정을 고를 때 씁니다.
- `--split test`: train+valid로 학습하고 test로 평가합니다. 최종 수치를 낼 때만 씁니다.
- `--users N`: 평가 사용자를 무작위 N명으로 줄입니다(기본 2000, 모델끼리 같은 사용자). 0이면 전부입니다. RMSE는 항상 평가 평점 전체로 계산합니다.
- 결과: `results/eval_<split>_topk.csv`, `eval_<split>_topk.png`, `eval_<split>_rmse.csv`, `eval_<split>_rmse.png`

### 지표

| 지표        | 정의                                                          |
| ----------- | ------------------------------------------------------------- |
| Precision@K | 추천한 K개 중 정답의 비율 (사용자별 계산 후 평균)             |
| Recall@K    | 정답 중 추천 K개에 들어간 비율 (사용자별 계산 후 평균)        |
| RMSE        | 평가 평점과 예측 평점의 제곱오차 평균의 제곱근. 낮을수록 좋음 |

- **정답**: 평가 데이터(valid 또는 test)에서 사용자가 4점 이상 준 레시피입니다. 4점 이상이 없는 사용자는 평가에서 빠지므로 결과 표에 평가 사용자 수(`n_users`)를 함께 적습니다.
- **추천 후보**: train에서 이미 평가한 레시피는 추천에서 뺍니다.
- **RMSE**: `model.predict(user_id, recipe_id)`가 있는 모델만 계산합니다. 콘텐츠 기반은 유사도 순위만 내므로 RMSE를 계산하지 않습니다. 학습에 없던 사용자나 레시피는 전체 평균 평점으로 예측하며, 예측이 가능했던 비율(`coverage`)을 함께 출력합니다.

### 인기순 기준선

`PopularityBaseline`(`src/evaluate.py`). train에서 평가 수가 많은 레시피 순서로(같으면 평균 평점 순) 모든 사용자에게 같은 목록을 추천하되, 사용자가 이미 평가한 레시피는 뺍니다. 평점 예측은 레시피의 평균 평점이고, train에 없던 레시피는 전체 평균을 씁니다. 개인화 모델은 이 기준선보다 좋아야 의미가 있습니다.

### 결과 (test, 평가 사용자 33,041명)

| 모델                      | Recall@10 | Recall@50 | Precision@10 | RMSE   |
| ------------------------- | --------- | --------- | ------------ | ------ |
| 인기순                    | 2.15%     | 6.03%     | 0.215%       | 0.7965 |
| 콘텐츠 기반               | 0.39%     | 0.94%     | 0.039%       | -      |
| 행렬 분해 (수정 전, 임시) | 0.32%     | 0.98%     | 0.032%       | 1.9698 |

행렬 분해 수치는 평점 평균을 빼지 않는 현재 구현으로 낸 임시 값이며, 구현을 고친 뒤 갱신합니다.
콘텐츠 기반 수치는 튜닝으로 고른 최종 설정(재료+태그+설명)으로 다시 낸 값입니다(`results/eval_test_popularity+content_based_topk.csv`). `eval_test_topk.csv/.png`는 행렬 분해를 고친 뒤 세 모델을 함께 다시 돌릴 때 갱신됩니다.
그래프: `results/eval_test_topk.png`, `results/eval_test_rmse.png`

### 해석할 때 주의할 점

- test와 valid는 사용자당 평점이 1개뿐입니다. 그래서 Recall@K는 "정답 1개를 추천 K개 안에 넣었는가"(Hit@K)와 같고, Precision@K는 항상 Recall@K를 K로 나눈 값이라 매우 작게 나옵니다.
- 추천 후보가 22만 개가 넘는데 정답은 1개라서 모든 모델의 절댓값이 낮습니다. 모델끼리의 상대 비교로 읽습니다.
- 콘텐츠 기반은 4점 이상 평가가 없는 사용자에게 인기순으로 대체해서 추천하므로, 그런 사용자의 점수에는 인기순 결과가 섞여 있습니다.
- 개발 중에는 평가 사용자를 2000명으로 줄여서 돌렸습니다. 이 경우 Recall@10의 오차가 약 ±0.3%p라서 작은 차이는 우연일 수 있습니다. 최종 수치는 전체 사용자로 냅니다.

담당: 곽가현

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
