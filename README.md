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
  recipe_parsing.py      # 재료·태그 문자열 파싱, TF-IDF 입력 텍스트 생성
  content_based.py       # ContentBasedRecommender
  temp_eval.py           # 임시 분할·Hit@K (공통 분할·평가 코드가 나오기 전까지 사용)
run_content_based.py     # 콘텐츠 기반 추천기 실행 (임시 분할로 동작 확인)
tune_content_based.py    # 콘텐츠 기반 튜닝 실험 (입력 필드, min_df, max_features)
results/                 # 실행 결과, plot
Team6_Proposal.pdf       # 제안서
```

## 모델 공통 인터페이스
평가 코드가 두 모델을 같은 방식으로 호출할 수 있도록 아래 형식에 맞춥니다.
```python
model.fit(recipes_df, train_df)        # train_df 컬럼: user_id, recipe_id, rating (rating 0 제거)
model.recommend(user_id, k=10) -> list # 추천 recipe_id 리스트 (점수 높은 순)
```

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
