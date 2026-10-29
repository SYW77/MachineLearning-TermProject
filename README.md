# Personalized Recipe Recommender (Team 6)

Food.com 레시피·평점 데이터로 만드는 개인 맞춤형 레시피 추천 시스템. 머신러닝(61375_004) 텀 프로젝트.

- 콘텐츠 기반 필터링: TF-IDF + 코사인 유사도
- 모델 기반 협업 필터링: SGD 행렬 분해
- 평가: RMSE, Precision@K, Recall@K, 인기순 기준선과 비교

## 팀 역할
| 이름 | 담당 |
|---|---|
| 이민지 | EDA, 전처리, 학습/검증/테스트 분할 |
| 장세창 | 행렬 분해 (협업 필터링) |
| 손연우 | TF-IDF 콘텐츠 기반 필터링, 재료·태그 파싱 |
| 곽가현 | 평가 코드, 인기순 기준선, 시각화, 발표 |

## 데이터 준비
1. Kaggle **"Food.com Recipes and Interactions"**(작성자 shuyangli94)를 내려받는다.
   - 이름이 비슷한 "Food.com - Recipes and Reviews"(recipes.csv / reviews.csv)는 **다른 데이터셋**이다.
2. 압축을 풀고 `RAW_recipes.csv`, `RAW_interactions.csv`를 `data/` 폴더에 넣는다.
   - `data/`는 용량 때문에 Git에 올리지 않는다(`.gitignore`).

## 폴더 구조
```
data/                    # 데이터 (Git 제외)
src/
  recipe_parsing.py      # 재료·태그 문자열 파싱, TF-IDF 입력 텍스트 생성
  content_based.py       # ContentBasedRecommender
run_content_based.py     # 콘텐츠 기반 추천기 실행 (임시 분할로 동작 확인)
results/                 # 실행 결과, plot
Team6_Proposal.pdf       # 제안서
```

## 모델 공통 인터페이스
평가 코드가 두 모델을 같은 방식으로 호출할 수 있도록 맞춘다.
```python
model.fit(recipes_df, train_df)        # train_df 컬럼: user_id, recipe_id, rating (rating 0 제거)
model.recommend(user_id, k=10) -> list # 추천 recipe_id 리스트 (점수 높은 순)
```

## 콘텐츠 기반 필터링 (손연우)
실행:
```
python run_content_based.py
```
1. `ingredients`, `tags` 문자열을 `ast.literal_eval`로 리스트로 바꾼다. 여러 단어 재료는 `olive_oil`처럼 하나로 묶는다.
2. 재료와 태그를 이어 붙인 텍스트를 `TfidfVectorizer`로 레시피 × 단어 희소 행렬로 만든다.
3. 사용자 프로필 = 학습 데이터에서 4점 이상 준 레시피 벡터의 평균.
4. 프로필과 전체 레시피의 코사인 유사도를 구하고, 이미 평가한 레시피를 빼고 상위 K개를 추천한다.
5. 4점 이상 레시피가 없는 사용자는 인기순으로 대체한다.

임시 분할(사용자별 가장 최근 평점 1개를 테스트로 사용)로 낸 결과입니다. 동작 확인용이며, 최종 수치는 공통 분할과 평가 코드로 다시 냅니다.

| K | Hit@K 콘텐츠 기반 | Hit@K 인기순 |
|---|---|---|
| 5 | 0.40% | 1.35% |
| 10 | 0.55% | 2.40% |
| 20 | 0.70% | 3.65% |
| 50 | 1.25% | 6.30% |

## 참고 자료 및 출처
외부 코드를 복사하지 않았으며, 아래 자료를 참고했다. 참고한 곳은 코드 주석에도 적었다.
- B. P. Majumder et al., "Generating Personalized Recipes from Historical User Preferences," EMNLP 2019 (데이터셋 원 논문)
- Kaggle, "Food.com Recipes and Interactions"
- scikit-learn 문서, `TfidfVectorizer`: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
- SciPy 문서, `scipy.sparse`: https://docs.scipy.org/doc/scipy/reference/sparse.html
- Python 문서, `ast.literal_eval`: https://docs.python.org/3/library/ast.html#ast.literal_eval
- Y. Koren, R. Bell, C. Volinsky, "Matrix Factorization Techniques for Recommender Systems," IEEE Computer, 2009
