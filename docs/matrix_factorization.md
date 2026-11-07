# 행렬 분해 연동

- src/matrix_factorization.py의 'MatrixFactorizationRecommender' 사용. SGD + L2 정규화.
- fit(recipes_df, train_df): 레시피 'id', 학습 데이터 'user_id', 'recipe_id', 'rating' 필요. 팀 공통 분할 사용.
- predict(user_id, recipe_id): 평점 예측값 하나 반환. 사용자 또는 레시피가 학습에 없으면 전체 학습 평균 반환. RMSE 평가에서 해당 행을 제외하지 않음.
- recommend(user_id, k=10): 학습된 레시피 중 이미 평가한 것을 제외하고 예측값 순으로 ID 리스트 반환. 후보가 부족하면 K개 미만 반환. 미학습 사용자는 학습 데이터 평가 횟수의 인기순 사용.
- 기본 파라미터는 동작 확인용이며 아직 튜닝 안함

담당: 장세창
