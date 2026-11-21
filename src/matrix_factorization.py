"""평점 평균 보정을 적용한 SGD + L2 행렬 분해 추천기.

예측 평점=전체 평균(mu)+p_u.q_i
평균을 빼고 학습하는 이유: 평점이 1~2개뿐인 사용자는 벡터가 거의 학습되지 않아서 예측값이 0 근처로 나옴
참고: Koren et al., Matrix Factorization Techniques for Recommender Systems, IEEE Computer, 2009
"""

import numpy as np
import pandas as pd


class MatrixFactorizationRecommender:
    def __init__(self, n_factors=20, learning_rate=0.01, reg=0.02,
                 n_epochs=20, random_state=42):
        # n_factors: 잠재 요인 수, reg: L2 정규화 계수(lambda), random_state: 난수 시드
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.reg = reg
        self.n_epochs = n_epochs
        self.random_state = random_state

    def fit(self, recipes_df, train_df):
        train = train_df[train_df["recipe_id"].isin(recipes_df["id"])]
        # pd.factorize: id를 0부터 시작하는 번호로 바꿈(행렬 인덱스로 쓰려고)
        # sort=True면 id 크기 순으로 번호를 매김, 반환값: 번호 배열, 원래 id 목록
        user_rows, user_ids = pd.factorize(train["user_id"], sort=True)
        item_rows, recipe_ids = pd.factorize(train["recipe_id"], sort=True)
        ratings = train["rating"].to_numpy(dtype=float)

        self.recipe_ids = recipe_ids.to_numpy()
        self.user_to_row = {uid: row for row, uid in enumerate(user_ids)}
        self.recipe_to_row = {rid: row for row, rid in enumerate(recipe_ids)}
        # 평균을 빼고 학습하고 예측할 때 다시 더함
        self.global_mean = float(ratings.mean())
        ratings = ratings - self.global_mean
        # 사용자별로 평가한 레시피 집합(추천에서 빼려고)
        self.rated_ids = train.groupby("user_id")["recipe_id"].apply(set).to_dict()
        # value_counts: 많이 나온 순으로 개수를 셈, 학습에 없는 사용자에게 인기순으로 추천할 때 씀
        self.popular_ids = train["recipe_id"].value_counts().index.tolist()

        # rng.normal(평균, 표준편차, 크기): 정규분포 난수로 잠재 벡터 초기화
        # 전부 0으로 시작하면 업데이트가 안 돼서 작은 값으로 시작함
        rng = np.random.default_rng(self.random_state)
        self.user_factors = rng.normal(0, 0.1, (len(user_ids), self.n_factors))
        self.item_factors = rng.normal(0, 0.1, (len(recipe_ids), self.n_factors))

        for _ in range(self.n_epochs):
            # rng.permutation: epoch마다 학습 순서를 섞음
            for row in rng.permutation(len(train)):
                u, i = user_rows[row], item_rows[row]
                # user를 먼저 바꾸니까 바꾸기 전 값을 복사해 둠
                user = self.user_factors[u].copy()
                item = self.item_factors[i].copy()
                error = ratings[row] - user.dot(item)  # 오차=(실제-평균)-예측
                # SGD 업데이트, reg는 L2 정규화 항
                self.user_factors[u] += self.learning_rate * (error * item - self.reg * user)
                self.item_factors[i] += self.learning_rate * (error * user - self.reg * item)

        return self

    def predict(self, user_id, recipe_id):
        u = self.user_to_row.get(user_id)
        i = self.recipe_to_row.get(recipe_id)
        if u is None or i is None:
            return self.global_mean
        return float(self.global_mean + self.user_factors[u].dot(self.item_factors[i]))

    def recommend(self, user_id, k=10):
        u = self.user_to_row.get(user_id)
        if u is None:
            return self.popular_ids[:k]  # 학습에 없는 사용자는 인기순

        scores = self.item_factors.dot(self.user_factors[u])
        # np.argsort: 오름차순 정렬 인덱스를 돌려줌, -scores를 넣어서 점수 높은 순
        ranked = self.recipe_ids[np.argsort(-scores)]
        return [int(rid) for rid in ranked if rid not in self.rated_ids[user_id]][:k]
