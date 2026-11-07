"""SGD와 L2 정규화를 사용하는 편향 항 없는 행렬 분해 추천기."""

import numpy as np
import pandas as pd


class MatrixFactorizationRecommender:
    def __init__(self, n_factors=20, learning_rate=0.01, reg=0.02,
                 n_epochs=20, random_state=42):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.reg = reg
        self.n_epochs = n_epochs
        self.random_state = random_state

    def fit(self, recipes_df, train_df):
        train = train_df[train_df["recipe_id"].isin(recipes_df["id"])]
        user_rows, user_ids = pd.factorize(train["user_id"], sort=True)
        item_rows, recipe_ids = pd.factorize(train["recipe_id"], sort=True)
        ratings = train["rating"].to_numpy(dtype=float)

        self.recipe_ids = recipe_ids.to_numpy()
        self.user_to_row = {uid: row for row, uid in enumerate(user_ids)}
        self.recipe_to_row = {rid: row for row, rid in enumerate(recipe_ids)}
        self.global_mean = float(ratings.mean())
        self.rated_ids = train.groupby("user_id")["recipe_id"].apply(set).to_dict()
        self.popular_ids = train["recipe_id"].value_counts().index.tolist()

        rng = np.random.default_rng(self.random_state)
        self.user_factors = rng.normal(0, 0.1, (len(user_ids), self.n_factors))
        self.item_factors = rng.normal(0, 0.1, (len(recipe_ids), self.n_factors))

        for _ in range(self.n_epochs):
            for row in rng.permutation(len(train)):
                u, i = user_rows[row], item_rows[row]
                user = self.user_factors[u].copy()
                item = self.item_factors[i].copy()
                error = ratings[row] - user.dot(item)
                self.user_factors[u] += self.learning_rate * (error * item - self.reg * user)
                self.item_factors[i] += self.learning_rate * (error * user - self.reg * item)

        return self

    def predict(self, user_id, recipe_id):
        u = self.user_to_row.get(user_id)
        i = self.recipe_to_row.get(recipe_id)
        if u is None or i is None:
            return self.global_mean
        return float(self.user_factors[u].dot(self.item_factors[i]))

    def recommend(self, user_id, k=10):
        u = self.user_to_row.get(user_id)
        if u is None:
            return self.popular_ids[:k]

        scores = self.item_factors.dot(self.user_factors[u])
        ranked = self.recipe_ids[np.argsort(-scores)]
        return [int(rid) for rid in ranked if rid not in self.rated_ids[user_id]][:k]
