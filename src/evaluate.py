"""공통 평가 코드: RMSE, Precision@K, Recall@K, 인기순 기준선, 시각화.

README의 모델 공통 인터페이스를 따른다.
    model.fit(recipes_df, train_df)         # train_df 컬럼: user_id, recipe_id, rating
    model.recommend(user_id, k=10) -> list  # 추천 recipe_id 리스트 (점수 높은 순)
    model.predict(user_id, recipe_id)       # (RMSE용) 예측 평점 -- 평점을 예측하는 모델만 필요

평가 데이터는 preprocess.load_splits()가 돌려주는 valid 또는 test (컬럼: user_id, recipe_id, date, rating).
튜닝은 valid로, 최종 수치는 test로 한 번만 낸다.
"""
import os

import matplotlib

matplotlib.use("Agg")  # 창을 띄우지 않고 파일로만 저장
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

USER, ITEM, RATING = "user_id", "recipe_id", "rating"


# ----------------------------------------------------------------- RMSE
def rmse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def evaluate_rmse(model, eval_df):
    """eval_df의 모든 (user, recipe) 평점을 model.predict로 예측해 RMSE를 계산한다.
    예측이 None/NaN인 행은 제외하고, 예측이 가능했던 비율(coverage)을 함께 돌려준다."""
    # (p:=...): predict 결과를 p에 저장하면서 바로 None인지 확인(파이썬 3.8 이상)
    preds = np.array(
        [np.nan if (p := model.predict(u, i)) is None else p
         for u, i in zip(eval_df[USER], eval_df[ITEM])],
        dtype=float,
    )
    ok = ~np.isnan(preds)
    return {
        "rmse": rmse(eval_df[RATING].values[ok], preds[ok]) if ok.any() else np.nan,
        "coverage": float(ok.mean()),
    }


# ------------------------------------------------- Precision@K, Recall@K
def evaluate_topk(model, eval_df, ks=(5, 10, 20, 50), rel_threshold=4.0,
                  max_users=None, seed=42):
    """사용자별 Top-K 추천을 eval_df의 '좋아한 레시피'와 비교해 평균 Precision@K, Recall@K를 구한다.

    rel_threshold: 이 점수 이상으로 평가한 레시피만 정답으로 본다. None이면 eval_df 전체가 정답.
                   (정답이 하나도 없는 사용자는 평가에서 빠지므로 n_users를 꼭 같이 보고할 것)
    max_users:     모델이 느릴 때 평가 사용자를 무작위로 줄인다. 모델끼리 같은 seed를 쓰면 같은 사용자가 뽑힌다.
    """
    rel = eval_df if rel_threshold is None else eval_df[eval_df[RATING] >= rel_threshold]
    liked = rel.groupby(USER)[ITEM].apply(set)  # 사용자별 정답 레시피 set
    if max_users is not None and len(liked) > max_users:
        # sample(n, random_state): n명 무작위 추출, seed가 같으면 모델끼리 같은 사용자로 비교됨
        liked = liked.sample(max_users, random_state=seed)

    ks = sorted(ks)
    max_k = ks[-1]
    precision = {k: [] for k in ks}
    recall = {k: [] for k in ks}
    for user, relevant in liked.items():
        recs = list(model.recommend(user, k=max_k))
        for k in ks:
            hits = len(set(recs[:k]) & relevant)  # 상위 k개 추천과 정답의 교집합 개수
            precision[k].append(hits / k)
            recall[k].append(hits / len(relevant))

    return pd.DataFrame({
        "k": ks,
        "precision": [np.mean(precision[k]) for k in ks],
        "recall": [np.mean(recall[k]) for k in ks],
        "n_users": len(liked),
    })


# ------------------------------------------------------ 인기순 기준선
class PopularityBaseline:
    """train에서 평가 수가 많은 레시피 순(동률이면 평균 평점 순)으로 추천한다.
    사용자가 train에서 이미 평가한 레시피는 빼고 추천한다.
    평점 예측은 레시피 평균 평점, train에 없던 레시피는 전체 평균을 쓴다."""

    def fit(self, recipes_df, train_df):
        # agg(["count", "mean"]): 레시피별 평점 수와 평균을 한 번에 계산
        g = train_df.groupby(ITEM)[RATING].agg(["count", "mean"])
        self.ranking = g.sort_values(["count", "mean"], ascending=False).index.to_numpy()
        self.item_mean = g["mean"].to_dict()
        self.global_mean = float(train_df[RATING].mean())
        self.seen = train_df.groupby(USER)[ITEM].apply(set).to_dict()
        return self

    def recommend(self, user_id, k=10):
        seen = self.seen.get(user_id, ())
        out = []
        for item in self.ranking:
            if item not in seen:
                out.append(item)
                if len(out) == k:
                    break
        return out

    def predict(self, user_id, recipe_id):
        return self.item_mean.get(recipe_id, self.global_mean)


# ---------------------------------------------------------- 시각화
def plot_topk(results, path="results/topk_comparison.png", title_suffix=""):
    """results: {모델이름: evaluate_topk 결과}. Precision@K, Recall@K를 나란히 선 그래프로 그린다."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ["precision", "recall"]):
        for name, df in results.items():
            ax.plot(df["k"], df[metric], marker="o", label=name)
        ax.set_xlabel("K")
        ax.set_ylabel(f"{metric.capitalize()}@K")
        ax.set_title(f"{metric.capitalize()}@K{title_suffix}")
        ax.set_xticks(list(next(iter(results.values()))["k"]))
        ax.grid(alpha=0.3)
        ax.legend()
    _save(fig, path)


def plot_rmse(rmse_results, path="results/rmse_comparison.png"):
    """rmse_results: {모델이름: RMSE 값}. 막대 그래프."""
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(list(rmse_results), list(rmse_results.values()))
    ax.bar_label(bars, fmt="%.3f")  # 막대 위에 값 표시
    ax.set_ylabel("RMSE (lower is better)")
    ax.set_title("RMSE comparison")
    _save(fig, path)


def _save(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)  # exist_ok=True: 폴더가 이미 있어도 에러 안 남
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------- 데모 (가짜 데이터로 동작 확인)
class _RandomModel:
    def fit(self, recipes_df, train_df):
        self.items = train_df[ITEM].unique()
        self.seen = train_df.groupby(USER)[ITEM].apply(set).to_dict()
        self.rng = np.random.default_rng(1)
        return self

    def recommend(self, user_id, k=10):
        seen = self.seen.get(user_id, ())
        return [i for i in self.rng.permutation(self.items)[: k + 200] if i not in seen][:k]


def _demo():
    rng = np.random.default_rng(0)
    n_users, n_items = 400, 300
    # zipf: 일부 레시피에 인기가 몰리도록 가짜 데이터를 만듦
    pop = rng.zipf(1.5, n_items).astype(float)
    pop /= pop.sum()
    rows = []
    for u in range(n_users):
        # choice(n, size, replace=False, p): 확률 p로 중복 없이 뽑음
        for t, i in enumerate(rng.choice(n_items, size=25, replace=False, p=pop)):
            rows.append((u, i, t, int(np.clip(round(rng.normal(4.2, 1.0)), 1, 5))))
    df = pd.DataFrame(rows, columns=[USER, ITEM, "date", RATING])
    # preprocess와 같은 방식: 사용자별 시간순으로 마지막 1개 test, 그 앞 1개 valid, 나머지 train
    df = df.sort_values([USER, "date"])
    pos = df.groupby(USER).cumcount(ascending=False)
    train, valid, test = df[pos >= 2], df[pos == 1], df[pos == 0]

    models = {"popularity": PopularityBaseline().fit(None, train),
              "random": _RandomModel().fit(None, train)}
    topk = {name: evaluate_topk(m, test) for name, m in models.items()}
    for name, r in topk.items():
        print(f"\n[{name}]\n{r.to_string(index=False)}")
    r = evaluate_rmse(models["popularity"], test)
    print(f"\n[popularity RMSE] {r}")

    plot_topk(topk, "results/demo_topk.png", " (demo)")
    plot_rmse({"popularity": r["rmse"]}, "results/demo_rmse.png")


if __name__ == "__main__":
    _demo()