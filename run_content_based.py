"""
run_content_based.py  (담당: Member C)

콘텐츠 기반 추천기를 '임시 분할'로 실행해 동작을 확인하는 스크립트.

!! 주의 !!
    - 여기서 나온 수치는 동작 확인용이다. 최종 보고용 수치가 아니다.
    - 최종 결과는 A의 공통 분할과 D의 평가 코드로 다시 낸다.

임시 분할 방법
    평점이 2개 이상인 사용자마다 '가장 최근(date 기준) 평점 1개'를 테스트로 뺀다.
    나머지는 모두 학습 데이터로 쓴다.

임시 확인 지표: Hit@K
    테스트 레시피에 4점 이상을 준 사용자만 대상으로 한다.
    그 레시피가 추천 상위 K개 안에 있으면 hit이다.
    테스트 아이템이 사용자당 1개이므로 Recall@K = Hit@K, Precision@K = Hit@K / K 이다.

실행 방법
    1) Kaggle "Food.com Recipes and Interactions"에서 받은 RAW_recipes.csv와
       RAW_interactions.csv를 data/ 폴더에 넣는다.
    2) python run_content_based.py
    결과: results/content_based_temp_eval.csv, results/content_based_temp_hitk.png
"""

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 화면 없이 PNG 파일로만 저장하는 백엔드
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))  # src/ 폴더의 모듈을 import할 수 있게 경로에 추가
from content_based import ContentBasedRecommender, LIKE_THRESHOLD  # noqa: E402

# 명령행 인자로 데이터 폴더를 바꿀 수 있다. 기본값은 ./data
DATA_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data"
RESULT_DIR = ROOT / "results"
N_EVAL_USERS = 2000  # 전체 사용자를 다 돌리면 오래 걸린다. 무작위 2000명만 확인한다.
KS = [5, 10, 20, 50]
SEED = 42


def temp_split(inter):
    """사용자별 가장 최근 평점 1개를 test로 빼는 임시 분할."""
    inter = inter.sort_values(["user_id", "date"])
    n_per_user = inter.groupby("user_id")["recipe_id"].transform("size")
    # cumcount(ascending=False): 그룹 안에서 뒤에서부터 0, 1, 2…를 매긴다 → 0이 가장 최근 평점
    is_last = inter.groupby("user_id").cumcount(ascending=False) == 0
    test_mask = is_last & (n_per_user >= 2)  # 평점이 1개뿐인 사용자는 학습에 남긴다
    return inter[~test_mask], inter[test_mask]


def main():
    t0 = time.time()
    recipes = pd.read_csv(DATA_DIR / "RAW_recipes.csv")
    inter = pd.read_csv(DATA_DIR / "RAW_interactions.csv", usecols=["user_id", "recipe_id", "date", "rating"])
    inter = inter[inter["rating"] > 0]  # rating 0 = 리뷰만 남긴 경우 → 제거
    print(f"레시피 {len(recipes):,}개, 평점 {len(inter):,}개, 사용자 {inter['user_id'].nunique():,}명")

    train, test = temp_split(inter)
    print(f"[임시 분할] train {len(train):,} / test {len(test):,}")

    model = ContentBasedRecommender(fields=("ingredients", "tags"), min_df=2).fit(recipes, train)
    print(f"TF-IDF 행렬: {model.X.shape[0]:,} 레시피 × {model.X.shape[1]:,} 단어, "
          f"0이 아닌 값 {model.X.nnz:,}개  ({time.time() - t0:.1f}s)")

    # ---- Hit@K 확인 (테스트 레시피를 4점 이상 준 사용자 중 무작위 표본) ----
    pos = test[test["rating"] >= LIKE_THRESHOLD]
    pos = pos.sample(min(N_EVAL_USERS, len(pos)), random_state=SEED)
    max_k = max(KS)
    rows = []
    for user_id, target in zip(pos["user_id"], pos["recipe_id"]):
        recs = model.recommend(user_id, k=max_k)
        popular = model._popular(model.rated_rows.get(user_id, []), max_k)  # 비교용 인기순
        has_profile = model.user_profile(user_id) is not None
        for k in KS:
            rows.append({"user_id": user_id, "k": k, "has_profile": has_profile,
                         "hit_content": target in recs[:k], "hit_popular": target in popular[:k]})
    res = pd.DataFrame(rows)

    summary = res.groupby("k")[["hit_content", "hit_popular"]].mean()
    summary["precision_content"] = summary["hit_content"] / summary.index
    print(f"\n평가 사용자 {len(pos):,}명 (프로필 있음 {res.drop_duplicates('user_id')['has_profile'].mean():.1%})")
    print(summary.round(4).to_string())

    RESULT_DIR.mkdir(exist_ok=True)
    summary.to_csv(RESULT_DIR / "content_based_temp_eval.csv")

    # ---- plot: K에 따른 Hit@K, 콘텐츠 기반 vs 인기순 ----
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(summary.index, summary["hit_content"], marker="o", label="Content-based (TF-IDF)")
    ax.plot(summary.index, summary["hit_popular"], marker="s", linestyle="--", label="Popularity (fallback)")
    ax.set_xlabel("K")
    ax.set_ylabel("Hit@K (= Recall@K, leave-last-one-out)")
    ax.set_title(f"Content-based sanity check (temporary split, {len(pos):,} users)")
    ax.set_xticks(KS)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "content_based_temp_hitk.png", dpi=150)

    # ---- 예시 한 명: 좋아한 레시피 vs 추천 레시피 이름 (눈으로 확인용) ----
    names = recipes.set_index("id")["name"]
    example = next(u for u in pos["user_id"] if len(model.liked_rows.get(u, [])) >= 3)
    liked_ids = model.recipe_ids[model.liked_rows[example]][:5]
    print(f"\n[예시] user {example}가 좋아한 레시피:")
    for rid in liked_ids:
        print("   -", names.get(rid))
    print("  추천 Top-5:")
    for rid in model.recommend(example, k=5):
        print("   -", names.get(rid))
    print(f"\n완료 ({time.time() - t0:.1f}s). 결과: {RESULT_DIR}")


if __name__ == "__main__":
    main()
