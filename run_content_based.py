"""
run_content_based.py  (담당: Member C)

콘텐츠 기반 추천기를 '임시 분할'로 실행해 동작을 확인하는 스크립트.

!! 주의 !!
    - 여기서 나온 수치는 동작 확인용이다. 최종 보고용 수치가 아니다.
    - 최종 결과는 A의 공통 분할과 D의 평가 코드로 다시 낸다.

임시 분할 (src/temp_eval.py)
    사용자별 가장 최근 평점 1개 → test, 그 앞 1개 → valid(튜닝용), 나머지 → train.
    이 스크립트는 튜닝을 하지 않는다. 그래서 train+valid로 학습하고 test로 확인한다.

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
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))  # src/ 폴더의 모듈을 import할 수 있게 경로에 추가
from content_based import ContentBasedRecommender  # noqa: E402
from temp_eval import temp_split, sample_eval_users, hit_at_k, popular_hit_at_k  # noqa: E402

# 명령행 인자로 데이터 폴더를 바꿀 수 있다. 기본값은 ./data
DATA_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data"
RESULT_DIR = ROOT / "results"
N_EVAL_USERS = 2000  # 전체 사용자를 다 돌리면 오래 걸린다. 무작위 2000명만 확인한다.
KS = [5, 10, 20, 50]

# 모델 설정. tune_content_based.py의 결과를 보고 고른 값을 여기에 넣는다.
FIELDS = ("ingredients", "tags")
MIN_DF = 2
MAX_FEATURES = None


def main():
    t0 = time.time()
    recipes = pd.read_csv(DATA_DIR / "RAW_recipes.csv")
    inter = pd.read_csv(DATA_DIR / "RAW_interactions.csv", usecols=["user_id", "recipe_id", "date", "rating"])
    inter = inter[inter["rating"] > 0]  # rating 0 = 리뷰만 남긴 경우 → 제거
    print(f"레시피 {len(recipes):,}개, 평점 {len(inter):,}개, 사용자 {inter['user_id'].nunique():,}명")

    train, valid, test = temp_split(inter)
    train = pd.concat([train, valid])  # 튜닝을 하지 않으므로 valid도 학습에 쓴다
    print(f"[임시 분할] train {len(train):,} / test {len(test):,}")

    model = ContentBasedRecommender(fields=FIELDS, min_df=MIN_DF, max_features=MAX_FEATURES)
    model.fit(recipes, train)
    print(f"TF-IDF 행렬: {model.X.shape[0]:,} 레시피 × {model.X.shape[1]:,} 단어, "
          f"0이 아닌 값 {model.X.nnz:,}개  ({time.time() - t0:.1f}s)")

    # ---- Hit@K 확인: 콘텐츠 기반 vs 인기순 ----
    eval_rows = sample_eval_users(test, N_EVAL_USERS)
    has_profile = sum(u in model.liked_rows for u in eval_rows["user_id"]) / len(eval_rows)
    summary = pd.DataFrame({
        "hit_content": hit_at_k(model, eval_rows, KS),
        "hit_popular": popular_hit_at_k(model, eval_rows, KS),
    })
    summary.index.name = "k"
    summary["precision_content"] = summary["hit_content"] / summary.index
    print(f"\n평가 사용자 {len(eval_rows):,}명 (프로필 있음 {has_profile:.1%})")
    print(summary.round(4).to_string())

    RESULT_DIR.mkdir(exist_ok=True)
    summary.to_csv(RESULT_DIR / "content_based_temp_eval.csv")

    # ---- plot: K에 따른 Hit@K, 콘텐츠 기반 vs 인기순 ----
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(summary.index, summary["hit_content"], marker="o", label="Content-based (TF-IDF)")
    ax.plot(summary.index, summary["hit_popular"], marker="s", linestyle="--", label="Popularity (fallback)")
    ax.set_xlabel("K")
    ax.set_ylabel("Hit@K (= Recall@K, leave-last-one-out)")
    ax.set_title(f"Content-based sanity check (temporary split, {len(eval_rows):,} users)")
    ax.set_xticks(KS)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "content_based_temp_hitk.png", dpi=150)

    # ---- 예시 한 명: 좋아한 레시피 vs 추천 레시피 이름 (눈으로 확인용) ----
    names = recipes.set_index("id")["name"]
    example = next(u for u in eval_rows["user_id"] if len(model.liked_rows.get(u, [])) >= 3)
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
