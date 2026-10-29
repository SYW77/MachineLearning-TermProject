"""
tune_content_based.py  (담당: Member C, 3주차 튜닝 실험)

콘텐츠 기반 추천기의 설정을 바꿔 가며 검증(valid) 데이터에서 Hit@K를 비교한다.

실험 (기본 설정: 재료+태그, min_df=2, max_features=None)
    한 번에 하나만 바꾸고 나머지는 기본값으로 둔다.
    1) 입력 필드  : 재료만 / 재료+태그 / 재료+태그+설명
    2) min_df     : 1, 2, 5, 20
    3) max_features: None(제한 없음), 1000, 3000, 5000

왜 valid로 튜닝하는가?
    test로 설정을 고르면 test 점수가 실제보다 좋게 나온다. test가 사실상 학습에 쓰인 셈이기 때문이다.
    그래서 사용자별 '끝에서 두 번째' 평점(valid)으로 설정을 고른다.
    test(마지막 평점)는 run_content_based.py에서 마지막 확인에만 쓴다.

!! 임시 분할 기준이다. A의 공통 분할이 나오면 그 valid로 다시 돌린다. !!

실행: python tune_content_based.py
결과: results/content_based_tuning.csv, results/content_based_tuning.png
"""

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 화면 없이 PNG 파일로만 저장하는 백엔드
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from content_based import ContentBasedRecommender  # noqa: E402
from recipe_parsing import prepare_recipes  # noqa: E402
from temp_eval import temp_split, sample_eval_users, hit_at_k, popular_hit_at_k  # noqa: E402

DATA_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data"
RESULT_DIR = ROOT / "results"
N_EVAL_USERS = 2000
KS = [10, 50]

DEFAULT = {"fields": ("ingredients", "tags"), "min_df": 2, "max_features": None}
FIELD_LABELS = {
    ("ingredients",): "ingredients",
    ("ingredients", "tags"): "ingr+tags",
    ("ingredients", "tags", "description"): "ingr+tags+desc",
}
# (실험 이름, 바꿀 파라미터 이름, 시도할 값들)
EXPERIMENTS = [
    ("Input fields", "fields", list(FIELD_LABELS)),
    ("min_df", "min_df", [1, 2, 5, 20]),
    ("max_features", "max_features", [None, 1000, 3000, 5000]),
]


def main():
    t0 = time.time()
    recipes = pd.read_csv(DATA_DIR / "RAW_recipes.csv")
    inter = pd.read_csv(DATA_DIR / "RAW_interactions.csv", usecols=["user_id", "recipe_id", "date", "rating"])
    inter = inter[inter["rating"] > 0]  # rating 0 = 리뷰만 남긴 경우 → 제거

    train, valid, _ = temp_split(inter)  # test는 여기서 쓰지 않는다
    eval_rows = sample_eval_users(valid, N_EVAL_USERS)
    print(f"[임시 분할] train {len(train):,} / valid {len(valid):,}, 평가 사용자 {len(eval_rows):,}명")

    # 재료·태그 파싱(ast.literal_eval)은 23만 행이라 수십 초 걸린다. 실험마다 반복하지 않도록 한 번만 한다.
    # 파싱된 리스트가 들어가면 fit 안의 parse_list_column은 그대로 통과시킨다.
    recipes = prepare_recipes(recipes)

    rows = []
    done = {}  # 같은 설정(기본값)을 여러 실험에서 다시 학습하지 않도록 결과를 저장해 둔다
    for exp_name, param, values in EXPERIMENTS:
        for value in values:
            config = {**DEFAULT, param: value}
            key = tuple(config.items())
            if key not in done:
                model = ContentBasedRecommender(**config).fit(recipes, train)
                hits = hit_at_k(model, eval_rows, KS)
                done[key] = (hits, model.X.shape[1])
                print(f"  {config}  단어 {model.X.shape[1]:,}개  "
                      + "  ".join(f"Hit@{k}={h:.4f}" for k, h in hits.items())
                      + f"  ({time.time() - t0:.0f}s)")
            hits, vocab = done[key]
            label = FIELD_LABELS[value] if param == "fields" else str(value)
            rows.append({"experiment": exp_name, "value": label, "vocab_size": vocab,
                         **{f"hit@{k}": h for k, h in hits.items()}})

    # 비교 기준: 같은 사용자들에게 인기순으로 추천한 결과 (모델 설정과 무관하므로 한 번만 계산)
    pop = popular_hit_at_k(model, eval_rows, KS)
    rows.append({"experiment": "Popularity", "value": "-", "vocab_size": 0,
                 **{f"hit@{k}": h for k, h in pop.items()}})

    res = pd.DataFrame(rows)
    RESULT_DIR.mkdir(exist_ok=True)
    res.to_csv(RESULT_DIR / "content_based_tuning.csv", index=False)
    print("\n" + res.round(4).to_string(index=False))

    # ---- plot: 실험별 Hit@10 막대 그래프, 인기순은 점선으로 표시 ----
    fig, axes = plt.subplots(1, len(EXPERIMENTS), figsize=(13, 4), sharey=True)
    for ax, (exp_name, _, _) in zip(axes, EXPERIMENTS):
        sub = res[res["experiment"] == exp_name]
        bars = ax.bar(sub["value"], sub["hit@10"], color="#5b8c5a")
        ax.bar_label(bars, fmt="%.4f", fontsize=8)  # 막대 위에 값 표시
        ax.axhline(pop[10], color="#e05a3a", linestyle="--", label="Popularity")
        ax.set_title(exp_name)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Hit@10 on validation")
    axes[0].legend()
    fig.suptitle(f"Content-based tuning (temporary split, {len(eval_rows):,} users)")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "content_based_tuning.png", dpi=150)
    print(f"\n완료 ({time.time() - t0:.0f}s). 결과: {RESULT_DIR}")


if __name__ == "__main__":
    main()
