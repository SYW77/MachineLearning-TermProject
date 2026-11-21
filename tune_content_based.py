"""
tune_content_based.py  (담당: Member C, 3주차 튜닝 실험)

콘텐츠 기반 추천기의 설정을 바꿔 가며 공통 분할의 valid에서 Precision@K·Recall@K를 비교한다.

실험 (기본 설정: 재료+태그, min_df=2, max_features=None)
    한 번에 하나만 바꾸고 나머지는 기본값으로 둔다.
    1) 입력 필드  : 재료만 / 재료+태그 / 재료+태그+설명
    2) min_df     : 1, 2, 5, 20
    3) max_features: None(제한 없음), 1000, 3000, 5000

왜 valid로 튜닝하는가?
    test로 설정을 고르면 test 점수가 실제보다 좋게 나온다. test가 사실상 학습에 쓰인 셈이기 때문이다.
    그래서 valid로 설정을 고르고, test는 run_evaluate.py --split test로 마지막에 한 번만 본다.

데이터와 지표
    - 분할: preprocess.load_splits (공통 분할. 먼저 python run_preprocess.py 실행)
    - 지표: evaluate.evaluate_topk (공통 평가 코드). valid는 사용자당 1개라서 Recall@K = Hit@K 이다.

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
from evaluate import PopularityBaseline, evaluate_topk  # noqa: E402
from preprocess import load_splits  # noqa: E402
from recipe_parsing import prepare_recipes  # noqa: E402

DATA_DIR = ROOT / "data"
RESULT_DIR = ROOT / "results"
# 평가 사용자 수. 처음 임시 실험(2,000명)은 설정 간 hit 차이가 몇 명뿐이라 우연과 구별하기 어려웠다.
# 그래서 10,000명으로 늘렸다. 전체(약 3만 3천 명)는 설정 9개를 돌리는 데 2시간 가까이 걸린다.
N_EVAL_USERS = 10000
KS = (10, 50)

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


def evaluate(model, valid):
    """공통 평가 코드로 valid를 평가하고 {"recall@10": ..., "precision@10": ..., ...}를 돌려준다."""
    # 모델이 달라도 seed가 같으면 같은 사용자들이 뽑힌다(evaluate_topk 설명 참고). 그래서 설정끼리 공정하게 비교된다.
    # k를 인덱스로 바꿔서 res.loc[10, "recall"]처럼 찾음
    res = evaluate_topk(model, valid, ks=KS, max_users=N_EVAL_USERS).set_index("k")
    out = {"n_users": int(res["n_users"].iloc[0])}
    for k in KS:
        out[f"recall@{k}"] = res.loc[k, "recall"]
        out[f"precision@{k}"] = res.loc[k, "precision"]
    return out


def main():
    t0 = time.time()
    train, valid, _ = load_splits(DATA_DIR / "processed")  # test는 여기서 쓰지 않는다
    recipes = pd.read_csv(DATA_DIR / "RAW_recipes.csv")
    print(f"[공통 분할] train {len(train):,} / valid {len(valid):,}")

    # 재료·태그 파싱(ast.literal_eval)은 23만 행이라 수십 초 걸린다. 실험마다 반복하지 않도록 한 번만 한다.
    # 파싱된 리스트가 들어가면 fit 안의 parse_list_column은 그대로 통과시킨다.
    recipes = prepare_recipes(recipes)

    rows = []
    done = {}  # 같은 설정(기본값)을 여러 실험에서 다시 학습하지 않도록 결과를 저장해 둔다
    for exp_name, param, values in EXPERIMENTS:
        for value in values:
            config = {**DEFAULT, param: value}  # 기본 설정에서 파라미터 하나만 바꿈
            key = tuple(config.items())  # dict는 키로 못 써서 튜플로 바꿈
            if key not in done:
                model = ContentBasedRecommender(**config).fit(recipes, train)
                done[key] = {"vocab_size": model.X.shape[1], **evaluate(model, valid)}
                r = done[key]
                print(f"  {config}  단어 {r['vocab_size']:,}개  "
                      + "  ".join(f"Recall@{k}={r[f'recall@{k}']:.4f}" for k in KS)
                      + f"  ({time.time() - t0:.0f}s)", flush=True)
            label = FIELD_LABELS[value] if param == "fields" else str(value)
            rows.append({"experiment": exp_name, "value": label, **done[key]})

    # 비교 기준: 공통 평가 코드의 인기순 기준선 (같은 사용자들)
    pop = evaluate(PopularityBaseline().fit(None, train), valid)
    rows.append({"experiment": "Popularity", "value": "-", "vocab_size": 0, **pop})

    res = pd.DataFrame(rows)
    RESULT_DIR.mkdir(exist_ok=True)
    res.to_csv(RESULT_DIR / "content_based_tuning.csv", index=False)
    print("\n" + res.round(5).to_string(index=False))

    # ---- plot: 실험별 Recall@10 막대 그래프, 인기순은 점선으로 표시 ----
    fig, axes = plt.subplots(1, len(EXPERIMENTS), figsize=(13, 4), sharey=True)  # sharey=True: y축 범위를 같게
    for ax, (exp_name, _, _) in zip(axes, EXPERIMENTS):
        sub = res[res["experiment"] == exp_name]
        bars = ax.bar(sub["value"], sub["recall@10"], color="#5b8c5a")
        ax.bar_label(bars, fmt="%.4f", fontsize=8)  # 막대 위에 값 표시
        ax.axhline(pop["recall@10"], color="#e05a3a", linestyle="--", label="Popularity")  # 인기순 기준선
        ax.set_title(exp_name)
        ax.tick_params(axis="x", rotation=15)  # x축 글자 기울이기
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Recall@10 on validation")
    axes[0].legend()
    fig.suptitle(f"Content-based tuning (shared split, valid, {pop['n_users']:,} users)")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "content_based_tuning.png", dpi=150)
    print(f"\n완료 ({time.time() - t0:.0f}s). 결과: {RESULT_DIR}")


if __name__ == "__main__":
    main()
