"""
run_content_based.py  (담당: Member C)

콘텐츠 기반 추천기를 공통 분할로 학습하고, 사용자 몇 명에게 실제로 무엇을 추천하는지 보여 주는 스크립트.
숫자 평가(Precision@K, Recall@K)는 공통 평가 코드 run_evaluate.py가 한다.
이 스크립트는 "추천이 내용상 말이 되는가"를 눈으로 확인하는 용도다.

실행 방법
    1) python run_preprocess.py   (공통 분할 data/processed/ 생성, 처음 한 번)
    2) python run_content_based.py
    결과: results/content_based_examples.txt
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))  # src/ 폴더의 모듈을 import할 수 있게 경로에 추가
from content_based import ContentBasedRecommender  # noqa: E402
from preprocess import load_splits  # noqa: E402

DATA_DIR = ROOT / "data"
RESULT_DIR = ROOT / "results"
N_EXAMPLES = 3  # 예시로 보여 줄 사용자 수
SEED = 42

# 모델 설정. run_evaluate.py의 콘텐츠 기반 설정과 같게 맞춘다 (tune_content_based.py 결과로 고른 값).
FIELDS = ("ingredients", "tags", "description")
MIN_DF = 2
MAX_FEATURES = None


def main():
    train, valid, test = load_splits(DATA_DIR / "processed")
    recipes = pd.read_csv(DATA_DIR / "RAW_recipes.csv")
    names = recipes.set_index("id")["name"]

    # 최종 평가(run_evaluate.py --split test)와 같이 train+valid로 학습한다.
    model = ContentBasedRecommender(fields=FIELDS, min_df=MIN_DF, max_features=MAX_FEATURES)
    model.fit(recipes, pd.concat([train, valid]))

    # 좋아한 레시피가 5개 이상인 test 사용자 중 무작위로 몇 명을 고른다. 너무 적으면 취향이 잘 안 보인다.
    candidates = [u for u in test["user_id"] if len(model.liked_rows.get(u, [])) >= 5]
    users = pd.Series(candidates).sample(N_EXAMPLES, random_state=SEED)
    test_item = test.set_index("user_id")["recipe_id"]

    lines = [f"TF-IDF 행렬: {model.X.shape[0]:,} 레시피 × {model.X.shape[1]:,} 단어", ""]
    for user_id in users:
        lines.append(f"[user {user_id}] 좋아한 레시피 (일부):")
        lines += [f"   - {names.get(rid)}" for rid in model.recipe_ids[model.liked_rows[user_id]][:5]]
        lines.append("  추천 Top-5:")
        lines += [f"   - {names.get(rid)}" for rid in model.recommend(user_id, k=5)]
        lines.append(f"  실제 test 레시피: {names.get(test_item[user_id])}")
        lines.append("")

    text = "\n".join(lines)
    print(text)
    RESULT_DIR.mkdir(exist_ok=True)
    (RESULT_DIR / "content_based_examples.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
