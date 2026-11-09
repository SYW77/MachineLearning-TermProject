"""공통 분할로 모델을 평가한다: 인기순 기준선 vs 콘텐츠 기반 vs 행렬 분해.

실행 (프로젝트 루트에서):
    python run_evaluate.py                          # valid로 세 모델 모두 평가 (튜닝·확인용)
    python run_evaluate.py --split test             # test로 최종 평가 (최종 수치용, train+valid로 학습)
    python run_evaluate.py --users 0                # 평가 사용자 전부 (오래 걸림)
    python run_evaluate.py --models popularity content_based   # 일부 모델만 (행렬 분해는 학습이 오래 걸린다)

결과: results/eval_<split>_topk.csv/.png, (RMSE가 있으면) eval_<split>_rmse.csv/.png
      --models로 일부만 돌리면 파일 이름 뒤에 모델 이름이 붙어서 전체 결과를 덮어쓰지 않는다.
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))  # src/ 폴더의 모듈을 import할 수 있게 경로에 추가
from preprocess import load_splits  # noqa: E402
from content_based import ContentBasedRecommender  # noqa: E402
from matrix_factorization import MatrixFactorizationRecommender  # noqa: E402
from evaluate import (  # noqa: E402
    PopularityBaseline, evaluate_topk, evaluate_rmse, plot_topk, plot_rmse,
)

KS = (5, 10, 20, 50)

# 콘텐츠 기반 설정. run_content_based.py / tune_content_based.py에서 고른 값과 맞춘다.
FIELDS = ("ingredients", "tags", "description")
MIN_DF = 2
MAX_FEATURES = None

# 행렬 분해 설정. 팀에서 정한 값이 있으면 여기만 바꾼다. (현재는 클래스 기본값과 같다)
MF_PARAMS = dict(n_factors=20, learning_rate=0.01, reg=0.02, n_epochs=20, random_state=42)

MODEL_FACTORIES = {
    "popularity": lambda: PopularityBaseline(),
    "content_based": lambda: ContentBasedRecommender(
        fields=FIELDS, min_df=MIN_DF, max_features=MAX_FEATURES),
    "matrix_factorization": lambda: MatrixFactorizationRecommender(**MF_PARAMS),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["valid", "test"], default="valid")
    ap.add_argument("--users", type=int, default=2000, help="평가 사용자 수. 0이면 전부")
    ap.add_argument("--models", nargs="+", choices=list(MODEL_FACTORIES),
                    default=list(MODEL_FACTORIES), help="평가할 모델 (기본: 전부)")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = ap.parse_args()

    t0 = time.time()
    train, valid, test = load_splits(str(args.data_dir / "processed"))
    recipes = pd.read_csv(args.data_dir / "RAW_recipes.csv")

    if args.split == "valid":
        fit_df, eval_df = train, valid
    else:
        # test를 평가할 때는 valid도 학습에 쓴다. 그래야 사용자가 valid에서 본 레시피도 추천에서 빠진다.
        fit_df, eval_df = pd.concat([train, valid]), test
    print(f"[{args.split}] 학습 평점 {len(fit_df):,}개 / 평가 평점 {len(eval_df):,}개")

    max_users = args.users or None
    topk, rmses = {}, {}
    for name in args.models:
        t = time.time()
        print(f"\n[{name}] 학습 중...", flush=True)
        model = MODEL_FACTORIES[name]()
        model.fit(recipes, fit_df)
        t_fit = time.time() - t
        print(f"[{name}] 학습 {t_fit:.1f}s, 평가 중...", flush=True)
        topk[name] = evaluate_topk(model, eval_df, ks=KS, max_users=max_users)
        print(topk[name].round(5).to_string(index=False))
        if hasattr(model, "predict"):  # 평점을 예측하는 모델만 RMSE를 계산한다
            r = evaluate_rmse(model, eval_df)
            rmses[name] = r["rmse"]
            print(f"RMSE {r['rmse']:.4f} (coverage {r['coverage']:.1%})")
        else:
            print("RMSE: predict()가 없어 건너뜀")
        print(f"[{name}] 총 {time.time() - t:.1f}s")

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    stem = f"eval_{args.split}"
    if set(args.models) != set(MODEL_FACTORIES):
        stem += "_" + "+".join(args.models)
    combined = pd.concat(topk, names=["model", "row"]).reset_index(level="model")
    combined.to_csv(out / f"{stem}_topk.csv", index=False)
    n_users = int(next(iter(topk.values()))["n_users"].iloc[0])
    plot_topk(topk, out / f"{stem}_topk.png", f" ({args.split}, {n_users:,} users)")
    if rmses:
        pd.Series(rmses, name="rmse").to_csv(out / f"{stem}_rmse.csv")
        plot_rmse(rmses, out / f"{stem}_rmse.png")
    print(f"\n완료 ({time.time() - t0:.1f}s). 결과: {out}")


if __name__ == "__main__":
    main()