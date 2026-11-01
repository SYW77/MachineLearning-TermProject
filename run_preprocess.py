"""
run_preprocess.py  (담당: Member A, 이민지)

전처리와 train / valid / test 분할을 실행하는 스크립트.
설정값(k-core, 최소 평점 수, valid/test 개수 등)은 src/preprocess.py 맨 위 상수에서 바꾼다.

실행 방법
    1) data/ 폴더에 RAW_interactions.csv를 넣는다.
    2) python run_preprocess.py              (데이터 폴더가 다르면: python run_preprocess.py <데이터폴더>)
    결과: data/processed/train.csv, valid.csv, test.csv, split_stats.json
    마지막에 preprocess.validate_splits로 분할을 검사하고 결과 표를 출력한다(실패하면 에러).

data/는 Git에 올리지 않으므로, 팀원은 각자 이 스크립트를 실행해 같은 분할 파일을 만든다.
같은 원본과 같은 설정이면 결과 파일이 항상 똑같다(동점은 원본 행 순서로 처리).
"""

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))  # src/ 폴더의 모듈을 import할 수 있게 경로에 추가
import preprocess  # noqa: E402

# 명령행 인자로 데이터 폴더를 바꿀 수 있다. 기본값은 ./data
DATA_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data"
OUT_DIR = ROOT / preprocess.OUTPUT_DIR


def main():
    t0 = time.time()
    stats, data = preprocess.run_pipeline(DATA_DIR / "RAW_interactions.csv", OUT_DIR, return_data=True)

    print("[설정]")
    for key, value in stats["config"].items():
        print(f"  {key} = {value}")

    # 단계별 행 수·사용자 수·레시피 수를 표로 출력한다.
    table = pd.DataFrame(stats["steps"]).set_index("step")
    print("\n[단계별 통계]")
    print(table.to_string(formatters={c: "{:,}".format for c in table.columns}))

    split_rows = sum(row["rows"] for row in stats["steps"] if row["step"].startswith("5."))
    print(f"\ntrain + valid + test = {split_rows:,}행 / 필터 후 전체 = {stats['rows_after_filtering']:,}행")

    # 저장된 파일을 다시 읽어 검사한다(팀원이 load_splits로 읽는 것과 같은 파일).
    # 실패하면 validate_splits가 실패 항목과 이유를 담아 에러를 내고 여기서 멈춘다.
    train, valid, test = preprocess.load_splits(OUT_DIR)
    checks = preprocess.validate_splits(train, valid, test, data["cleaned"])
    print("\n[분할 검증]")
    print(checks.to_string(index=False))
    print(f"\n완료 ({time.time() - t0:.1f}s). 결과: {OUT_DIR}")


if __name__ == "__main__":
    main()
