"""
preprocess.py  (담당: Member A, 이민지)

RAW_interactions.csv를 정리하고, 사용자별 시간순으로 train / valid / test를 나눈다.
모든 팀원(콘텐츠 기반, MF, 평가)이 이 파일이 만든 같은 분할을 쓴다.

처리 순서
    1) 필요한 4개 컬럼만 읽기 (user_id, recipe_id, date, rating)
    2) 원본 행 순서를 orig_order 컬럼으로 기록 → 같은 날짜 평가의 순서를 정하는 데 쓴다
    3) rating 0 제거 (리뷰만 남기고 평점은 주지 않은 경우)
    4) 같은 (user_id, recipe_id) 중복 검사 → 있으면 가장 최근 것만 남긴다
    5) KCORE_K가 정수면 반복 k-core 필터
    6) 사용자별 시간순 분할: 마지막 N_TEST개 → test, 그 앞 N_VALID개 → valid, 나머지 → train
    7) CSV 3개 + 단계별 통계 json 저장
    8) validate_splits로 분할 결과 검사 (run_preprocess.py가 마지막에 자동 실행)

다른 팀원이 쓰는 법
    from preprocess import load_splits
    train, valid, test = load_splits("data/processed")
    model.fit(recipes_df, train)

참고 자료 (코드 복사 없음, 공식 문서만 참고):
    - pandas DataFrame.sort_values
      https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_values.html
    - pandas GroupBy.cumcount
      https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.GroupBy.cumcount.html
    - Python json 모듈  https://docs.python.org/3/library/json.html
"""

import json  # 수업에서 다루지 않은 모듈 → save_splits 주석에서 설명
from pathlib import Path

import pandas as pd

# ============================================================================
# 팀 결정에 따라 바꿀 설정값 (여기만 고치면 된다)
# ============================================================================
MIN_USER_RATINGS = 3                # 평점이 이 개수 이상인 사용자만 valid/test 대상
KEEP_SPARSE_USERS_IN_TRAIN = True   # True면 평점이 MIN_USER_RATINGS 미만인 사용자도 train에 남긴다
KCORE_K = None                      # None이면 k-core 안 함. 정수면 사용자·레시피 모두 k개 이상 될 때까지 반복
N_VALID = 1                         # 사용자별 valid 개수 (test 바로 앞)
N_TEST = 1                          # 사용자별 test 개수 (가장 최근)
OUTPUT_DIR = "data/processed"       # 프로젝트 루트 기준 출력 폴더

# 저장하는 컬럼. 모델은 user_id, recipe_id, rating만 쓰고 date는 시간순 확인·분석용이다.
COLUMNS = ["user_id", "recipe_id", "date", "rating"]
# load_splits가 맞춰 주는 자료형. date는 "YYYY-MM-DD" 문자열 그대로 둔다.
DTYPES = {"user_id": "int64", "recipe_id": "int64", "rating": "int64", "date": str}


# ---------------------------------------------------------------- 단계별 함수
def load_interactions(path):
    """
    받는 것 : RAW_interactions.csv 경로
    하는 일 : 4개 컬럼만 읽고, 원본 파일에서 몇 번째 행이었는지 orig_order 컬럼에 기록한다.
    돌려주는 것: DataFrame (user_id, recipe_id, date, rating, orig_order)

    왜 orig_order가 필요한가?
        같은 사용자가 같은 날짜에 평가한 행이 전체의 41%나 된다.
        날짜만으로는 어느 것이 "마지막"인지 정할 수 없으므로,
        원본 파일 순서를 두 번째 기준으로 써서 매번 같은 결과가 나오게 한다.
    """
    # usecols: 읽을 컬럼 목록. review(긴 문장)를 안 읽어서 메모리와 시간을 아낀다.
    # dtype: 컬럼 자료형을 지정한다. date는 날짜 객체로 바꾸지 않고 문자열로 둔다.
    #   "YYYY-MM-DD" 문자열은 사전순 정렬 = 날짜순 정렬이라 그대로 비교해도 된다.
    df = pd.read_csv(path, usecols=COLUMNS, dtype=DTYPES)
    df["orig_order"] = range(len(df))  # 0, 1, 2, ... 원본 행 번호
    return df


def remove_zero_ratings(df):
    """
    받는 것 : 평점 DataFrame
    하는 일 : rating 0인 행을 지운다. 0점은 "리뷰만 쓰고 별점은 안 준" 경우라 평점이 아니다.
    돌려주는 것: rating 1~5만 남은 DataFrame
    """
    return df[df["rating"] > 0]


def drop_duplicate_pairs(df):
    """
    받는 것 : 평점 DataFrame (orig_order 포함)
    하는 일 : 같은 사용자가 같은 레시피를 여러 번 평가했으면 가장 최근 것만 남긴다.
              (가장 최근 = date가 가장 늦고, 같으면 원본에서 더 뒤에 있는 행)
    돌려주는 것: (중복 제거된 DataFrame, 지운 행 수)
    """
    # sort_values(by, kind): by의 컬럼 순서대로 정렬한다(앞 컬럼이 우선, 같으면 다음 컬럼으로 비교).
    #   kind="mergesort": 안정 정렬(값이 같은 행끼리는 원래 순서 유지).
    #   pandas 문서상 kind는 컬럼 1개로 정렬할 때만 적용되고, 여러 컬럼 정렬은 원래 안정 정렬로 동작한다.
    #   게다가 orig_order가 모든 행에서 다르므로 정렬 결과는 항상 하나로 정해진다.
    df = df.sort_values(["user_id", "recipe_id", "date", "orig_order"], kind="mergesort")
    # duplicated(subset, keep="last"): subset 컬럼 값이 같은 행 중 마지막 것만 False(=중복 아님)로 표시
    dup = df.duplicated(subset=["user_id", "recipe_id"], keep="last")
    return df[~dup], int(dup.sum())


def kcore_filter(df, k):
    """
    받는 것 : 평점 DataFrame, 정수 k
    하는 일 : 평점이 k개 미만인 사용자와 레시피를 지운다.
              지우고 나면 다른 사용자/레시피의 평점 수가 줄어 새로 k 미만이 될 수 있으므로,
              더 지울 것이 없을 때까지 반복한다.
    돌려주는 것: (k-core DataFrame, 반복 횟수)
    """
    n_iter = 0
    while True:
        n_iter += 1
        # value_counts(): 값별 개수. map(): 각 행의 user_id를 "그 사용자의 평점 수"로 바꾼다.
        user_cnt = df["user_id"].map(df["user_id"].value_counts())
        item_cnt = df["recipe_id"].map(df["recipe_id"].value_counts())
        keep = (user_cnt >= k) & (item_cnt >= k)
        if keep.all():  # 지울 행이 없으면 끝
            return df, n_iter
        df = df[keep]


def chronological_split(df, min_user_ratings=MIN_USER_RATINGS, n_valid=N_VALID, n_test=N_TEST,
                        keep_sparse_users=KEEP_SPARSE_USERS_IN_TRAIN):
    """
    받는 것 : 평점 DataFrame (orig_order 포함)과 분할 설정
    하는 일 : 사용자별로 날짜순 정렬 후
              - 평점 수 >= min_user_ratings 인 사용자: 마지막 n_test개 → test, 그 앞 n_valid개 → valid, 나머지 → train
              - 평점 수 <  min_user_ratings 인 사용자: keep_sparse_users면 전부 train, 아니면 제외
    돌려주는 것: (train, valid, test) DataFrame 3개

    예) 평점 5개인 사용자 (날짜순 a b c d e), n_valid=1, n_test=1
        → train: a b c / valid: d / test: e
    """
    # valid·test를 떼고도 train에 최소 1개가 남아야 콘텐츠 기반·MF가 그 사용자를 안다.
    if min_user_ratings < n_valid + n_test + 1:
        raise ValueError(f"MIN_USER_RATINGS는 N_VALID + N_TEST + 1 = {n_valid + n_test + 1} 이상이어야 합니다.")

    # 사용자 → 날짜 → 원본 순서로 정렬. orig_order가 같은 날짜 평가의 순서를 정한다(동점 처리).
    df = df.sort_values(["user_id", "date", "orig_order"], kind="mergesort")

    # groupby("user_id"): 사용자별로 묶는다.
    # transform("size"): 그룹 크기(그 사용자의 평점 수)를 그룹의 모든 행에 똑같이 붙여 돌려준다.
    #   결과 길이가 원래 행 수와 같아서 행마다 비교할 수 있다. (그냥 size()는 사용자 수 길이)
    n_ratings = df.groupby("user_id")["recipe_id"].transform("size")
    # cumcount(ascending=False): 그룹 안에서 순번을 매긴다. ascending=False면 뒤에서부터 0, 1, 2, ...
    #   → 0이 가장 최근 평점, 1이 그 바로 앞 평점.
    pos_from_end = df.groupby("user_id").cumcount(ascending=False)

    eligible = n_ratings >= min_user_ratings
    test_mask = eligible & (pos_from_end < n_test)
    valid_mask = eligible & (pos_from_end >= n_test) & (pos_from_end < n_test + n_valid)
    # 평점이 적은 사용자는 설정에 따라 train에 남기거나 아예 뺀다.
    train_mask = ~test_mask & ~valid_mask & (eligible | keep_sparse_users)
    return df[train_mask], df[valid_mask], df[test_mask]


# ---------------------------------------------------------------- 통계·저장·불러오기
def summarize(df):
    """
    받는 것 : 평점 DataFrame
    하는 일 : 행 수, 사용자 수, 레시피 수를 센다.
    돌려주는 것: {"rows": int, "users": int, "recipes": int}
    """
    # nunique(): 서로 다른 값의 개수
    return {"rows": len(df), "users": int(df["user_id"].nunique()), "recipes": int(df["recipe_id"].nunique())}


def save_splits(train, valid, test, stats, out_dir=OUTPUT_DIR):
    """
    받는 것 : train / valid / test DataFrame, 통계 딕셔너리, 출력 폴더
    하는 일 : train.csv, valid.csv, test.csv(컬럼 user_id, recipe_id, date, rating)와
              split_stats.json을 저장한다. orig_order는 내부용이라 저장하지 않는다.
    돌려주는 것: 없음
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)  # parents=True: 상위 폴더가 없으면 같이 만든다
    for name, part in [("train", train), ("valid", valid), ("test", test)]:
        # 사용자 → 날짜 → 원본 순서로 정렬해 저장한다. 매번 같은 순서라 파일이 똑같이 나온다.
        part = part.sort_values(["user_id", "date", "orig_order"], kind="mergesort")
        # index=False: 행 번호 컬럼을 저장하지 않는다.
        # lineterminator="\n": 줄바꿈을 OS와 상관없이 "\n"으로 고정한다.
        #   (Windows 기본값 "\r\n"이면 맥/리눅스 팀원과 파일 해시가 달라진다)
        part[COLUMNS].to_csv(out_dir / f"{name}.csv", index=False, lineterminator="\n")

    # json.dump(obj, fp, indent, ensure_ascii): 파이썬 딕셔너리를 JSON 텍스트 파일로 쓴다.
    #   indent=2: 사람이 읽기 좋게 2칸 들여쓰기, ensure_ascii=False: 한글을 \uXXXX로 바꾸지 않고 그대로 쓴다.
    #   실행 시각 같은 값은 넣지 않는다. 넣으면 같은 데이터라도 매번 파일이 달라진다.
    with open(out_dir / "split_stats.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


def load_splits(data_dir=OUTPUT_DIR):
    """
    받는 것 : 분할 파일이 있는 폴더 (기본값 data/processed, 실행 위치 기준 상대 경로)
    하는 일 : train.csv, valid.csv, test.csv를 읽고 자료형을 맞춘다.
              user_id, recipe_id, rating → int64, date → 문자열("YYYY-MM-DD")
    돌려주는 것: (train, valid, test) DataFrame 3개

    사용 예)
        train, valid, test = load_splits("data/processed")
        model.fit(recipes_df, train)              # 튜닝할 때
        model.fit(recipes_df, pd.concat([train, valid]))  # 최종 test 평가할 때 (팀 합의 필요)
    """
    data_dir = Path(data_dir)
    return tuple(pd.read_csv(data_dir / f"{name}.csv", dtype=DTYPES) for name in ("train", "valid", "test"))


# ---------------------------------------------------------------- 검증
def _n_users_out_of_order(earlier, later):
    """
    받는 것 : 사용자별 날짜 Series 2개 (index = user_id). earlier가 later보다 늦으면 안 된다.
    하는 일 : 두 Series에 모두 있는 사용자만 비교해 earlier > later인 사용자 수를 센다.
              (한쪽에만 있는 사용자는 "valid·test 사용자 ⊂ train 사용자" 항목에서 잡는다)
    돌려주는 것: 순서가 틀린 사용자 수
    """
    # concat(axis=1, join="inner"): user_id를 기준으로 옆으로 붙이고, 양쪽에 다 있는 사용자만 남긴다.
    both = pd.concat([earlier, later], axis=1, keys=["earlier", "later"], join="inner")
    return int((both["earlier"] > both["later"]).sum())


def validate_splits(train, valid, test, cleaned):
    """
    받는 것 : train / valid / test DataFrame, cleaned(분할에 들어간 전체 데이터.
              평점 적은 사용자를 빼는 설정이면 그 사용자를 뺀 것)
    하는 일 : 아래 5가지를 검사한다.
              1) 평점 0 없음
              2) 세 파일 간 (user_id, recipe_id) 겹침 없음
              3) valid·test 사용자가 모두 train에 있음
              4) 사용자별 train 최신 날짜 <= valid 날짜 <= test 날짜
              5) train + valid + test 행 수 = cleaned 행 수
    돌려주는 것: 검사 결과 표 DataFrame (항목, 결과, 내용). 하나라도 실패하면
              실패한 항목과 이유를 담아 ValueError를 낸다.
    """
    parts = {"train": train, "valid": valid, "test": test}
    key = ["user_id", "recipe_id"]
    results = []  # (항목, 통과 여부, 내용)

    # 1) 평점 0 없음
    zeros = {name: int((part["rating"] <= 0).sum()) for name, part in parts.items()}
    results.append(("평점 0 없음", sum(zeros.values()) == 0,
                    "rating <= 0 행 수 " + ", ".join(f"{n} {c:,}" for n, c in zeros.items())))

    # 2) (user_id, recipe_id) 겹침 없음
    # merge(on=key): 두 표에서 key 값이 같은 행끼리 붙인다. 결과 행 수 = 겹치는 쌍의 수.
    overlaps = {f"{a}-{b}": len(parts[a][key].merge(parts[b][key], on=key))
                for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]}
    results.append(("(user_id, recipe_id) 겹침 없음", sum(overlaps.values()) == 0,
                    "겹치는 쌍 " + ", ".join(f"{n} {c:,}" for n, c in overlaps.items())))

    # 3) valid·test 사용자 ⊂ train 사용자
    # isin: 값이 train의 user_id 중에 있으면 True. ~로 뒤집어 train에 없는 사용자를 고른다.
    missing = {name: int(parts[name].loc[~parts[name]["user_id"].isin(train["user_id"]), "user_id"].nunique())
               for name in ("valid", "test")}
    results.append(("valid·test 사용자 ⊂ train 사용자", sum(missing.values()) == 0,
                    "train에 없는 사용자 " + ", ".join(f"{n} {c:,}명" for n, c in missing.items())))

    # 4) 사용자별 날짜 순서
    # pd.to_datetime(format=...): "YYYY-MM-DD" 문자열을 날짜 자료형으로 바꾼다.
    #   문자열 그대로 groupby().max()를 하면 100만 행에서 20초 이상 걸리지만, 날짜 자료형은 1초 안에 끝난다.
    def user_dates(part, how):
        dates = pd.to_datetime(part["date"], format="%Y-%m-%d")
        return dates.groupby(part["user_id"]).agg(how)  # how: "max"(가장 늦은 날짜) 또는 "min"(가장 이른 날짜)

    bad_train_valid = _n_users_out_of_order(user_dates(train, "max"), user_dates(valid, "min"))
    bad_valid_test = _n_users_out_of_order(user_dates(valid, "max"), user_dates(test, "min"))
    results.append(("사용자별 날짜 순서 train <= valid <= test", bad_train_valid + bad_valid_test == 0,
                    f"순서가 틀린 사용자 train>valid {bad_train_valid:,}명, valid>test {bad_valid_test:,}명"))

    # 5) 행 수 합계
    n_split = len(train) + len(valid) + len(test)
    results.append(("행 수 합계 일치", n_split == len(cleaned),
                    f"train+valid+test {n_split:,}행 / 분할 전 {len(cleaned):,}행"))

    table = pd.DataFrame(results, columns=["항목", "통과", "내용"])
    failed = table[~table["통과"]]
    if len(failed) > 0:
        lines = [f"- {row.항목}: {row.내용}" for row in failed.itertuples()]
        raise ValueError(f"분할 검증 실패 ({len(failed)}개 항목)\n" + "\n".join(lines))
    table["통과"] = table["통과"].map({True: "통과", False: "실패"})
    table = table.rename(columns={"통과": "결과"})
    return table


# ---------------------------------------------------------------- 전체 실행
def run_pipeline(raw_path, out_dir=OUTPUT_DIR, kcore_k=KCORE_K, min_user_ratings=MIN_USER_RATINGS,
                 n_valid=N_VALID, n_test=N_TEST, keep_sparse_users=KEEP_SPARSE_USERS_IN_TRAIN,
                 return_data=False):
    """
    받는 것 : RAW_interactions.csv 경로, 출력 폴더, 설정값 (기본값은 파일 맨 위 상수),
              return_data (True면 검증에 쓸 DataFrame도 같이 돌려준다)
    하는 일 : 1)~7) 전 과정을 실행하고 파일을 저장한다.
    돌려주는 것: stats 딕셔너리 (설정값, 단계별 행/사용자/레시피 수, 분할 결과)
              return_data=True면 (stats, {"train", "valid", "test", "cleaned"} 딕셔너리)
    """
    steps = []  # 단계별 통계를 순서대로 모은다

    df = load_interactions(raw_path)
    steps.append({"step": "1. raw", **summarize(df)})

    df = remove_zero_ratings(df)
    steps.append({"step": "2. rating > 0", **summarize(df)})

    df, n_dup = drop_duplicate_pairs(df)
    steps.append({"step": f"3. dedup (removed {n_dup})", **summarize(df)})

    kcore_iters = None
    if kcore_k is not None:
        df, kcore_iters = kcore_filter(df, kcore_k)
        steps.append({"step": f"4. {kcore_k}-core ({kcore_iters} iters)", **summarize(df)})

    train, valid, test = chronological_split(df, min_user_ratings, n_valid, n_test, keep_sparse_users)
    # 분할에 들어가야 할 전체 데이터. 평점 적은 사용자를 빼는 설정이면 그 사용자의 행을 뺀 것.
    user_cnt = df["user_id"].map(df["user_id"].value_counts())
    cleaned = df if keep_sparse_users else df[user_cnt >= min_user_ratings]
    filtered_rows = len(cleaned)
    steps.append({"step": "5. split: train", **summarize(train)})
    steps.append({"step": "5. split: valid", **summarize(valid)})
    steps.append({"step": "5. split: test", **summarize(test)})

    stats = {
        "config": {
            "MIN_USER_RATINGS": min_user_ratings,
            "KEEP_SPARSE_USERS_IN_TRAIN": keep_sparse_users,
            "KCORE_K": kcore_k,
            "N_VALID": n_valid,
            "N_TEST": n_test,
        },
        "duplicates_removed": n_dup,
        "kcore_iterations": kcore_iters,
        "rows_after_filtering": filtered_rows,  # train + valid + test 행 수 합과 같아야 한다
        "steps": steps,
    }
    save_splits(train, valid, test, stats, out_dir)
    if return_data:
        return stats, {"train": train, "valid": valid, "test": test, "cleaned": cleaned}
    return stats
