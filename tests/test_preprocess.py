"""
test_preprocess.py  (담당: Member A, 이민지)

src/preprocess.py의 chronological_split과 validate_splits를 작은 가짜 데이터로 확인한다.
실행: 프로젝트 루트에서  python -m pytest tests -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest  # pytest.raises: with 안에서 그 예외가 나야 통과

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # src/ 폴더를 import 경로에 추가
from preprocess import chronological_split, validate_splits  # noqa: E402


def make_toy():
    """
    가짜 평점 데이터 (사용자 5명). orig_order = 이 표에서의 행 번호(원본 파일 순서 역할).
        사용자 1: 평점 5개, 날짜 모두 다름          → train 101~103, valid 104, test 105
        사용자 2: 평점 3개, 마지막 2개가 같은 날짜  → 원본에서 더 뒤인 203이 test, 202가 valid
        사용자 3: 평점 2개                          → MIN_USER_RATINGS(3) 미만이라 train에만
        사용자 4: 평점 1개                          → train에만
        사용자 5: 평점 4개, 파일 순서가 날짜순이 아니고 3월 1일이 동점
                  → 정렬 후 501(1월) 502(2월) 503(3월, 앞) 504(3월, 뒤) → valid 503, test 504
    """
    rows = [
        (1, 101, "2020-01-01", 5), (1, 102, "2020-01-02", 4), (1, 103, "2020-01-03", 3),
        (1, 104, "2020-01-04", 5), (1, 105, "2020-01-05", 2),
        (2, 201, "2020-01-01", 5), (2, 202, "2020-01-05", 4), (2, 203, "2020-01-05", 5),
        (3, 301, "2020-01-01", 4), (3, 302, "2020-02-01", 5),
        (4, 401, "2020-01-01", 3),
        (5, 503, "2020-03-01", 5), (5, 501, "2020-01-01", 4), (5, 504, "2020-03-01", 1),
        (5, 502, "2020-02-01", 5),
    ]
    df = pd.DataFrame(rows, columns=["user_id", "recipe_id", "date", "rating"])
    df["orig_order"] = range(len(df))
    return df


def pairs(df):
    """DataFrame의 (user_id, recipe_id) 쌍 집합"""
    return set(zip(df["user_id"], df["recipe_id"]))


# ---------------------------------------------------------------- chronological_split
def test_split_puts_latest_in_test_and_previous_in_valid():
    train, valid, test = chronological_split(make_toy(), min_user_ratings=3, n_valid=1, n_test=1,
                                             keep_sparse_users=True)
    assert pairs(test) == {(1, 105), (2, 203), (5, 504)}
    assert pairs(valid) == {(1, 104), (2, 202), (5, 503)}
    assert pairs(train) == {(1, 101), (1, 102), (1, 103), (2, 201), (3, 301), (3, 302), (4, 401),
                            (5, 501), (5, 502)}


def test_same_date_tie_is_decided_by_original_order():
    # 사용자 2의 같은 날짜 두 행(202, 203)의 원본 순서를 바꾸면 test도 바뀌어야 한다.
    df = make_toy()
    i202 = df.index[df["recipe_id"] == 202][0]
    i203 = df.index[df["recipe_id"] == 203][0]
    df.loc[i202, "orig_order"], df.loc[i203, "orig_order"] = df.loc[i203, "orig_order"], df.loc[i202, "orig_order"]
    _, valid, test = chronological_split(df, min_user_ratings=3, n_valid=1, n_test=1, keep_sparse_users=True)
    assert (2, 202) in pairs(test)
    assert (2, 203) in pairs(valid)


def test_split_is_same_when_run_twice():
    first = chronological_split(make_toy())
    second = chronological_split(make_toy())
    for a, b in zip(first, second):
        assert a.equals(b)


def test_sparse_users_kept_only_in_train():
    train, valid, test = chronological_split(make_toy(), min_user_ratings=3, keep_sparse_users=True)
    for user in (3, 4):
        assert user in set(train["user_id"])
        assert user not in set(valid["user_id"]) | set(test["user_id"])


def test_sparse_users_dropped_when_not_kept():
    train, valid, test = chronological_split(make_toy(), min_user_ratings=3, keep_sparse_users=False)
    all_users = set(train["user_id"]) | set(valid["user_id"]) | set(test["user_id"])
    assert all_users == {1, 2, 5}


def test_min_user_ratings_too_small_raises():
    # valid 1개 + test 1개를 떼면 train에 남는 것이 없으므로 에러가 나야 한다.
    with pytest.raises(ValueError):
        chronological_split(make_toy(), min_user_ratings=2, n_valid=1, n_test=1)


# ---------------------------------------------------------------- validate_splits
def split_toy():
    df = make_toy()
    train, valid, test = chronological_split(df, min_user_ratings=3, keep_sparse_users=True)
    return train.copy(), valid.copy(), test.copy(), df


def test_validate_passes_on_correct_split():
    train, valid, test, cleaned = split_toy()
    table = validate_splits(train, valid, test, cleaned)
    assert len(table) == 5
    assert (table["결과"] == "통과").all()


def test_validate_fails_on_zero_rating():
    train, valid, test, cleaned = split_toy()
    test.loc[test.index[0], "rating"] = 0
    with pytest.raises(ValueError) as err:
        validate_splits(train, valid, test, cleaned)
    assert "평점 0 없음" in str(err.value)


def test_validate_fails_on_overlapping_pair():
    # test의 한 행을 train에도 넣는다 → 같은 쌍이 두 파일에 있음 (행 수 합계도 함께 틀어짐)
    train, valid, test, cleaned = split_toy()
    train = pd.concat([train, test.iloc[[0]]])
    with pytest.raises(ValueError) as err:
        validate_splits(train, valid, test, cleaned)
    assert "겹침 없음" in str(err.value)


def test_validate_fails_when_test_user_missing_from_train():
    # 사용자 1의 train 행을 모두 test로 옮긴다 → 사용자 1이 train에 없음
    train, valid, test, cleaned = split_toy()
    moved = train[train["user_id"] == 1]
    train = train[train["user_id"] != 1]
    test = pd.concat([test, moved])
    with pytest.raises(ValueError) as err:
        validate_splits(train, valid, test, cleaned)
    assert "train 사용자" in str(err.value)


def test_validate_fails_on_wrong_date_order():
    # 사용자 1의 valid 날짜를 test보다 늦게 바꾼다
    train, valid, test, cleaned = split_toy()
    valid.loc[valid["user_id"] == 1, "date"] = "2021-01-01"
    with pytest.raises(ValueError) as err:
        validate_splits(train, valid, test, cleaned)
    message = str(err.value)
    assert "날짜 순서" in message
    assert "valid>test 1명" in message


def test_validate_fails_on_row_count_only():
    # train에서 1행만 지운다 → 다른 항목은 통과하고 행 수 합계만 실패해야 한다
    train, valid, test, cleaned = split_toy()
    train = train.iloc[1:]
    with pytest.raises(ValueError) as err:
        validate_splits(train, valid, test, cleaned)
    message = str(err.value)
    assert "(1개 항목)" in message
    assert "행 수 합계" in message
