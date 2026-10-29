"""
temp_eval.py  (담당: Member C)

A의 공통 분할과 D의 평가 코드가 나오기 전까지 쓰는 '임시' 분할·확인 지표.
run_content_based.py(동작 확인)와 tune_content_based.py(튜닝)가 함께 쓴다.
여기서 나온 수치는 최종 보고용이 아니다.
"""

import numpy as np

from content_based import LIKE_THRESHOLD


def temp_split(inter):
    """
    사용자별로 시간순 정렬한 뒤 마지막 평점 1개 → test, 그 바로 앞 1개 → valid, 나머지 → train.

    - test로 빼려면 평점이 2개 이상, valid까지 빼려면 3개 이상이어야 한다.
      그보다 적은 사용자의 평점은 train에 남긴다.
    - 튜닝은 valid로만 하고, test는 마지막 확인에만 쓴다.
      test로 튜닝하면 test 성능이 실제보다 좋게 나온다.
    """
    inter = inter.sort_values(["user_id", "date"])
    n = inter.groupby("user_id")["recipe_id"].transform("size")
    # cumcount(ascending=False): 그룹 안에서 뒤에서부터 0, 1, 2…를 매긴다 → 0이 가장 최근 평점
    pos_from_end = inter.groupby("user_id").cumcount(ascending=False)
    test_mask = (pos_from_end == 0) & (n >= 2)
    valid_mask = (pos_from_end == 1) & (n >= 3)
    return inter[~test_mask & ~valid_mask], inter[valid_mask], inter[test_mask]


def sample_eval_users(holdout, n_users, seed=42):
    """holdout(valid 또는 test)에서 4점 이상인 행만 골라 무작위로 n_users명을 뽑는다."""
    pos = holdout[holdout["rating"] >= LIKE_THRESHOLD]
    return pos.sample(min(n_users, len(pos)), random_state=seed)


def hit_at_k(model, eval_rows, ks):
    """
    Hit@K: 사용자가 실제로 높게 평가한 홀드아웃 레시피 1개가 추천 상위 K개 안에 있으면 1, 없으면 0.
    평가 대상 사용자 전체의 평균을 낸다. 사용자당 정답이 1개라서 Recall@K와 같다.

    반환: {K: hit 비율}
    """
    max_k = max(ks)
    hits = {k: 0 for k in ks}
    for user_id, target in zip(eval_rows["user_id"], eval_rows["recipe_id"]):
        recs = model.recommend(user_id, k=max_k)
        for k in ks:
            hits[k] += target in recs[:k]
    return {k: hits[k] / len(eval_rows) for k in ks}


def popular_hit_at_k(model, eval_rows, ks):
    """비교용: 같은 사용자들에게 인기순(model 내부 대체 목록)으로 추천했을 때의 Hit@K."""
    max_k = max(ks)
    hits = {k: 0 for k in ks}
    for user_id, target in zip(eval_rows["user_id"], eval_rows["recipe_id"]):
        recs = model._popular(model.rated_rows.get(user_id, np.array([], dtype=int)), max_k)
        for k in ks:
            hits[k] += target in recs[:k]
    return {k: hits[k] / len(eval_rows) for k in ks}
