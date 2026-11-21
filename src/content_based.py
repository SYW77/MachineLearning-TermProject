"""
content_based.py  (담당: Member C)

TF-IDF와 코사인 유사도를 쓰는 콘텐츠 기반 레시피 추천기.

전체 흐름
    1) 레시피마다 "재료 + 태그" 텍스트를 TF-IDF 벡터로 만든다.  → 레시피 × 단어 행렬 X
    2) 사용자가 학습 데이터에서 4점 이상 준 레시피 벡터를 평균 낸다. → 사용자 프로필 p
    3) p와 모든 레시피의 코사인 유사도를 한 번에 계산한다.       → X · p
    4) 이미 평가한 레시피를 빼고 점수 상위 K개를 추천한다.
    5) 4점 이상 레시피가 없는 사용자에게는 인기순으로 추천한다.

팀 공통 인터페이스
    model = ContentBasedRecommender()
    model.fit(recipes_df, train_df)        # train_df 컬럼: user_id, recipe_id, rating
    model.recommend(user_id, k=10) -> list # recipe_id 리스트

참고 자료 (코드 복사 없음, 공식 문서만 참고):
    - scikit-learn TfidfVectorizer 문서
      https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    - SciPy sparse 문서  https://docs.scipy.org/doc/scipy/reference/sparse.html
"""

import numpy as np
import pandas as pd

# scipy.sparse: 희소 행렬 모듈. 0이 대부분인 행렬에서 0이 아닌 값만 저장한다.
#   레시피 23만 개 × 단어 수만 개를 일반 배열로 만들면 수십 GB가 필요하다.
#   레시피 하나에는 단어가 20~30개뿐이라 희소 행렬로 저장하면 수십 MB면 된다.
#   csr_matrix(Compressed Sparse Row): 행 단위로 압축 저장하는 형식이다.
#   "특정 행 꺼내기"와 "행렬 × 벡터 곱"이 빠르다. 이 파일에서 하는 연산이 바로 이 두 가지다.
from scipy import sparse

# TfidfVectorizer: 문서 목록을 TF-IDF 희소 행렬로 바꿔 주는 scikit-learn 클래스.
from sklearn.feature_extraction.text import TfidfVectorizer

from recipe_parsing import prepare_recipes

LIKE_THRESHOLD = 4  # rating >= 4 이면 "좋아한 레시피"로 본다 (평가 기준과 같다).


class ContentBasedRecommender:
    def __init__(self, fields=("ingredients", "tags"), min_df=2, max_features=None):
        """
        파라미터 (3주차 튜닝 대상)
            fields       : 텍스트에 넣을 필드. recipe_parsing.build_recipe_text 참고.
            min_df       : 이 개수보다 적은 레시피에만 나온 단어는 버린다.
                           2면 딱 한 레시피에만 나온 단어(오타, 희귀 재료)를 버린다.
                           그런 단어는 다른 레시피와의 유사도에 기여하지 못하고 행렬만 키운다.
            max_features : 단어 수 상한. None이면 제한하지 않는다.
                           정수면 전체 문서에서 많이 나온 순서로 그 개수만 남긴다.
        """
        self.fields = fields
        self.min_df = min_df
        self.max_features = max_features

    # ------------------------------------------------------------------ fit
    def fit(self, recipes_df, train_df):
        """
        recipes_df : RAW_recipes.csv를 그대로 읽은 DataFrame (id, ingredients, tags, ...)
        train_df   : 학습용 평점. user_id, recipe_id, rating 컬럼. rating 0은 미리 제거한다.
        """
        # ---- 1. 레시피 텍스트 만들기 ----------------------------------------
        recipes = prepare_recipes(recipes_df, self.fields).reset_index(drop=True)
        self.recipe_ids = recipes["id"].to_numpy()
        # recipe_id → 행렬의 행 번호. 평점 데이터는 recipe_id로, 행렬은 행 번호로 접근한다.
        self.id_to_row = pd.Series(np.arange(len(recipes)), index=recipes["id"])  # recipe_id로 행 번호 찾기용

        # ---- 2. TF-IDF 행렬 만들기 -------------------------------------------
        # TF-IDF(t, d) = TF(t, d) × IDF(t)
        #   TF : 레시피 d에 단어 t가 나온 횟수
        #   IDF: log((1 + 전체 레시피 수) / (1 + t가 나온 레시피 수)) + 1
        #        "salt"처럼 거의 모든 레시피에 있는 재료는 IDF가 작다. 그래서 취향 구분에 덜 쓰인다.
        #        "tahini"처럼 드문 재료는 IDF가 커서 취향을 더 잘 드러낸다.
        #
        # TfidfVectorizer 파라미터 설명
        #   token_pattern=r"\S+" : 공백이 아닌 문자 덩어리를 단어 하나로 본다.
        #       기본값 r"(?u)\b\w\w+\b"를 쓰면 '30-minutes-or-less'가 하이픈에서 쪼개진다.
        #       재료·태그는 이미 공백 없는 토큰으로 만들었으므로 공백으로만 나눈다.
        #   lowercase=True       : 모두 소문자로 바꾼다. 'Salt'와 'salt'를 같은 단어로 본다.
        #   min_df, max_features : __init__ 주석 참고.
        #   norm="l2"            : 각 행(레시피 벡터)의 길이를 1로 맞춘다. (기본값)
        #       길이가 1인 벡터끼리는 cos(a, b) = a·b / (|a||b|) = a·b 이다.
        #       그래서 코사인 유사도를 내적 한 번으로 계산할 수 있다.
        #   smooth_idf=True      : IDF 분모·분자에 1을 더해 0으로 나누는 일을 막는다. (기본값)
        #   sublinear_tf=False   : TF를 log로 줄이지 않는다. (기본값)
        #       재료·태그는 한 레시피에 보통 한 번씩만 나오므로 의미가 거의 없다.
        #   dtype=np.float32     : 기본값 float64 대신 float32로 저장해 메모리를 절반으로 줄인다.
        self.vectorizer = TfidfVectorizer(
            token_pattern=r"\S+",
            lowercase=True,
            min_df=self.min_df,
            max_features=self.max_features,
            norm="l2",
            smooth_idf=True,
            sublinear_tf=False,
            dtype=np.float32,
        )
        # fit_transform(raw_documents): 단어 사전과 IDF를 학습하고, 바로 TF-IDF 행렬을 반환한다.
        #   반환값은 scipy.sparse.csr_matrix, 모양은 (레시피 수, 단어 수).
        # 레시피 텍스트만 쓰고 평점은 쓰지 않는다. 그래서 전체 레시피로 학습해도 테스트 평점이 새지 않는다.
        self.X = self.vectorizer.fit_transform(recipes["text"])

        # ---- 3. 사용자별 평가 기록 정리 -------------------------------------------
        # 레시피 데이터에 없는 recipe_id는 벡터가 없으므로 버린다.
        train = train_df[train_df["recipe_id"].isin(self.id_to_row.index)].copy()
        train["row"] = self.id_to_row.loc[train["recipe_id"]].to_numpy()

        # 사용자별로 "평가한 모든 레시피" (추천에서 제외할 목록)
        self.rated_rows = train.groupby("user_id")["row"].apply(np.array).to_dict()
        # 사용자별로 "4점 이상 준 레시피" (프로필을 만들 때 쓰는 목록)
        liked = train[train["rating"] >= LIKE_THRESHOLD]
        self.liked_rows = liked.groupby("user_id")["row"].apply(np.array).to_dict()

        # ---- 4. 인기순 대체 목록 -------------------------------------------------
        # 학습 데이터에서 평점이 많은 순서. 프로필을 만들 수 없는 사용자에게 쓴다.
        # (팀의 공식 인기순 기준선은 D가 만든다. 이건 추천기 내부의 대체용이다.)
        counts = train["row"].value_counts()  # 많은 순서로 정렬되어 나온다
        self.popular_rows = counts.index.to_numpy()
        return self

    # ------------------------------------------------------------ profile
    def user_profile(self, user_id):
        """
        사용자 프로필 = 4점 이상 준 레시피 TF-IDF 벡터들의 평균. 이 평균을 다시 길이 1로 맞춘다.
        4점 이상 레시피가 없으면 None을 반환한다.

        평균 벡터의 길이는 1보다 작다. 길이를 1로 맞춰야
        X · p 값이 진짜 코사인 유사도(-1~1, 여기선 0~1)가 된다.
        길이를 맞추지 않아도 순위는 같지만, 점수를 해석하기 쉽도록 맞춘다.
        """
        rows = self.liked_rows.get(user_id)
        if rows is None or len(rows) == 0:
            return None
        # X[rows]: 해당 행들만 꺼낸 희소 행렬. .mean(axis=0): 열별 평균 → 1 × 단어 수 (np.matrix)
        profile = np.asarray(self.X[rows].mean(axis=0)).ravel()
        norm = np.linalg.norm(profile)
        return profile / norm if norm > 0 else None

    # ---------------------------------------------------------- recommend
    def recommend(self, user_id, k=10):
        """user_id에게 추천할 recipe_id k개를 점수 높은 순서로 반환한다."""
        rated = self.rated_rows.get(user_id, np.array([], dtype=int))
        profile = self.user_profile(user_id)

        if profile is None:
            # 콜드 스타트: 좋아한 레시피가 없어 취향을 알 수 없다 → 인기순 추천
            return self._popular(rated, k)

        # 모든 레시피와의 코사인 유사도를 한 번에 계산한다: (레시피 수 × 단어 수) · (단어 수,) → (레시피 수,)
        # 레시피 × 레시피 유사도 행렬은 만들지 않는다. 23만² 크기라 메모리에 들어가지 않는다.
        scores = self.X @ profile
        scores[rated] = -np.inf  # 이미 평가한 레시피는 추천하지 않는다

        # np.argpartition(a, kth): 배열 전체를 정렬하지 않는다. kth번째 원소를 기준으로
        # 작은 값과 큰 값을 양쪽으로 나누기만 하므로 O(n)이다. 여기서는 상위 k개를 빠르게 고른다.
        # 23만 개를 전부 정렬(O(n log n))하는 것보다 빠르다. 고른 k개만 따로 정렬한다.
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return self.recipe_ids[top].tolist()

    def _popular(self, rated, k):
        """인기순 목록에서 이미 평가한 레시피를 빼고 앞에서 k개를 반환한다."""
        mask = ~np.isin(self.popular_rows, rated)  # np.isin: rated에 있는 위치가 True
        return self.recipe_ids[self.popular_rows[mask][:k]].tolist()
