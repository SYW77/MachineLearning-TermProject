"""
recipe_parsing.py  (담당: Member C)

RAW_recipes.csv의 `ingredients`, `tags` 컬럼을 파이썬 리스트로 파싱하고,
TF-IDF에 넣을 "레시피 텍스트" 한 줄을 만든다.

왜 파싱이 필요한가?
    CSV에는 리스트가 "['salt', 'olive oil']" 같은 *문자열*로 저장되어 있다.
    그대로 쓰면 대괄호, 따옴표, 쉼표까지 단어로 섞이므로 먼저 진짜 리스트로 바꾼다.

왜 여러 단어 재료를 하나로 묶는가?
    "olive oil"을 그대로 두면 TF-IDF가 "olive"와 "oil"을 별개 단어로 센다.
    그러면 "olive oil"과 "vegetable oil"이 "oil" 때문에 비슷해 보인다.
    "olive_oil"처럼 묶으면 재료 하나가 단어(토큰) 하나가 된다.

참고 자료 (코드 복사 없음, 공식 문서만 참고):
    - Python 공식 문서, ast.literal_eval
      https://docs.python.org/3/library/ast.html#ast.literal_eval
"""

import ast  # 수업에서 다루지 않은 모듈 → 아래 parse_list_column 주석에서 설명
import re   # 정규표현식. 설명(description) 문장을 정리하는 데만 쓴다.


def parse_list_column(value):
    """
    "['salt', 'olive oil']" 형태의 문자열을 ['salt', 'olive oil'] 리스트로 바꾼다.

    ast.literal_eval(node_or_string)
        - 문자열을 파이썬 "리터럴"(숫자, 문자열, 리스트, 딕셔너리 등)로만 해석한다.
        - 파라미터 node_or_string: 해석할 문자열.
        - eval()과 달리 함수 호출이나 import 같은 코드는 실행하지 않는다.
          그래서 외부 데이터(CSV)를 파싱할 때 안전하다.
        - 리터럴이 아닌 문자열이 들어오면 ValueError나 SyntaxError가 난다.

    값이 비어 있거나(NaN) 형식이 깨진 행은 빈 리스트로 처리한다.
    이렇게 하면 레시피 한 개 때문에 전체 실행이 멈추지 않는다.
    """
    if not isinstance(value, str):  # NaN(float) 등 문자열이 아닌 값
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []
    # 혹시 리스트가 아닌 값이 나오면 빈 리스트로 처리한다.
    return list(parsed) if isinstance(parsed, (list, tuple)) else []


def to_token(phrase):
    """
    재료나 태그 하나를 공백 없는 토큰 하나로 바꾼다.
        'Olive  Oil ' → 'olive_oil'
        '30-minutes-or-less' → '30-minutes-or-less'  (하이픈은 그대로 둔다)

    1) 소문자로 바꾸고 앞뒤 공백을 지운다.
    2) 사이에 있는 공백 덩어리(\\s+)를 밑줄 하나로 바꾼다.
    """
    return re.sub(r"\s+", "_", phrase.strip().lower())


def clean_description(text):
    """
    자유 문장인 description을 단어 나열로 정리한다. 3주차 '+설명' 실험에서만 쓴다.
        "It's GREAT, really!" → "it great really"

    알파벳·숫자가 아닌 문자는 공백으로 바꾸고, 한 글자 단어는 버린다.
    재료·태그와 달리 문장은 단어 단위로 쪼개야 하므로 밑줄로 묶지 않는다.
    """
    if not isinstance(text, str):
        return ""
    words = re.sub(r"[^a-z0-9]+", " ", text.lower()).split()
    return " ".join(w for w in words if len(w) > 1)


def build_recipe_text(row, fields=("ingredients", "tags")):
    """
    레시피 한 행(row)으로 TF-IDF 입력 문자열 하나를 만든다.

    파라미터
        row    : RAW_recipes의 한 행. ingredients·tags가 이미 리스트로 파싱되어 있어야 한다.
        fields : 사용할 입력 필드. 3주차 튜닝 실험용이다.
                 ("ingredients",)                        → 재료만
                 ("ingredients", "tags")                 → 재료 + 태그 (기본값)
                 ("ingredients", "tags", "description")  → + 설명

    예) ingredients=['olive oil', 'salt'], tags=['easy', '30-minutes-or-less']
        → "olive_oil salt easy 30-minutes-or-less"
    """
    parts = []
    if "ingredients" in fields:
        parts += [to_token(x) for x in row["ingredients"]]
    if "tags" in fields:
        parts += [to_token(x) for x in row["tags"]]
    if "description" in fields:
        parts.append(clean_description(row.get("description")))
    return " ".join(p for p in parts if p)


def prepare_recipes(recipes_df, fields=("ingredients", "tags")):
    """
    RAW_recipes DataFrame 전체에 위 함수들을 적용한다.

    반환: id, name, ingredients(list), tags(list), text 컬럼이 있는 새 DataFrame.
    원본 DataFrame은 바꾸지 않는다. 다른 팀원도 같은 원본을 쓰기 때문이다.
    """
    df = recipes_df.copy()
    df["ingredients"] = df["ingredients"].apply(parse_list_column)
    df["tags"] = df["tags"].apply(parse_list_column)
    # axis=1: 행 단위로 함수를 적용한다. 한 레시피의 여러 컬럼을 같이 봐야 하기 때문이다.
    df["text"] = df.apply(lambda r: build_recipe_text(r, fields), axis=1)
    return df
