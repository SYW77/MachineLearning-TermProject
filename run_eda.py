"""
run_eda.py  (담당: Member A, 이민지)

RAW_interactions.csv의 탐색적 데이터 분석(EDA) 그림과 표를 eda/ 폴더에 저장한다.
그래프의 제목·축 이름은 발표 자료용으로 영어로 쓴다.

결과
    eda/rating_distribution.png     평점 0~5 개수 (0은 전처리에서 지우는 값이라 회색)
    eda/ratings_per_user.png        사용자당 평점 수 히스토그램 (로그 스케일)
    eda/ratings_per_recipe.png      레시피당 평점 수 히스토그램 (로그 스케일)
    eda/ratings_per_year.png        연도별 평점 수
    eda/kcore_summary.csv           k-core k값(없음/2/3/5/10)별 남는 평점·사용자·레시피 수
    eda/min_count_tradeoff.png      최소 평점 수 기준(없음/2/3/5/10)별 남는 사용자 비율과 test 중 새 레시피 비율
    eda/test_users_train_count.png  test 사용자의 train 평점 수 구간(1~2/3~4/5~9/10+)별 사용자 수

실행: python run_eda.py   (데이터 폴더가 다르면: python run_eda.py <데이터폴더>)
"""

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 화면 없이 PNG 파일로만 저장하는 백엔드
import matplotlib.pyplot as plt
import matplotlib.ticker  # FuncFormatter: 축 눈금 숫자를 원하는 모양(예: 800K)으로 바꾸는 데 쓴다
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import preprocess  # noqa: E402

DATA_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data"
EDA_DIR = ROOT / "eda"
KCORE_KS = [None, 2, 3, 5, 10]

# 그래프 색. 막대는 한 가지 색만 쓰고(계열이 하나뿐), 글자·격자는 회색 계열로 눈에 덜 띄게 한다.
BAR = "#2a78d6"
BAR_2 = "#eb6834"       # 한 그림에 두 번째 지표가 있을 때 쓰는 색 (BAR와 색각이상 구분 검사 통과)
MUTED_BAR = "#b8b7b0"   # 지우는 값(평점 0) 표시용
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

# rcParams: matplotlib 전체 기본 스타일 딕셔너리. 여기서 한 번 바꾸면 모든 그래프에 적용된다.
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "text.color": TEXT,
    "xtick.color": TEXT_2, "ytick.color": TEXT_2,
    "axes.spines.top": False, "axes.spines.right": False,  # 위·오른쪽 테두리 숨김
    "axes.titlesize": 12, "axes.titleweight": "bold", "font.size": 10,
})


def style_axis(ax, grid_axis="y"):
    """
    받는 것 : matplotlib Axes, 격자를 그릴 축("x"/"y")
    하는 일 : 옅은 격자를 데이터 뒤에 깐다.
    돌려주는 것: 없음
    """
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)  # 격자를 막대 뒤로 보낸다


def plot_rating_distribution(inter, path):
    """
    받는 것 : 원본 평점 DataFrame, 저장 경로
    하는 일 : 평점 0~5 개수 막대그래프. 막대 위에 개수와 비율을 쓴다.
    돌려주는 것: 없음
    """
    # value_counts(): 값별 개수, sort_index(): 평점 0→5 순서로 정렬, reindex: 빠진 평점이 있으면 0으로
    counts = inter["rating"].value_counts().sort_index().reindex(range(6), fill_value=0)
    colors = [MUTED_BAR] + [BAR] * 5
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(counts.index, counts.values, color=colors, width=0.7)
    labels = [f"{c:,}\n({c / counts.sum():.1%})" for c in counts.values]
    ax.bar_label(bars, labels=labels, fontsize=8, color=TEXT_2, padding=2)  # 막대 위에 글자 표시
    ax.set_xticks(range(6), ["0\n(review only,\nremoved)", "1", "2", "3", "4", "5"])
    ax.set_xlabel("Rating")
    ax.set_ylabel("Number of interactions")
    ax.set_title(f"Rating distribution (all {counts.sum():,} interactions)")
    ax.set_ylim(0, counts.max() * 1.18)  # 막대 위 글자가 잘리지 않게 여유를 둔다
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e3:,.0f}K"))
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_count_histogram(counts, entity, path):
    """
    받는 것 : 개체(사용자/레시피)별 평점 수 Series, 개체 이름("user"/"recipe"), 저장 경로
    하는 일 : 평점 수 히스토그램을 x·y 모두 로그 스케일로 그린다.
              평점 수는 1개가 대부분이고 수천 개도 있어서(롱테일) 일반 축으로는 모양이 안 보인다.
    돌려주는 것: 없음
    """
    # np.logspace(start, stop, num): 10**start ~ 10**stop 구간을 로그 간격으로 num개 나눈 값.
    #   정수 경계로 바꾼 뒤 np.unique로 중복을 지운다 → 1, 2, 3, ... 처럼 작은 값은 촘촘하게.
    edges = np.unique(np.logspace(0, np.log10(counts.max() + 1), 40).astype(int))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(counts, bins=edges, color=BAR, edgecolor=SURFACE, linewidth=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"Ratings per {entity} (log scale)")
    ax.set_ylabel(f"Number of {entity}s (log scale)")
    ax.set_title(f"Ratings per {entity} (rating > 0, {len(counts):,} {entity}s)")
    one = (counts == 1).mean()
    note = (f"median = {counts.median():.0f}\n"
            f"exactly 1 rating: {one:.1%}\n"
            f"fewer than 5: {(counts < 5).mean():.1%}\n"
            f"max = {counts.max():,}")
    ax.text(0.97, 0.95, note, transform=ax.transAxes, ha="right", va="top", fontsize=9, color=TEXT_2)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ratings_per_year(pos, path):
    """
    받는 것 : rating > 0 DataFrame, 저장 경로
    하는 일 : 연도별 평점 수 막대그래프 (date 문자열 앞 4글자 = 연도)
    돌려주는 것: 없음
    """
    years = pos["date"].str[:4].astype(int).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(years.index, years.values, color=BAR, width=0.75)
    peak = years.idxmax()
    ax.annotate(f"peak {peak}: {years[peak]:,}", (peak, years[peak]), textcoords="offset points",
                xytext=(0, 4), ha="center", fontsize=8, color=TEXT_2)
    ax.set_ylim(0, years.max() * 1.12)  # 최고점 글자가 제목과 겹치지 않게 여유를 둔다
    ax.set_xticks(years.index, [str(y) if y % 2 == 0 else "" for y in years.index])
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of ratings")
    ax.set_title("Ratings per year (rating > 0)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e3:,.0f}K"))
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def kcore_table(pos, ks):
    """
    받는 것 : rating > 0 DataFrame, k 목록 (None = 필터 없음)
    하는 일 : k마다 preprocess.kcore_filter를 적용해 남는 평점·사용자·레시피 수와 비율을 센다.
    돌려주는 것: DataFrame (k, rows, users, recipes, *_share, iterations)
    """
    base = preprocess.summarize(pos)
    rows = []
    for k in ks:
        df, n_iter = (pos, 0) if k is None else preprocess.kcore_filter(pos, k)
        s = preprocess.summarize(df)
        rows.append({"k": "none" if k is None else k, **s,
                     **{f"{key}_share": round(s[key] / base[key], 4) for key in s},
                     "iterations": n_iter})
    return pd.DataFrame(rows)


def min_count_split_table(pos, ks):
    """
    받는 것 : rating > 0 DataFrame (orig_order 포함), 최소 평점 수 목록 (None = 필터 없음)
    하는 일 : 기준마다 평점 수가 기준 미만인 사용자·레시피를 반복해서 지운 뒤(preprocess.kcore_filter),
              run_preprocess.py와 같은 설정으로 시간순 분할을 하고
              남는 사용자 비율과 test 정답 중 새 레시피(train에 한 번도 없던 레시피) 비율을 센다.
    돌려주는 것: DataFrame (min_count, users_share, test_rows, new_recipe_share)
    """
    dedup, _ = preprocess.drop_duplicate_pairs(pos)
    n_users = dedup["user_id"].nunique()
    rows = []
    for k in ks:
        df = dedup if k is None else preprocess.kcore_filter(dedup, k)[0]
        train, _, test = preprocess.chronological_split(df)
        # isin: test의 recipe_id가 train에 있으면 True. ~로 뒤집어 "새 레시피"만 고르고 mean()으로 비율을 낸다.
        new_share = (~test["recipe_id"].isin(train["recipe_id"])).mean()
        rows.append({"min_count": "none" if k is None else k,
                     "users_share": round(df["user_id"].nunique() / n_users, 4),
                     "test_rows": len(test),
                     "new_recipe_share": round(new_share, 4)})
    return pd.DataFrame(rows)


def plot_min_count_tradeoff(table, path):
    """
    받는 것 : min_count_split_table 결과, 저장 경로
    하는 일 : 기준별 "남는 사용자 비율"(왼쪽)과 "test 중 새 레시피 비율"(오른쪽)을 나란히 막대그래프로 그린다.
              두 값은 크기 차이가 커서(100% vs 몇 %) 한 축에 같이 그리면 오른쪽 값이 안 보이므로 칸을 나눈다.
    돌려주는 것: 없음
    """
    labels = [str(k) for k in table["min_count"]]
    # subplots(1, 2, sharex=True): 가로로 그래프 2칸. 두 칸이 같은 x축(기준값)을 쓴다.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True)
    panels = [(axes[0], table["users_share"], BAR, "Users kept"),
              (axes[1], table["new_recipe_share"], BAR_2, "Test answers that are new recipes")]
    for ax, values, color, title in panels:
        bars = ax.bar(labels, values * 100, color=color, width=0.7)
        ax.bar_label(bars, labels=[f"{v:.1%}" for v in values], fontsize=8, color=TEXT_2, padding=2)
        ax.set_ylim(0, values.max() * 100 * 1.15)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.set_xlabel("Minimum rating count per user and recipe")
        ax.set_title(title)
        style_axis(ax)
    axes[0].set_ylabel("Share (%)")
    fig.suptitle("Stricter minimum rating count removes users and new recipes", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def count_test_users_by_train_ratings(pos):
    """
    받는 것 : rating > 0 DataFrame (orig_order 포함)
    하는 일 : run_preprocess.py와 같은 설정으로 분할한 뒤, test 사용자마다 train 평점 수를 세서
              1~2 / 3~4 / 5~9 / 10개 이상 구간으로 묶는다.
    돌려주는 것: 구간별 사용자 수 Series (index = 구간 이름)
    """
    dedup, _ = preprocess.drop_duplicate_pairs(pos)
    train, _, test = preprocess.chronological_split(dedup)
    counts = train.groupby("user_id").size().loc[test["user_id"]]  # test 사용자의 train 평점 수
    # pd.cut(x, bins, labels): 숫자를 구간으로 나눈다. bins=[0, 2, 4, 9, inf]면
    #   (0, 2] → "1-2", (2, 4] → "3-4", (4, 9] → "5-9", (9, inf] → "10+" (왼쪽 끝은 미포함, 오른쪽 끝은 포함)
    bins = pd.cut(counts, [0, 2, 4, 9, np.inf], labels=["1-2", "3-4", "5-9", "10+"])
    return bins.value_counts().sort_index()


def plot_test_user_train_counts(bins, path):
    """
    받는 것 : count_test_users_by_train_ratings 결과, 저장 경로
    하는 일 : 구간별 test 사용자 수 막대그래프. 막대 위에 사람 수와 비율을 쓴다.
    돌려주는 것: 없음
    """
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(bins.index.astype(str), bins.values, color=BAR, width=0.7)
    labels = [f"{c:,}\n({c / bins.sum():.1%})" for c in bins.values]
    ax.bar_label(bars, labels=labels, fontsize=8, color=TEXT_2, padding=2)
    ax.set_ylim(0, bins.max() * 1.2)
    ax.set_xlabel("Number of ratings in train")
    ax.set_ylabel("Number of test users")
    ax.set_title(f"Test users by number of train ratings ({bins.sum():,} users)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e3:,.0f}K"))
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    t0 = time.time()
    EDA_DIR.mkdir(exist_ok=True)
    # load_interactions: 4개 컬럼 + orig_order(원본 행 번호). orig_order는 분할(같은 날짜 동점 처리)에 필요하다.
    inter = preprocess.load_interactions(DATA_DIR / "RAW_interactions.csv")
    pos = preprocess.remove_zero_ratings(inter)

    plot_rating_distribution(inter, EDA_DIR / "rating_distribution.png")
    plot_count_histogram(pos["user_id"].value_counts(), "user", EDA_DIR / "ratings_per_user.png")
    plot_count_histogram(pos["recipe_id"].value_counts(), "recipe", EDA_DIR / "ratings_per_recipe.png")
    plot_ratings_per_year(pos, EDA_DIR / "ratings_per_year.png")

    table = kcore_table(pos, KCORE_KS)
    table.to_csv(EDA_DIR / "kcore_summary.csv", index=False, lineterminator="\n")
    print(table.to_string(index=False))

    tradeoff = min_count_split_table(pos, KCORE_KS)
    plot_min_count_tradeoff(tradeoff, EDA_DIR / "min_count_tradeoff.png")
    print("\n[최소 평점 수 기준별 남는 사용자 비율, test 중 새 레시피 비율]")
    print(tradeoff.to_string(index=False))

    bins = count_test_users_by_train_ratings(pos)
    plot_test_user_train_counts(bins, EDA_DIR / "test_users_train_count.png")
    print("\n[test 사용자의 train 평점 수 구간]")
    print(bins.to_string())
    print(f"\n완료 ({time.time() - t0:.1f}s). 결과: {EDA_DIR}")


if __name__ == "__main__":
    main()
