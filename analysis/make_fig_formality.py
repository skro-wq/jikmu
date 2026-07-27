"""<그림 2> 형식 수준별 평균 대응 강도 (GPT-5와 Claude Sonnet 5).

정본 CSV를 코드에 고정한다. 이전 판본은 그림을 별도 실행분에서 그려 표와 어긋날 위험이
있었으므로, 이 스크립트가 그림과 표 수치를 같은 자리에서 산출하고 대조까지 출력한다.

  python analysis/make_fig_formality.py

출력
  figures/fig2_formality.png
  표 2·표 3 대조표 (표준출력)

선생님 주석 대응
  #30 1.69 라벨이 주황 막대·범례와 겹치던 문제 → 범례를 축 위로 빼고 상단 여백 확보
  #31 '부분 이행' 주석이 회색으로 그래프 안에 묻히던 문제 → 축 아래 진한 잉크의 그림 주로 분리
색상은 dataviz 검증기(6개 검사) 통과 조합이다.
  node scripts/validate_palette.js "#1d63b0,#c8700a" --mode light  → 전 항목 PASS
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SIM = os.path.join(ROOT, "outputs")
os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
OUT = os.path.join(ROOT, "figures", "fig2_formality.png")

# 정본 실행분. 다른 CSV를 집지 않도록 파일명을 고정한다.
CANON = [("GPT-5", "sim_results_v12_gpt5.csv"),
         ("Claude Sonnet 5", "sim_results_v12_claude.csv")]

INTENSITY = {"수용": 0, "협상요청": 1, "청구": 2, "소송": 3}
LEVEL_LABEL = ["0\n규정 없음", "1\n서면통지 없음", "2\n협의 없음", "3\n모두 이행"]

# 한글 글꼴. macOS는 AppleGothic, 그 밖의 환경은 설치된 한글 글꼴로 바꿔 주십시오.
matplotlib.rcParams["font.family"] = os.environ.get("KO_FONT", "AppleGothic")
matplotlib.rcParams["axes.unicode_minus"] = False

PAPER = "#faf8f4"
INK = "#1b2430"
MUTED = "#5d6672"
SERIES = {"GPT-5": "#1d63b0", "Claude Sonnet 5": "#c8700a"}
HATCH = {"GPT-5": "", "Claude Sonnet 5": "///"}   # 흑백 인쇄·색각 대비용 2차 부호화

WIDTH_IN, HEIGHT_IN, DPI = 5.83, 3.15, 300       # 14.8 cm 본문 폭
LABEL_PT, TICK_PT, NOTE_PT = 9.0, 8.5, 8.0


def load():
    out = {}
    for name, fn in CANON:
        d = pd.read_csv(os.path.join(SIM, fn), encoding="utf-8-sig")
        d["강도"] = d["발명자결정"].map(INTENSITY)
        assert d["강도"].notna().all(), f"{fn}: 미분류 결정값"
        out[name] = d
    return out


def presumption_holds(reason):
    """표 3의 마지막 열: S2 추론 기록에 제6항 '단서'가 등장한 비율.

    이전 판본은 '추정이 성립한다'는 판단을 규칙으로 분류했으나, 그 규칙이 GPT-5의 표현에
    맞춰져 있어 Claude의 서술(예: "형식적 정당성은 만족하나 단서가 충족되지 않았다")을
    잡지 못했다. 64건 중 60건이 미분류로 남아 '불성립'으로 집계되었다. 두 모델 모두에서
    문자 그대로 셀 수 있는 지표로 교체한다. 조문의 어디까지 짚었는지를 재는 값이지
    법적 결론을 분류한 값이 아니다."""
    return reason.str.contains("단서")


def tables(data):
    rows = []
    for name, d in data.items():
        s2 = d[d["인지모드"] == "S2"]
        held = presumption_holds(s2["발명자이유"].astype(str))
        for lv in range(4):
            g = d[d["보상규정형식레벨"] == lv]
            m = s2["보상규정형식레벨"] == lv
            rows.append(dict(모델=name, 형식=lv, n=len(g),
                             평균강도=round(g["강도"].mean(), 2),
                             수용률=round((g["발명자결정"] == "수용").mean() * 100, 1),
                             단서언급률=round(held[m].mean() * 100, 1)))
    return pd.DataFrame(rows)


def draw(tab):
    fig, ax = plt.subplots(figsize=(WIDTH_IN, HEIGHT_IN))
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)

    # 부분 이행 구간(0·1·2)을 옅게 깔아 '움직이지 않는 구간'을 눈으로 보이게 한다.
    ax.axvspan(-0.5, 2.5, color="#eae5dc", zorder=0)

    width = 0.38
    for i, (name, colour) in enumerate(SERIES.items()):
        sub = tab[tab["모델"] == name].sort_values("형식")
        xs = [lv + (i - 0.5) * width for lv in range(4)]
        bars = ax.bar(xs, sub["평균강도"], width * 0.92, color=colour, zorder=3,
                      hatch=HATCH[name], edgecolor=PAPER, linewidth=0.8)
        for b, v in zip(bars, sub["평균강도"]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.045, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=TICK_PT, color=INK, zorder=4)

    ax.set_ylim(0, 2.15)                      # 라벨이 범례에 닿지 않도록 상단 여백 확보 (#30)
    ax.set_xlim(-0.5, 3.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels(LEVEL_LABEL, fontsize=TICK_PT, color=INK)
    ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
    ax.tick_params(axis="y", labelsize=TICK_PT, colors=INK, length=0)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("평균 대응 강도", fontsize=LABEL_PT, color=INK, labelpad=6)
    ax.grid(axis="y", color="#d8d2c8", linewidth=0.6, zorder=1)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK)
    ax.spines["bottom"].set_linewidth(0.8)

    handles = [Patch(facecolor=c, hatch=HATCH[n], edgecolor=PAPER, label=n)
               for n, c in SERIES.items()]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.01),
              ncol=2, frameon=False, fontsize=TICK_PT, labelcolor=INK,
              handlelength=1.4, columnspacing=1.6)

    # 주석은 그래프 밖으로 빼고 진하게 (#31)
    fig.text(0.5, 0.005,
             "형식 수준 0·1·2 구간(음영)에서는 두 모델 모두 대응이 움직이지 않고, "
             "완전 이행(3)에서만 낮아진다.",
             ha="center", va="bottom", fontsize=NOTE_PT, color=MUTED)

    fig.subplots_adjust(left=0.115, right=0.995, top=0.90, bottom=0.235)
    fig.savefig(OUT, dpi=DPI, facecolor=PAPER)
    plt.close(fig)


if __name__ == "__main__":
    tab = tables(load())
    draw(tab)
    print(f"saved {OUT}\n")
    print("=== 표 3 대조용 (정본 CSV 산출) ===")
    print(tab.to_string(index=False))
    print("\n=== 표 2 대조용 ===")
    for name, d in load().items():
        vc = d["발명자결정"].value_counts()
        cells = " | ".join(f"{k} {int(vc.get(k, 0))}건({vc.get(k, 0) / len(d) * 100:.1f}%)"
                           for k in INTENSITY)
        print(f"{name}: N={len(d)} | {cells}")
