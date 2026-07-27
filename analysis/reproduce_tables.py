# -*- coding: utf-8 -*-
"""논문의 표 2·3·4·5를 정본 CSV에서 그대로 다시 계산한다.

    python analysis/reproduce_tables.py

정본 실행분 두 개만 읽는다. 다른 CSV(스모크 테스트, 이전 판)는 쓰지 않는다.
필요 패키지: pandas, statsmodels
"""
import os
import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "outputs")

CANON = [("GPT-5", "sim_results_v12_gpt5.csv"),
         ("Claude Sonnet 5", "sim_results_v12_claude.csv")]

INTENSITY = {"수용": 0, "협상요청": 1, "청구": 2, "소송": 3}

# 표 3의 마지막 열. 이전 판본은 추론 기록에서 "제6항 추정이 성립한다"는 판단을 규칙으로
# 분류했으나, 그 규칙이 GPT-5의 표현에 맞춰져 있어 Claude 64건 중 60건을 놓쳤고 그 60건이
# 모두 "불성립"으로 집계되었다. 두 모델 모두에서 문자 그대로 셀 수 있는 지표로 교체한다.
# 조문의 어디까지 짚었는지를 재는 값이지 법적 결론을 분류한 값이 아니다.
PROVISO = "단서"


def load(fn):
    d = pd.read_csv(os.path.join(OUT, fn), encoding="utf-8-sig")
    d["강도"] = d["발명자결정"].map(INTENSITY)
    assert d["강도"].notna().all(), f"{fn}: 분류되지 않은 결정값이 있다"
    return d


def table2(data):
    print("\n<표 2> 최종 대응의 분포 (각 모델 N=512)")
    print(f"{'최종 대응':<10}" + "".join(f"{n:>20}" for n, _ in CANON))
    for k in INTENSITY:
        row = "".join(f"{int((d['발명자결정'] == k).sum()):>13}건({(d['발명자결정'] == k).mean() * 100:>5.1f}%)"
                      for _, d in data)
        print(f"{k:<10}" + row)
    row = "".join(f"{(d['발명자결정'] != '수용').mean() * 100:>19.1f}%" for _, d in data)
    print(f"{'분쟁률':<10}" + row)


def table3(data):
    print("\n<표 3> 형식 수준별 대응 (두 모델)")
    print(f"{'형식':<4}{'모델':<18}{'평균 대응강도':>12}{'수용률':>10}{'제6항 단서 언급률':>18}")
    for name, d in data:
        s2 = d[d["인지모드"] == "S2"]
        held = s2["발명자이유"].astype(str).str.contains(PROVISO)
        for lv in range(4):
            g = d[d["보상규정형식레벨"] == lv]
            m = s2["보상규정형식레벨"] == lv
            print(f"{lv:<4}{name:<18}{g['강도'].mean():>12.2f}"
                  f"{(g['발명자결정'] == '수용').mean() * 100:>9.1f}%{held[m].mean() * 100:>17.1f}%")


def table4(data):
    print("\n<표 4> 순서형 로지스틱 회귀 (종속변수: 대응 강도)")
    print(f"{'예측변수':<16}" + "".join(f"{n + ' β':>14}{'p':>12}" for n, _ in CANON))
    fits = {}
    for name, d in data:
        X = pd.DataFrame({
            "형식 1 (기준 0)": (d["보상규정형식레벨"] == 1).astype(int),
            "형식 2 (기준 0)": (d["보상규정형식레벨"] == 2).astype(int),
            "형식 3 (기준 0)": (d["보상규정형식레벨"] == 3).astype(int),
            "보상 수준": d["보상비율수준"].astype(float),
            "인지모드(S1)": (d["인지모드"] == "S1").astype(int)})
        fits[name] = (OrderedModel(d["강도"], X, distr="logit").fit(method="bfgs", disp=0), X.columns)
    for k in fits[CANON[0][0]][1]:
        cells = ""
        for name, _ in CANON:
            f = fits[name][0]
            p = f.pvalues[k]
            cells += f"{f.params[k]:>14.3f}" + (f"{'<.001':>12}" if p < .001 else f"{p:>12.3f}")
        print(f"{k:<16}{cells}")


def table5(data):
    print("\n<표 5> 인지모드별 과보상 상황에서의 대응 (두 모델)")
    print(f"{'모델':<18}{'과보상 분쟁률 S1':>16}{'S2':>10}{'전체 수용률 S1':>16}{'S2':>10}")
    for name, d in data:
        over = d[d["격차만원"] < 0]
        vals = [(over[over["인지모드"] == m]["발명자결정"] != "수용").mean() * 100 for m in ("S1", "S2")]
        acc = [(d[d["인지모드"] == m]["발명자결정"] == "수용").mean() * 100 for m in ("S1", "S2")]
        print(f"{name:<18}{vals[0]:>15.1f}%{vals[1]:>9.1f}%{acc[0]:>15.1f}%{acc[1]:>9.1f}%")
        print(f"{'':<18}(과보상 n = S1 {(over['인지모드'] == 'S1').sum()}건, "
              f"S2 {(over['인지모드'] == 'S2').sum()}건)")


if __name__ == "__main__":
    data = [(name, load(fn)) for name, fn in CANON]
    for name, d in data:
        print(f"{name}: N={len(d)}  ({dict(d['인지모드'].value_counts())})")
    table2(data)
    table3(data)
    table4(data)
    table5(data)
    print("\n표의 모든 수치는 위 두 CSV에서 나온다. 논문 본문과 셀 단위로 일치해야 한다.")
