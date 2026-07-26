"""실행 결과 진단 — 설계 반복에서 매번 같은 항목을 같은 방식으로 본다.

사용: python diagnose.py outputs/<파일>.csv

판정 기준 (본 실행 진행 여부):
  1. 무결성   파싱 실패·드롭 0, 셀 균형
  2. 분산     전체 분쟁률이 20~80% 안에 있을 것 (천장·바닥이면 어떤 가설도 검정 불가)
  3. 천장해소 보상비율 수준별 분쟁률이 단조롭게 벌어질 것
  4. H1여지   각 보상비율 수준 안에서 형식별 분쟁률이 변할 여지가 있을 것
  5. 감정다양 8종 중 4종 이상 사용
"""
import sys
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, spearmanr

f = sys.argv[1] if len(sys.argv) > 1 else "outputs/sim_results_v5.csv"
d = pd.read_csv(f, encoding="utf-8-sig")
d["분쟁"] = (d["최종결과"] != "수용").astype(int)
d["법적행동"] = d["최종결과"].isin(["청구", "소송"]).astype(int)
n = len(d)
ok = {}

print(f"=== {f}  N={n} ===\n")

bad = d["발명자이유"].astype(str).str.contains("Parsing|unrecognized|Error").sum()
cells = d.groupby(["보상규정형식레벨", "보상비율수준", "인지모드"]).size()
ok["1.무결성"] = bad == 0 and cells.nunique() == 1
print(f"[1] 무결성      파싱실패 {bad}건 | 셀 {len(cells)}개, 크기 {sorted(cells.unique())}"
      f" | 재시도 {int(d['재시도횟수'].sum())}")

rate = d["분쟁"].mean()
ok["2.분산"] = 0.20 <= rate <= 0.80
print(f"[2] 분산        분쟁률 {rate:.1%} | 결과분포 {d['최종결과'].value_counts().to_dict()}")

g = d.groupby("보상비율라벨").agg(과소보상률=("격차만원", lambda s: (s > 0).mean()),
                                분쟁률=("분쟁", "mean"), 법적행동률=("법적행동", "mean"), n=("분쟁", "size"))
g = g.reindex(["인색", "보통", "후함", "매우후함"]).dropna()
rho_g, p_g = spearmanr(d["보상비율수준"], d["분쟁"])
ok["3.천장해소"] = g["분쟁률"].max() - g["분쟁률"].min() >= 0.15
print(f"[3] 천장해소    보상비율 기울기 {g['분쟁률'].max() - g['분쟁률'].min():.2f} "
      f"(Spearman rho={rho_g:.3f}, p={p_g:.4f})")
print(g.round(3).to_string().replace("\n", "\n" + " " * 16))

print(f"\n[4] H1 여지")
sub = []
for lab in g.index:
    s = d[d["보상비율라벨"] == lab]
    r = s.groupby("보상규정형식레벨")["분쟁"].mean()
    sub.append(r.max() - r.min())
    print(f"{' ' * 16}{lab:<5} 형식별 {r.round(2).tolist()}  변동폭 {r.max() - r.min():.2f}")
# 게이트 4는 셀당 4관측의 max-min이어서 표본 잡음이 임계를 넘겼다. 실제로 v5 스모크에서
# 단조 '증가'(H1 반대 방향) 셀을 PASS시켰고 전체 H1은 p=.97이었다. 부호 있는 Spearman으로
# 바꾸고, 이진 분쟁이 아니라 서열 강도(escalation)를 본다. v3에서 이진 DV는 과소보상
# 부표본에서 포화(100/97.8/100/95.2%)였지만 법적행동률은 51.9/37.0/39.2/19.0%로
# 살아 있었다(beta=-0.417, p=.003). 신호는 '분쟁하느냐'가 아니라 '얼마나 세게'에 있다.
ESC = {"수용": 0, "협상": 1, "청구": 2, "소송": 3}
d["강도"] = d["최종결과"].map(ESC)
rho_e, p_e = spearmanr(d["보상규정형식레벨"], d["강도"])
rho_b, p_b = spearmanr(d["보상규정형식레벨"], d["분쟁"])
ok["4.H1여지"] = rho_e < 0
print(f"{' ' * 16}서열강도 Spearman rho={rho_e:+.3f}, p={p_e:.4f}"
      f"  | 이진 rho={rho_b:+.3f}, p={p_b:.4f}")
print(f"{' ' * 16}형식별 평균강도 {d.groupby('보상규정형식레벨')['강도'].mean().round(2).tolist()}"
      f" | 법적행동률 {d.groupby('보상규정형식레벨')['법적행동'].mean().round(3).tolist()}")
ct = pd.crosstab(d["보상규정형식레벨"], d["분쟁"])
if ct.shape[1] == 2:
    c, p, dof, _ = chi2_contingency(ct)
    v = np.sqrt(c / (n * (min(ct.shape) - 1)))
    rho, pr = spearmanr(d["보상규정형식레벨"], d["분쟁"])
    print(f"{' ' * 16}전체 H1: 형식별 {d.groupby('보상규정형식레벨')['분쟁'].mean().round(3).tolist()}")
    print(f"{' ' * 16}         chi2({dof})={c:.2f}, p={p:.4f}, V={v:.3f} | rho={rho:.3f}, p={pr:.4f}")

s1 = d[d["인지모드"] == "S1"]
emo = s1["감정동인"].value_counts()
ok["5.감정다양"] = len(emo) >= 4
ok["6.소송존재"] = (d["최종결과"] == "소송").sum() >= 3
print(f"\n[5] 감정다양    {len(emo)}종 사용 {emo.to_dict()}")

print(f"[6] 소송존재    소송 {(d[chr(34)+chr(52)+chr(34)] if False else (d['최종결과']=='소송').sum())}건")
print(f"\n[참고] H3 인지모드별 분쟁률 {d.groupby('인지모드')['분쟁'].mean().round(3).to_dict()}")
print(f"[참고] 격차↔충격배율 Spearman {spearmanr(d['격차만원'], d['충격배율'])[0]:.3f}")

print("\n" + "=" * 60)
for k, v in ok.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")
print(f"\n판정: {'본 실행 진행 가능' if all(ok.values()) else '보완 필요 — ' + ', '.join(k for k, v in ok.items() if not v)}")
