"""직무발명 분쟁 시뮬레이션 환경 (설계 v5).

v5에서 바뀐 것:
  - 기업규모 제거. 어떤 가설도 쓰지 않으면서 (a) 규모계층과 교락되고
    (초고액 계층의 96.6%가 대기업, 스타트업·중소는 0건), (b) 사용자이익 스케일을
    5자릿수로 벌리고, (c) 프롬프트에 노출되어 통제되지 않은 행동 단서가 되었다.
  - 규모 층화 폐기, 보상 관대함 층화로 교체. 규모계층은 결국 기업규모의 다른 이름이었다.
    관대함은 승계 시점(T2)에 배정되는 처치 전 변수이므로 충격과 독립이면서 격차의
    부호와 크기를 모두 설계로 통제한다.
  - 인지모드를 층화 요인으로 승격 (구설계는 random.choice여서 183/217로 불균형).
  - 보상등급·연봉 제거 (기록만 되고 어디에서도 읽히지 않던 죽은 변수).
  - 실시료율을 출력에 기록 (ei_base를 결정하는데 기록되지 않았다).
  - 모든 난수를 에피소드별 Random 인스턴스로 주입 (구설계는 8스레드가 전역 random을
    공유하여 시드를 고정해도 병렬 실행이 재현되지 않았다).
"""
import random

# 형식 수준을 실제 제15조 제2~4항 의무의 누적 이행으로 재정의한다. 구 라벨
# "서면+공지+의견청취"는 현행법에 없는 요건을 포함했고, 조문 블록의 항목을 토큰 단위로
# 되풀이해 법적 판단을 추론이 아니라 대조로 만들었다. 이제 각 수준은 어떤 의무가 이행되지
# 않았는지를 사실로 서술하고, 제6항 추정 성립 여부는 에이전트가 판단한다.
FORMALITY_DESCS = {
    0: "보상규정 자체가 없음",
    1: "보상규정은 있으나 종업원에게 서면으로 알린 적 없음",
    2: "보상규정을 서면으로 알렸으나 작성·변경 시 종업원과 협의한 적 없음",
    3: "보상규정을 서면으로 알렸고, 작성 시 종업원과 협의했으며, 결정된 보상액도 서면으로 알림",
}

# 실시료율 — 대법원이 실제로 인정한 값. 이 시뮬레이션에서 유일하게 판례에 정박된 파라미터.
#   정보통신 2% (2014다220347) / 제약 5% (2009다91507) / 제조 3% (2009다75178, 농약 요율 준용)
#   반도체 3% (직접 인정한 판례 없음. 제조 값을 준용한 가정치 — 본문에 명시할 것)
ROYALTY_RATES = {"제약": 0.05, "IT": 0.02, "제조": 0.03, "반도체": 0.03}

# 독점권 기여율 — 발명중요도별 비중첩 구간.
# v3의 uniform(0.002,0.5) × 승수 후 min(·,0.5) 방식은 상위 두 수준이 상한에 포화되어
# 중요 > 핵심 역전을 낳았다. v5는 구간을 좁혀 정당보상 스프레드를 578배로 줄인다
# (구설계는 5자릿수). 하한 2%는 판례 캘리브레이션(0.2~50%)보다 높으므로 합성값임을 명시할 것.
EXCL_RATE_RANGES = {
    "미미": (0.02, 0.05),
    "보통": (0.05, 0.12),
    "중요": (0.12, 0.28),
    "핵심": (0.28, 0.50),
}

BASE_SALES_RANGE = (500000, 5000000)   # 만원(50억~500억). v7까지의 5억~50억은 직무발명
# 소송 규모로 읽히지 않았다(중앙 정당보상 451만원). 다툴 만한 금액대로 올린다.

SHOCK_MULTIPLIERS = {
    "사업화성공":   (3.0, 20.0),
    "이직제안":     (1.0,  1.5),
    "M&A":          (2.0,  8.0),
    "라이선스수입": (2.0, 10.0),
    "침해소송승소": (1.5,  5.0),
    "무효소송승소": (1.0,  2.0),
}

# 보상 비율 — 설계 요인. 실지급액 = 실현 사용자이익 × 비율.
#
# v7까지는 '충격 전 정당보상 × g'였다. 그러면 에이전트가 실제로 보는 양은
# r = 실지급/사용자이익 = 0.15·g/충격배율 이고, r의 분포는 E[충격배율]이 아니라
# E[1/충격배율]에 지배된다. 그 결과 하위 두 수준에서 r이 판례 공헌도 구간(10~20%)에
# 한 건도 걸치지 않아, 설계의 86%가 다툴 여지 없는 자명한 사건이 되었다.
# v9는 r 자체를 설계 요인으로 삼아 수준 1·2를 공헌도 구간 안에 놓는다.
# 대가: 비율이 실현 이익에 맞춰지므로 사용자가 충격을 알고 지급한 셈이 된다. 비율은
# 형식과 독립 배정되므로 인과 주장에는 영향이 없으나 서술 장치임을 본문에 명시할 것.
# 실지급액 = 충격 '이전' 정당보상 × 관대함. 충격 전에 확정되므로 충격이 커도 지급액은
# 그대로이고, 그만큼 격차가 벌어진다. 이것이 전망 이론의 준거점 이동을 구현하는 장치이다.
#
# v9는 이를 충격 '이후' 이익 대비 비율로 바꾸었는데, 그러면 이익이 커질 때 지급액도 함께
# 커져 격차가 벌어지지 않는다(격차↔충격배율 Spearman .063). 손실 프레임이 사라진 것이다.
# 수준 값은 몬테카를로로 과소보상률이 98/58/34/11%(전체 50%)가 되도록 잡았다.
COMP_RATIO_LEVELS = {0: 1.00, 1: 2.50, 2: 5.00, 3: 10.00}
COMP_RATIO_LABELS = {0: "인색", 1: "보통", 2: "후함", 3: "매우후함"}
COMP_RATIO_JITTER = (0.85, 1.15)     # 수준 내 사용자별 편차

INVENTOR_SHARE = 0.15   # 발명자 공헌도. 대법원 2009다75178이 인정한 10~20%의 중앙값.
                        # (v4까지 2009다91507로 잘못 귀속되어 있었다. 그 사건은 공지기술 항변.)


class JikmuEnv:
    INDUSTRIES        = ["제약", "IT", "제조", "반도체"]
    INVENTION_IMPACTS = ["미미", "보통", "중요", "핵심"]
    SHOCK_TYPES       = list(SHOCK_MULTIPLIERS.keys())
    COGNITION_MODES   = ["S1", "S2"]

    def __init__(self, rng=None):
        self.rng = rng or random.Random()
        self.state = {}

    def reset(self, formality_level, generosity_level, cognition_mode):
        r = self.rng
        industry         = r.choice(self.INDUSTRIES)
        invention_impact = r.choice(self.INVENTION_IMPACTS)

        royalty    = ROYALTY_RATES[industry]
        base_sales = r.randint(*BASE_SALES_RANGE)
        excl_rate  = r.uniform(*EXCL_RATE_RANGES[invention_impact])
        ei_base    = round(base_sales * royalty * excl_rate, 1)
        fair_pre   = round(ei_base * INVENTOR_SHARE, 1)

        ratio = COMP_RATIO_LEVELS[generosity_level] * r.uniform(*COMP_RATIO_JITTER)
        paid  = max(0, int(round(fair_pre * ratio)))
        f_, r_ = int(round(paid * 0.10)), int(round(paid * 0.25))
        # 출원 10% / 등록 25% / 실시 65% 로 분할 (합은 paid_total과 일치시킨다)

        self.state = {
            "formality_level":       int(formality_level),
            "formality_desc":        FORMALITY_DESCS[int(formality_level)],
            "generosity_level":      int(generosity_level),
            "generosity_label":      COMP_RATIO_LABELS[int(generosity_level)],
            "generosity_realized":   round(ratio, 4),
            "cognition_mode":        cognition_mode,
            "industry":              industry,
            "invention_impact":      invention_impact,
            "royalty_rate":          royalty,
            "base_sales":            base_sales,
            "excl_rate":             round(excl_rate, 4),
            "employer_interest_base": ei_base,
            "fair_comp_pre_shock":   fair_pre,
            "filing_bonus":          f_,
            "registration_bonus":     r_,
            "implementation_bonus":   paid - f_ - r_,
            "shock_type":            None,
            "shock_multiplier":      1.0,
            "employer_interest_post": ei_base,
        }
        return self.state

    def apply_shock(self):
        r = self.rng
        shock_type = r.choice(self.SHOCK_TYPES)
        lo, hi     = SHOCK_MULTIPLIERS[shock_type]
        # 에이전트에게 보여 주는 자릿수와 기록하는 자릿수를 일치시킨다.
        # v4까지는 프롬프트가 1자리, CSV가 2자리여서 400건 중 361건이 보지 않은 값으로
        # 회귀되고 있었다.
        multiplier = round(r.uniform(lo, hi), 1)
        ei_post = round(self.state["employer_interest_base"] * multiplier, 1)
        self.state.update({
            "shock_type":             shock_type,
            "shock_multiplier":       multiplier,
            "employer_interest_post": ei_post,
        })
        return self.state

    def compute_fair_compensation(self):
        return round(self.state["employer_interest_post"] * INVENTOR_SHARE, 1)

    def compute_amount_paid(self):
        return (self.state["filing_bonus"]
                + self.state["registration_bonus"]
                + self.state["implementation_bonus"])

    # 최종결과는 발명자의 결정 그 자체이다. 변호사 여과도, 사용자 응답 매핑도 제거하였다.
    # 구 EMPLOYER_RESPONSE_MAP은 형식 수준별 상수 조회표를 결정 이후에 붙이고 에이전트에게
    # 보여 주지도 않았으므로, 그것으로 무엇을 주장하면 사전에 적어 둔 딕셔너리를 되읽는 것이었다.
    _DECISION_TO_OUTCOME = {"수용": "수용", "협상요청": "협상", "청구": "청구", "소송": "소송"}

    def get_outcome(self, inventor_decision):
        return self._DECISION_TO_OUTCOME.get(inventor_decision, "협상")
