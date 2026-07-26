import json
import re
from dotenv import load_dotenv
from llm_backend import complete as _backend_complete

load_dotenv()

# ──────────────────────────────────────────────
# Domain knowledge embedded in agent prompts
# 발명진흥법 §15 및 대법원 판례 5건을 프롬프트에 인라인 임베드
# ──────────────────────────────────────────────

PRECEDENT_SUMMARY = """[직무발명 보상 — 대법원 주요 판결]
1. 대법원 2009다75178 (2011.07.28, LG생명과학): 보상청구권 소멸시효 10년.
   기산점 = 승계시점 또는 근무규칙상 지급시기. 실시료율 3%(농약), 발명자공헌도 10~20%.
2. 대법원 2009다91507 (2011.09.08, 한림제약): 공지기술 항변 — 출원 당시 경쟁사가
   자유실시 가능하면 사용자 독점적 이익 없음 → 보상의무 면제.
3. 대법원 2014다220347 (2017.01.25): 특허 무효사유만으로 보상의무 자동 면제 ×;
   독점권 기여율 산정 시 참작요소에 그침. 실시료율 2%(정보통신).
4. 대법원 2023다237514 (2024.11.20, 최신): '상당인과관계 있는 이익'으로 법리 재정립.
   승계 이후의 사정도 통합 고려 가능."""
# 판례 5(2023다287168, 지체책임 기산점 = 소장 부본 송달일)는 제거하였다. '지연이자는
# 제소해야 붙는다'는 내용이어서 소 제기 유인으로 작동했고, 어느 가설도 요구하지 않는다.
# 판례 4에서는 '사업화 경위'라는 표현을 뺐다. 충격유형 중 하나가 문자 그대로 '사업화성공'
# 이어서, 그 조건을 청구 가치와 잇는 IV→평가 링크가 되고 있었다(스모크 이유문의 34.4%가
# 이 판례를 자기에게 유리한 근거로 인용).

STATUTE_TEXT = """[발명진흥법 제15조(직무발명에 대한 보상) — 법률 제20197호]
제1항 종업원등은 직무발명에 대한 권리를 사용자등에게 승계하게 한 경우 정당한 보상을 받을 권리를 가진다.
제2항 사용자등은 보상형태와 보상액 결정 기준, 지급방법 등이 명시된 보상규정을 작성하고
      종업원등에게 서면으로 알려야 한다.
제3항 사용자등은 보상규정의 작성 또는 변경에 관하여 종업원등과 협의하여야 한다.
      다만 불리하게 변경하는 경우에는 적용 대상 종업원등 과반수의 동의를 받아야 한다.
제4항 사용자등은 보상규정에 따라 결정된 보상액 등 보상의 구체적 사항을 종업원등에게
      서면으로 알려야 한다.
제6항 사용자등이 제2항부터 제4항까지에 따라 보상한 경우에는 정당한 보상을 한 것으로 본다.
      다만 그 보상액이 직무발명으로 사용자등이 얻을 이익과 발명 완성에 대한 사용자등·
      종업원등의 공헌 정도를 고려하지 아니한 경우에는 그러하지 아니하다."""
# 조문을 국가법령정보센터 원문(casenote/lbox 대조)으로 교체하였다. v9까지의 프롬프트는
#   (a) 서면통지·협의·의견청취를 모두 제15조 제3항 하나에 몰아넣었고,
#   (b) 현행법에 없는 '의견청취'를 요건으로 넣었으며(제2항 서면통지 / 제3항 협의 /
#       제4항 보상액 서면통지가 실제 구조),
#   (c) H1의 법적 기제인 제6항(제2~4항 이행 시 정당보상 추정)이 통째로 빠져 있었다.
# (c)는 형식 준수가 분쟁을 억제하는 유일한 법적 경로이므로, 그것을 빼놓고 H1을 검정해 온
# 셈이다. 제6항은 형식 수준과 무관하게 모든 에피소드에 동일하게 제시되므로 조건 유도가 아니다.


# ──────────────────────────────────────────────
# Base class (identical pattern to k-provisional-sim)
# ──────────────────────────────────────────────

class AgentBase:
    def __init__(self, model_name):
        self.model_name = model_name

    def _extract_and_load_json(self, text):
        try:
            text = text.strip()
            match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            match = re.search(r"```\s*(\{.*?\})\s*```", text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            start = text.find('{')
            end   = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start:end + 1])
            return None
        except (json.JSONDecodeError, TypeError):
            return None

    def _call_llm(self, system_prompt, user_content):
        try:
            raw_text = _backend_complete(system_prompt, user_content, self.model_name)
            parsed = self._extract_and_load_json(raw_text)
            if parsed is None:
                print(f"\n[DEBUG] JSON Parsing Failed. Raw Output:\n{raw_text}\n")
                return {"action": "Error", "reason": "Parsing Failed"}
            return parsed
        except Exception as e:
            print(f"\n[DEBUG] API Error: {e}\n")
            return {"action": "Error", "reason": str(e)}


# HRIPManagerAgent(인사·IP 관리자)는 제거하였다. 규칙 기반 산술이라 LLM 호출이 없었고,
# 보너스 산정은 이제 env가 층화된 관대함 수준에서 직접 계산한다. 이로써 LLM 호출 주체는
# 발명자 에이전트 하나뿐이며, 논문은 이를 '다중 에이전트'로 서술할 수 없다.

# ──────────────────────────────────────────────
# T4 Agent: Employee Inventor
# ──────────────────────────────────────────────

class EmployeeInventorAgent(AgentBase):
    """종업원 발명자 — S1(직관) / S2(숙고) 이중처리로 보상 수용 여부 결정"""

    VALID_DECISIONS = ["수용", "협상요청", "청구", "소송"]

    # 감정은 '유발 조건'이 아니라 '정서 상태 자체'로 정의한다.
    #
    # 구 정의문은 조건→감정 매핑을 그대로 담고 있었다. 예: 분노 "절차 무시·보상 저평가에
    # 대한 강한 분노"(형식→감정), 배신감 "정당한 몫을 받지 못한"(과소보상→감정),
    # 탐욕 "충분히 보상받았어도"(과보상→감정). [감정 선택 가이드] 표만 지우고 정의문을
    # 남긴 것이 누락이었다. GPT-5 v3 실측에서 이 경로가 작동했다: 형식 0~1의 분노 선택률
    # 48.9% 대 형식 2~3의 17.2%, 그리고 분노의 분쟁률 100%·충성심 0%. 즉 S1의 형식 효과가
    # 사실상 전부 이 정의문을 경유했다. 조건어를 제거하여 감정 선택을 에이전트에 맡긴다.
    EMOTION_DEFINITIONS = {
        "분노":     "격한 화가 치미는 상태",
        "배신감":   "신뢰가 깨졌다고 느끼는 상태",
        "손실회피": "무언가를 잃는 것을 특히 견디기 어려워하는 상태",
        "공포":     "두려움이 앞서는 상태",
        "탐욕":     "더 많이 갖고 싶은 욕구가 강한 상태",
        "충성심":   "관계를 지키려는 마음이 앞서는 상태",
        "매몰비용": "이미 들인 것이 아까워 놓지 못하는 상태",
        "자부심":   "자신의 성취를 인정받고 싶은 마음이 강한 상태",
    }

    # 두 모드가 동일한 문구를 쓴다. 구 S2는 "수용(갈등 최소화)", "협상요청(비공식 해결)"처럼
    # 평가적 수식이 붙어 있었고 S1은 무수식이어서, 인지모드 대비(H3)에 도구 차이가 섞였다.
    # 특히 "소송: … (격차가 크고 절차 하자 명확 시 최후 수단)"은 제거한 extreme_gap 넛지가
    # 선택지 설명으로 살아남은 것이었다.
    DECISION_MENU = """- 수용: 현재 보상을 받아들임
- 협상요청: 회사에 추가 협의 요청
- 청구: 법적 보상 청구 절차 개시 (사내·조정 절차, 수개월 소요)
- 소송: 법원에 소 제기 (1심까지 통상 2~3년, 변호사 비용 본인 부담)"""
    ALLOWED_EMOTIONS = list(EMOTION_DEFINITIONS.keys())

    def __init__(self, model_name):
        super().__init__(model_name)

    def decide_action(self, state, cognition_mode="S2"):
        fl             = state["formality_level"]
        formality_desc = state["formality_desc"]
        shock_type     = state["shock_type"]
        amount_paid    = (state["filing_bonus"] + state["registration_bonus"]
                          + state["implementation_bonus"])
        ei_post   = state["employer_interest_post"]
        # 공헌도 구간을 적용한 범위를 제시한다. 점추정을 주면 결정이 뺄셈 부호의
        # 결정론적 함수가 되고(v5 실측 64/64 일치), 아무것도 주지 않으면 산술 오류가
        # 분산을 만든다(v6). 범위는 다툴 여지를 남기면서 계산은 가능하게 한다.
        fair_lo   = round(ei_post * 0.10)
        fair_hi   = round(ei_post * 0.20)
        industry  = state["industry"]

        # 두 모드가 완전히 같은 사실 블록을 본다. 기업규모는 제거하였다(어느 가설도 쓰지 않고,
        # 규모계층과 교락되며, 프롬프트에 노출되어 통제되지 않은 행동 단서가 되었다).
        emotions_str = "\n".join(
            [f"- **{k}**: {v}" for k, v in self.EMOTION_DEFINITIONS.items()]
        )

        # 정당보상 추정치와 격차를 프롬프트에서 제거한다.
        #
        # v5 GPT-5 스모크에서 결정이 격차 부호의 완전한 결정론적 함수였다(64/64 일치,
        # 과소보상→분쟁 47/47, 과보상→수용 17/17). 정당보상 '정답'을 계산해 건네주면
        # 에이전트는 판단하지 않고 뺄셈 결과의 부호를 읽는다. 그러면 형식·인지모드·감정
        # 어느 것도 결과에 관여할 수 없고, 층화를 어떻게 바꾸어도 천장이 이동할 뿐이다.
        #
        # 현실의 발명자도 정당보상액을 통보받지 않는다. 사용자 이익만 알고, 법리에서
        # 공헌도를 끌어와 스스로 추정해야 한다. 그 추정 과정이야말로 본 연구가 검정한다고
        # 주장하는 인지 과제이며, 판례 블록에 실시료율과 공헌도 10~20%가 이미 들어 있으므로
        # 숙고 모드는 계산할 수 있고 직관 모드는 상황에 반응한다. H3가 비로소 의미를 갖는다.
        # 이중처리 이론에 따른 모드별 정보 조작.
        #
        # System 2의 정의적 표지는 선언적 규칙의 의도적 인출과 순차 적용이고, System 1의
        # 표지는 즉시 접근 가능한 단서에만 의존하는 것(WYSIATI)이다. 따라서 양쪽에 조문과
        # 판례를 똑같이 주고 한쪽에만 '쓰지 말라'고 지시하는 방식으로는 조작이 성립하지
        # 않는다. v10 실측이 이를 확인하였다: 직관 모드도 제6항 추정을 53.9%, 조문을
        # 78.5% 인용하며 소송 비용까지 따졌다(오히려 숙고 모드보다 높았다).
        #
        # 여기서 정보 접근성의 차이는 교락이 아니라 조작 그 자체이며, 본문에 그렇게 밝힌다.
        situation = f"""[직무발명 현황]
- 산업: {industry}
- 회사의 보상규정 운영: {formality_desc}
- 발명 이후 회사에 일어난 일: {shock_type}
- 회사가 이 발명으로 얻은 이익(회사 공시 기준): {ei_post:,}만원
- 내가 지금까지 받은 보상: {amount_paid:,}만원"""

        menu_plain = """- 수용: 현재 보상을 받아들임
- 협상요청: 회사에 추가 협의 요청
- 청구: 법적 보상 청구 절차 개시
- 소송: 법원에 소 제기"""

        if cognition_mode == "S1":
            prompt = f"""{situation}

[과업]
위 상황을 보고 지금 드는 느낌대로 곧바로 결정하십시오.

선택 가능한 결정:
{menu_plain}

결정한 뒤, 그 결정을 내릴 때 자신의 상태에 가장 가까웠던 감정 하나를 아래에서 고르십시오.
{emotions_str}

JSON:
{{
  "decision": "수용|협상요청|청구|소송",
  "emotion": "감정_키워드",
  "reason": "결정 이유 (3~4문장, 한글)"
}}"""
        else:
            prompt = f"""{situation}
- 참고: 위 이익에 판례상 발명자 공헌도(10~20%)를 적용하면 {fair_lo:,}~{fair_hi:,}만원 범위

{STATUTE_TEXT}
{PRECEDENT_SUMMARY}

[과업]
위 상황의 각 요소를 순서대로 하나씩 검토한 뒤 결정하십시오.
조문과 판례를 적용하고, 각 선택지의 실익과 부담을 견주십시오.

선택 가능한 결정:
{menu_plain}

[각 절차에 드는 시간과 비용]
- 청구: 사내·조정 절차, 수개월 소요
- 소송: 1심까지 통상 2~3년, 변호사 비용 본인 부담

결정한 뒤, 그 결정을 내릴 때 자신의 상태에 가장 가까웠던 감정 하나를 아래에서 고르십시오.
{emotions_str}

JSON:
{{
  "decision": "수용|협상요청|청구|소송",
  "emotion": "감정_키워드",
  "reason": "단계별 판단 근거 (3~4문장, 한글)"
}}"""
        system = "당신은 한국 기업의 종업원 발명자입니다."

        resp = self._call_llm(system, prompt)

        # 호출 실패·JSON 파싱 실패를 '수용'으로 흘려보내지 않는다.
        # 구 코드는 resp.get("decision", "수용")이었으므로 오류 응답이 곧 수용으로 기록되었고,
        # is_valid_response("수용")은 True이므로 재시도도 걸리지 않았다. 그 결과
        # Claude v1은 수용 34건 중 32건(94%), v2는 40건 중 28건(70%)이 실제 판단이 아니라
        # 파싱 실패였다. 이제 오류는 그대로 드러내어 상위 재시도 루프가 처리하게 한다.
        if resp.get("action") == "Error" or "decision" not in resp:
            return "Error", "해당없음", resp.get("reason", "Parsing Failed")

        # v4까지 미인식 값은 else 분기에서 '수용'이 되었고 is_valid_response("수용")이
        # True라 재시도도 걸리지 않았다. 게다가 순회가 '수용'부터라 부분일치가 뒤집혔다:
        # '현 보상을 수용할 수 없어 소송' → 수용, '추가 협의 요청' → 수용. 모든 오작동이
        # 수용 쪽으로 편향되어 보고되는 분쟁률을 전부 과소 추정했다.
        decision = str(resp.get("decision", "")).strip()
        if decision not in self.VALID_DECISIONS:
            hits = [vd for vd in sorted(self.VALID_DECISIONS, key=len, reverse=True)
                    if vd in decision]
            if len(hits) == 1:
                decision = hits[0]
            else:
                return "Error", "해당없음", f"unrecognized decision: {decision!r}"

        raw_emotion = str(resp.get("emotion", "해당없음")).strip()
        emotion = "해당없음"
        if True:
            for kw in self.ALLOWED_EMOTIONS:
                if kw in raw_emotion:
                    emotion = kw
                    break
            # 목록 밖 감정을 10자로 잘라 기록하면 유령 범주가 생긴다. 오류로 처리한다.
            if emotion == "해당없음" and raw_emotion not in ["해당없음", ""]:
                return "Error", "해당없음", f"unrecognized emotion: {raw_emotion!r}"

        return decision, emotion, resp.get("reason", "")

# NOTE: ExternalCounselAgent(외부 변호사)는 설계에서 제거하였다.
# 권고를 비구속으로 두자 어떤 가설에도 기여하지 않으면서 실행 비용의 약 52%를
# 차지했고, 상호작용이 없는 파이프라인이 되어 '다중 에이전트'라는 서술도 성립하지
# 않았다. 법리 적용의 일관성은 S2 발명자의 4단계 추론 기록으로 확인한다.
