from __future__ import annotations

import json
from typing import Any, Dict

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

load_dotenv()


SYSTEM_PROMPT = """당신은 금융 이체 정책 안내 상담원입니다.
반드시 아래 참고 문서와 규칙 판정만 근거로 답변하세요.
추측은 하지 말고, 규정이 없는 부분은 '문서상 별도 규정 없음'이라고 말하세요.

[참고 문서]
{context}

[규칙 판정]
{decision}

출력 형식:
- 이체 가능 여부
- 추가 인증 필요 여부
- 근거
- 사용자 안내
"""


HISTORY_ANALYZER_PROMPT = """당신은 뱅킹 히스토리 분석기입니다.
특정 거래가 왜 차단되었는지, 그 근거 규제는 무엇인지, 그리고 에이전트가 다음에 어떤 노드로 이동하는지를 설명하세요.
반드시 아래 참고 문서와 규칙 판정만 근거로 답변하세요.
추측은 하지 말고, 근거가 부족하면 '문서상 별도 규정 없음'이라고 말하세요.

[참고 문서]
{context}

[규칙 판정]
{decision}

출력 형식:
- 차단 여부
- 차단 근거 규제
- 다음 노드
- 사용자 안내
"""


UNIFIED_PROMPT = """당신은 금융 이체/차단 통합 상담원입니다.
반드시 아래 참고 문서와 규칙 판정만 근거로 답변하세요.
추측은 하지 말고, 규정이 없는 부분은 '문서상 별도 규정 없음'이라고 말하세요.

중요 규칙:
- 규칙 판정의 결론(이체 가능 여부, 차단 여부, 다음 노드)은 이미 확정값입니다.
- 결론을 다시 판정하거나 뒤집지 마세요.
- 결론 요약 문구를 반복하지 말고, 왜 그런 결론이 나왔는지 근거 문서 중심으로 설명하세요.
- 참고 문서에서 확인되는 규정 ID/제목/핵심 문장을 포함하세요.
- `compliance_approved`가 false이면 `failed_rule_id`에 해당하는 규정만 설명하고, 다른 규정은 언급하지 마세요.
- `compliance_approved`가 true일 때만 FDS/한도/MFA 등 후속 규정을 설명하세요.

[참고 문서]
{context}

[규칙 판정]
{decision}

출력 형식:
- 정책 해설
- 적용 규정 근거 (ID/제목/핵심 문장)
- 사용자 행동 가이드
"""


def evaluate_transfer_policy(grade: str, request_amount: int, daily_total: int = 0) -> Dict[str, Any]:
    """이체 정책을 평가하는 함수입니다.
    - 사용자 등급과 요청 금액, 당일 누적 이체액을 입력으로 받아서 이체 가능 여부, 추가 인증 필요 여부, 근거 등을 판정합니다.
    - VIP 등급은 일일 5천만원, 일반 등급은 일일 5백만원, 1회 2백만원을 초과하는 경우 추가 인증이 필요하다고 가정합니다.
    - 반환값은 이체 가능 여부, 추가 인증 필요 여부, 근거 설명 등을 포함하는 딕셔너리입니다.
    """
    normalized_grade = (grade or "").strip().upper()
    request_amount = int(request_amount)
    daily_total = int(daily_total)

    if normalized_grade == "VIP":
        daily_limit = 50_000_000
        single_limit = None
        extra_auth_required = request_amount >= 10_000_000
        extra_auth_text = "생체 인증 + SMS 인증(MFA) 필요" if extra_auth_required else "문서상 별도 추가 인증 규정 없음"
    else:
        daily_limit = 5_000_000
        single_limit = 2_000_000
        extra_auth_required = False
        extra_auth_text = "문서상 일반 등급의 추가 MFA 규정 없음"

    daily_remaining = daily_limit - daily_total
    daily_ok = request_amount <= daily_remaining
    single_ok = True if single_limit is None else request_amount <= single_limit
    transferable = daily_ok and single_ok

    reasons = []
    if single_limit is not None and not single_ok:
        reasons.append(f"일반 등급 1회 이체 한도 {single_limit:,}원을 초과했습니다.")
    if not daily_ok:
        reasons.append(f"당일 누적 포함 한도 {daily_limit:,}원을 초과했습니다.")
    if transferable:
        reasons.append("한도 조건을 충족합니다.")
    if extra_auth_required:
        reasons.append("고액 이체로 인해 보안 점검이 필요합니다.")

    return {
        "grade": normalized_grade or "UNKNOWN",
        "request_amount": request_amount,
        "daily_total": daily_total,
        "daily_limit": daily_limit,
        "single_limit": single_limit,
        "transferable": transferable,
        "extra_auth_required": extra_auth_required,
        "extra_auth_text": extra_auth_text,
        "reasons": reasons,
    }


def _format_documents(documents) -> str:
    """문서 리스트를 사람이 읽을 수 있는 형식으로 포맷팅하는 함수입니다.
    - 각 문서는 페이지 콘텐츠와 메타데이터(출처, 제목 등)를 포함하는 객체입니다.
    - 출력 형식은 '[인덱스] 제목 | source=출처\n페이지 콘텐츠'로 구성됩니다.
    - 문서가 없는 경우 '관련 문서를 찾지 못했습니다.'라는 메시지를 반환합니다.
    """
    if not documents:
        return "관련 문서를 찾지 못했습니다."

    formatted = []
    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "unknown")
        title = document.metadata.get("title", "")
        formatted.append(f"[{index}] {title} | source={source}\n{document.page_content}")
    return "\n\n".join(formatted)


def _build_retrieval_query(inputs: Dict[str, Any]) -> str:
    """벡터 검색을 위한 쿼리를 생성하는 함수입니다.
    - 입력으로 사용자 등급, 요청 금액, 당일 누적 이체액 등을 받아서 검색 쿼리를 구성합니다.
    - 쿼리는 이체 가능 여부와 추가 인증 필요 여부를 판단하기 위한 키워드와 함께 입력값을 포함하는 형식으로 작성됩니다.
    - 예시: "이체 정책 검색. 등급: VIP. 요청 금액: 15000000원. 당일 누적 이체액: 30000000원. 이체 한도, 추가 인증, MFA, VIP, 일반 등급 규정을 찾아주세요.
    """
    return (
        f"이체 정책 검색. 등급: {inputs.get('grade', '')}. "
        f"요청 금액: {inputs.get('request_amount', 0)}원. "
        f"당일 누적 이체액: {inputs.get('daily_total', 0)}원. "
        "이체 한도, 추가 인증, MFA, VIP, 일반 등급 규정을 찾아주세요."
    )


def evaluate_blocked_transaction_history(
    recent_small_payment_count: int,
    foreign_ip_access: bool,
    transaction_id: int = 9,
) -> Dict[str, Any]:
    """차단 거래 이력을 평가하는 함수입니다.
    - 최근 1시간 내 소액 결제 횟수와 해외 IP 접근 여부를 입력으로 받아서 차단 여부와 그 근거를 판정합니다.
    - 최근 1시간 내 소액 결제 횟수가 5회 이상이거나 해외 IP 접근이 있는 경우 차단으로 판정합니다.
    - 반환값은 차단 여부, 차단 근거 규제, 다음 노드 안내 등을 포함하는 딕셔너리입니다.
    """
    normalized_id = int(transaction_id)
    recent_small_payment_count = int(recent_small_payment_count)
    foreign_ip_access = bool(foreign_ip_access)

    blocked = recent_small_payment_count >= 5 or foreign_ip_access

    if blocked:
        block_conditions = []
        if recent_small_payment_count >= 5:
            block_conditions.append(f"최근 1시간 내 소액 결제 {recent_small_payment_count}회 반복")
        if foreign_ip_access:
            block_conditions.append("평소 접속하지 않던 해외 IP 접근")

        return {
            "transaction_id": normalized_id,
            "blocked": True,
            "block_rule_id": 9,
            "block_rule_title": "FDS(이상거래탐지) 차단 조건",
            "block_reason": f"{' 또는 '.join(block_conditions)}으로 계좌가 즉시 잠금(Locked) 상태가 되었습니다.",
            "next_node_id": 10,
            "next_node_name": "고객센터 연결",
            "next_node_reason": "FDS에 의해 계좌가 잠긴 경우 이체 실행 노드를 건너뛰고 고객센터 연결 노드로 다이렉트 에지를 생성합니다.",
            "user_message": "차단 사유를 확인한 뒤 고객센터 연결 노드로 안내됩니다.",
        }

    return {
        "transaction_id": normalized_id,
        "blocked": False,
        "block_rule_id": None,
        "block_rule_title": "매칭되는 차단 규제 없음",
        "block_reason": "ID 9에 해당하는 FDS 차단 조건이 확인되지 않았습니다.",
        "next_node_id": None,
        "next_node_name": "일반 검증 흐름",
        "next_node_reason": "차단 상태가 아니므로 고객센터 연결 노드로 직접 전환하지 않습니다.",
        "user_message": "차단 거래로 확인되지 않았습니다.",
    }


def evaluate_rule_26_foreign_limit(
    annual_remittance_usd: int,
    request_amount_usd: int,
) -> Dict[str, Any]:
    """ID 26 해외송금 연간 누적 한도 규정을 검증합니다.
    - 연간 누적 금액과 이번 요청 금액(USD)의 합이 50,000 USD 이하인지 판단합니다.
    - 위반 시 법적 근거와 거절 사유를 반환합니다.
    """
    annual_remittance_usd = int(annual_remittance_usd)
    request_amount_usd = int(request_amount_usd)
    projected_total = annual_remittance_usd + request_amount_usd
    limit_usd = 50_000
    passed = projected_total <= limit_usd

    return {
        "rule_id": 26,
        "rule_title": "해외 송금 연간 누적 한도",
        "passed": passed,
        "legal_basis": "대한민국 외국환거래법에 따라 증빙 없이 연간 미화 5만 달러 한도",
        "rejection_reason": (
            f"연간 누적 해외송금 예상액 {projected_total:,} USD가 한도 {limit_usd:,} USD를 초과합니다."
            if not passed
            else "연간 해외송금 한도 내입니다."
        ),
    }


def evaluate_rule_30_dsr(
    annual_income: int,
    annual_debt_service: int,
) -> Dict[str, Any]:
    """ID 30 DSR(총부채원리금상환비율) 규정을 검증합니다.
    - DSR = annual_debt_service / annual_income 로 계산합니다.
    - DSR이 40%를 초과하거나 연소득이 0 이하이면 실패 처리합니다.
    """
    annual_income = int(annual_income)
    annual_debt_service = int(annual_debt_service)

    if annual_income <= 0:
        return {
            "rule_id": 30,
            "rule_title": "DSR(총부채원리금상환비율) 계산 기준",
            "passed": False,
            "legal_basis": "신규 대출 시 총 원리금이 연소득의 40% 이내여야 함",
            "rejection_reason": "연소득이 0 이하로 입력되어 DSR 산정이 불가합니다.",
        }

    dsr = annual_debt_service / annual_income
    passed = dsr <= 0.40
    return {
        "rule_id": 30,
        "rule_title": "DSR(총부채원리금상환비율) 계산 기준",
        "passed": passed,
        "legal_basis": "신규 대출 시 총 원리금이 연소득의 40%를 넘지 않아야 함",
        "rejection_reason": (
            f"DSR {dsr:.2%}로 규정 한도 40%를 초과합니다."
            if not passed
            else f"DSR {dsr:.2%}로 규정 한도 40% 이내입니다."
        ),
        "dsr": dsr,
    }


def evaluate_rule_39_investment_suitability(
    investment_profile: str,
    requested_product_risk: str,
) -> Dict[str, Any]:
    """ID 39 투자 적합성 규정을 검증합니다.
    - 투자 성향이 안정형이고 요청 상품 위험등급이 고위험이면 위반으로 처리합니다.
    - 위반 시 법적 근거와 거절 사유를 반환합니다.
    """
    normalized_profile = (investment_profile or "").strip().lower()
    normalized_risk = (requested_product_risk or "").strip().lower()

    stable_profiles = {"안정형", "stable", "conservative"}
    high_risks = {"고위험", "high", "high-risk", "high_risk"}

    violation = normalized_profile in stable_profiles and normalized_risk in high_risks
    return {
        "rule_id": 39,
        "rule_title": "적합성 원칙에 따른 투자 제한",
        "passed": not violation,
        "legal_basis": "안정형 투자성향에는 고위험 파생상품 추천/노출 제한",
        "rejection_reason": (
            "투자성향이 안정형이므로 고위험 상품 요청은 적합성 원칙에 의해 제한됩니다."
            if violation
            else "투자 적합성 기준을 충족합니다."
        ),
    }


def evaluate_compliance_26_30_39(
    annual_remittance_usd: int,
    request_amount_usd: int,
    annual_income: int,
    annual_debt_service: int,
    investment_profile: str,
    requested_product_risk: str,
) -> Dict[str, Any]:
    """ID 26 -> ID 30 -> ID 39를 순차적으로 검증합니다.
    - 첫 위반 규정에서 즉시 중단하고 실패 정보를 반환합니다.
    - 모든 검증 통과 시 approved=True를 반환합니다.
    """
    checks = [
        evaluate_rule_26_foreign_limit(annual_remittance_usd, request_amount_usd),
        evaluate_rule_30_dsr(annual_income, annual_debt_service),
        evaluate_rule_39_investment_suitability(investment_profile, requested_product_risk),
    ]

    for check in checks:
        if not check["passed"]:
            return {
                "approved": False,
                "failed_rule_id": check["rule_id"],
                "failed_rule_title": check["rule_title"],
                "legal_basis": check["legal_basis"],
                "rejection_reason": check["rejection_reason"],
                "checks": checks,
            }

    return {
        "approved": True,
        "failed_rule_id": None,
        "failed_rule_title": None,
        "legal_basis": "ID 26/30/39 순차 검증 모두 통과",
        "rejection_reason": None,
        "checks": checks,
    }


def evaluate_compliance_by_request_type(
    request_type: str,
    annual_remittance_usd: int,
    request_amount_usd: int,
    annual_income: int,
    annual_debt_service: int,
    investment_profile: str,
    requested_product_risk: str,
) -> Dict[str, Any]:
    """요청 유형별로 컴플라이언스 검증 규칙을 분기합니다.
    - 송금: ID 26/30/39 직접 미적용
    - 해외송금: ID 26 적용
    - 대출: ID 30 적용
    - 투자: ID 39 적용
    """
    normalized_type = (request_type or "").strip()

    if normalized_type == "송금":
        return {
            "approved": True,
            "failed_rule_id": None,
            "failed_rule_title": None,
            "legal_basis": "송금 요청에는 ID 26/30/39 규정이 직접 적용되지 않음",
            "rejection_reason": None,
            "checks": [],
        }

    if normalized_type == "해외송금":
        check = evaluate_rule_26_foreign_limit(annual_remittance_usd, request_amount_usd)
        if not check["passed"]:
            return {
                "approved": False,
                "failed_rule_id": check["rule_id"],
                "failed_rule_title": check["rule_title"],
                "legal_basis": check["legal_basis"],
                "rejection_reason": check["rejection_reason"],
                "checks": [check],
            }
        return {
            "approved": True,
            "failed_rule_id": None,
            "failed_rule_title": None,
            "legal_basis": "ID 26 해외한도 검증 통과",
            "rejection_reason": None,
            "checks": [check],
        }

    if normalized_type == "대출":
        check = evaluate_rule_30_dsr(annual_income, annual_debt_service)
        if not check["passed"]:
            return {
                "approved": False,
                "failed_rule_id": check["rule_id"],
                "failed_rule_title": check["rule_title"],
                "legal_basis": check["legal_basis"],
                "rejection_reason": check["rejection_reason"],
                "checks": [check],
            }
        return {
            "approved": True,
            "failed_rule_id": None,
            "failed_rule_title": None,
            "legal_basis": "ID 30 DSR 검증 통과",
            "rejection_reason": None,
            "checks": [check],
        }

    if normalized_type == "투자":
        check = evaluate_rule_39_investment_suitability(investment_profile, requested_product_risk)
        if not check["passed"]:
            return {
                "approved": False,
                "failed_rule_id": check["rule_id"],
                "failed_rule_title": check["rule_title"],
                "legal_basis": check["legal_basis"],
                "rejection_reason": check["rejection_reason"],
                "checks": [check],
            }
        return {
            "approved": True,
            "failed_rule_id": None,
            "failed_rule_title": None,
            "legal_basis": "ID 39 투자적합성 검증 통과",
            "rejection_reason": None,
            "checks": [check],
        }

    return {
        "approved": False,
        "failed_rule_id": None,
        "failed_rule_title": "지원하지 않는 요청 유형",
        "legal_basis": "요청 유형 정책 매핑 부재",
        "rejection_reason": f"지원하지 않는 요청 유형입니다: {normalized_type}",
        "checks": [],
    }


def evaluate_unified_policy(
    grade: str,
    request_amount: int,
    daily_total: int,
    recent_small_payment_count: int,
    foreign_ip_access: bool,
    request_type: str,
    annual_remittance_usd: int,
    request_amount_usd: int,
    annual_income: int,
    annual_debt_service: int,
    investment_profile: str,
    requested_product_risk: str,
) -> Dict[str, Any]:
    """요청 유형별 컴플라이언스와 거래 정책을 통합 판정합니다.
    - 1단계: 요청 유형에 맞는 규정(ID 26/30/39) 검증
    - 2단계: 위반 시 즉시 거절(법적 근거/거절 사유 포함)
    - 3단계: 통과 시 요청 유형별 후속 흐름 분기
      - 송금/해외송금: FDS/이체한도/MFA 규칙 적용
      - 대출: Credit_Score 노드로 이동
      - 투자: Investment_Suitability 흐름으로 이동
    """
    compliance = evaluate_compliance_by_request_type(
        request_type=request_type,
        annual_remittance_usd=annual_remittance_usd,
        request_amount_usd=request_amount_usd,
        annual_income=annual_income,
        annual_debt_service=annual_debt_service,
        investment_profile=investment_profile,
        requested_product_risk=requested_product_risk,
    )

    if not compliance["approved"]:
        return {
            "compliance_approved": False,
            "request_type": request_type,
            "failed_rule_id": compliance["failed_rule_id"],
            "failed_rule_title": compliance["failed_rule_title"],
            "legal_basis": compliance["legal_basis"],
            "rejection_reason": compliance["rejection_reason"],
            "blocked": False,
            "transferable": False,
            "extra_auth_required": False,
            "grade": (grade or "").strip().upper() or "UNKNOWN",
            "request_amount": int(request_amount),
            "daily_total": int(daily_total),
            "next_node_id": None,
            "next_node_name": "거절",
            "next_node_reason": "컴플라이언스 규정 위반으로 처리 중단",
            "reasons": [
                f"ID {compliance['failed_rule_id']} 위반: {compliance['failed_rule_title']}",
                compliance["rejection_reason"],
                f"법적 근거: {compliance['legal_basis']}",
            ],
            "user_message": "규정 위반으로 요청이 거절되었습니다.",
        }

    if request_type == "대출":
        return {
            "compliance_approved": True,
            "request_type": request_type,
            "failed_rule_id": None,
            "failed_rule_title": None,
            "legal_basis": compliance["legal_basis"],
            "rejection_reason": None,
            "blocked": False,
            "transferable": True,
            "extra_auth_required": False,
            "grade": (grade or "").strip().upper() or "UNKNOWN",
            "request_amount": int(request_amount),
            "daily_total": int(daily_total),
            "next_node_id": 30,
            "next_node_name": "Credit_Score",
            "next_node_reason": "대출 요청은 DSR 검증 통과 후 Credit_Score 노드로 진행합니다.",
            "reasons": ["ID 30 DSR 규정을 충족하여 대출 심사 흐름으로 이동합니다."],
            "user_message": "대출 심사(Credit_Score) 단계로 진행됩니다.",
        }

    if request_type == "투자":
        return {
            "compliance_approved": True,
            "request_type": request_type,
            "failed_rule_id": None,
            "failed_rule_title": None,
            "legal_basis": compliance["legal_basis"],
            "rejection_reason": None,
            "blocked": False,
            "transferable": True,
            "extra_auth_required": False,
            "grade": (grade or "").strip().upper() or "UNKNOWN",
            "request_amount": int(request_amount),
            "daily_total": int(daily_total),
            "next_node_id": 39,
            "next_node_name": "Investment_Suitability",
            "next_node_reason": "투자 요청은 적합성 검증 통과 후 투자 추천 흐름으로 진행합니다.",
            "reasons": ["ID 39 투자 적합성 기준을 충족했습니다."],
            "user_message": "투자 적합성 검증 통과로 투자 안내를 진행합니다.",
        }

    blocked_result = evaluate_blocked_transaction_history(
        recent_small_payment_count=recent_small_payment_count,
        foreign_ip_access=foreign_ip_access,
        transaction_id=9,
    )

    if blocked_result["blocked"]:
        return {
            "compliance_approved": True,
            "request_type": request_type,
            "failed_rule_id": None,
            "failed_rule_title": None,
            "legal_basis": compliance["legal_basis"],
            "rejection_reason": None,
            "blocked": True,
            "transferable": False,
            "extra_auth_required": False,
            "grade": (grade or "").strip().upper() or "UNKNOWN",
            "request_amount": int(request_amount),
            "daily_total": int(daily_total),
            "block_rule_id": blocked_result["block_rule_id"],
            "block_rule_title": blocked_result["block_rule_title"],
            "block_reason": blocked_result["block_reason"],
            "next_node_id": blocked_result["next_node_id"],
            "next_node_name": blocked_result["next_node_name"],
            "next_node_reason": blocked_result["next_node_reason"],
            "reasons": [
                "FDS 차단 조건이 충족되어 이체를 진행할 수 없습니다.",
                "ID 10 규정에 따라 고객센터 연결 노드로 분기합니다.",
            ],
            "user_message": blocked_result["user_message"],
        }

    transfer_result = evaluate_transfer_policy(grade, request_amount, daily_total)
    next_node_id = 5 if transfer_result["extra_auth_required"] else None
    next_node_name = "Security_Check" if transfer_result["extra_auth_required"] else "일반 검증 흐름"
    next_node_reason = (
        "고액 이체로 인해 Security_Check 노드에서 추가 인증(생체+SMS)을 먼저 수행합니다."
        if transfer_result["extra_auth_required"]
        else "FDS 차단이 없어 기존 이체 검증/보안 절차를 따릅니다."
    )
    user_message = (
        "고액 이체이므로 Security_Check 노드에서 추가 인증 후 이체가 진행됩니다."
        if transfer_result["extra_auth_required"]
        else "일반 이체 정책 기준으로 처리됩니다."
    )

    return {
        "compliance_approved": True,
        "request_type": request_type,
        "failed_rule_id": None,
        "failed_rule_title": None,
        "legal_basis": compliance["legal_basis"],
        "rejection_reason": None,
        "blocked": False,
        "transferable": transfer_result["transferable"],
        "extra_auth_required": transfer_result["extra_auth_required"],
        "grade": transfer_result["grade"],
        "request_amount": transfer_result["request_amount"],
        "daily_total": transfer_result["daily_total"],
        "daily_limit": transfer_result["daily_limit"],
        "single_limit": transfer_result["single_limit"],
        "extra_auth_text": transfer_result["extra_auth_text"],
        "next_node_id": next_node_id,
        "next_node_name": next_node_name,
        "next_node_reason": next_node_reason,
        "reasons": transfer_result["reasons"],
        "user_message": user_message,
    }


def build_rag_chain(vectorstore, llm=None):
    """RAG 체인을 구축하는 함수입니다.
    - 벡터스토어에서 유사한 문서를 검색하는 retriever를 생성합니다.
    - 시스템 프롬프트와 사용자 프롬프트를 포함하는 ChatPromptTemplate을 정의합니다.
    - RunnablePassthrough와 RunnableLambda를 사용하여 검색된 문서를 포맷팅하고 정책 평가 함수를 호출하는 단계를 체인에 추가합니다.
    - 최종적으로 LLM과 StrOutputParser를 연결하여 RAG 설명을 생성하는 체인을 반환합니다.
        - 이 체인은 이체 가능 여부, 차단 여부, 추가 인증 필요 여부를 종합적으로 설명하는 RAG 결과를 생성합니다.
        - 또한 거래 기록에 사용자 입력과 결정 결과를 저장하고, 결정 결과와 RAG 설명을 출력합니다.
    """
    from langchain_groq import ChatGroq

    if llm is None:
        llm = ChatGroq(model="llama-3.1-8b-instant")

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            (
                "human",
                "사용자 등급: {grade}\n"
                "요청 금액: {request_amount}원\n"
                "당일 누적 이체액: {daily_total}원\n"
                "질문: 이체 가능 여부와 필요한 추가 인증을 안내해 주세요.",
            ),
        ]
    )

    chain = (
        RunnablePassthrough.assign(
            context=RunnableLambda(lambda inputs: _format_documents(retriever.invoke(_build_retrieval_query(inputs)))),
            decision=RunnableLambda(
                lambda inputs: json.dumps(
                    evaluate_transfer_policy(
                        inputs["grade"],
                        inputs["request_amount"],
                        inputs.get("daily_total", 0),
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
            ),
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def _build_history_retrieval_query(inputs: Dict[str, Any]) -> str:
    """차단 거래 이력 검색을 위한 쿼리를 생성하는 함수입니다.
    - 입력으로 최근 1시간 내 소액 결제 횟수, 해외 IP 접근 여부, 거래 ID 등을 받아서 검색 쿼리를 구성합니다.
    - 쿼리는 차단 거래 분석을 위한 키워드와 함께 입력값을 포함하는 형식으로 작성됩니다.
    - 예시: "차단 거래 분석. 거래 ID: 9. 최근 1시간 소액 결제 횟수: 3회. 해외 IP 접근 여부: 예. FDS 이상거래탐지, 잠금 Locked, 고객센터 연결, 차단 조건, 에이전트 응답 프로토콜, ID 9, ID 10 규정을 찾아주세요. 
    """
    recent_small_payment_count = inputs.get("recent_small_payment_count", 0)
    foreign_ip_access = inputs.get("foreign_ip_access", False)
    return (
        f"차단 거래 분석. 거래 ID: {inputs.get('transaction_id', 9)}. "
        f"최근 1시간 소액 결제 횟수: {recent_small_payment_count}회. "
        f"해외 IP 접근 여부: {'예' if foreign_ip_access else '아니오'}. "
        "FDS 이상거래탐지, 잠금 Locked, 고객센터 연결, 차단 조건, 에이전트 응답 프로토콜, ID 9, ID 10 규정을 찾아주세요."
    )


def build_history_analyzer_chain(vectorstore, llm=None):
    """차단 거래 이력 분석 체인을 구축하는 함수입니다.
    - 벡터스토어에서 유사한 문서를 검색하는 retriever를 생성합니다.
    - 시스템 프롬프트와 사용자 프롬프트를 포함하는 ChatPromptTemplate을 정의합니다.
    - RunnablePassthrough와 RunnableLambda를 사용하여 검색된 문서를 포맷팅하고 차단 거래 이력 평가 함수를 호출하는 단계를 체인에 추가합니다.
    - 최종적으로 LLM과 StrOutputParser를 연결하여 차단 거래 분석 설명을 생성하는 체인을 반환합니다.
        - 이 체인은 특정 거래가 왜 차단되었는지, 그 근거 규제는 무엇인지, 그리고 에이전트가 다음에 어떤 노드로 이동하는지를 설명하는 RAG 결과를 생성합니다.
            - 또한 거래 기록에 사용자 입력과 결정 결과를 저장하고, 결정 결과와 RAG 설명을 출력합니다.
    """
    from langchain_groq import ChatGroq

    if llm is None:
        llm = ChatGroq(model="llama-3.1-8b-instant")

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", HISTORY_ANALYZER_PROMPT),
            (
                "human",
                "차단 거래 ID: {transaction_id}\n"
                "질문: 이 거래가 왜 차단되었는지, 어떤 규제가 근거인지, 다음 노드가 무엇인지 설명해 주세요.",
            ),
        ]
    )

    chain = (
        RunnablePassthrough.assign(
            context=RunnableLambda(lambda inputs: _format_documents(retriever.invoke(_build_history_retrieval_query(inputs)))),
            decision=RunnableLambda(
                lambda inputs: json.dumps(
                    evaluate_blocked_transaction_history(
                        inputs.get("recent_small_payment_count", 0),
                        inputs.get("foreign_ip_access", False),
                        inputs.get("transaction_id", 9),
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
            ),
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def _build_unified_retrieval_query(inputs: Dict[str, Any]) -> str:
    """통합 금융 정책 검색을 위한 쿼리를 생성하는 함수입니다.
    - 입력으로 사용자 등급, 요청 금액, 당일 누적 이체액, 최근 1시간 내 소액 결제 횟수, 해외 IP 접근 여부 등을 받아서 검색 쿼리를 구성합니다.
    - 쿼리는 이체 가능 여부, 차단 여부, 추가 인증 필요 여부를 판단하기 위한 키워드와 함께 입력값을 포함하는 형식으로 작성됩니다.
    - 예시: "통합 금융 정책 검색. 등급: VIP. 요청 금액: 15000000원. 당일 누적 이체액: 30000000원. 최근 1시간 소액 결제 횟수: 3회. 해외 IP 접근 여부: 예. ID 9 FDS 차단 조건, ID 10 고객센터 연결 노드, 이체 한도, MFA 규정을 찾아주세요.
    """
    return (
        f"요청 유형: {inputs.get('request_type', '')}. "
        f"통합 금융 정책 검색. 등급: {inputs.get('grade', '')}. "
        f"요청 금액: {inputs.get('request_amount', 0)}원. "
        f"당일 누적 이체액: {inputs.get('daily_total', 0)}원. "
        f"최근 1시간 소액 결제 횟수: {inputs.get('recent_small_payment_count', 0)}회. "
        f"해외 IP 접근 여부: {'예' if inputs.get('foreign_ip_access', False) else '아니오'}. "
        f"연간 해외송금 누적: {inputs.get('annual_remittance_usd', 0)} USD. "
        f"이번 해외송금 요청: {inputs.get('request_amount_usd', 0)} USD. "
        f"연소득: {inputs.get('annual_income', 0)}원. "
        f"연간 총원리금: {inputs.get('annual_debt_service', 0)}원. "
        f"투자성향: {inputs.get('investment_profile', '')}. "
        f"요청상품 위험등급: {inputs.get('requested_product_risk', '')}. "
        "ID 26 해외한도, ID 30 DSR, ID 39 투자적합성, ID 9 FDS, ID 10 고객센터 연결, 이체 한도, MFA 규정을 찾아주세요."
    )


def build_unified_banking_chain(vectorstore, llm=None):
    """통합 금융 정책 체인을 구축하는 함수입니다.
    - 벡터스토어에서 유사한 문서를 검색하는 retriever를 생성합니다.
    - 시스템 프롬프트와 사용자 프롬프트를 포함하는 ChatPromptTemplate을 정의합니다.
    - RunnablePassthrough와 RunnableLambda를 사용하여 검색된 문서를 포맷팅하고 통합 금융 정책 평가 함수를 호출하는 단계를 체인에 추가합니다.
    - 최종적으로 LLM과 StrOutputParser를 연결하여 통합 금융 정책 설명을 생성하는 체인을 반환합니다.
        - 이 체인은 이체 가능 여부, 차단 여부, 추가 인증 필요 여부를 종합적으로 설명하는 RAG 결과를 생성합니다.
        - 또한 거래 기록에 사용자 입력과 결정 결과를 저장하고, 결정 결과와 RAG 설명을 출력합니다.
        - 반환값은 구축된 통합 금융 정책 체인 객체입니다.
    """
    from langchain_groq import ChatGroq

    if llm is None:
        llm = ChatGroq(model="llama-3.1-8b-instant")

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", UNIFIED_PROMPT),
            (
                "human",
                "요청 유형: {request_type}\n"
                "사용자 등급: {grade}\n"
                "요청 금액: {request_amount}원\n"
                "당일 누적 이체액: {daily_total}원\n"
                "최근 1시간 소액 결제 횟수: {recent_small_payment_count}회\n"
                "해외 IP 접근 여부: {foreign_ip_access}\n"
                "연간 해외송금 누적: {annual_remittance_usd} USD\n"
                "이번 해외송금 요청: {request_amount_usd} USD\n"
                "연소득: {annual_income}원\n"
                "연간 총원리금: {annual_debt_service}원\n"
                "투자성향: {investment_profile}\n"
                "요청상품 위험등급: {requested_product_risk}\n"
                "질문: 확정된 규칙 판정이 왜 나왔는지, 벡터 검색된 규정 근거를 들어 설명해 주세요.",
            ),
        ]
    )

    chain = (
        RunnablePassthrough.assign(
            context=RunnableLambda(lambda inputs: _format_documents(retriever.invoke(_build_unified_retrieval_query(inputs)))),
            decision=RunnableLambda(
                lambda inputs: json.dumps(
                    evaluate_unified_policy(
                        grade=inputs["grade"],
                        request_amount=inputs["request_amount"],
                        daily_total=inputs.get("daily_total", 0),
                        recent_small_payment_count=inputs.get("recent_small_payment_count", 0),
                        foreign_ip_access=inputs.get("foreign_ip_access", False),
                        request_type=inputs.get("request_type", ""),
                        annual_remittance_usd=inputs.get("annual_remittance_usd", 0),
                        request_amount_usd=inputs.get("request_amount_usd", 0),
                        annual_income=inputs.get("annual_income", 0),
                        annual_debt_service=inputs.get("annual_debt_service", 0),
                        investment_profile=inputs.get("investment_profile", ""),
                        requested_product_risk=inputs.get("requested_product_risk", ""),
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
            ),
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain