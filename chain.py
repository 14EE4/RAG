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


def evaluate_unified_policy(
    grade: str,
    request_amount: int,
    daily_total: int,
    recent_small_payment_count: int,
    foreign_ip_access: bool,
) -> Dict[str, Any]:
    """통합 금융 정책을 평가하는 함수입니다.
    - 사용자 등급, 요청 금액, 당일 누적 이체액, 최근 1시간 내 소액 결제 횟수, 해외 IP 접근 여부를 입력으로 받아서 이체 가능 여부, 차단 여부, 추가 인증 필요 여부 등을 종합적으로 판정합니다.
    - 먼저 차단 거래 이력 평가 함수를 호출하여 차단 여부를 판정합니다. 차단으로 판정되면 즉시 차단 결과를 반환합니다.
    - 차단으로 판정되지 않으면 이체 정책 평가 함수를 호출하여 이체 가능 여부와 추가 인증 필요 여부를 판정합니다.
    - 반환값은 이체 가능 여부, 차단 여부, 추가 인증 필요 여부,적용된 규정 근거, 다음 노드 안내 등을 포함하는 딕셔너리입니다.
    """
    blocked_result = evaluate_blocked_transaction_history(
        recent_small_payment_count=recent_small_payment_count,
        foreign_ip_access=foreign_ip_access,
        transaction_id=9,
    )

    if blocked_result["blocked"]:
        return {
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
        f"통합 금융 정책 검색. 등급: {inputs.get('grade', '')}. "
        f"요청 금액: {inputs.get('request_amount', 0)}원. "
        f"당일 누적 이체액: {inputs.get('daily_total', 0)}원. "
        f"최근 1시간 소액 결제 횟수: {inputs.get('recent_small_payment_count', 0)}회. "
        f"해외 IP 접근 여부: {'예' if inputs.get('foreign_ip_access', False) else '아니오'}. "
        "ID 9 FDS 차단 조건, ID 10 고객센터 연결 노드, 이체 한도, MFA 규정을 찾아주세요."
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
                "사용자 등급: {grade}\n"
                "요청 금액: {request_amount}원\n"
                "당일 누적 이체액: {daily_total}원\n"
                "최근 1시간 소액 결제 횟수: {recent_small_payment_count}회\n"
                "해외 IP 접근 여부: {foreign_ip_access}\n"
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