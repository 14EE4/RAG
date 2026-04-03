from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory

from chain import build_unified_banking_chain, evaluate_unified_policy
from vectorstore import init_vectorstore, load_vectorstore


VECTORSTORE_PATH = Path("./exp-faiss/vectorstore")
DATASET_PATH = Path("./libs/dataset.csv")
TRANSACTION_HISTORY_PATH = Path("./libs/transaction_history.jsonl")
SMALL_PAYMENT_THRESHOLD_KRW = 100_000
FX_KRW_PER_USD = 1_500

history = InMemoryChatMessageHistory()


def get_vectorstore():
	"""벡터스토어를 로드하거나 초기화합니다.
	- 기존에 저장된 벡터스토어가 있으면 로드합니다.
	- 없으면 데이터셋에서 문서를 로드하여 벡터스토어를 초기화하고 저장합니다.
	"""
	if VECTORSTORE_PATH.exists():
		return load_vectorstore(str(VECTORSTORE_PATH))
	return init_vectorstore(str(DATASET_PATH), str(VECTORSTORE_PATH))


def ask_int(prompt: str, default: int = 0) -> int:
	"""사용자에게 정수 입력을 요청하는 함수입니다.
	- 입력이 없으면 기본값을 반환합니다.
	- 입력에 쉼표가 포함된 경우 제거하고 정수로 변환합니다.
	"""
	raw_value = input(prompt).strip()
	if not raw_value:
		return default
	return int(raw_value.replace(",", ""))


def ask_grade() -> str:
	"""사용자에게 등급 입력을 요청하는 함수입니다.
	"""
	raw_value = input("사용자 등급을 선택하세요 (1=VIP, 2=일반): ").strip().lower()
	if raw_value in {"1", "vip"}:
		return "VIP"
	if raw_value in {"2", "일반", "normal"}:
		return "일반"
	return "일반"


def ask_request_type() -> str:
	"""요청 유형 입력을 표준 값으로 변환하는 함수입니다.
	- 1=송금, 2=해외송금, 3=대출, 4=투자 입력을 지원합니다.
	- 허용되지 않는 값은 기본값으로 송금을 반환합니다.
	"""
	raw_value = input("요청 유형을 선택하세요 (1=송금, 2=해외송금, 3=대출, 4=투자): ").strip().lower()
	if raw_value in {"1", "송금", "transfer"}:
		return "송금"
	if raw_value in {"2", "해외송금", "overseas"}:
		return "해외송금"
	if raw_value in {"3", "대출", "loan"}:
		return "대출"
	if raw_value in {"4", "투자", "investment"}:
		return "투자"
	return "송금"


def ask_yes_no(prompt: str, default: bool = False) -> bool:
	"""사용자에게 예/아니오 입력을 요청하는 함수입니다.
	- 입력이 없으면 기본값을 반환합니다.
	- 예/아니오 입력을 다양한 형태로 인식합니다.
    """
	raw_value = input(prompt).strip().lower()
	if not raw_value:
		return default
	return raw_value in {"y", "yes", "1", "true", "t", "예"}


def ask_investment_profile() -> str:
	"""투자 성향 입력을 표준 값으로 변환하는 함수입니다.
	- 1=안정형, 2=중립형, 3=공격형 입력을 지원합니다.
	- 허용되지 않는 값은 기본값으로 중립형을 반환합니다.
	"""
	raw_value = input("투자 성향을 선택하세요 (1=안정형, 2=중립형, 3=공격형): ").strip().lower()
	if raw_value in {"1", "안정형", "stable"}:
		return "안정형"
	if raw_value in {"2", "중립형", "neutral"}:
		return "중립형"
	if raw_value in {"3", "공격형", "aggressive"}:
		return "공격형"
	return "중립형"


def ask_product_risk() -> str:
	"""요청 상품 위험등급 입력을 표준 값으로 변환하는 함수입니다.
	- 1=저위험, 2=중위험, 3=고위험 입력을 지원합니다.
	- 허용되지 않는 값은 기본값으로 중위험을 반환합니다.
	"""
	raw_value = input("요청 상품 위험등급을 선택하세요 (1=저위험, 2=중위험, 3=고위험): ").strip().lower()
	if raw_value in {"1", "저위험", "low"}:
		return "저위험"
	if raw_value in {"2", "중위험", "medium"}:
		return "중위험"
	if raw_value in {"3", "고위험", "high"}:
		return "고위험"
	return "중위험"


def classify_ip_region(is_foreign_ip: bool) -> str:
	"""IP 접근 지역을 분류하는 함수입니다.
    - 해외 IP 접근이면 "foreign"을 반환하고, 그렇지 않으면 "domestic"을 반환합니다.
    """
	return "foreign" if is_foreign_ip else "domestic"


def append_transaction_history(record: dict):
	"""거래 기록을 JSONL 파일에 추가하는 함수입니다.
	- 기록은 JSON 형식으로 저장되며, 각 기록은 한 줄에 하나씩 저장됩니다.
	- 파일이 없으면 새로 생성됩니다.
    """
	TRANSACTION_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
	with open(TRANSACTION_HISTORY_PATH, mode="a", encoding="utf-8") as file:
		file.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_transaction_history() -> list[dict]:
	"""거래 기록을 JSONL 파일에서 로드하는 함수입니다.
    - 파일이 없으면 빈 리스트를 반환합니다.
    - 각 줄을 JSON으로 파싱하여 리스트로 반환합니다.
    - JSON 파싱 오류가 발생한 줄은 무시합니다.
"""
	if not TRANSACTION_HISTORY_PATH.exists():
		return []

	records: list[dict] = []
	with open(TRANSACTION_HISTORY_PATH, mode="r", encoding="utf-8") as file:
		for line in file:
			line = line.strip()
			if not line:
				continue
			try:
				records.append(json.loads(line))
			except json.JSONDecodeError:
				continue
	return records


def _parse_timestamp(value: str) -> datetime | None:
	"""ISO 8601 형식의 타임스탬프 문자열을 datetime 객체로 파싱하는 함수입니다.
    - 입력이 유효한 ISO 8601 형식이면 datetime 객체를 반환합니다
    - 그렇지 않으면 None을 반환합니다.
    """
	try:
		return datetime.fromisoformat(value)
	except (TypeError, ValueError):
		return None


def calculate_daily_total(records: list[dict], now: datetime) -> int:
	"""오늘 날짜에 해당하는 거래 기록의 요청 금액 합계를 계산하는 함수입니다.
    - 각 기록의 "timestamp" 필드를 ISO 8601 형식으로 파싱하여 날짜를 비교합니다.
    - "transferable"이 True인 기록만 합계에 포함됩니다.
    - 요청 금액은 "request_amount" 필드에서 정수로 변환하여 합산됩니다.
    """
	total = 0
	for record in records:
		ts = _parse_timestamp(record.get("timestamp"))
		if ts is None:
			continue
		if ts.date() != now.date():
			continue
		if not record.get("transferable", False):
			continue
		total += int(record.get("request_amount", 0))
	return total


def calculate_recent_small_payment_count(records: list[dict], now: datetime, request_amount: int) -> int:
	"""최근 1시간 이내의 소액 결제 건수를 계산하는 함수입니다.
    - 각 기록의 "timestamp" 필드를 ISO 8601 형식으로 파싱하여 현재 시간과 비교합니다.
    - "request_amount" 필드에서 정수로 변환하여 소액 결제 여부를 판단합니다.
    - 소액 결제 건수는 현재 요청도 소액인 경우 1을 추가하여 계산됩니다.
    """
	window_start = now.timestamp() - 3600
	count = 0
	for record in records:
		ts = _parse_timestamp(record.get("timestamp"))
		if ts is None:
			continue
		if ts.timestamp() < window_start:
			continue
		amount = int(record.get("request_amount", 0))
		if amount <= SMALL_PAYMENT_THRESHOLD_KRW:
			count += 1

	# 현재 요청도 소액이면 FDS 횟수 계산에 포함
	if int(request_amount) <= SMALL_PAYMENT_THRESHOLD_KRW:
		count += 1

	return count


def render_decision_summary(decision: dict) -> str:
	"""정책 결정 결과를 사람이 읽을 수 있는 형식으로 렌더링하는 함수입니다.
    - 결정 결과의 주요 필드(이체 가능 여부, 차단 여부, 추가 인증 필요 여부, 다음 노드)를 텍스트로 변환하여 요약합니다.
    - 결정 근거가 있는 경우 "근거:" 섹션을 추가하여 각 근거를 리스트 형식으로 나열합니다.
    - 사용자 안내 메시지가 있는 경우 "사용자 안내:" 섹션을 추가하여 안내 메시지를 포함합니다.
    - 최종적으로 모든 정보를 하나의 문자열로 결합하여 반환합니다.
    """
	transferable_text = "가능" if decision.get("transferable", False) else "불가"
	blocked_text = "차단" if decision.get("blocked", False) else "차단되지 않음"
	extra_auth_text = "필요" if decision.get("extra_auth_required", False) else "불필요"
	next_node = decision.get("next_node_name") or "일반 검증 흐름"

	lines = [
		f"요청 유형: {decision.get('request_type', '미정')}",
		f"컴플라이언스(ID 26/30/39): {'통과' if decision.get('compliance_approved', True) else '거절'}",
		f"이체 가능 여부: {transferable_text}",
		f"차단 여부(ID 9): {blocked_text}",
		f"추가 인증 필요 여부: {extra_auth_text}",
		f"다음 노드: {next_node}",
	]

	if not decision.get("compliance_approved", True):
		lines.append(f"거절 규정 ID: {decision.get('failed_rule_id')}")
		lines.append(f"거절 규정: {decision.get('failed_rule_title')}")
		lines.append(f"법적 근거: {decision.get('legal_basis')}")
		lines.append(f"거절 사유: {decision.get('rejection_reason')}")

	reasons = decision.get("reasons", [])
	if reasons:
		lines.append("근거:")
		for reason in reasons:
			lines.append(f"- {reason}")

	user_message = decision.get("user_message")
	if user_message:
		lines.append(f"사용자 안내: {user_message}")

	return "\n".join(lines)


def main():
	"""메인 함수입니다.
    - 환경 변수를 로드합니다.
    - 벡터스토어를 로드하거나 초기화합니다.
    - 통합 금융 정책 체인을 구축합니다.
    - 사용자로부터 입력을 받아 필요한 정보를 수집합니다.
    - 체인을 실행하여 RAG 설명을 생성합니다.
    - 정책 평가 함수를 호출하여 결정 결과를 얻습니다.
    - 거래 기록에 사용자 입력과 결정 결과를 저장합니다.
    - 결정 결과와 RAG 설명을 출력합니다.
    """
	load_dotenv()
	vectorstore = get_vectorstore()
	chain = build_unified_banking_chain(vectorstore)

	grade = ask_grade()
	request_type = ask_request_type()
	request_amount = ask_int("요청 금액을 입력하세요(원): ") if request_type in {"송금", "해외송금"} else 0
	foreign_ip_access = ask_yes_no("해외 IP 접근인가요? (y/n): ") if request_type in {"송금", "해외송금"} else False

	annual_remittance_usd = 0
	request_amount_usd = 0
	if request_type == "해외송금":
		annual_remittance_usd = ask_int("연간 해외송금 누적 금액을 입력하세요(USD): ", 0)
		request_amount_usd = math.ceil(request_amount / FX_KRW_PER_USD)

	annual_income = 0
	annual_debt_service = 0
	if request_type == "대출":
		annual_income = ask_int("연소득을 입력하세요(원): ", 0)
		annual_debt_service = ask_int("연간 총원리금 상환액을 입력하세요(원): ", 0)

	investment_profile = "중립형"
	requested_product_risk = "중위험"
	if request_type == "투자":
		investment_profile = ask_investment_profile()
		requested_product_risk = ask_product_risk()
	now = datetime.now()
	records = load_transaction_history()
	daily_total = calculate_daily_total(records, now) if request_type in {"송금", "해외송금"} else 0
	recent_small_payment_count = (
		calculate_recent_small_payment_count(records, now, request_amount)
		if request_type in {"송금", "해외송금"}
		else 0
	)
	ip_region = classify_ip_region(foreign_ip_access)

	input_payload = {
		"question": "ID 26, ID 30, ID 39를 순차 검증하고 위반 시 법적 근거와 거절 사유를 설명해 주세요.",
		"request_type": request_type,
		"grade": grade,
		"request_amount": request_amount,
		"daily_total": daily_total,
		"recent_small_payment_count": recent_small_payment_count,
		"foreign_ip_access": foreign_ip_access,
		"annual_remittance_usd": annual_remittance_usd,
		"request_amount_usd": request_amount_usd,
		"annual_income": annual_income,
		"annual_debt_service": annual_debt_service,
		"investment_profile": investment_profile,
		"requested_product_risk": requested_product_risk,
	}

	result = chain.invoke(input_payload)
	decision = evaluate_unified_policy(
		grade=grade,
		request_amount=request_amount,
		daily_total=daily_total,
		recent_small_payment_count=recent_small_payment_count,
		foreign_ip_access=foreign_ip_access,
		request_type=request_type,
		annual_remittance_usd=annual_remittance_usd,
		request_amount_usd=request_amount_usd,
		annual_income=annual_income,
		annual_debt_service=annual_debt_service,
		investment_profile=investment_profile,
		requested_product_risk=requested_product_risk,
	)

	history.add_user_message(
		f"grade={grade}, amount={request_amount}, daily_total={daily_total}, small_payments_1h={recent_small_payment_count}, region={ip_region}"
	)
	history.add_ai_message(result)

	log_record = {
		"timestamp": now.isoformat(timespec="seconds"),
		"request_type": request_type,
		"grade": grade,
		"request_amount": request_amount,
		"annual_remittance_usd": annual_remittance_usd,
		"request_amount_usd": request_amount_usd,
		"annual_income": annual_income,
		"annual_debt_service": annual_debt_service,
		"investment_profile": investment_profile,
		"requested_product_risk": requested_product_risk,
		"daily_total": daily_total,
		"recent_small_payment_count": recent_small_payment_count,
		"ip_region": ip_region,
		"compliance_approved": decision.get("compliance_approved", True),
		"failed_rule_id": decision.get("failed_rule_id"),
		"failed_rule_title": decision.get("failed_rule_title"),
		"legal_basis": decision.get("legal_basis"),
		"rejection_reason": decision.get("rejection_reason"),
		"blocked": decision.get("blocked", False),
		"transferable": decision.get("transferable", False),
		"extra_auth_required": decision.get("extra_auth_required", False),
		"next_node_id": decision.get("next_node_id"),
		"next_node_name": decision.get("next_node_name"),
	}
	append_transaction_history(log_record)

	print("\n=== 결정 결과(고정) ===")
	print(render_decision_summary(decision))
	print("\n=== RAG 설명(벡터 근거 기반 LLM) ===")
	print(result)
	print("\n=== 저장된 거래 요약 ===")
	print(json.dumps(log_record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
	main()
