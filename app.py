from __future__ import annotations

import json
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

history = InMemoryChatMessageHistory()


def get_vectorstore():
	if VECTORSTORE_PATH.exists():
		return load_vectorstore(str(VECTORSTORE_PATH))
	return init_vectorstore(str(DATASET_PATH), str(VECTORSTORE_PATH))


def ask_int(prompt: str, default: int = 0) -> int:
	raw_value = input(prompt).strip()
	if not raw_value:
		return default
	return int(raw_value.replace(",", ""))


def ask_grade() -> str:
	raw_value = input("사용자 등급을 선택하세요 (1=VIP, 2=일반): ").strip().lower()
	if raw_value in {"1", "vip"}:
		return "VIP"
	if raw_value in {"2", "일반", "normal"}:
		return "일반"
	return "일반"


def ask_yes_no(prompt: str, default: bool = False) -> bool:
	raw_value = input(prompt).strip().lower()
	if not raw_value:
		return default
	return raw_value in {"y", "yes", "1", "true", "t", "예"}


def classify_ip_region(is_foreign_ip: bool) -> str:
	return "foreign" if is_foreign_ip else "domestic"


def append_transaction_history(record: dict):
	TRANSACTION_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
	with open(TRANSACTION_HISTORY_PATH, mode="a", encoding="utf-8") as file:
		file.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_transaction_history() -> list[dict]:
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
	try:
		return datetime.fromisoformat(value)
	except (TypeError, ValueError):
		return None


def calculate_daily_total(records: list[dict], now: datetime) -> int:
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
	transferable_text = "가능" if decision.get("transferable", False) else "불가"
	blocked_text = "차단" if decision.get("blocked", False) else "차단되지 않음"
	extra_auth_text = "필요" if decision.get("extra_auth_required", False) else "불필요"
	next_node = decision.get("next_node_name") or "일반 검증 흐름"

	lines = [
		f"이체 가능 여부: {transferable_text}",
		f"차단 여부(ID 9): {blocked_text}",
		f"추가 인증 필요 여부: {extra_auth_text}",
		f"다음 노드: {next_node}",
	]

	reasons = decision.get("reasons", [])
	if reasons:
		lines.append("근거:")
		for reason in reasons:
			lines.append(f"- {reason}")

	user_message = decision.get("user_message")
	if user_message:
		lines.append(f"사용자 안내: {user_message}")

	return "\n".join(lines)


def render_rag_explanation(decision: dict) -> str:
	transferable_text = "가능" if decision.get("transferable", False) else "불가"
	blocked_text = "차단" if decision.get("blocked", False) else "차단되지 않음"
	extra_auth_text = "필요" if decision.get("extra_auth_required", False) else "불필요"
	next_node = decision.get("next_node_name") or "일반 검증 흐름"

	lines = [
		"통합 금융 정책 설명:",
		f"- 이체 가능 여부: {transferable_text}",
		f"- 차단 여부(ID 9): {blocked_text}",
		f"- 추가 인증 필요 여부: {extra_auth_text}",
		f"- 다음 노드: {next_node}",
	]

	reasons = decision.get("reasons", [])
	if reasons:
		lines.append("- 근거:")
		for reason in reasons:
			lines.append(f"  - {reason}")

	user_message = decision.get("user_message")
	if user_message:
		lines.append(f"- 사용자 안내: {user_message}")

	return "\n".join(lines)


def main():
	load_dotenv()
	vectorstore = get_vectorstore()
	chain = build_unified_banking_chain(vectorstore)

	grade = ask_grade()
	request_amount = ask_int("요청 금액을 입력하세요(원): ")
	foreign_ip_access = ask_yes_no("해외 IP 접근인가요? (y/n): ")
	now = datetime.now()
	records = load_transaction_history()
	daily_total = calculate_daily_total(records, now)
	recent_small_payment_count = calculate_recent_small_payment_count(records, now, request_amount)
	ip_region = classify_ip_region(foreign_ip_access)

	input_payload = {
		"question": "통합 금융 정책으로 이체 가능 여부, 차단 여부(ID 9), 다음 노드(ID 10), 추가 인증을 안내해 주세요.",
		"grade": grade,
		"request_amount": request_amount,
		"daily_total": daily_total,
		"recent_small_payment_count": recent_small_payment_count,
		"foreign_ip_access": foreign_ip_access,
	}

	result = chain.invoke(input_payload)
	decision = evaluate_unified_policy(
		grade=grade,
		request_amount=request_amount,
		daily_total=daily_total,
		recent_small_payment_count=recent_small_payment_count,
		foreign_ip_access=foreign_ip_access,
	)

	history.add_user_message(
		f"grade={grade}, amount={request_amount}, daily_total={daily_total}, small_payments_1h={recent_small_payment_count}, region={ip_region}"
	)
	history.add_ai_message(render_decision_summary(decision))

	log_record = {
		"timestamp": now.isoformat(timespec="seconds"),
		"grade": grade,
		"request_amount": request_amount,
		"daily_total": daily_total,
		"recent_small_payment_count": recent_small_payment_count,
		"ip_region": ip_region,
		"blocked": decision.get("blocked", False),
		"transferable": decision.get("transferable", False),
		"extra_auth_required": decision.get("extra_auth_required", False),
		"next_node_id": decision.get("next_node_id"),
		"next_node_name": decision.get("next_node_name"),
	}
	append_transaction_history(log_record)

	print("\n=== 결정 결과(고정) ===")
	print(render_decision_summary(decision))
	print("\n=== RAG 설명(결정값 기반) ===")
	print(render_rag_explanation(decision))
	print("\n=== 저장된 거래 요약 ===")
	print(json.dumps(log_record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
	main()
