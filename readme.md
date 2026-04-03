# RAG Banking Policy Assistant

규정 기반 금융 요청 판정 + RAG 근거 설명 프로그램입니다.

이 프로젝트는 다음 원칙으로 동작합니다.

1. 결정은 코드 규칙으로 고정합니다.
2. LLM은 벡터 검색된 규정 문서로 근거 설명만 생성합니다.

## 현재 지원 요청 유형

1. 송금
2. 해외송금
3. 대출
4. 투자

## 핵심 정책 로직

### 송금

- ID 9(FDS) 차단 조건 검사
- 차단 시 ID 10(고객센터 연결) 분기
- 차단이 아니면 이체 한도 및 MFA 규칙 적용

### 해외송금

- ID 26(연간 해외송금 누적 한도) 검사
- 한도 위반 시 즉시 거절 + 법적 근거 출력
- 한도 통과 시 송금 흐름(FDS/한도/MFA) 이어서 적용

### 대출

- ID 30(DSR 40% 규정) 검사
- 위반 시 즉시 거절 + 법적 근거 출력
- 통과 시 Credit_Score 흐름으로 이동

### 투자

- ID 39(투자 적합성) 검사
- 안정형 + 고위험 요청이면 즉시 거절 + 법적 근거 출력
- 통과 시 투자 안내 흐름으로 이동

## 해외송금 환율 처리

- 해외송금에서 금액은 KRW 한 번만 입력합니다.
- USD 환산은 고정 환율 `1 USD = 1500 KRW`를 사용합니다.
- 환산식: `request_amount_usd = ceil(request_amount_krw / 1500)`

## 자동 계산 항목

- 당일 누적 이체액: `libs/transaction_history.jsonl`에서 자동 합산
- 최근 1시간 소액 결제 횟수: 거래 로그 기반 자동 집계

## 주요 파일

- `app.py`: CLI 입력/출력, 자동 계산, 로그 저장
- `chain.py`: 정책 판정 함수, 순차 검증, RAG 체인
- `vectorstore.py`: CSV 문서 로드, 분할, FAISS 저장/로드
- `embeddings.py`: Ollama 임베딩 설정
- `libs/dataset.csv`: 규정 원천 데이터
- `libs/transaction_history.jsonl`: 거래 로그(JSONL)

## 실행 방법

1. 가상환경 활성화

```powershell
& d:\vscodeworkspace\RAG\venv\Scripts\Activate.ps1
```

2. 실행

```powershell
& d:/vscodeworkspace/RAG/venv/Scripts/python.exe d:/vscodeworkspace/RAG/app.py
```

## 입력 예시

### 해외송금

```text
사용자 등급을 선택하세요 (1=VIP, 2=일반): 1
요청 유형을 선택하세요 (1=송금, 2=해외송금, 3=대출, 4=투자): 2
요청 금액을 입력하세요(원): 1500000
해외 IP 접근인가요? (y/n): n
연간 해외송금 누적 금액을 입력하세요(USD): 49000
```

### 대출

```text
사용자 등급을 선택하세요 (1=VIP, 2=일반): 2
요청 유형을 선택하세요 (1=송금, 2=해외송금, 3=대출, 4=투자): 3
연소득을 입력하세요(원): 100000000
연간 총원리금 상환액을 입력하세요(원): 50000000
```

### 투자

```text
사용자 등급을 선택하세요 (1=VIP, 2=일반): 1
요청 유형을 선택하세요 (1=송금, 2=해외송금, 3=대출, 4=투자): 4
투자 성향을 선택하세요 (1=안정형, 2=중립형, 3=공격형): 1
요청 상품 위험등급을 선택하세요 (1=저위험, 2=중위험, 3=고위험): 3
```

## 출력 구조

1. 결정 결과(고정)
- 실제 시스템 판정 기준

2. RAG 설명(벡터 근거 기반 LLM)
- 데이터셋 규정 인용 기반 해설

3. 저장된 거래 요약
- 로그에 기록되는 최종 레코드

## 저장 로그 필드(요약)

- `request_type`, `grade`, `request_amount`
- `annual_remittance_usd`, `request_amount_usd`
- `annual_income`, `annual_debt_service`
- `investment_profile`, `requested_product_risk`
- `compliance_approved`, `failed_rule_id`, `legal_basis`, `rejection_reason`
- `blocked`, `transferable`, `extra_auth_required`, `next_node_name`

## 주의사항

1. RAG 설명은 근거 해설이며, 최종 결론은 고정 결정 결과를 기준으로 사용해야 합니다.
2. `.gitignore`에 `venv/`, `contents/`, `exp-faiss/`, `libs/transaction_history.jsonl`이 포함되어야 합니다.