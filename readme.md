# RAG Banking Policy Assistant

이 프로젝트는 금융 이체 정책을 RAG로 설명하고, 실제 판단은 코드 규칙으로 고정하는 통합형 프로그램입니다.

핵심 목표는 다음 두 가지입니다.

1. 결정 일관성: 이체 가능 여부, 차단 여부, 다음 노드는 코드 규칙으로 확정
2. 설명 품질: LLM은 벡터 검색된 규정 문서를 근거로 "왜" 그런 결정이 나왔는지 설명

## 주요 기능

1. 통합 입력 플로우
- 등급 선택: `1=VIP`, `2=일반`
- 요청 금액 입력
- 해외 IP 여부 입력

2. 자동 계산
- 당일 누적 이체액: `libs/transaction_history.jsonl` 기반 자동 합산
- 최근 1시간 소액 결제 횟수: 거래 로그 기반 자동 집계

3. 정책 판정(고정 로직)
- ID 9 (FDS 차단 조건):
	- 최근 1시간 소액 결제 `5회 이상` 또는 해외 IP 접근이면 차단
	- 차단 시 즉시 `고객센터 연결`(ID 10) 분기
- 일반 이체 한도:
	- 일반: 1회 200만 원, 1일 500만 원
	- VIP: 1일 5,000만 원
- VIP 고액 이체:
	- 1,000만 원 이상이면 `Security_Check` 노드(추가 인증) 분기

4. RAG 설명
- 벡터스토어에서 규정 문서 검색 후 LLM이 근거 설명 생성
- 결론은 재판정하지 않고, 확정된 결정을 근거 중심으로 해설

5. 거래 내역 저장
- `libs/transaction_history.jsonl`에 매 요청 결과 append
- 저장 필드: 시간, 등급, 금액, 자동 계산값, 차단 여부, 다음 노드 등

## 프로젝트 구조

- `app.py`: 실행 진입점, 입력 처리, 자동 집계, 결과 출력, 로그 저장
- `chain.py`: 정책 판정 함수 + RAG 체인 구성
- `vectorstore.py`: CSV 문서를 문서화하고 FAISS 벡터스토어 생성/로드
- `embeddings.py`: Ollama 임베딩 모델 설정
- `libs/dataset.csv`: 정책 원천 데이터
- `libs/transaction_history.jsonl`: 거래 로그 파일(실행 중 자동 생성/추가)

## 동작 개요

1. 사용자 입력 수집
2. 거래 로그에서 자동 지표 계산(당일 누적/1시간 소액 횟수)
3. 정책 로직으로 결정 확정
4. RAG 체인으로 근거 설명 생성
5. 결과 출력 및 JSONL 저장

## 실행 방법

1. 가상환경 활성화

```powershell
& d:\vscodeworkspace\RAG\venv\Scripts\Activate.ps1
```

2. 프로그램 실행

```powershell
& d:/vscodeworkspace/RAG/venv/Scripts/python.exe d:/vscodeworkspace/RAG/app.py
```

3. 입력 예시

```text
사용자 등급을 선택하세요 (1=VIP, 2=일반): 1
요청 금액을 입력하세요(원): 20000000
해외 IP 접근인가요? (y/n): n
```

## 출력 예시 해석

- `결정 결과(고정)`: 실제 시스템 판단 기준
- `RAG 설명(벡터 근거 기반 LLM)`: 데이터셋 규정 근거 해설
- `저장된 거래 요약`: 로그에 저장되는 최종 레코드

## 데이터/정책 확장 포인트

1. `libs/dataset.csv`에 규정 추가
2. `chain.py`의 정책 판정 함수 확장
3. 필요 시 GeoIP 자동 판별 로직 추가

## 주의사항

- `.gitignore`에 `venv/`, `contents/`, `exp-faiss/`, `libs/transaction_history.jsonl`이 포함되어 있어야 합니다.
- RAG 설명은 근거 해설 용도이며, 최종 판정값은 코드 규칙 결과를 기준으로 사용해야 합니다.