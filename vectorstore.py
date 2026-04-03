from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _load_csv_documents(file_path: str) -> List[Document]:
    """CSV 파일에서 문서를 로드하는 함수입니다.
    - CSV 파일을 읽어서 각 행을 Document 객체로 변환합니다.
    - Document의 page_content에는 id, 카테고리, 하위분류, 제목, 내용을 포함하는 텍스트가 저장됩니다.
    - Document의 metadata에는 source, id, category, sub_category, title이 포함됩니다
    - 반환값은 Document 객체의 리스트입니다.
    """
    documents: List[Document] = []

    with open(file_path, mode="r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            documents.append(
                Document(
                    page_content=(
                        f"id: {row.get('id', '')}\n"
                        f"카테고리: {row.get('category', '')}\n"
                        f"하위분류: {row.get('sub_category', '')}\n"
                        f"제목: {row.get('title', '')}\n"
                        f"내용: {row.get('content', '')}"
                    ),
                    metadata={
                        "source": file_path,
                        "id": row.get("id", ""),
                        "category": row.get("category", ""),
                        "sub_category": row.get("sub_category", ""),
                        "title": row.get("title", ""),
                    },
                )
            )

    return documents


def _load_json_documents(file_path: str) -> List[Document]:
    """JSON 파일에서 문서를 로드하는 함수입니다.
    - JSON 파일을 읽어서 각 항목을 Document 객체로 변환합니다.
    - Document의 page_content에는 question과 answer가 포함됩니다.
    - Document의 metadata에는 source가 포함됩니다.
    - 반환값은 Document 객체의 리스트입니다.
    """
    with open(file_path, mode="r", encoding="utf-8") as json_file:
        payload = json.load(json_file)

    if isinstance(payload, dict):
        payload = payload.get("data", [])

    documents: List[Document] = []
    for item in payload:
        documents.append(
            Document(
                page_content=f"질문: {item.get('question', '')}\n답변: {item.get('answer', '')}",
                metadata={"source": file_path},
            )
        )

    return documents


def load_documents(file_path='./libs/dataset.csv') -> List[Document]:
    """지식 베이스 파일에서 문서를 로드하는 함수입니다.
    - 입력된 파일 경로의 확장자에 따라 CSV 또는 JSON 형식으로 문서를 로드합니다.
    - 지원되는 형식은 .csv와 .json입니다. 그 외의 형식은 ValueError를 발생시킵니다.
    - 반환값은 Document 객체의 리스트입니다.
    """
    suffix = Path(file_path).suffix.lower()
    if suffix == ".csv":
        return _load_csv_documents(file_path)
    if suffix == ".json":
        return _load_json_documents(file_path)
    raise ValueError(f"Unsupported knowledge base format: {file_path}")


def split_docs(docs: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """문서를 텍스트 청크로 분할하는 함수입니다.
    - RecursiveCharacterTextSplitter를 사용하여 각 Document의 page_content를 chunk_size 길이로 분할합니다.
    - chunk_overlap은 청크 간의 겹치는 부분의 길이를 지정합니다.
    - 반환값은 분할된 Document 객체의 리스트입니다.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)


def embedding_docs(docs: List[Document]) -> FAISS:
    """문서 리스트를 벡터스토어로 임베딩하는 함수입니다.
    - get_embeddings 함수를 호출하여 OllamaEmbeddings 인스턴스를 가져옵니다.
    - FAISS.from_documents를 사용하여 문서 리스트와 임베딩 객체를 입력으로 받아서 벡터스토어를 생성합니다.
    - 반환값은 생성된 FAISS 벡터스토어 객체입니다.
    """
    from embeddings import get_embeddings

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
    return vectorstore


def save_vectorstore(vectorstore: FAISS, file_path="./exp-faiss/vectorstore"):
    """벡터스토어를 로컬 파일로 저장하는 함수입니다.
    - 입력된 벡터스토어 객체를 지정된 파일 경로에 저장합니다.
    - 저장하기 전에 파일 경로의 부모 디렉토리가 존재하지 않으면 생성합니다.
    - vectorstore.save_local 메서드를 사용하여 벡터스토어를 저장합니다.
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(file_path)


def load_vectorstore(file_path="./exp-faiss/vectorstore"):
    """로컬 파일에서 벡터스토어를 로드하는 함수입니다.
    - 입력된 파일 경로에서 벡터스토어를 로드합니다.
    - get_embeddings 함수를 호출하여 OllamaEmbeddings 인스턴스를 가져옵니다.
    - FAISS.load_local 메서드를 사용하여 파일 경로와 임베딩 객체를 입력으로 받아서 벡터스토어를 로드합니다.
    - allow_dangerous_deserialization=True 옵션을 사용하여 로드 시 발생할 수 있는 역직렬화 위험을 허용합니다.
    - 반환값은 로드된 FAISS 벡터스토어 객체입니다.
    """
    from embeddings import get_embeddings

    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(file_path, embeddings, allow_dangerous_deserialization=True)
    return vectorstore


def init_vectorstore(file_path='./libs/dataset.csv', persist_path="./exp-faiss/vectorstore"):
    """벡터스토어를 초기화하는 함수입니다.
    - 지정된 파일 경로에서 문서를 로드합니다.
    - 문서를 청크로 분할합니다.
    - 분할된 문서를 벡터스토어로 임베딩합니다.
    - 생성된 벡터스토어를 지정된 경로에 저장합니다.
    - 반환값은 생성된 FAISS 벡터스토어 객체입니다.
    """
    docs = load_documents(file_path)
    docs = split_docs(docs)
    vectorstore = embedding_docs(docs)
    save_vectorstore(vectorstore, persist_path)
    return vectorstore