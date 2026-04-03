from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _load_csv_documents(file_path: str) -> List[Document]:
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
    suffix = Path(file_path).suffix.lower()
    if suffix == ".csv":
        return _load_csv_documents(file_path)
    if suffix == ".json":
        return _load_json_documents(file_path)
    raise ValueError(f"Unsupported knowledge base format: {file_path}")


def split_docs(docs: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)


def embedding_docs(docs: List[Document]) -> FAISS:
    from embeddings import get_embeddings

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
    return vectorstore


def save_vectorstore(vectorstore: FAISS, file_path="./exp-faiss/vectorstore"):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(file_path)


def load_vectorstore(file_path="./exp-faiss/vectorstore"):
    from embeddings import get_embeddings

    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(file_path, embeddings, allow_dangerous_deserialization=True)
    return vectorstore


def init_vectorstore(file_path='./libs/dataset.csv', persist_path="./exp-faiss/vectorstore"):
    docs = load_documents(file_path)
    docs = split_docs(docs)
    vectorstore = embedding_docs(docs)
    save_vectorstore(vectorstore, persist_path)
    return vectorstore