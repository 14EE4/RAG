from langchain_ollama import OllamaEmbeddings

MODEL_NAME = "nomic-embed-text-v2-moe"

# Use explicit local endpoint to avoid issues with malformed OLLAMA_HOST.
OLLAMA_BASE_URL = "http://127.0.0.1:11434"

_embeddings = OllamaEmbeddings(model=MODEL_NAME, base_url=OLLAMA_BASE_URL)


def get_embeddings():
    """OllamaEmbeddings 인스턴스를 반환하는 함수입니다.
    - OllamaEmbeddings는 OLLAMA_BASE_URL에서 MODEL_NAME 모델을 사용하여 텍스트 임베딩을 생성하는 클래스입니다.
    - 이 함수는 전역적으로 생성된 _embeddings 인스턴스를 반환하여 애플리케이션 전체에서 동일한 임베딩 객체를 사용할 수 있도록 합니다.
    """
    return _embeddings

