from langchain_ollama import OllamaEmbeddings

MODEL_NAME = "nomic-embed-text-v2-moe"

# Use explicit local endpoint to avoid issues with malformed OLLAMA_HOST.
OLLAMA_BASE_URL = "http://127.0.0.1:11434"

_embeddings = OllamaEmbeddings(model=MODEL_NAME, base_url=OLLAMA_BASE_URL)


def get_embeddings():
    return _embeddings

