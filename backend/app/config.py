from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 7860
    frontend_origin: str = "*"

    stt_provider: str = "sarvam"
    sarvam_api_key: str | None = None
    sarvam_model: str = "saaras:v3"

    hf_token: str | None = None
    hf_llm_model: str = "Qwen/Qwen3-30B-A3B"
    hf_inference_provider: str = "auto"
    hf_embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    hf_reranker_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    hf_dataset_id: str = "ai4bharat/MSMARCO-XI"

    index_dir: str = "artifacts/index"
    index_max_docs: int = 5000
    dense_top_k: int = 20
    lexical_top_k: int = 20
    rerank_top_k: int = 5
    min_retrieval_score: float = 0.0
    max_context_chars: int = 12000
    benchmark_queries: int = 100


@lru_cache
def get_settings() -> Settings:
    return Settings()
