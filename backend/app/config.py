from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    stt_provider: str = "elevenlabs"
    elevenlabs_api_key: str | None = None
    sarvam_api_key: str | None = None

    hf_token: str | None = None
    hf_llm_model: str = "Qwen/Qwen3-30B-A3B"
    hf_inference_provider: str = "deepinfra"
    hf_embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    hf_reranker_model: str = "Qwen/Qwen3-Reranker-0.6B"
    hf_dataset_id: str = "ai4bharat/MSMARCO-XI"

    dense_top_k: int = 20
    lexical_top_k: int = 20
    rerank_top_k: int = 5
    chunk_size: int = 384
    chunk_overlap: int = 64
    min_retrieval_score: float = 0.20
    max_context_chars: int = 12000
    benchmark_queries: int = 100


@lru_cache
def get_settings() -> Settings:
    return Settings()
