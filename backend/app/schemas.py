from typing import Literal
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    language: str | None = None
    debug: bool = False


class Source(BaseModel):
    chunk_id: str
    document_id: str | None = None
    text: str
    score: float
    strategy: str


class QueryResponse(BaseModel):
    answer: str
    grounded: bool
    confidence: float
    refused: bool = False
    refusal_reason: str | None = None
    sources: list[Source] = []
    latency_ms: float
    stages_ms: dict[str, float] = {}


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    model: str
