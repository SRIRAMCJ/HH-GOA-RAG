from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.retrieval.hybrid import HybridRetriever, IndexedChunk
from app.schemas import HealthResponse, QueryRequest, QueryResponse

settings = get_settings()

# The production deployment will load the prebuilt MSMARCO-XI index at startup.
# Keeping the construction isolated makes the index provider replaceable.
_retriever = HybridRetriever([])

app = FastAPI(title="HH Goa RAG", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="hh-goa-rag", model=settings.hf_llm_model)


@app.post("/api/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    # Dataset/index loading and the full pipeline are wired in the next implementation stage.
    # This endpoint currently fails closed rather than fabricating an answer.
    return QueryResponse(
        answer="The retrieval index is not initialized yet.",
        grounded=False,
        confidence=0.0,
        refused=True,
        refusal_reason="index_not_initialized",
        sources=[],
        latency_ms=0.0,
        stages_ms={},
    )
