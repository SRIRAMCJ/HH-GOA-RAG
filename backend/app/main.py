from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.pipeline import RAGPipeline
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.index_store import DenseIndex
from app.retrieval.hybrid import IndexedChunk
from app.schemas import HealthResponse, QueryRequest, QueryResponse
from app.stt.sarvam import SarvamSTT

settings = get_settings()


def load_retriever() -> HybridRetriever:
    dense = DenseIndex.load(Path(settings.index_dir))
    if dense is None:
        return HybridRetriever([])
    return HybridRetriever(dense.chunks, dense)


_retriever = load_retriever()
_pipeline = RAGPipeline(settings, _retriever) if _retriever.chunks and settings.hf_token else None

app = FastAPI(title="HH Goa RAG", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="hh-goa-rag", model=settings.hf_llm_model)


@app.get("/api/status")
def status() -> dict:
    return {
        "index_ready": bool(_retriever.chunks),
        "chunks": len(_retriever.chunks),
        "llm_model": settings.hf_llm_model,
        "embedding_model": settings.hf_embedding_model,
        "stt_provider": settings.stt_provider,
    }


@app.post("/api/voice/transcribe")
async def transcribe_voice(
    file: UploadFile = File(...),
    language_code: str | None = Header(default=None, alias="X-Language-Code"),
):
    if settings.stt_provider.lower() != "sarvam":
        raise HTTPException(status_code=501, detail="Sarvam is the configured STT provider")
    if not settings.sarvam_api_key:
        raise HTTPException(status_code=503, detail="SARVAM_API_KEY is not configured")
    try:
        audio = await file.read()
        result = await SarvamSTT(settings).transcribe(audio, filename=file.filename or "audio.webm", language_code=language_code)
        return {
            "transcript": result.get("transcript", ""),
            "language_code": result.get("language_code"),
            "request_id": result.get("request_id"),
        }
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"STT provider error: {exc}") from exc


@app.post("/api/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="RAG index is not initialized. Build and mount artifacts/index first.")
    result = _pipeline.answer(request.query)
    return QueryResponse(**result)
