from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.retrieval.hybrid import HybridRetriever
from app.schemas import HealthResponse, QueryRequest, QueryResponse
from app.stt.sarvam import SarvamSTT

settings = get_settings()
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


@app.post("/api/voice/transcribe")
async def transcribe_voice(
    file: UploadFile = File(...),
    language_code: str | None = Header(default=None, alias="X-Language-Code"),
):
    if settings.stt_provider.lower() != "sarvam":
        raise HTTPException(status_code=501, detail="Only Sarvam STT is wired in this first backend stage")
    try:
        audio = await file.read()
        result = await SarvamSTT(settings).transcribe(
            audio,
            filename=file.filename or "audio.webm",
            language_code=language_code,
        )
        return {
            "transcript": result.get("transcript", ""),
            "language_code": result.get("language_code"),
            "request_id": result.get("request_id"),
        }
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"STT provider error: {exc}") from exc


@app.post("/api/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    # The MSMARCO-XI index is intentionally not checked into GitHub. The next stage
    # adds the remote/prebuilt index loader and connects this endpoint to RAGPipeline.
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
