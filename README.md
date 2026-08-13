# HH GOA RAG

Voice-enabled Retrieval-Augmented Generation system for Hacker House Goa 2026 Task 2.

## Official task targets

- Speech-to-text: Sarvam or ElevenLabs
- Dataset: `ai4bharat/MSMARCO-XI`
- Multi-strategy chunking
- Vector retrieval + lexical retrieval
- End-to-end RAG pipeline target: `< 200 ms`
- P50 / P70 / P100 latency reporting
- Structured model harness with retries, validation and recovery
- Guardrails for off-topic, unsafe, ungrounded and hallucinated answers

## Architecture

```text
Voice -> STT -> Query Processor -> Hybrid Retrieval -> Reranker -> HF LLM -> Grounding Guard -> Answer
                                      |                 |
                                    FAISS              BM25
```

The application is designed to run remotely. No local GPU is required. Large models and ML assets are configured through Hugging Face / hosted inference providers.

## Repository layout

- `backend/` FastAPI API and orchestration
- `backend/chunking/` multi-strategy chunking
- `backend/retrieval/` FAISS/BM25/hybrid retrieval
- `backend/models/` hosted model adapters
- `backend/guardrails/` input/context/output safety and grounding checks
- `backend/analytics/` latency instrumentation and benchmarks
- `frontend/` voice-first web client
- `scripts/` dataset/index preparation
- `.github/workflows/` CI

## Configuration

Copy `.env.example` to `.env` for local development or configure the same variables in the deployment platform. Secrets must never be committed.

## Status

Initial engineering scaffold. Dataset ingestion, retrieval benchmarking and hosted model integration are being implemented incrementally so every latency claim is measured rather than assumed.
