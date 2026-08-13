# HH GOA RAG

Voice-enabled Retrieval-Augmented Generation system for Hacker House Goa 2026 Task 2.

## Technical requirements mapped

| Requirement | Implementation |
|---|---|
| Speech-to-text | Sarvam Saaras v3 adapter |
| Dataset | `ai4bharat/MSMARCO-XI` |
| Chunking | sentence groups, sliding window, semantic grouping, metadata-aware |
| Retrieval | FAISS dense + BM25 lexical + Reciprocal Rank Fusion |
| Reranking | hosted semantic reranking over the top fused candidates |
| Generation | hosted Hugging Face `Qwen/Qwen3-30B-A3B` |
| Harness | structured I/O, validation, retry, timeout/error recovery |
| Guardrails | prompt-injection, unsafe input, empty/low-context and ungrounded-answer refusal |
| Latency | per-stage instrumentation + P50/P70/P100 benchmark |

## Architecture

```text
Browser
  |
  +--> Sarvam Saaras v3 --> transcript
  |
  +--> text query -------------------------------+
                                                 |
                                      query embedding
                                                 |
                                  +--------------+--------------+
                                  |                             |
                                FAISS                         BM25
                                  |                             |
                                  +-------------+---------------+
                                                |
                                         RRF hybrid top-k
                                                |
                                         semantic rerank
                                                |
                                         top 3-5 passages
                                                |
                                        Qwen3-30B-A3B
                                                |
                                         grounding guard
                                                |
                                             answer
```

## Cloud-first design

The application does not require a local GPU. Hugging Face Inference Providers are used for hosted generation and feature extraction; the dataset and model artifacts stay outside GitHub source control. The official Hugging Face dataset contains translated MS MARCO queries, answers and English/translated passages across Indic-language configurations.

## Dataset

`MSMARCO-XI` has language configurations including Tamil (`ta`), Hindi (`hi`), Bengali (`bn`), Telugu (`te`) and others. The builder uses the translated and English passage fields and keeps query/language metadata attached to every chunk.

Example build command on cloud/CI:

```bash
python scripts/build_index.py --config ta --split train --max-docs 5000
```

The output bundle is:

```text
artifacts/index/
├── vectors.faiss
├── chunks.jsonl
├── bm25_tokens.jsonl
└── manifest.json
```

The dataset itself is never committed to GitHub.

## Secrets

Configure these in the deployment platform/GitHub Actions secrets:

```text
HF_TOKEN
SARVAM_API_KEY
```

Never commit API keys or `.env` files.

## API

- `GET /health`
- `GET /api/status`
- `POST /api/voice/transcribe`
- `POST /api/query`

`POST /api/query` accepts:

```json
{"query":"your question","language":"ta","debug":true}
```

and returns the answer, grounding status, confidence, source chunks and stage latency.

## Benchmark

Run the benchmark against a deployed service:

```bash
python scripts/benchmark.py --url https://YOUR_HOST/api/query --queries tests/benchmark_queries.json
```

The report contains:

- P50
- P70
- P100
- mean latency
- failure rate
- under-200ms status
- per-stage timings

**Do not publish latency numbers until the benchmark has actually been run.**

## Deployment

The root `Dockerfile` packages the FastAPI API and frontend as one cloud service. A hosted environment must provide `HF_TOKEN`, `SARVAM_API_KEY` and a generated `artifacts/index` bundle. Hugging Face Docker Spaces can expose FastAPI applications on a configurable port and provide runtime secrets through environment variables.

## Submission checklist

- [ ] Dataset/index built from MSMARCO-XI
- [ ] Multi-strategy retrieval verified
- [ ] 100+ benchmark queries executed
- [ ] P50/P70/P100 recorded
- [ ] <200 ms target measured and reported honestly
- [ ] Voice demo working
- [ ] Guardrail refusal cases demonstrated
- [ ] Live deployment URL
- [ ] Process video
- [ ] Product demo video
- [ ] Social posts include `#RAGInGoa`
