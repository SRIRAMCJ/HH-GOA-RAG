# Hugging Face deployment

This repository is designed for cloud execution. The user's computer does not need to run the models.

## 1. Create a Hugging Face Space

Create a **Docker Space** and point it at the project source. Docker Spaces support FastAPI/custom containers. The container listens on port `7860`.

## 2. Add Space secrets

In Space Settings -> Secrets add:

- `HF_TOKEN` — a fine-grained token with Inference Providers permission
- `SARVAM_API_KEY` — Sarvam API key

## 3. Provide the index

The application expects:

```text
artifacts/index/vectors.faiss
artifacts/index/chunks.jsonl
```

Build it from GitHub Actions using:

**Actions -> Build MSMARCO-XI RAG Index -> Run workflow**

Use `ta` for Tamil or another official MSMARCO-XI language configuration. The workflow stores the generated index as an Actions artifact.

For a production deployment, copy/mount the generated bundle into `artifacts/index` using persistent cloud storage or a Hugging Face dataset/repository. Do not commit a full MSMARCO-XI corpus to GitHub.

## 4. Runtime variables

```text
HF_LLM_MODEL=Qwen/Qwen3-30B-A3B
HF_INFERENCE_PROVIDER=auto
HF_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
HF_RERANKER_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
INDEX_DIR=artifacts/index
STT_PROVIDER=sarvam
```

## 5. Verify

Open:

- `/health`
- `/api/status`

`/api/status` must report `index_ready: true` before `/api/query` is used.

## Important

The HH Goa `<200 ms` requirement must be demonstrated with the benchmark script. Hosted network/model latency is not guessed or hidden. Run `scripts/benchmark.py` against the deployed service and publish the real P50/P70/P100 numbers.
