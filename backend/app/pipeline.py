from time import perf_counter

from app.analytics.timing import timed
from app.config import Settings
from app.embeddings.hf_embeddings import HFEmbedder
from app.guardrails.checks import validate_answer, validate_context, validate_input
from app.models.hf_generator import HuggingFaceGenerator
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import HostedReranker


class RAGPipeline:
    def __init__(self, settings: Settings, retriever: HybridRetriever):
        self.settings = settings
        self.retriever = retriever
        self.generator = HuggingFaceGenerator(settings)
        self.embedder = HFEmbedder(settings.hf_token, settings.hf_embedding_model)
        self.reranker = HostedReranker(settings.hf_token, settings.hf_reranker_model)

    def answer(self, query: str) -> dict:
        stages: dict[str, float] = {}
        started = perf_counter()

        with timed(stages, "input_guard"):
            input_guard = validate_input(query)
        if not input_guard.allowed:
            return self._refusal(input_guard.reason or "blocked", stages, started)

        with timed(stages, "query_embedding"):
            query_vector = self.embedder.embed_one(query)

        with timed(stages, "retrieval"):
            lexical = self.retriever.lexical_search(query, self.settings.lexical_top_k)
            dense = self.retriever.dense_search(query_vector, self.settings.dense_top_k)
            results = self.retriever.fuse(dense, lexical, self.settings.dense_top_k)

        with timed(stages, "context_guard"):
            context_guard = validate_context(results, self.settings.min_retrieval_score)
        if not context_guard.allowed:
            return self._refusal(context_guard.reason or "no_context", stages, started)

        with timed(stages, "reranking"):
            ranked = self.reranker.rerank(query, results, self.settings.rerank_top_k)

        context = "\n\n".join(item.chunk.text for item in ranked)
        context = context[: self.settings.max_context_chars]

        generated = None
        generation_error = None
        for attempt in range(2):
            try:
                with timed(stages, "generation" if attempt == 0 else "generation_retry"):
                    generated = self.generator.generate(query, context)
                generation_error = None
                break
            except Exception as exc:
                generation_error = str(exc)
        if generated is None:
            return self._refusal(f"generation_error:{generation_error}", stages, started)

        with timed(stages, "answer_guard"):
            answer_guard = validate_answer(generated.answer, generated.grounded)
        if not answer_guard.allowed:
            return self._refusal(answer_guard.reason or "ungrounded", stages, started)

        total = (perf_counter() - started) * 1000.0
        return {
            "answer": generated.answer,
            "grounded": generated.grounded,
            "confidence": generated.confidence,
            "refused": False,
            "refusal_reason": None,
            "sources": [
                {
                    "chunk_id": item.chunk.chunk_id,
                    "document_id": item.chunk.document_id,
                    "text": item.chunk.text,
                    "score": float(item.score),
                    "strategy": item.chunk.strategy,
                    "metadata": item.chunk.metadata or {},
                }
                for item in ranked
            ],
            "latency_ms": total,
            "stages_ms": stages,
        }

    @staticmethod
    def _refusal(reason: str, stages: dict[str, float], started: float) -> dict:
        return {
            "answer": "I can't answer that from the provided dataset.",
            "grounded": False,
            "confidence": 0.0,
            "refused": True,
            "refusal_reason": reason,
            "sources": [],
            "latency_ms": (perf_counter() - started) * 1000.0,
            "stages_ms": stages,
        }
