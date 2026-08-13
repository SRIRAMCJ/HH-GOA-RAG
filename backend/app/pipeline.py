from time import perf_counter

from app.analytics.timing import timed
from app.config import Settings
from app.guardrails.checks import validate_answer, validate_context, validate_input
from app.models.hf_generator import HuggingFaceGenerator
from app.retrieval.hybrid import HybridRetriever


class RAGPipeline:
    def __init__(self, settings: Settings, retriever: HybridRetriever):
        self.settings = settings
        self.retriever = retriever
        self.generator = HuggingFaceGenerator(settings)

    def answer(self, query: str) -> dict:
        stages: dict[str, float] = {}
        started = perf_counter()

        with timed(stages, "input_guard"):
            input_guard = validate_input(query)
        if not input_guard.allowed:
            return self._refusal(input_guard.reason or "blocked", stages, started)

        with timed(stages, "retrieval"):
            lexical = self.retriever.lexical_search(query, self.settings.lexical_top_k)
            # Dense results will be injected once the FAISS index is built. The fusion
            # interface already supports both retrieval channels.
            results = self.retriever.fuse([], lexical, self.settings.dense_top_k)

        with timed(stages, "context_guard"):
            context_guard = validate_context(results, self.settings.min_retrieval_score)
        if not context_guard.allowed:
            return self._refusal(context_guard.reason or "no_context", stages, started)

        context = "\n\n".join(item.chunk.text for item in results[: self.settings.rerank_top_k])
        context = context[: self.settings.max_context_chars]

        with timed(stages, "generation"):
            generated = self.generator.generate(query, context)

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
                }
                for item in results[: self.settings.rerank_top_k]
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
