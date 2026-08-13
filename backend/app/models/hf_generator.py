import json
from dataclasses import dataclass

from huggingface_hub import InferenceClient

from app.config import Settings


@dataclass(slots=True)
class GenerationResult:
    answer: str
    confidence: float
    grounded: bool


class HuggingFaceGenerator:
    def __init__(self, settings: Settings):
        if not settings.hf_token:
            raise RuntimeError("HF_TOKEN is required for hosted generation")
        self.settings = settings
        self.client = InferenceClient(
            provider=settings.hf_inference_provider,
            api_key=settings.hf_token,
        )

    def generate(self, query: str, context: str) -> GenerationResult:
        system = (
            "You are the HH Goa RAG answer generator. Answer only from the supplied context. "
            "If the context does not support the answer, set grounded=false and say that the "
            "information is not available in the provided dataset. Never invent facts. "
            "Return strict JSON with keys: answer, confidence, grounded."
        )
        prompt = f"Question:\n{query}\n\nRetrieved context:\n{context}"
        response = self.client.chat.completions.create(
            model=self.settings.hf_llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=256,
        )
        content = response.choices[0].message.content or ""
        try:
            data = json.loads(content)
            return GenerationResult(
                answer=str(data.get("answer", "")),
                confidence=max(0.0, min(1.0, float(data.get("confidence", 0.0)))),
                grounded=bool(data.get("grounded", False)),
            )
        except (ValueError, TypeError, json.JSONDecodeError):
            # The orchestrator will treat malformed output as a recoverable model error.
            raise ValueError("Hosted LLM returned invalid structured output")
