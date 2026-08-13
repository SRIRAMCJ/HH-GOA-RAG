import json
import re
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
        self.client = InferenceClient(provider=settings.hf_inference_provider, api_key=settings.hf_token)

    def _completion(self, query: str, context: str):
        system = (
            "You are the HH Goa retrieval-grounded answer generator. Use ONLY the retrieved context. "
            "Never use outside knowledge. If the context does not directly support the answer, set grounded=false. "
            "Return ONLY valid JSON: {\"answer\":\"...\",\"confidence\":0.0,\"grounded\":true}."
        )
        prompt = f"Question:\n{query}\n\nRetrieved context:\n{context}"
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        try:
            return self.client.chat.completions.create(
                model=self.settings.hf_llm_model,
                messages=messages,
                temperature=0.0,
                max_tokens=220,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        except Exception:
            # Some inference providers do not accept provider-specific chat_template kwargs.
            return self.client.chat.completions.create(
                model=self.settings.hf_llm_model,
                messages=messages,
                temperature=0.0,
                max_tokens=220,
            )

    def generate(self, query: str, context: str) -> GenerationResult:
        response = self._completion(query, context)
        content = (response.choices[0].message.content or "").strip()
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.IGNORECASE).strip()
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if not match:
                raise ValueError("Hosted LLM returned invalid structured output") from exc
            data = json.loads(match.group(0))
        answer = str(data.get("answer", "")).strip()
        grounded = bool(data.get("grounded", False))
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
        if not answer:
            raise ValueError("Hosted LLM returned an empty answer")
        return GenerationResult(answer=answer, confidence=confidence, grounded=grounded)
