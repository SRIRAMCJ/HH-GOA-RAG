from dataclasses import dataclass


@dataclass(slots=True)
class GuardResult:
    allowed: bool
    reason: str | None = None


BLOCKED_PATTERNS = {
    "ignore previous instructions",
    "system prompt",
}


def validate_input(query: str) -> GuardResult:
    normalized = query.strip().lower()
    if not normalized:
        return GuardResult(False, "empty_query")
    if any(pattern in normalized for pattern in BLOCKED_PATTERNS):
        return GuardResult(False, "prompt_injection_attempt")
    return GuardResult(True)


def validate_context(results: list, min_score: float) -> GuardResult:
    if not results:
        return GuardResult(False, "no_relevant_context")
    if max(float(item.score) for item in results) < min_score:
        return GuardResult(False, "low_relevance_context")
    return GuardResult(True)


def validate_answer(answer: str, grounded: bool) -> GuardResult:
    if not answer.strip():
        return GuardResult(False, "empty_answer")
    if not grounded:
        return GuardResult(False, "answer_not_grounded")
    return GuardResult(True)
