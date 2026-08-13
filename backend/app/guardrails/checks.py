from dataclasses import dataclass


@dataclass(slots=True)
class GuardResult:
    allowed: bool
    reason: str | None = None


BLOCKED_PATTERNS = {
    "ignore previous instructions",
    "ignore all previous instructions",
    "reveal the system prompt",
    "show me the system prompt",
    "developer message",
}

UNSAFE_PATTERNS = {
    "how to make a bomb",
    "how to build a bomb",
    "make a weapon",
}


def validate_input(query: str) -> GuardResult:
    normalized = " ".join(query.strip().lower().split())
    if not normalized:
        return GuardResult(False, "empty_query")
    if len(normalized) > 2000:
        return GuardResult(False, "query_too_long")
    if any(pattern in normalized for pattern in BLOCKED_PATTERNS):
        return GuardResult(False, "prompt_injection_attempt")
    if any(pattern in normalized for pattern in UNSAFE_PATTERNS):
        return GuardResult(False, "unsafe_request")
    return GuardResult(True)


def validate_context(results: list, min_score: float = 0.0) -> GuardResult:
    if not results:
        return GuardResult(False, "no_relevant_context")
    if min_score > 0 and max(float(item.score) for item in results) < min_score:
        return GuardResult(False, "low_relevance_context")
    return GuardResult(True)


def validate_answer(answer: str, grounded: bool) -> GuardResult:
    if not answer.strip():
        return GuardResult(False, "empty_answer")
    if not grounded:
        return GuardResult(False, "answer_not_grounded")
    return GuardResult(True)
