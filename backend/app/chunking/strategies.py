from dataclasses import dataclass
import re


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    strategy: str
    start_sentence: int = 0
    end_sentence: int = 0


_SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s+")


def sentence_split(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text.strip()) if s.strip()]


def sentence_chunks(document_id: str, text: str, target_chars: int = 1400) -> list[Chunk]:
    sentences = sentence_split(text)
    chunks: list[Chunk] = []
    current: list[str] = []
    start = 0
    for i, sentence in enumerate(sentences):
        if current and len(" ".join(current)) + len(sentence) + 1 > target_chars:
            chunks.append(Chunk(f"{document_id}:sentence:{len(chunks)}", document_id, " ".join(current), "sentence", start, i - 1))
            current = []
            start = i
        current.append(sentence)
    if current:
        chunks.append(Chunk(f"{document_id}:sentence:{len(chunks)}", document_id, " ".join(current), "sentence", start, len(sentences) - 1))
    return chunks


def sliding_window_chunks(document_id: str, text: str, window: int = 1200, overlap: int = 200) -> list[Chunk]:
    text = " ".join(text.split())
    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        end = min(start + window, len(text))
        chunks.append(Chunk(f"{document_id}:window:{len(chunks)}", document_id, text[start:end], "sliding_window"))
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def metadata_aware_chunk(document_id: str, text: str, section: str | None = None) -> list[Chunk]:
    prefix = f"Section: {section}\n" if section else ""
    chunks = sentence_chunks(document_id, prefix + text, target_chars=1600)
    for chunk in chunks:
        chunk.strategy = "metadata_aware"
    return chunks
