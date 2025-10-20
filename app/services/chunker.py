"""
services/chunker.py

Three chunking strategies with different semantic preservation trade-offs:

  fixed     : Naive token-count slicing. Fast, predictable size.
               Best for: structured data, code, tables.

  sentence  : Sentence-boundary aware (default). Packs sentences into
               token-bounded windows with overlap.
               Best for: general prose, documentation, articles.

  paragraph : Splits on double-newlines first, then sentences within
               oversized paragraphs. Preserves topic coherence best.
               Best for: long-form content with clear section structure.

All strategies return chunks as plain strings. Token counts are computed
with tiktoken cl100k_base (same tokenizer as text-embedding-3-small).
"""

import re
import tiktoken

_enc = tiktoken.get_encoding("cl100k_base")


def _tokens(text: str) -> list[int]:
    return _enc.encode(text)


def _decode(tokens: list[int]) -> str:
    return _enc.decode(tokens)


# ── Fixed chunking ─────────────────────────────────────────────────────────────

def chunk_fixed(text: str, chunk_size: int, overlap: int) -> list[str]:
    tokens = _tokens(text)
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i : i + chunk_size]
        if chunk_tokens:
            chunks.append(_decode(chunk_tokens))
    return chunks


# ── Sentence chunking ──────────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using a simple regex that handles common cases."""
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'])', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def chunk_sentence(text: str, chunk_size: int, overlap: int) -> list[str]:
    sentences = _split_sentences(text)
    chunks = []
    current: list[int] = []

    for sentence in sentences:
        sent_tokens = _tokens(sentence)

        # Hard-split sentences that exceed chunk_size on their own
        if len(sent_tokens) > chunk_size:
            if current:
                chunks.append(_decode(current))
                current = []
            for i in range(0, len(sent_tokens), chunk_size - overlap):
                seg = sent_tokens[i : i + chunk_size]
                if seg:
                    chunks.append(_decode(seg))
            continue

        if len(current) + len(sent_tokens) > chunk_size:
            if current:
                chunks.append(_decode(current))
            current = current[-overlap:] + sent_tokens if overlap else sent_tokens
        else:
            current.extend(sent_tokens)

    if current:
        chunks.append(_decode(current))

    return chunks


# ── Paragraph chunking ─────────────────────────────────────────────────────────

def chunk_paragraph(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split on paragraph boundaries first (double newline). Each paragraph
    that fits within chunk_size becomes its own chunk. Paragraphs that
    exceed chunk_size are further split using sentence chunking.
    """
    paragraphs = re.split(r'\n\s*\n', text.strip())
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    buffer: list[int] = []

    for para in paragraphs:
        para_tokens = _tokens(para)

        if len(para_tokens) > chunk_size:
            # Flush buffer first
            if buffer:
                chunks.append(_decode(buffer))
                buffer = []
            # Split the oversized paragraph at sentence boundaries
            chunks.extend(chunk_sentence(para, chunk_size, overlap))
            continue

        if len(buffer) + len(para_tokens) > chunk_size:
            if buffer:
                chunks.append(_decode(buffer))
            buffer = buffer[-overlap:] + para_tokens if overlap else para_tokens
        else:
            buffer.extend(para_tokens)

    if buffer:
        chunks.append(_decode(buffer))

    return chunks


# ── Public interface ───────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    strategy: str = "sentence",
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[str]:
    if strategy == "fixed":
        return chunk_fixed(text, chunk_size, overlap)
    elif strategy == "paragraph":
        return chunk_paragraph(text, chunk_size, overlap)
    else:
        return chunk_sentence(text, chunk_size, overlap)


def token_count(text: str) -> int:
    return len(_tokens(text))


def tokenize_for_bm25(text: str) -> list[str]:
    """Lowercase word tokens for BM25 index: strip punctuation."""
    return re.findall(r'\b[a-z0-9]+\b', text.lower())
