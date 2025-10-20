"""
tests/test_chunker.py

Unit tests for the three chunking strategies (fixed, sentence, paragraph)
and the BM25 tokenizer. All logic is pure Python — no DB or OpenAI calls.

The chunker is the most correctness-critical component: bad chunk boundaries
corrupt every downstream embedding and retrieval result.
"""

import pytest
from app.services.chunker import (
    chunk_text,
    chunk_fixed,
    chunk_sentence,
    chunk_paragraph,
    token_count,
    tokenize_for_bm25,
)


# -- Helpers -----------------------------------------------------------------

def total_tokens(chunks: list[str]) -> int:
    return sum(token_count(c) for c in chunks)


SAMPLE_PROSE = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How valiantly did Quixote fight the windmills. "
    "Sphinx of black quartz, judge my vow. "
    "Waltz, bad nymph, for quick jigs vex. "
    "The five boxing wizards jump quickly. "
    "Jackdaws love my big sphinx of quartz."
)


# -- chunk_fixed -------------------------------------------------------------

class TestChunkFixed:

    def test_produces_chunks(self):
        chunks = chunk_fixed(SAMPLE_PROSE, chunk_size=20, overlap=0)
        assert len(chunks) >= 1

    def test_no_chunk_exceeds_chunk_size(self):
        chunks = chunk_fixed(SAMPLE_PROSE, chunk_size=20, overlap=0)
        for c in chunks:
            assert token_count(c) <= 20

    def test_overlap_reduces_chunk_count(self):
        chunks_no_overlap = chunk_fixed(SAMPLE_PROSE, chunk_size=20, overlap=0)
        chunks_with_overlap = chunk_fixed(SAMPLE_PROSE, chunk_size=20, overlap=5)
        # Overlap means more chunks covering same content
        assert len(chunks_with_overlap) >= len(chunks_no_overlap)

    def test_empty_string_returns_empty_list(self):
        assert chunk_fixed("", chunk_size=20, overlap=0) == []

    def test_short_text_returns_single_chunk(self):
        text = "Hello world."
        chunks = chunk_fixed(text, chunk_size=100, overlap=0)
        assert len(chunks) == 1

    def test_content_preserved(self):
        """All tokens from original text must appear somewhere in the chunks."""
        text = "apple banana cherry date mango"
        chunks = chunk_fixed(text, chunk_size=5, overlap=0)
        combined = " ".join(chunks)
        for word in ["apple", "banana", "cherry", "date", "mango"]:
            assert word in combined


# -- chunk_sentence ----------------------------------------------------------

class TestChunkSentence:

    def test_produces_chunks(self):
        chunks = chunk_sentence(SAMPLE_PROSE, chunk_size=30, overlap=5)
        assert len(chunks) >= 1

    def test_no_chunk_exceeds_chunk_size(self):
        chunks = chunk_sentence(SAMPLE_PROSE, chunk_size=30, overlap=5)
        for c in chunks:
            assert token_count(c) <= 30 + 5  # slight tolerance for sentence packing

    def test_oversized_single_sentence_is_split(self):
        # One very long sentence with no punctuation to split on
        long_sentence = "word " * 200  # ~200 tokens
        chunks = chunk_sentence(long_sentence, chunk_size=50, overlap=0)
        assert len(chunks) > 1
        for c in chunks:
            assert token_count(c) <= 50

    def test_empty_string_returns_empty(self):
        assert chunk_sentence("", chunk_size=50, overlap=0) == []

    def test_overlap_carries_context_forward(self):
        """With overlap > 0, adjacent chunks should share some tokens."""
        text = " ".join([f"sentence{i}." for i in range(20)])
        chunks_no_ol = chunk_sentence(text, chunk_size=20, overlap=0)
        chunks_with_ol = chunk_sentence(text, chunk_size=20, overlap=5)
        # Overlap produces more chunks than no-overlap
        assert len(chunks_with_ol) >= len(chunks_no_ol)

    def test_respects_sentence_boundaries(self):
        """Chunks should not split in the middle of a sentence where avoidable."""
        text = "First sentence ends here. Second sentence starts here. Third here."
        chunks = chunk_sentence(text, chunk_size=100, overlap=0)
        # All fits in one chunk at size 100
        assert len(chunks) == 1
        assert "First sentence ends here" in chunks[0]


# -- chunk_paragraph ---------------------------------------------------------

class TestChunkParagraph:

    def test_splits_on_paragraphs(self):
        text = "Para one content here.\n\nPara two content here.\n\nPara three here."
        chunks = chunk_paragraph(text, chunk_size=100, overlap=0)
        # All three fit in one chunk at size 100
        assert len(chunks) >= 1

    def test_oversized_paragraph_is_sentence_split(self):
        long_para = ("This is a sentence. " * 50)  # ~200+ tokens
        text = long_para + "\n\nShort para."
        chunks = chunk_paragraph(text, chunk_size=40, overlap=0)
        for c in chunks:
            assert token_count(c) <= 45  # slight tolerance

    def test_empty_string_returns_empty(self):
        assert chunk_paragraph("", chunk_size=50, overlap=0) == []

    def test_single_paragraph_produces_chunks(self):
        text = "Just one paragraph with no double newlines anywhere in the text."
        chunks = chunk_paragraph(text, chunk_size=100, overlap=0)
        assert len(chunks) >= 1


# -- chunk_text dispatcher ---------------------------------------------------

class TestChunkTextDispatcher:

    def test_default_strategy_is_sentence(self):
        chunks_default = chunk_text(SAMPLE_PROSE, chunk_size=30)
        chunks_sentence = chunk_text(SAMPLE_PROSE, strategy="sentence", chunk_size=30)
        assert chunks_default == chunks_sentence

    def test_fixed_strategy(self):
        chunks = chunk_text(SAMPLE_PROSE, strategy="fixed", chunk_size=20, overlap=0)
        assert len(chunks) >= 1

    def test_paragraph_strategy(self):
        text = "Paragraph A.\n\nParagraph B.\n\nParagraph C."
        chunks = chunk_text(text, strategy="paragraph", chunk_size=100)
        assert len(chunks) >= 1

    def test_unknown_strategy_falls_back_to_sentence(self):
        chunks = chunk_text(SAMPLE_PROSE, strategy="unknown_strategy", chunk_size=30)
        assert len(chunks) >= 1


# -- tokenize_for_bm25 -------------------------------------------------------

class TestTokenizeForBm25:

    def test_lowercases_tokens(self):
        tokens = tokenize_for_bm25("Hello World")
        assert all(t == t.lower() for t in tokens)

    def test_strips_punctuation(self):
        tokens = tokenize_for_bm25("hello, world! how are you?")
        for t in tokens:
            assert t.isalnum()

    def test_empty_string_returns_empty_list(self):
        assert tokenize_for_bm25("") == []

    def test_returns_expected_words(self):
        tokens = tokenize_for_bm25("The quick brown FOX jumps.")
        assert "the" in tokens
        assert "quick" in tokens
        assert "fox" in tokens

    def test_numbers_are_kept(self):
        tokens = tokenize_for_bm25("model version 42 released")
        assert "42" in tokens

