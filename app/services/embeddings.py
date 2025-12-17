"""
services/embeddings.py

Thin wrapper around OpenAI embeddings with batching and basic retry.
Keeps embedding logic out of the ingestion and retrieval paths.
"""

import logging
import time
from openai import OpenAI, RateLimitError
from app.config import settings

log = logging.getLogger(__name__)

BATCH_SIZE = 100  # OpenAI max per request


def embed_texts(texts: list[str], client: OpenAI) -> list[list[float]]:
    """
    Embed a list of texts in batches. Returns embeddings in the same order.
    Retries once on rate limit with a 20-second backoff.
    """
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        attempt = 0
        while attempt < 2:
            try:
                response = client.embeddings.create(
                    model=settings.embedding_model,
                    input=batch,
                )
                batch_embeddings = [
                    item.embedding
                    for item in sorted(response.data, key=lambda x: x.index)
                ]
                all_embeddings.extend(batch_embeddings)
                break
            except RateLimitError:
                if attempt == 0:
                    log.warning("Rate limit hit, backing off 20s...")
                    time.sleep(20)
                    attempt += 1
                else:
                    raise

    return all_embeddings


def embed_query(query: str, client: OpenAI) -> list[float]:
    """Embed a single query string."""
    return embed_texts([query], client)[0]
