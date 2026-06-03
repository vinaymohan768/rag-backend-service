-- init.sql: RAG backend service schema

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ── Collections ───────────────────────────────────────────────────────────────
-- Namespaces for organizing documents. Each collection is independently
-- searchable: e.g., "product-docs", "support-kb", "legal-contracts"
CREATE TABLE IF NOT EXISTS collections (
    id          UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    name        TEXT        NOT NULL UNIQUE,
    description TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

INSERT INTO collections (name, description) VALUES
    ('default', 'Default collection')
ON CONFLICT DO NOTHING;


-- ── Documents ─────────────────────────────────────────────────────────────────
-- Source document registry. Tracks what's been ingested, when, and its status.
CREATE TABLE IF NOT EXISTS documents (
    id              UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    collection_id   UUID        NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    title           TEXT        NOT NULL,
    source          TEXT        NOT NULL,
    char_count      INT         NOT NULL,
    chunk_count     INT         NOT NULL DEFAULT 0,
    chunk_strategy  TEXT        NOT NULL DEFAULT 'sentence',
    status          TEXT        NOT NULL DEFAULT 'pending'
                                CHECK (status IN ('pending', 'processing', 'ready', 'error')),
    error_message   TEXT,
    metadata        JSONB       NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_documents_collection
    ON documents (collection_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_documents_status
    ON documents (status);


-- ── Chunks with embeddings ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS chunks (
    id              UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id     UUID        NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    collection_id   UUID        NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    chunk_index     INT         NOT NULL,
    content         TEXT        NOT NULL,
    token_count     INT         NOT NULL,
    embedding       vector(1536),
    -- BM25 support: store lowercased token list for keyword search
    tokens          TEXT[],
    metadata        JSONB       NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- IVFFlat for ANN vector search: partition per collection via WHERE clause
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_chunks_document
    ON chunks (document_id, chunk_index);

CREATE INDEX IF NOT EXISTS idx_chunks_collection
    ON chunks (collection_id);

-- GIN index on token array for fast BM25 keyword matching
CREATE INDEX IF NOT EXISTS idx_chunks_tokens
    ON chunks USING GIN (tokens);


-- ── Search audit log ──────────────────────────────────────────────────────────
-- Records every search query for analytics, debugging, and quality tracking.
CREATE TABLE IF NOT EXISTS search_log (
    id              BIGSERIAL   PRIMARY KEY,
    collection_id   UUID        REFERENCES collections(id),
    query           TEXT        NOT NULL,
    top_k           INT         NOT NULL,
    rerank_k        INT         NOT NULL,
    strategy        TEXT        NOT NULL,   -- "vector" | "hybrid"
    result_count    INT         NOT NULL,
    latency_ms      INT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_search_log_collection
    ON search_log (collection_id, created_at DESC);
