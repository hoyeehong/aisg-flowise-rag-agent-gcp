-- Retrieval schema. Applied idempotently by PgVectorStore.ensure_schema().
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id            BIGSERIAL PRIMARY KEY,
    -- Content hash makes ingestion idempotent: re-running a pipeline over an unchanged
    -- document updates rows in place instead of duplicating the corpus.
    content_hash  TEXT        NOT NULL,
    tenant_id     TEXT        NOT NULL DEFAULT 'default',
    source        TEXT        NOT NULL,
    page          INTEGER,
    text          TEXT        NOT NULL,
    -- Generated rather than trigger-maintained, so it can never drift from `text`.
    text_search   TSVECTOR    GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    embedding     VECTOR(768) NOT NULL,
    embedder      TEXT        NOT NULL,
    redactions    JSONB       NOT NULL DEFAULT '{}'::jsonb,
    ingested_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, content_hash)
);

-- Vector search. HNSW over cosine distance; embeddings are stored unit-normalised so
-- `<=>` is a true cosine distance.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Lexical search, the other half of hybrid retrieval.
CREATE INDEX IF NOT EXISTS chunks_text_search_gin
    ON chunks USING gin (text_search);

-- Tenant isolation and source filters are the common access patterns.
CREATE INDEX IF NOT EXISTS chunks_tenant_source ON chunks (tenant_id, source);
