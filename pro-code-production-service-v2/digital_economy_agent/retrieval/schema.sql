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

-- When the *source document* was last changed, as asserted by whoever published the
-- ingestion event. Distinct from ingested_at, which records when this row was written.
-- Nullable: batch ingestion has no such notion, and a NULL on either side of the
-- comparison below means "no version information", so the write applies.
--
-- Added by ALTER rather than in the CREATE TABLE above so an existing database picks it
-- up too; IF NOT EXISTS keeps ensure_schema() idempotent.
--
-- What it guards: the conditional DO UPDATE in PgVectorStore.upsert, so a redelivered
-- older version cannot replace a newer row for the same chunk. Kafka orders per
-- partition and Pub/Sub only with ordering keys, so v1 arriving after v2 is normal.
--
-- What it does NOT guard: chunks a newer version deleted. Edited text hashes
-- differently, so it inserts as its own row and never reaches ON CONFLICT -- a
-- superseded version's chunks are added alongside the current ones. Removing them
-- needs document reconciliation, which is not implemented; see the README limitations
-- and test_a_superseded_version_leaves_its_chunks_behind.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS source_updated_at TIMESTAMPTZ;

-- Vector search. HNSW over cosine distance; embeddings are stored unit-normalised so
-- `<=>` is a true cosine distance.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Lexical search, the other half of hybrid retrieval.
CREATE INDEX IF NOT EXISTS chunks_text_search_gin
    ON chunks USING gin (text_search);

-- Tenant isolation and source filters are the common access patterns.
CREATE INDEX IF NOT EXISTS chunks_tenant_source ON chunks (tenant_id, source);

-- Row-level security. The `WHERE tenant_id = %s` in application SQL stays, but it is
-- now defence in depth rather than the boundary itself: before this, a query path that
-- forgot the filter read the 'default' tenant silently, and no test caught it because
-- in tests almost everything *is* 'default'. The policy makes the database refuse.
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
-- Without FORCE, a table owner bypasses its own policies. The service connects as the
-- owner in local development and in CI, so omitting this would leave exactly those
-- environments unprotected -- and they are the ones the tests run in.
ALTER TABLE chunks FORCE ROW LEVEL SECURITY;

-- `CREATE POLICY` has no IF NOT EXISTS, and this file is applied on every service
-- start, so the drop is what keeps ensure_schema() idempotent.
DROP POLICY IF EXISTS chunks_tenant_isolation ON chunks;
CREATE POLICY chunks_tenant_isolation ON chunks
    -- current_setting(..., missing_ok => true) yields NULL when app.tenant_id was
    -- never set, and `tenant_id = NULL` is NULL, not true -- so an unscoped
    -- connection reads zero rows rather than every row. Failing closed on a
    -- forgotten SET is the entire point.
    USING       (tenant_id = current_setting('app.tenant_id', true))
    WITH CHECK  (tenant_id = current_setting('app.tenant_id', true));
