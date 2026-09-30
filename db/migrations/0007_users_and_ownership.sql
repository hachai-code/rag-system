-- User identity + per-user ownership. The `user` table mirrors fastapi-users'
-- SQLAlchemyBaseUserTableUUID so db/migrate.py stays the single schema source (the
-- SQLAlchemy model in rag/auth.py maps this table, it does not create it).

CREATE TABLE IF NOT EXISTS "user" (
    id              UUID         PRIMARY KEY,
    email           VARCHAR(320) NOT NULL UNIQUE,
    hashed_password VARCHAR(1024) NOT NULL,
    is_active       BOOLEAN      NOT NULL DEFAULT true,
    is_superuser    BOOLEAN      NOT NULL DEFAULT false,
    is_verified     BOOLEAN      NOT NULL DEFAULT false
);

-- Per-user scoping for the deep agent's private data. Nullable: rows that predate auth
-- (a rebuildable cache, transient paused runs) become orphans no user's filter matches.
ALTER TABLE agent_threads ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES "user"(id);
ALTER TABLE qa_memory     ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES "user"(id);

CREATE INDEX IF NOT EXISTS qa_memory_user_idx ON qa_memory (user_id);
