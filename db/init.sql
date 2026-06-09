-- Runs automatically the first time the container initializes an empty data
-- volume. Enables pgvector in the `rag` database so the app never has to.
CREATE EXTENSION IF NOT EXISTS vector;
