"""Apply pending SQL migrations, tracked in a schema_migrations table.

Run once after `docker compose up -d`, and again whenever new migrations land:

    uv run python -m db.migrate

Each db/migrations/NNNN_*.sql file is applied once, in filename order, inside its own
transaction, and recorded in schema_migrations so re-runs are no-ops. Migrations use
CREATE ... IF NOT EXISTS, so a DB provisioned before this system existed catches up
cleanly. To add a change: drop a new higher-numbered .sql file in db/migrations/.
"""

from pathlib import Path

import psycopg

from rag.db import DB_URL

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def migrate() -> None:
    with psycopg.connect(DB_URL) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS schema_migrations (
                   version    TEXT PRIMARY KEY,
                   applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
               )"""
        )
        conn.commit()
        applied = {r[0] for r in conn.execute("SELECT version FROM schema_migrations").fetchall()}

        pending = sorted(p for p in MIGRATIONS_DIR.glob("*.sql") if p.stem not in applied)
        if not pending:
            print("schema up to date")
            return

        for path in pending:
            print(f"applying {path.name}")
            conn.execute(path.read_text())  # no params -> psycopg runs the whole file
            conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (path.stem,))
            conn.commit()
        print(f"applied {len(pending)} migration(s)")


if __name__ == "__main__":
    migrate()
