"""Apply versioned SQL migrations to a TimescaleDB instance."""

from __future__ import annotations

import os
from pathlib import Path

import psycopg


def main() -> None:
    database_url = os.environ.get("QUANT_PAIRS_DATABASE_URL")
    if not database_url:
        raise SystemExit("set QUANT_PAIRS_DATABASE_URL before applying migrations")
    migrations = sorted((Path(__file__).parents[1] / "db" / "migrations").glob("*.sql"))
    with psycopg.connect(database_url, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS public.schema_migration (
                    version TEXT PRIMARY KEY,
                    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """)
            cursor.execute("SELECT version FROM public.schema_migration")
            applied = {row[0] for row in cursor.fetchall()}
            for migration in migrations:
                if migration.name not in applied:
                    cursor.execute(migration.read_text())
                    cursor.execute(
                        "INSERT INTO public.schema_migration (version) VALUES (%s)",
                        (migration.name,),
                    )
                    print(f"applied {migration.name}")


if __name__ == "__main__":
    main()
