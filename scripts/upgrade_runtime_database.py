"""Back up and migrate an existing runtime SQLite database without reseeding it."""
from __future__ import annotations

import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
DB_PATH = ROOT_DIR / "data" / "sqlite_data" / "project.db"
SESSION_TABLES = ("user_profiles", "stage_progress", "user_qa", "user_answers", "session_states")


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
    ).fetchone() is not None


def needs_upgrade(conn: sqlite3.Connection) -> bool:
    if table_exists(conn, "learning_package_claims"):
        foreign_keys = conn.execute("PRAGMA foreign_key_list(learning_package_claims)").fetchall()
        grouped: dict[int, list[tuple]] = {}
        for row in foreign_keys:
            grouped.setdefault(row[0], []).append(row)
        valid_source_fk = any(
            rows[0][2] == "learning_package_sources"
            and [item[3] for item in sorted(rows, key=lambda item: item[1])] == ["trace_id", "source_ref"]
            and [item[4] for item in sorted(rows, key=lambda item: item[1])] == ["trace_id", "source_id"]
            for rows in grouped.values()
        )
        if not valid_source_fk:
            return True

    for table in SESSION_TABLES:
        if table_exists(conn, table):
            orphan = conn.execute(
                f"SELECT 1 FROM {table} AS legacy "
                "LEFT JOIN sessions AS current ON current.session_id = legacy.session_id "
                "WHERE legacy.session_id IS NOT NULL AND current.session_id IS NULL LIMIT 1"
            ).fetchone()
            if orphan:
                return True
    return False


def main() -> int:
    if not DB_PATH.exists():
        print("[SKIP] runtime database does not exist; first-run initialization will create it.")
        return 0

    with sqlite3.connect(DB_PATH) as conn:
        required = needs_upgrade(conn)
    if not required:
        print("[OK] runtime database migration is already current.")
        return 0

    backup_dir = DB_PATH.parent / "backups"
    backup_dir.mkdir(exist_ok=True)
    backup_path = backup_dir / f"project.before-upgrade.{datetime.now():%Y%m%d-%H%M%S}.db"
    with sqlite3.connect(DB_PATH) as source, sqlite3.connect(backup_path) as target:
        source.backup(target)
    print(f"[BACKUP] {backup_path.relative_to(ROOT_DIR)}")

    sys.path.insert(0, str(BACKEND_DIR))
    from db.sqlite_client import SQLiteClient

    client = SQLiteClient()
    client.close()
    print("[OK] runtime database migration completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
