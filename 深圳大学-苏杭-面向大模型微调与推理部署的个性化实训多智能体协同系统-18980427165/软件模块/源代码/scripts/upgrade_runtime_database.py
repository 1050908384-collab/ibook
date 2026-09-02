"""Back up and migrate an existing runtime SQLite database without user-data loss."""
from __future__ import annotations

import shutil
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
DB_PATH = ROOT_DIR / "data" / "sqlite_data" / "project.db"
SESSION_TABLES = ("user_profiles", "stage_progress", "user_qa", "user_answers", "session_states")
SEEDS_DIR = ROOT_DIR / "data" / "seeds"


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
    ).fetchone() is not None


def needs_upgrade(conn: sqlite3.Connection) -> bool:
    # A runtime database is deliberately excluded from Git.  Teammates can
    # therefore carry forward a database made by an older curriculum version.
    # Compare identities, not just non-zero row counts, before serving it.
    expected_blocks = {
        item["block_id"]
        for item in json.loads((SEEDS_DIR / "knowledge_blocks.json").read_text(encoding="utf-8"))
    }
    expected_subtopics = {
        item["subtopic_id"]
        for item in json.loads((SEEDS_DIR / "knowledge_subtopics.json").read_text(encoding="utf-8"))
    }
    expected_question_ids = {
        item["question_id"]
        for item in json.loads((SEEDS_DIR / "test_questions.json").read_text(encoding="utf-8"))
    }
    if not table_exists(conn, "knowledge_blocks") or not table_exists(conn, "knowledge_subtopics"):
        return True
    actual_blocks = {row[0] for row in conn.execute("SELECT block_id FROM knowledge_blocks")}
    actual_subtopics = {row[0] for row in conn.execute("SELECT subtopic_id FROM knowledge_subtopics")}
    # Extra legacy nodes are harmless once all current IDs are present. Do not
    # force a destructive re-upgrade merely because an old learner record
    # still references a retired topic.
    if not expected_blocks.issubset(actual_blocks) or not expected_subtopics.issubset(actual_subtopics):
        return True

    # The question bank is versioned through the seed file as well. Older
    # runtime databases may have all current knowledge nodes but still miss
    # newly added assessment questions. Sync the authoritative catalog before
    # the read-only health check rejects such a database.
    if not table_exists(conn, "test_questions"):
        return True
    actual_question_ids = {
        row[0] for row in conn.execute("SELECT question_id FROM test_questions")
    }
    if not expected_question_ids.issubset(actual_question_ids):
        return True

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
    sys.path.insert(0, str(ROOT_DIR / "scripts"))
    import insert_knowledge_data

    # This updates only authoritative catalogue rows. User sessions, progress,
    # answers, and Q&A history live in separate tables and are retained.
    insert_knowledge_data.main()
    print("[OK] runtime database migration completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
