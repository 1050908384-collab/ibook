"""Read-only startup health check for the runtime SQLite database."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DB_PATH = ROOT_DIR / "data" / "sqlite_data" / "project.db"
MINIMUM_COUNTS = {
    "knowledge_blocks": 15,
    "knowledge_subtopics": 73,
    "knowledge_docs": 68,
    "knowledge_chunks": 1,
    "test_questions": 217,
    "learning_resources": 1,
    "block_resources": 15,
    "subtopic_resources": 73,
}


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    raise SystemExit(1)


def main() -> int:
    if not DB_PATH.exists():
        fail(f"database not found: {DB_PATH}")

    uri = f"file:{DB_PATH.as_posix()}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as exc:
        fail(f"database cannot be opened read-only: {exc}")
    try:
        integrity = conn.execute("PRAGMA integrity_check").fetchall()
        if integrity != [("ok",)]:
            fail(f"integrity_check: {integrity}")
        foreign_key_errors = conn.execute("PRAGMA foreign_key_check").fetchall()
        if foreign_key_errors:
            fail(f"foreign_key_check returned {len(foreign_key_errors)} row(s): {foreign_key_errors[:3]}")
        for table, minimum in MINIMUM_COUNTS.items():
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            if count < minimum:
                fail(f"{table}: expected at least {minimum}, found {count}")
            print(f"[PASS] {table}: {count}")
    except sqlite3.Error as exc:
        fail(f"database query failed: {exc}")
    finally:
        conn.close()
    print("[PASS] integrity_check")
    print("[PASS] foreign_key_check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
