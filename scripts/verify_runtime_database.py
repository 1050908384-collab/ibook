"""Read-only startup health check for the runtime SQLite database."""
from __future__ import annotations


# ---------- 路径设置：python scripts\xxx.py 直接运行时能正确找到 backend / data ----------

import sys
from pathlib import Path
_PROJ_ROOT_ = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT_) not in sys.path: sys.path.insert(0, str(_PROJ_ROOT_))
_BE_ = Path(_PROJ_ROOT_, "backend")
if str(_BE_) not in sys.path: sys.path.insert(0, str(_BE_))

import sys as _sys_py
from pathlib import Path as _Path_py
_PROJ_ROOT_ = _Path_py(__file__).resolve().parent.parent
if str(_PROJ_ROOT_) not in _sys_py.path:
    _sys_py.path.insert(0, str(_PROJ_ROOT_))
if str(_PROJ_ROOT_ / "backend") not in _sys_py.path:
    _sys_py.path.insert(0, str(_PROJ_ROOT_ / "backend"))
# -------------------------------------------------------------------------------------

import sqlite3
import sys
import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DB_PATH = ROOT_DIR / "data" / "sqlite_data" / "project.db"
MINIMUM_COUNTS = {
    "knowledge_blocks": 15,
    "knowledge_subtopics": 73,
    "knowledge_docs": 68,
    "knowledge_chunks": 1,
    "learning_resources": 1,
    "block_resources": 15,
    "subtopic_resources": 73,
}


def expected_question_count() -> int:
    seed_path = ROOT_DIR / "data" / "seeds" / "test_questions.json"
    try:
        with seed_path.open("r", encoding="utf-8") as handle:
            questions = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read question seed: {exc}")
    if not isinstance(questions, list) or not questions:
        fail("test_questions seed is empty or malformed")
    return len(questions)


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
        question_count = conn.execute("SELECT COUNT(*) FROM test_questions").fetchone()[0]
        minimum_questions = expected_question_count()
        if question_count < minimum_questions:
            fail(f"test_questions: expected at least {minimum_questions}, found {question_count}")
        print(f"[PASS] test_questions: {question_count} (seed: {minimum_questions})")
    except sqlite3.Error as exc:
        fail(f"database query failed: {exc}")
    finally:
        conn.close()
    print("[PASS] integrity_check")
    print("[PASS] foreign_key_check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
