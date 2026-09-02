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

# ---------- 路径设置：python scripts\xxx.py 直接运行时能正确找到 backend / data ----------
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
import uuid
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DB_PATH = ROOT_DIR / "data" / "sqlite_data" / "project.db"

EXPECTED_TABLES = [
    "review_traces",
    "learning_package_sources",
    "learning_package_claims",
    "review_debate_issues",
    "review_debate_rounds",
    "review_debate_responses",
]

EXPECTED_COLUMNS = {
    "review_traces": {
        "trace_id",
        "session_id",
        "stage_id",
        "target_type",
        "package_version",
        "package_status",
        "review_mode",
        "unresolved_issue_count",
        "unresolved_must_fix_count",
        "final_status",
        "final_summary",
        "created_at",
        "updated_at",
        "closed_at",
    },
    "learning_package_sources": {
        "trace_id",
        "source_id",
        "source_type",
        "title",
        "locator",
        "content_ref",
        "source_version",
        "metadata_json",
        "created_at",
        "updated_at",
    },
    "learning_package_claims": {
        "trace_id",
        "claim_id",
        "text",
        "source_ref",
        "source_span",
        "section_ref",
        "claim_type",
        "is_key_technical",
        "confidence",
        "status",
        "created_at",
        "updated_at",
    },
    "review_debate_issues": {
        "issue_id",
        "trace_id",
        "canonical_error_id",
        "error_id",
        "claim_id",
        "error_type",
        "severity",
        "must_fix",
        "must_fix_reason",
        "description",
        "location",
        "evidence",
        "suggestion",
        "source_refs",
        "change_request",
        "current_status",
        "final_disposition",
        "opened_in_round_num",
        "resolved_in_round_num",
        "created_at",
        "updated_at",
    },
    "review_debate_rounds": {
        "round_id",
        "trace_id",
        "round_num",
        "review_iteration",
        "round_status",
        "orchestrator_verdict",
        "verdict_summary",
        "created_at",
        "closed_at",
    },
    "review_debate_responses": {
        "response_id",
        "round_id",
        "issue_id",
        "actor",
        "response_type",
        "disposition",
        "rationale",
        "source_refs",
        "patched_sections",
        "created_at",
    },
}


class CheckFailure(Exception):
    pass


def fetch_table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {row[0] for row in rows}


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {row[1] for row in rows}


def foreign_keys(conn: sqlite3.Connection, table: str) -> list[tuple]:
    return conn.execute(f"PRAGMA foreign_key_list({table})").fetchall()


def indexes(conn: sqlite3.Connection, table: str) -> list[tuple]:
    return conn.execute(f"PRAGMA index_list({table})").fetchall()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailure(message)


def check_tables_exist(conn: sqlite3.Connection) -> None:
    actual = fetch_table_names(conn)
    missing = [name for name in EXPECTED_TABLES if name not in actual]
    require(not missing, f"missing tables: {missing}")


def check_columns(conn: sqlite3.Connection) -> None:
    for table, expected in EXPECTED_COLUMNS.items():
        actual = table_columns(conn, table)
        missing = sorted(expected - actual)
        require(not missing, f"{table} missing columns: {missing}")


def check_fk_targets(conn: sqlite3.Connection) -> None:
    expected = {
        "review_traces": {("stage_id", "stage_progress"), ("session_id", "sessions")},
        "learning_package_sources": {("trace_id", "review_traces")},
        "learning_package_claims": {
            ("trace_id", "review_traces"),
            ("trace_id/source_ref", "learning_package_sources"),
        },
        "review_debate_issues": {
            ("trace_id", "review_traces"),
            ("trace_id/claim_id", "learning_package_claims"),
        },
        "review_debate_rounds": {("trace_id", "review_traces")},
        "review_debate_responses": {
            ("issue_id", "review_debate_issues"),
            ("round_id", "review_debate_rounds"),
        },
    }

    for table, table_expected in expected.items():
        rows = foreign_keys(conn, table)
        refs = {}
        for fk_id, seq, ref_table, from_col, to_col, *_ in rows:
            require("legacy_migration" not in ref_table, f"{table} references legacy table {ref_table}")
            refs.setdefault((fk_id, ref_table), []).append((seq, from_col, to_col))

        normalized = set()
        for (_, ref_table), items in refs.items():
            items = sorted(items)
            from_cols = "/".join(item[1] for item in items)
            normalized.add((from_cols, ref_table))

        missing = table_expected - normalized
        require(not missing, f"{table} foreign keys mismatch: missing {sorted(missing)}")


def check_composite_fk_shape(conn: sqlite3.Connection) -> None:
    composite_expectations = {
        "learning_package_claims": ("learning_package_sources", ["trace_id", "source_ref"], ["trace_id", "source_id"]),
        "review_debate_issues": ("learning_package_claims", ["trace_id", "claim_id"], ["trace_id", "claim_id"]),
    }
    for table, (ref_table, from_cols, to_cols) in composite_expectations.items():
        groups = {}
        for row in foreign_keys(conn, table):
            fk_id, seq, target_table, from_col, to_col, *_ = row
            if target_table != ref_table:
                continue
            groups.setdefault(fk_id, []).append((seq, from_col, to_col))
        matched = False
        for items in groups.values():
            items = sorted(items)
            if [item[1] for item in items] == from_cols and [item[2] for item in items] == to_cols:
                matched = True
                break
        require(matched, f"{table} composite FK to {ref_table} is not defined correctly")


def check_unique_indexes(conn: sqlite3.Connection) -> None:
    expected = {
        "review_debate_issues": {"trace_id", "canonical_error_id"},
        "review_debate_rounds": {"trace_id", "round_num"},
        "review_debate_responses": {"round_id", "issue_id", "actor", "response_type"},
    }
    for table, expected_cols in expected.items():
        found = False
        for _, index_name, unique, *_ in indexes(conn, table):
            if unique != 1:
                continue
            cols = {
                row[2]
                for row in conn.execute(f"PRAGMA index_info({index_name})").fetchall()
            }
            if cols == expected_cols:
                found = True
                break
        require(found, f"{table} missing unique index for {sorted(expected_cols)}")


def check_runtime_constraints(conn: sqlite3.Connection) -> None:
    suffix = uuid.uuid4().hex[:8]
    session_id = f"schema_check_session_{suffix}"
    stage_id = f"schema_check_stage_{suffix}"
    trace_id = f"schema_check_trace_{suffix}"
    source_id = "src_check"
    claim_id = "clm_check"
    issue_id = f"schema_check_issue_{suffix}"
    round_id = f"schema_check_round_{suffix}"
    response_id = f"schema_check_response_{suffix}"

    conn.execute("BEGIN")
    try:
        conn.execute(
            "INSERT INTO sessions (session_id, current_stage, status) VALUES (?, 1, 'active')",
            (session_id,),
        )
        conn.execute(
            """
            INSERT INTO stage_progress (stage_id, session_id, stage_num, block_id, block_name, status)
            VALUES (?, ?, 1, 'SCHEMA-CHECK', 'Schema Check', 'pending')
            """,
            (stage_id, session_id),
        )
        conn.execute(
            """
            INSERT INTO review_traces (
                trace_id, session_id, stage_id, target_type, package_status, review_mode, final_status
            ) VALUES (?, ?, ?, 'learning_package', 'draft', 'debate', 'in_progress')
            """,
            (trace_id, session_id, stage_id),
        )
        conn.execute(
            """
            INSERT INTO learning_package_sources (
                trace_id, source_id, source_type, title
            ) VALUES (?, ?, 'knowledge_doc', 'Schema Check Source')
            """,
            (trace_id, source_id),
        )
        conn.execute(
            """
            INSERT INTO learning_package_claims (
                trace_id, claim_id, text, source_ref, section_ref, claim_type, is_key_technical, confidence, status
            ) VALUES (?, ?, 'ok', ?, 'tutorial:sec_1', 'concept', 1, 0.9, 'active')
            """,
            (trace_id, claim_id, source_id),
        )
        conn.execute(
            """
            INSERT INTO review_debate_issues (
                issue_id, trace_id, canonical_error_id, error_id, claim_id, error_type, current_status
            ) VALUES (?, ?, 'canon-check', 'canon-check_r1', ?, 'fact_error', 'open')
            """,
            (issue_id, trace_id, claim_id),
        )
        conn.execute(
            """
            INSERT INTO review_debate_rounds (
                round_id, trace_id, round_num, round_status, orchestrator_verdict
            ) VALUES (?, ?, 1, 'open', 'pending')
            """,
            (round_id, trace_id),
        )
        conn.execute(
            """
            INSERT INTO review_debate_responses (
                response_id, round_id, issue_id, actor, response_type, disposition
            ) VALUES (?, ?, ?, 'agent3', 'author_reply', 'accepted')
            """,
            (response_id, round_id, issue_id),
        )

        try:
            conn.execute(
                """
                INSERT INTO learning_package_claims (
                    trace_id, claim_id, text, section_ref, claim_type, is_key_technical, confidence, status
                ) VALUES (?, 'clm_bad_source', 'bad', 'tutorial:sec_2', 'concept', 1, 0.9, 'active')
                """,
                (trace_id,),
            )
            raise CheckFailure("key technical claim without source_ref was not rejected")
        except sqlite3.IntegrityError:
            pass

        try:
            conn.execute(
                """
                INSERT INTO learning_package_claims (
                    trace_id, claim_id, text, source_ref, section_ref, claim_type, is_key_technical, confidence, status
                ) VALUES (?, 'clm_bad_conf', 'bad', ?, 'tutorial:sec_2', 'concept', 0, 1.2, 'active')
                """,
                (trace_id, source_id),
            )
            raise CheckFailure("confidence > 1 was not rejected")
        except sqlite3.IntegrityError:
            pass

        try:
            conn.execute(
                """
                INSERT INTO review_debate_issues (
                    issue_id, trace_id, canonical_error_id, error_id, claim_id, error_type, current_status
                ) VALUES ('dup_issue', ?, 'canon-check', 'canon-check_r2', ?, 'fact_error', 'open')
                """,
                (trace_id, claim_id),
            )
            raise CheckFailure("duplicate canonical_error_id in same trace was not rejected")
        except sqlite3.IntegrityError:
            pass

        try:
            conn.execute(
                """
                INSERT INTO review_debate_rounds (
                    round_id, trace_id, round_num, round_status, orchestrator_verdict
                ) VALUES ('dup_round', ?, 1, 'open', 'pending')
                """,
                (trace_id,),
            )
            raise CheckFailure("duplicate round_num in same trace was not rejected")
        except sqlite3.IntegrityError:
            pass

        other_trace = f"schema_check_trace_other_{suffix}"
        conn.execute(
            """
            INSERT INTO review_traces (
                trace_id, session_id, stage_id, target_type, package_status, review_mode, final_status
            ) VALUES (?, ?, ?, 'learning_package', 'draft', 'debate', 'in_progress')
            """,
            (other_trace, session_id, stage_id),
        )
        try:
            conn.execute(
                """
                INSERT INTO learning_package_claims (
                    trace_id, claim_id, text, source_ref, section_ref, claim_type, is_key_technical, confidence, status
                ) VALUES (?, 'clm_cross_trace', 'bad', ?, 'tutorial:sec_2', 'concept', 0, 0.8, 'active')
                """,
                (other_trace, source_id),
            )
            raise CheckFailure("cross-trace source_ref was not rejected by composite FK")
        except sqlite3.IntegrityError:
            pass
    finally:
        conn.rollback()


def main() -> int:
    if not DB_PATH.exists():
        print(f"[FAIL] database not found: {DB_PATH}")
        return 1

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    checks = [
        ("tables_exist", check_tables_exist),
        ("columns", check_columns),
        ("foreign_key_targets", check_fk_targets),
        ("composite_foreign_keys", check_composite_fk_shape),
        ("unique_indexes", check_unique_indexes),
        ("runtime_constraints", check_runtime_constraints),
    ]

    failed = False
    try:
        for name, fn in checks:
            try:
                fn(conn)
                print(f"[PASS] {name}")
            except CheckFailure as exc:
                failed = True
                print(f"[FAIL] {name}: {exc}")
    finally:
        conn.close()

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
