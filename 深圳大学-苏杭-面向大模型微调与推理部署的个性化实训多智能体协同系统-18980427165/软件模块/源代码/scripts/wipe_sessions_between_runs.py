# -*- coding: utf-8 -*-
"""Wipe session-scoped data between isolated re-runs."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))
import os
os.chdir(str(BACKEND))
from db.sqlite_client import SQLiteClient
from sqlalchemy import text

TABLES = [
    "sessions",
    "user_profiles",
    "stage_progress",
    "user_qa",
    "session_states",
    "learning_packages",
]


def main():
    cli = SQLiteClient()
    with cli.engine.connect().execution_options(autocommit=False) as conn:
        conn.execute(text("PRAGMA foreign_keys = OFF"))
        for t in TABLES:
            try:
                r = conn.execute(text(f"DELETE FROM {t}"))
                print(f"{t}: deleted {r.rowcount}")
            except Exception as exc:
                print(f"{t}: SKIP ({type(exc).__name__}: {str(exc)[:100]})")
        conn.commit()
        conn.execute(text("PRAGMA foreign_keys = ON"))
        # Verify zero on those that actually exist
        zero = True
        for t in TABLES:
            try:
                c = conn.execute(text(f"SELECT COUNT(*) FROM {t}")).fetchone()
                n = c[0]
                print(f"REMAIN {t}: {n}")
                if n != 0:
                    zero = False
            except Exception:
                print(f"REMAIN {t}: N/A (table absent)")
    cli.close()
    if zero:
        print("OK_CLEAN")
        return 0
    print("FAIL_NOT_CLEAN")
    return 1


if __name__ == "__main__":
    sys.exit(main())
