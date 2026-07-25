import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, BACKEND_DIR)

from backend.db.sqlite_client import SQLiteClient


def init_db():
    client = SQLiteClient()
    db_url = client.database_url
    print("[OK] Active SQLite schema initialized successfully")
    print(f"[INFO] Active database: {db_url}")


if __name__ == "__main__":
    init_db()
