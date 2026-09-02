
# ---------- 路径设置：python scripts\xxx.py 直接运行时能正确找到 backend / data ----------
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
