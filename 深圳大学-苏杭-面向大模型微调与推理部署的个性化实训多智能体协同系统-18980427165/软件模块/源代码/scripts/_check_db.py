
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

import sys
sys.path.insert(0, 'backend')
from db.sqlite_client import SQLiteClient
from sqlalchemy import text

db = SQLiteClient()
conn = db.engine.connect()

tables = ['knowledge_blocks', 'knowledge_subtopics', 'knowledge_docs', 'test_questions']
for table in tables:
    try:
        count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
        print(f"{table}: {count} rows")
    except Exception as e:
        print(f"{table}: ERROR - {e}")

conn.close()
db.close()
