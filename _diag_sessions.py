import sys as _sys_py
from pathlib import Path as _Path_py
_PROJ_ROOT_ = _Path_py(__file__).resolve().parent.parent
_sys_py.path.insert(0, str(_PROJ_ROOT_))
_sys_py.path.insert(0, str(_PROJ_ROOT_ / "backend"))

import sqlite3, json
c = sqlite3.connect("data/sqlite_data/project.db")
# 列出所有表
tbls = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()]
print("tables:", tbls)

# 查 sessions
print("\n[ sessions ]")
for r in c.execute("SELECT session_id, goal, status, current_stage FROM sessions ORDER BY created_at DESC LIMIT 10").fetchall():
    print(f"  sid={r[0][:16]}…  status={r[2]}  stage={r[3]}  goal={str(r[1])[:100]}")
print("\n[ session_states 条数 ]:", c.execute("SELECT COUNT(*) FROM session_states").fetchone()[0])
