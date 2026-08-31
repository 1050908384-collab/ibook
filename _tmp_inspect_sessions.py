import sqlite3
import json
import sys

DB_PATH = r'd:\我的项目\揭榜挂帅\ai_agent-trainer\data\sqlite_data\project.db'
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# List all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
tables = cursor.fetchall()
print('=== ALL TABLES ===')
for t in tables:
    print(' -', t[0])
print()

# Check sessions with learning goal (KB-01 Linux)
print('=== SESSIONS (targeting Linux KB-01 or goal mentions Linux) ===')
cursor.execute("SELECT session_id, user_id, goal, created_at FROM sessions ORDER BY created_at DESC")
rows = cursor.fetchall()
linux_sessions = []
for r in rows:
    sid, uid, goal, cat = r
    goal_str = (goal or '')
    print(f'  [{sid[:8]}..] user={uid} goal={goal_str[:60]} created={cat}')
    if 'linux' in goal_str.lower() or 'linux' in (goal or '').lower():
        linux_sessions.append(sid)
print()
print(f'Potential Linux sessions: {linux_sessions}')

# Now look for stage_progress / learning_package table to find content
print()
print('=== CHECK learning_package TABLE ===')
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%package%'")
print([r[0] for r in cursor.fetchall()])

cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%stage%'")
print('Stage-like tables:', [r[0] for r in cursor.fetchall()])

# stage_progress
print()
print('=== STAGE_PROGRESS COLUMNS ===')
cursor.execute("PRAGMA table_info(stage_progress)")
for col in cursor.fetchall():
    print(' ', col)

# Find learning packages from stage_progress or material_cache or similar
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%material%' OR name LIKE '%cache%' OR name LIKE '%content%'")
print('Content-like tables:', [r[0] for r in cursor.fetchall()])

conn.close()
