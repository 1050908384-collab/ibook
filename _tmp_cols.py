import sqlite3
import json

DB_PATH = r'd:\我的项目\揭榜挂帅\ai_agent-trainer\data\sqlite_data\project.db'
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# First get sessions columns
cursor.execute("PRAGMA table_info(sessions)")
print('SESSIONS COLS:', [c[1] for c in cursor.fetchall()])

cursor.execute("PRAGMA table_info(session_states)")
print('SESSION_STATES COLS:', [c[1] for c in cursor.fetchall()])

cursor.execute("PRAGMA table_info(learner_profiles)")
print('LEARNER_PROFILES COLS:', [c[1] for c in cursor.fetchall()])

cursor.execute("PRAGMA table_info(learning_package_sources)")
print('LEARNING_PACKAGE_SOURCES COLS:', [c[1] for c in cursor.fetchall()])

cursor.execute("PRAGMA table_info(learning_resources)")
print('LEARNING_RESOURCES COLS:', [c[1] for c in cursor.fetchall()])

conn.close()
