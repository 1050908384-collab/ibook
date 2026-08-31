import sqlite3
import json

DB_PATH = r'd:\我的项目\揭榜挂帅\ai_agent-trainer\data\sqlite_data\project.db'
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

linux_sids = [
    '51197dbb-1d3c-45f3-8b3e-769628af1ec8',
    '838db607-41bf-4eae-a6a1-34d4a777f6fc'
]

for sid in linux_sids:
    print(f'\n{"="*80}')
    print(f'SESSION: {sid}')
    print(f'{"="*80}')
    
    # session info
    cursor.execute("SELECT session_id, user_id, goal, created_at, profile_confirmed FROM sessions WHERE session_id=?", (sid,))
    s = cursor.fetchone()
    print(f'\n[SESSIONS] goal={s[2]}, created={s[3]}, profile_confirmed={s[4]}')
    
    # learner_profile
    print(f'\n--- LEARNER_PROFILE ---')
    cursor.execute("SELECT profile_id, session_id, skill_tags, mastered_block_ids, learning_style, hardware_info, daily_hours, background_text FROM learner_profiles WHERE session_id=?", (sid,))
    profs = cursor.fetchall()
    for p in profs:
        print(f'  profile_id={p[0]}')
        print(f'  skill_tags={p[2]}')
        print(f'  mastered_block_ids={p[3]}')
        print(f'  learning_style={p[4]}')
        print(f'  hardware_info={p[5]}')
        print(f'  daily_hours={p[6]}')
        print(f'  background_text={p[7]}')
    
    # session_states - likely stores learning_package JSON
    print(f'\n--- SESSION_STATES (schema preview) ---')
    cursor.execute("PRAGMA table_info(session_states)")
    for c in cursor.fetchall():
        print(f'  col={c}')
    
    cursor.execute("SELECT state_key, state_value FROM session_states WHERE session_id=? ORDER BY state_key", (sid,))
    for k, v in cursor.fetchall():
        print(f'\n  [state_key] = {k}')
        preview = (v or '')[:2000]
        print(f'  preview: {preview}')
    
    # stage_progress
    print(f'\n--- STAGE_PROGRESS ---')
    cursor.execute("SELECT stage_id, stage_num, block_id, block_name, status, subtopic_id, estimated_hours, strategy_used FROM stage_progress WHERE session_id=? ORDER BY stage_num", (sid,))
    for sp in cursor.fetchall():
        print(f'  stage#{sp[1]}: block={sp[2]}({sp[3]}) status={sp[4]} subtopic={sp[5]} hours={sp[6]} strategy={sp[7]}')
    
    # learning_package_sources
    print(f'\n--- LEARNING_PACKAGE_SOURCES ---')
    cursor.execute("PRAGMA table_info(learning_package_sources)")
    for c in cursor.fetchall():
        print(f'  col={c}')
    cursor.execute("SELECT source_id, stage_id, source_type, title, content FROM learning_package_sources ls JOIN stage_progress sp ON ls.stage_id=sp.stage_id WHERE sp.session_id=?", (sid,))
    for lp in cursor.fetchall():
        print(f'  source: type={lp[2]} title={lp[3]}')
        content = (lp[4] or '')[:1500]
        print(f'  content preview: {content}\n')

conn.close()
