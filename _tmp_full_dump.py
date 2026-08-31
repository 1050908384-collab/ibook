import sqlite3
import json

DB_PATH = r'd:\我的项目\揭榜挂帅\ai_agent-trainer\data\sqlite_data\project.db'
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# 1. Learner_profiles by user_id from the sessions
cursor.execute("SELECT DISTINCT user_id FROM sessions WHERE session_id IN ('51197dbb-1d3c-45f3-8b3e-769628af1ec8','838db607-41bf-4eae-a6a1-34d4a777f6fc')")
uids = [r[0] for r in cursor.fetchall()]
print('User IDs from sessions:', uids)

for uid in uids:
    cursor.execute("SELECT * FROM learner_profiles WHERE user_id=?", (uid,))
    rows = cursor.fetchall()
    cursor.execute("PRAGMA table_info(learner_profiles)")
    cols = [c[1] for c in cursor.fetchall()]
    print(f'\n=== LEARNER_PROFILES for {uid} ===')
    if rows:
        for row in rows:
            for c, v in zip(cols, row):
                vs = str(v)[:500]
                print(f'  {c}: {vs}')
    else:
        print('  (NO ROW in learner_profiles)')

# 2. Dump first-level of state_json fully for both sessions
for sid in ['51197dbb-1d3c-45f3-8b3e-769628af1ec8', '838db607-41bf-4eae-a6a1-34d4a777f6fc']:
    cursor.execute("SELECT state_json FROM session_states WHERE session_id=?", (sid,))
    row = cursor.fetchone()
    state = json.loads(row[0])
    out_path = f'd:\\我的项目\\揭榜挂帅\\ai_agent-trainer\\tmp\\state_{sid[:8]}.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        # Full dump
        json.dump(state, f, ensure_ascii=False, indent=2)
    print(f'\nWritten full state: {out_path}')
    # Print type summary for each key
    print(f'\n{sid[:8]} summary:')
    for k in sorted(state.keys()):
        v = state[k]
        t = type(v).__name__
        if isinstance(v, dict):
            print(f'  {k}: dict with keys {list(v.keys())[:20]}')
        elif isinstance(v, list):
            print(f'  {k}: list len={len(v)}, types sample={[type(x).__name__ for x in v[:3]]}')
        elif isinstance(v, str):
            print(f'  {k}: str len={len(v)} sample="{v[:120]}"')
        else:
            print(f'  {k}: {t} = {str(v)[:200]}')

conn.close()
