import sqlite3
import json

DB_PATH = r'd:\我的项目\揭榜挂帅\ai_agent-trainer\data\sqlite_data\project.db'
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

linux_sids = [
    '51197dbb-1d3c-45f3-8b3e-769628af1ec8',   # 15:30
    '838db607-41bf-4eae-a6a1-34d4a777f6fc'    # 15:29
]

# Fetch full state_json for each, dump full profile + package sections to individual files for diff
for i, sid in enumerate(linux_sids):
    cursor.execute("SELECT state_json FROM session_states WHERE session_id=?", (sid,))
    row = cursor.fetchone()
    if not row or not row[0]:
        print(f'Session {sid}: NO state_json')
        continue
    state = json.loads(row[0])
    
    # Write profile
    prof = state.get('profile')
    lp_path = f'd:\\我的项目\\揭榜挂帅\\ai_agent-trainer\\tmp\\session{i+1}_profile.json'
    with open(lp_path, 'w', encoding='utf-8') as f:
        json.dump(prof, f, ensure_ascii=False, indent=2)
    print(f'Written: {lp_path}')
    
    # Write learning_package
    lpkg = state.get('learning_package')
    lp_path2 = f'd:\\我的项目\\揭榜挂帅\\ai_agent-trainer\\tmp\\session{i+1}_learning_package.json'
    if isinstance(lpkg, (dict, list)):
        with open(lp_path2, 'w', encoding='utf-8') as f:
            json.dump(lpkg, f, ensure_ascii=False, indent=2)
    else:
        with open(lp_path2, 'w', encoding='utf-8') as f:
            f.write(str(lpkg or ''))
    print(f'Written: {lp_path2}')
    
    # Also dump stage_info
    st_info = state.get('stage_info')
    si_path = f'd:\\我的项目\\揭榜挂帅\\ai_agent-trainer\\tmp\\session{i+1}_stage_info.json'
    with open(si_path, 'w', encoding='utf-8') as f:
        json.dump(st_info, f, ensure_ascii=False, indent=2)
    print(f'Written: {si_path}')
    
    print(f'  profile_confirmed={state.get("profile_confirmed")}')
    print(f'  profile_modified={state.get("profile_modified")}')
    print(f'  phase={state.get("phase")}')
    print(f'  current_stage={state.get("current_stage")}')
    print(f'  status={state.get("status")}')
    print()

conn.close()
print('Done. All files in tmp/ folder.')
