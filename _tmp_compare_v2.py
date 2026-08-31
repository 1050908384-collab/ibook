import sqlite3
import json

DB_PATH = r'd:\我的项目\揭榜挂帅\ai_agent-trainer\data\sqlite_data\project.db'
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

linux_sids = [
    '51197dbb-1d3c-45f3-8b3e-769628af1ec8',
    '838db607-41bf-4eae-a6a1-34d4a777f6fc'
]

results = {}
for sid in linux_sids:
    data = {}
    print(f'\n{"="*80}')
    print(f'SESSION: {sid}')
    print(f'{"="*80}')
    
    # session info
    cursor.execute("SELECT session_id, user_id, goal, current_stage, status, target_daily_hours, target_total_hours FROM sessions WHERE session_id=?", (sid,))
    s = cursor.fetchone()
    data['session'] = {
        'session_id': s[0], 'user_id': s[1], 'goal': s[2],
        'current_stage': s[3], 'status': s[4],
        'target_daily_hours': s[5], 'target_total_hours': s[6]
    }
    print(f"  goal={s[2]}  user={s[1][:18]}...  current_stage={s[3]}  status={s[4]}")
    print(f"  daily_hours={s[5]} total_hours={s[6]}")
    
    # session_states.state_json - this is where learner_profile & learning_package live
    cursor.execute("SELECT state_json FROM session_states WHERE session_id=?", (sid,))
    row = cursor.fetchone()
    if row and row[0]:
        try:
            state = json.loads(row[0])
            data['state_keys'] = list(state.keys())
            print(f'\n  session_states keys: {list(state.keys())}')
            
            # Print learner_profile summary
            if 'learner_profile' in state:
                lp = state['learner_profile']
                data['learner_profile'] = lp
                print(f'\n  --- LEARNER_PROFILE (from state_json) ---')
                for k, v in lp.items():
                    vstr = str(v)[:500]
                    print(f'    {k}: {vstr}')
            elif 'profile' in state:
                lp = state['profile']
                data['learner_profile'] = lp
                print(f'\n  --- LEARNER_PROFILE (state["profile"]) ---')
                for k, v in lp.items():
                    vstr = str(v)[:500]
                    print(f'    {k}: {vstr}')
            else:
                print(f'\n  ! NO learner_profile / profile key in state_json')
                # Dump top-level string values to understand layout
                for k, v in state.items():
                    if isinstance(v, (str, int, float, bool)):
                        print(f'    state.{k} = {str(v)[:200]}')
                    elif isinstance(v, dict):
                        print(f'    state.{k} keys = {list(v.keys())[:30]}')
                    elif isinstance(v, list):
                        print(f'    state.{k} list len={len(v)} sample={str(v[:2])[:200]}')
            
            # Find learning_package or tutorial content
            print(f'\n  --- LOOKING FOR LEARNING PACKAGE ---')
            for lk in ['learning_package', 'tutorial', 'package', 'stage_1_package', 'materials']:
                if lk in state:
                    v = state[lk]
                    if isinstance(v, str):
                        data['package_preview_' + lk] = v[:800]
                        print(f'    state["{lk}"] (len={len(v)}) PREVIEW:')
                        print(v[:800])
                    elif isinstance(v, dict):
                        print(f'    state["{lk}"] keys: {list(v.keys())}')
                        # Try known sub-keys
                        for sk in ['tutorial', 'content', 'md', 'markdown', 'main_tutorial', 'learning_package.md']:
                            if sk in v:
                                sub = v[sk]
                                ps = sub[:800] if isinstance(sub, str) else str(sub)[:500]
                                data['pkg_dict_' + sk] = ps
                                print(f'      ["{sk}"] (len={len(sub) if isinstance(sub,str) else "?"}): {ps}')
                    elif isinstance(v, list):
                        print(f'    state["{lk}"] list len={len(v)}')
                        for i, item in enumerate(v[:2]):
                            print(f'      [{i}] type={type(item).__name__} val={str(item)[:300]}')
            
            # Stages
            if 'stages' in state:
                stages = state['stages']
                print(f'\n  --- STAGES ({len(stages)} stages) ---')
                for i, st in enumerate(stages):
                    stage_keys = list(st.keys()) if isinstance(st, dict) else []
                    print(f'    stage[{i}] keys={stage_keys[:20]}')
                    if isinstance(st, dict):
                        for sk in ['stage_num', 'block_id', 'block_name', 'subtopic_ids', 'package', 'tutorial']:
                            if sk in st:
                                sv = st[sk]
                                if isinstance(sv, str) and len(sv) < 500:
                                    print(f'      {sk}: {sv}')
                                elif isinstance(sv, str):
                                    print(f'      {sk}: len={len(sv)} preview={sv[:300]}')
                                elif isinstance(sv, list):
                                    print(f'      {sk}: {sv[:10]}')
                                else:
                                    print(f'      {sk}: {str(sv)[:300]}')
        except json.JSONDecodeError as e:
            print(f'  JSON DECODE ERROR: {e}')
            print(f'  raw[:500]: {row[0][:500]}')
    
    # stage_progress
    cursor.execute("SELECT stage_id, stage_num, block_id, block_name, status, subtopic_id, estimated_hours, strategy_used FROM stage_progress WHERE session_id=? ORDER BY stage_num", (sid,))
    print(f'\n  --- STAGE_PROGRESS ---')
    for sp in cursor.fetchall():
        print(f'    stage#{sp[1]}: block={sp[2]}({sp[3]}) status={sp[4]} sub={sp[5]} hrs={sp[6]} strategy={sp[7]}')
    
    results[sid] = data

conn.close()

# Now do a comparison
print(f'\n\n{"#"*80}')
print(f'COMPARISON: LEARNER PROFILE DIFFERENCES')
print(f'{"#"*80}')
s1, s2 = linux_sids
lp1 = results[s1].get('learner_profile') or {}
lp2 = results[s2].get('learner_profile') or {}

all_keys = set(list(lp1.keys()) + list(lp2.keys()))
for k in sorted(all_keys):
    v1 = lp1.get(k)
    v2 = lp2.get(k)
    same = (v1 == v2)
    marker = "  SAME" if same else "  DIFF"
    s1v = str(v1)[:200]
    s2v = str(v2)[:200]
    print(f'\n[{k}]{marker}')
    print(f'  Session1: {s1v}')
    print(f'  Session2: {s2v}')

# Compare learning_package if we have any string previews
print(f'\n\n{"#"*80}')
print(f'COMPARISON: LEARNING PACKAGE DIFFERENCES')
print(f'{"#"*80}')
pk_keys = set()
for sid in linux_sids:
    for k in results[sid].keys():
        if k.startswith('package_') or k.startswith('pkg_'):
            pk_keys.add(k)
for k in sorted(pk_keys):
    v1 = results[s1].get(k, '')
    v2 = results[s2].get(k, '')
    same = (v1 == v2)
    print(f'\n[{k}] {"SAME" if same else "DIFFERENT"} (len1={len(v1)} len2={len(v2)})')
    if not same:
        # find first diff position
        minlen = min(len(v1), len(v2))
        diffpos = 0
        while diffpos < minlen and v1[diffpos] == v2[diffpos]:
            diffpos += 1
        print(f'  First diff at pos {diffpos}')
        print(f'  S1 around diff: ...{v1[max(0,diffpos-40):diffpos+40]}...')
        print(f'  S2 around diff: ...{v2[max(0,diffpos-40):diffpos+40]}...')
