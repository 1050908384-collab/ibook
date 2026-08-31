import json

p1 = r'd:\我的项目\揭榜挂帅\ai_agent-trainer\tmp\state_51197dbb.json'  # 15:30 - later created
p2 = r'd:\我的项目\揭榜挂帅\ai_agent-trainer\tmp\state_838db607.json'  # 15:29 - first created

with open(p1, encoding='utf-8') as f:
    s1 = json.load(f)
with open(p2, encoding='utf-8') as f:
    s2 = json.load(f)

print('=== LEARNING PACKAGE COMPARISON ===')
lp1 = s1.get('learning_package') or {}
lp2 = s2.get('learning_package') or {}
print(f'session1 (15:30) block={lp1.get("block_id")} stage_id={str(lp1.get("stage_id"))[:12]}...')
print(f'session2 (15:29) block={lp2.get("block_id")} stage_id={str(lp2.get("stage_id"))[:12]}...')

allkeys = set(list(lp1.keys()) + list(lp2.keys()))
for k in sorted(allkeys):
    v1 = lp1.get(k)
    v2 = lp2.get(k)
    if isinstance(v1, str) and isinstance(v2, str):
        same = (v1 == v2)
        print(f'\n  lp["{k}"]: len1={len(v1)} len2={len(v2)}  SAME? {same}')
        if not same:
            minlen = min(len(v1), len(v2))
            i = 0
            while i < minlen and v1[i] == v2[i]:
                i += 1
            print(f'    1st diff at pos {i}')
            print(f'    S1: ...{v1[max(0,i-50):i+150]}...')
            print(f'    S2: ...{v2[max(0,i-50):i+150]}...')
    elif isinstance(v1, list) and isinstance(v2, list):
        s1j = json.dumps(v1, ensure_ascii=False, sort_keys=True)
        s2j = json.dumps(v2, ensure_ascii=False, sort_keys=True)
        same = (s1j == s2j)
        print(f'\n  lp["{k}"]: list len1={len(v1)} len2={len(v2)}  SAME? {same}')
        if not same:
            print(f'    S1[:500]: {s1j[:500]}')
            print(f'    S2[:500]: {s2j[:500]}')
    elif isinstance(v1, dict) and isinstance(v2, dict):
        s1j = json.dumps(v1, ensure_ascii=False, sort_keys=True)
        s2j = json.dumps(v2, ensure_ascii=False, sort_keys=True)
        same = (s1j == s2j)
        print(f'\n  lp["{k}"]: dict keys1={list(v1.keys())[:15]}  SAME? {same}')
        if not same:
            print(f'    S1[:500]: {s1j[:500]}')
            print(f'    S2[:500]: {s2j[:500]}')
    else:
        same = (v1 == v2)
        print(f'\n  lp["{k}"]: same? {same}  v1={str(v1)[:200]}  v2={str(v2)[:200]}')

# Write tutorials out separately for long diff
tut1 = lp1.get('tutorial') or ''
tut2 = lp2.get('tutorial') or ''
with open(r'd:\我的项目\揭榜挂帅\ai_agent-trainer\tmp\s1_tutorial.md', 'w', encoding='utf-8') as f:
    f.write(tut1)
with open(r'd:\我的项目\揭榜挂帅\ai_agent-trainer\tmp\s2_tutorial.md', 'w', encoding='utf-8') as f:
    f.write(tut2)
print(f'\nTutorials written. s1_tutorial.md len={len(tut1)}  s2_tutorial.md len={len(tut2)}')

# Compare stage_info as well
print('\n\n=== STAGE_INFO COMPARISON ===')
si1 = s1.get('stage_info') or {}
si2 = s2.get('stage_info') or {}
for k in sorted(set(list(si1.keys()) + list(si2.keys()))):
    v1 = si1.get(k)
    v2 = si2.get(k)
    if isinstance(v1, (dict, list)):
        v1s = json.dumps(v1, ensure_ascii=False, sort_keys=True)
    else:
        v1s = str(v1)
    if isinstance(v2, (dict, list)):
        v2s = json.dumps(v2, ensure_ascii=False, sort_keys=True)
    else:
        v2s = str(v2)
    same = (v1s == v2s)
    print(f'  si["{k}"]: same? {same}')
    if not same:
        print(f'    S1: {v1s[:300]}')
        print(f'    S2: {v2s[:300]}')

# Dump interactions for each session - to understand what the user answered (profile differences)
print('\n\n=== INTERACTIONS COMPARISON (profile building phase) ===')
for label, s in [('S1_15:30', s1), ('S2_15:29', s2)]:
    print(f'\n--- {label}: {len(s.get("interactions", []))} interactions ---')
    for idx, ia in enumerate(s.get('interactions', [])):
        role = ia.get('role', '?')
        content = (ia.get('content') or '')
        if isinstance(content, str):
            c = content[:300].replace('\n', ' | ')
        else:
            c = json.dumps(content, ensure_ascii=False)[:300]
        print(f'  [{idx}] {role}: {c}')
