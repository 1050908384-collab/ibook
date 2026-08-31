import json
from pathlib import Path
ROOT = Path(r"d:\我的项目\揭榜挂帅\ai_agent-trainer")
RT = ROOT / "测试数据与评测材料" / "02_50组个性化主评测" / "raw_outputs" / "case_001" / "stage_1" / "review_trace.json"
rt = json.loads(RT.read_text(encoding="utf-8"))
print("top keys:", list(rt.keys()))
rh = rt.get("review_history", [])
print(f"review_history len={len(rh)}")
for i, it in enumerate(rh):
    print(f"round {i} keys:", list(it.keys()))
    found = None
    for k in ("issues","errors","review_issues","found_issues","items","reviews"):
        v = it.get(k)
        if isinstance(v, list) and v:
            found = (k, v)
            break
        if isinstance(v, list):
            found = (k, v)
            break
    if found:
        kname, lst = found
        print(f"  -> via '{kname}': list len={len(lst)}")
        if lst and isinstance(lst[0], dict):
            print(f"     0th issue keys:", list(lst[0].keys())[:10])
            for kk in list(lst[0].keys())[:5]:
                vv = lst[0][kk]
                if isinstance(vv, str) and len(vv) > 120: vv = vv[:120] + "..."
                print(f"       {kk}: {vv}")
    else:
        print("  -> no issues list-like key found")
        print(f"     sample items (str/int keys):", [(k, str(v)[:80]) for k, v in it.items() if k not in ("review_passed","package_status","needs_human_review","unresolved_issue_count","unresolved_must_fix_count","round")])
    flags = {k: it.get(k) for k in ("review_passed","unresolved_issue_count","unresolved_must_fix_count","package_status","round","iteration")}
    print(f"  flags: {flags}")
    print()
