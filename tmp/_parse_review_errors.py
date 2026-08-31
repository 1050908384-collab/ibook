import json, ast, re
from pathlib import Path
ROOT = Path(r"d:\我的项目\揭榜挂帅\ai_agent-trainer")
RT = ROOT / "测试数据与评测材料" / "02_50组个性化主评测" / "raw_outputs" / "case_001" / "stage_1" / "review_trace.json"
rt = json.loads(RT.read_text(encoding="utf-8"))
rh = rt["review_history"]
# try parse technical string
it = rh[0]
tech_str = it["technical"]
print("technical type:", type(tech_str).__name__)
print("first 300 chars:", tech_str[:300])
try:
    tech_obj = ast.literal_eval(tech_str)
    print("ast.literal_eval OK -> keys:", list(tech_obj.keys()) if isinstance(tech_obj, dict) else "not dict")
    errs = tech_obj.get("errors") if isinstance(tech_obj, dict) else None
    print(f"errors len = {len(errs) if isinstance(errs, list) else 'N/A'}")
    if isinstance(errs, list) and errs:
        print("  0th err keys:", list(errs[0].keys())[:8] if isinstance(errs[0], dict) else errs[0])
except Exception as e:
    print("ast fails:", e)
# top level unresolved counts
print("\ntop-level unresolved:", rt.get("unresolved_issue_count"), "must_fix:", rt.get("unresolved_must_fix_count"))
print("iterations:", rt.get("iterations"), "passed:", rt.get("review_passed"))
