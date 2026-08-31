import json, csv
from pathlib import Path
ROOT = Path(r"d:\我的项目\揭榜挂帅\ai_agent-trainer")
RAW = ROOT / "测试数据与评测材料" / "02_50组个性化主评测" / "raw_outputs"
CSV_PATH = ROOT / "测试数据与评测材料" / "02_50组个性化主评测" / "evaluation_cases.csv"

# 1. stage 2 declaration from CSV (single source of truth)
decl = {}
with CSV_PATH.open(encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        decl[r["case_id"]] = str(r.get("run_stage_2", "False")).strip().lower() == "true"
stage2_true = sorted([c for c, v in decl.items() if v])
print("CSV run_stage_2=True count:", len(stage2_true), stage2_true)

# 2. per-case learning package chars + summary
pkg_chars = {}
for cid in [f"case_{i:03d}" for i in range(1, 51)]:
    d = RAW / cid
    n = 0
    for stage in ("stage_1", "stage_2"):
        p = d / stage / "learning_package.md"
        if p.exists():
            n += len(p.read_text(encoding="utf-8"))
    pkg_chars[cid] = n
print("total_package_chars =", sum(pkg_chars.values()))
short = sorted(pkg_chars.items(), key=lambda x: x[1])[:10]
long_ = sorted(pkg_chars.items(), key=lambda x: x[1])[-10:]
print("\nshortest 10:")
for cid, n in short:
    print(f"  {cid}: {n}")
print("\nlongest 10:")
for cid, n in long_:
    print(f"  {cid}: {n}")
print("\nre-run 3 LP chars:")
for cid in ("case_016", "case_026", "case_046"):
    p1 = RAW / cid / "stage_1" / "learning_package.md"
    n1 = len(p1.read_text(encoding="utf-8")) if p1.exists() else 0
    p2 = RAW / cid / "stage_2" / "learning_package.md"
    n2 = len(p2.read_text(encoding="utf-8")) if p2.exists() else 0
    print(f"  {cid}: stage1={n1} stage2={n2} total={n1+n2}")
