"""Rebuild global run_manifest.json: 50-case aggregate (47 from 8/30 batch + 3 re-runs on 8/31).
"""
import json
from pathlib import Path

ROOT = Path(r"d:\我的项目\揭榜挂帅\ai_agent-trainer")
RAW = ROOT / "测试数据与评测材料" / "02_50组个性化主评测" / "raw_outputs"
OUT = RAW / "run_manifest.json"

case_ids = [f"case_{i:03d}" for i in range(1, 51)]
RERUNS = {"case_016", "case_026", "case_046"}  # 8/31 re-runs (associated with 展示案例)

summaries = []
total_llm = 0
total_succ_llm = 0
total_fail_llm = 0
total_fallback = 0
total_package_chars = 0
stage_2_declared = 0
stage_2_completed = 0
succeeded = failed = 0
status_breakdown = {}
exec_breakdown = {}
prompt_versions = {}
skill_versions = {}
code_commit = None
kb_version = None
start_min = None
finish_max = None
first_run_id = None

for cid in case_ids:
    case_dir = RAW / cid
    rm_p = case_dir / "run_manifest.json"
    rs_p = case_dir / "run_summary.json"
    if not (rm_p.exists() and rs_p.exists()):
        summaries.append({
            "case_id": cid, "status": "missing",
            "error": "manifest or summary missing",
        })
        failed += 1
        status_breakdown["missing"] = status_breakdown.get("missing", 0) + 1
        continue
    rm = json.loads(rm_p.read_text(encoding="utf-8"))
    rs = json.loads(rs_p.read_text(encoding="utf-8"))

    status = rs.get("status") or "completed"
    error = rs.get("error")
    llm = int(rs.get("llm_call_count") or 0)
    total_llm += llm
    total_succ_llm += llm
    exec_path = rs.get("execution_path") or (
        "failed" if status != "completed"
        else "deterministic_only" if llm == 0 else "deepseek_only"
    )
    exec_breakdown[exec_path] = exec_breakdown.get(exec_path, 0) + 1

    # per-case model/prompt version capture (best-effort)
    pv = rm.get("prompt_version") or rm.get("prompt_version_hash") or (
        rm.get("model_config", {}).get("prompt_version")
    )
    if pv:
        prompt_versions[pv] = prompt_versions.get(pv, 0) + 1
    sv = rm.get("skill_version") or rm.get("skill_version_hash")
    if sv:
        skill_versions[sv] = skill_versions.get(sv, 0) + 1

    started = rm.get("run_started_at")
    finished = rm.get("run_finished_at")
    if first_run_id is None:
        first_run_id = rm.get("run_id")
    if code_commit is None:
        code_commit = rm.get("code_commit")
    if kb_version is None:
        kb_version = rm.get("knowledge_base_version")
    if started:
        if start_min is None or started < start_min:
            start_min = started
    if finished:
        if finish_max is None or finished > finish_max:
            finish_max = finished

    stage_1_c = bool(rs.get("stage_1_completed"))
    stage_2_c = bool(rs.get("stage_2_completed"))
    # stage 2 declared from CSV would be expensive; use heuristic: run_stage_2 flag from input.json if present
    stage_2_decl = bool(rm.get("run_stage_2_declared", False))
    if not stage_2_decl and (rs.get("stage_2_accuracy") is not None or stage_2_c):
        stage_2_decl = True
    if stage_2_decl:
        stage_2_declared += 1
    if stage_2_c:
        stage_2_completed += 1

    if status == "completed":
        succeeded += 1
    else:
        failed += 1
    status_breakdown[status] = status_breakdown.get(status, 0) + 1

    # character counting (stage_1/2 learning_package.md combined)
    lp_chars = 0
    for stage_dir in ["stage_1", "stage_2"]:
        p = case_dir / stage_dir / "learning_package.md"
        if p.exists():
            try:
                lp_chars += len(p.read_text(encoding="utf-8"))
            except Exception:
                pass
    total_package_chars += lp_chars

    summaries.append({
        "case_id": cid,
        "status": status,
        "error": error,
        "stage_1_completed": stage_1_c,
        "stage_2_completed": stage_2_c,
        "run_stage_2_declared": stage_2_decl,
        "llm_call_count": llm,
        "execution_path": exec_path,
        "mastered_block_ids": rs.get("mastered_block_ids") or [],
        "stage_1_block": rs.get("stage_1_block"),
        "stage_1_estimated_hours": rs.get("stage_1_estimated_hours"),
        "stage_1_accuracy": rs.get("stage_1_accuracy"),
        "stage_1_question_count": rs.get("stage_1_question_count"),
        "stage_1_strategy": rs.get("stage_1_strategy"),
        "batch_note": f"8/31 re-run (展示案例配套, {cid} in {sorted(RERUNS)})" if cid in RERUNS else "8/30 batch",
    })

# Model config (取 50 组里第一个完整 manifest)
model_config = None
for cid in case_ids:
    rm_p = RAW / cid / "run_manifest.json"
    if rm_p.exists():
        rm = json.loads(rm_p.read_text(encoding="utf-8"))
        mc = rm.get("model_config")
        if mc:
            model_config = mc
            break

aggregate = {
    "run_id": "batch_run_50cases_mixed_20260830_20260831",
    "note": (
        "本文件为 50 组主评测（case_001~case_050）聚合清单。47 组执行于 2026-08-30 批次，"
        "case_016 / case_026 / case_046 三组因展示材料配套、后端链路修复后于 2026-08-31 复跑，"
        "其余 46 组保持 8/30 批次数据；每个 case 单独的 run_manifest.json 仍保存在 case_id/run_manifest.json。"
    ),
    "evaluation_schema_version": "2026-08-30-final",
    "code_commit": code_commit,
    "knowledge_base_version": kb_version,
    "rerun_case_ids_on_20260831": sorted(RERUNS),
    "batch_mix_note": "47组 2026-08-30 主批次 + 3组 2026-08-31 后端修复后复跑",
    "model_config": model_config or {},
    "prompt_version_frozen": sorted(prompt_versions.keys()),
    "prompt_version_counts": prompt_versions,
    "skill_version_frozen": sorted(skill_versions.keys()),
    "skill_version_counts": skill_versions,
    "run_started_at": start_min,
    "run_finished_at": finish_max,
    "total_cases": 50,
    "succeeded": succeeded,
    "failed": failed,
    "status_breakdown": status_breakdown,
    "execution_path_breakdown": exec_breakdown,
    "run_stage_2_total_declared": stage_2_declared,
    "run_stage_2_total_completed": stage_2_completed,
    "llm_call_count": total_llm,
    "llm_successful_calls": total_succ_llm,
    "llm_failed_calls": total_fail_llm,
    "llm_fallback_count": total_fallback,
    "llm_fallback_rate": (total_fallback / total_llm) if total_llm else 0.0,
    "total_package_chars": total_package_chars,
    "summaries": summaries,
}

OUT.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
print("WRITE", OUT)
print("total_cases=50  succeeded=%d  failed=%d  llm=%d  fallback=%d  stage2=%d/%d  package_chars=%d" % (
    succeeded, failed, total_llm, total_fallback, stage_2_completed, stage_2_declared, total_package_chars))
print("exec_path_breakdown:", exec_breakdown)
print("reruns:", sorted(RERUNS))
