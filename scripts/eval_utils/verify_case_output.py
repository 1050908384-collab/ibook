# -*- coding: utf-8 -*-
"""运行完好检查 —— 验证评测输出目录是否完整、四个核心文件是否齐全。

用法:
  python scripts/eval_utils/verify_case_output.py --case case_001
  python scripts/eval_utils/verify_case_output.py --dir raw_outputs
  python scripts/eval_utils/verify_case_output.py --dir raw_outputs_case001_final_20260830_115417
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = PROJECT_ROOT / "测试数据与评测材料" / "02_50组个性化主评测"

CORE_FILES = [
    "learner_profile.json",
    "stage_planning_output.json",
    "learning_package.md",
    "review_trace.json",
]

REQUIRED_FILES = CORE_FILES + [
    "diagnostic_output.json",
    "assessment_questions.json",
]


def check_case(case_dir: Path) -> dict:
    """检查单个 case 目录。返回 {status, errors, info}"""
    stage1 = case_dir / "stage_1"
    errors = []
    info = {}

    # 1. 顶层文件
    for fname in ["run_summary.json", "run_manifest.json", "llm_calls.jsonl"]:
        p = case_dir / fname
        if not p.exists():
            errors.append(f"缺失顶层文件: {fname}")
        elif p.stat().st_size == 0:
            errors.append(f"顶层文件为空: {fname}")

    # 2. stage_1 四个核心文件
    if not stage1.exists():
        errors.append("缺失 stage_1 目录")
    else:
        for fname in REQUIRED_FILES:
            p = stage1 / fname
            if not p.exists():
                errors.append(f"stage_1 缺失文件: {fname}")
                continue
            size = p.stat().st_size
            info[f"stage_1/{fname}"] = size
            if size < 20:
                errors.append(f"stage_1 文件过小 (<20B): {fname} ({size}B)")

    # 3. 解析 learner_profile.json 关键字段
    lp_path = stage1 / "learner_profile.json"
    if lp_path.exists() and lp_path.stat().st_size > 20:
        try:
            lp = json.loads(lp_path.read_text(encoding="utf-8"))
            mastered = lp.get("mastered_block_ids", [])
            target = lp.get("target_block_ids", [])
            if mastered and not target:
                errors.append("mastered_block_ids 有值但 target_block_ids 为空")
            # 安全检查：如果有 mastered，不应该包含 KB-02/04/09（零基础常见误判）
            zero_foundation_suspects = {"KB-02", "KB-04", "KB-09"}
            overlap = zero_foundation_suspects & set(mastered)
            if overlap:
                errors.append(f"⚠️ mastered_block_ids 疑似误判: 包含 {overlap} (零基础不应已掌握)")
            info["mastered_block_ids"] = mastered
            info["target_block_ids"] = target
            info["active_stage_block_id"] = lp.get("active_stage_block_id")
            info["goal"] = lp.get("goal")
        except json.JSONDecodeError as e:
            errors.append(f"learner_profile.json JSON 解析失败: {e}")

    # 4. 解析 stage_planning_output.json
    sp_path = stage1 / "stage_planning_output.json"
    if sp_path.exists() and sp_path.stat().st_size > 20:
        try:
            sp = json.loads(sp_path.read_text(encoding="utf-8"))
            sub_ids = sp.get("subtopic_ids", [])
            if not sub_ids:
                errors.append("⚠️ stage_planning_output: subtopic_ids 为空（空阶段风险）")
            info["stage_block_id"] = sp.get("block_id")
            info["stage_subtopic_ids"] = sub_ids
            info["stage_estimated_hours"] = sp.get("estimated_hours")
        except json.JSONDecodeError as e:
            errors.append(f"stage_planning_output.json JSON 解析失败: {e}")

    # 5. learning_package.md 大小检查
    md_path = stage1 / "learning_package.md"
    if md_path.exists():
        size = md_path.stat().st_size
        if size < 1000:
            errors.append(f"learning_package.md 过小 (<1KB): {size}B")
        elif size > 500_000:
            errors.append(f"learning_package.md 异常大 (>500KB): {size}B")

    # 6. review_trace.json 审核轮次检查
    rt_path = stage1 / "review_trace.json"
    if rt_path.exists() and rt_path.stat().st_size > 20:
        try:
            rt = json.loads(rt_path.read_text(encoding="utf-8"))
            iterations = rt.get("iterations", 0)
            rh = rt.get("review_history", [])
            if iterations == 0 and not rh:
                errors.append("⚠️ review_trace 无审核轮次（假成功风险）")
            info["review_iterations"] = iterations
            info["review_history_len"] = len(rh)
            info["review_passed"] = rt.get("review_passed")
            info["needs_human_review"] = rt.get("needs_human_review")
        except json.JSONDecodeError as e:
            errors.append(f"review_trace.json JSON 解析失败: {e}")

    # 7. llm_calls.jsonl 模型分布
    calls_path = case_dir / "llm_calls.jsonl"
    if calls_path.exists():
        try:
            calls = [json.loads(l) for l in calls_path.read_text(encoding="utf-8").splitlines() if l.strip()]
            models = Counter(c.get("actual_model", "?") for c in calls)
            fb_count = sum(1 for c in calls if c.get("fallback_triggered"))
            fb_reasons = Counter()
            for c in calls:
                if c.get("fallback_triggered"):
                    err = (c.get("error_type") or "(empty)")[:80]
                    fb_reasons[err] += 1
            info["llm_total_calls"] = len(calls)
            info["llm_model_dist"] = dict(models)
            info["llm_fallback_count"] = fb_count
            if fb_reasons:
                info["llm_fallback_reasons"] = dict(fb_reasons)
            # finish_reason=length 检查
            length_fb = sum(1 for c in calls if "finish_reason=length" in (c.get("error_type") or ""))
            if length_fb > 0:
                errors.append(f"⚠️ 有 {length_fb} 次 finish_reason=length 回退（thinking disabled 未生效？）")
        except Exception as e:
            errors.append(f"llm_calls.jsonl 解析失败: {e}")

    # 8. run_summary.json
    rs_path = case_dir / "run_summary.json"
    if rs_path.exists():
        try:
            rs = json.loads(rs_path.read_text(encoding="utf-8"))
            status = rs.get("status")
            if status != "completed":
                errors.append(f"run_summary status={status} (非 completed)")
            info["run_status"] = status
            info["execution_path"] = rs.get("execution_path")
            info["accuracy"] = rs.get("stage_1_accuracy")
        except json.JSONDecodeError as e:
            errors.append(f"run_summary.json JSON 解析失败: {e}")

    return {
        "case_id": case_dir.name,
        "status": "✅ PASS" if not errors else f"❌ FAIL ({len(errors)} 项问题)",
        "errors": errors,
        "info": info,
    }


def main():
    parser = argparse.ArgumentParser(description="评测输出完好检查")
    parser.add_argument("--case", help="检查单个 case，如 case_001")
    parser.add_argument("--dir", help="检查整个输出目录，如 raw_outputs")
    args = parser.parse_args()

    if not args.case and not args.dir:
        parser.print_help()
        return 1

    results = []

    if args.case:
        case_dir = EVAL_DIR / "raw_outputs" / args.case
        if not case_dir.exists():
            # 也试试归档目录
            for archive in EVAL_DIR.glob("raw_outputs*"):
                candidate = archive / args.case
                if candidate.exists():
                    case_dir = candidate
                    print(f"[VERIFY] 在归档目录找到: {candidate}")
                    break
            else:
                print(f"[VERIFY] ❌ 找不到 {args.case}")
                return 1
        results.append(check_case(case_dir))

    elif args.dir:
        base = EVAL_DIR / args.dir
        if not base.exists():
            print(f"[VERIFY] ❌ 目录不存在: {base}")
            return 1
        case_dirs = sorted([d for d in base.iterdir() if d.is_dir() and d.name.startswith("case_")])
        if not case_dirs:
            print(f"[VERIFY] ❌ {base} 下没有 case_* 目录")
            return 1
        print(f"[VERIFY] 扫描 {len(case_dirs)} 个 case...\n")
        for cd in case_dirs:
            results.append(check_case(cd))

    # 输出结果
    for r in results:
        print(f"{'='*60}")
        print(f"  {r['case_id']}  {r['status']}")
        print(f"{'='*60}")
        for k, v in r["info"].items():
            if isinstance(v, dict):
                print(f"  {k}:")
                for kk, vv in v.items():
                    print(f"    {kk}: {vv}")
            else:
                print(f"  {k}: {v}")
        if r["errors"]:
            print(f"\n  问题 ({len(r['errors'])}):")
            for e in r["errors"]:
                print(f"    ❌ {e}")
        print()

    fail_count = sum(1 for r in results if r["errors"])
    print(f"[VERIFY] 总计: {len(results)} 个 case, {len(results)-fail_count} 通过, {fail_count} 有问题")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
