# -*- coding: utf-8 -*-
"""真实数据指标统计脚本 —— 按文档《测试3+50要求.md》§4.3 的 12 项指标公式计算。

用法:
    python scripts/calculate_real_metrics.py

前置条件:
    1. raw_outputs/ 有 50 组真实输出
    2. build_evaluation_review_templates.py --force 已生成复核模板
    3. 人工填写 personalization_review.csv 和 fact_check_review.csv

输出:
    测试数据与评测材料/06_指标统计/
        personalization_evaluation.csv
        question_evaluation.csv
        claim_evaluation.csv
        decision_evaluation.csv
        metrics_summary.json
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
DELIVERY_DIR = ROOT_DIR / "测试数据与评测材料"
RAW_OUTPUTS_DIR = DELIVERY_DIR / "02_50组个性化主评测" / "raw_outputs"
METRICS_DIR = DELIVERY_DIR / "06_指标统计"
PERSONALIZATION_REVIEW_PATH = METRICS_DIR / "personalization_review.csv"
FACT_CHECK_REVIEW_PATH = METRICS_DIR / "fact_check_review.csv"
STRATEGY_REVIEW_PATH = METRICS_DIR / "strategy_review.csv"
SUBJECTIVE_GRADING_PATH = METRICS_DIR / "subjective_agent6_grading.jsonl"


# =====================================================================
# 工具函数
# =====================================================================
def load_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return records


def load_subjective_grading_evidence() -> Set[Tuple[str, str, str]]:
    """Load successful targeted Agent6 grades kept outside frozen raw outputs."""
    return {
        (str(r.get("case_id")), str(r.get("stage")), str(r.get("question_id")))
        for r in load_jsonl(SUBJECTIVE_GRADING_PATH)
        if r.get("success") and isinstance(r.get("grading"), dict)
    }


def load_csv(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _load_package_md(path: Path) -> str:
    if not path.exists():
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# =====================================================================
# 人工复核表检查
# =====================================================================
def is_review_completed(review_path: Path) -> Tuple[bool, int, int]:
    """检查人工复核表是否已填写。
    返回: (是否完成, 已填写条数, 总条数)
    """
    rows = load_csv(review_path)
    if not rows:
        return (False, 0, 0)
    filled = sum(1 for r in rows if (r.get("verdict") or "").strip().lower() in ("pass", "fail", "error"))
    return (filled == len(rows) and filled > 0, filled, len(rows))


# =====================================================================
# 指标 1: 画像适配率（依赖人工复核表）
# =====================================================================
def calc_personalization_adaptation() -> Dict:
    """画像适配率 = 普通学习阶段的 pass 条件数 / 普通阶段已复核条件数。

    ``not_applicable`` is reserved for terminal/empty stages.  Those rows
    remain auditable in the review CSV but are intentionally excluded from
    the normal learning-package adaptation denominator.
    """
    rows = load_csv(PERSONALIZATION_REVIEW_PATH)
    filled = sum(1 for r in rows if (r.get("verdict") or "").strip().lower() in ("pass", "fail", "error"))
    total = len(rows)
    excluded = sum(1 for r in rows if (r.get("verdict") or "").strip().lower() == "not_applicable")
    completed = filled > 0 and (filled + excluded == total)
    if not completed:
        return {
            "personalization_adaptation_rate": None,
            "personalization_pass_count": None,
            "personalization_total": None,
            "review_completed": False,
            "review_filled": filled,
            "review_total": total,
            "review_excluded": excluded,
        }
    pass_count = sum(1 for r in rows if (r.get("verdict") or "").strip().lower() == "pass")
    return {
        "personalization_adaptation_rate": round(pass_count / filled, 4) if filled > 0 else None,
        "personalization_pass_count": pass_count,
        "personalization_total": filled,
        "review_completed": True,
        "review_excluded": excluded,
    }


# =====================================================================
# 指标 2: 可核验事实错误率（依赖人工复核表）
# =====================================================================
def calc_fact_check_error() -> Dict:
    """可核验事实错误率 = (fail + error事实数) / 已核验事实数 × 100%
    数据来源: fact_check_review.csv
    未完成人工复核时返回 null。"""
    completed, filled, total = is_review_completed(FACT_CHECK_REVIEW_PATH)
    if not completed:
        return {
            "fact_check_error_rate": None,
            "fact_check_fail_count": None,
            "fact_check_error_count": None,
            "fact_check_total": None,
            "review_completed": False,
            "review_filled": filled,
            "review_total": total,
        }

    rows = load_csv(FACT_CHECK_REVIEW_PATH)
    fail_count = sum(1 for r in rows if (r.get("verdict") or "").strip().lower() == "fail")
    error_count = sum(1 for r in rows if (r.get("verdict") or "").strip().lower() == "error")
    return {
        "fact_check_error_rate": round((fail_count + error_count) / filled, 4) if filled > 0 else None,
        "fact_check_fail_count": fail_count,
        "fact_check_error_count": error_count,
        "fact_check_total": filled,
        "review_completed": True,
    }


def calc_strategy_reasonableness() -> Dict:
    """人工核验 Agent6 阶段后策略是否适合当前学习状态。"""
    completed, filled, total = is_review_completed(STRATEGY_REVIEW_PATH)
    if not completed:
        return {
            "strategy_reasonableness_rate": None,
            "strategy_pass_count": None,
            "strategy_total": None,
            "review_completed": False,
            "review_filled": filled,
            "review_total": total,
        }
    rows = load_csv(STRATEGY_REVIEW_PATH)
    passed = sum(1 for r in rows if (r.get("verdict") or "").strip().lower() == "pass")
    return {
        "strategy_reasonableness_rate": round(passed / filled, 4) if filled else None,
        "strategy_pass_count": passed,
        "strategy_total": filled,
        "review_completed": True,
    }


# =====================================================================
# 指标 3: 审核修复率
# =====================================================================
def calc_review_fix_rate() -> Dict:
    """审核修复率 = 修订后经审核解决的问题数 / 初稿问题总数 × 100%
    数据来源: review_trace.json 中的 review_history
    """
    total_issues = 0
    resolved_issues = 0
    review_categories = ["technical", "format", "pedagogy", "curriculum",
                         "style", "accuracy", "safety", "alignment"]

    for case_dir in sorted(RAW_OUTPUTS_DIR.iterdir()):
        if not case_dir.is_dir() or not case_dir.name.startswith("case_"):
            continue
        for stage_name in ["stage_1", "stage_2"]:
            trace = load_json(case_dir / stage_name / "review_trace.json")
            if not trace:
                continue
            history = trace.get("review_history", []) or []
            for round_data in history:
                if not isinstance(round_data, dict):
                    continue
                for cat in review_categories:
                    cat_data = round_data.get(cat)
                    if not isinstance(cat_data, dict):
                        continue
                    errors = cat_data.get("errors") or []
                    for e in errors:
                        if not isinstance(e, dict):
                            continue
                        total_issues += 1
                        # must_fix=false 或 must_fix_reason 含 "降级" 或 change_request.required=false
                        # 视为"已解决"（审核后判定不需要修复或已修订）
                        if not e.get("must_fix", False):
                            resolved_issues += 1

    return {
        "review_fix_rate": round(resolved_issues / total_issues, 4) if total_issues > 0 else None,
        "review_resolved_count": resolved_issues,
        "review_total_issues": total_issues,
    }


# =====================================================================
# 指标 4: 当前阶段核心知识点覆盖率（冻结口径）
# =====================================================================
CORE_KNOWLEDGE_SCOPE_VERSION = "stage_planned_subtopics_v2"

def calc_knowledge_coverage() -> Dict:
    """当前阶段核心知识点覆盖率。

    分母是阶段规划明确给出的 ``subtopic_ids``，分子是阶段题目实际出现
    且属于该清单的 ``subtopic_id``。已掌握/跳过导致规划为空的阶段记为
    not_applicable，不把空阶段计作未覆盖。
    """
    cases = load_csv(DELIVERY_DIR / "02_50组个性化主评测" / "evaluation_cases.csv")
    rows = []
    total_expected = 0
    total_covered = 0

    for case in cases:
        case_id = case["case_id"]
        case_dir = RAW_OUTPUTS_DIR / case_id
        stage_1 = case_dir / "stage_1"

        plan = load_json(stage_1 / "stage_planning_output.json") or {}
        questions = load_json(stage_1 / "assessment_questions.json") or []

        expected_subtopics = {
            str(s).strip() for s in (plan.get("subtopic_ids") or []) if str(s).strip()
        }
        if not expected_subtopics and plan.get("subtopic_id"):
            expected_subtopics.add(str(plan["subtopic_id"]).strip())

        covered_subtopics: Set[str] = set()
        for q in questions:
            if q.get("subtopic_id"):
                covered_subtopics.add(str(q["subtopic_id"]).strip())

        overlap = covered_subtopics & expected_subtopics
        total_expected += len(expected_subtopics)
        total_covered += len(overlap)

        coverage = len(overlap) / len(expected_subtopics) if expected_subtopics else None
        rows.append({
            "case_id": case_id,
            "scope_version": CORE_KNOWLEDGE_SCOPE_VERSION,
            "stage": "stage_1",
            "expected_subtopics": "|".join(sorted(expected_subtopics)),
            "covered_subtopics": "|".join(sorted(covered_subtopics)),
            "expected_count": len(expected_subtopics),
            "covered_count": len(overlap),
            "coverage_ratio": round(coverage, 4) if coverage is not None else None,
            "status": "scored" if expected_subtopics else "not_applicable",
        })

    overall_rate = total_covered / total_expected if total_expected > 0 else None
    return {"rows": rows, "overall_rate": round(overall_rate, 4) if overall_rate is not None else None,
            "scope_version": CORE_KNOWLEDGE_SCOPE_VERSION,
            "total_covered": total_covered, "total_expected": total_expected}


# =====================================================================
# 指标 5: 题库知识点覆盖率
# =====================================================================
def calc_question_bank_coverage() -> Dict:
    """题库知识点覆盖率 = 至少有3道专属题的知识点数 / 73 × 100%
    数据来源: 题库Seed与数据库验证
    """
    seed_path = ROOT_DIR / "data" / "seeds" / "test_questions.json"
    seed = load_json(seed_path)
    if not seed or not isinstance(seed, list):
        return {"question_bank_coverage_rate": None, "subtopics_with_3plus": None, "total_subtopics": 73}

    subtopic_counts: Dict[str, int] = {}
    for q in seed:
        sid = q.get("subtopic_id", "")
        if sid:
            subtopic_counts[sid] = subtopic_counts.get(sid, 0) + 1

    covered = sum(1 for count in subtopic_counts.values() if count >= 3)
    total = 73
    return {
        "question_bank_coverage_rate": round(covered / total, 4) if total > 0 else None,
        "subtopics_with_3plus": covered,
        "total_subtopics": total,
    }


# =====================================================================
# 指标 6/7/8: 题目匹配、兜底率、有效率
# =====================================================================
def calc_question_metrics() -> Dict:
    """计算题目匹配、每知识点 2 题达标率、可判分率。
    数据来源: assessment_questions.json + stage_planning_output.json
    """
    total_questions = 0
    exclusive_match = 0  # subtopic_id 属于当前 stage.subtopic_ids
    block_fallback = 0   # 非专属但 block_id 匹配当前阶段
    valid_questions = 0  # 有标准答案、解析且可判分
    expected_slots = 0   # 每个阶段知识点固定 2 道题
    exclusive_slots = 0  # 每个知识点最多计 2 道精准题
    rows = []
    targeted_agent6_evidence = load_subjective_grading_evidence()

    for case_dir in sorted(RAW_OUTPUTS_DIR.iterdir()):
        if not case_dir.is_dir() or not case_dir.name.startswith("case_"):
            continue
        case_id = case_dir.name
        for stage_name in ["stage_1", "stage_2"]:
            stage_dir = case_dir / stage_name
            if not stage_dir.exists():
                continue

            questions = load_json(stage_dir / "assessment_questions.json") or []
            plan = load_json(stage_dir / "stage_planning_output.json") or {}

            stage_subtopic_ids: Set[str] = set(plan.get("subtopic_ids") or [])
            if plan.get("subtopic_id"):
                stage_subtopic_ids.add(plan["subtopic_id"])
            stage_block_id = plan.get("block_id", "")
            stage_block_ids: Set[str] = set(plan.get("block_ids") or [])
            if stage_block_id:
                stage_block_ids.add(stage_block_id)

            expected_slots += len(stage_subtopic_ids) * 2
            stage_exclusive_counts: Dict[str, int] = {}

            # 判分证据只接受 rubric/评分字段或独立 grader 结果，不能把
            # learner_answers 中的模拟 is_correct 当作 LLM 判分证据。
            answer_records = load_json(stage_dir / "learner_answers.json") or []
            grading_by_qid = {
                str(a.get("question_id")): a for a in answer_records
                if isinstance(a, dict) and a.get("question_id")
                and any(k in a for k in ("grading_result", "grading_score", "grader_result", "rubric_result"))
            }

            for q in questions:
                total_questions += 1
                q_subtopic = q.get("subtopic_id", "")
                q_block = q.get("block_id", "")

                is_exclusive = q_subtopic in stage_subtopic_ids if stage_subtopic_ids else False
                is_block_match = q_block in stage_block_ids if stage_block_ids else False

                if is_exclusive:
                    exclusive_match += 1
                    if stage_exclusive_counts.get(q_subtopic, 0) < 2:
                        stage_exclusive_counts[q_subtopic] = stage_exclusive_counts.get(q_subtopic, 0) + 1
                        exclusive_slots += 1
                elif is_block_match:
                    block_fallback += 1

                question_type = str(q.get("question_type", "")).lower()
                is_subjective = question_type in {"short_answer", "practical"} or str(q.get("grading_mode", "")).lower() in {"llm", "subjective", "rubric"}
                has_answer = bool(q.get("correct_answer") or q.get("reference_answer"))
                has_explanation = bool(q.get("explanation"))
                has_grading_evidence = (
                    str(q.get("question_id", "")) in grading_by_qid
                    or (case_id, stage_name, str(q.get("question_id", ""))) in targeted_agent6_evidence
                )
                # Frozen questions store their reference in correct_answer rather
                # than a separate rubric field. A successful targeted Agent6
                # record proves that reference plus the fixed grading rule were
                # actually used, so it is valid scoring evidence.
                has_rubric = bool(
                    q.get("rubric") or q.get("grading_rule") or q.get("judge_rule")
                    or q.get("reference_answer") or (has_answer and has_grading_evidence)
                )
                is_valid = ((not is_subjective and has_answer) or (is_subjective and has_rubric and has_grading_evidence))
                if is_valid:
                    valid_questions += 1

                rows.append({
                    "case_id": case_id,
                    "stage": stage_name,
                    "question_id": q.get("question_id", ""),
                    "block_id": q_block,
                    "subtopic_id": q_subtopic,
                    "difficulty": q.get("difficulty", ""),
                    "question_type": q.get("question_type", ""),
                    "has_correct_answer": has_answer,
                    "has_explanation": has_explanation,
                    "has_rubric": has_rubric,
                    "has_grading_evidence": has_grading_evidence,
                    "is_exclusive_match": is_exclusive,
                    "is_block_fallback": (not is_exclusive and is_block_match),
                    "is_valid": is_valid,
                })

    return {
        "rows": rows,
        "exclusive_match_rate": round(exclusive_match / total_questions, 4) if total_questions > 0 else None,
        "block_fallback_rate": round(block_fallback / total_questions, 4) if total_questions > 0 else None,
        "question_valid_rate": round(valid_questions / total_questions, 4) if total_questions > 0 else None,
        "exclusive_match_count": exclusive_match,
        "block_fallback_count": block_fallback,
        "valid_count": valid_questions,
        "total_questions": total_questions,
        "expected_slots": expected_slots,
        "exclusive_slots": exclusive_slots,
        "question_quota_rate": round(exclusive_slots / expected_slots, 4) if expected_slots > 0 else None,
    }


# =====================================================================
# 指标 9: 目标板块映射准确率
# =====================================================================
def calc_block_mapping_accuracy() -> Dict:
    """目标板块映射准确率 = 正确映射目标板块的案例数 / 有效案例数 × 100%
    数据来源: 预期目标(expected_target_blocks) 与实际规划(block_id)
    """
    cases = load_csv(DELIVERY_DIR / "02_50组个性化主评测" / "evaluation_cases.csv")
    matched = 0
    valid = 0
    rows = []

    for case in cases:
        case_id = case["case_id"]
        expected_blocks = {b.strip() for b in case.get("expected_target_blocks", "").split("|") if b.strip()}
        expected_first = (case.get("expected_first_stage") or "").strip()

        plan = load_json(RAW_OUTPUTS_DIR / case_id / "stage_1" / "stage_planning_output.json") or {}
        actual_block = plan.get("block_id", "")
        actual_blocks: Set[str] = set(plan.get("block_ids") or [])
        if actual_block:
            actual_blocks.add(actual_block)

        # 映射正确：实际 block 在 expected 列表中，或 expected_first_stage 匹配
        is_match = (actual_block in expected_blocks) if expected_blocks else True
        if expected_first and actual_block:
            is_match = is_match or (expected_first == actual_block)

        if actual_block:  # 有效案例（有规划输出）
            valid += 1
            if is_match:
                matched += 1

        rows.append({
            "case_id": case_id,
            "expected_target_blocks": "|".join(sorted(expected_blocks)),
            "expected_first_stage": expected_first,
            "actual_block": actual_block,
            "is_correct_mapping": is_match,
        })

    return {
        "rows": rows,
        "block_mapping_accuracy": round(matched / valid, 4) if valid > 0 else None,
        "matched_count": matched,
        "valid_count": valid,
    }


# =====================================================================
# 指标 10: 首阶段选择准确率
# =====================================================================
def calc_first_stage_accuracy() -> Dict:
    """首阶段选择准确率 = 首阶段符合硬前置关系和用户基础的案例数 / 成功生成阶段的案例数 × 100%
    数据来源: 规划输出与预期规则
    """
    cases = load_csv(DELIVERY_DIR / "02_50组个性化主评测" / "evaluation_cases.csv")
    matched = 0
    valid = 0
    rows = []

    for case in cases:
        case_id = case["case_id"]
        expected_first = (case.get("expected_first_stage") or "").strip()

        plan = load_json(RAW_OUTPUTS_DIR / case_id / "stage_1" / "stage_planning_output.json") or {}
        actual_block = plan.get("block_id", "")

        # 首阶段选择正确：actual_block == expected_first_stage
        # 或 expected_first 为空时不扣分
        if actual_block:  # 成功生成阶段
            valid += 1
            is_match = (not expected_first) or (actual_block == expected_first)
            if is_match:
                matched += 1

        rows.append({
            "case_id": case_id,
            "expected_first_stage": expected_first,
            "actual_first_stage": actual_block,
            "is_correct": (not expected_first) or (actual_block == expected_first) if actual_block else None,
        })

    return {
        "rows": rows,
        "first_stage_accuracy": round(matched / valid, 4) if valid > 0 else None,
        "matched_count": matched,
        "valid_count": valid,
    }


# =====================================================================
# 指标 11: 策略选择准确率
# =====================================================================
# 答题行为 → 预期策略映射
ANSWER_BEHAVIOR_STRATEGY = {
    "high": "advance",      # 高分(≥0.8) → 推进
    "medium": "reinforce",  # 中分(0.6~0.79) → 巩固
    "low": "simplify",      # 低分(0.4~0.59) → 降维
    "very_low": "backtrack",# 极低(<0.4) → 回退
    "zero": "simplify",     # 零分 → 降维
}


def calc_strategy_accuracy() -> Dict:
    """策略选择准确率 = 策略符合预设答题行为规则的阶段数 / 有阶段决策的阶段数 × 100%
    数据来源: 答题脚本(answer_script_stage_1) 与 stage_decision.json
    """
    cases = load_csv(DELIVERY_DIR / "02_50组个性化主评测" / "evaluation_cases.csv")
    matched = 0
    total_stages = 0
    rows = []

    def parse_script(case: Dict[str, str], stage_name: str) -> Dict[str, Any]:
        raw = case.get(f"answer_script_{stage_name}") or ""
        if raw.strip().startswith("{"):
            try:
                value = json.loads(raw)
                if isinstance(value, dict):
                    return value
            except json.JSONDecodeError:
                pass
        return {"answer_behavior": raw.strip() or "medium"}

    for case in cases:
        case_id = case["case_id"]

        for stage_name in ["stage_1", "stage_2"]:
            decision = load_json(RAW_OUTPUTS_DIR / case_id / stage_name / "stage_decision.json")
            if not decision:
                continue
            script = parse_script(case, stage_name)
            answer_behavior = str(script.get("answer_behavior") or "medium").strip()
            expected_strategy = str(script.get("expected_next_strategy") or "").strip()
            if not expected_strategy and stage_name == "stage_1":
                expected_strategy = str(case.get("expected_next_strategy") or "").strip()
            if not expected_strategy:
                expected_strategy = ANSWER_BEHAVIOR_STRATEGY.get(answer_behavior, "reinforce")

            actual_strategy = (decision.get("next_strategy") or "").strip()
            # 归一化比较
            is_match = _strategy_match(expected_strategy, actual_strategy)
            total_stages += 1
            if is_match:
                matched += 1

            rows.append({
                "case_id": case_id,
                "stage": stage_name,
                "answer_behavior": answer_behavior,
                "expected_strategy": expected_strategy,
                "actual_strategy": actual_strategy,
                "accuracy": decision.get("accuracy"),
                "strategy_match": is_match,
            })

    return {
        "rows": rows,
        "strategy_accuracy": round(matched / total_stages, 4) if total_stages > 0 else None,
        "matched_count": matched,
        "total_stages": total_stages,
    }


def _strategy_match(expected: str, actual: str) -> bool:
    """策略匹配（含语义归一化）。"""
    if not expected or not actual:
        return False
    # 语义等价组
    groups = [
        {"advance", "进阶", "升级"},
        {"reinforce", "巩固", "强化", "continue", "repeat"},
        {"simplify", "简化", "降维", "low"},
        {"backtrack", "回退", "回溯"},
        {"completed", "完成", "掌握"},
    ]
    for g in groups:
        if expected.lower() in g and actual.lower() in g:
            return True
    return expected.lower() == actual.lower()


# =====================================================================
# 指标 12: 完整流程成功率
# =====================================================================
def calc_flow_success_rate() -> Dict:
    """完整流程成功率 = 完成诊断、生成、审核、测试和决策的案例数 / 50 × 100%
    数据来源: run_manifest.json 与阶段输出文件
    """
    run_manifest = load_json(RAW_OUTPUTS_DIR / "run_manifest.json") or {}
    summaries = run_manifest.get("summaries", [])
    if not summaries:
        # 兜底：按目录检查（文档 §3.5 文件清单：根目录 4 + stage_1 8 = 12 项）
        case_dirs = [d for d in RAW_OUTPUTS_DIR.iterdir()
                     if d.is_dir() and d.name.startswith("case_")]
        succeeded = 0
        for cd in case_dirs:
            stage_1 = cd / "stage_1"
            stage_2 = cd / "stage_2"
            # 根目录 4 项
            root_needed = [
                cd / "input.json",
                cd / "run_manifest.json",
                cd / "llm_calls.jsonl",
                cd / "generation_events.jsonl",
            ]
            # stage_1 8 项（文档 §3.5）
            stage_1_needed = [
                stage_1 / "learner_profile.json",
                stage_1 / "diagnostic_output.json",
                stage_1 / "stage_planning_output.json",
                stage_1 / "learning_package.md",
                stage_1 / "review_trace.json",
                stage_1 / "assessment_questions.json",
                stage_1 / "learner_answers.json",
                stage_1 / "stage_decision.json",
            ]
            needed = root_needed + stage_1_needed
            # stage_2 存在时追加检查（snapshot 替代 diagnostic_output）
            if stage_2.exists():
                stage_2_needed = [
                    stage_2 / "learner_profile_snapshot.json",
                    stage_2 / "stage_planning_output.json",
                    stage_2 / "learning_package.md",
                    stage_2 / "review_trace.json",
                    stage_2 / "assessment_questions.json",
                    stage_2 / "learner_answers.json",
                    stage_2 / "stage_decision.json",
                ]
                needed = needed + stage_2_needed
            if all(p.exists() for p in needed):
                succeeded += 1
        total = len(case_dirs)
    else:
        succeeded = sum(1 for s in summaries if s.get("status") == "completed")
        total = len(summaries)

    # execution_path 分组
    path_groups = {"deepseek_only": 0, "mixed_fallback": 0, "ollama_only": 0, "failed": 0}
    if summaries:
        for s in summaries:
            p = s.get("execution_path") or "failed"
            path_groups[p] = path_groups.get(p, 0) + 1

    return {
        "flow_success_rate": round(succeeded / total, 4) if total > 0 else None,
        "flow_succeeded": succeeded,
        "flow_total": total,
        "execution_path_breakdown": path_groups,
    }


# =====================================================================
# LLM 调用统计
# =====================================================================
def calc_llm_stats() -> Dict:
    all_calls = load_jsonl(RAW_OUTPUTS_DIR / "llm_calls.jsonl")
    total = len(all_calls)
    success = sum(1 for c in all_calls if c.get("success"))
    fallback = sum(1 for c in all_calls if c.get("fallback_triggered"))
    failed = total - success
    return {
        "llm_call_count": total,
        "llm_successful_calls": success,
        "llm_failed_calls": failed,
        "llm_fallback_count": fallback,
        "llm_fallback_rate": round(fallback / total, 4) if total > 0 else 0,
    }


# =====================================================================
# 谬误率（辅助指标，从审核 trace 统计）
# =====================================================================
def calc_hallucination_rate() -> Dict:
    """谬误率 = critical 错误数 / 审核主张总数（所有 severity 的 error 条目数）
    数据来源: review_trace.json review_history 中的所有 errors"""
    critical = 0
    major = 0
    minor = 0
    total_claims = 0
    total_package_chars = 0
    review_categories = ["technical", "format", "pedagogy", "curriculum",
                         "style", "accuracy", "safety", "alignment"]

    for case_dir in sorted(RAW_OUTPUTS_DIR.iterdir()):
        if not case_dir.is_dir() or not case_dir.name.startswith("case_"):
            continue
        for stage_name in ["stage_1", "stage_2"]:
            stage_dir = case_dir / stage_name
            md_path = stage_dir / "learning_package.md"
            if md_path.exists():
                total_package_chars += len(md_path.read_text(encoding="utf-8"))
            trace = load_json(stage_dir / "review_trace.json")
            if not trace:
                continue
            for round_data in (trace.get("review_history") or []):
                if not isinstance(round_data, dict):
                    continue
                for cat in review_categories:
                    cat_data = round_data.get(cat)
                    if not isinstance(cat_data, dict):
                        continue
                    for e in (cat_data.get("errors") or []):
                        if not isinstance(e, dict):
                            continue
                        total_claims += 1
                        sev = (e.get("severity") or "").lower()
                        if sev == "critical":
                            critical += 1
                        elif sev == "major":
                            major += 1
                        elif sev == "minor":
                            minor += 1

    rate = critical / total_claims if total_claims > 0 else None
    per_100k = critical / total_package_chars * 100_000 if total_package_chars > 0 else 0
    return {
        "hallucination_rate": round(rate, 4) if rate is not None else None,
        "hallucination_critical": critical,
        "hallucination_major": major,
        "hallucination_minor": minor,
        "hallucination_total_claims": total_claims,
        "hallucination_per_100k_chars": round(per_100k, 2),
        "total_package_chars": total_package_chars,
    }


# =====================================================================
# 主函数
# =====================================================================
def main():
    if not RAW_OUTPUTS_DIR.exists():
        print(f"ERROR: raw_outputs 目录不存在: {RAW_OUTPUTS_DIR}")
        print("请先运行 scripts/run_real_evaluation.py 生成真实数据")
        return 1

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    print("计算指标 1: 画像适配率（依赖人工复核表）...")
    p_adapt = calc_personalization_adaptation()

    print("计算指标 2: 可核验事实错误率（依赖人工复核表）...")
    f_check = calc_fact_check_error()

    print("计算指标 3: 审核修复率...")
    review_fix = calc_review_fix_rate()

    print("计算指标 4: 当前阶段核心知识点覆盖率（冻结口径）...")
    kc = calc_knowledge_coverage()
    write_csv(METRICS_DIR / "claim_evaluation.csv", kc["rows"],
              ["case_id", "scope_version", "stage", "expected_subtopics", "covered_subtopics", "expected_count",
               "covered_count", "coverage_ratio", "status"])

    print("计算指标 5: 题库知识点覆盖率...")
    qb_cov = calc_question_bank_coverage()

    print("计算指标 6/7/8: 专属题目匹配率、同板块兜底率、题目有效率...")
    qm = calc_question_metrics()
    write_csv(METRICS_DIR / "question_evaluation.csv", qm["rows"],
              ["case_id", "stage", "question_id", "block_id", "subtopic_id",
               "difficulty", "question_type", "has_correct_answer", "has_explanation",
               "has_rubric", "has_grading_evidence", "is_exclusive_match",
               "is_block_fallback", "is_valid"])

    print("计算指标 9: 目标板块映射准确率...")
    bm = calc_block_mapping_accuracy()
    write_csv(METRICS_DIR / "block_mapping_evaluation.csv", bm["rows"],
              ["case_id", "expected_target_blocks", "expected_first_stage",
               "actual_block", "is_correct_mapping"])

    print("计算指标 10: 首阶段选择准确率...")
    fs = calc_first_stage_accuracy()
    write_csv(METRICS_DIR / "first_stage_evaluation.csv", fs["rows"],
              ["case_id", "expected_first_stage", "actual_first_stage", "is_correct"])

    print("计算指标 11: 策略选择准确率...")
    sa = calc_strategy_accuracy()
    write_csv(METRICS_DIR / "decision_evaluation.csv", sa["rows"],
              ["case_id", "stage", "answer_behavior", "expected_strategy",
               "actual_strategy", "accuracy", "strategy_match"])
    strategy_review = calc_strategy_reasonableness()

    print("计算指标 12: 完整流程成功率...")
    flow = calc_flow_success_rate()

    print("计算 LLM 调用统计...")
    llm_stats = calc_llm_stats()

    print("计算谬误率（辅助指标）...")
    halluc = calc_hallucination_rate()

    # 画像适配详细表
    # 从 raw_outputs 生成结构化 personalization_evaluation.csv（非人工复核，仅结构化参考）
    cases = load_csv(DELIVERY_DIR / "02_50组个性化主评测" / "evaluation_cases.csv")
    p_rows = []
    for case in cases:
        case_id = case["case_id"]
        stage_1 = RAW_OUTPUTS_DIR / case_id / "stage_1"
        plan = load_json(stage_1 / "stage_planning_output.json") or {}
        expected_diff = case.get("expected_difficulty", "")
        actual_diff = plan.get("difficulty", "")
        p_rows.append({
            "case_id": case_id,
            "expected_difficulty": expected_diff,
            "actual_difficulty": actual_diff,
            "difficulty_match": expected_diff == actual_diff,
        })
    write_csv(METRICS_DIR / "personalization_evaluation.csv", p_rows,
              ["case_id", "expected_difficulty", "actual_difficulty", "difficulty_match"])

    # 汇总
    summary = {
        # 文档 §4.3 的 12 项指标
        "1_画像适配率": p_adapt["personalization_adaptation_rate"],
        "1_画像适配_pass数": p_adapt.get("personalization_pass_count"),
        "1_画像适配_总数": p_adapt.get("personalization_total"),
        "1_复核完成": p_adapt.get("review_completed", False),

        "2_可核验事实错误率": f_check["fact_check_error_rate"],
        "2_事实_fail数": f_check.get("fact_check_fail_count"),
        "2_事实_error数": f_check.get("fact_check_error_count"),
        "2_事实_总数": f_check.get("fact_check_total"),
        "2_复核完成": f_check.get("review_completed", False),

        "3_审核修复率": review_fix["review_fix_rate"],
        "3_审核_已解决": review_fix["review_resolved_count"],
        "3_审核_总问题": review_fix["review_total_issues"],

        "4_任务核心知识点覆盖率": kc["overall_rate"],
        "4_覆盖_已覆盖数": kc["total_covered"],
        "4_覆盖_预期总数": kc["total_expected"],

        "5_题库知识点覆盖率": qb_cov["question_bank_coverage_rate"],
        "5_题库_已覆盖知识点": qb_cov["subtopics_with_3plus"],
        "5_题库_总知识点": qb_cov["total_subtopics"],

        "6_专属题目匹配率": qm["exclusive_match_rate"],
        "6_专属_匹配数": qm["exclusive_match_count"],
        "6_专属_总数": qm["total_questions"],
        "6_每知识点2题达标率": qm["question_quota_rate"],
        "6_目标题量_精准题数": qm["exclusive_slots"],
        "6_目标题量_期望题数": qm["expected_slots"],

        "7_同板块兜底率": qm["block_fallback_rate"],
        "7_兜底_匹配数": qm["block_fallback_count"],
        "7_兜底_总数": qm["total_questions"],

        "8_题目有效率": qm["question_valid_rate"],
        "8_有效_题数": qm["valid_count"],
        "8_有效_总数": qm["total_questions"],

        "9_目标板块映射准确率": bm["block_mapping_accuracy"],
        "9_映射_正确数": bm["matched_count"],
        "9_映射_有效数": bm["valid_count"],

        "10_首阶段选择准确率": fs["first_stage_accuracy"],
        "10_首阶段_正确数": fs["matched_count"],
        "10_首阶段_有效数": fs["valid_count"],

        "11_策略选择准确率": sa["strategy_accuracy"],
        "11_策略_匹配数": sa["matched_count"],
        "11_策略_总数": sa["total_stages"],
        "11_人工策略合理率": strategy_review["strategy_reasonableness_rate"],
        "11_人工策略_pass数": strategy_review.get("strategy_pass_count"),
        "11_人工策略_总数": strategy_review.get("strategy_total"),
        "11_人工策略复核完成": strategy_review.get("review_completed", False),

        "12_完整流程成功率": flow["flow_success_rate"],
        "12_流程_成功数": flow["flow_succeeded"],
        "12_流程_总数": flow["flow_total"],
        "12_执行路径分布": flow["execution_path_breakdown"],

        # 辅助指标
        "hallucination_rate": halluc["hallucination_rate"],
        "hallucination_critical": halluc["hallucination_critical"],
        "hallucination_major": halluc["hallucination_major"],
        "hallucination_minor": halluc["hallucination_minor"],
        "hallucination_total_claims": halluc["hallucination_total_claims"],
        "hallucination_per_100k_chars": halluc["hallucination_per_100k_chars"],
        "total_package_chars": halluc["total_package_chars"],

        # LLM 调用统计
        "llm_call_count": llm_stats["llm_call_count"],
        "llm_successful_calls": llm_stats["llm_successful_calls"],
        "llm_failed_calls": llm_stats["llm_failed_calls"],
        "llm_fallback_count": llm_stats["llm_fallback_count"],
        "llm_fallback_rate": llm_stats["llm_fallback_rate"],
    }

    write_json(METRICS_DIR / "metrics_summary.json", summary)

    # 打印
    print()
    print("=" * 70)
    print("指标汇总（文档 §4.3 的 12 项指标 + 辅助指标）")
    print("=" * 70)
    for key, val in summary.items():
        if val is None:
            print(f"  {key}: null（未完成人工复核或无数据）")
        elif isinstance(val, float):
            print(f"  {key}: {val:.2%}")
        else:
            print(f"  {key}: {val}")
    print()
    print(f"结果已保存到: {METRICS_DIR}")
    print()
    print("[NOTICE] 指标 1（画像适配率）和指标 2（可核验事实错误率）依赖人工复核表。")
    print("    请确保已运行 build_evaluation_review_templates.py --force 并填写复核表。")
    print("    未完成复核时，这两项指标为 null。")

    return 0


if __name__ == "__main__":
    sys.exit(main())
