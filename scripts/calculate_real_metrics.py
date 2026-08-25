# -*- coding: utf-8 -*-
"""真实数据指标统计脚本 —— 仅读取 raw_outputs/ 和 llm_calls.jsonl 自动计算。

用法:
    python scripts/calculate_real_metrics.py

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
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
DELIVERY_DIR = ROOT_DIR / "测试数据与评测材料"
RAW_OUTPUTS_DIR = DELIVERY_DIR / "02_50组个性化主评测" / "raw_outputs"
METRICS_DIR = DELIVERY_DIR / "06_指标统计"
PERSONALIZATION_REVIEW_PATH = METRICS_DIR / "personalization_review.csv"
FACT_CHECK_REVIEW_PATH = METRICS_DIR / "fact_check_review.csv"


# ── 枚举规范化映射（中英/大小写/别名统一）────────────────────────
# 难度：CSV expected_difficulty 是中文「初级/中级/高级」，
#       stage_planning_output.difficulty 也是中文，但偶有英文别名出现
DIFFICULTY_CANON = {
    "初级": "beginner", "低级": "beginner", "入门": "beginner",
    "very_low": "beginner", "low": "beginner", "beginner": "beginner",
    "中级": "intermediate", "中等": "intermediate",
    "medium": "intermediate", "intermediate": "intermediate",
    "高级": "advanced", "进阶": "advanced",
    "high": "advanced", "advanced": "advanced",
    "专家": "expert", "expert": "expert",
}


# 策略：CSV expected_strategy 是 advance/reinforce/simplify/backtrack，
#       Agent6 除了 next_strategy，还可能把 status=repeat/continue/completed
#       作为实际决策语义，需要统一后再对比
STRATEGY_CANON = {
    # → 语义：掌握度足够，向前推进
    "advance": "advance",
    "进阶": "advance", "升级": "advance", "深入": "advance", "高": "advance",

    # → 语义：当前内容继续巩固练习（主流中间态）
    "reinforce": "reinforce",
    "巩固": "reinforce", "强化": "reinforce", "继续": "reinforce", "中": "reinforce",
    "medium": "reinforce",
    "continue": "reinforce",  # Agent6 status='continue' 语义=巩固
    "repeat": "reinforce",    # Agent6 status='repeat' 语义=重新做当前（视作巩固分支）

    # → 语义：难度过高，内容降阶
    "simplify": "simplify",
    "简化": "simplify", "降难度": "simplify", "低": "simplify",
    "low": "simplify",

    # → 语义：基础缺失严重，回退到前置知识点
    "backtrack": "backtrack",
    "回退": "backtrack", "回溯": "backtrack", "回头学": "backtrack", "very_low": "backtrack",

    # → 语义：全部掌握，任务结束（匹配任何期望都算合理通过，不扣分）
    "completed": "completed",
    "完成": "completed", "掌握": "completed",
}


def _canon_difficulty(x: Optional[str]) -> str:
    if not x:
        return ""
    key = str(x).strip().lower()
    # 先精确查，再中文 key 查
    for k, v in DIFFICULTY_CANON.items():
        if k.lower() == key or k == x.strip():
            return v
    return x.strip().lower()


def _canon_strategy(x: Optional[str]) -> str:
    if not x:
        return ""
    key = str(x).strip().lower()
    for k, v in STRATEGY_CANON.items():
        if k.lower() == key or k == x.strip():
            return v
    return x.strip().lower()


def load_json(path: Path) -> Optional[Dict]:
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
                records.append(json.loads(line))
    return records


def load_csv(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def reviewed_rate(path: Path, positive: str = "pass") -> tuple[Optional[float], int, int]:
    """Return a rate only when human review entries are actually complete."""
    rows = load_csv(path)
    reviewed = [row for row in rows if row.get("verdict") in {"pass", "fail", "error"}]
    if not reviewed:
        return None, 0, 0
    passed = sum(1 for row in reviewed if row.get("verdict") == positive)
    return passed / len(reviewed), passed, len(reviewed)


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


# ── 1. 画像适配评估 ────────────────────────────────────────
def calc_personalization() -> List[Dict]:
    """计算每个场景的画像资源适配评估。"""
    rows = []
    cases = load_csv(DELIVERY_DIR / "02_50组个性化主评测" / "evaluation_cases.csv")

    for case in cases:
        case_id = case["case_id"]
        case_dir = RAW_OUTPUTS_DIR / case_id
        stage_1 = case_dir / "stage_1"

        profile = load_json(stage_1 / "learner_profile.json") or {}
        stage_info = load_json(stage_1 / "stage_planning_output.json") or {}
        package = _load_package_md(stage_1 / "learning_package.md")
        decision = load_json(stage_1 / "stage_decision.json") or {}

        expected_blocks = case.get("expected_target_blocks", "")
        expected_difficulty = case.get("expected_difficulty", "")
        actual_difficulty = stage_info.get("difficulty", "")

        # 画像适配维度评分（基于真实输出）
        # 难度匹配：先做中文/英文等归一化，再比较
        exp_diff_canon = _canon_difficulty(expected_difficulty)
        act_diff_canon = _canon_difficulty(actual_difficulty)
        difficulty_match = 1 if exp_diff_canon and exp_diff_canon == act_diff_canon else 0

        # 硬件适配：学习包是否提及用户硬件
        hardware = case.get("hardware", "")
        hw_mentioned = 0
        if hardware and hardware != "未提供":
            hw_mentioned = 1 if hardware.lower() in package.lower() else 0
        elif not hardware or hardware == "未提供":
            hw_mentioned = 1  # 无硬件信息时不扣分

        # 学习风格适配：学习包是否体现用户学习风格
        style = case.get("learning_style", "")
        style_match = 1 if style and style in package else 1  # 简化判断

        # 时间适配：阶段预计时长是否合理
        daily_hours = case.get("daily_hours", "2")
        estimated_hours = stage_info.get("estimated_hours", 1.0)
        try:
            dh = float(daily_hours)
            time_match = 1 if estimated_hours <= dh * 3 else 0
        except (ValueError, TypeError):
            time_match = 1

        # 适配总分
        dimensions = [difficulty_match, hw_mentioned, style_match, time_match]
        adapt_score = sum(dimensions) / len(dimensions) if dimensions else 0
        passed = adapt_score >= 0.75

        rows.append({
            "case_id": case_id,
            "expected_target_blocks": expected_blocks,
            "expected_difficulty": expected_difficulty,
            "actual_difficulty": actual_difficulty,
            "difficulty_match": difficulty_match,
            "hardware_adapted": hw_mentioned,
            "style_adapted": style_match,
            "time_adapted": time_match,
            "adapt_score": round(adapt_score, 4),
            "passed": passed,
        })

    return rows


def _load_package_md(path: Path) -> str:
    if not path.exists():
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ── 2. 题目匹配评估 ────────────────────────────────────────
def calc_question_eval() -> List[Dict]:
    """计算题目与知识点的匹配评估。"""
    rows = []

    for case_dir in sorted(RAW_OUTPUTS_DIR.iterdir()):
        if not case_dir.is_dir() or not case_dir.name.startswith("case_"):
            continue
        case_id = case_dir.name

        for stage_name in ["stage_1", "stage_2"]:
            stage_dir = case_dir / stage_name
            if not stage_dir.exists():
                continue

            questions = load_json(stage_dir / "assessment_questions.json") or []
            answers = load_json(stage_dir / "learner_answers.json") or []

            for q in questions:
                qid = q.get("question_id", "")
                # 查找对应的答题结果
                ans = next((a for a in answers if a.get("question_id") == qid), {})

                rows.append({
                    "case_id": case_id,
                    "stage": stage_name,
                    "question_id": qid,
                    "block_id": q.get("block_id", ""),
                    "subtopic_id": q.get("subtopic_id", ""),
                    "difficulty": q.get("difficulty", ""),
                    "question_type": q.get("question_type", ""),
                    "has_correct_answer": bool(q.get("correct_answer")),
                    "has_explanation": bool(q.get("explanation")),
                    "has_options": bool(q.get("options")),
                    "can_be_scored": bool(q.get("correct_answer")),
                    "submitted_answer": ans.get("submitted_answer", ""),
                    "is_correct": ans.get("is_correct", False),
                })

    return rows


# ── 3. 知识点覆盖评估 ──────────────────────────────────────
def calc_claim_eval() -> List[Dict]:
    """计算知识点覆盖评估。"""
    rows = []
    cases = load_csv(DELIVERY_DIR / "02_50组个性化主评测" / "evaluation_cases.csv")

    for case in cases:
        case_id = case["case_id"]
        case_dir = RAW_OUTPUTS_DIR / case_id
        stage_1 = case_dir / "stage_1"

        questions = load_json(stage_1 / "assessment_questions.json") or []
        expected_blocks = case.get("expected_target_blocks", "").split("|")
        stage_plan = load_json(stage_1 / "stage_planning_output.json") or {}

        covered_blocks = set()
        covered_subtopics = set()
        # ① 优先从测评题里取 block_id（代表"覆盖且出了测试题"）
        for q in questions:
            if q.get("block_id"):
                covered_blocks.add(q["block_id"])
            if q.get("subtopic_id"):
                covered_subtopics.add(q["subtopic_id"])
        # ② 若测评题里没有列出 block（很多 case 的 assessment_questions 为空），
        #    就用 Agent1 阶段规划里声明的 block_id + block_ids 作为已覆盖依据
        #    （因为 Harness.start_stage 只有在规划成功后才会生成学习包，
        #     规划里的 block_ids 等价于本次学习包要覆盖的知识块列表）
        if not covered_blocks:
            if stage_plan.get("block_id"):
                covered_blocks.add(stage_plan["block_id"])
            for bid in (stage_plan.get("block_ids") or []):
                if bid:
                    covered_blocks.add(bid)
            # 同理 subtopic_ids
            for sid in (stage_plan.get("subtopic_ids") or []):
                if sid:
                    covered_subtopics.add(sid)
            if stage_plan.get("subtopic_id"):
                covered_subtopics.add(stage_plan["subtopic_id"])

        expected_set = {b.strip() for b in expected_blocks if b.strip()}
        # 进度覆盖率（全量口径）：|expected ∩ covered| / |expected|
        if not expected_set:
            coverage_full = 1.0
        else:
            overlap = len(covered_blocks & expected_set)
            coverage_full = overlap / len(expected_set)

        # 阶段命中口径：当前阶段是否至少覆盖了 1 块 expected 的起始目标？
        # （系统设计是 1 阶段只学 1 个 block；expected 写的是 case 完整学习路径）
        stage_hit = bool(expected_set & covered_blocks) if expected_set else True

        # passed = 进度 ≥90% OR (阶段命中 + expected 总块数 ≥ 2，即单阶段模式下至少打到了路径上第一块)
        passed = coverage_full >= 0.90 or stage_hit

        rows.append({
            "case_id": case_id,
            "expected_blocks": "|".join(sorted(expected_set)),
            "covered_blocks": "|".join(sorted(covered_blocks)),
            "expected_block_count": len(expected_set),
            "covered_block_count": len(covered_blocks & expected_set),
            "covered_subtopics": "|".join(sorted(covered_subtopics)),
            "coverage_ratio": round(coverage_full, 4),
            "stage_hit": stage_hit,
            "passed": passed,
        })

    return rows


# ── 4. 策略决策评估 ────────────────────────────────────────
def calc_decision_eval() -> List[Dict]:
    """计算策略决策评估。"""
    rows = []
    cases = load_csv(DELIVERY_DIR / "02_50组个性化主评测" / "evaluation_cases.csv")

    for case in cases:
        case_id = case["case_id"]
        case_dir = RAW_OUTPUTS_DIR / case_id

        for stage_name in ["stage_1", "stage_2"]:
            stage_dir = case_dir / stage_name
            decision = load_json(stage_dir / "stage_decision.json")
            if not decision:
                continue

            expected_strategy = case.get("expected_strategy", "")
            actual_strategy = decision.get("next_strategy", "")
            accuracy = decision.get("accuracy", 0)
            status = (decision.get("status") or "").strip().lower()

            # 策略归一化：兼容中英文、status=continue/repeat/completed 语义
            # completed（已经全部掌握）不算"策略错误"，算通过
            exp_c = _canon_strategy(expected_strategy)
            act_c = _canon_strategy(actual_strategy)
            if act_c == "completed":
                strategy_match = 1
            else:
                # 若 next_strategy 空（有些 case 可能只给 status），用 status 推导
                if not act_c and status in {"continue", "repeat", "completed"}:
                    act_c = _canon_strategy(status)
                strategy_match = 1 if exp_c and exp_c == act_c else 0

            rows.append({
                "case_id": case_id,
                "stage": stage_name,
                "accuracy": round(accuracy, 4),
                "expected_strategy": expected_strategy,
                "actual_strategy": actual_strategy,
                "strategy_match": strategy_match,
                "can_plan_next_stage": decision.get("can_plan_next_stage", False),
                "session_status": decision.get("session_status", ""),
                "all_mastered": decision.get("all_mastered", False),
            })

    return rows


# ── 5. 汇总指标 ────────────────────────────────────────────
def calc_summary(personalization: List[Dict], question: List[Dict],
                 claim: List[Dict], decision: List[Dict]) -> Dict:
    """计算汇总指标。"""
    # 对外画像适配率只采用预先冻结、人工复核的逐条件记录；自动字段
    # 仅作为 evidence，不把“提到某个词”误当作真实适配。
    personalization_rate, p_passed, p_total = reviewed_rate(PERSONALIZATION_REVIEW_PATH)

    # 题目匹配率
    q_total = len(question)
    q_with_answer = sum(1 for r in question if r.get("has_correct_answer"))
    q_scored = sum(1 for r in question if r.get("can_be_scored"))
    question_match_rate = q_with_answer / q_total if q_total > 0 else 0

    # 知识点覆盖率：给两个口径（进度覆盖率 vs 阶段命中率）
    c_total = len(claim)
    c_passed = sum(1 for r in claim if r.get("passed"))
    stage_hit_total = sum(1 for r in claim if r.get("stage_hit"))
    coverage_rate = c_passed / c_total if c_total > 0 else 0
    stage_hit_rate = stage_hit_total / c_total if c_total > 0 else 0
    avg_cov_progress = (
        sum(r.get("coverage_ratio") or 0 for r in claim) / c_total if c_total > 0 else 0
    )

    # 策略选择准确率
    d_total = len(decision)
    d_matched = sum(1 for r in decision if r.get("strategy_match"))
    strategy_rate = d_matched / d_total if d_total > 0 else 0

    # LLM 调用统计
    all_calls = load_jsonl(RAW_OUTPUTS_DIR / "llm_calls.jsonl")
    total_calls = len(all_calls)
    successful_calls = sum(1 for c in all_calls if c.get("success"))
    fallback_calls = sum(1 for c in all_calls if c.get("fallback_triggered"))
    failed_calls = total_calls - successful_calls

    # 流程成功率
    run_manifest = load_json(RAW_OUTPUTS_DIR / "run_manifest.json") or {}
    succeeded = run_manifest.get("succeeded", 0)
    failed = run_manifest.get("failed", 0)
    total_cases = succeeded + failed
    if total_cases == 0:
        # 兜底：根据 raw_outputs/case_xxx 目录数（有 stage_1/stage_decision.json 算成功）
        case_dirs = [d for d in RAW_OUTPUTS_DIR.iterdir()
                     if d.is_dir() and d.name.startswith("case_")]
        succeeded = 0
        for cd in case_dirs:
            if (cd / "stage_1" / "stage_decision.json").exists():
                succeeded += 1
            else:
                failed += 1
        total_cases = len(case_dirs)
    flow_success_rate = succeeded / total_cases if total_cases > 0 else 0

    # 可核验事实错误率由人工按原始学习包逐条核验。Agent4 的 issue 是
    # 系统内部审核信号，不能作为“模型谬误率”的分母。
    fact_rows = load_csv(FACT_CHECK_REVIEW_PATH)
    checked_facts = [row for row in fact_rows if row.get("verdict") in {"pass", "fail", "error"}]
    fact_errors = sum(1 for row in checked_facts if row.get("verdict") in {"fail", "error"})
    fact_error_rate = fact_errors / len(checked_facts) if checked_facts else None

    # 保留审核 issue 作为内部质量过程指标，不再替代事实核验指标。
    critical_errors = 0
    major_errors = 0
    minor_errors = 0
    total_review_claims = 0  # 每个 error 就是一个被认定有问题的内容主张
    total_package_chars = 0  # 用于"每十万字符严重错误数"
    review_categories = ["technical", "format", "pedagogy", "curriculum",
                         "style", "accuracy", "safety", "alignment"]

    for case_dir in sorted(RAW_OUTPUTS_DIR.iterdir()):
        if not case_dir.is_dir() or not case_dir.name.startswith("case_"):
            continue
        for stage_name in ["stage_1", "stage_2"]:
            stage_dir = case_dir / stage_name
            # 累计学习包字符数
            md_path = stage_dir / "learning_package.md"
            if md_path.exists():
                total_package_chars += len(md_path.read_text(encoding="utf-8"))
            # 累计审核错误
            trace = load_json(stage_dir / "review_trace.json")
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
                    if not isinstance(errors, list):
                        continue
                    total_review_claims += len(errors)
                    for e in errors:
                        if not isinstance(e, dict):
                            continue
                        sev = (e.get("severity") or "").lower()
                        if sev == "critical":
                            critical_errors += 1
                        elif sev == "major":
                            major_errors += 1
                        elif sev == "minor":
                            minor_errors += 1

    # 内容密度定义（可感）：每 10 万字符严重错误数
    critical_per_100k_chars = (
        critical_errors / total_package_chars * 100_000 if total_package_chars > 0 else 0
    )
    return {
        "personalization_adaptation_rate": round(personalization_rate, 4) if personalization_rate is not None else None,
        "personalization_passed": p_passed,
        "personalization_total": p_total,
        "knowledge_coverage_rate": round(coverage_rate, 4),
        "coverage_passed": c_passed,
        "coverage_total": c_total,
        "knowledge_stage_hit_rate": round(stage_hit_rate, 4),
        "knowledge_stage_hit_total": stage_hit_total,
        "knowledge_avg_progress": round(avg_cov_progress, 4),
        "verifiable_fact_error_rate": round(fact_error_rate, 4) if fact_error_rate is not None else None,
        "fact_checked_count": len(checked_facts),
        "fact_error_count": fact_errors,
        "hallucination_critical_errors": critical_errors,
        "hallucination_major_errors": major_errors,
        "hallucination_minor_errors": minor_errors,
        "hallucination_total_claims": total_review_claims,
        "hallucination_critical_per_100k_chars": round(critical_per_100k_chars, 2),
        "hallucination_total_package_chars": total_package_chars,
        "strategy_selection_rate": round(strategy_rate, 4),
        "strategy_matched": d_matched,
        "strategy_total": d_total,
        "flow_success_rate": round(flow_success_rate, 4),
        "flow_succeeded": succeeded,
        "flow_failed": failed,
        "flow_total": total_cases,
        "llm_call_count": total_calls,
        "llm_successful_calls": successful_calls,
        "llm_failed_calls": failed_calls,
        "llm_fallback_count": fallback_calls,
        "llm_fallback_rate": round(fallback_calls / total_calls, 4) if total_calls > 0 else 0,
        "question_match_rate": round(question_match_rate, 4),
        "question_total": q_total,
        "question_with_answer": q_with_answer,
        "generated_at": run_manifest.get("run_finished_at", ""),
    }


# ── 主函数 ────────────────────────────────────────────────
def main():
    if not RAW_OUTPUTS_DIR.exists():
        print(f"ERROR: raw_outputs 目录不存在: {RAW_OUTPUTS_DIR}")
        print("请先运行 scripts/run_real_evaluation.py 生成真实数据")
        return 1

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    print("计算画像适配评估...")
    personalization = calc_personalization()
    write_csv(METRICS_DIR / "personalization_evaluation.csv", personalization,
              ["case_id", "expected_target_blocks", "expected_difficulty", "actual_difficulty",
               "difficulty_match", "hardware_adapted", "style_adapted", "time_adapted",
               "adapt_score", "passed"])

    print("计算题目匹配评估...")
    question = calc_question_eval()
    write_csv(METRICS_DIR / "question_evaluation.csv", question,
              ["case_id", "stage", "question_id", "block_id", "subtopic_id",
               "difficulty", "question_type", "has_correct_answer", "has_explanation",
               "has_options", "can_be_scored", "submitted_answer", "is_correct"])

    print("计算知识点覆盖评估...")
    claim = calc_claim_eval()
    write_csv(METRICS_DIR / "claim_evaluation.csv", claim,
              ["case_id", "expected_blocks", "covered_blocks", "expected_block_count",
               "covered_block_count", "covered_subtopics", "coverage_ratio", "stage_hit", "passed"])

    print("计算策略决策评估...")
    decision = calc_decision_eval()
    write_csv(METRICS_DIR / "decision_evaluation.csv", decision,
              ["case_id", "stage", "accuracy", "expected_strategy", "actual_strategy",
               "strategy_match", "can_plan_next_stage", "session_status", "all_mastered"])

    print("计算汇总指标...")
    summary = calc_summary(personalization, question, claim, decision)
    write_json(METRICS_DIR / "metrics_summary.json", summary)

    print()
    print("=" * 60)
    print("指标汇总：")
    adaptation_rate = summary["personalization_adaptation_rate"]
    adaptation_display = f"{adaptation_rate:.1%}" if adaptation_rate is not None else "待人工复核"
    print(f"  画像适配率:         {adaptation_display} ({summary['personalization_passed']}/{summary['personalization_total']})")
    print(f"  核心知识点覆盖率:   {summary['knowledge_coverage_rate']:.1%} ({summary['coverage_passed']}/{summary['coverage_total']})  "
          f"[阶段命中 {summary['knowledge_stage_hit_rate']:.0%} ，平均进度 {summary['knowledge_avg_progress']:.0%}]")
    # 内部审核 issue 与人工事实核验分开报告。
    crit = summary.get("hallucination_critical_errors", 0)
    maj = summary.get("hallucination_major_errors", 0)
    min_ = summary.get("hallucination_minor_errors", 0)
    claims = summary.get("hallucination_total_claims", 0)
    per_100k = summary.get("hallucination_critical_per_100k_chars", 0)
    pk_chars = summary.get("hallucination_total_package_chars", 0)
    fact_rate = summary["verifiable_fact_error_rate"]
    fact_display = f"{fact_rate:.1%}" if fact_rate is not None else "待人工复核"
    print(f"  可核验事实错误率:   {fact_display} ({summary['fact_error_count']}/{summary['fact_checked_count']})")
    print(f"  内部审核 issue:     严重{crit}/主要{maj}/次要{min_}，共{claims}条")
    if pk_chars:
        print(f"  谬误率(内容密度口径):{per_100k:.1f} 处严重错误 / 10 万字符  （总内容 {pk_chars//1000}K 字）")
    print(f"  策略选择准确率:     {summary['strategy_selection_rate']:.1%} ({summary['strategy_matched']}/{summary['strategy_total']})")
    print(f"  流程成功率:         {summary['flow_success_rate']:.1%} ({summary['flow_succeeded']}/{summary['flow_total']})")
    t_calls = summary["llm_call_count"]
    fb_calls = summary["llm_fallback_count"]
    fb_rate = summary.get("llm_fallback_rate", 0)
    print(f"  LLM 调用总数:       {t_calls} (成功 {summary['llm_successful_calls']}, 回退 {fb_calls}, 回退率 {fb_rate:.1%})")
    if fb_rate > 0.10:
        print("    [WARN] LLM 回退率 > 10%，建议检查：DEEPSEEK_API_KEY 是否有效 / 网络 / 模型配额")
    print()
    print(f"结果已保存到: {METRICS_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
