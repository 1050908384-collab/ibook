"""Create human-review templates from the same 50 main-evaluation cases.

This script never invokes a model.  Reviewers fill the generated CSV files
after ``run_real_evaluation.py --all`` has produced the one authoritative
``raw_outputs`` set.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "测试数据与评测材料" / "02_50组个性化主评测"
METRICS = ROOT / "测试数据与评测材料" / "06_指标统计"


def read_cases() -> list[dict[str, str]]:
    with (MAIN / "evaluation_cases.csv").open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_if_missing(path: Path, rows: list[dict[str, str]], fields: list[str], force: bool) -> None:
    if path.exists() and not force:
        print(f"[SKIP] {path.name} already exists; use --force only before review starts")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] {path.name}: {len(rows)} rows")


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    personalization_rows = []
    fact_rows = []
    strategy_rows = []
    dynamic_loop_rows = []
    for case in read_cases():
        case_id = case["case_id"]
        conditions = [
            (
                "target_path", "目标与路径适配：Agent1 的 block_id/subtopic_ids 命中 expected_target_blocks，满足 prerequisites；Agent3 学习包覆盖该阶段全部 subtopic_ids。",
                "Agent1 + Agent3", "stage_1/diagnostic_output.json; stage_1/stage_planning_output.json; stage_1/learning_package.md",
            ),
            (
                "difficulty", f"难度适配：Agent1 difficulty 与预期 {case['expected_difficulty']} 一致或有合理前置补偿；Agent3 的解释深度、代码和练习不超出当前阶段能力。",
                "Agent1 + Agent3", "stage_1/stage_planning_output.json; stage_1/learning_package.md; stage_1/assessment_questions.json",
            ),
            (
                "time_budget", f"时间适配：Agent1 estimated_hours、recommended_daily_hours、suggested_study_days 与每日 {case['daily_hours']} 小时相容；Agent3 学习包工作量与该阶段范围相称。",
                "Agent1 + Agent3", "stage_1/stage_planning_output.json; stage_1/learning_package.md",
            ),
            (
                "hardware", f"硬件适配：Agent1 hardware_notes 识别用户条件“{case['hardware']}”；Agent3 的模型、参数、命令和实训步骤在该条件下可执行，或提供明确替代方案。",
                "Agent1 + Agent3", "stage_1/stage_planning_output.json; stage_1/learning_package.md",
            ),
            (
                "learning_style", f"风格适配：Agent1 learning_style_notes 回应偏好“{case['learning_style']}”；Agent3 的学习包结构、解释和实操比例实际体现该偏好。",
                "Agent1 + Agent3", "stage_1/stage_planning_output.json; stage_1/learning_package.md",
            ),
        ]
        for condition_id, expected_rule, agent_owner, evidence_path in conditions:
            personalization_rows.append({
                "case_id": case_id, "condition_id": condition_id,
                "agent_owner": agent_owner, "expected_rule": expected_rule, "evidence_path": evidence_path,
                "actual_evidence": "", "verdict": "not_reviewed", "reviewer_id": "", "notes": "",
            })
        for claim_index in range(1, 4):
            fact_rows.append({
                "case_id": case_id, "stage": "stage_1", "claim_index": str(claim_index),
                "source_path": "stage_1/learning_package.md", "claim_text": "",
                "evidence_source": "", "verdict": "not_reviewed", "reviewer_id": "", "notes": "",
            })

        for stage_name in ("stage_1", "stage_2"):
            decision_path = MAIN / "raw_outputs" / case_id / stage_name / "stage_decision.json"
            if not decision_path.exists():
                continue
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            raw_script = case.get(f"answer_script_{stage_name}", "")
            try:
                script = json.loads(raw_script) if raw_script.strip().startswith("{") else {}
            except json.JSONDecodeError:
                script = {}
            expected = script.get("expected_next_strategy") or (
                case.get("expected_next_strategy", "") if stage_name == "stage_1" else ""
            )
            strategy_rows.append({
                "case_id": case_id, "stage": stage_name,
                "expected_strategy": expected,
                "actual_strategy": decision.get("next_strategy", ""),
                "accuracy": decision.get("accuracy", ""),
                "evidence_path": f"{stage_name}/stage_decision.json; evaluation_cases.csv",
                "actual_evidence": "", "verdict": "not_reviewed", "reviewer_id": "", "notes": "",
            })

        stage_2_dir = MAIN / "raw_outputs" / case_id / "stage_2"
        if stage_2_dir.exists():
            stage_1_decision = json.loads((MAIN / "raw_outputs" / case_id / "stage_1" / "stage_decision.json").read_text(encoding="utf-8"))
            stage_2_plan_raw = json.loads((stage_2_dir / "stage_planning_output.json").read_text(encoding="utf-8"))
            # Stage-2 planning is returned by the next-stage API and may wrap
            # the actual plan in next_stage; preserve an empty subtopic list as
            # review evidence rather than inventing coverage from the package.
            stage_2_plan = stage_2_plan_raw.get("next_stage", stage_2_plan_raw)
            loop_checks = [
                (
                    "agent6_to_agent1", "Agent1 的 stage_2 规划应接收并响应 stage_1 的 Agent6 next_strategy；策略、知识点范围、难度或前置补偿之间应有可追溯联系。",
                    "stage_1/stage_decision.json; stage_2/stage_planning_output.json",
                ),
                (
                    "agent1_replanning", "Agent1 的 stage_2 subtopic_ids、difficulty、estimated_hours 应与用户更新后的掌握状态和策略相符，而不是机械复制 stage_1。",
                    "stage_1/stage_decision.json; stage_2/learner_profile_snapshot.json; stage_2/stage_planning_output.json",
                ),
                (
                    "agent3_regeneration", "Agent3 的 stage_2 学习包、题目和实训内容应覆盖 stage_2 全部 subtopic_ids，并体现 Agent1 的重新规划结果。",
                    "stage_2/stage_planning_output.json; stage_2/learning_package.md; stage_2/assessment_questions.json",
                ),
            ]
            for check_id, expected_rule, evidence_path in loop_checks:
                dynamic_loop_rows.append({
                    "case_id": case_id, "check_id": check_id,
                    "stage_1_strategy": stage_1_decision.get("next_strategy", ""),
                    "stage_2_subtopic_ids": "|".join(stage_2_plan.get("subtopic_ids") or []),
                    "expected_rule": expected_rule, "evidence_path": evidence_path,
                    "actual_evidence": "", "verdict": "not_reviewed", "reviewer_id": "", "notes": "",
                })

    write_if_missing(METRICS / "personalization_review.csv", personalization_rows,
                     list(personalization_rows[0]), args.force)
    write_if_missing(METRICS / "fact_check_review.csv", fact_rows,
                     list(fact_rows[0]), args.force)
    write_if_missing(METRICS / "strategy_review.csv", strategy_rows,
                     list(strategy_rows[0]), args.force)
    write_if_missing(METRICS / "dynamic_loop_review.csv", dynamic_loop_rows,
                     list(dynamic_loop_rows[0]), args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
