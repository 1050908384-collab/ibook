"""Create human-review templates from the same 50 main-evaluation cases.

This script never invokes a model.  Reviewers fill the generated CSV files
after ``run_real_evaluation.py --all`` has produced the one authoritative
``raw_outputs`` set.
"""
from __future__ import annotations

import csv
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
    for case in read_cases():
        case_id = case["case_id"]
        conditions = [
            ("stage_target_path", "Agent1 选择的 block_id/subtopic_ids 应匹配用户目标、已有基础与 expected_target_blocks", "stage_1/diagnostic_output.json; stage_1/stage_planning_output.json"),
            ("stage_prerequisites", "Agent1 不得跳过硬前置知识；若补前置，应给出合理原因", "stage_1/diagnostic_output.json; stage_1/stage_planning_output.json"),
            ("stage_difficulty", f"Agent1 阶段难度应适配预期 {case['expected_difficulty']} 或有明确前置补偿理由", "stage_1/stage_planning_output.json"),
            ("stage_time_budget", f"Agent1 计划时长应适配每日 {case['daily_hours']} 小时预算", "stage_1/stage_planning_output.json"),
            ("package_difficulty", "Agent3 学习包的讲解、代码和练习难度应与当前阶段及学习者基础一致", "stage_1/stage_planning_output.json; stage_1/learning_package.md"),
            ("package_hardware", f"Agent3 给出的模型、参数、命令和实操方案应可在硬件条件下执行：{case['hardware']}", "stage_1/learning_package.md"),
            ("package_learning_style", f"Agent3 的资源组织和练习方式应体现学习偏好：{case['learning_style']}", "stage_1/learning_package.md"),
            ("package_practice_feasibility", "Agent3 的实训步骤应具备可执行前置、明确输入输出和安全边界", "stage_1/learning_package.md"),
            ("stage_strategy", "Agent6 阶段后策略应与实际 accuracy、掌握状态、迭代历史及测试用例预期策略一致", "stage_1/stage_decision.json; evaluation_cases.csv"),
        ]
        for condition_id, expected_rule, evidence_path in conditions:
            personalization_rows.append({
                "case_id": case_id, "condition_id": condition_id,
                "expected_rule": expected_rule, "evidence_path": evidence_path,
                "actual_evidence": "", "verdict": "not_reviewed", "reviewer_id": "", "notes": "",
            })
        for claim_index in range(1, 4):
            fact_rows.append({
                "case_id": case_id, "stage": "stage_1", "claim_index": str(claim_index),
                "source_path": "stage_1/learning_package.md", "claim_text": "",
                "evidence_source": "", "verdict": "not_reviewed", "reviewer_id": "", "notes": "",
            })

    write_if_missing(METRICS / "personalization_review.csv", personalization_rows,
                     list(personalization_rows[0]), args.force)
    write_if_missing(METRICS / "fact_check_review.csv", fact_rows,
                     list(fact_rows[0]), args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
