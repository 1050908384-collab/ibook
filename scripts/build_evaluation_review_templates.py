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
            ("target_path", "阶段 block_id/subtopic_ids 应命中 expected_target_blocks 的当前起始学习路径", "stage_1/stage_planning_output.json"),
            ("difficulty", f"阶段难度应与预期 {case['expected_difficulty']} 一致或有明确、合理的前置补偿理由", "stage_1/stage_planning_output.json"),
            ("time_budget", f"计划应适配每日 {case['daily_hours']} 小时的时间预算", "stage_1/stage_planning_output.json"),
            ("hardware", f"学习包或阶段说明应回应硬件约束：{case['hardware']}", "stage_1/stage_planning_output.json; stage_1/learning_package.md"),
            ("learning_style", f"学习包的内容组织应体现偏好：{case['learning_style']}", "stage_1/learning_package.md"),
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
