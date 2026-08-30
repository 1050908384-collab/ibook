"""补齐并校验50组主评测场景的可追溯元数据。

用法：
    python scripts/prepare_evaluation_cases.py --write
    python scripts/prepare_evaluation_cases.py --check

--write 只在正式测试开始前使用一次；场景冻结后只能使用 --check。
"""

from __future__ import annotations


import sys
from pathlib import Path
_PROJ_ROOT_ = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT_) not in sys.path: sys.path.insert(0, str(_PROJ_ROOT_))
_BE_ = Path(_PROJ_ROOT_, "backend")
if str(_BE_) not in sys.path: sys.path.insert(0, str(_BE_))

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
EVALUATION_DIR = ROOT_DIR / "测试数据与评测材料" / "02_50组个性化主评测"
CSV_PATH = EVALUATION_DIR / "evaluation_cases.csv"
JSONL_PATH = EVALUATION_DIR / "evaluation_cases.jsonl"

CATEGORY_RANGES = (
    (range(1, 9), "学情诊断与前置补全"),
    (range(9, 16), "已掌握知识识别与跳过"),
    (range(16, 26), "LoRA/QLoRA实训适配"),
    (range(26, 34), "vLLM部署与工程适配"),
    (range(34, 40), "时间与学习风格适配"),
    (range(45, 51), "测试反馈与路径调整"),
    (range(46, 51), "异常与边界输入"),
)

EXPECTED_CATEGORY_COUNTS = {
    "学情诊断与前置补全": 8,
    "已掌握知识识别与跳过": 7,
    "LoRA/QLoRA实训适配": 10,
    "vLLM部署与工程适配": 8,
    "时间与学习风格适配": 6,
    "测试反馈与路径调整": 6,
    "异常与边界输入": 5,
}

SKILL_DEFAULTS = {
    "case_001": "无Python、深度学习与大模型相关基础",
    "case_005": "无Python、深度学习与大模型相关基础",
    "case_006": "无Transformer与大模型相关基础",
    "case_008": "无Python、深度学习与大模型相关基础",
    "case_043": "无Python、深度学习与大模型相关基础",
    "case_049": "无Python、深度学习与大模型相关基础",
    "case_050": "未提供，系统应先追问学习基础",
}

# 为此前未作为预期目标出现的核心板块补齐代表案例，避免50组目标只集中在LoRA与部署。
TARGET_BLOCK_OVERRIDES = {
    "case_006": "KB-01|KB-02|KB-07|KB-08",
    "case_024": "KB-12|KB-13|KB-14",
    "case_034": "KB-08|KB-09|KB-10",
}


def case_number(case_id: str) -> int:
    try:
        return int(case_id.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"非法 case_id: {case_id}") from exc


def category_for(case_id: str) -> str:
    number = case_number(case_id)
    for number_range, category in CATEGORY_RANGES:
        if number in number_range:
            return category
    raise ValueError(f"case_id 超出正式50组范围: {case_id}")


def build_tags(row: dict[str, str]) -> str:
    tags = [category_for(row["case_id"])]
    for value in (row.get("learning_goal", ""), row.get("learning_style", ""), row.get("hardware", "")):
        if value and value not in tags:
            tags.append(value)
    if row.get("run_stage_2", "").strip().lower() in {"true", "1", "yes"}:
        tags.append("multi_stage")
    return "|".join(tags)


def normalize(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    for row in rows:
        case_id = row["case_id"]
        row["primary_category"] = category_for(case_id)
        row["tags"] = build_tags(row)
        if not row.get("existing_skills", "").strip():
            row["existing_skills"] = SKILL_DEFAULTS.get(case_id, "未提供")
        if case_id in TARGET_BLOCK_OVERRIDES:
            row["expected_target_blocks"] = TARGET_BLOCK_OVERRIDES[case_id]
        targets = [item.strip() for item in row["expected_target_blocks"].split("|") if item.strip()]
        row["expected_first_stage"] = targets[0]
    return rows


def validate(rows: list[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    required_fields = (
        "case_id", "primary_category", "tags", "background", "existing_skills",
        "learning_goal", "daily_hours", "learning_style", "hardware",
        "expected_target_blocks", "expected_first_stage", "expected_difficulty",
        "expected_strategy", "run_stage_2", "answer_script_stage_1",
    )
    if len(rows) != 50:
        errors.append(f"场景数量必须为50，当前为{len(rows)}")

    ids = [row.get("case_id", "") for row in rows]
    if len(set(ids)) != len(ids):
        errors.append("case_id 存在重复")
    expected_ids = {f"case_{number:03d}" for number in range(1, 51)}
    if set(ids) != expected_ids:
        errors.append("case_id 必须完整覆盖 case_001 至 case_050")

    for row in rows:
        missing = [field for field in required_fields if not row.get(field, "").strip()]
        if missing:
            errors.append(f"{row.get('case_id', '<unknown>')} 缺少字段：{','.join(missing)}")

    category_counts = Counter(row.get("primary_category", "") for row in rows)
    if dict(category_counts) != EXPECTED_CATEGORY_COUNTS:
        errors.append(f"主类别数量不符合冻结方案：{dict(category_counts)}")

    multi_stage_ids = {
        row["case_id"] for row in rows
        if row.get("run_stage_2", "").strip().lower() in {"true", "1", "yes"}
    }
    expected_multi_stage_ids = {f"case_{number:03d}" for number in range(45, 51)}
    if multi_stage_ids != expected_multi_stage_ids:
        errors.append("多阶段案例必须是CSV中run_stage_2=True的连续区间（共6个）")

    # answer_script_stage_1 / stage_2 JSON 完整性校验（文档 §3.1）
    LEGAL_BEHAVIORS = {"high", "medium", "low", "very_low"}
    LEGAL_STRATEGIES = {"fixed_by_question_id", "random_distractor"}
    for row in rows:
        cid = row.get("case_id", "<unknown>")
        for stage_field in ("answer_script_stage_1", "answer_script_stage_2"):
            raw = row.get(stage_field, "")
            if stage_field == "answer_script_stage_2" and (
                row.get("run_stage_2", "").strip().lower() not in {"true", "1", "yes"}
            ):
                # 非多阶段案例：answer_script_stage_2 允许空
                continue
            if not isinstance(raw, str) or not raw.strip():
                errors.append(f"{cid} {stage_field} 不能为空（必须为 JSON 结构）")
                continue
            try:
                obj = json.loads(raw.strip()) if isinstance(raw, str) else raw
            except json.JSONDecodeError as exc:
                errors.append(f"{cid} {stage_field} 不是合法 JSON：{exc}")
                continue
            if not isinstance(obj, dict):
                errors.append(f"{cid} {stage_field} 必须是 JSON 对象（dict）")
                continue
            missing_sub = [
                k for k in (
                    "answer_behavior", "target_accuracy_range",
                    "wrong_answer_strategy", "expected_next_strategy",
                )
                if k not in obj or (isinstance(obj[k], str) and not obj[k].strip())
            ]
            if missing_sub:
                errors.append(f"{cid} {stage_field} 缺少子字段：{','.join(missing_sub)}")
                continue
            if obj["answer_behavior"] not in LEGAL_BEHAVIORS:
                errors.append(
                    f"{cid} {stage_field}.answer_behavior={obj['answer_behavior']} "
                    f"不在合法集合 {LEGAL_BEHAVIORS}"
                )
            rng = obj.get("target_accuracy_range")
            if not (isinstance(rng, (list, tuple)) and len(rng) == 2
                    and all(isinstance(v, (int, float)) for v in rng)
                    and 0.0 <= float(rng[0]) <= float(rng[1]) <= 1.0):
                errors.append(
                    f"{cid} {stage_field}.target_accuracy_range 必须是 [a, b] 且 0≤a≤b≤1，当前：{rng}"
                )
            if obj["wrong_answer_strategy"] not in LEGAL_STRATEGIES:
                errors.append(
                    f"{cid} {stage_field}.wrong_answer_strategy={obj['wrong_answer_strategy']} "
                    f"不在合法集合 {LEGAL_STRATEGIES}"
                )

    covered_blocks = {
        block.strip()
        for row in rows
        for block in row.get("expected_target_blocks", "").split("|")
        if block.strip()
    }
    expected_blocks = {f"KB-{number:02d}" for number in range(1, 16)}
    missing_blocks = sorted(expected_blocks - covered_blocks)
    if missing_blocks:
        errors.append(f"预期目标未覆盖核心板块：{','.join(missing_blocks)}")
    if JSONL_PATH.exists():
        with JSONL_PATH.open("r", encoding="utf-8") as handle:
            jsonl_rows = [json.loads(line) for line in handle if line.strip()]
        if jsonl_rows != rows:
            errors.append("evaluation_cases.csv 与 evaluation_cases.jsonl 内容不一致")
    else:
        errors.append("evaluation_cases.jsonl 不存在")
    return errors


def read_csv() -> tuple[list[dict[str, str]], list[str]]:
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), reader.fieldnames or []


def write_cases(rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    additions = ["primary_category", "tags", "expected_first_stage"]
    output_fields = fieldnames + [field for field in additions if field not in fieldnames]
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(rows)
    with JSONL_PATH.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="50组主评测场景预处理与预检")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="补齐冻结元数据并同步CSV/JSONL")
    mode.add_argument("--check", action="store_true", help="只校验已冻结场景，不写文件")
    args = parser.parse_args()

    rows, fieldnames = read_csv()
    if args.write:
        rows = normalize(rows)
        write_cases(rows, fieldnames)
        print("已补齐主类别、标签、首阶段预期，并同步 evaluation_cases.csv/jsonl")

    errors = validate(rows)
    if errors:
        print("场景预检失败：")
        for error in errors:
            print(f"- {error}")
        return 1

    print("场景预检通过：50组、7类分布、15个知识板块、6个多阶段案例、必填字段完整。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
