# -*- coding: utf-8 -*-
"""Build the runtime question seed from the curated Markdown question bank.

The source course uses historical section numbers.  The mapping file is the
single authority that links those sections to the current KBS identifiers.
Questions without an approved mapping are written to an audit report instead
of being assigned to a misleading block.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEEDS = ROOT / "data" / "seeds"
SOURCE = SEEDS / "source" / "测试题目.md"
MAPPING = SEEDS / "question_source_mapping.json"
SUBTOPICS = SEEDS / "knowledge_subtopics.json"
OUTPUT = SEEDS / "test_questions.json"
AUDIT = SEEDS / "test_question_mapping_audit.json"
SUPPLEMENTAL = SEEDS / "supplemental_test_questions.json"

TOPIC_RE = re.compile(r"^#{3,4}\s*知识点\s*(\d+)\.(\d+)[：:]")
BLOCK_RE = re.compile(r"^#{2,4}\s*板块\s*([一二三四五六七八九十]+|\d+)[：:]")
QUESTION_RE = re.compile(r"^\*\*题\s*(\d+)\s*[^*]*\*\*")
TYPE_RE = re.compile(r"[（(](选择|填空|判断(?:\+解释)?|实操|代码阅读|代码|简答|流程|评估)[）)]")
# Source answers use both ``**答案：B**`` and ``**答案：** B``.  Accepting
# both is essential: otherwise most fill-in and practical questions are
# silently omitted from the runtime bank.
ANSWER_RE = re.compile(r"^✅\s*\*\*答案：(?:([^*]*?)\*\*|\*\*\s*(.*))\s*$")
EXPECTED_RE = re.compile(r"^✅\s*\*\*预期输出(?:示例|应包含)?：(?:([^*]*?)\*\*|\*\*\s*(.*))\s*$")
OPTION_RE = re.compile(r"^([A-D])\.\s*(.+)$")
INLINE_OPTION_RE = re.compile(r"(?:^|\s)([A-D])\.\s*")

TYPE_MAP = {
    "选择": "single_choice",
    "填空": "fill_blank",
    "判断": "true_false",
    "判断+解释": "true_false",
    "实操": "practical",
    "代码阅读": "short_answer",
    "代码": "short_answer",
    "简答": "short_answer",
    "流程": "short_answer",
    "评估": "short_answer",
}
PRACTICAL_TOPICS = {"1.1", "2.1", "3.1"}

DIFFICULTY_MAP = (("高级", "高级"), ("进阶", "中级"), ("基础", "初级"))
CHINESE_NUMBERS = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
                   "十一": 11, "十二": 12, "十三": 13, "十四": 14, "十五": 15}


def clean(lines: list[str]) -> str:
    return "\n".join(line.rstrip() for line in lines).strip()


def question_type(header: str, topic: str) -> str:
    match = TYPE_RE.search(header)
    if match:
        return TYPE_MAP[match.group(1)]
    # The installation-validation headings intentionally omit the type label.
    if topic in PRACTICAL_TOPICS:
        return "practical"
    return "short_answer"


def difficulty(header: str) -> str:
    for marker, value in DIFFICULTY_MAP:
        if marker in header:
            return value
    return "初级"


def parse_questions(source: Path) -> tuple[list[dict], list[dict]]:
    lines = source.read_text(encoding="utf-8").splitlines()
    topic = ""
    parsed: list[dict] = []
    skipped: list[dict] = []
    index = 0
    while index < len(lines):
        topic_match = TOPIC_RE.match(lines[index])
        if topic_match:
            topic = f"{topic_match.group(1)}.{topic_match.group(2)}"
            index += 1
            continue
        question_match = QUESTION_RE.match(lines[index])
        if not question_match:
            index += 1
            continue

        header = lines[index]
        end = index + 1
        while end < len(lines) and not QUESTION_RE.match(lines[end]) and not TOPIC_RE.match(lines[end]) and not BLOCK_RE.match(lines[end]):
            end += 1
        body = lines[index + 1:end]
        answer_index = next((pos for pos, line in enumerate(body) if ANSWER_RE.match(line) or EXPECTED_RE.match(line)), None)
        if answer_index is None:
            skipped.append({"topic": topic, "question": header, "reason": "missing_answer_marker"})
            index = end
            continue
        answer_match = ANSWER_RE.match(body[answer_index]) or EXPECTED_RE.match(body[answer_index])
        stem_lines = body[:answer_index]
        inline_answer = ""
        if answer_match:
            inline_answer = next((group.strip() for group in answer_match.groups() if group and group.strip()), "")
        answer_lines = ([inline_answer] if inline_answer else []) + body[answer_index + 1:]
        explanation_lines = body[answer_index + 1:]

        options = []
        prompt_lines = []
        for line in stem_lines:
            inline_options = list(INLINE_OPTION_RE.finditer(line))
            if len(inline_options) >= 2 and inline_options[0].start() == 0:
                for option_index, match in enumerate(inline_options):
                    option_end = (
                        inline_options[option_index + 1].start()
                        if option_index + 1 < len(inline_options)
                        else len(line)
                    )
                    option_text = line[match.end():option_end].strip()
                    if option_text:
                        options.append({"key": match.group(1), "text": option_text})
                continue
            option_match = OPTION_RE.match(line.strip())
            if option_match:
                options.append({"key": option_match.group(1), "text": option_match.group(2).strip()})
            else:
                prompt_lines.append(line)
        prompt = clean(prompt_lines)
        answer = clean(answer_lines)
        if not prompt:
            skipped.append({"topic": topic, "question": header, "reason": "empty_prompt"})
            index = end
            continue
        parsed.append({
            "topic": topic,
            "ordinal": question_match.group(1),
            "header": header,
            "question_text": prompt,
            "options": options or None,
            "answer": answer,
            "explanation": clean(explanation_lines),
            "question_type": question_type(header, topic),
            "difficulty": difficulty(header),
        })
        index = end
    return parsed, skipped


def normalize_answer(record: dict) -> tuple[str, str | None, str]:
    answer = record["answer"]
    qtype = record["question_type"]
    if qtype == "single_choice":
        match = re.search(r"(?:答案：?\s*)?([A-D])(?:[。．、\s]|$)", answer)
        return (match.group(1) if match else answer, None, "exact_match")
    if qtype == "true_false":
        return ("正确" if "正确" in answer[:12] else "错误", None, "exact_match")
    if qtype == "practical":
        expected = answer
        commands = re.findall(r"```(?:bash|shell|powershell)?\s*\n(.*?)```", answer, re.DOTALL)
        # The grader compares learner output semantically; retain expected output
        # while using the command/reference answer as a non-empty fallback.
        reference = " ||| ".join(command.strip() for command in commands if command.strip()) or answer
        return (reference, expected, "command_output_check")
    return (answer, None, "semantic_match")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="validate without writing the seed")
    args = parser.parse_args()

    mapping = json.loads(MAPPING.read_text(encoding="utf-8"))["topic_mapping"]
    subtopics = json.loads(SUBTOPICS.read_text(encoding="utf-8"))
    valid = {item["subtopic_id"]: item["block_id"] for item in subtopics}
    parsed, skipped = parse_questions(SOURCE)
    records: list[dict] = []
    unmapped: list[dict] = []

    sequence_by_subtopic: Counter[str] = Counter()
    for item in parsed:
        subtopic_id = mapping.get(item["topic"])
        if not subtopic_id:
            unmapped.append({"topic": item["topic"], "question": item["header"], "reason": "no_runtime_mapping"})
            continue
        if subtopic_id not in valid:
            raise ValueError(f"Mapping points to unknown subtopic: {subtopic_id}")
        correct_answer, expected_output, grading_mode = normalize_answer(item)
        qtype = item["question_type"]
        sequence_by_subtopic[subtopic_id] += 1
        record = {
            "question_id": f"Q-{subtopic_id.replace('KBS-', 'KB').replace('-', '')}-{sequence_by_subtopic[subtopic_id]:03d}",
            "block_id": valid[subtopic_id],
            "subtopic_id": subtopic_id,
            "difficulty": item["difficulty"],
            "question_type": qtype,
            "question_text": item["question_text"],
            "options": item["options"],
            "correct_answer": correct_answer,
            "explanation": item["explanation"],
            "expected_output": expected_output,
            "grading_mode": grading_mode,
            "source": "测试题目.md",
        }
        records.append(record)

    duplicate_ids = [qid for qid, count in Counter(item["question_id"] for item in records).items() if count > 1]
    if duplicate_ids:
        raise ValueError(f"Duplicate generated question IDs: {duplicate_ids[:5]}")
    source_mapped_count = len(records)
    supplemental = json.loads(SUPPLEMENTAL.read_text(encoding="utf-8"))
    normalized_supplemental: list[dict] = []
    for item in supplemental:
        if item["subtopic_id"] not in valid or valid[item["subtopic_id"]] != item["block_id"]:
            raise ValueError(f"Invalid supplemental question mapping: {item['question_id']}")
        normalized = dict(item)
        normalized.setdefault("options", None)
        normalized.setdefault("explanation", "")
        normalized.setdefault("expected_output", None)
        normalized.setdefault("grading_mode", "semantic_match")
        normalized.setdefault("source", "runtime_supplement")
        normalized_supplemental.append(normalized)
    records.extend(normalized_supplemental)
    duplicate_ids = [qid for qid, count in Counter(item["question_id"] for item in records).items() if count > 1]
    if duplicate_ids:
        raise ValueError(f"Duplicate question IDs after supplement merge: {duplicate_ids[:5]}")
    coverage = Counter(item["subtopic_id"] for item in records)
    audit = {
        "source": str(SOURCE.relative_to(ROOT)),
        "parsed_questions": len(parsed),
        "mapped_source_questions": source_mapped_count,
        "supplemental_questions": len(supplemental),
        "runtime_questions": len(records),
        "parsed_question_types": dict(sorted(Counter(item["question_type"] for item in parsed).items())),
        "runtime_question_types": dict(sorted(Counter(item["question_type"] for item in records).items())),
        "unmapped_questions": unmapped,
        "parse_skipped_questions": skipped,
        "question_count_by_block": dict(sorted(Counter(item["block_id"] for item in records).items())),
        "question_count_by_subtopic": dict(sorted(coverage.items())),
        "subtopics_without_questions": [item["subtopic_id"] for item in subtopics if not coverage[item["subtopic_id"]]],
    }
    print(json.dumps({key: audit[key] for key in ("parsed_questions", "mapped_source_questions", "supplemental_questions", "runtime_questions", "question_count_by_block", "subtopics_without_questions")}, ensure_ascii=False, indent=2))
    if not args.check:
        OUTPUT.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
