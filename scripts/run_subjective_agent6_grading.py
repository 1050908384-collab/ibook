"""Run an auditable, targeted Agent6 grading pass for frozen subjective answers.

The script deliberately writes only to 06_指标统计.  It never regenerates a
profile, learning package, question, answer, or any file under raw_outputs.
Use --dry-run to inspect the exact workload.  A network call requires --execute.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from config import settings  # noqa: E402
from llm.client import LLMClient  # noqa: E402

QUEUE = ROOT / "测试数据与评测材料" / "06_指标统计" / "subjective_grading_queue.csv"
OUT = ROOT / "测试数据与评测材料" / "06_指标统计" / "subjective_agent6_grading.jsonl"


def sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def grading_prompt(row: dict[str, str]) -> str:
    return f"""你是学情评估与动态决策 Agent（Agent6），负责对大模型微调与推理部署实训题进行独立评分。
只根据题目、参考答案和学习者回答评分；不要假设学习者做过未写出的步骤。

题型：{row['question_type']}
题目：{row['question_text']}
参考答案/关键要点：{row['reference_answer']}
学习者回答：{row['submitted_answer']}

评分规则：pass 表示核心要点正确；partial 表示方向正确但遗漏关键步骤或有小错误；fail 表示核心概念/关键步骤错误或无法回答。实操题还须核查可执行性、安全性和结果完整性。
仅返回 JSON 对象：{{"score": 0 或 1, "max_score": 1, "verdict": "pass|partial|fail", "reason": "不超过120字的中文依据", "missing_points": ["..."]}}"""


def parse_result(response: str) -> dict[str, Any]:
    try:
        value = json.loads(response)
    except json.JSONDecodeError:
        start, end = response.find("{"), response.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("model response is not a JSON object")
        value = json.loads(response[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("model response JSON must be an object")
    verdict = str(value.get("verdict", "")).lower()
    if verdict not in {"pass", "partial", "fail"}:
        raise ValueError("verdict must be pass, partial, or fail")
    score = float(value.get("score"))
    max_score = float(value.get("max_score"))
    if max_score <= 0 or score < 0 or score > max_score:
        raise ValueError("invalid score/max_score")
    return {
        "score": score,
        "max_score": max_score,
        "verdict": verdict,
        "reason": str(value.get("reason", "")).strip(),
        "missing_points": value.get("missing_points") if isinstance(value.get("missing_points"), list) else [],
    }


def load_queue() -> list[dict[str, str]]:
    import csv
    with QUEUE.open(encoding="utf-8-sig", newline="") as handle:
        return [r for r in csv.DictReader(handle) if r.get("review_status") == "needs_agent6_grading"]


def load_completed() -> set[tuple[str, str, str]]:
    if not OUT.exists():
        return set()
    completed = set()
    for line in OUT.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("success") and item.get("grading"):
            completed.add((str(item.get("case_id")), str(item.get("stage")), str(item.get("question_id"))))
    return completed


async def execute(rows: list[dict[str, str]], force: bool) -> tuple[int, int]:
    existing = load_completed() if not force else set()
    client = LLMClient()
    if not client._has_deepseek_credentials():
        await client.deepseek_client.aclose()
        await client.client.aclose()
        raise RuntimeError(
            "No configured cloud credential. Set DeepSeek variables, or set "
            "LLM_PROVIDER=openai_compatible with OPENAI_COMPATIBLE_API_KEY/API_BASE/MODEL."
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    written = failed = 0
    try:
        with OUT.open("a", encoding="utf-8") as handle:
            for row in rows:
                identity = (row["case_id"], row["stage"], row["question_id"])
                if identity in existing:
                    continue
                prompt = grading_prompt(row)
                record: dict[str, Any] = {
                    "case_id": row["case_id"], "stage": row["stage"], "question_id": row["question_id"],
                    "question_type": row["question_type"], "grader": "agent6_subjective_grading_v1",
                    "requested_provider": settings.LLM_PROVIDER, "requested_model": client.deepseek_model,
                    "graded_at": datetime.now(timezone.utc).isoformat(), "prompt_sha256": sha256(prompt),
                }
                try:
                    response = await client.complete(prompt, max_tokens=512, temperature=0.0, json_mode=True)
                    # A local fallback is useful in the product, but cannot be labelled
                    # as the requested cloud Agent6 evidence in the competition report.
                    if client.last_provider != client.cloud_provider:
                        raise RuntimeError(f"cloud grading did not complete; provider={client.last_provider}")
                    record.update({
                        "success": True, "actual_provider": client.last_provider,
                        "actual_model": client.last_model, "output_sha256": sha256(response),
                        "grading": parse_result(response),
                    })
                    written += 1
                except Exception as exc:
                    record.update({"success": False, "error": str(exc)})
                    failed += 1
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
    finally:
        await client.deepseek_client.aclose()
        await client.client.aclose()
    return written, failed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Permit real cloud Agent6 calls")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Regrade rows already recorded successfully")
    args = parser.parse_args()
    if args.dry_run == args.execute:
        parser.error("choose exactly one of --dry-run or --execute")
    rows = load_queue()
    if args.limit:
        rows = rows[:args.limit]
    if args.dry_run:
        print(json.dumps({"pending_subjective_questions": len(rows), "output": str(OUT), "network_calls": 0}, ensure_ascii=False))
        return 0
    import asyncio
    written, failed = asyncio.run(execute(rows, args.force))
    print(json.dumps({"graded": written, "failed": failed, "output": str(OUT)}, ensure_ascii=False))
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
