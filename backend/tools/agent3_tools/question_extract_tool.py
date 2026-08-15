# -*- coding: utf-8 -*-
"""
question_extract_tool — Agent3 自测题提取工具
==============================================
功能：从 test_questions 表按 block_id / block_name 提取 1~3 道题目。
题型优先级：客观题（填空/选择/判断/简答）优先，主观题（简答/实操）其次。
"""

import json
import logging
import random
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

from config import BACKEND_DIR, DATA_DIR, settings

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

MOCK_FILE = DATA_DIR / "test_data" / "mock_question_extract_tool.json"


OBJECTIVE_TYPES = {"fill_blank", "single_choice", "true_false", "multi_choice", "match", "ordering"}
SUBJECTIVE_TYPES = {"short_answer", "practical", "essay", "case_analysis"}


def question_extract(input_data: dict, config: dict = None, database_url: str = None) -> dict:
    config = config or {}
    use_mock = config.get("use_mock", True)

    block_id = input_data.get("block_id", "")
    block_name = input_data.get("block_name", "")
    # 上限放开：内部按每个 subtopic 独立抽样后再补齐，总量通常是
    # len(subtopic_ids) * questions_per_subtopic (≈ 2 per subtopic)，
    # 单个阶段最多 5~6 subtopic 时约 10~12 题，15 的 cap 足够且防异常大值。
    stids_in = input_data.get("subtopic_ids") or []
    max_allowed = max(15, (len(stids_in) or 1) * 3)
    count = max(1, min(int(input_data.get("count", 2)), max_allowed))

    if not block_id and not block_name:
        return {
            "status": "error",
            "data": None,
            "message": "block_id or block_name is required",
            "mock_used": use_mock,
        }

    try:
        if use_mock:
            return _mock_extract(input_data, count)

        questions, db_available = _extract_questions(
            block_id,
            block_name,
            count,
            input_data.get("subtopic_ids"),
            database_url,
        )
        return {
            "status": "success",
            "data": {"questions": questions, "db_available": db_available},
            "message": _build_message(questions),
            "mock_used": False,
        }
    except Exception as exc:
        logger.exception("question_extract failed")
        return {
            "status": "error",
            "data": None,
            "message": f"question_extract failed: {exc}",
            "mock_used": use_mock,
        }


def _extract_questions(
    block_id: str,
    block_name: str,
    count: int,
    subtopic_ids: List[str] | None = None,
    database_url: str = None,
) -> tuple[List[Dict[str, Any]], bool]:
    db_path = _get_db_path(database_url)
    if not db_path.exists():
        logger.warning("Database not found for question extraction: %s", db_path)
        return [], False

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            QPS = 2  # 每个 subtopic 抽 2 题（与 sqlite_client.list_questions_for_stage 一致）
            picked_ids: set[str] = set()
            rows: list[tuple[Any, ...]] = []
            _BASE = (
                "SELECT question_id, block_id, subtopic_id, question_type, question_text, options, "
                "correct_answer, difficulty, explanation, expected_output, grading_mode, source, hint "
                "FROM test_questions "
            )
            _QTYPE_ORDER = (
                "ORDER BY CASE "
                "WHEN question_type IN ('fill_blank','single_choice','true_false','multi_choice','判断','选择','填空','match','ordering') THEN 0 "
                "WHEN question_type IN ('short_answer','practical','essay','case_analysis','简答') THEN 1 "
                "ELSE 2 END, difficulty ASC, question_id ASC "
            )
            if block_id:
                stids = list(dict.fromkeys(s for s in (subtopic_ids or []) if s))
                if stids:
                    # 严格按 subtopic 抽：每个 subtopic 最多 QPS 题，
                    # 0 题跳过，1 题就出 1 题。绝不从其他 subtopic / block 池凑题。
                    per_sql = _BASE + "WHERE block_id = ? AND subtopic_id = ? " + _QTYPE_ORDER + "LIMIT ?"
                    for stid in stids:
                        for r in conn.execute(per_sql, (block_id, stid, QPS)).fetchall():
                            qid = r[0]
                            if qid in picked_ids:
                                continue
                            picked_ids.add(qid)
                            rows.append(r)
                else:
                    # 无 subtopic → 全局 LIMIT（旧行为）
                    sql = _BASE + "WHERE block_id = ? " + _QTYPE_ORDER + "LIMIT ?"
                    rows = list(conn.execute(sql, (block_id, count)).fetchall())
            else:
                sql = (
                    "SELECT q.question_id, q.block_id, q.subtopic_id, q.question_type, q.question_text, q.options, "
                    "q.correct_answer, q.difficulty, q.explanation, q.expected_output, q.grading_mode, q.source, q.hint "
                    "FROM test_questions q JOIN knowledge_blocks b ON q.block_id = b.block_id "
                    "WHERE b.block_name = ? " + _QTYPE_ORDER + "LIMIT ?"
                )
                rows = list(conn.execute(sql, (block_name, count)).fetchall())
        finally:
            conn.close()

        _TRUE_FALSE_TYPES = {"true_false", "判断", "判断题", "yes_no"}
        questions: List[Dict[str, Any]] = []
        for row in rows:
            options: List[Any] = []
            if row[5]:
                try:
                    options = json.loads(row[5])
                    if not isinstance(options, list):
                        options = []
                except json.JSONDecodeError:
                    options = []
            qtype = row[3] or ""
            if not options and str(qtype).strip().lower() in _TRUE_FALSE_TYPES:
                options = ["正确", "错误"]
            questions.append({
                "question_id": row[0],
                "block_id": row[1],
                "subtopic_id": row[2],
                "question_type": qtype,
                "question_text": row[4],
                "options": options,
                "correct_answer": row[6],
                "difficulty": row[7],
                "explanation": row[8] or (row[12] if len(row) > 12 else "") or "",
                "expected_output": row[9],
                "grading_mode": row[10],
                "source": row[11],
                "hint": row[12] if len(row) > 12 else "",
            })
        return questions, True
    except sqlite3.Error as exc:
        logger.warning("Failed to query test_questions: %s", exc)
        return [], False


def _get_db_path(database_url: str = None) -> Path:
    url = database_url or settings.DATABASE_URL
    raw = url.replace("sqlite:///", "", 1)
    path = Path(raw)
    if not path.is_absolute():
        path = (BACKEND_DIR / path).resolve()
    return path


def _build_message(questions: List[Dict[str, Any]]) -> str:
    if not questions:
        return "No matched test questions found"
    return f"Extracted {len(questions)} question(s)"


def _mock_extract(input_data: dict, count: int) -> dict:
    scenario = input_data.get("scenario") or "normal"
    scenarios = _load_mock()
    if scenario not in scenarios:
        scenario = "normal"

    all_questions = scenarios.get(scenario, {}).get("questions", [])
    questions = random.sample(all_questions, count) if len(all_questions) > count else all_questions[:count]
    return {
        "status": "success",
        "data": {"questions": questions, "db_available": False},
        "message": f"[Mock] {_build_message(questions)}",
        "mock_used": True,
    }


def _load_mock() -> dict:
    if not MOCK_FILE.exists():
        logger.warning("Mock file not found: %s", MOCK_FILE)
        return {}
    try:
        with MOCK_FILE.open("r", encoding="utf-8") as file:
            return json.load(file).get("scenarios", {})
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read mock question file: %s", exc)
        return {}


class QuestionExtractTool:
    def __init__(self, database_url: str = None):
        self._default_config = {"use_mock": False}
        self._database_url = database_url

    async def extract_questions(self, tutorial: str, stage_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        result = question_extract(
            {
                "block_id": stage_info.get("block_id", ""),
                "block_name": stage_info.get("block_name", ""),
                "subtopic_ids": stage_info.get("subtopic_ids", []),
                "count": stage_info.get("question_count", 2),
            },
            config=self._default_config,
            database_url=self._database_url,
        )
        if result.get("status") != "success":
            return []
        return result["data"].get("questions", [])
