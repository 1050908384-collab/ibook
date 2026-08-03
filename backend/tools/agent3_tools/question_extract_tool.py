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


def question_extract(input_data: dict, config: dict = None) -> dict:
    config = config or {}
    use_mock = config.get("use_mock", True)

    block_id = input_data.get("block_id", "")
    block_name = input_data.get("block_name", "")
    count = min(max(int(input_data.get("count", 2)), 1), 3)

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

        questions, db_available = _extract_questions(block_id, block_name, count)
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


def _extract_questions(block_id: str, block_name: str, count: int) -> tuple[List[Dict[str, Any]], bool]:
    db_path = _get_db_path()
    if not db_path.exists():
        logger.warning("Database not found for question extraction: %s", db_path)
        return [], False

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            if block_id:
                sql = """
                    SELECT question_id, block_id, subtopic_id, question_type, question_text, options,
                           correct_answer, difficulty, explanation, expected_output, grading_mode, source, hint
                    FROM test_questions
                    WHERE block_id = ?
                    ORDER BY CASE
                        WHEN question_type IN ('fill_blank', 'single_choice', 'true_false', 'multi_choice', 'match', 'ordering') THEN 0
                        WHEN question_type IN ('short_answer', 'practical', 'essay', 'case_analysis') THEN 1
                        ELSE 2
                    END, difficulty ASC, question_id ASC
                    LIMIT ?
                """
                params = [block_id, count]
            else:
                sql = """
                    SELECT q.question_id, q.block_id, q.subtopic_id, q.question_type, q.question_text, q.options,
                           q.correct_answer, q.difficulty, q.explanation, q.expected_output, q.grading_mode, q.source, q.hint
                    FROM test_questions q
                    JOIN knowledge_blocks b ON q.block_id = b.block_id
                    WHERE b.block_name = ?
                    ORDER BY CASE
                        WHEN q.question_type IN ('fill_blank', 'single_choice', 'true_false', 'multi_choice', 'match', 'ordering') THEN 0
                        WHEN q.question_type IN ('short_answer', 'practical', 'essay', 'case_analysis') THEN 1
                        ELSE 2
                    END, q.difficulty ASC, q.question_id ASC
                    LIMIT ?
                """
                params = [block_name, count]
            rows = conn.execute(sql, params).fetchall()
        finally:
            conn.close()

        questions: List[Dict[str, Any]] = []
        for row in rows:
            options = []
            if row[5]:
                try:
                    options = json.loads(row[5])
                    if not isinstance(options, list):
                        options = []
                except json.JSONDecodeError:
                    options = []
            questions.append({
                "question_id": row[0],
                "block_id": row[1],
                "subtopic_id": row[2],
                "question_type": row[3],
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


def _get_db_path() -> Path:
    url = settings.DATABASE_URL
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
    def __init__(self):
        self._default_config = {"use_mock": False}

    async def extract_questions(self, tutorial: str, stage_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        result = question_extract(
            {
                "block_id": stage_info.get("block_id", ""),
                "block_name": stage_info.get("block_name", ""),
                "count": stage_info.get("question_count", 2),
            },
            config=self._default_config,
        )
        if result.get("status") != "success":
            return []
        return result["data"].get("questions", [])
