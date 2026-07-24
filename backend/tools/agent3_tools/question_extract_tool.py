# -*- coding: utf-8 -*-
"""
question_extract_tool — Agent3 自测题抽取工具
================================================
功能：从 test_questions 表抽取当前阶段的自测题（填空/选择优先）。
    - 真实模式：查 test_questions 表，按 block_id 过滤，抽取1~3道题
    - Mock 模式：返回 ./data/test_data/mock_question_extract_tool.json 预设题目

统一接口：
    def question_extract(input_data: dict, config: dict = None) -> dict
    返回：{"status", "data", "message", "mock_used"}

规则对齐（content_generation.skill.md）：
    - 自测题由用户主动点击自测时单独抽取，不纳入教程生成阶段
    - 优先抽取填空/选择题（question_type IN ('填空', '选择')）
    - 抽取数量：1~3道

保留 QuestionExtractTool 类作为向后兼容适配器。
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


# ------------------------------------------------------------------
# 统一函数式接口（规范实现）
# ------------------------------------------------------------------
def question_extract(input_data: dict, config: dict = None) -> dict:
    """
    抽取自测题。

    参数：
        input_data: dict
            - block_id: str          板块ID（过滤条件）
            - block_name: str        板块名称（备选过滤条件）
            - count: int             抽取数量（默认2，范围1~3）
            - scenario: str          Mock 模式下选择场景：normal|boundary|exception
        config: dict
            - use_mock: bool         是否使用 Mock（默认 True）

    返回：
        {
            "status": "success" | "error",
            "data": {
                "questions": [
                    {
                        "question_id": str,
                        "question_type": str,
                        "question_text": str,
                        "options": List[str],
                        "correct_answer": str,
                        "explanation": str,
                        "difficulty": str
                    }
                ],
                "db_available": bool
            },
            "message": str,
            "mock_used": bool
        }
    """
    config = config or {}
    use_mock = config.get("use_mock", True)

    block_id = input_data.get("block_id", "")
    block_name = input_data.get("block_name", "")
    count = min(max(input_data.get("count", 2), 1), 3)

    if not block_id and not block_name:
        return {
            "status": "error",
            "data": None,
            "message": "输入未包含 block_id 或 block_name 字段，无法抽取题目",
            "mock_used": use_mock,
        }

    try:
        if use_mock:
            return _mock_extract(input_data, count)

        questions, db_available = _extract_questions(block_id, block_name, count)
        return {
            "status": "success",
            "data": {
                "questions": questions,
                "db_available": db_available,
            },
            "message": _build_message(questions),
            "mock_used": False,
        }
    except Exception as e:
        logger.exception("question_extract 执行失败")
        return {
            "status": "error",
            "data": None,
            "message": f"question_extract 执行异常: {e}",
            "mock_used": use_mock,
        }


# ------------------------------------------------------------------
# 真实模式：数据库抽取
# ------------------------------------------------------------------
def _extract_questions(block_id: str, block_name: str, count: int) -> tuple[List[Dict[str, Any]], bool]:
    """
    SQL: 从 test_questions 表抽取题目（优先填空/选择）。
    """
    db_path = _get_db_path()
    if not db_path.exists():
        logger.warning("数据库不存在: %s，question_extract 无可用题目", db_path)
        return [], False

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            params = []
            if block_id:
                sql = """
                    SELECT question_id, question_type, question_text, options, 
                           correct_answer, explanation, difficulty
                    FROM test_questions
                    WHERE block_id = ?
                    ORDER BY question_type IN ('填空', '选择') DESC, difficulty ASC, created_at DESC
                    LIMIT ?
                    """
                params = [block_id, count]
            elif block_name:
                sql = """
                    SELECT q.question_id, q.question_type, q.question_text, q.options, 
                           q.correct_answer, q.explanation, q.difficulty
                    FROM test_questions q
                    JOIN knowledge_blocks b ON q.block_id = b.block_id
                    WHERE b.block_name = ?
                    ORDER BY q.question_type IN ('填空', '选择') DESC, q.difficulty ASC, q.created_at DESC
                    LIMIT ?
                    """
                params = [block_name, count]
            else:
                conn.close()
                return [], True

            cur = conn.execute(sql, params)
            rows = cur.fetchall()
        finally:
            conn.close()

        questions = []
        for r in rows:
            options = []
            if r[3]:
                try:
                    options = json.loads(r[3])
                    if not isinstance(options, list):
                        options = []
                except json.JSONDecodeError:
                    options = []

            questions.append({
                "question_id": r[0],
                "question_type": r[1],
                "question_text": r[2],
                "options": options,
                "correct_answer": r[4],
                "explanation": r[5] or "",
                "difficulty": r[6],
            })

        return questions, True
    except sqlite3.Error as e:
        logger.warning("查询 test_questions 失败: %s", e)
        return [], False


# ------------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------------
def _get_db_path() -> Path:
    """解析 settings.DATABASE_URL 为绝对路径。"""
    url = settings.DATABASE_URL
    raw = url.replace("sqlite:///", "", 1)
    p = Path(raw)
    if not p.is_absolute():
        p = (BACKEND_DIR / p).resolve()
    return p


def _build_message(questions: List[Dict[str, Any]]) -> str:
    if not questions:
        return "未找到匹配的自测题"
    types = [q["question_type"] for q in questions]
    return f"抽取到 {len(questions)} 道自测题：{', '.join(types)}"


# ------------------------------------------------------------------
# Mock 模式
# ------------------------------------------------------------------
def _mock_extract(input_data: dict, count: int) -> dict:
    """从 mock JSON 读取预设场景。"""
    scenario = input_data.get("scenario") or "normal"
    scenarios = _load_mock()
    if scenario not in scenarios:
        scenario = "normal"

    all_questions = scenarios.get(scenario, {}).get("questions", [])
    if len(all_questions) > count:
        questions = random.sample(all_questions, count)
    else:
        questions = all_questions[:count]

    return {
        "status": "success",
        "data": {
            "questions": questions,
            "db_available": False,
        },
        "message": f"[Mock] {_build_message(questions)}",
        "mock_used": True,
    }


def _load_mock() -> dict:
    if not MOCK_FILE.exists():
        logger.warning("Mock 文件不存在: %s", MOCK_FILE)
        return {}
    try:
        with MOCK_FILE.open("r", encoding="utf-8") as f:
            return json.load(f).get("scenarios", {})
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("读取 Mock 文件失败: %s", e)
        return {}


# ------------------------------------------------------------------
# 向后兼容适配器
# ------------------------------------------------------------------
class QuestionExtractTool:
    def __init__(self):
        self._default_config = {"use_mock": True}

    async def extract_questions(self, block_id: str, count: int = 2) -> List[Dict[str, Any]]:
        result = question_extract({
            "block_id": block_id,
            "count": count,
        }, config=self._default_config)
        if result.get("status") != "success":
            return []
        return result["data"].get("questions", [])
