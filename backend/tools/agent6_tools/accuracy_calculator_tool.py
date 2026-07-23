"""
accuracy_calculator_tool — Agent6 正确率计算工具
================================================
功能：输入 session_id 和 stage，从 user_answers 表统计该阶段答题正确率（0~1），
返回正确数和总数。Mock 模式返回预设正确率。

统一接口：
    def accuracy_calculator(input_data: dict, config: dict = None) -> dict
    返回：{"status", "data", "message", "mock_used"}

依赖表：user_answers（stage / is_correct）
    数据库表字段：stage INT, is_correct BOOLEAN，参见 数据库框架.md 第8节。
    SQL：SELECT COUNT(*) FROM user_answers WHERE session_id=? AND stage=?
         SELECT SUM(CASE WHEN is_correct=1 THEN 1 ELSE 0 END) FROM ...

规则对齐：iteration_strategy.skill.md 规则1（accuracy 是五档策略选择的基础）。
保留 AccuracyCalculatorTool 类作为向后兼容适配器。
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import BACKEND_DIR, DATA_DIR, settings

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

MOCK_FILE = DATA_DIR / "test_data" / "mock_accuracy_calculator.json"


# ------------------------------------------------------------------
# 统一函数式接口（规范实现）
# ------------------------------------------------------------------
def accuracy_calculator(input_data: dict, config: dict = None) -> dict:
    """
    计算指定 session/stage 的答题正确率。

    参数：
        input_data: dict
            - session_id: str       会话ID
            - stage: int            阶段编号（可选，默认所有阶段）
            - scenario: str         Mock 场景：high_match|medium_match|low_match
            - answers: list         可选，直接传入答题记录（绕过DB查询）
        config: dict
            - use_mock: bool        是否使用 Mock（默认 True）

    返回：
        {
            "status": "success" | "error",
            "data": {
                "total": int,
                "correct": int,
                "wrong": int,
                "accuracy": float,           # 0~1
                "stage": int,
                "session_id": str,
                "source": "mock" | "db" | "direct",
                "db_available": bool
            },
            "message": str,
            "mock_used": bool
        }
    """
    config = config or {}
    use_mock = config.get("use_mock", True)

    input_data = input_data or {}
    session_id = str(input_data.get("session_id") or "")
    stage = input_data.get("stage")

    # 允许直接传入 answers 列表（单测/快速调用场景）
    answers = input_data.get("answers")
    if answers is not None:
        try:
            total, correct = _count_answers(answers)
            accuracy = correct / total if total > 0 else 0.0
            return {
                "status": "success",
                "data": {
                    "total": total,
                    "correct": correct,
                    "wrong": total - correct,
                    "accuracy": round(accuracy, 4),
                    "stage": stage,
                    "session_id": session_id,
                    "source": "direct",
                    "db_available": False,
                },
                "message": f"直接传入答题记录：正确 {correct}/{total}，正确率 {accuracy:.0%}",
                "mock_used": False,
            }
        except Exception as e:
            return {
                "status": "error",
                "data": None,
                "message": f"answers 解析失败: {e}",
                "mock_used": False,
            }

    if not session_id:
        return {
            "status": "error",
            "data": None,
            "message": "输入需包含 session_id",
            "mock_used": use_mock,
        }

    try:
        if use_mock:
            return _mock_calculate(session_id, stage, input_data)

        # 真实模式：查 user_answers 表
        total, correct, db_available = _query_answers(session_id, stage)
        wrong = max(0, total - correct)
        accuracy = (correct / total) if total > 0 else 0.0
        return {
            "status": "success",
            "data": {
                "total": total,
                "correct": correct,
                "wrong": wrong,
                "accuracy": round(accuracy, 4),
                "stage": stage,
                "session_id": session_id,
                "source": "db",
                "db_available": db_available,
            },
            "message": _build_message(total, correct, accuracy, stage),
            "mock_used": False,
        }
    except Exception as e:
        logger.exception("accuracy_calculator 执行失败")
        return {
            "status": "error",
            "data": None,
            "message": f"accuracy_calculator 执行异常: {e}",
            "mock_used": use_mock,
        }


# ------------------------------------------------------------------
# 内部实现：直接answers + DB查询
# ------------------------------------------------------------------
def _count_answers(answers: List[Dict[str, Any]]) -> tuple[int, int]:
    """从答案列表统计 total 与 correct。支持布尔和字符串两种 is_correct 表示。"""
    total = 0
    correct = 0
    for a in answers:
        if not isinstance(a, dict):
            continue
        if a.get("question_id") is None and a.get("answer_id") is None:
            continue  # 必须是答题记录
        total += 1
        v = a.get("is_correct")
        if v is True or v == 1 or v == "1" or (isinstance(v, str) and v.lower() == "true"):
            correct += 1
    return total, correct


def _get_db_path() -> Path:
    url = settings.DATABASE_URL
    raw = url.replace("sqlite:///", "", 1)
    p = Path(raw)
    if not p.is_absolute():
        p = (BACKEND_DIR / p).resolve()
    return p


def _query_answers(session_id: str, stage: Optional[int]) -> tuple[int, int, bool]:
    """
    SQL：从 user_answers 表统计 total 与 correct。
    user_answers 表结构：answer_id, session_id, stage, question_id,
                        user_answer, is_correct, time_spent, created_at。
    数据库不存在时返回 (0, 0, False)，调用方降级为 0 正确率。
    """
    db_path = _get_db_path()
    if not db_path.exists():
        logger.warning("数据库不存在: %s", db_path)
        return 0, 0, False
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            if stage is None:
                # SQL: 仅按 session_id 聚合所有阶段的答题记录
                cur = conn.execute(
                    """
                    -- 统计该会话全部答题记录
                    SELECT COUNT(*), SUM(CASE WHEN is_correct=1 THEN 1 ELSE 0 END)
                    FROM user_answers
                    WHERE session_id = ?
                    """,
                    (session_id,),
                )
            else:
                # SQL: 限定到指定阶段
                cur = conn.execute(
                    """
                    -- 统计指定阶段答题记录，stage 来自 user_answers.stage
                    SELECT COUNT(*), SUM(CASE WHEN is_correct=1 THEN 1 ELSE 0 END)
                    FROM user_answers
                    WHERE session_id = ? AND stage = ?
                    """,
                    (session_id, int(stage)),
                )
            row = cur.fetchone()
        finally:
            conn.close()
        total = int(row[0] or 0)
        correct = int(row[1] or 0)
        return total, correct, True
    except sqlite3.Error as e:
        logger.warning("查询 user_answers 失败: %s", e)
        return 0, 0, False


def _build_message(total: int, correct: int, accuracy: float, stage: Optional[int]) -> str:
    stage_part = f"阶段{stage}" if stage is not None else "全阶段"
    return f"{stage_part}答题：正确 {correct}/{total}，正确率 {accuracy:.0%}"


# ------------------------------------------------------------------
# Mock 模式
# ------------------------------------------------------------------
def _mock_calculate(session_id: str, stage: Optional[int], input_data: dict) -> dict:
    """按 scenario 或正确率自动选择 Mock 场景。"""
    scenario = _pick_scenario(input_data)
    scenarios = _load_mock()
    if scenario not in scenarios:
        scenario = "medium_match"
    data = json.loads(json.dumps(scenarios[scenario]))
    data["session_id"] = session_id
    if stage is not None:
        data["stage"] = stage
    data["source"] = "mock"
    return {
        "status": "success",
        "data": data,
        "message": f"[Mock] 正确率 {data.get('accuracy', 0):.0%}",
        "mock_used": True,
    }


def _pick_scenario(input_data: dict) -> str:
    scenario = (input_data or {}).get("scenario")
    if scenario:
        return scenario
    # 按 answers 内容自动归类
    answers = (input_data or {}).get("answers")
    if answers:
        total, correct = _count_answers(answers)
        if total == 0:
            return "medium_match"
        acc = correct / total
        if acc >= 0.80:
            return "high_match"
        if acc >= 0.60:
            return "medium_match"
        return "low_match"
    return "medium_match"


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
# 保留原 StageDecisionAgent 调用的 async 类接口，委托给函数式核心。
#   calculate(answers) → 返回 float 正确率
# ------------------------------------------------------------------
class AccuracyCalculatorTool:
    def __init__(self):
        self._default_config = {"use_mock": True}

    async def calculate(self, answers: list) -> float:
        if not answers:
            return 0.0
        result = accuracy_calculator({"answers": answers}, config=self._default_config)
        if result.get("status") != "success":
            return 0.0
        return float(result["data"].get("accuracy", 0.0))
