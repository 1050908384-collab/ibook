# -*- coding: utf-8 -*-
"""
calculate_confidence_tool — Agent1 置信度计算工具
================================================
功能：评估用户画像的信息完整度，输出置信度(0~1)，用于决定是否需要追问。
    - 真实模式：基于 skill_tags/knowledge_gaps/goal/daily_hours/learning_style 综合评分
    - Mock 模式：返回 ./data/test_data/mock_calculate_confidence_tool.json 预设置信度

统一接口：
    def calculate_confidence(input_data: dict, config: dict = None) -> dict
    返回：{"status", "data", "message", "mock_used"}

规则对齐（learner_profiling.skill.md 规则2）：
    - LLM正常返回 → confidence=0.9
    - LLM回退正则 → confidence=0.75
    - 信息不足 → confidence<0.7 → 触发追问
    - 缺目标 → 默认"通用学习目标"，confidence=0.75

保留 CalculateConfidenceTool 类作为向后兼容适配器。
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from config import DATA_DIR

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

MOCK_FILE = DATA_DIR / "test_data" / "mock_calculate_confidence_tool.json"


# ------------------------------------------------------------------
# 统一函数式接口（规范实现）
# ------------------------------------------------------------------
def calculate_confidence(input_data: dict, config: dict = None) -> dict:
    """
    计算用户画像的信息完整度置信度。

    参数：
        input_data: dict
            - user_input: str           用户原始输入
            - skill_tags: List[str]     已掌握技能标签
            - knowledge_gaps: List[Dict] 知识盲区
            - goal: str                 学习目标（可选）
            - daily_hours: float        每日学习时长（可选）
            - learning_style: str       学习风格（可选）
            - parsed_by_llm: bool       是否由LLM解析（可选，用于决定基准分）
        config: dict
            - use_mock: bool            是否使用 Mock（默认 True）

    返回：
        {
            "status": "success" | "error",
            "data": {
                "confidence": float,          0~1 置信度
                "needs_more_info": bool,      是否需要追问（confidence < 0.7）
                "follow_up_questions": List[str], 追问列表
                "breakdown": {                各维度评分明细
                    "skills_score": float,
                    "gaps_score": float,
                    "goal_score": float,
                    "hours_score": float,
                    "style_score": float
                }
            },
            "message": str,
            "mock_used": bool
        }
    """
    config = config or {}
    use_mock = config.get("use_mock", True)

    try:
        if use_mock:
            return _mock_calculate(input_data)

        result = _calculate(input_data)
        return {
            "status": "success",
            "data": result,
            "message": _build_message(result),
            "mock_used": False,
        }
    except Exception as e:
        logger.exception("calculate_confidence 执行失败")
        return {
            "status": "error",
            "data": None,
            "message": f"calculate_confidence 执行异常: {e}",
            "mock_used": use_mock,
        }


# ------------------------------------------------------------------
# 真实模式：置信度计算
# ------------------------------------------------------------------
def _calculate(input_data: dict) -> dict:
    """基于各字段完整性综合计算置信度。"""
    skill_tags = input_data.get("skill_tags", [])
    knowledge_gaps = input_data.get("knowledge_gaps", [])
    goal = input_data.get("goal", "")
    daily_hours = input_data.get("daily_hours")
    learning_style = input_data.get("learning_style", "")
    parsed_by_llm = input_data.get("parsed_by_llm", False)

    base = 0.9 if parsed_by_llm else 0.75

    skills_score = min(len(skill_tags) * 0.1, 0.3)
    gaps_score = min(len(knowledge_gaps) * 0.1, 0.3)
    goal_score = 0.2 if goal and goal.strip() else 0.0
    hours_score = 0.1 if daily_hours and daily_hours > 0 else 0.0
    style_score = 0.1 if learning_style else 0.0

    confidence = base * 0.5 + (skills_score + gaps_score + goal_score + hours_score + style_score) * 0.5
    confidence = max(0.3, min(0.95, confidence))

    needs_more_info = confidence < 0.7
    follow_up_questions = _generate_follow_up(input_data)

    return {
        "confidence": round(confidence, 4),
        "needs_more_info": needs_more_info,
        "follow_up_questions": follow_up_questions,
        "breakdown": {
            "skills_score": round(skills_score, 4),
            "gaps_score": round(gaps_score, 4),
            "goal_score": round(goal_score, 4),
            "hours_score": round(hours_score, 4),
            "style_score": round(style_score, 4),
        },
    }


def _generate_follow_up(input_data: dict) -> List[str]:
    """根据缺失字段生成追问。"""
    questions = []
    if not input_data.get("goal"):
        questions.append("你希望达成什么学习目标？")
    if not input_data.get("skill_tags") or len(input_data.get("skill_tags", [])) == 0:
        questions.append("你目前已掌握哪些技能？")
    if input_data.get("daily_hours") is None:
        questions.append("你每天可投入多少小时学习？")
    if not input_data.get("learning_style"):
        questions.append("你偏好哪种学习方式？（理论学习/动手实践/两者结合）")
    return questions


def _build_message(result: dict) -> str:
    confidence = result["confidence"]
    needs_more_info = result["needs_more_info"]
    questions = result["follow_up_questions"]

    if needs_more_info:
        return f"置信度 {confidence:.4f} < 0.7，需要追问 {len(questions)} 个问题"
    return f"置信度 {confidence:.4f}，信息完整，无需追问"


# ------------------------------------------------------------------
# Mock 模式
# ------------------------------------------------------------------
def _mock_calculate(input_data: dict) -> dict:
    """从 mock JSON 读取预设场景。"""
    scenario = (input_data or {}).get("scenario") or "normal"
    scenarios = _load_mock()
    if scenario not in scenarios:
        scenario = "normal"

    data = dict(scenarios[scenario])
    data.setdefault("needs_more_info", False)
    data.setdefault("follow_up_questions", [])
    data.setdefault("breakdown", {})

    return {
        "status": "success",
        "data": data,
        "message": f"[Mock] {_build_message(data)}",
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
class CalculateConfidenceTool:
    def __init__(self):
        self._default_config = {"use_mock": False}

    async def calculate(
        self,
        user_input: str,
        skill_tags: List[str],
        knowledge_gaps: List[Dict[str, Any]],
        goal: str = "",
        daily_hours: float | None = None,
        learning_style: str = "",
        parsed_by_llm: bool = False,
    ) -> float:
        result = calculate_confidence({
            "user_input": user_input,
            "skill_tags": skill_tags,
            "knowledge_gaps": knowledge_gaps,
            "goal": goal,
            "daily_hours": daily_hours,
            "learning_style": learning_style,
            "parsed_by_llm": parsed_by_llm,
        }, config=self._default_config)
        if result.get("status") != "success":
            return 0.3
        return result["data"].get("confidence", 0.3)
