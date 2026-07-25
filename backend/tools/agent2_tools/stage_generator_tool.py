# -*- coding: utf-8 -*-
"""
stage_generator_tool — Agent2 阶段计划生成工具
================================================
功能：根据知识盲区和学习目标生成学习阶段计划，对齐 StageInfo 契约。
    - 真实模式：基于 knowledge_blocks 表数据生成阶段（含 estimated_hours/prerequisites）
    - Mock 模式：返回 ./data/test_data/mock_stage_generator_tool.json 预设阶段计划

统一接口：
    def stage_generator(input_data: dict, config: dict = None) -> dict
    返回：{"status", "data", "message", "mock_used"}

规则对齐（stage_planning.skill.md）：
    - 规则1：仅输出单个学习阶段，取首个未掌握盲区
    - 规则2：stage_id/block_id 由 session_id+stage_num+gap.skill 确定性派生（uuid5）
    - 规则3：estimated_hours 默认2.0（框架默认，实际从表读取），prerequisites 沿用 gap.prerequisites
    - 规则4：字段须符合 StageInfo 契约

保留 StageGeneratorTool 类作为向后兼容适配器。
"""

import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Dict

from config import DATA_DIR
from llm.client import LLMClient

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

MOCK_FILE = DATA_DIR / "test_data" / "mock_stage_generator_tool.json"


# ------------------------------------------------------------------
# 统一函数式接口（规范实现）
# ------------------------------------------------------------------
def stage_generator(input_data: dict, config: dict = None) -> dict:
    """
    生成学习阶段计划。

    参数：
        input_data: dict
            - session_id: str        会话标识（用于派生确定性ID）
            - gap: Dict              知识盲区（含 skill/level）
            - block_info: Dict       板块详情（可选，含 estimated_hours/prerequisites）
            - goal: str              学习目标
            - stage_num: int         当前阶段序号（默认1）
            - scenario: str          Mock 模式下选择场景：normal|boundary|exception
        config: dict
            - use_mock: bool         是否使用 Mock（默认 True）

    返回：
        {
            "status": "success" | "error",
            "data": {
                "stage": {
                    "stage_id": str,
                    "stage_num": int,
                    "block_id": str,
                    "block_name": str,
                    "estimated_hours": float,
                    "gap_skill": str,
                    "prerequisites": List[str],
                    "title": str,
                    "description": str,
                    "objectives": List[str],
                    "difficulty": str
                },
                "generated_by_llm": bool
            },
            "message": str,
            "mock_used": bool
        }
    """
    config = config or {}
    use_mock = config.get("use_mock", True)

    session_id = input_data.get("session_id", "")
    gap = input_data.get("gap", {})
    goal = input_data.get("goal", "")
    stage_num = input_data.get("stage_num", 1)

    skill_name = gap.get("skill", "未知技能")
    level = gap.get("level", "beginner")

    if not session_id:
        return {
            "status": "error",
            "data": None,
            "message": "输入未包含 session_id 字段，无法生成阶段ID",
            "mock_used": use_mock,
        }

    try:
        if use_mock:
            return _mock_generate(input_data)

        block_info = input_data.get("block_info", {})
        result = _generate_stage(session_id, gap, block_info, goal, stage_num)
        return {
            "status": "success",
            "data": result,
            "message": f"阶段生成完成: {result['stage']['title']}",
            "mock_used": False,
        }
    except Exception as e:
        logger.exception("stage_generator 执行失败")
        fallback = _fallback_stage(session_id, skill_name, level, goal, stage_num)
        return {
            "status": "success",
            "data": {
                "stage": fallback,
                "generated_by_llm": False,
            },
            "message": f"LLM生成异常，已使用回退策略: {e}",
            "mock_used": False,
        }


# ------------------------------------------------------------------
# 真实模式：LLM 生成阶段计划
# ------------------------------------------------------------------
def _generate_stage(session_id: str, gap: Dict[str, Any], block_info: Dict[str, Any],
                    goal: str, stage_num: int) -> dict:
    """生成阶段计划（确定性描述，稳定派生ID）。"""
    skill_name = gap.get("skill", "未知技能")
    level = gap.get("level", "beginner")
    llm_data = {}

    stage_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{session_id}:{stage_num}:{skill_name}"))
    block_id = block_info.get("block_id") or f"KB-{uuid.uuid5(uuid.NAMESPACE_DNS, skill_name).hex[:8].upper()}"
    difficulty_map = {"beginner": "初级", "intermediate": "中级", "advanced": "高级"}

    return {
        "stage": {
            "stage_id": stage_id,
            "stage_num": stage_num,
            "block_id": block_id,
            "block_name": skill_name,
            "estimated_hours": block_info.get("estimated_hours", 2.0),
            "gap_skill": skill_name,
            "prerequisites": block_info.get("prerequisites", []),
            "title": llm_data.get("title", f"学习{skill_name}"),
            "description": llm_data.get("description", f"围绕{skill_name}完成一次可执行的阶段学习与练习。"),
            "objectives": llm_data.get("objectives", [
                f"理解{skill_name}的基本概念",
                f"掌握{skill_name}的核心技术",
                f"能够独立完成{skill_name}相关任务"
            ]),
            "difficulty": llm_data.get("difficulty", block_info.get("difficulty") or difficulty_map.get(level, "中级")),
        },
        "generated_by_llm": False,
    }


def _fallback_stage(session_id: str, skill_name: str, level: str, goal: str, stage_num: int) -> dict:
    """LLM失败时的回退阶段计划。"""
    stage_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{session_id}:{stage_num}:{skill_name}"))
    block_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{skill_name}"))

    difficulty_map = {"beginner": "初级", "intermediate": "中级", "advanced": "高级"}

    return {
        "stage_id": stage_id,
        "stage_num": stage_num,
        "block_id": block_id,
        "block_name": skill_name,
        "estimated_hours": 2.0,
        "gap_skill": skill_name,
        "prerequisites": [],
        "title": f"学习{skill_name}",
        "description": f"掌握{skill_name}的核心概念和实践技能",
        "objectives": [
            f"理解{skill_name}的基本概念",
            f"掌握{skill_name}的核心技术",
            f"能够独立完成{skill_name}相关任务"
        ],
        "difficulty": difficulty_map.get(level, "中级"),
    }


# ------------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------------
def _clean_response(response: str) -> str:
    """清理LLM响应中的标记。"""
    import re
    response = response.replace("<think>", "").replace("</think>", "")
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    response = re.sub(r'```json\s*', '', response)
    response = re.sub(r'\s*```', '', response)
    return response.strip()


# ------------------------------------------------------------------
# Mock 模式
# ------------------------------------------------------------------
def _mock_generate(input_data: dict) -> dict:
    """从 mock JSON 读取预设场景。"""
    scenario = input_data.get("scenario") or "normal"
    scenarios = _load_mock()
    if scenario not in scenarios:
        scenario = "normal"

    data = dict(scenarios[scenario])
    data["stage"]["stage_num"] = input_data.get("stage_num", 1)
    data.setdefault("generated_by_llm", True)

    return {
        "status": "success",
        "data": data,
        "message": f"[Mock] 阶段生成完成: {data['stage']['title']}",
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
class StageGeneratorTool:
    def __init__(self):
        self._default_config = {"use_mock": False}
        self.llm_client = LLMClient()

    async def generate_stage(self, session_id: str, gap: Dict[str, Any],
                             block_info: Dict[str, Any], goal: str) -> Dict[str, Any]:
        result = stage_generator({
            "session_id": session_id,
            "gap": gap,
            "block_info": block_info,
            "goal": goal,
        }, config=self._default_config)
        if result.get("status") != "success":
            return _fallback_stage(session_id, gap.get("skill", "未知技能"),
                                   gap.get("level", "beginner"), goal, 1)
        return result["data"].get("stage", {})
