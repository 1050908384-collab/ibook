# -*- coding: utf-8 -*-
"""
stage_generator_tool — Agent1 阶段计划生成工具（原Agent2工具，已合并至Agent1）
========================================================================================
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
            - profile: Dict          用户画像（可选，用于个性化）
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
                    "difficulty": str,
                    "recommended_daily_hours": float,
                    "suggested_study_days": int,
                    "learning_style_notes": str,
                    "hardware_notes": str,
                    "personalization_summary": str
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
    profile = input_data.get("profile", {})

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
        result = _generate_stage(session_id, gap, block_info, goal, stage_num, profile)
        return {
            "status": "success",
            "data": result,
            "message": f"阶段生成完成: {result['stage']['title']}",
            "mock_used": False,
        }
    except Exception as e:
        logger.exception("stage_generator 执行失败")
        fallback = _fallback_stage(session_id, skill_name, level, goal, stage_num, profile)
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
# 真实模式：确定性生成阶段计划
# ------------------------------------------------------------------
def _generate_stage(session_id: str, gap: Dict[str, Any], block_info: Dict[str, Any],
                    goal: str, stage_num: int, profile: Dict[str, Any] = None) -> dict:
    """生成阶段计划（确定性描述，稳定派生ID）。"""
    profile = profile or {}
    skill_name = gap.get("skill", "未知技能")
    level = gap.get("level", "beginner")
    llm_data = {}

    stage_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{session_id}:{stage_num}:{skill_name}"))
    block_id = block_info.get("block_id") or f"KB-{uuid.uuid5(uuid.NAMESPACE_DNS, skill_name).hex[:8].upper()}"
    difficulty_map = {"beginner": "初级", "intermediate": "中级", "advanced": "高级"}

    daily_hours = float(profile.get("daily_hours", 2.0))
    learning_style = profile.get("learning_style", "mixed")
    hardware = profile.get("hardware", "")
    grade = profile.get("grade", "")

    base_hours = float(block_info.get("estimated_hours", 2.0))
    adjusted_hours = _adjust_hours_by_profile(base_hours, daily_hours, level)
    study_days = max(1, round(adjusted_hours / max(daily_hours, 0.5)))

    learning_style_notes = _generate_style_notes(learning_style, skill_name)
    hardware_notes = _generate_hardware_notes(hardware, skill_name)
    personalization_summary = _generate_summary(grade, daily_hours, learning_style, adjusted_hours, study_days)

    return {
        "stage": {
            "stage_id": stage_id,
            "stage_num": stage_num,
            "block_id": block_id,
            "block_name": skill_name,
            "estimated_hours": adjusted_hours,
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
            "recommended_daily_hours": daily_hours,
            "suggested_study_days": study_days,
            "learning_style_notes": learning_style_notes,
            "hardware_notes": hardware_notes,
            "personalization_summary": personalization_summary,
        },
        "generated_by_llm": False,
    }


def _adjust_hours_by_profile(base_hours: float, daily_hours: float, level: str) -> float:
    """根据用户画像调整预估时长。"""
    if daily_hours >= 4:
        speed_factor = 1.2
    elif daily_hours >= 2:
        speed_factor = 1.0
    elif daily_hours >= 1:
        speed_factor = 0.8
    else:
        speed_factor = 0.6

    level_multiplier = {"beginner": 1.2, "intermediate": 1.0, "advanced": 0.9}
    adjusted = base_hours * level_multiplier.get(level, 1.0) / speed_factor
    return round(min(adjusted, 16.0), 1)


def _generate_style_notes(learning_style: str, skill_name: str) -> str:
    """根据学习风格生成建议。"""
    style_map = {
        "practice": f"动手型学习者：建议先完成{skill_name}的最小可运行操作，再记录命令、报错与解决过程",
        "visual": f"视觉型学习者：建议多关注{skill_name}的架构图、流程图和可视化演示",
        "auditory": f"听觉型学习者：建议配合{skill_name}的视频讲解和语音教程",
        "reading": f"阅读型学习者：建议详细阅读{skill_name}的官方文档和技术书籍",
        "kinesthetic": f"动手型学习者：建议边学边练，在代码中实践{skill_name}的每个知识点",
        "mixed": f"混合型学习者：建议结合阅读、观看和实践多种方式学习{skill_name}",
    }
    return style_map.get(learning_style, style_map["mixed"])


def _generate_hardware_notes(hardware: str, skill_name: str) -> str:
    """根据硬件条件生成提示。"""
    if not hardware:
        return ""
    hardware_lower = hardware.lower()
    if any(keyword in skill_name for keyword in ("安装", "环境准备", "环境配置", "Linux环境")):
        return f"当前设备为 {hardware}：本阶段先完成环境与依赖检查；训练阶段再根据显存设置 batch size。"
    # NVIDIA GPU 检测：包含 RTX、GTX、或具体型号
    if any(gpu in hardware_lower for gpu in ["rtx", "gtx", "gpu", "cuda", "nvidia", 
                                               "4090", "4080", "4070", "4060", "4050",
                                               "3090", "3080", "3070", "3060",
                                               "2090", "2080", "2070"]):
        return f"检测到独立GPU：{skill_name}训练可加速运行，建议尝试更大的batch size"
    # AMD GPU 检测
    elif any(gpu in hardware_lower for gpu in ["rx", "radeon", "amd gpu"]):
        return f"检测到AMD GPU：{skill_name}训练可加速运行，注意兼容性设置"
    # CPU 检测
    elif any(cpu in hardware_lower for cpu in ["cpu", "核", "i3", "i5", "i7", "i9", 
                                                 "ryzen", "xeon", "arm"]):
        return f"检测到CPU环境：{skill_name}推理可正常运行，训练建议使用更小的batch size"
    return f"当前环境：{skill_name}学习资源已适配您的设备条件"


def _generate_summary(grade: str, daily_hours: float, learning_style: str,
                      estimated_hours: float, study_days: int) -> str:
    """生成个性化摘要。"""
    parts = []
    if grade:
        parts.append(f"适合{grade}学习者")
    parts.append(f"预计{estimated_hours}小时完成")
    parts.append(f"建议分{study_days}天学习")
    if daily_hours:
        parts.append(f"每天{daily_hours}小时")
    style_label = {"practice": "动手型", "visual": "视觉型", "auditory": "听觉型", "reading": "阅读型",
                   "kinesthetic": "动手型", "mixed": "综合型"}.get(learning_style, "综合型")
    parts.append(f"{style_label}学习风格")
    return " | ".join(parts)


def _fallback_stage(session_id: str, skill_name: str, level: str, goal: str,
                    stage_num: int, profile: Dict[str, Any] = None) -> dict:
    """LLM失败时的回退阶段计划。"""
    profile = profile or {}
    stage_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{session_id}:{stage_num}:{skill_name}"))
    block_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{skill_name}"))

    difficulty_map = {"beginner": "初级", "intermediate": "中级", "advanced": "高级"}

    daily_hours = float(profile.get("daily_hours", 2.0))
    learning_style = profile.get("learning_style", "mixed")
    hardware = profile.get("hardware", "")
    grade = profile.get("grade", "")

    adjusted_hours = _adjust_hours_by_profile(2.0, daily_hours, level)
    study_days = max(1, round(adjusted_hours / max(daily_hours, 0.5)))

    return {
        "stage_id": stage_id,
        "stage_num": stage_num,
        "block_id": block_id,
        "block_name": skill_name,
        "estimated_hours": adjusted_hours,
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
        "recommended_daily_hours": daily_hours,
        "suggested_study_days": study_days,
        "learning_style_notes": _generate_style_notes(learning_style, skill_name),
        "hardware_notes": _generate_hardware_notes(hardware, skill_name),
        "personalization_summary": _generate_summary(grade, daily_hours, learning_style, adjusted_hours, study_days),
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
                             block_info: Dict[str, Any], goal: str, stage_num: int = 1,
                             profile: Dict[str, Any] = None) -> Dict[str, Any]:
        input_data = {
            "session_id": session_id,
            "gap": gap,
            "block_info": block_info,
            "goal": goal,
            "stage_num": stage_num,
        }
        if profile:
            input_data["profile"] = profile
        result = stage_generator(input_data, config=self._default_config)
        if result.get("status") != "success":
            return _fallback_stage(session_id, gap.get("skill", "未知技能"),
                                   gap.get("level", "beginner"), goal, stage_num, profile)
        return result["data"].get("stage", {})
