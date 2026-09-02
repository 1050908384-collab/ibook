# -*- coding: utf-8 -*-
"""
difficulty_analyzer_tool — 难度分析工具（原Agent5，合并至专业知识与教学质量审核 Agent）
================================================
功能：输入学习包内容与用户画像，输出内容难度等级（初级/中级/高级）及与用户水平的匹配度（0~1）。
    - 真实模式：调用 LLM（llm/client.py 同一 Ollama 端点）评估难度；LLM 不可用时降级为启发式
    - Mock 模式：返回 ./data/test_data/mock_difficulty_analyzer.json 预设匹配度

统一接口：
    def difficulty_analyzer(input_data: dict, config: dict = None) -> dict
    返回：{"status", "data", "message", "mock_used"}

依赖画像字段：grade、skill_tags、knowledge_gaps

规则对齐（learner_model.skill.md）：
    - 规则1 用户类型识别：学生 / 工程师 / 研究者（由 grade+goal+skill_tags 推断）
    - 规则2 难度适配映射：
        beginner(初级)   概念密度 ≤3      match: 1.0(区间内) / 0.6(±2) / 0.3(严重偏离)
        intermediate(中级) 4~7
        advanced(高级)    ≥8
    - difficulty_match < 0.5 → difficulty_mismatch(major)

实现说明：
    统一接口为同步函数，而 llm/client.py 的 LLMClient.complete 为 async。
    为在同步/异步上下文均可调用 LLM，真实模式用同步 httpx 直连 Ollama 端点
    （与 LLMClient 同一 /api/generate、同一 model），失败时降级启发式。
    概念密度计算复用 cognitive_load_tool 的 compute_density。
    保留 DifficultyAnalyzerTool 类作为向后兼容适配器。
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

import httpx

from config import DATA_DIR, settings
from tools.agent4_tools.cognitive_load_tool import compute_density

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

MOCK_FILE = DATA_DIR / "test_data" / "mock_difficulty_analyzer.json"

# 难度等级与概念密度区间（learner_model.skill.md 规则2）
_LEVEL_RANGES = {  # level -> (lower, upper) 概念密度
    "beginner": (0, 3),
    "intermediate": (4, 7),
    "advanced": (8, 9999),
}
_LEVEL_CN = {"beginner": "初级", "intermediate": "中级", "advanced": "高级"}
_CN_TO_LEVEL = {"初级": "beginner", "中级": "intermediate", "高级": "advanced"}
_LEVEL_RANK = {"beginner": 1, "intermediate": 2, "advanced": 3}


# ------------------------------------------------------------------
# 统一函数式接口（规范实现）
# ------------------------------------------------------------------
def difficulty_analyzer(input_data: dict, config: dict = None) -> dict:
    """
    分析内容难度与用户水平的匹配度。

    参数：
        input_data: dict
            - content: str              待分析内容
            - profile: dict             用户画像（grade / skill_tags / knowledge_gaps）
            - grade / skill_tags / knowledge_gaps  亦可顶层传入
            - scenario: str             Mock 场景：high_match|medium_match|low_match
        config: dict
            - use_mock: bool            是否使用 Mock（默认 True）

    返回：
        {
            "status": "success" | "error",
            "data": {
                "content_difficulty": "初级"|"中级"|"高级",
                "difficulty_match": float,        # 0~1
                "user_type": str,                 # 学生/工程师/研究者/未识别
                "user_level": "beginner"|...,
                "concept_density": float,
                "issues": [{issue_type, severity, location, description, suggestion}],
                "summary": str,
                "llm_used": bool,
                "fallback_used": bool
            },
            "message": str,
            "mock_used": bool
        }
    """
    config = config or {}
    use_mock = config.get("use_mock", True)

    content = (input_data or {}).get("content") or ""
    profile = _extract_profile(input_data)

    if not content.strip():
        return {
            "status": "error",
            "data": None,
            "message": "输入未包含 content 字段，无法分析难度",
            "mock_used": use_mock,
        }

    try:
        if use_mock:
            return _mock_analyze(content, profile, input_data)

        # 真实模式：先算启发式基线（密度、用户类型、用户水平），再尝试 LLM 覆盖
        density, _ = compute_density(content)
        user_type, type_note = _infer_user_type(profile)
        user_level = _infer_user_level(profile, content)
        heur_level = _density_to_level(density)
        heur_match = _match_score(density, user_level)

        llm_result, llm_used, fallback_used = _llm_judge(content, profile, user_type, user_level)

        if llm_result is not None:
            content_difficulty = llm_result.get("content_difficulty", _LEVEL_CN[heur_level])
            difficulty_match = _clamp(float(llm_result.get("difficulty_match", heur_match)))
            issues = llm_result.get("issues") or _build_issues(content_difficulty, user_level, difficulty_match)
            summary = llm_result.get("reason") or _build_summary(content_difficulty, difficulty_match, user_type, user_level, density, True)
        else:
            content_difficulty = _LEVEL_CN[heur_level]
            difficulty_match = heur_match
            issues = _build_issues(content_difficulty, user_level, difficulty_match)
            summary = _build_summary(content_difficulty, difficulty_match, user_type, user_level, density, False)
            if type_note:
                summary = f"{type_note}。{summary}"

        return {
            "status": "success",
            "data": {
                "content_difficulty": content_difficulty,
                "difficulty_match": round(difficulty_match, 2),
                "user_type": user_type,
                "user_level": user_level,
                "concept_density": round(density, 2),
                "issues": issues,
                "summary": summary,
                "llm_used": llm_used,
                "fallback_used": fallback_used,
            },
            "message": _build_message(content_difficulty, difficulty_match, llm_used, fallback_used),
            "mock_used": False,
        }
    except Exception as e:
        logger.exception("difficulty_analyzer 执行失败")
        return {
            "status": "error",
            "data": None,
            "message": f"difficulty_analyzer 执行异常: {e}",
            "mock_used": use_mock,
        }


# ------------------------------------------------------------------
# 画像提取与用户推断（learner_model.skill.md 规则1/2）
# ------------------------------------------------------------------
def _extract_profile(input_data: dict) -> dict:
    """合并顶层画像字段与 profile 嵌套字段。"""
    if not isinstance(input_data, dict):
        return {}
    profile = dict(input_data.get("profile") or {})
    for key in ("grade", "skill_tags", "knowledge_gaps", "goal", "learning_style", "professional_background"):
        if input_data.get(key) is not None and not profile.get(key):
            profile[key] = input_data[key]
    return profile


def _infer_user_type(profile: dict) -> tuple[str, str]:
    """识别用户类型：学生/工程师/研究者，未识别默认学生（带 note）。"""
    text = " ".join([
        str(profile.get("grade", "")),
        str(profile.get("goal", "")),
        str(profile.get("professional_background", "")),
    ])
    if any(k in text for k in ("博士", "研究", "学术", "论文", "教授", "科研")):
        return "研究者", ""
    if any(k in text for k in ("工程师", "开发", "后端", "前端", "从业", "职场", "算法工程师")):
        return "工程师", ""
    if any(k in text for k in ("学生", "大一", "大二", "大三", "大四", "研一", "研二", "研三", "本科", "研究生", "年级")):
        return "学生", ""
    if not text.strip():
        return "学生", "用户类型未识别，采用默认基准（学生）"
    return "学生", "用户类型未识别，采用默认基准（学生）"


def _infer_user_level(profile: dict, content: str) -> str:
    """
    推断用户水平：优先匹配内容主题相关的 knowledge_gap 的 level；
    否则取最弱 gap；无 gap 默认 beginner。
    兼容 profile.py 的 {skill,level} 与 DB 的 {topic,status,progress} 两种结构。
    """
    gaps = profile.get("knowledge_gaps") or []
    if not gaps:
        return "beginner"

    # 优先取主题出现在内容中的 gap
    target_gap = None
    for g in gaps:
        if not isinstance(g, dict):
            continue
        topic = g.get("skill") or g.get("topic") or ""
        if topic and topic in content:
            target_gap = g
            break
    if target_gap is None:
        # 取 progress 最低（最弱）的 gap
        def _prog(g):
            try:
                return float(g.get("progress", 0))
            except (TypeError, ValueError):
                return 0.0
        target_gap = min(
            (g for g in gaps if isinstance(g, dict)),
            key=_prog,
            default=gaps[0],
        )
    return _gap_to_level(target_gap)


def _gap_to_level(gap: dict) -> str:
    """从单个 gap 推断 level（level 字段 > progress > status）。"""
    level = (gap.get("level") or "").lower()
    if level in _LEVEL_RANK:
        return level
    if level in _CN_TO_LEVEL:
        return _CN_TO_LEVEL[level]
    # progress 推断
    try:
        p = float(gap.get("progress", 0))
        if p < 0.3:
            return "beginner"
        if p < 0.7:
            return "intermediate"
        return "advanced"
    except (TypeError, ValueError):
        pass
    # status 推断
    status = str(gap.get("status", ""))
    if any(k in status for k in ("未开始", "未掌握", "不懂")):
        return "beginner"
    if any(k in status for k in ("部分掌握", "了解")):
        return "intermediate"
    if any(k in status for k in ("掌握", "熟练", "精通")):
        return "advanced"
    return "beginner"


# ------------------------------------------------------------------
# 启发式难度与匹配度（learner_model.skill.md 规则2）
# ------------------------------------------------------------------
def _density_to_level(density: float) -> str:
    if density <= 3:
        return "beginner"
    if density <= 7:
        return "intermediate"
    return "advanced"


def _match_score(density: float, user_level: str) -> float:
    """难度匹配度：1.0(区间内) / 0.6(±2) / 0.3(严重偏离)。"""
    low, high = _LEVEL_RANGES[user_level]
    if low <= density <= high:
        return 1.0
    if (low - 2) <= density <= (high + 2):
        return 0.6
    return 0.3


def _build_issues(content_difficulty: str, user_level: str, match: float) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    content_rank = _LEVEL_RANK.get(_CN_TO_LEVEL.get(content_difficulty, "intermediate"), 2)
    user_rank = _LEVEL_RANK.get(user_level, 1)
    if match < 0.5:
        issue_type = "too_hard" if content_rank > user_rank else "too_easy"
        issues.append({
            "issue_type": "difficulty_mismatch",
            "severity": "major",
            "location": "内容整体",
            "description": f"内容难度 {content_difficulty} 与用户水平 {_LEVEL_CN[user_level]} 严重不匹配，匹配度 {match:.2f}。",
            "suggestion": "too_hard" if issue_type == "too_hard" else "too_easy",
        })
        issues[-1]["suggestion"] = (
            "增加基础扫盲段落、降低概念密度、拆分为多个阶段。"
            if issue_type == "too_hard"
            else "提升内容深度、增加工程化实践与高阶案例。"
        )
    elif match < 0.8:
        issues.append({
            "issue_type": "difficulty_slightly_off",
            "severity": "minor",
            "location": "核心概念段落",
            "description": f"内容难度 {content_difficulty} 略偏离用户水平 {_LEVEL_CN[user_level]}，匹配度 {match:.2f}。",
            "suggestion": "微调概念密度或补充过渡说明。",
        })
    return issues


def _build_summary(content_difficulty, match, user_type, user_level, density, llm_used) -> str:
    method = "LLM评估" if llm_used else "启发式估算"
    return (
        f"[{method}] 内容难度 {content_difficulty}，用户为 {user_type}（{_LEVEL_CN[user_level]}），"
        f"匹配度 {match:.2f}，概念密度 {density:.1f}/千字。"
    )


def _build_message(content_difficulty, match, llm_used, fallback_used) -> str:
    if fallback_used:
        return f"LLM不可用，降级启发式：难度 {content_difficulty}，匹配度 {match:.2f}"
    if llm_used:
        return f"LLM评估：难度 {content_difficulty}，匹配度 {match:.2f}"
    return f"难度 {content_difficulty}，匹配度 {match:.2f}"


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


# ------------------------------------------------------------------
# 真实模式：LLM 评估（同步 httpx，调用 DeepSeek Chat Completions 接口）
# ------------------------------------------------------------------
_LLM_SYSTEM_PROMPT = """你是一位专业的教学质量评估专家。根据用户画像评估学习内容的难度是否匹配用户水平。
只返回 JSON，字段：content_difficulty(初级/中级/高级)、difficulty_match(0-1浮点)、reason(简述)、issues(数组，每项含issue_type/severity/description/suggestion，无问题返回[])。"""

_LLM_PROMPT_TEMPLATE = """用户画像：{profile_summary}

学习内容（前2000字符）：
{content}

请评估内容难度等级及与用户水平的匹配度，返回JSON。"""


def _llm_judge(content: str, profile: dict, user_type: str, user_level: str) -> tuple[Optional[dict], bool, bool]:
    """
    调用 LLM 评估难度。返回 (result_dict|None, llm_used, fallback_used)。
    经 LLMClient.complete_sync 调用：云端 DeepSeek 失败时自动回退本地
    deepseek-r1:7b（LLM_FALLBACK_PROVIDER=ollama 时），仍失败则降级启发式。
    """
    from llm.client import LLMClient

    profile_summary = (
        f"类型={user_type}, 水平={_LEVEL_CN[user_level]}, "
        f"年级={profile.get('grade','未知')}, "
        f"技能={profile.get('skill_tags','[]')}, "
        f"知识盲区={profile.get('knowledge_gaps','[]')}"
    )
    messages = [
        {"role": "system", "content": _LLM_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _LLM_PROMPT_TEMPLATE.format(
                profile_summary=profile_summary,
                content=content[:2000],
            ),
        },
    ]
    response = LLMClient().complete_sync(
        messages, max_tokens=1024, temperature=0.3, json_mode=True,
    )
    if not response or response.startswith("Error:"):
        logger.warning("LLM 调用失败，降级启发式: %s", response[:200])
        return None, False, True

    text = _clean_response(response)
    parsed = _parse_json(text)
    if parsed is None:
        logger.warning("LLM 响应无法解析，降级启发式: %s", text[:200])
        return None, False, True

    # 校验字段
    cd = parsed.get("content_difficulty")
    if cd not in ("初级", "中级", "高级"):
        return None, False, True
    try:
        dm = float(parsed.get("difficulty_match", 0))
    except (TypeError, ValueError):
        return None, False, True
    parsed["difficulty_match"] = _clamp(dm)
    return parsed, True, False


def _clean_response(text: str) -> str:
    text = text.replace("<think>", "").replace("</think>", "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"\s*```", "", text)
    return text.strip()


def _parse_json(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                return None
        return None


# ------------------------------------------------------------------
# Mock 模式
# ------------------------------------------------------------------
def _mock_analyze(content: str, profile: dict, input_data: dict) -> dict:
    """从 mock JSON 读取预设场景，按 scenario 或画像/密度自动选择。"""
    scenario = _pick_scenario(content, profile, input_data)
    scenarios = _load_mock()
    if scenario not in scenarios:
        scenario = "medium_match"

    data = json.loads(json.dumps(scenarios[scenario]))  # 深拷贝
    data["content_preview"] = content[:200]
    # 用画像信息回填，便于核对
    user_type, _ = _infer_user_type(profile)
    data["user_type"] = data.get("user_type", user_type)

    return {
        "status": "success",
        "data": data,
        "message": f"[Mock] 难度 {data.get('content_difficulty')}，匹配度 {data.get('difficulty_match')}",
        "mock_used": True,
    }


def _pick_scenario(content: str, profile: dict, input_data: dict) -> str:
    """优先 scenario；否则按启发式匹配度自动归类为 高/中/低 场景。"""
    scenario = (input_data or {}).get("scenario")
    if scenario:
        return scenario
    density, _ = compute_density(content)
    user_level = _infer_user_level(profile, content)
    match = _match_score(density, user_level)
    if match >= 0.8:
        return "high_match"
    if match >= 0.5:
        return "medium_match"
    return "low_match"


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
# 保留原调用的 async 类接口，委托给函数式核心。
# 支持画像传入：profile 可包含 grade / skill_tags / knowledge_gaps / learning_style
# analyze 返回 issues 列表（兼容旧调用方 print 的 type/message 字段）。
# ------------------------------------------------------------------
class DifficultyAnalyzerTool:
    def __init__(self):
        self._default_config = {"use_mock": True}

    def _clean_response(self, response: str) -> str:
        return _clean_response(response)

    async def analyze(self, content: str, profile: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        input_data: Dict[str, Any] = {"content": content}
        if profile is not None:
            input_data["profile"] = profile
        result = difficulty_analyzer(input_data, config=self._default_config)
        if result.get("status") != "success":
            return []
        issues = result["data"].get("issues", [])
        for it in issues:
            it["type"] = it.get("issue_type", "difficulty_mismatch")
            it["message"] = it.get("description", "")
        return issues
