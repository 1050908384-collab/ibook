# -*- coding: utf-8 -*-
"""
extract_skills_tool — Agent1 技能与盲区提取工具
================================================
功能：从用户自然语言输入提取已掌握技能标签（skill_tags）和知识盲区（knowledge_gaps）。
    - 真实模式：调用 LLM 做结构化提取，对齐 learner_profiling.skill.md 规则1
    - Mock 模式：返回 ./data/test_data/mock_extract_skills_tool.json 预设提取结果

统一接口：
    def extract_skills(input_data: dict, config: dict = None) -> dict
    返回：{"status", "data", "message", "mock_used"}

规则对齐（learner_profiling.skill.md）：
    - 规则1：skill_tags 为已掌握技能列表；knowledge_gaps 每项含 skill 与 level(beginner/intermediate/advanced)
    - 规则2：LLM返回非JSON时回退确定性正则匹配

保留 ExtractSkillsTool 类作为向后兼容适配器。
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

from config import DATA_DIR
from llm.client import LLMClient

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

MOCK_FILE = DATA_DIR / "test_data" / "mock_extract_skills_tool.json"


# ------------------------------------------------------------------
# 统一函数式接口（规范实现）
# ------------------------------------------------------------------
def extract_skills(input_data: dict, config: dict = None) -> dict:
    """
    从用户输入提取技能标签和知识盲区。

    参数：
        input_data: dict
            - user_input: str    用户原始输入（优先）
            - content: str       同义字段，user_input 缺失时使用
            - scenario: str      Mock 模式下选择场景：normal|boundary|exception
        config: dict
            - use_mock: bool     是否使用 Mock（默认 True）
            - mode: str          提取模式："all"(同时提取) | "skills"(仅技能) | "gaps"(仅盲区)

    返回：
        {
            "status": "success" | "error",
            "data": {
                "skill_tags": List[str],         已掌握技能标签
                "knowledge_gaps": List[Dict],    知识盲区（每项含 skill/level）
                "parsed_by_llm": bool,           是否由 LLM 解析成功
                "fallback_used": bool            是否使用回退策略
            },
            "message": str,
            "mock_used": bool
        }
    """
    config = config or {}
    use_mock = config.get("use_mock", True)
    mode = config.get("mode", "all")

    user_input = (input_data or {}).get("user_input") or (input_data or {}).get("content") or ""
    if not user_input.strip():
        return {
            "status": "error",
            "data": None,
            "message": "输入未包含 user_input 或 content 字段，无法提取",
            "mock_used": use_mock,
        }

    try:
        if use_mock:
            return _mock_extract(user_input, input_data, mode)

        if mode == "all":
            result = _extract_both(user_input)
        elif mode == "skills":
            result = {"skill_tags": _extract_skill_tags(user_input), "knowledge_gaps": []}
        elif mode == "gaps":
            result = {"skill_tags": [], "knowledge_gaps": _extract_knowledge_gaps(user_input)}
        else:
            result = _extract_both(user_input)

        result["parsed_by_llm"] = True
        result["fallback_used"] = False
        return {
            "status": "success",
            "data": result,
            "message": _build_message(result),
            "mock_used": False,
        }
    except Exception as e:
        logger.exception("extract_skills 执行失败")
        fallback = _fallback_extract(user_input)
        return {
            "status": "success",
            "data": {
                "skill_tags": fallback.get("skill_tags", []),
                "knowledge_gaps": fallback.get("knowledge_gaps", []),
                "parsed_by_llm": False,
                "fallback_used": True,
            },
            "message": f"LLM提取异常，已回退确定性提取: {e}",
            "mock_used": False,
        }


# ------------------------------------------------------------------
# 真实模式：LLM 结构化提取
# ------------------------------------------------------------------
_LLM_SYSTEM_PROMPT = """你是一位专业的学习者画像分析专家。从用户输入中提取：
1. skill_tags：用户已掌握的技能标签列表（如["Python", "PyTorch", "Transformer"]）
2. knowledge_gaps：用户想学习的知识盲区列表，每项含 skill 和 level（beginner/intermediate/advanced）
只返回严格JSON格式，不要额外文本。"""

_LLM_PROMPT_TEMPLATE = """请从以下用户输入中提取画像信息：

用户输入：{user_input}

返回JSON格式：
{{
  "skill_tags": ["技能1", "技能2"],
  "knowledge_gaps": [{{"skill": "想学的技能", "level": "beginner"}}]
}}"""


def _extract_both(user_input: str) -> dict:
    """同时提取技能标签和知识盲区（LLM优先，失败回退）。"""
    try:
        llm_result = _llm_extract(user_input)
        if llm_result is not None:
            return llm_result
    except Exception as e:
        logger.warning("LLM提取失败，回退确定性提取: %s", e)
    return _fallback_extract(user_input)


def _extract_skill_tags(user_input: str) -> List[str]:
    """仅提取已掌握技能标签。"""
    try:
        result = _llm_extract(user_input)
        if result is not None:
            return result.get("skill_tags", [])
    except Exception:
        pass
    return _fallback_skills(user_input)


def _extract_knowledge_gaps(user_input: str) -> List[Dict[str, Any]]:
    """仅提取知识盲区。"""
    try:
        result = _llm_extract(user_input)
        if result is not None:
            return result.get("knowledge_gaps", [])
    except Exception:
        pass
    return _fallback_gaps(user_input)


def _llm_extract(user_input: str) -> dict | None:
    """调用 LLM 做结构化提取，失败返回 None。"""
    import httpx
    from config import settings

    prompt = _LLM_PROMPT_TEMPLATE.format(user_input=user_input)
    # 复用与 llm/client.py 相同的占位符判定逻辑，避免未替换模板 key 时发请求
    placeholder_tokens = {
        "", "your-api-key-here", "your-deepseek-api-key",
        "your_api_key_here", "your_deepseek_api_key",
        "sk-yourkey", "sk-your-key", "placeholder", "xxx", "xxxx",
    }
    normalized_key = (settings.DEEPSEEK_API_KEY or "").strip().strip('"').strip("'").lower()
    key_ok = (
        bool(settings.DEEPSEEK_API_KEY)
        and normalized_key not in placeholder_tokens
        and not normalized_key.startswith("your-")
        and not normalized_key.startswith("your_")
    )
    if settings.LLM_PROVIDER.lower() != "deepseek" or not key_ok:
        return None
    api_base = settings.DEEPSEEK_API_BASE.rstrip("/")
    endpoint = (
        f"{api_base}/chat/completions"
        if api_base.endswith("/v1")
        else f"{api_base}/v1/chat/completions"
    )
    payload = {
        "model": settings.DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": _LLM_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 1024,
        "temperature": 0.3,
        "response_format": {"type": "json_object"},
    }
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.post(
                endpoint,
                json=payload,
                headers={"Authorization": f"Bearer {settings.DEEPSEEK_API_KEY}"},
            )
            resp.raise_for_status()
            text = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        text = _clean_response(text)
        parsed = json.loads(text)
        if "skill_tags" in parsed or "knowledge_gaps" in parsed:
            return {
                "skill_tags": parsed.get("skill_tags", []),
                "knowledge_gaps": parsed.get("knowledge_gaps", []),
            }
        return None
    except Exception as e:
        logger.warning("LLM提取解析失败: %s", e)
        return None


# ------------------------------------------------------------------
# 回退策略：确定性正则提取
# ------------------------------------------------------------------
def _fallback_extract(user_input: str) -> dict:
    """LLM失败时的回退提取策略。"""
    return {
        "skill_tags": _fallback_skills(user_input),
        "knowledge_gaps": _fallback_gaps(user_input),
    }


def _fallback_skills(user_input: str) -> List[str]:
    """基于关键词的技能标签回退提取。"""
    skill_keywords = {
        "python", "pytorch", "tensorflow", "机器学习", "深度学习", "大模型",
        "transformer", "nlp", "计算机视觉", "cv", "数据挖掘", "算法",
        "编程", "开发", "人工智能", "ai", "数据分析", "sql", "数据库",
    }
    tags = []
    for kw in skill_keywords:
        if kw.lower() in user_input.lower():
            tags.append(kw)
    return tags or ["编程基础"]


def _fallback_gaps(user_input: str) -> List[Dict[str, Any]]:
    """基于关键词的知识盲区回退提取（动态推导，非硬编码）。"""
    gap_patterns = [
        (r"(?:想学|学习|入门|了解|熟悉)\s*(.+)", "beginner"),
        (r"(?:提升|掌握|精通)\s*(.+)", "intermediate"),
        (r"(?:进阶|深入)\s*(.+)", "advanced"),
    ]
    for pattern, level in gap_patterns:
        m = re.search(pattern, user_input)
        if m:
            skill_text = m.group(1).strip()[:20]
            if skill_text:
                return [{"skill": skill_text, "level": level}]

    topic_keywords = {
        "LoRA": ["lora", "qlora", "高效微调"],
        "Transformer": ["transformer", "注意力机制"],
        "RAG": ["rag", "检索增强"],
        "微调": ["微调", "fine-tune", "finetune"],
        "大模型": ["大模型", "llm", "llm"],
        "PyTorch": ["pytorch"],
        "深度学习": ["深度学习", "dl"],
        "强化学习": ["强化学习", "rl"],
    }
    for skill, kws in topic_keywords.items():
        if any(kw.lower() in user_input.lower() for kw in kws):
            return [{"skill": skill, "level": "beginner"}]
    return []


# ------------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------------
def _clean_response(response: str) -> str:
    """清理LLM响应中的标记。"""
    response = response.replace("<think>", "").replace("</think>", "")
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    response = re.sub(r'```json\s*', '', response)
    response = re.sub(r'\s*```', '', response)
    return response.strip()


def _build_message(result: dict) -> str:
    tags = result.get("skill_tags", [])
    gaps = result.get("knowledge_gaps", [])
    fallback = result.get("fallback_used", False)

    base = f"提取到 {len(tags)} 个技能标签，{len(gaps)} 个知识盲区"
    if fallback:
        return f"{base}（使用回退策略）"
    return base


# ------------------------------------------------------------------
# Mock 模式
# ------------------------------------------------------------------
def _mock_extract(user_input: str, input_data: dict, mode: str) -> dict:
    """从 mock JSON 读取预设场景。"""
    scenario = (input_data or {}).get("scenario") or "normal"
    scenarios = _load_mock()
    if scenario not in scenarios:
        scenario = "normal"

    data = dict(scenarios[scenario])
    if mode == "skills":
        data["knowledge_gaps"] = []
    elif mode == "gaps":
        data["skill_tags"] = []

    data["parsed_by_llm"] = True
    data["fallback_used"] = False

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
class ExtractSkillsTool:
    def __init__(self):
        self._default_config = {"use_mock": False}
        self.llm_client = LLMClient()

    async def extract_skills(self, user_input: str) -> List[str]:
        result = extract_skills({"user_input": user_input}, config=self._default_config)
        if result.get("status") != "success":
            return []
        return result["data"].get("skill_tags", [])

    async def extract_gaps(self, user_input: str) -> List[Dict[str, Any]]:
        result = extract_skills({"user_input": user_input}, config={"use_mock": self._default_config["use_mock"], "mode": "gaps"})
        if result.get("status") != "success":
            return []
        return result["data"].get("knowledge_gaps", [])
