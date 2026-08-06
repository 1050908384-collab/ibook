# -*- coding: utf-8 -*-
"""
llm_generate_tool — Agent3 LLM内容生成工具
================================================
功能：调用LLM生成教程、回答用户问题，并根据审核意见修订内容。
    - 真实模式：调用 LLM 生成四段式教程/回答，对齐 content_generation.skill.md
    - Mock 模式：返回 ./data/test_data/mock_llm_generate_tool.json 预设内容

统一接口：
    def llm_generate(input_data: dict, config: dict = None) -> dict
    返回：{"status", "data", "message", "mock_used"}

规则对齐（content_generation.skill.md + question_answering.skill.md）：
    - 教程四段式：已知回顾→核心概念拆解→实操演示→常见坑点（不包含自测题）
    - 答疑三段式：结论→原理→示例（不编造来源，聚焦当前阶段）
    - 修订优先级：critical > major > minor，保留正确内容

保留 LLMGenerateTool 类作为向后兼容适配器。
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

from config import BACKEND_DIR, DATA_DIR, settings
from llm.client import LLMClient

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

MOCK_FILE = DATA_DIR / "test_data" / "mock_llm_generate_tool.json"


# ------------------------------------------------------------------
# 统一函数式接口（规范实现）
# ------------------------------------------------------------------
def llm_generate(input_data: dict, config: dict = None) -> dict:
    """
    LLM生成教程或回答。

    参数：
        input_data: dict
            - mode: str              生成模式："tutorial"(教程) | "answer"(回答) | "fix_tutorial"(修订教程) | "fix_answer"(修订回答)
            - block_id: str          板块ID（教程模式）
            - gap: Dict              知识盲区（教程模式）
            - question: str          用户问题（回答模式）
            - tutorial: str          原教程内容（修订模式）
            - answer: str            原回答内容（修订模式）
            - errors: List[Dict]     审核错误列表（修订模式）
            - related_docs: List[Dict] 相关文档（可选，用于RAG）
            - scenario: str          Mock 模式下选择场景：normal|boundary|exception
        config: dict
            - use_mock: bool         是否使用 Mock（默认 True）

    返回：
        {
            "status": "success" | "error",
            "data": {
                "content": str,          生成的教程或回答
                "code_examples": List,   提取的代码示例（教程模式）
                "version": int,          版本号（修订模式，+1）
                "generated_by_llm": bool
            },
            "message": str,
            "mock_used": bool
        }
    """
    config = config or {}
    use_mock = config.get("use_mock", True)

    mode = input_data.get("mode", "tutorial")

    try:
        if use_mock:
            return _mock_generate(input_data)

        if mode == "tutorial":
            result = _generate_tutorial(input_data)
        elif mode == "answer":
            result = _generate_answer(input_data)
        elif mode == "fix_tutorial":
            result = _fix_tutorial(input_data)
        elif mode == "fix_answer":
            result = _fix_answer(input_data)
        else:
            result = _generate_tutorial(input_data)

        return {
            "status": "success",
            "data": result,
            "message": _build_message(mode, result),
            "mock_used": False,
        }
    except Exception as e:
        logger.exception("llm_generate 执行失败")
        fallback = _fallback_content(mode, input_data)
        return {
            "status": "success",
            "data": {
                "content": fallback,
                "code_examples": [],
                "version": input_data.get("version", 1),
                "generated_by_llm": False,
            },
            "message": f"LLM生成异常，已使用回退策略: {e}",
            "mock_used": False,
        }


# ------------------------------------------------------------------
# 真实模式：LLM 生成教程
# ------------------------------------------------------------------
def _generate_tutorial(input_data: dict) -> dict:
    """生成四段式教程：已知回顾→核心概念拆解→实操演示→常见坑点。"""
    block_id = input_data.get("block_id", "")
    gap = input_data.get("gap", {}) or {}
    skill_name = gap.get("skill") or _resolve_block_name(block_id) or block_id or "未知技能"

    system_prompt = """你是一个专业的技术教程编写专家。请用通俗易懂的语言编写一篇技术教程。

教程结构（四段式）：
1. 已知回顾：链接读者已掌握的相关技能，建立知识桥梁
2. 核心概念拆解：解释核心概念的定义与原理，每段不超过2个新概念
3. 实操演示：提供可运行的代码示例或真实场景案例
4. 常见坑点：列出2~3个典型误区或陷阱

重要：教程中不包含自测题，自测题由用户主动点击自测时单独抽取。
代码示例请遵循PEP8规范，包含必要的注释和异常处理。
目标读者是希望学习该技术的初学者到中级学习者。"""

    prompt = f"""请编写一篇关于"{skill_name}"的技术教程，包括：
1. 已知回顾：假设读者已掌握编程基础，请链接相关前置知识
2. 核心概念拆解：解释{skill_name}的定义和工作原理
3. 实操演示：提供一个简单可运行的代码示例
4. 常见坑点：学习{skill_name}时常见的问题和解决方案

请用markdown格式编写，内容要详细但易于理解。"""

    llm_client = LLMClient()
    response = llm_client.complete(prompt, system_prompt)

    code_examples = _extract_code_examples(response)

    return {
        "content": response,
        "code_examples": code_examples,
        "version": 1,
        "generated_by_llm": True,
    }


# ------------------------------------------------------------------
# 真实模式：LLM 生成回答
# ------------------------------------------------------------------
def _generate_answer(input_data: dict) -> dict:
    """生成回答：结论先行→原理支撑→示例佐证。"""
    question = input_data.get("question", "")

    system_prompt = """你是一个专业的技术问答助手。请准确、详细地回答用户的问题。

回答结构（三段式）：
1. 结论先行：首句给出明确结论
2. 原理支撑：解释原理或机制，不编造来源，无法确认时明确说明
3. 示例佐证：需要时提供代码或数值示例

重要：与当前学习阶段无关的问题简短作答，并提示聚焦本阶段学习。"""

    prompt = f"""问题：{question}

请详细回答这个问题，确保技术准确性和易懂性。如果涉及代码，请提供示例代码。"""

    llm_client = LLMClient()
    response = llm_client.complete(prompt, system_prompt)

    return {
        "content": response,
        "code_examples": [],
        "version": 1,
        "generated_by_llm": True,
    }


# ------------------------------------------------------------------
# 真实模式：LLM 修订教程
# ------------------------------------------------------------------
def _fix_tutorial(input_data: dict) -> dict:
    """根据审核意见修订教程。"""
    tutorial = input_data.get("tutorial", "")
    errors = input_data.get("errors", [])
    version = input_data.get("version", 1)

    error_str = "\n".join([f"- [{e.get('severity', 'unknown')}] {e.get('description', '')}" for e in errors])

    system_prompt = """你是一个专业的技术编辑。请根据反馈意见修改教程内容，确保技术准确性。

重要：
- 只修改被指出有问题的部分，保持其他正确内容不变
- critical错误必须修复，major错误尽量修复，minor错误酌情修复
- 教程结构保持四段式（已知回顾→核心概念拆解→实操演示→常见坑点）
- 不添加自测题，自测题由用户主动点击时单独抽取"""

    prompt = f"""原教程内容：
{tutorial}

需要修复的问题：
{error_str}

请修改教程，解决以上问题，并保持内容的完整性和可读性。"""

    llm_client = LLMClient()
    response = llm_client.complete(prompt, system_prompt)

    code_examples = _extract_code_examples(response)

    return {
        "content": response,
        "code_examples": code_examples,
        "version": version + 1,
        "generated_by_llm": True,
    }


# ------------------------------------------------------------------
# 真实模式：LLM 修订回答
# ------------------------------------------------------------------
def _fix_answer(input_data: dict) -> dict:
    """根据审核意见修订回答。"""
    answer = input_data.get("answer", "")
    errors = input_data.get("errors", [])
    question = input_data.get("question", "")

    tech_errors = [e for e in errors if e.get("category") == "technical"]
    quality_errors = [e for e in errors if e.get("category") == "quality"]

    tech_error_str = "\n".join([f"- [技术错误] [{e.get('type', 'unknown')}] {e.get('message', '')}" for e in tech_errors])
    quality_error_str = "\n".join([f"- [质量问题] [{e.get('type', 'unknown')}] {e.get('message', '')}" for e in quality_errors])

    all_errors = []
    if tech_error_str:
        all_errors.append(f"技术错误：\n{tech_error_str}")
    if quality_error_str:
        all_errors.append(f"质量问题：\n{quality_error_str}")
    error_str = "\n\n".join(all_errors)

    system_prompt = """你是一个专业的技术编辑。请根据反馈意见修改回答。

重要：只修改被指出有问题的部分，保持其他正确内容不变，不要删减任何信息。"""

    question_part = f"原始问题：{question}\n\n" if question else ""
    prompt = f"""{question_part}原回答：
{answer}

需要修复的问题：
{error_str}

请修改回答，只针对上述问题进行修正。保持回答的完整性和详细程度，不要删减正确的内容。"""

    llm_client = LLMClient()
    response = llm_client.complete(prompt, system_prompt)

    return {
        "content": response,
        "code_examples": [],
        "version": 1,
        "generated_by_llm": True,
    }


# ------------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------------
import re


def _extract_code_examples(content: str) -> List[Dict[str, Any]]:
    """从教程中提取代码示例。"""
    examples = []
    code_blocks = re.findall(r'```(\w*)\n(.*?)```', content, re.DOTALL)
    for lang, code in code_blocks:
        examples.append({
            "title": f"代码示例{len(examples) + 1}",
            "language": lang or "python",
            "code": code.strip(),
        })
    return examples


def _resolve_block_name(block_id: str) -> str:
    """当 gap 为空时，根据 block_id 反查 knowledge_blocks.block_name 作为技能名。

    避免把 block_id（uuid 哈希）直接塞进 prompt 当技能名。
    查询失败或无记录时返回空字符串，由调用方继续回退。
    """
    if not block_id:
        return ""
    try:
        url = getattr(settings, "DATABASE_URL", "")
        raw = url.replace("sqlite:///", "", 1)
        p = Path(raw)
        if not p.is_absolute():
            p = (BACKEND_DIR / p).resolve()
        if not p.exists():
            return ""
        conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        try:
            cur = conn.execute(
                "SELECT block_name FROM knowledge_blocks WHERE block_id = ? LIMIT 1",
                (block_id,),
            )
            row = cur.fetchone()
        finally:
            conn.close()
        return row[0] if row else ""
    except Exception as e:
        logger.warning("反查 block_name 失败 (block_id=%s): %s", block_id, e)
        return ""


def _fallback_content(mode: str, input_data: dict) -> str:
    """LLM失败时的回退内容。"""
    if mode == "tutorial":
        gap = input_data.get("gap", {}) or {}
        block_id = input_data.get("block_id", "")
        skill_name = gap.get("skill") or _resolve_block_name(block_id) or "未知技能"
        return f"""# {skill_name}入门

## 已知回顾
假设你已掌握编程基础。

## 核心概念拆解
{skill_name}是一个重要的技术概念。

## 实操演示
```python
# {skill_name}示例代码
pass
```

## 常见坑点
- 学习时注意理解核心原理"""
    elif mode == "answer":
        question = input_data.get("question", "")
        return f"关于\"{question}\"的问题，建议参考相关教程内容。"
    elif mode == "fix_tutorial":
        return input_data.get("tutorial", "")
    else:
        return input_data.get("answer", "")


def _build_message(mode: str, result: dict) -> str:
    if mode == "tutorial":
        return f"教程生成完成，提取到 {len(result.get('code_examples', []))} 个代码示例"
    elif mode == "answer":
        return "回答生成完成"
    elif mode == "fix_tutorial":
        return f"教程修订完成，版本 {result.get('version', 1)}"
    else:
        return "回答修订完成"


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
    data.setdefault("code_examples", [])
    data.setdefault("version", 1)
    data.setdefault("generated_by_llm", True)

    return {
        "status": "success",
        "data": data,
        "message": f"[Mock] {_build_message(input_data.get('mode', 'tutorial'), data)}",
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
class LLMGenerateTool:
    def __init__(self):
        self._default_config = {"use_mock": True}
        self.llm_client = LLMClient()

    async def generate_tutorial(self, block_id: str, gap: Dict[str, Any],
                                related_docs: List[Dict[str, Any]] = None) -> str:
        result = llm_generate({
            "mode": "tutorial",
            "block_id": block_id,
            "gap": gap,
            "related_docs": related_docs or [],
        }, config=self._default_config)
        if result.get("status") != "success":
            return _fallback_content("tutorial", {"gap": gap})
        return result["data"].get("content", "")

    async def fix_tutorial(self, tutorial: str, errors: List[Dict[str, Any]]) -> str:
        result = llm_generate({
            "mode": "fix_tutorial",
            "tutorial": tutorial,
            "errors": errors,
        }, config=self._default_config)
        if result.get("status") != "success":
            return tutorial
        return result["data"].get("content", tutorial)

    async def adjust_tutorial_difficulty(self, tutorial: str, errors: List[Dict[str, Any]]) -> str:
        return tutorial

    async def generate_answer(self, question: str, related_docs: List[Dict[str, Any]] = None,
                              tutorial: str = "", qa_history: List[Dict[str, Any]] = None,
                              profile_summary: str = "") -> str:
        """生成技术类回答：向量检索 + LLM 生成，整合 memory 上下文。

        新增 memory 参数（可选）：
        - tutorial: 当前学习包教程内容，作为技术参考材料
        - qa_history: 历史问答记录，用于理解用户上下文
        - profile_summary: 用户画像摘要，个性化回答
        """
        result = llm_generate({
            "mode": "answer",
            "question": question,
            "related_docs": related_docs or [],
        }, config=self._default_config)
        if result.get("status") != "success":
            return _fallback_content("answer", {"question": question})
        base_answer = result["data"].get("content", "")

        # 如果有 memory 上下文，用 LLM 整合生成更贴合用户的回答
        if tutorial or qa_history or profile_summary:
            integrated = await self._integrate_memory_to_answer(
                question, base_answer, tutorial, qa_history or [], profile_summary
            )
            return integrated
        return base_answer

    async def _integrate_memory_to_answer(
        self,
        question: str,
        base_answer: str,
        tutorial: str,
        qa_history: List[Dict[str, Any]],
        profile_summary: str,
    ) -> str:
        """将 memory 上下文整合到技术回答中，使回答更贴合用户当前学习进度。"""
        history_text = ""
        if qa_history:
            recent = qa_history[-5:]  # 只取最近5轮，避免 prompt 过长
            history_text = "\n".join([
                f"用户问：{item.get('question', '')}\n答：{item.get('answer', '')[:200]}"
                for item in recent
                if isinstance(item, dict)
            ])

        tutorial_text = (tutorial or "")[:2000]  # 截断，避免 prompt 过长
        profile_text = profile_summary or ""

        system_prompt = """你是一个专业的技术辅导老师。请基于以下信息整合生成最终回答：

1. 保持技术准确性，不编造来源
2. 结合用户的学习进度和历史问题，让回答更贴合用户当前水平
3. 如果学习包教程中有相关内容，优先引用并补充说明
4. 回答结构保持：结论先行→原理支撑→示例佐证"""

        prompt = f"""用户问题：{question}

基础回答（基于向量检索生成）：
{base_answer}

用户画像摘要：
{profile_text}

当前学习包教程片段：
{tutorial_text}

历史问答记录：
{history_text}

请整合以上信息，生成一个更贴合用户当前学习进度和水平的回答。保持技术准确性，结合用户的历史问题和学习内容。"""

        try:
            response = await self.llm_client.complete(prompt, system_prompt)
            if response and not response.startswith("Error:"):
                return response
            return base_answer
        except Exception as e:
            logger.warning("整合 memory 到回答失败，返回基础回答: %s", e)
            return base_answer

    async def classify_question(self, question: str) -> str:
        """用 LLM 判断问题是技术类还是日常类。

        返回值："technical"（技术类）或 "daily"（日常类）。

        分类标准：
        - technical: 涉及代码、算法、概念解释、技术原理、API 使用、报错排查等
        - daily: 学习计划咨询、情绪疏导、学习鼓励、非技术闲聊等
        """
        system_prompt = """你是一个问题分类器。请判断用户的问题属于"技术类"还是"日常类"。

分类标准：
- technical（技术类）：涉及代码、算法、概念解释、技术原理、API使用、报错排查、编程实现、框架工具等技术性问题
- daily（日常类）：学习计划咨询（如"我该怎么学"）、学习状态/情绪（如"今天状态不好"）、学习意义咨询（如"学这个有什么用"）、非技术闲聊等

请只输出一个JSON：{"type": "technical"} 或 {"type": "daily"}"""

        prompt = f"""请判断以下问题属于技术类还是日常类：

用户问题：{question}

请输出JSON格式结果。"""

        try:
            response = await self.llm_client.complete(
                prompt, system_prompt, temperature=0.1, json_mode=True
            )
            if response and not response.startswith("Error:"):
                # 尝试解析 JSON
                import json as _json
                try:
                    result = _json.loads(response)
                    q_type = result.get("type", "technical").lower()
                    if q_type in ("technical", "daily"):
                        return q_type
                except _json.JSONDecodeError:
                    # JSON 解析失败，用关键词兜底
                    pass
            # LLM 失败时用关键词兜底分类
            return self._classify_by_keywords(question)
        except Exception as e:
            logger.warning("LLM 分类问题失败，使用关键词兜底: %s", e)
            return self._classify_by_keywords(question)

    @staticmethod
    def _classify_by_keywords(question: str) -> str:
        """LLM 不可用时的关键词兜底分类。"""
        # 日常类关键词
        daily_keywords = [
            "怎么学", "如何学", "学习计划", "学习路线", "状态不好", "学不进去",
            "有什么用", "为什么学", "坚持", "放弃", "焦虑", "累", "难",
            "鼓励", "你好", "谢谢", "再见", "今天", "心情",
        ]
        # 技术类关键词
        tech_keywords = [
            "代码", "报错", "error", "bug", "算法", "原理", "公式", "实现",
            "函数", "类", "接口", "api", "框架", "配置", "安装", "部署",
            "python", "pytorch", "transformer", "怎么写", "如何实现",
        ]
        q_lower = question.lower()
        for kw in tech_keywords:
            if kw in q_lower:
                return "technical"
        for kw in daily_keywords:
            if kw in question:
                return "daily"
        # 默认按技术类处理（保守策略，确保技术问题不被误判为日常）
        return "technical"

    async def generate_daily_answer(
        self,
        question: str,
        tutorial: str,
        qa_history: List[Dict[str, Any]],
        profile_summary: str,
    ) -> str:
        """生成日常类回答：只用历史对话 memory + 用户画像。

        日常类回答不涉及技术准确性，无需 Agent4 审核。
        注：tutorial 参数保留用于向后兼容，但不再使用。
        """
        history_text = ""
        if qa_history:
            recent = qa_history[-5:]
            history_text = "\n".join([
                f"用户问：{item.get('question', '')}\n答：{item.get('answer', '')[:200]}"
                for item in recent
                if isinstance(item, dict)
            ])

        profile_text = profile_summary or "暂无用户画像信息"

        system_prompt = """你是一个温暖、专业的学习导师。用户提出的是日常类问题（非技术问题）。

请基于以下信息生成有温度、有针对性的回答：
1. 结合用户画像（学习目标、水平、偏好），给出个性化建议
2. 参考历史对话，理解用户当前的学习状态和情绪
3. 不要引用技术学习材料，保持轻松自然的对话风格
4. 语气要亲切自然，像朋友间的对话，避免说教
5. 回答要简洁，不要过长（一般不超过300字）"""

        prompt = f"""用户问题：{question}

用户画像摘要：
{profile_text}

历史问答记录：
{history_text}

请生成一个温暖、有针对性的回答。"""

        try:
            response = await self.llm_client.complete(prompt, system_prompt, temperature=0.8)
            if response and not response.startswith("Error:"):
                return response
            return self._daily_fallback(question, profile_summary)
        except Exception as e:
            logger.warning("生成日常类回答失败，使用兜底: %s", e)
            return self._daily_fallback(question, profile_summary)

    @staticmethod
    def _daily_fallback(question: str, profile_summary: str) -> str:
        """LLM 不可用时日常类回答的兜底内容。"""
        goal = ""
        if profile_summary:
            for line in profile_summary.split("\n"):
                if "目标" in line or "goal" in line.lower():
                    goal = line.split("：")[-1].split(":")[-1].strip()
                    break
        if goal:
            return f"我理解你的感受。你目前的学习目标是「{goal}」，学习过程中遇到困惑很正常。可以告诉我具体是哪方面让你困扰吗？"
        return "我理解你的感受。学习过程中遇到困惑很正常，可以告诉我具体是哪方面让你困扰吗？我们一起想办法。"

    async def fix_answer(self, answer: str, errors: List[Dict[str, Any]], question: str = "") -> str:
        return await self._modify_answer(answer, errors, [], question)

    async def adjust_answer(self, answer: str, errors: List[Dict[str, Any]], question: str = "") -> str:
        return await self._modify_answer(answer, [], errors, question)

    async def _modify_answer(self, answer: str, tech_errors: List[Dict[str, Any]],
                             quality_errors: List[Dict[str, Any]], question: str = "") -> str:
        all_errors = []
        for e in tech_errors:
            all_errors.append({"category": "technical", "type": e.get("type"), "message": e.get("message")})
        for e in quality_errors:
            all_errors.append({"category": "quality", "type": e.get("type"), "message": e.get("message")})

        result = llm_generate({
            "mode": "fix_answer",
            "answer": answer,
            "errors": all_errors,
            "question": question,
        }, config=self._default_config)
        if result.get("status") != "success":
            return answer
        return result["data"].get("content", answer)
