# -*- coding: utf-8 -*-
"""
code_generation_tool — Agent3 代码示例生成工具
================================================
功能：生成或修复教程中的代码示例，遵循 code_style.skill.md 规范。
    - 真实模式：查 knowledge_docs 表 doc_type='code' 的文档作为代码示例来源
    - Mock 模式：返回 ./data/test_data/mock_code_generation_tool.json 预设代码示例

统一接口：
    def code_generation(input_data: dict, config: dict = None) -> dict
    返回：{"status", "data", "message", "mock_used"}

规则对齐（code_style.skill.md）：
    - 规则1：PEP8 风格（行宽88、snake_case、导入分层）
    - 规则2：docstring 与注释（顶层函数/类必须有docstring）
    - 规则3：异常处理（IO/模型加载/网络必须try-except）
    - 规则4：配置化与安全（禁止硬编码路径/密钥）

保留 CodeGenerationTool 类作为向后兼容适配器。
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

MOCK_FILE = DATA_DIR / "test_data" / "mock_code_generation_tool.json"


# ------------------------------------------------------------------
# 统一函数式接口（规范实现）
# ------------------------------------------------------------------
def code_generation(input_data: dict, config: dict = None) -> dict:
    """
    生成或修复代码示例。

    参数：
        input_data: dict
            - mode: str              模式："generate"(生成) | "fix"(修复)
            - block_id: str          板块ID（生成模式）
            - skill_name: str        技能名称（备选）
            - code_examples: List[Dict] 代码示例列表（修复模式）
            - errors: List[Dict]     审核错误列表（修复模式）
            - scenario: str          Mock 模式下选择场景：normal|boundary|exception
        config: dict
            - use_mock: bool         是否使用 Mock（默认 True）

    返回：
        {
            "status": "success" | "error",
            "data": {
                "code_examples": [
                    {"title", "language", "code", "description"}
                ],
                "generated_by_llm": bool,
                "db_available": bool
            },
            "message": str,
            "mock_used": bool
        }
    """
    config = config or {}
    use_mock = config.get("use_mock", True)

    mode = input_data.get("mode", "generate")

    try:
        if use_mock:
            return _mock_generate(input_data)

        if mode == "generate":
            result = _generate_code(input_data)
        elif mode == "fix":
            result = _fix_code(input_data)
        else:
            result = _generate_code(input_data)

        return {
            "status": "success",
            "data": result,
            "message": _build_message(mode, result),
            "mock_used": False,
        }
    except Exception as e:
        logger.exception("code_generation 执行失败")
        fallback = input_data.get("code_examples", [])
        return {
            "status": "success",
            "data": {
                "code_examples": fallback,
                "generated_by_llm": False,
                "db_available": True,
            },
            "message": f"代码生成异常，已保留原代码示例: {e}",
            "mock_used": False,
        }


# ------------------------------------------------------------------
# 真实模式：代码生成
# ------------------------------------------------------------------
def _generate_code(input_data: dict) -> dict:
    """生成代码示例（优先从知识库读取，不足时LLM补充）。"""
    block_id = input_data.get("block_id", "")
    skill_name = input_data.get("skill_name", "") or block_id or "未知技能"

    db_examples, db_available = _load_code_docs(block_id)

    if db_examples:
        return {
            "code_examples": db_examples,
            "generated_by_llm": False,
            "db_available": db_available,
        }

    examples = _llm_generate_code(skill_name)
    return {
        "code_examples": examples,
        "generated_by_llm": True,
        "db_available": db_available,
    }


def _load_code_docs(block_id: str) -> tuple[List[Dict[str, Any]], bool]:
    """
    SQL: 从 knowledge_docs 表读取 doc_type='code' 的文档作为代码示例。
    """
    db_path = _get_db_path()
    if not db_path.exists():
        logger.warning("数据库不存在: %s，code_generation 无可用代码文档", db_path)
        return [], False

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            cur = conn.execute(
                """
                SELECT title, content, source
                FROM knowledge_docs
                WHERE block_id = ? AND doc_type = 'code'
                ORDER BY created_at DESC
                LIMIT 5
                """,
                (block_id,),
            )
            rows = cur.fetchall()
        finally:
            conn.close()

        examples = []
        for r in rows:
            examples.append({
                "title": r[0],
                "language": "python",
                "code": r[1],
                "description": r[2] or "",
            })
        return examples, True
    except sqlite3.Error as e:
        logger.warning("读取 knowledge_docs(code) 失败: %s", e)
        return [], False


def _llm_generate_code(skill_name: str) -> List[Dict[str, Any]]:
    """LLM生成代码示例（知识库无数据时回退）。"""
    system_prompt = """你是一个专业的Python代码生成专家。请生成一段符合PEP8规范的代码示例。

要求：
1. 包含完整的顶层函数或类，必须有docstring
2. 关键步骤有注释
3. 文件IO、模型加载等操作必须有try-except
4. 禁止硬编码路径、密钥、GPU编号
5. 使用PyTorch 2.x推荐API（model.to(device)而非model.cuda()）
6. 代码可运行，不要省略必要的导入和初始化"""

    prompt = f"""请为"{skill_name}"生成一个完整的代码示例，包含：
- 必要的导入
- 一个完整的函数或类（带docstring）
- 关键步骤注释
- 异常处理

代码功能：展示{skill_name}的核心用法。"""

    llm_client = LLMClient()
    response = llm_client.complete(prompt, system_prompt)

    import re
    code_blocks = re.findall(r'```(\w*)\n(.*?)```', response, re.DOTALL)
    if code_blocks:
        lang, code = code_blocks[0]
        return [{
            "title": f"{skill_name}代码示例",
            "language": lang or "python",
            "code": code.strip(),
            "description": f"{skill_name}核心用法示例",
        }]
    return []


# ------------------------------------------------------------------
# 真实模式：代码修复
# ------------------------------------------------------------------
def _fix_code(input_data: dict) -> dict:
    """根据审核意见修复代码示例。"""
    code_examples = input_data.get("code_examples", [])
    errors = input_data.get("errors", [])

    if not code_examples or not errors:
        return {
            "code_examples": code_examples,
            "generated_by_llm": False,
            "db_available": True,
        }

    error_str = "\n".join([f"- [{e.get('severity', 'unknown')}] {e.get('description', '')}" for e in errors])

    system_prompt = """你是一个专业的Python代码编辑。请根据反馈意见修复代码。

要求：
1. 修复PEP8风格问题（行宽、命名、导入顺序）
2. 补充缺失的docstring和注释
3. 添加缺失的异常处理
4. 将硬编码路径/密钥改为配置读取
5. 将过时API改为推荐写法
6. 只修改有问题的部分，保持功能不变"""

    prompt = f"""原代码示例：
{json.dumps(code_examples, indent=2, ensure_ascii=False)}

需要修复的问题：
{error_str}

请修复代码示例，返回修复后的代码示例列表（JSON格式）。"""

    llm_client = LLMClient()
    response = llm_client.complete(prompt, system_prompt)

    import re
    match = re.search(r'\[.*\]', response, re.DOTALL)
    if match:
        try:
            fixed = json.loads(match.group())
            if isinstance(fixed, list):
                return {
                    "code_examples": fixed,
                    "generated_by_llm": True,
                    "db_available": True,
                }
        except Exception as e:
            logger.warning("代码修复解析失败: %s", e)

    return {
        "code_examples": code_examples,
        "generated_by_llm": False,
        "db_available": True,
    }


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


def _build_message(mode: str, result: dict) -> str:
    count = len(result.get("code_examples", []))
    if mode == "generate":
        if result.get("generated_by_llm"):
            return f"LLM生成 {count} 个代码示例"
        return f"从知识库读取 {count} 个代码示例"
    return f"修复 {count} 个代码示例"


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
    data.setdefault("generated_by_llm", True)
    data.setdefault("db_available", False)

    return {
        "status": "success",
        "data": data,
        "message": f"[Mock] {_build_message(input_data.get('mode', 'generate'), data)}",
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
class CodeGenerationTool:
    def __init__(self):
        self._default_config = {"use_mock": True}

    async def generate_code(self, block_id: str) -> List[Dict[str, Any]]:
        result = code_generation({
            "mode": "generate",
            "block_id": block_id,
        }, config=self._default_config)
        if result.get("status") != "success":
            return []
        return result["data"].get("code_examples", [])

    async def fix_code(self, code_examples: List[Dict[str, Any]], errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        result = code_generation({
            "mode": "fix",
            "code_examples": code_examples,
            "errors": errors,
        }, config=self._default_config)
        if result.get("status") != "success":
            return code_examples
        return result["data"].get("code_examples", code_examples)
