# -*- coding: utf-8 -*-
"""
template_render_tool — 个性化学习资源生成 Agent 教程模板渲染工具
================================================
功能：使用 Jinja2 模板引擎渲染教程内容。
    - 真实模式：从模板文件读取并渲染
    - Mock 模式：返回 ./data/test_data/mock_template_render_tool.json 预设渲染结果

统一接口：
    def template_render(input_data: dict, config: dict = None) -> dict
    返回：{"status", "data", "message", "mock_used"}

规则对齐（content_generation.skill.md）：
    - 模板支持变量替换：{{ block_name }}、{{ estimated_hours }}、{{ prerequisites }}
    - 模板支持条件渲染和循环

保留 TemplateRenderTool 类作为向后兼容适配器。
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

from config import DATA_DIR

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

MOCK_FILE = DATA_DIR / "test_data" / "mock_template_render_tool.json"


# ------------------------------------------------------------------
# 统一函数式接口（规范实现）
# ------------------------------------------------------------------
def template_render(input_data: dict, config: dict = None) -> dict:
    """
    渲染教程模板。

    参数：
        input_data: dict
            - template_name: str     模板名称（不带扩展名）
            - context: dict          模板变量上下文
            - scenario: str          Mock 模式下选择场景：normal|boundary|exception
        config: dict
            - use_mock: bool         是否使用 Mock（默认 True）

    返回：
        {
            "status": "success" | "error",
            "data": {
                "rendered_content": str,
                "template_name": str
            },
            "message": str,
            "mock_used": bool
        }
    """
    config = config or {}
    use_mock = config.get("use_mock", True)

    template_name = input_data.get("template_name", "")
    context = input_data.get("context", {})

    if not template_name:
        return {
            "status": "error",
            "data": None,
            "message": "输入未包含 template_name 字段，无法渲染",
            "mock_used": use_mock,
        }

    try:
        if use_mock:
            return _mock_render(input_data)

        result = _render_template(template_name, context)
        return {
            "status": "success",
            "data": result,
            "message": f"模板渲染完成: {template_name}",
            "mock_used": False,
        }
    except Exception as e:
        logger.exception("template_render 执行失败")
        return {
            "status": "error",
            "data": None,
            "message": f"template_render 执行异常: {e}",
            "mock_used": use_mock,
        }


# ------------------------------------------------------------------
# 真实模式：Jinja2 模板渲染
# ------------------------------------------------------------------
def _render_template(template_name: str, context: dict) -> dict:
    """使用 Jinja2 渲染模板。"""
    template_dir = DATA_DIR / "templates"
    template_path = template_dir / f"{template_name}.jinja2"

    if not template_path.exists():
        logger.warning("模板文件不存在: %s，使用默认模板", template_path)
        return _render_default_template(context)

    try:
        from jinja2 import Environment, FileSystemLoader

        env = Environment(loader=FileSystemLoader(str(template_dir)))
        template = env.get_template(f"{template_name}.jinja2")
        rendered = template.render(**context)

        return {
            "rendered_content": rendered,
            "template_name": template_name,
        }
    except Exception as e:
        logger.warning("Jinja2渲染失败: %s，使用默认模板", e)
        return _render_default_template(context)


def _render_default_template(context: dict) -> dict:
    """默认模板渲染（无Jinja2或模板不存在时）。"""
    block_name = context.get("block_name", "未知技能")
    estimated_hours = context.get("estimated_hours", 2.0)
    prerequisites = context.get("prerequisites", [])

    prereq_str = ", ".join(prerequisites) if prerequisites else "无"

    content = f"""# {block_name}

## 学习目标
掌握{block_name}的核心概念和实践技能

## 预计时长
{estimated_hours}小时

## 先修知识
{prereq_str}

## 学习内容
详细内容将由LLM生成..."""

    return {
        "rendered_content": content,
        "template_name": "default",
    }


# ------------------------------------------------------------------
# Mock 模式
# ------------------------------------------------------------------
def _mock_render(input_data: dict) -> dict:
    """从 mock JSON 读取预设场景。"""
    scenario = input_data.get("scenario") or "normal"
    scenarios = _load_mock()
    if scenario not in scenarios:
        scenario = "normal"

    data = dict(scenarios[scenario])
    data.setdefault("template_name", input_data.get("template_name", "default"))

    return {
        "status": "success",
        "data": data,
        "message": f"[Mock] 模板渲染完成: {data['template_name']}",
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
class TemplateRenderTool:
    def __init__(self):
        self._default_config = {"use_mock": True}

    async def render(self, template_name: str, context: Dict[str, Any]) -> str:
        result = template_render({
            "template_name": template_name,
            "context": context,
        }, config=self._default_config)
        if result.get("status") != "success":
            return ""
        return result["data"].get("rendered_content", "")
