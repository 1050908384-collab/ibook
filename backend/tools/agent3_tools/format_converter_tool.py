# -*- coding: utf-8 -*-
"""
format_converter_tool — Agent3 格式转换工具
================================================
功能：在不同格式之间转换教程内容（Markdown↔HTML↔JSON）。
    - 真实模式：使用 Python 库进行格式转换
    - Mock 模式：返回 ./data/test_data/mock_format_converter_tool.json 预设转换结果

统一接口：
    def format_converter(input_data: dict, config: dict = None) -> dict
    返回：{"status", "data", "message", "mock_used"}

规则对齐（content_generation.skill.md）：
    - 转换保持内容完整性，不丢失代码块、表格、列表等结构
    - Markdown→HTML 保留代码高亮标记

保留 FormatConverterTool 类作为向后兼容适配器。
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

from config import DATA_DIR

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

MOCK_FILE = DATA_DIR / "test_data" / "mock_format_converter_tool.json"


# ------------------------------------------------------------------
# 统一函数式接口（规范实现）
# ------------------------------------------------------------------
def format_converter(input_data: dict, config: dict = None) -> dict:
    """
    格式转换。

    参数：
        input_data: dict
            - mode: str              转换模式："md_to_html" | "html_to_md" | "md_to_json" | "json_to_md"
            - content: str           待转换内容
            - scenario: str          Mock 模式下选择场景：normal|boundary|exception
        config: dict
            - use_mock: bool         是否使用 Mock（默认 True）

    返回：
        {
            "status": "success" | "error",
            "data": {
                "converted_content": str,
                "source_format": str,
                "target_format": str
            },
            "message": str,
            "mock_used": bool
        }
    """
    config = config or {}
    use_mock = config.get("use_mock", True)

    mode = input_data.get("mode", "md_to_html")
    content = input_data.get("content", "")

    if not content.strip():
        return {
            "status": "error",
            "data": None,
            "message": "输入未包含 content 字段或内容为空，无法转换",
            "mock_used": use_mock,
        }

    try:
        if use_mock:
            return _mock_convert(input_data)

        result = _convert(mode, content)
        return {
            "status": "success",
            "data": result,
            "message": f"格式转换完成: {result['source_format']}→{result['target_format']}",
            "mock_used": False,
        }
    except Exception as e:
        logger.exception("format_converter 执行失败")
        return {
            "status": "error",
            "data": None,
            "message": f"format_converter 执行异常: {e}",
            "mock_used": use_mock,
        }


# ------------------------------------------------------------------
# 真实模式：格式转换
# ------------------------------------------------------------------
def _convert(mode: str, content: str) -> dict:
    """根据模式执行格式转换。"""
    format_map = {
        "md_to_html": ("markdown", "html", _md_to_html),
        "html_to_md": ("html", "markdown", _html_to_md),
        "md_to_json": ("markdown", "json", _md_to_json),
        "json_to_md": ("json", "markdown", _json_to_md),
    }

    if mode not in format_map:
        logger.warning("未知转换模式: %s，使用 md_to_html", mode)
        mode = "md_to_html"

    source_format, target_format, converter = format_map[mode]
    converted = converter(content)

    return {
        "converted_content": converted,
        "source_format": source_format,
        "target_format": target_format,
    }


def _md_to_html(content: str) -> str:
    """Markdown → HTML。"""
    try:
        import markdown
        return markdown.markdown(content, extensions=["fenced_code", "tables", "toc"])
    except ImportError:
        logger.warning("markdown库未安装，使用简单转换")
        return _simple_md_to_html(content)


def _simple_md_to_html(content: str) -> str:
    """简单 Markdown→HTML 转换（无依赖）。"""
    import re
    html = content
    html = re.sub(r'^### (.*)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.*)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^# (.*)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
    html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'\*(.*?)\*', r'<em>\1</em>', html)
    html = re.sub(r'```(\w*)\n(.*?)```', r'<pre><code class="language-\1">\2</code></pre>', html, flags=re.DOTALL)
    html = re.sub(r'`([^`]+)`', r'<code>\1</code>', html)
    html = re.sub(r'^- (.*)$', r'<li>\1</li>', html, flags=re.MULTILINE)
    html = re.sub(r'(<li>.*</li>)', r'<ul>\1</ul>', html)
    html = html.replace('\n', '<br>')
    return html


def _html_to_md(content: str) -> str:
    """HTML → Markdown。"""
    try:
        from markdownify import markdownify as md
        return md(content)
    except ImportError:
        logger.warning("markdownify库未安装，返回原内容")
        return content


def _md_to_json(content: str) -> str:
    """Markdown → JSON。"""
    import re

    sections = []
    current_section = {"title": "", "content": ""}

    lines = content.split('\n')
    for line in lines:
        if line.startswith('#'):
            if current_section["title"]:
                sections.append(current_section)
            level = len(line) - len(line.lstrip('#'))
            current_section = {
                "title": line.strip('#').strip(),
                "level": level,
                "content": "",
            }
        else:
            current_section["content"] += line + '\n'

    if current_section["title"]:
        sections.append(current_section)

    result = {"type": "markdown", "sections": sections}
    return json.dumps(result, ensure_ascii=False, indent=2)


def _json_to_md(content: str) -> str:
    """JSON → Markdown。"""
    try:
        data = json.loads(content)
        sections = data.get("sections", [])
        md_lines = []
        for section in sections:
            level = section.get("level", 1)
            title = section.get("title", "")
            content_text = section.get("content", "")
            md_lines.append('#' * level + ' ' + title)
            md_lines.append(content_text)
        return '\n'.join(md_lines)
    except Exception as e:
        logger.warning("JSON解析失败: %s，返回原内容", e)
        return content


# ------------------------------------------------------------------
# Mock 模式
# ------------------------------------------------------------------
def _mock_convert(input_data: dict) -> dict:
    """从 mock JSON 读取预设场景。"""
    scenario = input_data.get("scenario") or "normal"
    scenarios = _load_mock()
    if scenario not in scenarios:
        scenario = "normal"

    data = dict(scenarios[scenario])

    mode = input_data.get("mode", "md_to_html")
    format_map = {
        "md_to_html": ("markdown", "html"),
        "html_to_md": ("html", "markdown"),
        "md_to_json": ("markdown", "json"),
        "json_to_md": ("json", "markdown"),
    }
    source_format, target_format = format_map.get(mode, ("markdown", "html"))

    data.setdefault("source_format", source_format)
    data.setdefault("target_format", target_format)

    return {
        "status": "success",
        "data": data,
        "message": f"[Mock] 格式转换完成: {source_format}→{target_format}",
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
class FormatConverterTool:
    def __init__(self):
        self._default_config = {"use_mock": True}

    async def convert(self, content: str, mode: str = "md_to_html") -> str:
        result = format_converter({
            "mode": mode,
            "content": content,
        }, config=self._default_config)
        if result.get("status") != "success":
            return content
        return result["data"].get("converted_content", content)
