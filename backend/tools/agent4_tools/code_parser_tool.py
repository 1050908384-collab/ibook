"""
code_parser_tool — Agent4 代码语法检查工具
================================================
功能：使用 Python 标准库 ast 检查代码语法，返回是否有错误及错误行号/偏移。
特性：纯计算，不依赖外部服务，无 Mock（mock_used 始终为 False）。

统一接口：
    def code_parser(input_data: dict, config: dict = None) -> dict
    返回：{"status", "data", "message", "mock_used"}

对齐说明：
    - 错误结构对齐 models/review.py 的 ReviewError（error_type/severity/location/description/...）
    - 语法错误统一映射为 severity="critical"（见 technical_review.skill.md 规则3：语法错误属 Critical 阻断性错误）
    - 保留 CodeParserTool 类作为向后兼容适配器，供现有 agent_04_tech_review.py 调用
"""

import ast
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


# ------------------------------------------------------------------
# 统一函数式接口（规范实现）
# ------------------------------------------------------------------
def code_parser(input_data: dict, config: dict = None) -> dict:
    """
    用 ast 库检查代码语法错误。

    参数：
        input_data: dict
            - code: str              单段代码（优先使用）
            - code_examples: list    多段代码，元素可为 {"code": "..."} 或裸字符串
            - scenario: str          可选，本工具为纯计算，忽略该字段
        config: dict                 本工具为纯计算，忽略 use_mock 等字段

    返回：
        {
            "status": "success" | "error",
            "data": {
                "has_errors": bool,
                "errors": [
                    {
                        "error_type": "SyntaxError" | "IndentationError" | "TabError" | "ValueError",
                        "severity": "critical",
                        "line": int | None,            # 出错行号
                        "offset": int | None,          # 列偏移
                        "end_line": int | None,
                        "location": "代码块#{idx} 第{line}行",
                        "description": str,            # 错误信息（ast 的 e.msg）
                        "suggestion": str              # 修复建议
                    }
                ],
                "total_checked": int,
                "code_length": int
            },
            "message": str,
            "mock_used": False
        }
    """
    # 本工具为纯计算，config 仅做形式化接收，不切换 Mock
    _ = config or {}

    try:
        codes = _extract_codes(input_data)
        if not codes:
            return {
                "status": "error",
                "data": None,
                "message": "输入未包含 code 或 code_examples 字段，无法解析",
                "mock_used": False,
            }

        all_errors: List[Dict[str, Any]] = []
        for idx, code in enumerate(codes):
            for err in _parse_one(code):
                err["code_block_index"] = idx
                err["location"] = f"代码块#{idx} 第{err.get('line')}行"
                all_errors.append(err)

        has_errors = len(all_errors) > 0
        if has_errors:
            msg = f"已检查 {len(codes)} 段代码，发现 {len(all_errors)} 个语法错误"
        else:
            msg = f"已检查 {len(codes)} 段代码，语法正确"

        return {
            "status": "success",
            "data": {
                "has_errors": has_errors,
                "errors": all_errors,
                "total_checked": len(codes),
                "code_length": sum(len(c) for c in codes),
            },
            "message": msg,
            "mock_used": False,
        }
    except Exception as e:  # 兜底：任何异常都不应让调用方崩溃
        logger.exception("code_parser 执行失败")
        return {
            "status": "error",
            "data": None,
            "message": f"code_parser 执行异常: {e}",
            "mock_used": False,
        }


# ------------------------------------------------------------------
# 内部实现
# ------------------------------------------------------------------
def _extract_codes(input_data: Any) -> List[str]:
    """从 input_data 中归一化提取待检查的代码段列表。"""
    codes: List[str] = []
    if isinstance(input_data, str):
        codes.append(input_data)
        return codes
    if not isinstance(input_data, dict):
        return codes

    if input_data.get("code"):
        codes.append(input_data["code"])
    for ce in input_data.get("code_examples", []) or []:
        if isinstance(ce, dict) and ce.get("code"):
            codes.append(ce["code"])
        elif isinstance(ce, str):
            codes.append(ce)
    return codes


def _parse_one(code: str) -> List[Dict[str, Any]]:
    """对单段代码执行 ast.parse，返回语法错误列表。"""
    if not isinstance(code, str) or not code.strip():
        return []

    try:
        # ast.parse 仅做语法解析，不执行代码，安全
        ast.parse(code)
        return []
    except SyntaxError as e:
        # IndentationError / TabError 均为 SyntaxError 子类，type 名可区分
        return [{
            "error_type": type(e).__name__,
            "severity": "critical",  # 语法错误属 Critical 阻断性错误
            "line": e.lineno,
            "offset": e.offset,
            "end_line": getattr(e, "end_lineno", e.lineno),
            "description": e.msg or type(e).__name__,
            "suggestion": _syntax_suggestion(type(e).__name__, e.msg),
        }]
    except ValueError as e:
        # 含 NUL 字节等非法字符时 ast 会抛 ValueError
        return [{
            "error_type": "ValueError",
            "severity": "critical",
            "line": None,
            "offset": None,
            "end_line": None,
            "description": str(e),
            "suggestion": "清理代码中的非法字符（如 NUL 字节、编码错误字符）后重试",
        }]


def _syntax_suggestion(error_type: str, message: str) -> str:
    """根据错误类型与信息给出可操作的修复建议。"""
    msg = (message or "").lower()
    if error_type in ("IndentationError", "TabError") or "indent" in msg:
        return "检查缩进是否统一使用 4 个空格，避免空格与 Tab 混用"
    if "unexpected eof" in msg or "eof" in msg:
        return "代码不完整，检查括号、冒号、引号是否闭合，或函数/类定义是否缺少体"
    if "invalid syntax" in msg:
        return "检查该行是否存在拼写错误、缺失冒号、运算符误用或关键字冲突"
    if "unmatched" in msg or "closing" in msg:
        return "检查括号/引号/三引号是否成对闭合"
    return "参考 Python 语法手册修正该行语法"


# ------------------------------------------------------------------
# 向后兼容适配器
# 保留原 TechnicalReviewAgent 调用的 async 类接口，委托给函数式核心。
# 后续 Agent 层迁移到统一函数式接口后可移除。
# ------------------------------------------------------------------
class CodeParserTool:
    def __init__(self):
        pass

    async def parse_code(self, code: str) -> List[Dict[str, Any]]:
        """委托给 code_parser，返回错误列表（兼容旧调用方）。"""
        result = code_parser({"code": code})
        if result.get("status") != "success":
            return []
        return result["data"].get("errors", [])
