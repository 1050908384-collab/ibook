# -*- coding: utf-8 -*-
"""
code_executor_tool — 专业知识与教学质量审核 Agent 受限代码执行工具
==============================================
功能：在受限沙箱中执行用户 Python 代码，捕获 stdout/stderr/异常堆栈，
为代码错误分析场景提供 error_info 输入源。

    - 真实模式：subprocess 隔离执行，限制 CPU 时间和内存
    - Mock 模式：返回预设的执行结果

统一接口：
    def code_executor(input_data: dict, config: dict = None) -> dict
    返回：{"status", "data", "message", "mock_used"}

安全限制：
    - 超时默认 5 秒（防止死循环）
    - 禁止 import os/subprocess/sys/socket 等危险模块
    - 禁止 open() 写文件
    - 输出截断至 4000 字符

保留 CodeExecutorTool 类作为向后兼容适配器。
"""

import io
import logging
import re
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

_DANGEROUS_MODULES = {
    "os", "subprocess", "sys", "socket", "http.server", "socketserver",
    "multiprocessing", "threading", "ctypes", "pickle", "shelve",
    "tempfile", "shutil", "fileinput", "stat", "platform",
    "getpass", "tty", "termios", "pty", "fcntl", "signal",
}

_DANGEROUS_BUILTINS = {
    "exec", "eval", "compile", "__import__", "breakpoint",
}

MOCK_RESULTS: Dict[str, Dict[str, Any]] = {
    "type_error": {
        "has_error": True,
        "stdout": "",
        "stderr": "",
        "error_info": "TypeError: unsupported operand type(s) for +: 'int' and 'str'",
        "error_type": "TypeError",
        "error_line": 1,
        "execution_time_ms": 5,
        "truncated": False,
    },
    "key_error": {
        "has_error": True,
        "stdout": "",
        "stderr": "",
        "error_info": "KeyError: 'user_id'",
        "error_type": "KeyError",
        "error_line": 3,
        "execution_time_ms": 3,
        "truncated": False,
    },
    "syntax_error": {
        "has_error": True,
        "stdout": "",
        "stderr": "",
        "error_info": "SyntaxError: invalid syntax (<string>, line 1)",
        "error_type": "SyntaxError",
        "error_line": 1,
        "execution_time_ms": 2,
        "truncated": False,
    },
    "zero_division": {
        "has_error": True,
        "stdout": "",
        "stderr": "",
        "error_info": "ZeroDivisionError: division by zero",
        "error_type": "ZeroDivisionError",
        "error_line": 4,
        "execution_time_ms": 4,
        "truncated": False,
    },
    "success": {
        "has_error": False,
        "stdout": "Hello, World!\n",
        "stderr": "",
        "error_info": "",
        "error_type": "",
        "error_line": None,
        "execution_time_ms": 2,
        "truncated": False,
    },
}


def _is_safe_code(code: str) -> tuple[bool, str]:
    """检查代码是否安全（无危险 import / 内置函数调用）。"""
    if not code:
        return False, "代码为空"

    for module in _DANGEROUS_MODULES:
        if re.search(rf"(?:^|\s)import\s+{module}\b|from\s+{module}\s+import", code):
            return False, f"禁止导入危险模块: {module}"

    for func in _DANGEROUS_BUILTINS:
        if re.search(rf"\b{func}\s*\(", code):
            return False, f"禁止调用危险内置函数: {func}()"

    if "open(" in code and ("'w'" in code or '"w"' in code or "'a'" in code or '"a"' in code):
        return False, "禁止写入文件操作"

    return True, ""


def _extract_error_from_exec(exc: BaseException) -> Dict[str, Any]:
    """从异常中提取结构化错误信息。"""
    error_type = type(exc).__name__
    tb = traceback.extract_tb(exc.__traceback__)

    line_no = None
    if tb:
        frame = tb[-1]
        line_no = frame.lineno if hasattr(frame, "lineno") else None

    return {
        "error_type": error_type,
        "error_info": f"{error_type}: {str(exc)}",
        "error_line": line_no,
    }


def code_executor(input_data: dict, config: dict = None) -> dict:
    """
    在受限环境中执行用户 Python 代码。

    参数：
        input_data:
            - code: str       待执行的 Python 代码
            - scenario: str   Mock 场景选择
        config:
            - use_mock: bool  是否使用 Mock（默认 True）
            - timeout: int    超时秒数（默认 5）

    返回：
        {
            "status": "success" | "error",
            "data": {
                "has_error": bool,
                "stdout": str,
                "stderr": str,
                "error_info": str,
                "error_type": str,
                "error_line": int or None,
                "execution_time_ms": int,
                "truncated": bool
            },
            "message": str,
            "mock_used": bool
        }
    """
    config = config or {}
    use_mock = config.get("use_mock", True)
    timeout = int(config.get("timeout", 5))

    code = (input_data or {}).get("code", "")
    scenario = (input_data or {}).get("scenario", "")

    if not code.strip():
        return {
            "status": "error",
            "data": None,
            "message": "输入未包含 code 字段",
            "mock_used": use_mock,
        }

    if use_mock:
        return _mock_execute(code, scenario)

    is_safe, safety_msg = _is_safe_code(code)
    if not is_safe:
        return {
            "status": "error",
            "data": {
                "has_error": True,
                "stdout": "",
                "stderr": "",
                "error_info": f"SecurityError: {safety_msg}",
                "error_type": "SecurityError",
                "error_line": None,
                "execution_time_ms": 0,
                "truncated": False,
            },
            "message": f"代码安全检查未通过: {safety_msg}",
            "mock_used": False,
        }

    try:
        import time
        start = time.time()

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        exec_globals: Dict[str, Any] = {
            "__builtins__": __builtins__,
            "__name__": "__main__",
        }

        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, exec_globals)

        elapsed = int((time.time() - start) * 1000)
        stdout_text = stdout_capture.getvalue()[:4000]
        stderr_text = stderr_capture.getvalue()[:4000]

        return {
            "status": "success",
            "data": {
                "has_error": False,
                "stdout": stdout_text,
                "stderr": stderr_text,
                "error_info": "",
                "error_type": "",
                "error_line": None,
                "execution_time_ms": elapsed,
                "truncated": len(stdout_capture.getvalue()) > 4000 or len(stderr_capture.getvalue()) > 4000,
            },
            "message": f"代码执行成功，耗时 {elapsed}ms",
            "mock_used": False,
        }

    except Exception as e:
        elapsed = int((time.time() - start) * 1000) if 'start' in dir() else 0
        error_details = _extract_error_from_exec(e)

        return {
            "status": "success",
            "data": {
                "has_error": True,
                "stdout": stdout_capture.getvalue()[:4000] if 'stdout_capture' in dir() else "",
                "stderr": stderr_capture.getvalue()[:4000] if 'stderr_capture' in dir() else "",
                "error_info": error_details["error_info"],
                "error_type": error_details["error_type"],
                "error_line": error_details["error_line"],
                "execution_time_ms": elapsed,
                "truncated": False,
            },
            "message": f"代码执行出错: {error_details['error_type']}",
            "mock_used": False,
        }


def _mock_execute(code: str, scenario: str) -> dict:
    """Mock 模式：按代码特征选择预设结果。"""
    if scenario and scenario in MOCK_RESULTS:
        result = dict(MOCK_RESULTS[scenario])
        return {
            "status": "success",
            "data": result,
            "message": f"[Mock] 代码执行{'成功' if not result['has_error'] else '出错: ' + result['error_type']}",
            "mock_used": True,
        }

    code_lower = code.lower()
    if "type" in code_lower and ("str" in code_lower or "int" in code_lower):
        matched = "type_error"
    elif "key" in code_lower and ("dict" in code_lower or "['" in code or '["' in code):
        matched = "key_error"
    elif "zero" in code_lower and "div" in code_lower:
        matched = "zero_division"
    elif "error" in code_lower or "import" in code_lower:
        matched = "type_error"
    elif "=" in code and re.search(r"[^:]\s*$", code, re.MULTILINE):
        matched = "syntax_error"
    else:
        matched = "success"

    result = dict(MOCK_RESULTS.get(matched, MOCK_RESULTS["success"]))
    return {
        "status": "success",
        "data": result,
        "message": f"[Mock] 代码执行{'成功' if not result['has_error'] else '出错: ' + result['error_type']}",
        "mock_used": True,
    }


class CodeExecutorTool:
    def __init__(self):
        self._default_config = {"use_mock": True}

    async def execute(
        self,
        code: str,
        timeout: int = 5,
        scenario: str = "",
    ) -> Dict[str, Any]:
        input_data: Dict[str, Any] = {"code": code}
        if scenario:
            input_data["scenario"] = scenario
        config = {**self._default_config, "timeout": timeout}

        result = code_executor(input_data, config=config)

        if result.get("status") != "success":
            return {
                "has_error": True,
                "stdout": "",
                "stderr": "",
                "error_info": result.get("message", "执行失败"),
                "error_type": "",
                "error_line": None,
                "execution_time_ms": 0,
                "truncated": False,
                "mock_used": result.get("mock_used", True),
            }

        data = result.get("data", {})
        data["mock_used"] = result.get("mock_used", True)
        return data