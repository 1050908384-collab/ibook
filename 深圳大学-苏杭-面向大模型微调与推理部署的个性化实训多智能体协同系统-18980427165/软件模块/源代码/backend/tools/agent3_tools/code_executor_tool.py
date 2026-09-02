# -*- coding: utf-8 -*-
"""
code_executor_tool — 个性化学习资源生成 Agent 代码执行工具
================================================
功能：在本地执行用户提供的Python代码，捕获执行结果、报错信息或运行输出。
    - 真实模式：使用 subprocess 执行代码，捕获 stdout/stderr 和返回码
    - Mock 模式：返回 ./data/test_data/mock_code_executor_tool.json 预设执行结果

统一接口：
    def code_executor(input_data: dict, config: dict = None) -> dict
    返回：{"status", "data", "message", "mock_used"}

安全说明：
    - 代码在独立进程中运行，超时自动终止
    - 不启用沙箱（用户配置要求高），但限制执行时间
    - 建议仅在用户明确授权时调用

保留 CodeExecutorTool 类作为向后兼容适配器。
"""

import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

from config import DATA_DIR

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

MOCK_FILE = DATA_DIR / "test_data" / "mock_code_executor_tool.json"


def code_executor(input_data: dict, config: dict = None) -> dict:
    config = config or {}
    use_mock = config.get("use_mock", True)

    code = input_data.get("code", "")
    timeout = input_data.get("timeout", 10)

    if not code.strip():
        return {
            "status": "error",
            "data": None,
            "message": "输入未包含 code 字段或代码为空",
            "mock_used": use_mock,
        }

    try:
        if use_mock:
            return _mock_execute(input_data)

        result = _execute_code(code, timeout)
        return {
            "status": "success",
            "data": result,
            "message": _build_message(result),
            "mock_used": False,
        }
    except Exception as e:
        logger.exception("code_executor 执行失败")
        return {
            "status": "success",
            "data": {
                "output": "",
                "error": f"代码执行异常: {str(e)}",
                "return_code": -1,
                "timeout": False,
            },
            "message": f"代码执行异常: {e}",
            "mock_used": False,
        }


def _execute_code(code: str, timeout: int) -> dict:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        temp_file = f.name

    try:
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )

        return {
            "output": result.stdout,
            "error": result.stderr,
            "return_code": result.returncode,
            "timeout": False,
        }
    except subprocess.TimeoutExpired:
        return {
            "output": "",
            "error": f"代码执行超时（超过 {timeout} 秒）",
            "return_code": -2,
            "timeout": True,
        }
    finally:
        Path(temp_file).unlink(missing_ok=True)


def _build_message(result: dict) -> str:
    if result.get("timeout"):
        return "代码执行超时"
    if result.get("return_code") == 0:
        output = result.get("output", "")
        return f"代码执行成功，输出: {output[:50]}..." if output else "代码执行成功，无输出"
    return f"代码执行失败，返回码: {result.get('return_code')}"


def _mock_execute(input_data: dict) -> dict:
    scenario = input_data.get("scenario") or "normal"
    scenarios = _load_mock()
    
    if not scenarios:
        data = {"output": "代码执行成功（Mock模式）", "error": "", "return_code": 0, "timeout": False}
    else:
        if scenario not in scenarios:
            scenario = "normal"
        data = dict(scenarios[scenario])
        data.setdefault("output", "")
        data.setdefault("error", "")
        data.setdefault("return_code", 0)
        data.setdefault("timeout", False)

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


class CodeExecutorTool:
    def __init__(self):
        self._default_config = {"use_mock": True}

    async def execute_code(self, code: str, timeout: int = 10) -> Dict[str, Any]:
        result = code_executor({
            "code": code,
            "timeout": timeout,
        }, config=self._default_config)
        if result.get("status") != "success":
            return {
                "output": "",
                "error": result.get("message", "执行失败"),
                "return_code": -1,
                "timeout": False,
            }
        return result["data"]