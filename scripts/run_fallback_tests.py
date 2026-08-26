# -*- coding: utf-8 -*-
"""12 组回退可靠性测试 —— 受控故障注入。

用法:
    # 全部 12 组故障注入测试
    python scripts/run_fallback_tests.py

    # 仅运行特定故障类
    python scripts/run_fallback_tests.py --types api_key_empty,http_429_timeout

故障类型（12 类）:
    F01  api_key_empty           — DEEPSEEK_API_KEY 为空或占位符
    F02  http_401_auth_fail      — 401 鉴权失败
    F03  http_403_denied         — 403 权限拒绝
    F04  http_404_invalid_model  — 404 模型/端点无效
    F05  http_429_rate_limit     — 429 限流
    F06  connect_timeout         — 连接超时
    F07  read_timeout            — 读取超时
    F08  http_500_server         — 500/502/503 服务端错误
    F09  empty_content_body      — 返回 200 但 content 为空（仅 reasoning_content）
    F10  ollama_model_unavailable — 本地首选模型不存在
    F11  ollama_down             — Ollama 服务未启动（端口不可达）
    F12  all_models_unavailable  — 云端+本地都不可用（期望失败+提示人工）

注意：
    - F01、F09、F10、F11、F12 通过临时环境变量 + monkey-patch 注入。
    - F02~F08（真实 HTTP 错误）通过 httpx MockTransport 注入，不访问外网。
    - 每条测试都会记录：fault_type、inject_method、actual_taken_model、
      retry_count、flow_final_status、state_continuity、evidence。
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(BACKEND_DIR))

DELIVERY_DIR = ROOT_DIR / "测试数据与评测材料"
FALLBACK_DIR = DELIVERY_DIR / "03_回退可靠性测试"

# 底座场景（4 个代表）
BASE_CASE_IDS = ["case_001", "case_018", "case_026", "case_040"]

# 从主评测 evaluation_cases.csv 加载场景输入
def load_case_input(case_id: str) -> Dict[str, str]:
    csv_path = DELIVERY_DIR / "02_50组个性化主评测" / "evaluation_cases.csv"
    import csv as _csv
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        for row in _csv.DictReader(f):
            if row["case_id"] == case_id:
                return row
    raise ValueError(f"Case not found: {case_id}")


def build_user_input(case: Dict[str, str]) -> str:
    parts = []
    for label, key in [
        ("我的学习背景", "background"),
        ("已掌握技能", "existing_skills"),
        ("我的学习目标", "learning_goal"),
        ("我每天能学", "daily_hours"),
        ("我喜欢怎么学", "learning_style"),
        ("我的硬件条件", "hardware"),
    ]:
        v = case.get(key, "").strip()
        if v and v != "未提供":
            suffix = "小时" if key == "daily_hours" else ""
            parts.append(f"{label}：{v}{suffix}")
    return "\n".join(parts)


# =====================================================================
# 故障注入层
# =====================================================================

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 故障配置：每个类型的注入方式 + 场景描述
FAULT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "api_key_empty": {
        "label": "API Key 缺失",
        "inject": "env_patch",
        "env_overrides": {"DEEPSEEK_API_KEY": ""},
        "expect_fallback_to_ollama": True,
        "expect_flow_failure": False,
    },
    "http_401_auth_fail": {
        "label": "401 鉴权失败",
        "inject": "mock_transport",
        "status_code": 401,
        "body": b'{"error":{"message":"Invalid API Key"}}',
        "expect_fallback_to_ollama": True,
        "expect_flow_failure": False,
    },
    "http_403_denied": {
        "label": "403 权限拒绝",
        "inject": "mock_transport",
        "status_code": 403,
        "body": b'{"error":{"message":"Account suspended"}}',
        "expect_fallback_to_ollama": True,
        "expect_flow_failure": False,
    },
    "http_404_invalid_model": {
        "label": "404 模型无效",
        "inject": "mock_transport",
        "status_code": 404,
        "body": b'{"error":{"message":"model not found"}}',
        "expect_fallback_to_ollama": True,
        "expect_flow_failure": False,
    },
    "http_429_rate_limit": {
        "label": "429 限流",
        "inject": "mock_transport",
        "status_code": 429,
        "body": b'{"error":{"message":"Rate limit exceeded"}}',
        "expect_fallback_to_ollama": True,
        "expect_flow_failure": False,
    },
    "connect_timeout": {
        "label": "连接超时",
        "inject": "mock_transport_timeout",
        "timeout_type": "connect",
        "expect_fallback_to_ollama": True,
        "expect_flow_failure": False,
    },
    "read_timeout": {
        "label": "读取超时",
        "inject": "mock_transport_timeout",
        "timeout_type": "read",
        "expect_fallback_to_ollama": True,
        "expect_flow_failure": False,
    },
    "http_500_server": {
        "label": "500 服务端错误",
        "inject": "mock_transport",
        "status_code": 500,
        "body": b'{"error":{"message":"Internal server error"}}',
        "expect_fallback_to_ollama": True,
        "expect_flow_failure": False,
    },
    "empty_content_body": {
        "label": "空正文（仅推理内容）",
        "inject": "mock_transport",
        "status_code": 200,
        "body": b'{"choices":[{"finish_reason":"length","message":{"role":"assistant","reasoning_content":"I think... step by step..."}}]}',
        "expect_fallback_to_ollama": True,
        "expect_flow_failure": False,
    },
    "ollama_model_unavailable": {
        "label": "Ollama 首选模型不存在",
        "inject": "env_patch",
        "env_overrides": {
            "LLM_PROVIDER": "ollama",
            "LLM_FALLBACK_PROVIDER": "ollama",
            "OLLAMA_MODEL": "nonexistent-model:999b",
            "OLLAMA_FALLBACK_MODELS": "qwen2.5:7b",
            "DEEPSEEK_API_KEY": "",
        },
        "expect_fallback_to_ollama": True,
        "expect_flow_failure": False,
    },
    "ollama_down": {
        "label": "Ollama 服务未启动",
        "inject": "env_patch",
        "env_overrides": {
            "LLM_PROVIDER": "ollama",
            "LLM_FALLBACK_PROVIDER": "",
            "OLLAMA_BASE_URL": "http://127.0.0.1:1",  # 故意错误端口
            "DEEPSEEK_API_KEY": "",
        },
        "expect_fallback_to_ollama": False,
        "expect_flow_failure": True,
    },
    "all_models_unavailable": {
        "label": "云端+本地全部不可用",
        "inject": "env_patch",
        "env_overrides": {
            "LLM_PROVIDER": "deepseek",
            "LLM_FALLBACK_PROVIDER": "ollama",
            "DEEPSEEK_API_KEY": "",
            "OLLAMA_BASE_URL": "http://127.0.0.1:1",
        },
        "expect_fallback_to_ollama": False,
        "expect_flow_failure": True,
    },
}


def apply_env_overrides(overrides: Dict[str, str]):
    """临时覆盖环境变量（影响 pydantic-settings 重新读取的字段）。"""
    for k, v in overrides.items():
        os.environ[k] = v


def restore_env(saved: Dict[str, str]):
    for k, old_v in saved.items():
        if old_v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = old_v


def refresh_settings_inplace() -> None:
    """把当前 os.environ 同步到 **现有** config.settings 实例。

    关键点：Python 的 `from config import settings` 会在导入时绑定对象引用。
    如果我们用 `config.settings = Settings(...)` 替换对象，那些已完成 `from` 导入
    的模块（harness / agents / llm.client）仍会指向旧对象。所以改为**原地修改
    现有 settings 对象的每个字段值**，保持引用不变，让所有模块同时看到新值。
    """
    import config as config_module
    from config import Settings, settings as global_settings

    # 保留 backend/.env 中冻结的本地模型链，同时允许本测试的 os.environ
    # 覆盖故障注入字段。
    fresh = Settings(_env_file=BACKEND_DIR / ".env")
    for field_name in fresh.model_fields:
        try:
            setattr(global_settings, field_name, getattr(fresh, field_name))
        except (AttributeError, ValueError):
            # 某些只读/计算字段跳过，不影响主体逻辑
            continue
    # 清理 pydantic 可能缓存的私有状态（保险起见）
    for attr in ("__pydantic_fields_set__", "__pydantic_private__"):
        if hasattr(global_settings, attr):
            try:
                delattr(global_settings, attr)
            except Exception:
                pass


_llm_patch_applied_in_fallback = False
_faulty_mock_transport: "FaultyMockTransport | None" = None


def apply_llm_patch_once() -> None:
    """和 run_real_evaluation.py 中同样的 LLM 调用拦截逻辑（只执行一次）。"""
    global _llm_patch_applied_in_fallback
    if _llm_patch_applied_in_fallback:
        return
    _llm_patch_applied_in_fallback = True
    # 复用 run_real_evaluation 中的 patch 函数，保证 _llm_call_records 被写入
    from scripts.run_real_evaluation import _patch_llm_client
    _patch_llm_client()


def apply_mock_transport_globally(mock_tp: "FaultyMockTransport") -> None:
    """在 LLMClient 类级别注入 mock transport：之后所有新建/现有实例的
    deepseek_client 请求都会走 mock 逻辑，不会漏网。

    - Monkey-patch `LLMClient._build_async_deepseek_client`：新 build 的 client
      直接用 mock transport。
    - 对已创建的 LLMClient 实例：如果已经有 deepseek_client，重建一个带 mock
      transport 的新 httpx.AsyncClient 替换掉旧的。
    """
    from llm import client as llm_client_module
    import httpx as _httpx

    OriginalBuilder = llm_client_module.LLMClient._build_async_deepseek_client

    def MockedBuilder(self, timeout, proxy_url):
        # 忽略 timeout / proxy_url，强制用 mock transport
        return _httpx.AsyncClient(
            timeout=timeout,
            transport=mock_tp,
            trust_env=False,
        )

    llm_client_module.LLMClient._build_async_deepseek_client = MockedBuilder

    # 兜底：遍历所有已创建的 LLMClient 实例（通过 gc 模块抓取），重建其 deepseek_client
    import gc
    for obj in gc.get_objects():
        # run_real_evaluation 会把 LLMClient 替换为 InstrumentedLLMClient，
        # 因此不能只按精确类名筛选，否则 Harness 已创建的客户端仍会访问云端。
        if isinstance(obj, llm_client_module.LLMClient):
            try:
                old_timeout = getattr(obj.deepseek_client, "timeout", None) or _httpx.Timeout(180.0)
                obj.deepseek_client = _httpx.AsyncClient(
                    timeout=old_timeout,
                    transport=mock_tp,
                    trust_env=False,
                )
            except Exception:
                pass


# =====================================================================
# MockTransport: 用于注入真实 HTTP 状态码/超时，不触网
# =====================================================================
try:
    import httpx
except ImportError:
    httpx = None


class FaultyMockTransport:
    """返回指定状态码或触发指定超时类型，不访问外网。"""

    def __init__(self, fault_type: str, status_code: int | None = None,
                 body: bytes | None = None, timeout_type: str | None = None):
        self.fault_type = fault_type
        self.status_code = status_code
        self.body = body
        self.timeout_type = timeout_type

    def handle_async_request(self, request):
        if self.timeout_type == "connect":
            raise httpx.ConnectTimeout("Mocked connect timeout")
        if self.timeout_type == "read":
            raise httpx.ReadTimeout("Mocked read timeout")
        # 返回预置 HTTP 响应
        return httpx.Response(
            status_code=self.status_code or 500,
            content=self.body or b"{}",
            request=request,
        )


# =====================================================================
# 单个故障运行
# =====================================================================
async def run_single_fault(fault_type: str, base_case_id: str) -> Dict[str, Any]:
    """对指定底座场景注入指定故障，返回结构化结果。"""
    from llm import client as llm_client_module

    config = FAULT_CONFIGS[fault_type]
    run_started = _now_iso()
    result: Dict[str, Any] = {
        "fault_id": f"{fault_type.upper()}-{base_case_id}",
        "fault_type": fault_type,
        "fault_label": config["label"],
        "base_case_id": base_case_id,
        "inject_method": config["inject"],
        "run_started_at": run_started,
        "run_finished_at": None,
        "flow_final_status": "running",
        "actual_taken_model": [],
        "retry_count": 0,
        "fallback_triggered": False,
        "state_continuity": False,
        "expect_flow_failure": config["expect_flow_failure"],
        "error_message": None,
        "evidence": {},
    }

    # 0. 确保 LLM 调用拦截已启用
    apply_llm_patch_once()
    # 必须在安装拦截器后导入 Harness。Harness 的 Agent 会在导入时绑定
    # LLMClient；反过来会导致故障测试实际运行却没有路由日志。
    from orchestrator.harness import Harness

    # 1. 保存环境并覆盖
    env_overrides = config.get("env_overrides", {}) or {}
    saved_env = {k: os.environ.get(k) for k in env_overrides}
    apply_env_overrides(env_overrides)
    # 把新环境变量同步到现有 config.settings（所有已导入的引用都能看到变化）
    refresh_settings_inplace()

    # 保存 LLMClient builder 原函数以便恢复
    orig_builder = getattr(llm_client_module.LLMClient, "_build_async_deepseek_client", None)

    try:
        harness = Harness()

        # 2. 对于 mock_transport 类故障：在 LLMClient 类级别全局注入 mock，
        #    无论是 Harness 已创建的 Agent 内部 llm_client，还是后续懒加载的
        #    grading/工具内新建的 llm_client，所有请求都会命中 mock。
        if config["inject"].startswith("mock_transport"):
            if httpx is None:
                raise RuntimeError("httpx 未安装，无法使用 mock_transport 注入")
            if config["inject"] == "mock_transport":
                mock_tp = FaultyMockTransport(
                    fault_type,
                    status_code=config.get("status_code"),
                    body=config.get("body"),
                )
            else:
                mock_tp = FaultyMockTransport(
                    fault_type,
                    timeout_type=config.get("timeout_type"),
                )
            apply_mock_transport_globally(mock_tp)

        # 记录本次运行前 LLM 调用索引位置（后面只截取属于本场景的调用，避免跨场景污染）
        from scripts.run_real_evaluation import _llm_call_records as lcr_global
        lcr_start_idx = len(lcr_global)

        # 3. 创建会话 + 输入
        session = harness.state_manager.create_session(None)
        session_id = session["session_id"]
        case_row = load_case_input(base_case_id)
        user_input = build_user_input(case_row)
        result["evidence"]["session_id"] = session_id
        result["evidence"]["user_input_hash"] = user_input[:80]

        # 4. 诊断 → 阶段启动（捕获异常或结果）
        diag = await harness.process_initial_input(session_id, user_input)
        if diag.get("status") == "ready":
            profile = diag.get("profile", {})
            stage_info = diag.get("stage_info", {})
            result["evidence"]["profile_confidence"] = profile.get("confidence", 0)
            result["evidence"]["planned_block"] = stage_info.get("block_name", "")

            try:
                stage_res = await harness.start_stage(session_id)
                result["evidence"]["review_passed"] = stage_res.get("review_passed")
                result["evidence"]["iterations"] = stage_res.get("iterations", 0)
                result["flow_final_status"] = "success"
            except Exception as exc:
                result["flow_final_status"] = "exception_during_start"
                result["error_message"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        else:
            result["flow_final_status"] = f"diagnostic_{diag.get('status')}"

        # 5. 从 LLM 调用历史（_llm_call_records）提取本场景专属记录
        try:
            current_calls = lcr_global[lcr_start_idx:]
            if current_calls:
                result["actual_taken_model"] = list(dict.fromkeys(
                    [c["actual_model"] for c in current_calls if c.get("actual_model")]
                ))
                result["retry_count"] = sum(c.get("retry_count", 0) for c in current_calls)
                result["fallback_triggered"] = any(c.get("fallback_triggered") for c in current_calls)
                result["evidence"]["llm_call_count"] = len(current_calls)
                result["evidence"]["failed_calls"] = sum(
                    1 for c in current_calls if not c.get("success")
                )
                # 详细证据：保存每条调用的 request→actual 模型路由，便于人工审查
                result["evidence"]["llm_route_samples"] = [
                    {
                        "agent": c.get("agent_name"),
                        "task": c.get("task_type"),
                        "request": c.get("request_model"),
                        "actual": c.get("actual_model"),
                        "success": c.get("success"),
                        "fallback": c.get("fallback_triggered"),
                        "err": c.get("error_type") or "",
                    }
                    for c in current_calls[:20]
                ]
                # 保存完整、无提示词的调用证据，供回退测试人工复核。
                result["evidence"]["llm_calls"] = current_calls
                successful_models = list(dict.fromkeys(
                    [c.get("actual_model") for c in current_calls
                     if c.get("success") and c.get("actual_model")]
                ))
                result["evidence"]["successful_models"] = successful_models
                # Deterministic content may still let the harness finish after
                # all model calls fail. That is not a model-flow success.
                if result["expect_flow_failure"] and not successful_models:
                    result["flow_final_status"] = "model_unavailable"
                    result["error_message"] = (
                        "所有配置的模型调用均失败；已禁止将确定性兜底结果视为成功，"
                        "需要人工处理或恢复模型服务。"
                    )
        except Exception as exc:
            result["evidence"]["lcr_error"] = f"{type(exc).__name__}:{str(exc)[:200]}"

        # 6. 连续性检查：fallback 模式下 profile 是否仍保存、阶段是否仍生成
        evidence = result.get("evidence", {})
        result["state_continuity"] = bool(
            evidence.get("profile_confidence", 0) > 0
            and evidence.get("planned_block")
            and evidence.get("review_passed") is True
            and result.get("flow_final_status") == "success"
        )

    except Exception as exc:
        result["flow_final_status"] = "crashed"
        result["error_message"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        result["evidence"]["traceback"] = traceback.format_exc()[:1000]
    finally:
        # 恢复原 builder（如有保存），避免 mock 污染后续故障场景
        if orig_builder is not None:
            try:
                llm_client_module.LLMClient._build_async_deepseek_client = orig_builder
            except Exception:
                pass
        restore_env(saved_env)
        # 恢复后再刷新一次 settings，确保回到正常状态
        refresh_settings_inplace()

    result["run_finished_at"] = _now_iso()
    # 根据期望标注匹配结果
    passed = False
    if result["expect_flow_failure"]:
        # 本地模型不可用时必须明确失败，不能用确定性内容伪造成功。
        passed = result["flow_final_status"] not in {"success", "diagnostic_ready"}
    else:
        actual_models = result.get("actual_taken_model") or []
        used_ollama = any(
            str(model).lower() == "ollama"
            or "ollama" in str(model).lower()
            or "deepseek-r1" in str(model).lower()
            or "qwen" in str(model).lower()
            for model in actual_models
        )
        passed = (
            result["flow_final_status"] == "success"
            and result["fallback_triggered"]
            and used_ollama
            and result["state_continuity"]
        )
    result["passed"] = passed
    return result


# =====================================================================
# 主流程
# =====================================================================
async def main():
    parser = argparse.ArgumentParser(description="12 组回退可靠性测试（受控故障注入）")
    parser.add_argument("--types", default="all",
                        help="逗号分隔的故障类型，如 api_key_empty,http_429_rate_limit")
    parser.add_argument("--base-cases", default=",".join(BASE_CASE_IDS),
                        help="逗号分隔的底座场景 ID")
    parser.add_argument("--dry", action="store_true",
                        help="仅打印将运行的组合，不执行")
    args = parser.parse_args()

    FALLBACK_DIR.mkdir(parents=True, exist_ok=True)

    # 故障类型列表
    if args.types == "all":
        fault_types = list(FAULT_CONFIGS.keys())
    else:
        fault_types = [t.strip() for t in args.types.split(",") if t.strip()]

    base_cases = [x.strip() for x in args.base_cases.split(",") if x.strip()]

    # 运行组合：故障 × 底座（原则 12 类 × 至少 1 底座 = 12；按需可多底座取代表）
    combos: List[tuple] = []
    for ft in fault_types:
        # 按《测试数据建议方法》：每类故障取 1 个底座即可，保证 12 条记录
        combos.append((ft, base_cases[0]))

    if args.dry:
        print("=== 将要运行的组合 ===")
        for ft, bc in combos:
            print(f"  {ft:<24} -> {bc}  {FAULT_CONFIGS[ft]['label']}")
        print(f"总计 {len(combos)} 条")
        return 0

    records = []
    for i, (ft, bc) in enumerate(combos, 1):
        print(f"[{i}/{len(combos)}] {ft} ({FAULT_CONFIGS[ft]['label']}) on {bc} ...")
        try:
            rec = await run_single_fault(ft, bc)
        except Exception as exc:
            rec = {
                "fault_id": f"{ft.upper()}-{bc}",
                "fault_type": ft,
                "fault_label": FAULT_CONFIGS[ft]["label"],
                "base_case_id": bc,
                "flow_final_status": "outer_crashed",
                "error_message": f"{type(exc).__name__}: {str(exc)[:500]}",
                "passed": False,
            }
        print(f"  -> {rec.get('flow_final_status')} passed={rec.get('passed')} "
              f"fallback={rec.get('fallback_triggered')}")
        records.append(rec)

    # 筛选运行只更新指定故障，保留已有其他真实证据；完整运行仍覆盖重建。
    if args.types != "all":
        existing_by_id = {}
        csv_path_existing = FALLBACK_DIR / "fallback_cases.csv"
        if csv_path_existing.exists():
            try:
                with open(csv_path_existing, "r", encoding="utf-8-sig", newline="") as f:
                    for row in csv.DictReader(f):
                        if row.get("fault_id"):
                            existing_by_id[row["fault_id"]] = row
            except Exception:
                pass
        jsonl_path_existing = FALLBACK_DIR / "fallback_traces.jsonl"
        if jsonl_path_existing.exists():
            try:
                with open(jsonl_path_existing, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            item = json.loads(line)
                            if item.get("fault_id"):
                                existing_by_id[item["fault_id"]] = item
            except Exception:
                pass
        for old in existing_by_id.values():
            if not any(r.get("fault_id") == old.get("fault_id") for r in records):
                records.append(old)

    # 保存 fallback_cases.csv
    csv_path = FALLBACK_DIR / "fallback_cases.csv"
    fieldnames = [
        "fault_id", "fault_type", "fault_label", "base_case_id",
        "inject_method", "flow_final_status", "expect_flow_failure",
        "actual_taken_model", "retry_count", "fallback_triggered",
        "state_continuity", "error_message",
        "run_started_at", "run_finished_at", "passed",
    ]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            row = dict(r)
            row["actual_taken_model"] = "|".join(r.get("actual_taken_model") or [])
            writer.writerow(row)

    # 保存 jsonl 证据
    jsonl_path = FALLBACK_DIR / "fallback_traces.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    # 打印摘要
    total = len(records)
    passed = sum(1 for r in records if r.get("passed"))
    triggered_fb = sum(1 for r in records if r.get("fallback_triggered"))
    failed_status = sum(1 for r in records
                        if r.get("expect_flow_failure") and
                        r.get("flow_final_status") in {"crashed", "exception_during_start"})

    print()
    print("=" * 60)
    print(f"回退可靠性测试：{passed}/{total} 通过")
    print(f"触发回退：{triggered_fb}/{total}")
    print(f"期望失败且确实失败：{failed_status}")
    print(f"结果文件：{csv_path}")
    print(f"证据文件：{jsonl_path}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
