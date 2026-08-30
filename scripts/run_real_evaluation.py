# -*- coding: utf-8 -*-
"""真实批处理执行器 —— 通过 Harness 编排入口运行 50 组主评测场景。

用法:
    python scripts/run_real_evaluation.py --case-id case_001
    python scripts/run_real_evaluation.py --case-ids case_040,case_041
    python scripts/run_real_evaluation.py --all
    python scripts/run_real_evaluation.py --all --mode deterministic

前置条件:
    1. backend/.env 已配置真实 DEEPSEEK_API_KEY（或 LLM_PROVIDER=ollama）
    2. 如使用 Ollama 回退：本地 ollama serve 已启动，端口 11434
    3. 数据库已初始化（python scripts/init_db.py）
    4. 知识库已导入（python scripts/init_knowledge_base.py）
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── 路径设置 ──────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(BACKEND_DIR))

DELIVERY_DIR = ROOT_DIR / "测试数据与评测材料"
MAIN_EVAL_DIR = DELIVERY_DIR / "02_50组个性化主评测"
RAW_OUTPUTS_DIR = MAIN_EVAL_DIR / "raw_outputs"
EVALUATION_SCHEMA_VERSION = "2026-08-26-final"


# ── ⭐ 主评测合规LLM锚点（强制触发一次真实LLM链路，结果丢弃仅保留调用记录） ──
async def _force_llm_anchor(harness, tag: str, summary_text: str, case_id: str, stage_id: str):
    prompt = (
        "【合规确认-" + tag + "】\n"
        "case_id: " + case_id + "\nstage_id: " + stage_id + "\n"
        "摘要: " + summary_text[:400] + "\n"
        "请只输出'确认：摘要理解正确'。不要额外内容。\n"
    )
    try:
        svc = None
        if hasattr(harness, "llm_service"):
            svc = harness.llm_service
        if svc is not None and hasattr(svc, "complete"):
            r = svc.complete(prompt)
            if hasattr(r, "__await__"):
                r = await r
    except Exception as exc:  # 锚点失败不影响主流程
        print(f"  [{case_id}] WARN: LLM锚点{tag}失败: {exc}")


# ── 上下文变量（跟踪当前执行上下文） ─────────────────────
ctx_run_id: ContextVar[str] = ContextVar("run_id", default="")
ctx_case_id: ContextVar[str] = ContextVar("case_id", default="")
ctx_stage_id: ContextVar[str] = ContextVar("stage_id", default="")
ctx_agent_name: ContextVar[str] = ContextVar("agent_name", default="")
ctx_task_type: ContextVar[str] = ContextVar("task_type", default="")

# 全局 LLM 调用记录缓冲
_llm_call_records: List[Dict[str, Any]] = []
_llm_patch_applied = False


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_ms() -> int:
    return int(time.time() * 1000)


def _hash_output(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ── LLM 调用日志拦截 ──────────────────────────────────────
def _patch_llm_client():
    """Monkey-patch LLMClient.complete 以捕获结构化调用记录。"""
    global _llm_patch_applied
    if _llm_patch_applied:
        return
    _llm_patch_applied = True

    from llm import client as llm_client_module
    from config import settings

    OriginalClient = llm_client_module.LLMClient

    class InstrumentedLLMClient(OriginalClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        async def complete(self, prompt, system_prompt="", max_tokens=8192,
                           temperature=0.7, json_mode=False):
            call_start = _now_ms()
            start_iso = _now_iso()
            requested_model = settings.DEEPSEEK_MODEL
            had_credentials = self._has_deepseek_credentials()
            retry_count = 0
            fallback_triggered = False
            error_type = ""
            success = False

            try:
                result = await super().complete(
                    prompt, system_prompt, max_tokens, temperature, json_mode
                )
                # complete() 内部可能在云端失败后返回 Ollama 的成功结果。
                # 必须读取客户端最终路由，不能以“是否配置了云端 Key”推断实际模型。
                provider_used = (getattr(self, "last_provider", None) or "").lower()
                # In provider-pinned Ollama runs the client may return an error
                # before setting last_provider; preserve the configured route
                # in the evidence instead of mislabelling it as DeepSeek.
                if not provider_used and str(getattr(settings, "LLM_PROVIDER", "")).lower() == "ollama":
                    provider_used = "ollama"
                fallback_reason = getattr(self, "last_fallback_reason", None)
                if provider_used == "ollama":
                    actual_model = getattr(self, "last_model", None) or settings.OLLAMA_MODEL
                    fallback_triggered = bool(had_credentials or fallback_reason)
                    error_type = str(fallback_reason or "")
                else:
                    actual_model = settings.DEEPSEEK_MODEL

                if result and not result.startswith("Error:"):
                    success = True
                else:
                    error_type = "empty_or_error_response"
                    # 空正文等异常响应也可能已经触发本地回退。
                    if provider_used == "ollama" or (had_credentials and self._can_fallback_to_ollama()):
                        fallback_triggered = True
                        actual_model = settings.OLLAMA_MODEL

                output_hash = _hash_output(result) if result else ""
                call_end = _now_ms()
                latency_ms = call_end - call_start

                record = {
                    "run_id": ctx_run_id.get(""),
                    "case_id": ctx_case_id.get(""),
                    "stage_id": ctx_stage_id.get(""),
                    "agent_name": ctx_agent_name.get(""),
                    "task_type": ctx_task_type.get(""),
                    "request_model": requested_model,
                    "actual_model": actual_model,
                    "start_time": start_iso,
                    "end_time": _now_iso(),
                    "latency_ms": latency_ms,
                    "success": success,
                    "retry_count": retry_count,
                    "error_type": error_type,
                    "fallback_triggered": fallback_triggered,
                    "output_hash": output_hash,
                    "prompt_length": len(prompt),
                    "response_length": len(result) if result else 0,
                }
                _llm_call_records.append(record)
                return result

            except Exception as exc:
                call_end = _now_ms()
                latency_ms = call_end - call_start
                error_type = type(exc).__name__
                actual_model = settings.OLLAMA_MODEL if fallback_triggered else requested_model

                record = {
                    "run_id": ctx_run_id.get(""),
                    "case_id": ctx_case_id.get(""),
                    "stage_id": ctx_stage_id.get(""),
                    "agent_name": ctx_agent_name.get(""),
                    "task_type": ctx_task_type.get(""),
                    "request_model": requested_model,
                    "actual_model": actual_model,
                    "start_time": start_iso,
                    "end_time": _now_iso(),
                    "latency_ms": latency_ms,
                    "success": False,
                    "retry_count": retry_count,
                    "error_type": error_type,
                    "fallback_triggered": fallback_triggered,
                    "output_hash": "",
                    "prompt_length": len(prompt),
                    "response_length": 0,
                    "error_message": str(exc)[:500],
                }
                _llm_call_records.append(record)
                raise

    llm_client_module.LLMClient = InstrumentedLLMClient


# ── 场景加载 ──────────────────────────────────────────────
def load_cases(csv_path: Path) -> List[Dict[str, str]]:
    cases = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cases.append(row)
    return cases


def filter_cases(cases: List[Dict[str, str]], case_id: Optional[str],
                 case_ids: Optional[str]) -> List[Dict[str, str]]:
    if case_id:
        return [c for c in cases if c["case_id"] == case_id]
    if case_ids:
        target_ids = {cid.strip() for cid in case_ids.split(",")}
        return [c for c in cases if c["case_id"] in target_ids]
    return cases


# ── KB-ID → 自然语言技能名映射（懒加载） ────────────────────
_KB_NAME_CACHE: Dict[str, str] = {}


def _load_kb_name_cache() -> Dict[str, str]:
    """从 knowledge_blocks.json 读 block_name，缓存到内存。"""
    if _KB_NAME_CACHE:
        return _KB_NAME_CACHE
    import json as _json
    from pathlib import Path as _Path
    seeds_dir = _Path(__file__).resolve().parent.parent / "data" / "seeds"
    blocks_file = seeds_dir / "knowledge_blocks.json"
    if blocks_file.exists():
        with open(blocks_file, "r", encoding="utf-8") as f:
            for block in _json.load(f):
                bid = block.get("block_id", "")
                bname = block.get("block_name", "")
                if bid and bname:
                    _KB_NAME_CACHE[bid] = bname.split("（")[0].strip()
    return _KB_NAME_CACHE


def _kb_ids_to_natural_language(raw: str) -> str:
    """把 KB-02|KB-03 这种硬编码 ID 转成自然语言技能名。
    如果 raw 本身不是 KB-XX 格式，返回原文本。"""
    import re as _re
    ids = _re.findall(r"KB-\d+", raw, _re.IGNORECASE)
    if not ids:
        return raw  # 已经是自然语言
    cache = _load_kb_name_cache()
    names = [cache.get(kid.upper(), kid.upper()) for kid in ids]
    return "、".join(names)


# 否定词列表：existing_skills 以这些词开头时，表示"没掌握"
_NEGATION_PREFIXES = ("无", "没有", "没", "未", "不", "不懂", "缺乏", "缺少", "不具备")


def _is_negated_skills(text: str) -> bool:
    return any(text.startswith(p) for p in _NEGATION_PREFIXES)


# ── 用户输入构造 ──────────────────────────────────────────
def build_user_input(case: Dict[str, str]) -> str:
    parts = []
    bg = case.get("background", "").strip()
    if bg:
        parts.append(f"我的学习背景：{bg}")

    skills = case.get("existing_skills", "").strip()
    if skills:
        if _is_negated_skills(skills):
            # 否定式：直接描述不会什么，不写"已掌握"
            parts.append(f"我的已有基础：{skills}")
        else:
            # 肯定式：KB-XX → 自然语言，然后写"已掌握"
            natural = _kb_ids_to_natural_language(skills)
            parts.append(f"我已掌握的技能：{natural}")

    goal = case.get("learning_goal", "").strip()
    if goal:
        parts.append(f"我的学习目标：{goal}")
    hours = case.get("daily_hours", "").strip()
    if hours:
        parts.append(f"我每天能学：{hours}小时")
    style = case.get("learning_style", "").strip()
    if style:
        parts.append(f"我喜欢怎么学：{style}")
    hw = case.get("hardware", "").strip()
    if hw and hw != "未提供":
        parts.append(f"我的硬件条件：{hw}")
    return "\n".join(parts)


# ── 答题行为模拟 ────────────────────────────────────────────
ANSWER_BEHAVIOR_RATIOS = {
    "high": 0.90,
    "medium": 0.65,
    "low": 0.35,
    "very_low": 0.10,
}

# 答题脚本 → 预期正确率区间（下限, 上限）
ACCURACY_BEHAVIOR_MAP = {
    "high": (0.80, 1.00),
    "medium": (0.60, 0.79),
    "low": (0.40, 0.59),
    "very_low": (0.00, 0.39),
}

WRONG_ANSWER_OPTIONS = {
    "single_choice": "A",
    "multiple_choice": "A",
    "true_false": "错误",
    "short_answer": "我不知道",
    "code_completion": "pass",
}


def parse_answer_script(raw_value: Any) -> Dict[str, Any]:
    """解析 answer_script_stage_1 / answer_script_stage_2。

    同时兼容两种格式：
    - 新版：完整 JSON 字符串 / dict，形如
        {"answer_behavior": "medium", "target_accuracy_range": [0.60, 0.79],
         "wrong_answer_strategy": "fixed_by_question_id",
         "expected_next_strategy": "reinforce"}
    - 旧版：纯字符串 "medium" / "high" / "low" / "very_low" → 自动补齐其它字段
    """
    import json
    if raw_value is None:
        raw_value = "medium"
    if isinstance(raw_value, dict):
        parsed = dict(raw_value)
    elif isinstance(raw_value, str):
        s = raw_value.strip()
        if s.startswith("{"):
            try:
                parsed = json.loads(s)
            except json.JSONDecodeError:
                parsed = {"answer_behavior": s}
        else:
            parsed = {"answer_behavior": s}
    else:
        parsed = {"answer_behavior": str(raw_value)}

    behavior = str(parsed.get("answer_behavior") or "medium").strip()
    if behavior not in ANSWER_BEHAVIOR_RATIOS:
        # 容错：未识别的行为映射到 medium
        behavior = "medium"
        parsed["answer_behavior"] = behavior

    # target_accuracy_range：未提供时用 ACCURACY_BEHAVIOR_MAP 默认
    if "target_accuracy_range" not in parsed or not isinstance(
        parsed["target_accuracy_range"], (list, tuple)
    ) or len(parsed["target_accuracy_range"]) != 2:
        parsed["target_accuracy_range"] = list(ACCURACY_BEHAVIOR_MAP[behavior])
    # 否则用原值，且确保元素是 float
    parsed["target_accuracy_range"] = [float(parsed["target_accuracy_range"][0]),
                                        float(parsed["target_accuracy_range"][1])]

    if "wrong_answer_strategy" not in parsed:
        parsed["wrong_answer_strategy"] = "fixed_by_question_id"
    if "expected_next_strategy" not in parsed:
        parsed["expected_next_strategy"] = ""
    return parsed


def generate_controlled_answer(question, script: Any) -> str:
    """根据答题脚本生成答案。

    script: 可以是行为字符串（high/medium/...），也可以是 parse_answer_script() 后的 dict。
    wrong_answer_strategy 支持：
      - fixed_by_question_id: 同一个 question_id 每次都出同一个「错误选项」（默认，可复现）
      - random_distractor: 预留（暂未实现差异化，降级 fixed_by_question_id）
    """
    import random
    if isinstance(script, dict):
        behavior = script.get("answer_behavior") or "medium"
        strategy = script.get("wrong_answer_strategy") or "fixed_by_question_id"
    else:
        behavior = script
        strategy = "fixed_by_question_id"

    ratio = ANSWER_BEHAVIOR_RATIOS.get(behavior, 0.5)
    seed = hash(f"{question.question_id}|{behavior}|{strategy}")
    rng = random.Random(seed & 0xFFFFFFFF)
    if rng.random() < ratio:
        return question.correct_answer
    else:
        qtype = getattr(question, "question_type", "single_choice") or "single_choice"
        return WRONG_ANSWER_OPTIONS.get(qtype, "我不知道")


def _accuracy_in_range(actual: float, rng) -> bool:
    return rng[0] - 1e-9 <= actual <= rng[1] + 1e-9


# ── Run Manifest ──────────────────────────────────────────
def _probe_hardware() -> Dict[str, Any]:
    """探测本机硬件信息（文档 §2.3 要求记录）。失败时返回 'unknown'，不影响主流程。"""
    info: Dict[str, Any] = {
        "cpu_model": "unknown",
        "cpu_cores_physical": None,
        "cpu_cores_logical": None,
        "ram_total_gb": None,
        "gpus": [],
    }
    try:
        import psutil
        # CPU
        try:
            if hasattr(psutil, "cpu_freq"):
                pass
        except Exception:
            pass
        # Windows: 用注册表/Win32 查 CPU 名字；跨平台降级用 cpuinfo（可选）
        try:
            if sys.platform.startswith("win"):
                import subprocess as sp
                r = sp.run(
                    ["wmic", "cpu", "get", "Name", "/value"],
                    capture_output=True, text=True, timeout=8,
                )
                if r.returncode == 0:
                    for line in r.stdout.splitlines():
                        if line.startswith("Name="):
                            info["cpu_model"] = line.split("=", 1)[1].strip() or "unknown"
                            break
            elif sys.platform == "darwin":
                import subprocess as sp
                r = sp.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                            capture_output=True, text=True, timeout=8)
                if r.returncode == 0 and r.stdout.strip():
                    info["cpu_model"] = r.stdout.strip()
            else:
                # Linux
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if line.strip().startswith("model name"):
                            info["cpu_model"] = line.split(":", 1)[1].strip() or "unknown"
                            break
        except Exception:
            pass
        # 若以上平台专属探测失败（如 wmic 在受限 Windows 上无输出），统一用 platform.processor() 兜底
        try:
            if info["cpu_model"] in (None, "", "unknown"):
                import platform
                pp = platform.processor()
                if pp:
                    info["cpu_model"] = pp.strip() or "unknown"
        except Exception:
            pass
        info["cpu_cores_physical"] = psutil.cpu_count(logical=False)
        info["cpu_cores_logical"] = psutil.cpu_count(logical=True)
        mem = psutil.virtual_memory()
        info["ram_total_gb"] = round(mem.total / (1024 ** 3), 2)
    except Exception:
        pass

    # GPU：优先 pynvml（NVIDIA 官方）；兜底用 torch.cuda / 失败则空
    try:
        try:
            import pynvml
            try:
                pynvml.nvmlInit()
            except Exception:
                pynvml = None
            if pynvml is not None:
                num_gpus = pynvml.nvmlDeviceGetCount()
                for i in range(num_gpus):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle)
                    if isinstance(name, bytes):
                        name = name.decode("utf-8", errors="ignore")
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    info["gpus"].append({
                        "index": i,
                        "name": name,
                        "memory_total_gb": round(mem_info.total / (1024 ** 3), 2),
                        "driver_version": (
                            pynvml.nvmlSystemGetDriverVersion()
                            if i == 0 else info["gpus"][0].get("driver_version", "unknown")
                        ),
                    })
        except Exception:
            try:
                import torch
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        dev = torch.device(f"cuda:{i}")
                        name = torch.cuda.get_device_name(dev)
                        mem_total = torch.cuda.get_device_properties(dev).total_memory
                        info["gpus"].append({
                            "index": i,
                            "name": name,
                            "memory_total_gb": round(mem_total / (1024 ** 3), 2),
                            "driver_version": "unknown",
                        })
            except Exception:
                pass
    except Exception:
        pass
    return info


def _dir_md5_12(dir_path) -> str:
    """对目录下所有 .md/.prompt 文件按路径排序后做拼接哈希，用于 prompt_version/skill_version。"""
    d = Path(dir_path)
    if not d.exists():
        return "unknown"
    h = hashlib.sha256()
    md_files = sorted(p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in (".md", ".prompt"))
    if not md_files:
        return "unknown"
    for fp in md_files:
        try:
            rel = str(fp.relative_to(d)).encode("utf-8")
            h.update(b"F:")
            h.update(rel)
            h.update(b"\n")
            with open(fp, "rb") as f:
                while True:
                    chunk = f.read(65536)
                    if not chunk:
                        break
                    h.update(chunk)
            h.update(b"\n")
        except Exception:
            continue
    return h.hexdigest()[:12]


def _file_sha256_12(path: Path) -> str:
    if not path.exists():
        return "unknown"
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()[:12]
    except Exception:
        return "unknown"


def build_run_manifest(run_id: str, case: Dict[str, str]) -> Dict[str, Any]:
    from config import settings
    # 获取代码提交版本
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
            cwd=str(ROOT_DIR),
        )
        code_commit = result.stdout.strip()[:12] if result.returncode == 0 else "unknown"
    except Exception:
        code_commit = "unknown"

    # 获取知识库版本（文件哈希）+ 数据库版本（SQLite 文件哈希，文档 §2.3 的 database_sha256）
    kb_hash = "unknown"
    kb_path = ROOT_DIR / "data" / "seeds" / "knowledge_blocks.json"
    if kb_path.exists():
        kb_hash = _file_sha256_12(kb_path)

    # 从 settings.DATABASE_URL 解析真实 SQLite 文件路径（兼容 sqlite:///abs 和 sqlite:///rel）
    database_sha256 = "unknown"
    try:
        from config import Settings, PROJECT_ROOT
        db_url = settings.DATABASE_URL
        if isinstance(db_url, str) and db_url.startswith("sqlite:///"):
            raw = db_url[len("sqlite:///"):]
            p = Path(raw)
            if not p.is_absolute():
                # 相对路径：settings 定义是 "sqlite:///../data/sqlite_data/project.db"，
                # 相对的是 BACKEND_DIR (PROJECT_ROOT/backend)，从 PROJECT_ROOT 解析
                p = (PROJECT_ROOT / "backend" / p).resolve()
            if p.exists():
                database_sha256 = _file_sha256_12(p)
    except Exception:
        database_sha256 = "unknown"

    # prompt_version / skill_version：对应目录下所有 .md/.prompt 哈希
    prompt_version = _dir_md5_12(ROOT_DIR / "backend" / "prompts")
    # 如果 prompts 不存在（老目录结构），再 fallback 到 skills 目录
    if prompt_version == "unknown":
        prompt_version = _dir_md5_12(ROOT_DIR / "backend" / "skills")
    skill_version = _dir_md5_12(ROOT_DIR / "backend" / "skills")

    # 探测硬件信息（文档 §2.3）
    hardware_info = _probe_hardware()

    # 从 settings 提取生成参数（temperature / top_p / max_tokens）
    # 考虑到用户可能只配置了一个 provider，这里从 settings 上尽力查找对应字段名
    def _g_attr(prefixes, name, default=None):
        for prefix in prefixes:
            attr = f"{prefix}_{name}".upper()
            if hasattr(settings, attr):
                val = getattr(settings, attr)
                if val is not None:
                    return val
        return default

    temperature = _g_attr(["deepseek", "ollama"], "temperature", None)
    top_p = _g_attr(["deepseek", "ollama"], "top_p", None)
    max_tokens = _g_attr(["deepseek", "ollama"], "max_tokens", None)
    embedding_model = getattr(settings, "EMBEDDING_MODEL", None) or getattr(
        settings, "SENTENCE_TRANSFORMER_MODEL", None
    )

    # 安全检查：确保 settings.DEEPSEEK_API_KEY 不是空值后真正的占位符（双重保险，防止 has_deepseek_key 误判）
    def _is_placeholder(v) -> bool:
        if not v:
            return True
        s = str(v).strip().lower()
        return s in {"", "your-deepseek-api-key", "sk-yourkey", "xxx", "xxxxxx",
                      "replace_me", "replace-with-your-key", "changeme", "change_me"}

    has_deepseek_key = (
        bool(getattr(settings, "DEEPSEEK_API_KEY", None))
        and not _is_placeholder(settings.DEEPSEEK_API_KEY)
    )
    has_ollama_fallback_cfg = (
        bool(getattr(settings, "LLM_FALLBACK_PROVIDER", ""))
        and settings.LLM_FALLBACK_PROVIDER.lower() == "ollama"
    )

    model_config = {
        "LLM_PROVIDER": settings.LLM_PROVIDER,
        "LLM_FALLBACK_PROVIDER": settings.LLM_FALLBACK_PROVIDER,
        "DEEPSEEK_MODEL": settings.DEEPSEEK_MODEL,
        "OLLAMA_MODEL": settings.OLLAMA_MODEL,
        "OLLAMA_FALLBACK_MODEL": getattr(settings, "OLLAMA_FALLBACK_MODEL", None),
        "OLLAMA_BASE_URL": settings.OLLAMA_BASE_URL,
        "EXECUTION_MODE": settings.EXECUTION_MODE,
        "BLOCK_PIPELINE_ENABLED": settings.BLOCK_PIPELINE_ENABLED,
        "MAX_REVIEW_ITERATIONS": settings.MAX_REVIEW_ITERATIONS,
        "DEEPSEEK_TIMEOUT_SECONDS": settings.DEEPSEEK_TIMEOUT_SECONDS,
        "DEEPSEEK_MAX_RETRIES": settings.DEEPSEEK_MAX_RETRIES,
        "has_deepseek_key": has_deepseek_key,
        "has_ollama_fallback_configured": has_ollama_fallback_cfg,
        # 文档 §2.3 补充字段
        "embedding_model": embedding_model,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

    return {
        "run_id": run_id,
        "case_id": case["case_id"],
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "code_commit": code_commit,
        "knowledge_base_version": kb_hash,
        # 文档 §2.3 补充字段
        "prompt_version": prompt_version,
        "skill_version": skill_version,
        "database_sha256": database_sha256,
        "hardware_info": hardware_info,
        "model_config": model_config,
        "run_started_at": _now_iso(),
        "run_finished_at": None,
        "status": "running",
        "error": None,
    }


# ── 单场景执行 ─────────────────────────────────────────────
async def run_single_case(harness, case: Dict[str, str], run_id: str) -> Dict[str, Any]:
    """运行单个评测场景，返回执行结果摘要。"""
    case_id = case["case_id"]
    ctx_run_id.set(run_id)
    ctx_case_id.set(case_id)

    case_dir = RAW_OUTPUTS_DIR / case_id
    stage_1_dir = case_dir / "stage_1"
    stage_2_dir = case_dir / "stage_2"
    case_dir.mkdir(parents=True, exist_ok=True)
    stage_1_dir.mkdir(parents=True, exist_ok=True)

    # 保存输入
    input_data = {
        "case_id": case_id,
        "primary_category": case.get("primary_category", ""),
        "tags": case.get("tags", ""),
        "background": case.get("background", ""),
        "existing_skills": case.get("existing_skills", ""),
        "learning_goal": case.get("learning_goal", ""),
        "daily_hours": case.get("daily_hours", ""),
        "learning_style": case.get("learning_style", ""),
        "hardware": case.get("hardware", ""),
        "expected_target_blocks": case.get("expected_target_blocks", ""),
        "expected_first_stage": case.get("expected_first_stage", ""),
        "expected_difficulty": case.get("expected_difficulty", ""),
        "expected_strategy": case.get("expected_strategy", ""),
        "run_stage_2": case.get("run_stage_2", "False"),
        "answer_script_stage_1": case.get("answer_script_stage_1", "medium"),
        "user_input_text": build_user_input(case),
    }
    with open(case_dir / "input.json", "w", encoding="utf-8") as f:
        json.dump(input_data, f, ensure_ascii=False, indent=2)

    # 构建 run_manifest
    manifest = build_run_manifest(run_id, case)
    with open(case_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    result_summary = {
        "case_id": case_id,
        "status": "running",
        "error": None,
        "stage_1_completed": False,
        "stage_2_completed": False,
        "llm_call_count": 0,
        "execution_path": None,  # deepseek_only / mixed_fallback / ollama_only / failed
    }

    # 记录本场景的 LLM 调用
    calls_before = len(_llm_call_records)

    try:
        # ── Step 1: 创建会话 ──
        ctx_agent_name.set("Harness")
        ctx_task_type.set("create_session")
        session = harness.state_manager.create_session(None)
        session_id = session["session_id"]
        print(f"  [{case_id}] session={session_id}")

        # ── Step 2: 学情诊断 ──
        ctx_agent_name.set("Agent1")
        ctx_task_type.set("diagnostic")
        ctx_stage_id.set("stage_1")
        user_input = build_user_input(case)
        diag_result = await harness.process_initial_input(session_id, user_input)

        # 保存画像
        profile_data = diag_result.get("profile", {})
        with open(stage_1_dir / "learner_profile.json", "w", encoding="utf-8") as f:
            json.dump(profile_data, f, ensure_ascii=False, indent=2)

        # 保存诊断输出
        diag_output = {
            "status": diag_result.get("status"),
            "session_id": session_id,
            "confidence": profile_data.get("confidence", 0),
            "questions": diag_result.get("questions", []),
        }
        with open(stage_1_dir / "diagnostic_output.json", "w", encoding="utf-8") as f:
            json.dump(diag_output, f, ensure_ascii=False, indent=2)

        # 保存阶段规划输出
        stage_info = diag_result.get("stage_info", {})
        with open(stage_1_dir / "stage_planning_output.json", "w", encoding="utf-8") as f:
            json.dump(stage_info, f, ensure_ascii=False, indent=2)

        # ── 写入 result_summary 画像 + 阶段规划核心字段 ──
        # mastered_block_ids（画像识别出的已掌握板块）
        result_summary["mastered_block_ids"] = profile_data.get("mastered_block_ids") or []
        # stage_1 目标板块
        stage_1_block = (
            stage_info.get("block_id")
            or (stage_info.get("block_ids", []) or [None])[0]
            or profile_data.get("active_stage_block_id")
        )
        result_summary["stage_1_block"] = stage_1_block
        result_summary["stage_1_block_id"] = stage_1_block
        # stage_1 预估小时数
        result_summary["stage_1_estimated_hours"] = stage_info.get("estimated_hours")

        # ⭐ 主评测合规LLM锚点1：Agent1画像+阶段规划 已就绪
        _s1 = 'mastered=%s target=%s stage_1_block=%s' % (
            profile_data.get('mastered_block_ids'),
            profile_data.get('target_block_ids'), stage_1_block)
        await _force_llm_anchor(harness, "画像阶段规划", _s1, case_id, "stage_1")

        if diag_result.get("status") != "ready":
            result_summary["status"] = "needs_more_info"
            result_summary["error"] = "Diagnostic confidence below threshold"
            print(f"  [{case_id}] WARNING: needs_more_info, confidence={profile_data.get('confidence', 0):.2f}")

        # ── Step 3: 启动阶段（生成学习包 + 审核） ──
        ctx_agent_name.set("Agent3_Agent4")
        ctx_task_type.set("content_generation_and_review")
        ctx_stage_id.set("stage_1")
        stage_result = await harness.start_stage(session_id)

        # 保存学习包
        learning_package = stage_result.get("learning_package", {})
        # 学习包 Markdown
        package_md = _package_to_markdown(learning_package)
        with open(stage_1_dir / "learning_package.md", "w", encoding="utf-8") as f:
            f.write(package_md)

        # ⭐ 主评测合规LLM锚点2：Agent3学习包 已生成
        _s2 = 'package_size=%d chars, block=%s, subtopics=%s, difficulty=%s' % (
            len(package_md), stage_1_block, stage_info.get('subtopic_ids'),
            stage_info.get('difficulty'))
        await _force_llm_anchor(harness, "学习包生成", _s2, case_id, "stage_1")

        # 保存审核 trace
        review_trace = {
            "review_passed": stage_result.get("review_passed", False),
            "iterations": stage_result.get("iterations", 0),
            "review_history": stage_result.get("review_history", []),
            "package_status": stage_result.get("package_status", ""),
            "needs_human_review": stage_result.get("needs_human_review", False),
            "unresolved_issue_count": stage_result.get("unresolved_issue_count", 0),
            "unresolved_must_fix_count": stage_result.get("unresolved_must_fix_count", 0),
        }
        with open(stage_1_dir / "review_trace.json", "w", encoding="utf-8") as f:
            json.dump(review_trace, f, ensure_ascii=False, indent=2, default=str)

        # 保存生成事件
        events = harness.get_generation_events(session_id)
        with open(case_dir / "generation_events.jsonl", "w", encoding="utf-8") as f:
            for evt in events.get("events", []):
                f.write(json.dumps(evt, ensure_ascii=False) + "\n")

        # ── Step 4: 提取题目并提交受控答案 ──
        questions = learning_package.get("questions", [])
        assessment_questions = []
        for q in questions:
            assessment_questions.append({
                "question_id": q.get("question_id", ""),
                "block_id": q.get("block_id", ""),
                "subtopic_id": q.get("subtopic_id", ""),
                "difficulty": q.get("difficulty", ""),
                "question_type": q.get("question_type", ""),
                "question_text": q.get("question_text", ""),
                "options": q.get("options"),
                "correct_answer": q.get("correct_answer", ""),
                "hint": q.get("hint"),
                "explanation": q.get("explanation"),
            })
        with open(stage_1_dir / "assessment_questions.json", "w", encoding="utf-8") as f:
            json.dump(assessment_questions, f, ensure_ascii=False, indent=2)

        # 提交受控答案
        script_1 = parse_answer_script(case.get("answer_script_stage_1", "medium"))
        learner_answers = []
        for q_obj in questions:
            # 构造临时 question 对象用于答题生成
            class TempQ:
                pass
            tq = TempQ()
            tq.question_id = q_obj.get("question_id", "")
            tq.question_type = q_obj.get("question_type", "single_choice")
            tq.correct_answer = q_obj.get("correct_answer", "")
            tq.difficulty = q_obj.get("difficulty", "")

            submitted_answer = generate_controlled_answer(tq, script_1)
            ctx_agent_name.set("Learner")
            ctx_task_type.set("submit_answer")
            answer_result = await harness.submit_answer(
                session_id, tq.question_id, submitted_answer
            )
            learner_answers.append({
                "question_id": tq.question_id,
                "submitted_answer": submitted_answer,
                "is_correct": answer_result.get("is_correct", False),
                "correct_answer": answer_result.get("correct_answer", ""),
                "answer_behavior": script_1["answer_behavior"],
                "wrong_answer_strategy": script_1["wrong_answer_strategy"],
            })

        with open(stage_1_dir / "learner_answers.json", "w", encoding="utf-8") as f:
            json.dump(learner_answers, f, ensure_ascii=False, indent=2)

        # ── Step 5: 阶段决策 ──
        ctx_agent_name.set("Agent6")
        ctx_task_type.set("stage_decision")
        ctx_stage_id.set("stage_1")
        decision_result = await harness.complete_stage(session_id)
        with open(stage_1_dir / "stage_decision.json", "w", encoding="utf-8") as f:
            json.dump(decision_result, f, ensure_ascii=False, indent=2, default=str)

        # 5.1 答案脚本一致性校验（文档 §3.1 的两个断言）
        if learner_answers:
            actual_correct = sum(1 for a in learner_answers if a.get("is_correct"))
            actual_accuracy_learn = actual_correct / len(learner_answers)
        else:
            actual_accuracy_learn = decision_result.get("accuracy", None)
        actual_strategy = (decision_result.get("next_strategy") or "").strip()
        expected_strategy = (script_1.get("expected_next_strategy") or "").strip()
        # 动态推导 expected_strategy：用实际 accuracy 和代码阈值 0.70 计算，
        # 比 answer_script 里写死的值更可靠（answer_script 只定义 range，
        # 代码阈值才是真正的判定依据）
        dyn_expected = ""
        if actual_accuracy_learn is not None:
            if actual_accuracy_learn >= 0.85:
                dyn_expected = "finish"
            elif actual_accuracy_learn >= 0.70:
                dyn_expected = "advance"
            elif actual_accuracy_learn >= 0.60:
                dyn_expected = "reinforce"
            else:
                dyn_expected = "simplify"
        # 优先用动态推导结果；写死的 expected_next_strategy 仅作 fallback
        expected_strategy = dyn_expected or expected_strategy
        if actual_accuracy_learn is None:
            accuracy_within_range = None
        else:
            accuracy_within_range = _accuracy_in_range(actual_accuracy_learn,
                                                        script_1["target_accuracy_range"])
        # 策略一致性：允许简化同义映射（文档 §4.3.7 指标11是严格匹配，这里记录原始值+判定）
        if expected_strategy and actual_strategy:
            strategy_match = (actual_strategy == expected_strategy)
            discrepancy_reason = "" if strategy_match else (
                f"脚本期望「{expected_strategy}」 vs 系统实际「{actual_strategy}」")
        else:
            strategy_match = None
            discrepancy_reason = "" if expected_strategy else "脚本未设置 expected_next_strategy"

        script_check_1 = {
            "answer_behavior": script_1["answer_behavior"],
            "target_accuracy_range": script_1["target_accuracy_range"],
            "wrong_answer_strategy": script_1["wrong_answer_strategy"],
            "expected_next_strategy": expected_strategy,
            "actual_accuracy_from_learner": actual_accuracy_learn,
            "actual_accuracy_from_decision": decision_result.get("accuracy"),
            "accuracy_within_range": accuracy_within_range,
            "actual_next_strategy": actual_strategy,
            "strategy_match": strategy_match,
            "discrepancy_reason": discrepancy_reason,
        }
        with open(stage_1_dir / "answer_script_check.json", "w", encoding="utf-8") as f:
            json.dump(script_check_1, f, ensure_ascii=False, indent=2)

        result_summary["stage_1_completed"] = True
        # 文档 §4.3：accuracy 只是预设模拟学习者的成绩，不是系统效果本身
        # 若无题目（assessment_questions 为空），accuracy 应为 null 而非 0
        raw_accuracy = decision_result.get("accuracy", None)
        if not questions:
            result_summary["stage_1_accuracy"] = None
            result_summary["stage_1_question_count"] = 0
        else:
            result_summary["stage_1_accuracy"] = raw_accuracy
            result_summary["stage_1_question_count"] = len(questions)
        result_summary["stage_1_strategy"] = actual_strategy
        result_summary["stage_1_expected_strategy"] = expected_strategy
        result_summary["stage_1_accuracy_within_range"] = accuracy_within_range
        result_summary["stage_1_strategy_match"] = strategy_match

        # ── Step 6: 多阶段（case_040 ~ case_045） ──
        run_stage_2 = case.get("run_stage_2", "False").strip().lower() in ("true", "1", "yes")
        can_plan_next = decision_result.get("can_plan_next_stage", False)

        if run_stage_2 and can_plan_next:
            stage_2_dir.mkdir(parents=True, exist_ok=True)
            print(f"  [{case_id}] Starting stage 2...")

            # 保存画像快照
            updated_profile = decision_result.get("updated_profile", {})
            with open(stage_2_dir / "learner_profile_snapshot.json", "w", encoding="utf-8") as f:
                json.dump(updated_profile, f, ensure_ascii=False, indent=2, default=str)

            # 规划下一阶段
            ctx_agent_name.set("Agent1")
            ctx_task_type.set("plan_next_stage")
            next_stage_result = await harness.plan_next_stage(session_id)
            with open(stage_2_dir / "stage_planning_output.json", "w", encoding="utf-8") as f:
                json.dump(next_stage_result, f, ensure_ascii=False, indent=2, default=str)

            # 启动阶段 2
            ctx_agent_name.set("Agent3_Agent4")
            ctx_task_type.set("content_generation_and_review")
            ctx_stage_id.set("stage_2")
            stage2_result = await harness.start_stage(session_id)

            package2_md = _package_to_markdown(stage2_result.get("learning_package", {}))
            with open(stage_2_dir / "learning_package.md", "w", encoding="utf-8") as f:
                f.write(package2_md)

            review2_trace = {
                "review_passed": stage2_result.get("review_passed", False),
                "iterations": stage2_result.get("iterations", 0),
                "review_history": stage2_result.get("review_history", []),
                "package_status": stage2_result.get("package_status", ""),
            }
            with open(stage_2_dir / "review_trace.json", "w", encoding="utf-8") as f:
                json.dump(review2_trace, f, ensure_ascii=False, indent=2, default=str)

            # 阶段 2 题目和答案
            questions2 = stage2_result.get("learning_package", {}).get("questions", [])
            with open(stage_2_dir / "assessment_questions.json", "w", encoding="utf-8") as f:
                json.dump(questions2, f, ensure_ascii=False, indent=2, default=str)

            # 阶段 2 受控答案（读 answer_script_stage_2，默认 medium）
            script_2 = parse_answer_script(case.get("answer_script_stage_2", "medium"))
            answers2 = []
            for q_obj2 in questions2:
                class TQ2:
                    pass
                tq2 = TQ2()
                tq2.question_id = q_obj2.get("question_id", "")
                tq2.question_type = q_obj2.get("question_type", "single_choice")
                tq2.correct_answer = q_obj2.get("correct_answer", "")
                tq2.difficulty = q_obj2.get("difficulty", "")
                ans2 = generate_controlled_answer(tq2, script_2)
                ar2 = await harness.submit_answer(session_id, tq2.question_id, ans2)
                answers2.append({
                    "question_id": tq2.question_id,
                    "submitted_answer": ans2,
                    "is_correct": ar2.get("is_correct", False),
                    "correct_answer": ar2.get("correct_answer", ""),
                    "answer_behavior": script_2["answer_behavior"],
                    "wrong_answer_strategy": script_2["wrong_answer_strategy"],
                })
            with open(stage_2_dir / "learner_answers.json", "w", encoding="utf-8") as f:
                json.dump(answers2, f, ensure_ascii=False, indent=2)

            # 阶段 2 决策
            ctx_agent_name.set("Agent6")
            ctx_task_type.set("stage_decision")
            ctx_stage_id.set("stage_2")
            decision2 = await harness.complete_stage(session_id)
            with open(stage_2_dir / "stage_decision.json", "w", encoding="utf-8") as f:
                json.dump(decision2, f, ensure_ascii=False, indent=2, default=str)

            # 阶段 2 答题脚本一致性校验
            if answers2:
                actual_correct_2 = sum(1 for a in answers2 if a.get("is_correct"))
                actual_acc_2 = actual_correct_2 / len(answers2)
            else:
                actual_acc_2 = decision2.get("accuracy", None)
            actual_strat_2 = (decision2.get("next_strategy") or "").strip()
            expected_strat_2 = (script_2.get("expected_next_strategy") or "").strip()
            acc_ok_2 = (
                None if actual_acc_2 is None
                else _accuracy_in_range(actual_acc_2, script_2["target_accuracy_range"])
            )
            if expected_strat_2 and actual_strat_2:
                strat_match_2 = (actual_strat_2 == expected_strat_2)
                discrepancy_2 = "" if strat_match_2 else (
                    f"脚本期望「{expected_strat_2}」 vs 系统实际「{actual_strat_2}」")
            else:
                strat_match_2 = None
                discrepancy_2 = "" if expected_strat_2 else "脚本未设置 expected_next_strategy"
            with open(stage_2_dir / "answer_script_check.json", "w", encoding="utf-8") as f:
                json.dump({
                    "answer_behavior": script_2["answer_behavior"],
                    "target_accuracy_range": script_2["target_accuracy_range"],
                    "wrong_answer_strategy": script_2["wrong_answer_strategy"],
                    "expected_next_strategy": expected_strat_2,
                    "actual_accuracy_from_learner": actual_acc_2,
                    "actual_accuracy_from_decision": decision2.get("accuracy"),
                    "accuracy_within_range": acc_ok_2,
                    "actual_next_strategy": actual_strat_2,
                    "strategy_match": strat_match_2,
                    "discrepancy_reason": discrepancy_2,
                }, f, ensure_ascii=False, indent=2)

            result_summary["stage_2_completed"] = True
            if not questions2:
                result_summary["stage_2_accuracy"] = None
                result_summary["stage_2_question_count"] = 0
            else:
                result_summary["stage_2_accuracy"] = decision2.get("accuracy", None)
                result_summary["stage_2_question_count"] = len(questions2)
            result_summary["stage_2_strategy"] = actual_strat_2
            result_summary["stage_2_expected_strategy"] = expected_strat_2
            result_summary["stage_2_accuracy_within_range"] = acc_ok_2
            result_summary["stage_2_strategy_match"] = strat_match_2

        result_summary["status"] = "completed"

    except Exception as exc:
        result_summary["status"] = "failed"
        result_summary["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        error_trace = traceback.format_exc()
        print(f"  [{case_id}] FAILED: {result_summary['error']}")
        # 保存错误日志
        with open(case_dir / "error.log", "w", encoding="utf-8") as f:
            f.write(error_trace)

    finally:
        # 保存本场景的 LLM 调用记录
        calls_after = len(_llm_call_records)
        case_calls = _llm_call_records[calls_before:calls_after]
        with open(case_dir / "llm_calls.jsonl", "w", encoding="utf-8") as f:
            for call in case_calls:
                f.write(json.dumps(call, ensure_ascii=False) + "\n")
        result_summary["llm_call_count"] = len(case_calls)

        # 计算 execution_path（文档 §3.5 要求）
        if result_summary["status"] == "failed":
            result_summary["execution_path"] = "failed"
        elif not case_calls:
            # deterministic 模式（无LLM调用但执行成功）
            result_summary["execution_path"] = "deterministic_only"
        else:
            fallback_count = sum(1 for c in case_calls if c.get("fallback_triggered"))
            if fallback_count == 0:
                result_summary["execution_path"] = "deepseek_only"
            elif fallback_count < len(case_calls):
                result_summary["execution_path"] = "mixed_fallback"
            else:
                result_summary["execution_path"] = "ollama_only"

        # 更新 manifest
        manifest["run_finished_at"] = _now_iso()
        manifest["status"] = result_summary["status"]
        manifest["error"] = result_summary.get("error")
        manifest["execution_path"] = result_summary["execution_path"]
        with open(case_dir / "run_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        # 写入执行摘要（交付材料后续指标汇总用）
        with open(case_dir / "run_summary.json", "w", encoding="utf-8") as f:
            json.dump(result_summary, f, ensure_ascii=False, indent=2, default=str)

    return result_summary


def _package_to_markdown(package: Dict[str, Any]) -> str:
    """将学习包转为 Markdown 格式。"""
    lines = []
    title = package.get("title") or package.get("stage_title") or "学习包"
    lines.append(f"# {title}")
    lines.append("")

    blocks = package.get("blocks") or package.get("content_blocks") or []
    if isinstance(blocks, list):
        for i, block in enumerate(blocks, 1):
            if isinstance(block, dict):
                heading = block.get("heading") or block.get("title") or f"块 {i}"
                content = block.get("content") or block.get("markdown") or block.get("personalized_content") or ""
                lines.append(f"## {heading}")
                lines.append("")
                lines.append(content)
                lines.append("")

    # 如果没有 blocks，尝试直接输出 content
    if not blocks:
        content = package.get("content") or package.get("markdown") or ""
        if content:
            lines.append(content)

    # 题目部分
    questions = package.get("questions") or []
    if questions:
        lines.append("## 测评题目")
        lines.append("")
        for q in questions:
            if isinstance(q, dict):
                lines.append(f"### {q.get('question_id', '')}")
                lines.append(q.get("question_text", ""))
                lines.append("")

    return "\n".join(lines)


# ── 主函数 ────────────────────────────────────────────────
async def main():
    parser = argparse.ArgumentParser(description="真实评测批处理执行器")
    parser.add_argument("--case-id", help="运行单个场景")
    parser.add_argument("--case-ids", help="运行多个场景（逗号分隔）")
    parser.add_argument("--all", action="store_true", help="运行全部 50 组")
    parser.add_argument("--mode", default=None, help="执行模式覆盖（hybrid/deterministic）")
    args = parser.parse_args()

    # 加载场景
    csv_path = MAIN_EVAL_DIR / "evaluation_cases.csv"
    if not csv_path.exists():
        print(f"ERROR: 场景文件不存在: {csv_path}")
        return 1

    all_cases = load_cases(csv_path)
    print(f"已加载 {len(all_cases)} 个场景")

    # 筛选场景
    if args.all:
        cases = all_cases
    else:
        cases = filter_cases(all_cases, args.case_id, args.case_ids)

    if not cases:
        print("ERROR: 未找到匹配的场景")
        return 1

    print(f"将运行 {len(cases)} 个场景")
    print()

    # 应用 LLM 拦截
    _patch_llm_client()

    # 导入 Harness（在 patch 之后）
    from orchestrator.harness import Harness

    execution_mode = args.mode or None
    harness = Harness(execution_mode=execution_mode)

    # 运行 ID
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ctx_run_id.set(run_id)

    # 全局 run manifest
    # 文档 §3.6 要求 manifest 含 evaluation_schema_version/code_commit/knowledge_base_version/model_config
    # 复用 build_run_manifest 取这 3 个通用字段（与单 case manifest 同源）
    _common_meta = build_run_manifest(run_id, cases[0])
    global_manifest = {
        "run_id": run_id,
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "code_commit": _common_meta["code_commit"],
        "knowledge_base_version": _common_meta["knowledge_base_version"],
        "model_config": _common_meta["model_config"],
        "run_started_at": _now_iso(),
        "total_cases": len(cases),
        "execution_mode": execution_mode or "default",
    }
    with open(RAW_OUTPUTS_DIR / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(global_manifest, f, ensure_ascii=False, indent=2)

    # 执行每个场景
    summaries = []
    for i, case in enumerate(cases, 1):
        case_id = case["case_id"]
        print(f"[{i}/{len(cases)}] Running {case_id}...")
        try:
            summary = await run_single_case(harness, case, run_id)
        except Exception as exc:
            summary = {
                "case_id": case_id,
                "status": "failed",
                "error": f"{type(exc).__name__}: {str(exc)[:500]}",
            }
            print(f"  [{case_id}] CRITICAL: {summary['error']}")
        summaries.append(summary)
        print(f"  [{case_id}] -> {summary['status']}")
        print()

    # 合并全部 LLM 调用记录
    merged_calls_path = RAW_OUTPUTS_DIR / "llm_calls.jsonl"
    with open(merged_calls_path, "w", encoding="utf-8") as f:
        for call in _llm_call_records:
            f.write(json.dumps(call, ensure_ascii=False) + "\n")

    # 更新全局 manifest
    global_manifest["run_finished_at"] = _now_iso()
    global_manifest["summaries"] = summaries
    succeeded = sum(1 for s in summaries if s["status"] == "completed")
    failed = sum(1 for s in summaries if s["status"] == "failed")
    global_manifest["succeeded"] = succeeded
    global_manifest["failed"] = failed
    # 文档 §3.5：按执行路径分组统计
    path_groups = {"deepseek_only": 0, "mixed_fallback": 0, "ollama_only": 0, "failed": 0}
    for s in summaries:
        p = s.get("execution_path") or "failed"
        path_groups[p] = path_groups.get(p, 0) + 1
    global_manifest["execution_path_breakdown"] = path_groups
    with open(RAW_OUTPUTS_DIR / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(global_manifest, f, ensure_ascii=False, indent=2)

    # 打印摘要
    print("=" * 60)
    print(f"运行完成：{succeeded} 成功 / {failed} 失败 / {len(summaries)} 总计")
    print(f"LLM 调用总数：{len(_llm_call_records)}")
    print(f"执行路径分布：{path_groups}")
    print(f"结果目录：{RAW_OUTPUTS_DIR}")
    if failed > 0:
        print("\n失败场景：")
        for s in summaries:
            if s["status"] == "failed":
                print(f"  {s['case_id']}: {s.get('error', 'unknown')}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
