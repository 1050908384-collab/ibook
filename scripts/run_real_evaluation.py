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
                # 判断实际使用的模型
                if had_credentials:
                    actual_model = settings.DEEPSEEK_MODEL
                else:
                    actual_model = settings.OLLAMA_MODEL
                    fallback_triggered = True

                if result and not result.startswith("Error:"):
                    success = True
                else:
                    error_type = "empty_or_error_response"
                    # 检查是否回退到 Ollama
                    if had_credentials and self._can_fallback_to_ollama():
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


# ── 用户输入构造 ──────────────────────────────────────────
def build_user_input(case: Dict[str, str]) -> str:
    parts = []
    bg = case.get("background", "").strip()
    if bg:
        parts.append(f"我的学习背景：{bg}")
    skills = case.get("existing_skills", "").strip()
    if skills:
        parts.append(f"已掌握技能：{skills}")
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

WRONG_ANSWER_OPTIONS = {
    "single_choice": "A",
    "multiple_choice": "A",
    "true_false": "错误",
    "short_answer": "我不知道",
    "code_completion": "pass",
}


def generate_controlled_answer(question, behavior: str) -> str:
    """根据受控答题行为生成答案。

    behavior: high(90%正确) / medium(65%) / low(35%) / very_low(10%)
    """
    import random
    rng = random.Random(hash(question.question_id + behavior))
    ratio = ANSWER_BEHAVIOR_RATIOS.get(behavior, 0.5)
    if rng.random() < ratio:
        return question.correct_answer
    else:
        qtype = getattr(question, "question_type", "single_choice") or "single_choice"
        return WRONG_ANSWER_OPTIONS.get(qtype, "我不知道")


# ── Run Manifest ──────────────────────────────────────────
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

    # 获取知识库版本（文件哈希）
    kb_hash = "unknown"
    kb_path = ROOT_DIR / "data" / "seeds" / "knowledge_blocks.json"
    if kb_path.exists():
        with open(kb_path, "rb") as f:
            kb_hash = hashlib.sha256(f.read()).hexdigest()[:12]

    # 模型配置（不记录 API Key）
    model_config = {
        "LLM_PROVIDER": settings.LLM_PROVIDER,
        "LLM_FALLBACK_PROVIDER": settings.LLM_FALLBACK_PROVIDER,
        "DEEPSEEK_MODEL": settings.DEEPSEEK_MODEL,
        "OLLAMA_MODEL": settings.OLLAMA_MODEL,
        "OLLAMA_BASE_URL": settings.OLLAMA_BASE_URL,
        "EXECUTION_MODE": settings.EXECUTION_MODE,
        "BLOCK_PIPELINE_ENABLED": settings.BLOCK_PIPELINE_ENABLED,
        "MAX_REVIEW_ITERATIONS": settings.MAX_REVIEW_ITERATIONS,
        "DEEPSEEK_TIMEOUT_SECONDS": settings.DEEPSEEK_TIMEOUT_SECONDS,
        "DEEPSEEK_MAX_RETRIES": settings.DEEPSEEK_MAX_RETRIES,
        "has_deepseek_key": bool(settings.DEEPSEEK_API_KEY) and
            settings.DEEPSEEK_API_KEY not in ("", "your-deepseek-api-key"),
    }

    return {
        "run_id": run_id,
        "case_id": case["case_id"],
        "code_commit": code_commit,
        "knowledge_base_version": kb_hash,
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
        "background": case.get("background", ""),
        "existing_skills": case.get("existing_skills", ""),
        "learning_goal": case.get("learning_goal", ""),
        "daily_hours": case.get("daily_hours", ""),
        "learning_style": case.get("learning_style", ""),
        "hardware": case.get("hardware", ""),
        "expected_target_blocks": case.get("expected_target_blocks", ""),
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
        answer_behavior = case.get("answer_script_stage_1", "medium")
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

            submitted_answer = generate_controlled_answer(tq, answer_behavior)
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
                "answer_behavior": answer_behavior,
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

        result_summary["stage_1_completed"] = True
        result_summary["stage_1_accuracy"] = decision_result.get("accuracy", 0)
        result_summary["stage_1_strategy"] = decision_result.get("next_strategy", "")

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

            # 阶段 2 受控答案（中等行为）
            answers2 = []
            for q_obj2 in questions2:
                class TQ2:
                    pass
                tq2 = TQ2()
                tq2.question_id = q_obj2.get("question_id", "")
                tq2.question_type = q_obj2.get("question_type", "single_choice")
                tq2.correct_answer = q_obj2.get("correct_answer", "")
                tq2.difficulty = q_obj2.get("difficulty", "")
                ans2 = generate_controlled_answer(tq2, "medium")
                ar2 = await harness.submit_answer(session_id, tq2.question_id, ans2)
                answers2.append({
                    "question_id": tq2.question_id,
                    "submitted_answer": ans2,
                    "is_correct": ar2.get("is_correct", False),
                    "correct_answer": ar2.get("correct_answer", ""),
                    "answer_behavior": "medium",
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

            result_summary["stage_2_completed"] = True
            result_summary["stage_2_accuracy"] = decision2.get("accuracy", 0)
            result_summary["stage_2_strategy"] = decision2.get("next_strategy", "")

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

        # 更新 manifest
        manifest["run_finished_at"] = _now_iso()
        manifest["status"] = result_summary["status"]
        manifest["error"] = result_summary.get("error")
        with open(case_dir / "run_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

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
    global_manifest = {
        "run_id": run_id,
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
    with open(RAW_OUTPUTS_DIR / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(global_manifest, f, ensure_ascii=False, indent=2)

    # 打印摘要
    print("=" * 60)
    print(f"运行完成：{succeeded} 成功 / {failed} 失败 / {len(summaries)} 总计")
    print(f"LLM 调用总数：{len(_llm_call_records)}")
    print(f"结果目录：{RAW_OUTPUTS_DIR}")
    if failed > 0:
        print("\n失败场景：")
        for s in summaries:
            if s["status"] == "failed":
                print(f"  {s['case_id']}: {s.get('error', 'unknown')}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
