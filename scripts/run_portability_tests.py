# -*- coding: utf-8 -*-
"""12 x 2 模型可替换性测试。

来源 case（与《测试数据建议方法》一致）:
    case_004, case_010, case_017, case_020,
    case_026, case_029, case_034, case_038,
    case_040, case_043, case_048, case_049

两种配置:
    配置 A: 强制 DeepSeek 云端 —— LLM_PROVIDER=deepseek, FALLBACK_PROVIDER=""
    配置 B: 强制 Ollama 本地 —— LLM_PROVIDER=ollama,  FALLBACK_PROVIDER=""

用法:
    python scripts/run_portability_tests.py
    python scripts/run_portability_tests.py --cases case_004,case_026
    python scripts/run_portability_tests.py --providers DeepSeek
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(BACKEND_DIR))

DELIVERY_DIR = ROOT_DIR / "测试数据与评测材料"
PORT_DIR = DELIVERY_DIR / "05_模型可替换性测试"

DEFAULT_CASES = [
    "case_004", "case_010", "case_017", "case_020",
    "case_026", "case_029", "case_034", "case_038",
    "case_040", "case_043", "case_048", "case_049",
]

PROVIDERS = ["DeepSeek", "Ollama"]
RAW_OUTPUTS_DIR = DELIVERY_DIR / "02_50组个性化主评测" / "raw_outputs"
REQUIRED_EVALUATION_SCHEMA_VERSION = "2026-08-26-final"


# =====================================================================
# 主评测结果复用（针对 DeepSeek-only 配置，要求主评测没走 Ollama fallback）
# =====================================================================
def extract_deepseek_from_main(case_id: str) -> Dict[str, Any] | None:
    """若主评测 raw_outputs/{case_id}/ 里：
       - has_deepseek_key=true
       - llm_calls 中所有 actual_models 都不含 ollama 字符串（没有回退）
    则视为等价 DeepSeek-only 可替换性结果，直接抽取。"""
    case_dir = RAW_OUTPUTS_DIR / case_id
    stage_1 = case_dir / "stage_1"
    needed = [
        case_dir / "run_manifest.json",
        case_dir / "llm_calls.jsonl",
        stage_1 / "learner_profile.json",
        stage_1 / "stage_planning_output.json",
        stage_1 / "review_trace.json",
        stage_1 / "assessment_questions.json",
        stage_1 / "stage_decision.json",
    ]
    if not all(p.exists() for p in needed):
        return None

    with open(needed[0], "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("evaluation_schema_version") != REQUIRED_EVALUATION_SCHEMA_VERSION:
        return None
    cfg = manifest.get("model_config", {}) or {}
    if not cfg.get("has_deepseek_key") or cfg.get("LLM_PROVIDER", "").lower() != "deepseek":
        return None

    llm_calls: List[Dict[str, Any]] = []
    with open(needed[1], "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    llm_calls.append(json.loads(line))
                except Exception:
                    pass

    actual_models: set[str] = set()
    total_calls = len(llm_calls)
    failed_calls = 0
    fallback_count = 0
    for c in llm_calls:
        if c.get("actual_model"):
            actual_models.add(c["actual_model"])
        if c.get("fallback_triggered"):
            fallback_count += 1
        if not c.get("success", True):
            failed_calls += 1

    # 严格：不能有任何 Ollama 模型名
    if any("ollama" in m.lower() or "deepseek-r1" in m.lower() or "qwen" in m.lower()
           for m in actual_models):
        return None

    with open(needed[2], "r", encoding="utf-8") as f:
        profile = json.load(f)
    with open(needed[3], "r", encoding="utf-8") as f:
        plan = json.load(f)
    with open(needed[4], "r", encoding="utf-8") as f:
        review = json.load(f)
    with open(needed[5], "r", encoding="utf-8") as f:
        questions = json.load(f)
    with open(needed[6], "r", encoding="utf-8") as f:
        decision = json.load(f)

    total_latency_ms = sum(int(c.get("latency_ms") or 0) for c in llm_calls)
    qcount = len(questions) if isinstance(questions, list) else 0

    return {
        "case_id": case_id,
        "provider": "DeepSeek",
        "status": "success",
        "profile_confidence": profile.get("confidence"),
        "planned_block": plan.get("block_name"),
        "review_passed": review.get("review_passed"),
        "review_iterations": review.get("iterations", 0),
        "question_count": qcount,
        "accuracy": decision.get("accuracy"),
        "next_strategy": decision.get("next_strategy"),
        "actual_models": "|".join(sorted(actual_models)),
        "total_llm_calls": total_calls,
        "failed_llm_calls": failed_calls,
        "fallback_count": fallback_count,
        "total_latency_ms": total_latency_ms,
        "package_chars": 0,  # 复用模式下不重算，留 0；若需要，可加 learning_package.md 读取
        "used_only_expected_provider": True,
        "data_source": f"reused_from_main_evaluation run_id={manifest.get('run_id')}",
        "source_run_id": manifest.get("run_id"),
        "run_started_at": manifest.get("run_started_at"),
        "run_finished_at": manifest.get("run_finished_at"),
    }


# =====================================================================
# 工具
# =====================================================================
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


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def apply_env_overrides(overrides: Dict[str, str]) -> Dict[str, str]:
    saved = {k: os.environ.get(k) for k in overrides}
    for k, v in overrides.items():
        os.environ[k] = v
    return saved


def restore_env(saved: Dict[str, str]):
    for k, old_v in saved.items():
        if old_v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = old_v


PROVIDER_ENV = {
    "DeepSeek": {
        # 强制 DeepSeek 云端，关闭回退（强制走这一路，用于证明可替换）
        "LLM_PROVIDER": "deepseek",
        "LLM_FALLBACK_PROVIDER": "",
    },
    "Ollama": {
        # 强制 Ollama 本地
        "LLM_PROVIDER": "ollama",
        "LLM_FALLBACK_PROVIDER": "",
        # 让 DEEPSEEK API Key 逻辑不生效（置空避免误尝试）
        "DEEPSEEK_API_KEY": "",
    },
}


async def run_case_provider(case_id: str, provider: str,
                            reuse_main: bool = True) -> Dict[str, Any]:
    # DeepSeek-only：主评测未发生回退则直接抽取（零 LLM 消耗）
    if reuse_main and provider == "DeepSeek":
        reused = extract_deepseek_from_main(case_id)
        if reused is not None:
            return reused

    from orchestrator.harness import Harness
    from config import Settings

    env_overrides = PROVIDER_ENV[provider]
    saved_env = apply_env_overrides(env_overrides)

    run_started = _now_iso()
    t0 = time.time()
    try:
        # 实例化新的 settings，让它重新读 env
        fresh = Settings(_env_file=None)
        harness = Harness()
        session = harness.state_manager.create_session(None)
        session_id = session["session_id"]
        user_input = build_user_input(load_case_input(case_id))

        diag = await harness.process_initial_input(session_id, user_input)
        if diag.get("status") != "ready":
            return {
                "case_id": case_id,
                "provider": provider,
                "status": f"diagnostic_{diag.get('status')}",
                "error": "Diagnostic confidence below threshold",
                "run_started_at": run_started,
                "run_finished_at": _now_iso(),
            }

        profile = diag.get("profile", {})
        stage_info = diag.get("stage_info", {})
        stage_res = await harness.start_stage(session_id)

        learning_package = stage_res.get("learning_package", {})
        questions = learning_package.get("questions", []) or []
        # 受控答题 70%
        import random
        rng = random.Random(hash(case_id + provider) & 0xFFFFFF)
        for q in questions:
            qid = q.get("question_id", "")
            correct = q.get("correct_answer", "")
            submitted = correct if rng.random() < 0.7 else "错误答案"
            if qid:
                try:
                    await harness.submit_answer(session_id, qid, submitted)
                except Exception:
                    pass

        decision = await harness.complete_stage(session_id)

        t1 = time.time()
        latency_ms = int((t1 - t0) * 1000)

        # 从 LLM 调用记录提取模型名/成功率
        actual_models = set()
        total_calls = 0
        failed_calls = 0
        fallback_count = 0
        try:
            from scripts.run_real_evaluation import _llm_call_records as lcr
            recent = lcr[-50:]
            total_calls = len(recent)
            for c in recent:
                if c.get("actual_model"):
                    actual_models.add(c["actual_model"])
                if not c.get("success", True):
                    failed_calls += 1
                if c.get("fallback_triggered"):
                    fallback_count += 1
        except Exception:
            pass

        return {
            "case_id": case_id,
            "provider": provider,
            "status": "success",
            "profile_confidence": profile.get("confidence"),
            "planned_block": stage_info.get("block_name"),
            "review_passed": stage_res.get("review_passed"),
            "review_iterations": stage_res.get("iterations", 0),
            "question_count": len(questions),
            "accuracy": decision.get("accuracy"),
            "next_strategy": decision.get("next_strategy"),
            "actual_models": "|".join(sorted(actual_models)),
            "total_llm_calls": total_calls,
            "failed_llm_calls": failed_calls,
            "fallback_count": fallback_count,
            "total_latency_ms": latency_ms,
            "package_chars": len(
                (learning_package.get("title") or "") +
                "".join(
                    (b.get("personalized_content") or b.get("content") or "")
                    for b in (learning_package.get("blocks") or learning_package.get("content_blocks") or [])
                    if isinstance(b, dict)
                )
            ),
            "used_only_expected_provider": (
                # DeepSeek 模式：实际只出现 deepseek-chat，没有 Ollama 回退
                (provider == "DeepSeek" and all("ollama" not in m.lower() for m in actual_models))
                or
                # Ollama 模式：只出现 Ollama 模型
                (provider == "Ollama" and all(
                    any(name in m.lower() for name in ["deepseek-r1", "qwen", "ollama"])
                    for m in actual_models
                )) if actual_models else False
            ),
            "run_started_at": run_started,
            "run_finished_at": _now_iso(),
        }
    except Exception as exc:
        return {
            "case_id": case_id,
            "provider": provider,
            "status": "crashed",
            "error": f"{type(exc).__name__}: {str(exc)[:500]}",
            "run_started_at": run_started,
            "run_finished_at": _now_iso(),
        }
    finally:
        restore_env(saved_env)


async def main():
    parser = argparse.ArgumentParser(description="12×2 模型可替换性测试")
    parser.add_argument("--cases", default=",".join(DEFAULT_CASES))
    parser.add_argument("--providers", default=",".join(PROVIDERS))
    parser.add_argument("--no-reuse-main", action="store_true",
                        help="关闭主评测结果复用（默认开启；已跑过的 DeepSeek-only 12 条可直接从主评测抽）")
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()

    cases = [x.strip() for x in args.cases.split(",") if x.strip()]
    providers = [x.strip() for x in args.providers.split(",") if x.strip()]
    reuse_main = not args.no_reuse_main

    PORT_DIR.mkdir(parents=True, exist_ok=True)

    if args.dry:
        total = 0
        reused_count = 0
        for c in cases:
            for p in providers:
                total += 1
                if reuse_main and p == "DeepSeek" and extract_deepseek_from_main(c) is not None:
                    reused_count += 1
        print(f"可替换性测试规划: {len(cases)} cases × {len(providers)} providers = {total} 条")
        for c in cases:
            line = []
            for p in providers:
                if reuse_main and p == "DeepSeek" and extract_deepseek_from_main(c) is not None:
                    line.append(f"{p}[REUSE]")
                else:
                    line.append(f"{p}[RUN]")
            print(f"  {c}: " + ", ".join(line))
        print(f"\n  [PLAN] 预计真 LLM 调用条数: {total - reused_count}（DeepSeek {reused_count} 条从主评测抽取，零消耗）")
        return 0

    rows = []
    reused_global = 0
    real_runs = 0
    for i, case_id in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] Portability {case_id}")
        for p in providers:
            print(f"  · {p} ...", end=" ", flush=True)
            row = await run_case_provider(case_id, p, reuse_main=reuse_main)
            if row.get("data_source", "").startswith("reused_from_main_evaluation"):
                reused_global += 1
                print(f"REUSE (main: {row.get('source_run_id','')}) calls={row.get('total_llm_calls',0)} latency={row.get('total_latency_ms',0)}ms")
            else:
                real_runs += 1
                print(f"{row.get('status')} calls={row.get('total_llm_calls',0)} latency={row.get('total_latency_ms',0)}ms")
            rows.append(row)

    csv_path = PORT_DIR / "portability_cases.csv"
    fields = [
        "case_id", "provider", "status",
        "profile_confidence", "planned_block",
        "review_passed", "review_iterations",
        "question_count", "accuracy", "next_strategy",
        "actual_models", "total_llm_calls", "failed_llm_calls",
        "fallback_count", "package_chars",
        "used_only_expected_provider", "total_latency_ms",
        "data_source", "source_run_id",
        "run_started_at", "run_finished_at", "error",
    ]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    success = [r for r in rows if r.get("status") == "success"]
    ds_ok = [r for r in success if r.get("provider") == "DeepSeek" and r.get("used_only_expected_provider")]
    ol_ok = [r for r in success if r.get("provider") == "Ollama" and r.get("used_only_expected_provider")]
    print()
    print(f"完成 {len(rows)} 条 -> {csv_path}")
    print(f"  成功: {len(success)}/{len(rows)}")
    print(f"  [REUSE] 主评测复用(零 token): {reused_global} 条")
    print(f"  [RUN] 真 LLM 执行(产生消耗): {real_runs} 条")
    ds_total = sum(1 for r in success if r.get("provider") == "DeepSeek")
    ol_total = sum(1 for r in success if r.get("provider") == "Ollama")
    print(f"  DeepSeek-only 成功率: {len(ds_ok)}/{ds_total}")
    print(f"  Ollama-only 成功率:   {len(ol_ok)}/{ol_total}")
    if success:
        ds_lat = sum(r.get("total_latency_ms", 0) for r in success if r.get("provider") == "DeepSeek")
        ol_lat = sum(r.get("total_latency_ms", 0) for r in success if r.get("provider") == "Ollama")
        ds_cnt = max(1, sum(1 for r in success if r.get("provider") == "DeepSeek"))
        ol_cnt = max(1, sum(1 for r in success if r.get("provider") == "Ollama"))
        print(f"  平均延迟: DeepSeek {ds_lat/ds_cnt:.0f}ms / Ollama {ol_lat/ol_cnt:.0f}ms")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
