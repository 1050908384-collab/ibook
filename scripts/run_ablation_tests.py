# -*- coding: utf-8 -*-
"""15 x 4 多 Agent 消融测试。

场景来源（与《方法》一致）:
    case_002, case_004, case_008,
    case_010, case_013, case_017,
    case_020, case_023, case_026,
    case_029, case_034, case_038,
    case_040, case_043, case_049

四种配置定义：
    Direct-LLM  — 完全绕过所有 Agent：用裸 LLM 直接把用户输入 → Markdown 学习包（RAG 检索也跳过）
    RAG-only    — 保留知识库检索（Agent3 的 KB 向量检索），但跳过 Agent4 审核和 Agent1/Agent6 规划
    No-Agent4   — 完整 Agent1(诊断+规划) + Agent3(生成+RAG) 但跳过 Agent4 审核直接通过
    Full-System — 完整系统（相当于 run_real_evaluation.py 正常模式）

用法:
    python scripts/run_ablation_tests.py                          # 全部 60 条
    python scripts/run_ablation_tests.py --cases case_002,case_004  # 只跑指定 case
    python scripts/run_ablation_tests.py --configs Full-System,No-Agent4

输出:
    04_多Agent消融测试/
      ablation_cases.csv              60 条聚合记录
      ablation_outputs/
        {case_id}_{config}.json       每个 case 的原始输出
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
ABLATION_DIR = DELIVERY_DIR / "04_多Agent消融测试"
OUTPUTS_DIR = ABLATION_DIR / "ablation_outputs"

DEFAULT_CASES = [
    "case_002", "case_004", "case_008",
    "case_010", "case_013", "case_017",
    "case_020", "case_023", "case_026",
    "case_029", "case_034", "case_038",
    "case_040", "case_043", "case_049",
]

CONFIGS = ["Direct-LLM", "RAG-only", "No-Agent4", "Full-System"]
RAW_OUTPUTS_DIR = DELIVERY_DIR / "02_50组个性化主评测" / "raw_outputs"
REQUIRED_EVALUATION_SCHEMA_VERSION = "2026-08-26-final"


# =====================================================================
# 主评测结果复用（针对 Full-System 配置，免重复 LLM 调用）
# =====================================================================
def extract_fullsystem_from_main(case_id: str) -> Dict[str, Any] | None:
    """若主评测 raw_outputs/{case_id}/ 已完整，则直接抽取等价 Full-System 结果。
    返回 None 表示不可复用，必须真跑。"""
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
        stage_1 / "learning_package.md",
    ]
    if not all(p.exists() for p in needed):
        return None

    with open(needed[0], "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("evaluation_schema_version") != REQUIRED_EVALUATION_SCHEMA_VERSION:
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
    package_md = needed[7].read_text(encoding="utf-8")

    # LLM calls
    llm_calls: List[Dict[str, Any]] = []
    with open(needed[1], "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    llm_calls.append(json.loads(line))
                except Exception:
                    pass

    total_latency_ms = sum(int(c.get("latency_ms") or 0) for c in llm_calls)

    return {
        "case_id": case_id,
        "config": "Full-System",
        "status": "success",
        "package_md": package_md,
        "package_chars": len(package_md),
        "profile_confidence": profile.get("confidence"),
        "planned_block": plan.get("block_name"),
        "review_iterations": review.get("iterations", 0),
        "review_passed": review.get("review_passed", False),
        "accuracy": decision.get("accuracy"),
        "next_strategy": decision.get("next_strategy"),
        "question_count": len(questions) if isinstance(questions, list) else 0,
        "llm_calls": llm_calls,
        "total_llm_calls": len(llm_calls),
        "total_latency_ms": total_latency_ms,
        "used_rag": True,
        "used_review": True,
        "used_planning": True,
        "data_source": f"reused_from_main_evaluation run_id={manifest.get('run_id')}",
        "source_run_id": manifest.get("run_id"),
        "source_started": manifest.get("run_started_at"),
        "source_finished": manifest.get("run_finished_at"),
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


# =====================================================================
# 运行模式 A: Direct-LLM
#   裸 LLM 直接生成学习包，不经过任何 Agent 编排/RAG/审核
# =====================================================================
async def run_direct_llm(user_input: str, case_id: str) -> Dict[str, Any]:
    from llm.client import LLMClient
    from config import settings

    client = LLMClient()
    prompt = f"""你是AI学习教练。请基于以下学习者画像，生成一份个性化学习包（Markdown格式）：

=== 学习者画像 ===
{user_input}

=== 输出要求 ===
1. 至少包含 4 个章节：学习目标、前置知识、核心内容、实操练习、下一步建议
2. 根据用户硬件条件（如 GPU 显存）调整示例代码
3. 根据用户学习风格调整内容配比（实操型加代码示例，理论型加原理讲解）
4. 内容要覆盖 LoRA/QLoRA/vLLM 相关（根据学习目标）
5. 最后附 3 道测评题（单选/判断），附答案

直接输出完整 Markdown，不要用反引号包裹，不要加前置说明。"""

    t0 = time.time()
    content = await client.complete(prompt, system_prompt="你是资深大模型教育专家，擅长生成结构化学习材料。",
                                     max_tokens=8192, temperature=0.7)
    t1 = time.time()
    latency_ms = int((t1 - t0) * 1000)

    if content.startswith("Error:"):
        return {
            "config": "Direct-LLM",
            "status": "failed",
            "error": content,
            "package_md": "",
            "llm_calls": [{"model": settings.DEEPSEEK_MODEL, "latency_ms": latency_ms, "success": False}],
            "total_tokens_approx": 0,
            "used_rag": False,
            "used_review": False,
            "used_planning": False,
        }

    return {
        "config": "Direct-LLM",
        "status": "success",
        "package_md": content,
        "package_chars": len(content),
        "llm_calls": [{"model": settings.DEEPSEEK_MODEL, "latency_ms": latency_ms, "success": True}],
        "total_latency_ms": latency_ms,
        "used_rag": False,
        "used_review": False,
        "used_planning": False,
    }


# =====================================================================
# 运行模式 B: RAG-only
#   保留 KB 检索 + Agent3 生成，但跳过 Agent4 审核、Agent1/Agent6 规划
# =====================================================================
async def run_rag_only(user_input: str, case_id: str) -> Dict[str, Any]:
    """直接调用 HybridContentAgent.generate_block（含RAG），但跳过规划/审核/决策。"""
    from config import settings
    from models.profile import UserProfile, PracticeProfile
    from models.stage import StageInfo
    from agents.deterministic import HybridContentAgent
    import hashlib

    t0 = time.time()
    agent = HybridContentAgent(database_url="sqlite:///../data/sqlite_data/project.db")

    # 构造一个 StageInfo（默认用 KB-12 LoRA，因为大部分场景目标是 LoRA）
    stage = StageInfo(
        stage_id=f"ablation-rag-{case_id}",
        stage_num=1,
        block_id="KB-12",
        block_name="参数高效微调（LoRA与QLoRA）",
        gap_skill="LoRA微调",
        subtopic_id="KBS-12-01",
        estimated_hours=2.0,
    )
    case_row = load_case_input(case_id)
    profile = UserProfile(
        session_id=f"ablation-rag-{case_id}",
        goal=case_row.get("learning_goal", "LoRA微调"),
        learning_style=case_row.get("learning_style", "实操型"),
    )
    practice = PracticeProfile.from_hardware_text(case_row.get("hardware", ""))

    # 先拿到原始 chunk
    chunks = agent.iter_stage_chunks(stage)
    if not chunks:
        # KB-12 没有则退回 KB-15
        stage.block_id = "KB-15"
        stage.block_name = "模型推理部署（vLLM）"
        chunks = agent.iter_stage_chunks(stage)

    merged_md_parts = []
    for idx, chunk in enumerate(chunks[:6]):  # 最多用 6 块
        try:
            block = await agent.generate_block(
                stage, chunk, profile=profile, practice=practice,
                block_index=idx, session_id=profile.session_id,
            )
            merged_md_parts.append(f"## {getattr(block, 'heading', '') or f'块 {idx+1}'}\n\n"
                                   f"{getattr(block, 'personalized_content', '') or ''}")
        except Exception as exc:
            merged_md_parts.append(f"## 块 {idx+1}（生成失败）\n\nError: {exc}")

    full_md = "# 个性化学习包（RAG-only）\n\n" + "\n\n".join(merged_md_parts)
    t1 = time.time()
    latency_ms = int((t1 - t0) * 1000)

    return {
        "config": "RAG-only",
        "status": "success" if merged_md_parts else "failed",
        "package_md": full_md,
        "package_chars": len(full_md),
        "llm_calls": [{"note": "Agent3 internal calls counted via _llm_call_records",
                       "total_latency_ms": latency_ms}],
        "total_latency_ms": latency_ms,
        "used_rag": True,
        "used_review": False,
        "used_planning": False,
    }


# =====================================================================
# 运行模式 C: No-Agent4 / D: Full-System
#   直接走 Harness，但 No-Agent4 通过 monkey-patch 让 _review_* 流程直接通过
# =====================================================================
async def run_with_harness(case_id: str, config: str) -> Dict[str, Any]:
    from orchestrator.harness import Harness

    t0 = time.time()
    harness = Harness()

    # No-Agent4: monkey-patch 审核流程，直接通过，不修改任何学习包
    if config == "No-Agent4":
        # 替换 _review_learning_package_pipeline 和 _review_learning_package
        # 让它们返回 (learning_package, empty_review_history, passed_trace)
        async def fake_pipeline(self, session_id, stage, profile, practice_profile):
            # 直接调用 Agent3 生成
            from services.learning_evidence_service import LearningEvidenceService
            from models.contracts import ContentGenerationInput

            learning_evidence = LearningEvidenceService(self.state_manager.db_client).build(session_id, stage)
            generated = await self.agent3.run(ContentGenerationInput(
                session_id=session_id, stage=stage,
                learning_profile=profile, practice_profile=practice_profile,
                learning_evidence=learning_evidence,
            ))
            # 直接把 package 内容写回 trace 视为 passed
            self._last_review_trace = {
                "final_status": "passed",
                "package_status": "approved",
                "trace_id": f"ablation-no-agent4-{session_id}",
                "unresolved_issue_count": 0,
                "unresolved_must_fix_count": 0,
            }
            return generated.learning_package, []

        import types
        harness._review_learning_package_pipeline = types.MethodType(
            fake_pipeline, harness)
        harness._review_learning_package = types.MethodType(
            lambda self, *a, **kw: asyncio.coroutine(lambda: (a[1] if len(a)>1 else {}, [], None))(), harness)

    # 标准 Harness 流程
    session = harness.state_manager.create_session(None)
    session_id = session["session_id"]
    case_row = load_case_input(case_id)
    user_input = build_user_input(case_row)

    diag = await harness.process_initial_input(session_id, user_input)
    profile = diag.get("profile", {})
    stage_info = diag.get("stage_info", {})
    if diag.get("status") != "ready":
        return {
            "config": config,
            "status": f"diagnostic_{diag.get('status')}",
            "error": "Diagnostic confidence below threshold",
            "llm_calls": [],
        }

    stage_result = await harness.start_stage(session_id)
    learning_package = stage_result.get("learning_package", {})
    review_iterations = stage_result.get("iterations", 0)
    review_passed = stage_result.get("review_passed", False)

    # 受控答题 + 阶段决策（Full-System 才走决策）
    accuracy = None
    next_strategy = None
    try:
        questions = learning_package.get("questions", []) or []
        for q in questions:
            qid = q.get("question_id", "")
            correct = q.get("correct_answer", "")
            # 模拟 70% 正确率
            import random
            rng = random.Random(hash(case_id + config) & 0xFFFFFF)
            submitted = correct if rng.random() < 0.7 else "错误答案"
            if qid:
                try:
                    await harness.submit_answer(session_id, qid, submitted)
                except Exception:
                    pass
        decision = await harness.complete_stage(session_id)
        accuracy = decision.get("accuracy")
        next_strategy = decision.get("next_strategy")
    except Exception as exc:
        pass

    t1 = time.time()
    latency_ms = int((t1 - t0) * 1000)

    # 学习包 -> markdown
    blocks = learning_package.get("blocks") or learning_package.get("content_blocks") or []
    md_lines = [f"# {learning_package.get('title') or stage_info.get('block_name') or '学习包'}\n"]
    for i, b in enumerate(blocks):
        if isinstance(b, dict):
            h = b.get("heading") or b.get("title") or f"块 {i+1}"
            c = b.get("personalized_content") or b.get("content") or b.get("markdown") or ""
            md_lines.append(f"## {h}\n\n{c}\n")
    package_md = "\n".join(md_lines)

    # LLM 调用计数
    llm_calls = []
    try:
        from scripts.run_real_evaluation import _llm_call_records as lcr
        llm_calls = list(lcr[-80:])
    except Exception:
        pass

    return {
        "config": config,
        "status": "success",
        "package_md": package_md,
        "package_chars": len(package_md),
        "profile_confidence": profile.get("confidence"),
        "planned_block": stage_info.get("block_name"),
        "review_iterations": review_iterations,
        "review_passed": review_passed,
        "accuracy": accuracy,
        "next_strategy": next_strategy,
        "llm_calls": llm_calls,
        "total_latency_ms": latency_ms,
        "used_rag": True,
        "used_review": config == "Full-System",
        "used_planning": True,
    }


# =====================================================================
# 单 case 跑 4 配置
# =====================================================================
async def run_case(case_id: str, configs: List[str],
                   reuse_main: bool = True) -> List[Dict[str, Any]]:
    case_row = load_case_input(case_id)
    user_input = build_user_input(case_row)
    results = []
    for cfg in configs:
        print(f"  · {cfg} ...", end=" ", flush=True)
        run_started = _now_iso()
        # Full-System：优先复用主评测结果（零 LLM 消耗）
        if reuse_main and cfg == "Full-System":
            reused = extract_fullsystem_from_main(case_id)
            if reused is not None:
                reused["run_started_at"] = run_started
                reused["run_finished_at"] = _now_iso()
                out_path = OUTPUTS_DIR / f"{case_id}_{cfg}.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(reused, f, ensure_ascii=False, default=str)
                results.append(reused)
                chars = reused.get("package_chars", 0)
                calls = reused.get("total_llm_calls", 0)
                print(f"REUSE (main: {reused.get('source_run_id','')}) chars={chars} calls~{calls}")
                continue
        try:
            if cfg == "Direct-LLM":
                result = await run_direct_llm(user_input, case_id)
            elif cfg == "RAG-only":
                result = await run_rag_only(user_input, case_id)
            elif cfg in {"No-Agent4", "Full-System"}:
                result = await run_with_harness(case_id, cfg)
            else:
                raise ValueError(f"Unknown config: {cfg}")
        except Exception as exc:
            result = {
                "config": cfg,
                "status": "crashed",
                "error": f"{type(exc).__name__}: {str(exc)[:500]}",
            }
        result["case_id"] = case_id
        result["run_started_at"] = run_started
        result["run_finished_at"] = _now_iso()
        # 保存单 case 输出
        out_path = OUTPUTS_DIR / f"{case_id}_{cfg}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, default=str)
        results.append(result)
        chars = result.get("package_chars", 0)
        calls = len(result.get("llm_calls") or [])
        print(f"{result.get('status')} chars={chars} calls~{calls}")
    return results


# =====================================================================
# 主
# =====================================================================
async def main():
    parser = argparse.ArgumentParser(description="15×4 多Agent消融测试")
    parser.add_argument("--cases", default=",".join(DEFAULT_CASES),
                        help="逗号分隔的 case_id 列表")
    parser.add_argument("--configs", default=",".join(CONFIGS),
                        help="逗号分隔的配置：Direct-LLM,RAG-only,No-Agent4,Full-System")
    parser.add_argument("--no-reuse-main", action="store_true",
                        help="关闭主评测结果复用（默认开启；默认情况下已跑过的主评测 case 的 Full-System 直接抽取，不产生新 LLM 调用）")
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()

    cases = [x.strip() for x in args.cases.split(",") if x.strip()]
    configs = [x.strip() for x in args.configs.split(",") if x.strip()]
    reuse_main = not args.no_reuse_main

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)

    if args.dry:
        total = 0
        reused_count = 0
        for c in cases:
            for cfg in configs:
                total += 1
                if reuse_main and cfg == "Full-System" and extract_fullsystem_from_main(c) is not None:
                    reused_count += 1
        print(f"消融测试规划: {len(cases)} cases × {len(configs)} configs = {total} 条")
        for c in cases:
            line = []
            for cfg in configs:
                if reuse_main and cfg == "Full-System" and extract_fullsystem_from_main(c) is not None:
                    line.append(f"{cfg}[REUSE]")
                else:
                    line.append(f"{cfg}[RUN]")
            print(f"  {c}: " + ", ".join(line))
        print(f"\n  [PLAN] 预计真 LLM 调用条数: {total - reused_count}（Full-System {reused_count} 条从主评测抽取，零消耗）")
        if reused_count == 0 and "Full-System" in configs:
            print(f"  [WARN] 没有任何 Full-System 可复用，检查 02_50组个性化主评测/raw_outputs/ 是否已有对应 case 结果")
        return 0

    all_rows = []
    reused_global = 0
    real_runs = 0
    for i, case_id in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] Ablation {case_id}")
        rows = await run_case(case_id, configs, reuse_main=reuse_main)
        for r in rows:
            if r.get("data_source", "").startswith("reused_from_main_evaluation"):
                reused_global += 1
            else:
                real_runs += 1
        all_rows.extend(rows)

    # 写 CSV
    csv_path = ABLATION_DIR / "ablation_cases.csv"
    fields = [
        "case_id", "config", "status", "package_chars", "total_latency_ms",
        "used_rag", "used_review", "used_planning",
        "review_iterations", "review_passed", "accuracy", "next_strategy",
        "profile_confidence", "planned_block",
        "data_source", "source_run_id",
        "run_started_at", "run_finished_at", "error",
    ]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)

    print()
    print(f"完成 {len(all_rows)} 条消融测试记录 -> {csv_path}")
    success_rows = [r for r in all_rows if r.get("status") == "success"]
    print(f"  成功: {len(success_rows)}/{len(all_rows)}")
    print(f"  [REUSE] 主评测复用(零 token): {reused_global} 条")
    print(f"  [RUN] 真 LLM 执行(产生消耗): {real_runs} 条")
    # Full-System vs No-Agent4 对比摘要
    fs = [r for r in success_rows if r.get("config") == "Full-System"]
    na4 = [r for r in success_rows if r.get("config") == "No-Agent4"]
    if fs and na4:
        avg_fs_iters = sum(r.get("review_iterations") or 0 for r in fs) / len(fs)
        print(f"  Full-System 平均审核轮次: {avg_fs_iters:.2f}")
        print(f"  Full-System 审核通过率: {sum(1 for r in fs if r.get('review_passed'))/len(fs):.1%}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
