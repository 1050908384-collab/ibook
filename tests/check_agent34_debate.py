# -*- coding: utf-8 -*-
"""
Agent3 ↔ Agent4 辩论链路独立测试 —— 完整审核/答辩/复核/裁决/修订循环

复刻 backend/orchestrator/harness.py::_review_learning_package 的完整辩论逻辑，
但不实例化 Harness/StateManager（避免 DB 依赖），Orchestrator 裁决用本地精简函数实现。

一轮辩论 = 5 步（对齐审核辩论协议第 4~5 节）：
  1. Agent4.review              -> issues（含 canonical_error_id / must_fix）
  2. Agent3.reply_to_review     -> author_replies（accepted / partially_accepted / rejected）
  3. Agent4.rebut               -> verdicts（persist / withdraw / downgrade）
  4. adjudicate（本地裁决）     -> accepted_issues + unresolved_must_fix_count
  5. Agent3.modify_package      -> 修订后 package -> 进下一轮

终止条件（对齐 harness.py 与协议第 7 节）：
  - 某轮 Agent4.review.passed=true                     -> passed
  - 某轮 unresolved_must_fix_count=0                   -> passed
  - 达到 MAX_REVIEW_ITERATIONS 仍有未决 must_fix       -> needs_human_review

独立运行：
  python tests/check_agent34_debate.py
"""
import asyncio
import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, ROOT_DIR)

from config import settings
from agents.deterministic import (
    DeterministicContentAgent,
    DeterministicTechnicalReviewAgent,
    HybridContentAgent,
    HybridTechnicalReviewAgent,
)
from models.contracts import ReviewInput
from models.profile import UserProfile
from models.review import ReviewError
from models.stage import Claim, LearningPackage, Source, StageInfo


def pretty(label: str, obj) -> None:
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump(mode="json")
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def choose_mode() -> str:
    print("\n选择执行模式：")
    print("  1) hybrid        Agent3/4 使用本地 LLM（Ollama），查看真实辩论（默认）")
    print("  2) deterministic Agent3/4 纯规则/工具链路，不调 LLM，速度快")
    choice = input("请输入 1 或 2（回车=1）: ").strip()
    return "deterministic" if choice == "2" else "hybrid"


def adjudicate_issues(
    errors: list[ReviewError],
    verdicts: dict[str, str],
) -> tuple[list[dict], int]:
    """Orchestrator 裁决（复刻 harness.py::_adjudicate_issues，去掉 DB 落库）。

    返回 (accepted_issues, unresolved_must_fix_count)：
    - withdraw  -> accepted_issues（Agent3 接受处理）
    - downgrade -> 降级 severity + 解除 must_fix，进 accepted_issues
    - persist   -> must_fix=true 强制 Agent3 修改并计入未决；must_fix=false 跳过
    """
    accepted: list[dict] = []
    unresolved_must_fix_count = 0
    for err in errors:
        cid = err.canonical_error_id or ""
        verdict = verdicts.get(cid, "persist")
        if verdict == "withdraw":
            accepted.append(err.model_dump(mode="json"))
        elif verdict == "downgrade":
            err.severity = "minor" if err.severity == "major" else err.severity
            if err.must_fix:
                err.must_fix = False
                err.must_fix_reason = "Agent4 复核降级，解除 must_fix"
            accepted.append(err.model_dump(mode="json"))
        else:  # persist
            if err.must_fix:
                accepted.append(err.model_dump(mode="json"))
                unresolved_must_fix_count += 1
    return accepted, unresolved_must_fix_count


# 默认学习包：故意包含事实错误，用于触发 Agent4 审核 → 辩论
DEFAULT_TUTORIAL = (
    "# LoRA 微调\n\n"
    "LoRA 的核心原理是对原始权重 W 执行 SVD 分解，再训练 U、Σ、V 三个矩阵。\n"
    "若 W 维度为 d_out×d_in，LoRA 可训练参数量为 d_out*d_in。\n"
    "LoRA 必须冻结原权重，只训练低秩增量 ΔW=(α/r)BA。\n"
    "QLoRA 是 LoRA 的扩展，必须使用 4-bit 量化。\n"
)


def build_default_package(session_id: str, stage: StageInfo) -> LearningPackage:
    sources = [
        Source(
            source_id="src_kb12_0",
            source_type="knowledge_doc",
            title="LoRA 原理文档",
            locator="doc_lora_01",
        ),
        Source(
            source_id="src_kb12_1",
            source_type="knowledge_doc",
            title="QLoRA 量化文档",
            locator="doc_qlora_01",
        ),
    ]
    claims = [
        Claim(
            claim_id="clm_001",
            text="LoRA 冻结原权重，训练低秩增量 ΔW=(α/r)BA",
            source_ref="src_kb12_0",
            section_ref="tutorial:sec_1",
            claim_type="concept",
            is_key_technical=True,
            confidence=0.85,
        ),
        Claim(
            claim_id="clm_002",
            text="LoRA 对原始权重 W 执行 SVD 分解",  # 事实错误，应被 Agent4 抓出
            source_ref=None,
            section_ref="tutorial:sec_1",
            claim_type="fact",
            is_key_technical=True,
            confidence=0.4,
        ),
        Claim(
            claim_id="clm_003",
            text="QLoRA 必须使用 4-bit 量化",
            source_ref="src_kb12_1",
            section_ref="tutorial:sec_3",
            claim_type="fact",
            is_key_technical=True,
            confidence=0.8,
        ),
    ]
    return LearningPackage(
        session_id=session_id,
        stage_id=stage.stage_id,
        block_id=stage.block_id,
        block_name=stage.block_name,
        tutorial=DEFAULT_TUTORIAL,
        sources=sources,
        claims=claims,
        version=1,
    )


async def run_debate():
    print("=" * 60)
    print("Agent3 ↔ Agent4 辩论链路独立测试")
    print(f"（最多 {settings.MAX_REVIEW_ITERATIONS} 轮，复刻 harness._review_learning_package）")
    print("=" * 60)

    mode = choose_mode()
    if mode == "hybrid":
        agent3 = HybridContentAgent()
        agent4 = HybridTechnicalReviewAgent()
    else:
        agent3 = DeterministicContentAgent()
        agent4 = DeterministicTechnicalReviewAgent()

    session_id = "test-agent34-debate"
    stage = StageInfo(
        stage_id="stage-test-34", stage_num=1, block_id="KB-12",
        block_name="参数高效微调（LoRA与QLoRA）", gap_skill="LoRA",
    )
    profile = UserProfile(session_id=session_id, goal="掌握LoRA微调")

    print("\n请输入待审核的 tutorial 内容（直接回车使用默认示例，默认故意包含事实错误以触发辩论）：")
    print("-" * 60)
    print(DEFAULT_TUTORIAL)
    print("-" * 60)
    custom = input(">>> ").strip()
    package = build_default_package(session_id, stage)
    if custom:
        package.tutorial = custom

    pretty("【初始输入】LearningPackage", {
        "block_name": package.block_name,
        "version": package.version,
        "tutorial_length": len(package.tutorial),
        "sources_count": len(package.sources),
        "claims_count": len(package.claims),
        "claims": [c.model_dump(mode="json") for c in package.claims],
    })

    history: list[dict] = []
    final_status = "in_progress"
    package_status = "draft"

    for iteration in range(1, settings.MAX_REVIEW_ITERATIONS + 1):
        print(f"\n{'#' * 60}\n# 辩论第 {iteration} 轮（最多 {settings.MAX_REVIEW_ITERATIONS} 轮）\n{'#' * 60}")

        # 步骤1：Agent4 审核
        print(f"\n[Step 1] Agent4.review（round_index={iteration}）...")
        review_request = ReviewInput(
            session_id=session_id,
            target_type="learning_package",
            content=package.tutorial,
            stage=stage,
            profile=profile,
            sources=[s.model_dump(mode="json") for s in package.sources],
            claims=[c.model_dump(mode="json") for c in package.claims],
            round_index=iteration,
        )
        review_result = await agent4.run(review_request)
        pretty(f"【Agent4 输出】第{iteration}轮 issues（共{len(review_result.errors)}条）", {
            "passed": review_result.passed,
            "errors": [e.model_dump(mode="json") for e in review_result.errors],
        })

        if review_result.passed:
            final_status = "passed"
            package_status = "approved"
            print(f"\n=== 第{iteration}轮 Agent4 直接通过，辩论结束 ===")
            history.append({"iteration": iteration, "passed": True, "errors": 0})
            break

        # 步骤2：Agent3 答辩
        print(f"\n[Step 2] Agent3.reply_to_review（对每条 issue 逐条回复）...")
        issues = [e.model_dump(mode="json") for e in review_result.errors]
        package_dict = package.model_dump(mode="json")
        author_replies = await agent3.reply_to_review(session_id, issues, package_dict)
        pretty(f"【Agent3 输出】author_replies（共{len(author_replies)}条）", author_replies)

        # 步骤3：Agent4 复核裁决
        print(f"\n[Step 3] Agent4.rebut（对 Agent3 答辩做 persist/withdraw/downgrade）...")
        verdicts = await agent4.rebut(session_id, issues, author_replies)
        pretty("【Agent4 输出】verdicts（persist=坚持 / withdraw=撤销 / downgrade=降级）", verdicts)

        # 步骤4：Orchestrator 裁决（本地精简版）
        print(f"\n[Step 4] Orchestrator 裁决（adjudicate_issues）...")
        accepted_issues, unresolved_must_fix_count = adjudicate_issues(
            review_result.errors, verdicts
        )
        pretty("【Orchestrator 输出】", {
            "accepted_issues_count": len(accepted_issues),
            "unresolved_must_fix_count": unresolved_must_fix_count,
            "accepted_issues": accepted_issues,
        })

        # 步骤5：Agent3 修改学习包
        if accepted_issues:
            print(f"\n[Step 5] Agent3.modify_package（处理 {len(accepted_issues)} 条已裁决 issue）...")
            revised_dict = await agent3.modify_package(
                session_id,
                package_dict,
                {"tech": review_result.errors, "quality": []},
                accepted_issues,
            )
            package = LearningPackage.model_validate(revised_dict)
            pretty(f"【Agent3 输出】修订后 package（version={package.version}）", {
                "version": package.version,
                "tutorial_length": len(package.tutorial),
                "claims_status": [
                    {"claim_id": c.claim_id, "status": c.status}
                    for c in package.claims
                ],
                "tutorial_tail": package.tutorial[-300:],
            })
        else:
            print(f"\n[Step 5] 无 accepted_issues，跳过 modify_package")

        history.append({
            "iteration": iteration,
            "passed": False,
            "errors": len(review_result.errors),
            "unresolved_must_fix": unresolved_must_fix_count,
        })

        # 终止判定
        if unresolved_must_fix_count == 0:
            final_status = "passed"
            package_status = "approved"
            print(f"\n=== 第{iteration}轮辩论后通过（无未决 must_fix）===")
            break
        if iteration == settings.MAX_REVIEW_ITERATIONS:
            final_status = "needs_human_review"
            package_status = "provisional"
            print(
                f"\n=== 达到最大轮次仍有 {unresolved_must_fix_count} 个 must_fix 未决，"
                f"标记为 needs_human_review（package 降级为 provisional）==="
            )
            break
        print(f"\n=== 第{iteration}轮仍有 {unresolved_must_fix_count} 个 must_fix 未决，进入下一轮 ===")

    # 最终总结
    pretty("【辩论最终结果】", {
        "final_status": final_status,
        "package_status": package_status,
        "total_rounds": len(history),
        "history": history,
    })
    print("\n状态说明：")
    print("  passed              Agent4 通过或所有 must_fix 已解决")
    print("  needs_human_review  达到最大轮次仍有 must_fix 未决（非阻塞，package 降级为 provisional）")


if __name__ == "__main__":
    try:
        asyncio.run(run_debate())
    except KeyboardInterrupt:
        print("\n已中断。")
    except Exception:
        import traceback
        traceback.print_exc()
