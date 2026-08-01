# -*- coding: utf-8 -*-
"""
Agent4 独立测试 —— 技术+质量审核 Agent（review + rebut 辩论协议）

职责（对齐 backend/agents/deterministic.py）：
  - 审核 learning_package / answer -> TechnicalReviewResult
    每个 error 强制携带：canonical_error_id / error_id / source_refs / change_request /
    must_fix / must_fix_reason / claim_id（审核辩论协议 2.3 / 2.4 / 第 6 节）
  - rebut 复核：对 Agent3 的逐条回复做裁决 -> {canonical_error_id: persist|withdraw|downgrade}

独立运行：
  python tests/check_agent4_review.py

说明：
  - 默认示例内容「故意包含事实错误」（LoRA 不是 SVD 分解；参数量算错），
    用于触发规则检查 / 幻觉检测，直观看到审核输出。
  - rebut 步骤会自动按 Agent3 风格构造 author_replies（must_fix->partially_accepted，其余->accepted），
    无需手写即可观察 persist/withdraw/downgrade 裁决。
"""
import asyncio
import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, ROOT_DIR)

from agents.deterministic import DeterministicTechnicalReviewAgent, HybridTechnicalReviewAgent
from models.contracts import ReviewInput
from models.profile import UserProfile
from models.stage import StageInfo


def pretty(label: str, obj) -> None:
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump(mode="json")
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def choose_mode() -> str:
    print("\n选择执行模式：")
    print("  1) hybrid        使用本地 LLM（Ollama），查看真实模型输出（默认）")
    print("  2) deterministic 纯规则/工具链路，不调用 LLM，速度快")
    choice = input("请输入 1 或 2（回车=1）: ").strip()
    return "deterministic" if choice == "2" else "hybrid"


DEFAULT_CONTENT_LP = (
    "# LoRA 微调\n\n"
    "LoRA 的核心原理是对原始权重 W 执行 SVD 分解，再训练 U、Σ、V 三个矩阵。\n"
    "若 W 维度为 d_out×d_in，LoRA 可训练参数量为 d_out*d_in。\n"
    "LoRA 必须冻结原权重，只训练低秩增量 ΔW=(α/r)BA。\n"
)
DEFAULT_CONTENT_ANS = "LoRA 通过对权重矩阵做 SVD 分解来减少参数量，训练 U、Σ、V 三个矩阵。"


async def main():
    print("=" * 60)
    print("Agent4 技术+质量审核 Agent 独立测试（review + rebut）")
    print("=" * 60)

    mode = choose_mode()
    agent = HybridTechnicalReviewAgent() if mode == "hybrid" else DeterministicTechnicalReviewAgent()

    # 选择审核目标类型
    print("\n选择审核目标类型：")
    print("  1) learning_package  审核学习包 tutorial（默认）")
    print("  2) answer            审核问答回答")
    t = input("请输入 1 或 2（回车=1）: ").strip()
    target_type = "answer" if t == "2" else "learning_package"

    default_content = DEFAULT_CONTENT_LP if target_type == "learning_package" else DEFAULT_CONTENT_ANS
    print(f"\n请输入待审核内容（直接回车使用默认示例，默认故意包含事实错误以触发审核）：")
    print("-" * 60)
    print(default_content)
    print("-" * 60)
    content = input(">>> ").strip()
    if not content:
        content = default_content

    session_id = "test-agent4-session"
    stage = StageInfo(
        stage_id="stage-test-04", stage_num=1, block_id="KB-12",
        block_name="参数高效微调（LoRA与QLoRA）", gap_skill="LoRA",
    )
    profile = UserProfile(session_id=session_id, goal="掌握LoRA微调")

    # 构造 sources / claims，让辩论协议字段 enrichment 生效
    sources = [{
        "source_id": "src_kb12_0",
        "source_type": "knowledge_doc",
        "title": "LoRA 原理文档",
        "locator": "doc_lora_01",
    }]
    claims = [{
        "claim_id": "clm_001",
        "text": "LoRA 冻结原权重，训练低秩增量 ΔW=(α/r)BA",
        "source_ref": "src_kb12_0",
        "section_ref": "tutorial:sec_0",
        "claim_type": "concept",
        "is_key_technical": True,
        "confidence": 0.85,
        "status": "active",
    }]

    request = ReviewInput(
        session_id=session_id,
        target_type=target_type,
        content=content,
        stage=stage,
        profile=profile,
        sources=sources,
        claims=claims,
        round_index=1,
    )
    pretty("【输入】ReviewInput", {
        "target_type": target_type,
        "content": content,
        "sources": sources,
        "claims": claims,
        "round_index": 1,
    })

    # Step 1: 审核
    print("\n[Step 1] 运行 run() -> TechnicalReviewResult ...")
    result = await agent.run(request)
    pretty("【输出】TechnicalReviewResult 概要", {
        "passed": result.passed,
        "errors_count": len(result.errors),
        "review_details": result.review_details,
    })
    print("\n--- errors（含辩论协议字段 canonical_error_id / must_fix / change_request / source_refs）---")
    print(json.dumps([e.model_dump(mode="json") for e in result.errors], ensure_ascii=False, indent=2, default=str))

    # Step 2: rebut 复核
    if not result.errors:
        print("\n无 errors，跳过 rebut 测试。")
        return
    ans = input("\n是否继续测试 rebut 复核裁决？(y/N): ").strip().lower()
    if ans != "y":
        return

    # 自动构造 Agent3 风格 author_replies（must_fix->partially_accepted，其余->accepted）
    issues = [e.model_dump(mode="json") for e in result.errors]
    author_replies = []
    for iss in issues:
        if iss.get("must_fix"):
            disposition = "partially_accepted"
            refs = iss.get("source_refs") or ["src_kb12_0"]
        else:
            disposition = "accepted"
            refs = iss.get("source_refs") or []
        author_replies.append({
            "canonical_error_id": iss.get("canonical_error_id"),
            "disposition": disposition,
            "reason": f"测试回复：{iss.get('description', '')}",
            "source_refs": refs,
            "patched_sections": [iss.get("location") or "tutorial:sec_0"],
        })
    pretty("【输入】rebut 的 author_replies（自动按 Agent3 风格构造）", author_replies)
    print("\n[Step 2] 运行 rebut() 复核裁决 ...")
    verdicts = await agent.rebut(session_id, issues, author_replies)
    pretty("【输出】rebut verdicts（persist=坚持 / withdraw=撤销 / downgrade=降级）", verdicts)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n已中断。")
    except Exception:
        import traceback
        traceback.print_exc()
