# -*- coding: utf-8 -*-
"""
Agent3 独立测试 —— 内容生成 Agent（学习包 + 答疑）

职责（对齐 backend/agents/deterministic.py）：
  - 依据 StageInfo 生成 LearningPackage（tutorial / code_examples / questions / sources / claims）
  - 依据用户问题生成回答（generate_answer）

独立运行：
  python tests/check_agent3_content.py

说明：
  - hybrid 模式：先走确定性生成，再用本地 LLM 增强 tutorial / answer（与 Harness 实际行为一致）。
  - deterministic 模式：纯工具链路（向量检索 + FAQ），不调 LLM。
"""
import asyncio
import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, ROOT_DIR)

from agents.deterministic import DeterministicContentAgent, HybridContentAgent
from models.contracts import AnswerGenerationInput, ContentGenerationInput
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


async def main():
    print("=" * 60)
    print("Agent3 内容生成 Agent 独立测试（学习包 + 答疑）")
    print("=" * 60)

    mode = choose_mode()
    agent = HybridContentAgent() if mode == "hybrid" else DeterministicContentAgent()

    # 构造 StageInfo（用户可覆盖）
    print("\n请输入阶段信息（直接回车使用默认）：")
    block_name = input("block_name（回车=参数高效微调（LoRA与QLoRA））: ").strip() or "参数高效微调（LoRA与QLoRA）"
    block_id = input("block_id（回车=KB-12）: ").strip() or "KB-12"
    gap_skill = input("gap_skill（回车=LoRA）: ").strip() or "LoRA"

    session_id = "test-agent3-session"
    stage = StageInfo(
        stage_id="stage-test-03",
        stage_num=1,
        block_id=block_id,
        block_name=block_name,
        gap_skill=gap_skill,
        estimated_hours=2.0,
        prerequisites=[],
    )
    profile = UserProfile(session_id=session_id, goal=f"掌握{gap_skill}")
    request = ContentGenerationInput(session_id=session_id, stage=stage)

    pretty("【输入】ContentGenerationInput", {
        "session_id": session_id,
        "stage": stage.model_dump(mode="json"),
    })

    # Step 1: 内容生成 -> LearningPackage
    print("\n[Step 1] 运行 run() 生成 LearningPackage ...")
    output = await agent.run(request)
    pkg = output.learning_package
    pretty("【输出】LearningPackage 元信息", {
        "block_name": pkg.block_name,
        "block_id": pkg.block_id,
        "version": pkg.version,
        "tutorial_length": len(pkg.tutorial),
        "questions_count": len(pkg.questions),
        "sources_count": len(pkg.sources),
        "claims_count": len(pkg.claims),
        "review_trace_id": pkg.review_trace_id,
    })
    print("\n--- tutorial ---")
    print(pkg.tutorial)
    print("\n--- questions ---")
    print(json.dumps([q.model_dump(mode="json") for q in pkg.questions], ensure_ascii=False, indent=2, default=str))
    print("\n--- sources ---")
    print(json.dumps([s.model_dump(mode="json") for s in pkg.sources], ensure_ascii=False, indent=2, default=str))
    print("\n--- claims ---")
    print(json.dumps([c.model_dump(mode="json") for c in pkg.claims], ensure_ascii=False, indent=2, default=str))

    # Step 2: 答疑
    ans = input("\n是否继续测试答疑 generate_answer？(y/N): ").strip().lower()
    if ans != "y":
        print("已结束。")
        return
    question = input("请输入问题（回车=什么是LoRA？它是如何工作的？）: ").strip() or "什么是LoRA？它是如何工作的？"
    answer_request = AnswerGenerationInput(
        session_id=session_id, question=question, stage=stage, profile=profile
    )
    pretty("【输入】AnswerGenerationInput", {
        "question": question,
        "stage.block_name": stage.block_name,
        "profile.goal": profile.goal,
    })
    print("\n[Step 2] 运行 generate_answer() ...")
    answer_output = await agent.generate_answer(answer_request)
    print("\n--- answer ---")
    print(answer_output.answer)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n已中断。")
    except Exception:
        import traceback
        traceback.print_exc()
