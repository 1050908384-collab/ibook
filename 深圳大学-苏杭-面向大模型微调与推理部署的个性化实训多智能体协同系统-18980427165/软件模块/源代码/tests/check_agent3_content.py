# -*- coding: utf-8 -*-
"""
个性化学习资源生成 Agent 独立测试 —— 内容生成 Agent（学习包 + 答疑）

职责（对齐 backend/agents/deterministic.py）：
  - 依据 StageInfo 生成 LearningPackage（tutorial / code_examples / questions / sources / claims）
  - 依据用户问题生成回答（generate_answer）

独立运行：
  python tests/check_agent3_content.py

输出内容：
  1. Agent 配置（模式、工具列表）
  2. 输入（ContentGenerationInput: session_id + StageInfo）
  3. 运行中主要参数（向量检索结果、题目提取结果 — 直接调用工具预览中间数据）
  4. 个性化学习资源生成 Agent run() 输出（LearningPackage: tutorial / questions / sources / claims）
  5. 答疑中间参数（知识文档向量检索）
  6. 答疑输出（answer）

说明：
  - hybrid 模式：先走确定性生成，再用本地 LLM 增强 tutorial / answer（与 Harness 实际行为一致）。
  - deterministic 模式：纯工具链路（知识文档向量检索），不调 LLM。
"""
import asyncio
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

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


def section(title: str) -> None:
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


def show_agent_config(agent) -> None:
    """展示 Agent 的工具列表和配置。"""
    section("Agent 配置")
    agent_type = type(agent).__name__
    print(f"  agent_type:    {agent_type}")
    print(f"  has_llm:       {hasattr(agent, 'llm_client')}")
    tools = []
    for attr in sorted(vars(agent)):
        if attr.startswith("_"):
            continue
        val = getattr(agent, attr)
        tools.append(f"    .{attr}  →  {type(val).__name__}")
    print(f"  tools/attrs ({len(tools)}):")
    for line in tools:
        print(line)


async def show_runtime_params_content(agent, stage: StageInfo) -> dict:
    """直接调用 个性化学习资源生成 Agent 的工具，预览内容生成运行中的主要参数。"""
    section("运行中主要参数（个性化学习资源生成 Agent 内部工具调用预览）")
    params = {}

    # 1. 向量检索（按 block_id 检索相关文档）
    print(f"\n  [工具] vector_search_tool.search_by_block(block_id='{stage.block_id}')")
    try:
        related_docs = await agent.vector_search_tool.search_by_block(stage.block_id)
        params["related_docs"] = related_docs
        print(f"    → 检索到 {len(related_docs)} 篇文档")
        for i, doc in enumerate(related_docs[:3]):
            content_preview = (doc.get("content", "") or "")[:100]
            print(f"      [{i}] doc_id={doc.get('doc_id', '?')}, title={doc.get('title', '?')}, chunk_id={doc.get('chunk_id', '?')}")
            print(f"          content: {content_preview}...")
        if len(related_docs) > 3:
            print(f"      ... 共 {len(related_docs)} 篇")
    except Exception as e:
        params["related_docs"] = f"[ERROR] {e}"
        print(f"    → [ERROR] {e}")

    # 2. 题目提取
    print(f"\n  [工具] question_extract_tool.extract_questions(tutorial, config)")
    try:
        sample_tutorial = f"# {stage.block_name}\n\n核心概念与学习步骤。"
        extracted_questions = await agent.question_extract_tool.extract_questions(
            sample_tutorial,
            {"block_id": stage.block_id, "block_name": stage.block_name, "question_count": 2},
        )
        params["extracted_questions"] = extracted_questions
        print(f"    → 提取到 {len(extracted_questions)} 道题")
        for i, q in enumerate(extracted_questions):
            if isinstance(q, dict):
                print(f"      [{i}] type={q.get('question_type', '?')}, text={q.get('question_text', '?')[:60]}")
    except Exception as e:
        params["extracted_questions"] = f"[ERROR] {e}"
        print(f"    → [ERROR] {e}")

    # 3. 视频资源检索
    print(f"\n  [工具] video_search_tool.search_by_block(block_id='{stage.block_id}', block_name='{stage.block_name}')")
    try:
        video_results = await agent.video_search_tool.search_by_block(
            block_id=stage.block_id, block_name=stage.block_name, top_k=5
        )
        params["video_results"] = video_results
        print(f"    → 检索到 {len(video_results)} 个视频")
        for i, v in enumerate(video_results[:5]):
            vid = v.get("video_id", "?")
            tp = v.get("topic", "?")
            tt = v.get("title", "?")[:60]
            url = v.get("url", "?")[:80]
            dur = v.get("duration_hours", 0)
            print(f"      [{i}] {vid} | {tp} ({dur}h)")
            print(f"          {tt}")
            print(f"          url: {url}")
    except Exception as e:
        params["video_results"] = f"[ERROR] {e}"
        print(f"    → [ERROR] {e}")

    return params


async def show_runtime_params_answer(agent, question: str, stage: StageInfo) -> dict:
    """答疑的中间参数。"""
    section("答疑中间参数（个性化学习资源生成 Agent.generate_answer 内部工具调用预览）")
    params = {}

    # 知识文档向量检索（按问题检索）
    print(f"\n  [工具] vector_search_tool.search_by_query(question, top_k=3)")
    try:
        query_docs = await agent.vector_search_tool.search_by_query(question, top_k=3)
        params["query_docs"] = query_docs
        print(f"    → 检索到 {len(query_docs)} 篇相关文档")
        for i, doc in enumerate(query_docs[:3]):
            content_preview = (doc.get("content", "") or "")[:100]
            print(f"      [{i}] doc_id={doc.get('doc_id', '?')}, content: {content_preview}...")
    except Exception as e:
        params["query_docs"] = f"[ERROR] {e}"
        print(f"    → [ERROR] {e}")

    return params


def choose_mode() -> str:
    print("\n选择执行模式：")
    print("  1) hybrid        使用本地 LLM（Ollama），查看真实模型输出（默认）")
    print("  2) deterministic 纯规则/工具链路，不调用 LLM，速度快")
    choice = input("请输入 1 或 2（回车=1）: ").strip()
    return "deterministic" if choice == "2" else "hybrid"


async def main():
    print("=" * 60)
    print("个性化学习资源生成 Agent 内容生成 Agent 独立测试（学习包 + 答疑）")
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

    # ========== 1. Agent 配置 ==========
    show_agent_config(agent)

    # ========== 2. 输入 ==========
    pretty("【输入】ContentGenerationInput", {
        "session_id": session_id,
        "stage": stage.model_dump(mode="json"),
    })

    # ========== 3. 运行中主要参数 ==========
    runtime_params = await show_runtime_params_content(agent, stage)
    pretty("【运行参数汇总】", runtime_params)

    # ========== 4. 个性化学习资源生成 Agent run() 输出 ==========
    section("[Step 1] 运行 run() 生成 LearningPackage")
    output = await agent.run(request)
    pkg = output.learning_package
    pretty("【输出】LearningPackage 元信息", {
        "block_name": pkg.block_name,
        "block_id": pkg.block_id,
        "version": pkg.version,
        "review_trace_id": pkg.review_trace_id,
        "tutorial_length": len(pkg.tutorial),
        "questions_count": len(pkg.questions),
        "sources_count": len(pkg.sources),
        "claims_count": len(pkg.claims),
    })
    print("\n--- tutorial ---")
    print(pkg.tutorial)
    print("\n--- questions ---")
    print(json.dumps([q.model_dump(mode="json") for q in pkg.questions], ensure_ascii=False, indent=2, default=str))
    print("\n--- sources ---")
    if pkg.sources:
        doc_sources = [s for s in pkg.sources if s.source_type != "video"]
        video_sources = [s for s in pkg.sources if s.source_type == "video"]
        print(f"  文档来源: {len(doc_sources)} 条")
        for s in doc_sources:
            print(f"    [{s.source_id}] type={s.source_type}, title={s.title}")
        print(f"  视频来源: {len(video_sources)} 条")
        for s in video_sources:
            meta = s.metadata_json or {}
            topic = meta.get("topic", "")
            dur = meta.get("duration_hours", 0)
            print(f"    [{s.source_id}] {topic} ({dur}h)")
            print(f"      title: {s.title}")
            print(f"      url: {s.locator}")
    else:
        print("  (空)")
    print("\n--- claims ---")
    if pkg.claims:
        for c in pkg.claims:
            print(f"  [{c.claim_id}] type={c.claim_type}, key_tech={c.is_key_technical}, conf={c.confidence}, status={c.status}")
            print(f"    text: {c.text}")
            print(f"    source_ref: {c.source_ref}, section_ref: {c.section_ref}")
    else:
        print("  (空)")

    # ========== 5. 答疑 ==========
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
        "stage.block_id": stage.block_id,
        "profile.goal": profile.goal,
    })

    # 答疑中间参数
    answer_params = await show_runtime_params_answer(agent, question, stage)
    pretty("【答疑参数汇总】", answer_params)

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
