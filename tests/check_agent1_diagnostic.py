# -*- coding: utf-8 -*-
"""
Agent1 独立测试 —— 诊断 Agent（学习者画像 + 阶段规划）

职责（对齐 backend/agents/deterministic.py）：
  - 解析学习者自述文本 -> 生成 UserProfile + followup_questions（置信度不足时追问）
  - 置信度达标后，按画像生成第一个学习阶段 StageInfo（原 Agent2 功能已并入 Agent1）

独立运行：
  python tests/check_agent1_diagnostic.py

输出内容：
  1. Agent 配置（模式、工具列表）
  2. 输入（DiagnosticInput）
  3. 运行中主要参数（技能提取、知识图谱查询、置信度计算 — 直接调用工具预览中间数据）
  4. Agent1 run() 输出（UserProfile、followup_questions）
  5. 阶段规划中间参数（知识板块查询、可学习子知识点候选）
  6. 阶段规划输出（StageInfo）

说明：
  - hybrid 模式调用本地 Ollama（需已启动）；deterministic 模式纯规则/工具，不调 LLM。
  - generate_stage 在 hybrid 模式下仍走确定性实现（与 Harness 实际行为一致）。
"""
import asyncio
import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, ROOT_DIR)

from agents.deterministic import DeterministicDiagnosticAgent, HybridDiagnosticAgent
from models.contracts import DiagnosticInput, StagePlanningInput


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


async def show_runtime_params_diagnostic(agent, text: str) -> dict:
    """直接调用 Agent1 的工具，预览运行中的主要参数。"""
    section("运行中主要参数（Agent1 内部工具调用预览）")
    params = {}

    # 1. 技能提取
    print("\n  [工具] extract_skills_tool.extract_skills(text)")
    try:
        extracted_tags = await agent.extract_skills_tool.extract_skills(text)
        params["extracted_skills"] = extracted_tags
        print(f"    → extracted_skills: {extracted_tags}")
    except Exception as e:
        params["extracted_skills"] = f"[ERROR] {e}"
        print(f"    → [ERROR] {e}")

    # 2. 知识缺口提取
    print("\n  [工具] extract_skills_tool.extract_gaps(text)")
    try:
        extracted_gaps = await agent.extract_skills_tool.extract_gaps(text)
        params["extracted_gaps"] = extracted_gaps
        print(f"    → extracted_gaps: {json.dumps(extracted_gaps, ensure_ascii=False)}")
    except Exception as e:
        params["extracted_gaps"] = f"[ERROR] {e}"
        print(f"    → [ERROR] {e}")

    # 3. 知识图谱查询（前置知识）
    print("\n  [工具] graph_query_tool.query_prerequisites(skill)")
    from agents.deterministic import SKILL_NAMES
    target_text = text
    import re
    match = re.search(r"(?:想学习|想学|学习目标(?:是|为)?|目标(?:是|为)?)(.+)", text, flags=re.IGNORECASE)
    if match:
        target_text = match.group(1).strip(" ，。,.！！")
    targets = [s for s in SKILL_NAMES if s.lower() in target_text.lower()]
    if not targets:
        targets = ["LoRA"]
    for skill in targets:
        try:
            prereqs = await agent.graph_query_tool.query_prerequisites(skill)
            params[f"prerequisites[{skill}]"] = prereqs
            print(f"    → {skill}: {prereqs}")
        except Exception as e:
            params[f"prerequisites[{skill}]"] = f"[ERROR] {e}"
            print(f"    → {skill}: [ERROR] {e}")

    # 4. 置信度计算
    print("\n  [工具] calculate_confidence_tool.calculate(text, mastered, gaps)")
    mastered = params.get("extracted_skills", [])
    gaps_for_conf = params.get("extracted_gaps", [])
    try:
        confidence = await agent.calculate_confidence_tool.calculate(
            text, mastered, gaps_for_conf
        )
        params["calculated_confidence"] = confidence
        print(f"    → confidence: {confidence:.4f}")
    except Exception as e:
        params["calculated_confidence"] = f"[ERROR] {e}"
        print(f"    → [ERROR] {e}")

    return params


async def show_runtime_params_stage(agent, profile, stage_num: int, session_id: str) -> dict:
    """阶段规划的中间参数。"""
    section("阶段规划中间参数（Agent1.generate_stage 内部工具调用预览）")
    params = {}
    gap = next(
        (item for item in profile.knowledge_gaps if float(item.progress or 0.0) < 1.0),
        None,
    )
    if gap is None:
        print("  [跳过] 无未掌握的知识缺口")
        return params
    print(f"  当前缺口: skill={gap.skill}, level={getattr(gap, 'level', '?')}, progress={getattr(gap, 'progress', 0)}")

    # 1. 知识板块查询
    print(f"\n  [工具] knowledge_block_query_tool.query_block('{gap.skill}')")
    try:
        preferred_block = await agent.knowledge_block_query_tool.query_block(gap.skill)
        params["preferred_block"] = preferred_block
        print(f"    → block_id: {preferred_block.get('block_id', '?')}")
        print(f"    → block_name: {preferred_block.get('block_name', '?')}")
        print(f"    → estimated_hours: {preferred_block.get('estimated_hours', '?')}")
    except Exception as e:
        params["preferred_block"] = f"[ERROR] {e}"
        print(f"    → [ERROR] {e}")

    # 2. 可学习子知识点候选
    print(f"\n  [DB] db_client.list_learnable_subtopics(session_id='{session_id}')")
    try:
        candidates = agent.db_client.list_learnable_subtopics(
            session_id=session_id,
            preferred_block_id=preferred_block.get("block_id") if isinstance(preferred_block, dict) else None,
            max_hours=profile.daily_hours,
        )
        params["subtopic_candidates"] = candidates
        print(f"    → 候选数: {len(candidates)}")
        for i, c in enumerate(candidates[:5]):
            print(f"      [{i}] subtopic_id={c.get('subtopic_id', '?')}, name={c.get('subtopic_name', '?')}, block={c.get('block_id', '?')}, hours={c.get('estimated_hours', '?')}")
        if len(candidates) > 5:
            print(f"      ... 共 {len(candidates)} 个")
    except Exception as e:
        params["subtopic_candidates"] = f"[ERROR] {e}"
        print(f"    → [ERROR] {e}")

    # 3. 阶段可用视频资源
    block_id_for_video = preferred_block.get("block_id") if isinstance(preferred_block, dict) else None
    block_name_for_video = preferred_block.get("block_name", gap.skill) if isinstance(preferred_block, dict) else gap.skill
    print(f"\n  [视频] video_search_tool.search_by_block(block_id='{block_id_for_video}', block_name='{block_name_for_video}')")
    try:
        from tools.agent3_tools.video_search_tool import search_videos
        video_list = search_videos(block_id=block_id_for_video, block_name=block_name_for_video, top_k=5)
        params["available_videos"] = video_list
        print(f"    → 可用视频: {len(video_list)} 个")
        for i, v in enumerate(video_list[:5]):
            vid = v.get("video_id", "?")
            tp = v.get("topic", "?")
            tt = v.get("title", "?")[:50]
            dur = v.get("duration_hours", 0)
            print(f"      [{i}] {vid} | {tp} ({dur}h) - {tt}")
    except Exception as e:
        params["available_videos"] = f"[ERROR] {e}"
        print(f"    → [ERROR] {e}")

    # 4. 阶段生成器
    print(f"\n  [工具] stage_generator_tool.generate_stage(...)")
    try:
        generated_stage = await agent.stage_generator_tool.generate_stage(
            session_id,
            gap.model_dump(mode="json"),
            preferred_block if isinstance(preferred_block, dict) else {},
            profile.goal,
            stage_num,
        )
        params["generated_stage"] = generated_stage
        print(f"    → stage_id: {generated_stage.get('stage_id', '?')}")
        print(f"    → block_name: {generated_stage.get('block_name', '?')}")
    except Exception as e:
        params["generated_stage"] = f"[ERROR] {e}"
        print(f"    → [ERROR] {e}")

    return params


def choose_mode() -> str:
    print("\n选择执行模式：")
    print("  1) hybrid        使用本地 LLM（Ollama），查看真实模型输出（默认）")
    print("  2) deterministic 纯规则/工具链路，不调用 LLM，速度快")
    choice = input("请输入 1 或 2（回车=1）: ").strip()
    return "deterministic" if choice == "2" else "hybrid"


DEFAULT_INPUT = (
    "我的身份背景：研究生，有PyTorch基础，用过transformer库\n"
    "我的已有基础：Python、PyTorch\n"
    "我的学习目标：掌握LoRA微调，能独立部署到生产环境\n"
    "我每天可以学习：2小时\n"
    "我的学习方式：偏爱动手实操\n"
    "我的设备条件：RTX 4060"
)


async def main():
    print("=" * 60)
    print("Agent1 诊断 Agent 独立测试（画像 + 阶段规划）")
    print("=" * 60)

    mode = choose_mode()
    agent = HybridDiagnosticAgent() if mode == "hybrid" else DeterministicDiagnosticAgent()

    print("\n请输入学习者自述（直接回车使用默认示例）：")
    print("-" * 60)
    print(DEFAULT_INPUT)
    print("-" * 60)
    user_input = input(">>> ").strip()
    if not user_input:
        user_input = DEFAULT_INPUT

    session_id = "test-agent1-session"

    # ========== 1. Agent 配置 ==========
    show_agent_config(agent)

    # ========== 2. 输入 ==========
    request = DiagnosticInput(session_id=session_id, user_input=user_input)
    pretty("【输入】DiagnosticInput", {"session_id": session_id, "user_input": user_input})

    # ========== 3. 运行中主要参数 ==========
    runtime_params = await show_runtime_params_diagnostic(agent, user_input)
    pretty("【运行参数汇总】", runtime_params)

    # ========== 4. Agent1 run() 输出 ==========
    section("[Step 1] 运行 run() 生成 UserProfile")
    diagnostic = await agent.run(request)
    pretty("【输出】DiagnosticOutput.profile", diagnostic.profile)
    if diagnostic.followup_questions:
        pretty("【输出】followup_questions（置信度不足时返回追问）", diagnostic.followup_questions)
    else:
        print("\n（无 followup_questions，表示置信度达标）")

    # ========== 5. 阶段规划 ==========
    print(f"\nprofile.confidence = {diagnostic.profile.confidence:.2f}")
    ans = input("\n是否继续生成学习阶段 StageInfo？(y/N): ").strip().lower()
    if ans != "y":
        print("已结束。")
        return
    stage_num_in = input("stage_num（回车=1）: ").strip()
    stage_num = int(stage_num_in) if stage_num_in else 1

    plan_request = StagePlanningInput(
        session_id=session_id,
        profile=diagnostic.profile,
        stage_num=stage_num,
    )
    pretty("【输入】StagePlanningInput", {
        "session_id": session_id,
        "stage_num": stage_num,
        "profile.goal": diagnostic.profile.goal,
        "knowledge_gaps": [g.model_dump(mode="json") for g in diagnostic.profile.knowledge_gaps],
    })

    # 阶段规划中间参数
    stage_params = await show_runtime_params_stage(agent, diagnostic.profile, stage_num, session_id)
    pretty("【阶段规划参数汇总】", stage_params)

    print(f"\n[Step 2] 运行 generate_stage() 生成 StageInfo ...")
    try:
        plan = await agent.generate_stage(plan_request)
        pretty("【输出】StagePlanningOutput.stage", plan.stage)
    except ValueError as e:
        print(f"\n[跳过] generate_stage 未生成阶段：{e}")
        print("（通常是该 session 在 DB 中已无可学习子知识点，可换一个 session_id 或检查 seed 数据）")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n已中断。")
    except Exception:
        import traceback
        traceback.print_exc()
