# -*- coding: utf-8 -*-
"""
Agent1 独立测试 —— 诊断 Agent（学习者画像 + 阶段规划）

职责（对齐 backend/agents/deterministic.py）：
  - 解析学习者自述文本 -> 生成 UserProfile + followup_questions（置信度不足时追问）
  - 置信度达标后，按画像生成第一个学习阶段 StageInfo（原 Agent2 功能已并入 Agent1）

独立运行：
  python tests/check_agent1_diagnostic.py

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
    request = DiagnosticInput(session_id=session_id, user_input=user_input)

    pretty("【输入】DiagnosticInput", {"session_id": session_id, "user_input": user_input})

    # Step 1: 诊断 -> UserProfile
    print("\n[Step 1] 运行 run() 生成 UserProfile ...")
    diagnostic = await agent.run(request)
    pretty("【输出】DiagnosticOutput.profile", diagnostic.profile)
    if diagnostic.followup_questions:
        pretty("【输出】followup_questions（置信度不足时返回追问）", diagnostic.followup_questions)
    else:
        print("\n（无 followup_questions，表示置信度达标）")

    # Step 2: 阶段规划
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
    print("\n[Step 2] 运行 generate_stage() 生成 StageInfo ...")
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
