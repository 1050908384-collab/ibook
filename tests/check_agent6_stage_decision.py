# -*- coding: utf-8 -*-
"""
Agent6 独立测试 —— 阶段决策 Agent

职责（对齐 backend/agents/deterministic.py）：
  - 阶段结束后，依据 accuracy / qa_history / interactions 更新画像（掌握度、节奏、信心、困惑标记）
  - 产出决策 decision（continue / repeat / completed）+ 策略 strategy + 阶段总结 stage_summary
  - 透传 iteration_history 激活 backtrack / finish 策略（iteration_strategy.skill.md）

独立运行：
  python tests/check_agent6_stage_decision.py

说明：
  - Agent6 仅有 deterministic 版本（无 hybrid），Harness 中即 DeterministicStageDecisionAgent，本测试与之一致。
  - accuracy >= 0.8 -> continue；< 0.8 -> repeat；全部掌握或 strategy=finish -> completed。
"""
import asyncio
import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, ROOT_DIR)

from agents.deterministic import DeterministicStageDecisionAgent
from models.contracts import StageDecisionInput
from models.profile import KnowledgeGap, UserProfile
from models.stage import StageInfo


def pretty(label: str, obj) -> None:
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump(mode="json")
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


async def main():
    print("=" * 60)
    print("Agent6 阶段决策 Agent 独立测试（画像更新 + 决策 + 总结）")
    print("=" * 60)
    print("说明：Agent6 仅 deterministic 版本（无 hybrid），与 Harness 实际使用一致。")

    agent = DeterministicStageDecisionAgent()

    # 输入 accuracy
    acc_in = input("\n请输入本阶段正确率 accuracy（0~1，回车=0.85）: ").strip()
    try:
        accuracy = float(acc_in) if acc_in else 0.85
        accuracy = max(0.0, min(1.0, accuracy))
    except ValueError:
        accuracy = 0.85
    print(f"accuracy = {accuracy}")

    # 构造 profile（含一个学习中、一个未掌握的 gap，便于观察决策差异）
    session_id = "test-agent6-session"
    profile = UserProfile(
        session_id=session_id,
        skill_tags=["Python", "PyTorch"],
        knowledge_gaps=[
            KnowledgeGap(skill="LoRA", status="学习中", progress=0.5, prerequisites=["PyTorch"]),
            KnowledgeGap(skill="QLoRA", status="未掌握", progress=0.0, prerequisites=["LoRA"]),
        ],
        goal="掌握LoRA微调",
    )
    stage = StageInfo(
        stage_id="stage-test-06", stage_num=1, block_id="KB-12",
        block_name="参数高效微调（LoRA与QLoRA）", gap_skill="LoRA",
        estimated_hours=2.0,
    )
    qa_history = [
        {"question_id": "Q1", "is_correct": True, "emotion_score": 0.8},
        {"question_id": "Q2", "is_correct": accuracy >= 0.5, "emotion_score": 0.6},
    ]
    interactions = [
        {"time_spent": 5400, "expected_time": 7200},  # 1.5h 实际 / 2h 预期
    ]

    request = StageDecisionInput(
        session_id=session_id,
        profile=profile,
        stage=stage,
        accuracy=accuracy,
        qa_history=qa_history,
        interactions=interactions,
    )
    pretty("【输入】StageDecisionInput", {
        "accuracy": accuracy,
        "stage": stage.model_dump(mode="json"),
        "profile.knowledge_gaps": [g.model_dump(mode="json") for g in profile.knowledge_gaps],
        "qa_history": qa_history,
        "interactions": interactions,
    })

    # Step 1: 阶段决策
    print("\n[Step 1] 运行 run() -> StageDecisionOutput ...")
    output = await agent.run(request)
    pretty("【输出】StageDecisionOutput（核心字段）", {
        "decision": output.decision,
        "strategy": output.strategy,
        "stage_summary": output.stage_summary,
    })
    pretty("【输出】更新后的 profile.knowledge_gaps（观察 status / progress 变化）",
           [g.model_dump(mode="json") for g in output.profile.knowledge_gaps])
    pretty("【输出】更新后的 profile（rhythm / confidence_index / confusion_markers）", {
        "learning_rhythm": output.profile.learning_rhythm,
        "confidence_index": output.profile.confidence_index,
        "confusion_markers": output.profile.confusion_markers,
        "version": output.profile.version,
    })

    print("\n决策参考：")
    print("  accuracy >= 0.8  -> continue（推进下一阶段）")
    print("  accuracy <  0.8  -> repeat（重试当前阶段）")
    print("  全部已掌握 / strategy=finish -> completed（结业）")
    print("  strategy=backtrack -> repeat（回退前置知识）")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n已中断。")
    except Exception:
        import traceback
        traceback.print_exc()
