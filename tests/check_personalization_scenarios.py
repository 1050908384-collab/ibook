# -*- coding: utf-8 -*-
"""Compare Agent3 output for multiple learner profiles on one real source chunk.

Run:
  python tests/check_personalization_scenarios.py
  python tests/check_personalization_scenarios.py --mode hybrid
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
sys.path.insert(0, BACKEND_DIR)

from agents.deterministic import DeterministicContentAgent, HybridContentAgent
from models.profile import PracticeProfile, UserProfile
from models.stage import StageInfo


SCENARIOS = [
    {
        "name": "Windows零基础实操型",
        "goal": "在 Windows 上用 WSL2 准备后续 LoRA 微调环境",
        "learning_style": "实操型",
        "hardware": "Windows 11 + RTX 4060 8GB",
    },
    {
        "name": "原理优先阅读型",
        "goal": "理解 Linux 环境选择的原理与适用边界",
        "learning_style": "阅读型",
        "hardware": "",
    },
    {
        "name": "已有GPU的部署目标型",
        "goal": "为 vLLM 推理部署准备 Ubuntu 命令行环境",
        "learning_style": "示例驱动",
        "hardware": "RTX 4090 24GB",
    },
]


async def main(mode: str) -> int:
    agent = HybridContentAgent() if mode == "hybrid" else DeterministicContentAgent()
    stage = StageInfo(
        stage_id="personalization-demo-linux",
        stage_num=1,
        block_id="KB-01",
        block_name="Linux环境与命令行操作",
        gap_skill="Linux环境准备",
        subtopic_id="KBS-01-00",
    )
    chunks = agent.iter_stage_chunks(stage)
    if not chunks:
        print("未找到 KB-01/KBS-01-00 的原始知识块，无法执行场景验收。")
        return 1

    source_chunk = chunks[0]
    print(f"模式：{mode}；来源块：{source_chunk.get('chunk_id')}；标题：{source_chunk.get('heading_path')}")
    actions = set()
    reasons = set()
    try:
        for index, scenario in enumerate(SCENARIOS, 1):
            profile = UserProfile(
                session_id=f"personalization-demo-{index}",
                goal=scenario["goal"],
                learning_style=scenario["learning_style"],
            )
            practice = PracticeProfile.from_hardware_text(scenario["hardware"])
            block = await agent.generate_block(
                stage,
                source_chunk,
                profile=profile,
                practice=practice,
                block_index=0,
                session_id=profile.session_id,
            )
            actions.add(block.personalization_action)
            reasons.add(block.personalization_reason)
            print(f"\n=== 场景{index}：{scenario['name']} ===")
            print(f"动作：{block.personalization_action}")
            print(f"依据：{block.personalization_reason}")
            print(f"任务数：{len(block.teaching_tasks)}；来源：{block.fact_source_chunk_ids}")
            print("Markdown预览：")
            print(block.personalized_content[:900])
            if block.personalized_content.lstrip().startswith("Error:"):
                print("FAIL：模型错误信息不能作为学习材料展示。")
                return 1

        if len(reasons) != len(SCENARIOS):
            print("FAIL：三种画像未生成不同的个性化依据。")
            return 1
        if "explain_and_practice" not in actions:
            print("FAIL：实操型画像未生成实操教学动作。")
            return 1
        print("\nPASS：三种画像具有不同的个性化依据，且实操型场景包含实操教学动作。")
        return 0
    finally:
        if mode == "hybrid":
            await agent.llm_client.client.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("deterministic", "hybrid"), default="deterministic")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.mode)))
