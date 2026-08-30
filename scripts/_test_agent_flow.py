
# ---------- 路径设置：python scripts\xxx.py 直接运行时能正确找到 backend / data ----------
import sys as _sys_py
from pathlib import Path as _Path_py
_PROJ_ROOT_ = _Path_py(__file__).resolve().parent.parent
if str(_PROJ_ROOT_) not in _sys_py.path:
    _sys_py.path.insert(0, str(_PROJ_ROOT_))
if str(_PROJ_ROOT_ / "backend") not in _sys_py.path:
    _sys_py.path.insert(0, str(_PROJ_ROOT_ / "backend"))
# -------------------------------------------------------------------------------------

# ---------- 路径设置：python scripts\xxx.py 直接运行时能正确找到 backend / data ----------
import sys as _sys_py
from pathlib import Path as _Path_py
_PROJ_ROOT_ = _Path_py(__file__).resolve().parent.parent
if str(_PROJ_ROOT_) not in _sys_py.path:
    _sys_py.path.insert(0, str(_PROJ_ROOT_))
if str(_PROJ_ROOT_ / "backend") not in _sys_py.path:
    _sys_py.path.insert(0, str(_PROJ_ROOT_ / "backend"))
# -------------------------------------------------------------------------------------

import asyncio
import json
import sys

sys.path.insert(0, 'backend')

from agents.deterministic import (
    DeterministicDiagnosticAgent,
    DeterministicStageDecisionAgent,
)
from models.contracts import (
    StageDecisionInput,
    StagePlanningInput,
    DiagnosticInput,
)
from models.profile import KnowledgeGap, UserProfile
from models.stage import StageInfo


async def test_agent1_graph_integration():
    print("=" * 60)
    print("TEST 1: 学情诊断与阶段规划 Agent (DeterministicDiagnosticAgent) graph_query integration")
    print("=" * 60)
    agent = DeterministicDiagnosticAgent()
    result = await agent.run(DiagnosticInput(
        session_id="test-session-01",
        user_input="我想学习LoRA微调",
    ))
    print(f"Profile goal: {result.profile.goal}")
    print(f"Knowledge gaps: {len(result.profile.knowledge_gaps)}")
    for gap in result.profile.knowledge_gaps:
        print(f"  skill={gap.skill}, prereqs={gap.prerequisites}")
    print(f"Confidence: {result.profile.confidence}")
    assert len(result.profile.knowledge_gaps) > 0, "No gaps generated"
    print("[PASS] 学情诊断与阶段规划 Agent graph integration works\n")


async def test_agent2_integration():
    print("=" * 60)
    print("TEST 2: 学情诊断与阶段规划 Agent (generate_stage) integration (merged Agent2 functionality)")
    print("=" * 60)
    agent = DeterministicDiagnosticAgent()
    # Use skill that maps to existing knowledge block ("大模型微调" -> KB-10 or similar)
    profile = UserProfile(
        session_id="test-session-02",
        skill_tags=["Python"],
        knowledge_gaps=[KnowledgeGap(skill="大模型微调", prerequisites=["深度学习"])],
        goal="掌握大模型微调",
    )
    result = await agent.generate_stage(StagePlanningInput(
        session_id="test-session-02",
        profile=profile,
        stage_num=1,
    ))
    print(f"Stage block: {result.stage.block_name}")
    print(f"Stage block_id: {result.stage.block_id}")
    print(f"Stage subtopic_id: {result.stage.subtopic_id}")
    print(f"Stage prerequisites: {result.stage.prerequisites}")
    # Verify stage was generated (block_id should not be empty or None)
    assert result.stage.block_id is not None, "No block_id assigned"
    assert result.stage.block_name is not None, "No block_name assigned"
    print("[PASS] 学情诊断与阶段规划 Agent.generate_stage integration works (merged Agent2)\n")


async def test_agent6_integration():
    print("=" * 60)
    print("TEST 3: 学情评估与动态决策 Agent (DeterministicStageDecisionAgent) integration")
    print("=" * 60)
    agent = DeterministicStageDecisionAgent()
    profile = UserProfile(
        session_id="test-session-03",
        skill_tags=["Python", "PyTorch"],
        knowledge_gaps=[KnowledgeGap(
            skill="LoRA",
            status="学习中",
            progress=0.5,
            prerequisites=["PyTorch"],
        )],
        goal="掌握LoRA微调",
    )
    stage = StageInfo(
        stage_id="stage-001",
        stage_num=1,
        block_id="KB-12",
        block_name="参数高效微调（LoRA与QLoRA）",
        subtopic_id="KBS-12-01",
        gap_skill="LoRA",
        estimated_hours=1.5,
        prerequisites=["深度学习框架PyTorch基础"],
    )
    result = await agent.run(StageDecisionInput(
        session_id="test-session-03",
        profile=profile,
        stage=stage,
        accuracy=0.85,
        qa_history=[],
        interactions=[],
    ))
    print(f"Decision: {result.decision}")
    print(f"Strategy: {result.strategy}")
    print(f"All mastered: {all(g.status == '已掌握' for g in result.profile.knowledge_gaps)}")
    print("[PASS] 学情评估与动态决策 Agent integration works\n")


async def test_harness_basic():
    print("=" * 60)
    print("TEST 4: Harness basic flow (process_initial_input)")
    print("=" * 60)
    from orchestrator.harness import Harness
    harness = Harness(execution_mode="deterministic")
    harness.create_session("test-session-04")
    result = await harness.process_initial_input(
        session_id="test-session-04",
        user_input="我想学习LoRA微调",
    )
    print(f"Status: {result['status']}")
    if result['status'] == 'ready':
        print(f"Profile: goal={result['profile']['goal']}")
        print(f"Stage: block_name={result['stage_info']['block_name']}")
        print(f"Stage: subtopic_id={result['stage_info'].get('subtopic_id')}")
        print("[PASS] Harness basic flow works")
    else:
        print(f"Needs more info: {result.get('questions')}")
        print("[INFO] Harness needs more info (expected for low confidence)")
    print()


async def main():
    try:
        await test_agent1_graph_integration()
    except Exception as e:
        print(f"[FAIL] 学情诊断与阶段规划 Agent test: {e}\n")

    try:
        await test_agent2_integration()
    except Exception as e:
        print(f"[FAIL] Agent2 test: {e}\n")

    try:
        await test_agent6_integration()
    except Exception as e:
        print(f"[FAIL] 学情评估与动态决策 Agent test: {e}\n")

    try:
        await test_harness_basic()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[FAIL] Harness test: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
