
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

import os
import sys
import asyncio

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, BACKEND_DIR)

from backend.orchestrator.harness import Harness

async def run_demo():
    print("=== AI Agent Trainer Demo ===")
    print()
    
    harness = Harness()
    
    session_id = harness.state_manager.create_session(None)["session_id"]
    print(f"Created session: {session_id}")
    print()
    
    user_input = """
    我的学习背景：研究生，有PyTorch基础，用过transformer库
    我的学习目标：掌握LoRA微调，能独立部署到生产环境
    我每天能学：2小时
    我喜欢怎么学：偏爱动手实操，不太喜欢纯理论讲解
    """
    
    print("Step 1: User initial input")
    print("-" * 50)
    print(user_input)
    print()
    
    result = await harness.process_initial_input(session_id, user_input)
    print("Step 2: Profile generated")
    print("-" * 50)
    print(f"Confidence: {result['profile']['confidence']}")
    print(f"Skill tags: {result['profile']['skill_tags']}")
    print(f"Knowledge gaps: {[g['skill'] for g in result['profile']['knowledge_gaps']]}")
    print()
    
    print("Step 3: Stage generated")
    print("-" * 50)
    print(f"Stage: {result['stage_info']['block_name']}")
    print(f"Estimated hours: {result['stage_info']['estimated_hours']}")
    print()
    
    print("Step 4: Starting stage...")
    print("-" * 50)
    stage_result = await harness.start_stage(session_id)
    print(f"Review passed: {stage_result['review_passed']}")
    print(f"Iterations: {stage_result['iterations']}")
    print()
    
    print("Step 5: Answering a question")
    print("-" * 50)
    question = "什么是LoRA？它是如何工作的？"
    print(f"Question: {question}")
    answer_result = await harness.handle_question(session_id, question)
    print(f"Answer length: {len(answer_result['answer'])} characters")
    print(f"Tech pass: {answer_result['tech_pass']}")
    print(f"Quality pass: {answer_result['quality_pass']}")
    print()
    
    print("Step 6: Completing stage")
    print("-" * 50)
    complete_result = await harness.complete_stage(session_id)
    print(f"Status: {complete_result['status']}")
    print(f"Accuracy: {complete_result['accuracy']}")
    print()
    
    print("=== Demo completed ===")

if __name__ == "__main__":
    asyncio.run(run_demo())
