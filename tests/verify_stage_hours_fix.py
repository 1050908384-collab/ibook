"""非交互式验证：学习阶段内容询问的预估时长与知识点显示。

用 deterministic 模式（不依赖 ollama）跑完整流程：
  process_initial_input -> start_stage -> handle_question("现在学习阶段内容是什么")
检查回答中的 estimated_hours 与 subtopic_name 是否符合预期。
"""
import asyncio
import sys
import tempfile
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path[:0] = [str(PROJECT_ROOT), str(BACKEND_DIR)]

from db.sqlite_client import SQLiteClient
from orchestrator.harness import Harness
from orchestrator.state_manager import StateManager
from scripts.insert_knowledge_data import (
    BLOCKS_FILE_PATH,
    DOCS_FILE_PATH,
    QUESTIONS_FILE_PATH,
    SUBTOPICS_FILE_PATH,
    insert_knowledge_blocks,
    insert_knowledge_docs,
    insert_knowledge_subtopics,
    insert_test_questions,
    load_required_list,
)


async def run_case(goal: str, daily_hours: float, question: str, mode: str = "deterministic") -> None:
    print("=" * 72)
    print(f"[{mode}] 目标: {goal}  |  每日: {daily_hours}h  |  问题: {question}")
    print("=" * 72)

    with tempfile.TemporaryDirectory(prefix="verify-hours-") as temp_dir:
        db_path = Path(temp_dir) / "verify.db"
        client = SQLiteClient(f"sqlite:///{db_path.as_posix()}")
        try:
            insert_knowledge_blocks(client, load_required_list(BLOCKS_FILE_PATH))
            insert_knowledge_docs(client, load_required_list(DOCS_FILE_PATH))
            insert_knowledge_subtopics(client, load_required_list(SUBTOPICS_FILE_PATH))
            insert_test_questions(client, load_required_list(QUESTIONS_FILE_PATH))

            harness = Harness(StateManager(client), execution_mode=mode)
            session_id = f"verify-hours-{uuid.uuid4().hex[:8]}"
            harness.state_manager.create_session(session_id)

            user_input = "\n".join((
                f"学习目标：{goal}",
                "已掌握基础：无",
                f"每日学习时长：{daily_hours}小时",
                "学习方式：mixed",
            ))
            initial = await harness.process_initial_input(session_id, user_input)
            if initial["status"] != "ready":
                print("  诊断未就绪:")
                for q in initial.get("questions", []):
                    print(f"    - {q}")
                return

            stage_info = initial["stage_info"]
            print(f"\n[阶段信息]")
            print(f"  block: {stage_info['block_name']} (id={stage_info['block_id']})")
            print(f"  subtopic: {stage_info.get('subtopic_name', '')} (id={stage_info.get('subtopic_id', '')})")
            print(f"  estimated_hours: {stage_info.get('estimated_hours', '?')}")
            print(f"  suggested_study_days: {stage_info.get('suggested_study_days', '?')}")

            started = await harness.start_stage(session_id)
            print(f"\n[启动阶段] review_passed={started['review_passed']}")

            result = await harness.handle_question(session_id, question)
            answer = result.get("answer", "")
            print(f"\n[Agent3 回答]")
            print("-" * 72)
            print(answer)
            print("-" * 72)

            # 断言（精确匹配，避免子串误判）
            ok_hours = "**预估时长**：0 小时" not in answer
            ok_kp = "**知识点**：环境准备：安装Python 3.10+" in answer
            ok_obj = "（环境准备：安装Python 3.10+，配置conda" not in answer
            print(f"\n[校验]")
            print(f"  预估时长非0      : {'✅ 通过' if ok_hours else '❌ 仍为0小时'}")
            print(f"  知识点行无括号残留: {'✅ 通过' if ok_kp else '❌ 知识点行异常'}")
            print(f"  objectives无长串括号: {'✅ 通过' if ok_obj else '❌ objectives仍有长串'}")
        finally:
            client.close()


async def main() -> None:
    # deterministic 模式验证
    await run_case("Python编程基础", 5.0, "现在学习阶段内容是什么")
    # hybrid 模式验证（用户实际使用的模式，需要 ollama）
    await run_case("Python编程基础", 5.0, "现在学习阶段内容是什么", mode="hybrid")


if __name__ == "__main__":
    asyncio.run(main())
