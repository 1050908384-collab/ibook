"""验证 agent3 技术类问题不再答非所问。

复现用户场景：学习阶段 Python 编程基础，问"我先检验语法掌握程度要怎么看"，
确认回答与"语法掌握/检验"相关，而非"检查 Python 安装"（环境准备内容）。
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
    BLOCKS_FILE_PATH, DOCS_FILE_PATH, QUESTIONS_FILE_PATH, SUBTOPICS_FILE_PATH,
    insert_knowledge_blocks, insert_knowledge_docs, insert_knowledge_subtopics, insert_test_questions,
    load_required_list,
)


async def main():
    with tempfile.TemporaryDirectory(prefix="verify-relevance-") as temp_dir:
        db_path = Path(temp_dir) / "relevance.db"
        client = SQLiteClient(f"sqlite:///{db_path.as_posix()}")
        try:
            insert_knowledge_blocks(client, load_required_list(BLOCKS_FILE_PATH))
            insert_knowledge_docs(client, load_required_list(DOCS_FILE_PATH))
            insert_knowledge_subtopics(client, load_required_list(SUBTOPICS_FILE_PATH))
            insert_test_questions(client, load_required_list(QUESTIONS_FILE_PATH))

            harness = Harness(StateManager(client), execution_mode="hybrid")
            session_id = f"verify-rel-{uuid.uuid4().hex[:8]}"
            harness.state_manager.create_session(session_id)

            user_input = "\n".join((
                "学习目标：Python编程基础",
                "已掌握基础：无",
                "每日学习时长：5.0小时",
                "学习方式：mixed",
            ))
            initial = await harness.process_initial_input(session_id, user_input)
            assert initial["status"] == "ready"
            await harness.start_stage(session_id)

            # 用户问题（复现用户场景）
            question = "我先检验语法掌握程度要怎么看"
            result = await harness.handle_question(session_id, question)
            answer = result.get("answer", "")

            print("=" * 72)
            print(f"【用户问题】{question}")
            print("=" * 72)
            print(answer)
            print("=" * 72)

            # 断言1：回答不应是"检查 Python 安装"（环境准备内容）
            bad_keywords = ["python --version", "检查Python是否安装", "确认Python路径", "which python", "Python已经安装"]
            has_bad = any(kw in answer for kw in bad_keywords)

            # 断言2：回答不应是 try-except 异常处理（LLM 误解"检验"为"捕捉错误"）
            bad_llm_keywords = ["try-except", "try/except", "捕捉异常", "捕获异常", "FileNotFoundError"]
            has_bad_llm = any(kw in answer for kw in bad_llm_keywords)

            # 断言3：应走自测专门路径，包含自测方案结构
            has_self_assessment_title = "检验本阶段掌握程度" in answer
            has_self_assessment_section = "自测题目" in answer or "自测建议" in answer
            has_subtopic_list = "本阶段知识点" in answer or "知识点" in answer

            # 断言4：应包含本阶段知识点名称（环境准备/语法/数据结构之一）
            has_stage_topic = any(kw in answer for kw in ["语法", "数据结构", "环境准备"])

            print(f"\n[校验]")
            print(f"  不含环境准备答案     : {'✅ 通过' if not has_bad else '❌ 仍含检查Python安装内容'}")
            print(f"  不含try-except误解   : {'✅ 通过' if not has_bad_llm else '❌ 仍含异常捕获内容'}")
            print(f"  含自测方案标题       : {'✅ 通过' if has_self_assessment_title else '❌ 未走自测专门路径'}")
            print(f"  含自测题目/建议      : {'✅ 通过' if has_self_assessment_section else '❌ 未给出自测方案'}")
            print(f"  含知识点列表         : {'✅ 通过' if has_subtopic_list else '❌ 未展示知识点'}")
            print(f"  含本阶段知识点名称   : {'✅ 通过' if has_stage_topic else '❌ 未含阶段知识点'}")

            all_pass = (
                not has_bad and not has_bad_llm
                and has_self_assessment_title and has_self_assessment_section
                and has_subtopic_list and has_stage_topic
            )
            print(f"\n  总体: {'✅ 全部通过' if all_pass else '❌ 存在未通过项'}")
        finally:
            client.close()


if __name__ == "__main__":
    asyncio.run(main())
