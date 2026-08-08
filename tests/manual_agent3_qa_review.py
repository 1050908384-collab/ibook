"""Interactive acceptance check for Agent3 question-answering + Agent4 review.

Run from the project root:
    python tests/manual_agent3_qa_review.py

The script creates an isolated SQLite database, loads the knowledge seeds,
runs initial diagnostic + start_stage to establish a learning context, then
enters an interactive loop where the user asks questions. Each technical
question goes through the Agent3 → Agent4 review → Agent3 revise loop, and
every round of the review debate is printed in detail.

Daily (greeting / chit-chat) questions skip Agent4; technical questions
trigger the full Agent3↔Agent4 review iteration.

Commands available in the Q&A loop:
  - type a question to get an answer
  - next      : complete current stage and advance to the next one
  - status    : show current stage / profile info
  - exit/quit : end the session
"""

import asyncio
import os
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


def prompt(label: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{label}{suffix}: ").strip()
    return value or default


def print_review_round(round_num: int, total: int, round_data: dict) -> None:
    """详细打印一轮 Agent4 审核结果。"""
    technical = round_data.get("technical", {}) or {}
    quality = round_data.get("quality", {}) or {}
    tech_passed = technical.get("passed", False)
    quality_passed = quality.get("passed", False)
    tech_errors = technical.get("errors", []) or []

    print(f"\n--- 第 {round_num}/{total} 轮审核 ---")
    print(f"  技术审核 (Agent4): {'✅ 通过' if tech_passed else '❌ 未通过'}")
    print(f"  质量审核        : {'✅ 通过' if quality_passed else '❌ 未通过'}")

    if tech_errors:
        print(f"  Agent4 提出 {len(tech_errors)} 条 issue：")
        for idx, err in enumerate(tech_errors, 1):
            must_fix = err.get("must_fix", False)
            severity = err.get("severity", "")
            cid = err.get("canonical_error_id", "")
            print(f"    [{idx}] 类型: {err.get('error_type', '')}  |  严重度: {severity}  |  must_fix: {must_fix}")
            print(f"        描述: {err.get('description', '')}")
            if err.get("suggestion"):
                print(f"        建议: {err['suggestion']}")
            if cid:
                print(f"        issue_id: {cid}")
    else:
        print("  Agent4 未提出 issue")

    # 非最后一轮且未通过 → Agent3 会修改回答后进入下一轮
    if round_num < total and not (tech_passed and quality_passed):
        print("  → 本轮未通过，Agent3 修改回答后重新提交 Agent4 审核")


def print_current_stage(harness: Harness, session_id: str) -> None:
    """打印当前阶段信息（与 manual_agent1_stage_planning.py 对齐）。"""
    session = harness.state_manager.get_session(session_id)
    if not session:
        print("  (无 session)")
        return
    stage = session.get("stage_info") or {}
    pkg = session.get("learning_package") or {}
    prerequisites = stage.get("prerequisites") or []
    prerequisites_str = ", ".join(prerequisites) if prerequisites else "none"
    print(f"  Stage {stage.get('stage_num', '?')}: {stage.get('block_id', '?')} / {stage.get('subtopic_id', '?')}")
    print(f"    Block: {stage.get('block_name', '?')}")
    print(f"    Subtopic: {stage.get('subtopic_name', '?')}")
    print(f"    Estimated hours: {stage.get('estimated_hours', '?')}")
    print(f"    Daily study time: {stage.get('recommended_daily_hours', '?')}")
    print(f"    Suggested study days: {stage.get('suggested_study_days', '?')}")
    print(f"    Direct hard prerequisites: {prerequisites_str}")
    if pkg:
        print(f"    学习包题目数: {len(pkg.get('questions', []))}")


async def advance_stage(harness: Harness, session_id: str) -> bool:
    """完成当前阶段 → 规划下一阶段 → 启动下一阶段。

    返回 True 表示成功推进，False 表示无法推进（已全部掌握或出错）。
    """
    # 1. 完成当前阶段
    print("\n[推进阶段] 完成当前阶段 ...")
    completed = await harness.complete_stage(session_id)
    status = completed.get("status", "")
    accuracy = completed.get("accuracy", 0.0)
    all_mastered = completed.get("all_mastered", False)
    can_plan = completed.get("can_plan_next_stage", False)

    print(f"  阶段结果: {status}  |  正确率: {accuracy:.2f}  |  全部掌握: {all_mastered}")

    if all_mastered:
        print("  🎉 全部知识盲区已掌握，学习完成！")
        final_report = completed.get("final_report")
        if final_report:
            print("\n  --- 最终报告 ---")
            if isinstance(final_report, dict):
                for k, v in final_report.items():
                    print(f"  {k}: {v}")
            else:
                print(f"  {final_report}")
        return False

    if not can_plan:
        print("  当前无法规划下一阶段。")
        return False

    # 2. 规划下一阶段
    print("[推进阶段] 规划下一阶段 ...")
    planned = await harness.plan_next_stage(session_id)
    next_stage = planned.get("next_stage", {}) or {}
    print(f"  下一阶段: {next_stage.get('block_name', '?')} "
          f"(subtopic={next_stage.get('subtopic_name', '?')})")

    # 3. 启动下一阶段
    print("[推进阶段] 启动下一阶段，生成学习包 ...")
    started = await harness.start_stage(session_id)
    print(f"  学习包审核通过: {started.get('review_passed', False)}")
    print(f"  阶段测试题数  : {len(started.get('learning_package', {}).get('questions', []))}")
    print_current_stage(harness, session_id)
    return True


async def main() -> None:
    print("=" * 72)
    print("Agent3 问答质量 + Agent3↔Agent4 审核辩论  交互测试")
    print("=" * 72)

    # 执行模式：hybrid 有真实 LLM 回答，deterministic 为模板回答
    mode = prompt("执行模式 (deterministic / hybrid)", "hybrid")
    if mode not in ("deterministic", "hybrid"):
        raise SystemExit("模式必须是 deterministic 或 hybrid")
    os.environ["EXECUTION_MODE"] = mode

    # === 输入字段与 manual_agent1_stage_planning.py 对齐 ===
    # Hardware is collected separately and is not sent to Agent1.
    goal = prompt("Learning goal", "I want to learn LoRA")
    mastered = prompt("Mastered skills (comma-separated; optional)")
    daily_hours_text = prompt("Daily study hours", "2")
    learning_style = prompt("Learning style", "mixed")
    hardware = prompt("Practice hardware/environment (Agent3-only; 例: Windows 11, RTX 5060)", "")

    try:
        daily_hours = float(daily_hours_text)
    except ValueError:
        raise SystemExit("Daily study hours must be numeric.")

    session_id = f"manual-agent3-qa-{uuid.uuid4().hex[:8]}"
    # user_input 只包含 Agent1 需要的信息，不包含 hardware
    user_input = "\n".join((
        f"学习目标：{goal}",
        f"已掌握基础：{mastered or '无'}",
        f"每日学习时长：{daily_hours}小时",
        f"学习方式：{learning_style}",
    ))

    with tempfile.TemporaryDirectory(prefix="agent3-qa-") as temp_dir:
        db_path = Path(temp_dir) / "agent3_qa.db"
        client = SQLiteClient(f"sqlite:///{db_path.as_posix()}")
        try:
            # 加载知识数据（题目表供 start_stage 使用）
            insert_knowledge_blocks(client, load_required_list(BLOCKS_FILE_PATH))
            insert_knowledge_docs(client, load_required_list(DOCS_FILE_PATH))
            insert_knowledge_subtopics(client, load_required_list(SUBTOPICS_FILE_PATH))
            insert_test_questions(client, load_required_list(QUESTIONS_FILE_PATH))

            harness = Harness(StateManager(client), execution_mode=mode)
            harness.state_manager.create_session(session_id)

            # === 初始化诊断 + 阶段规划 ===
            print("\n[初始化] 诊断 + 阶段规划 ...")
            initial = await harness.process_initial_input(session_id, user_input)
            if initial["status"] != "ready":
                print("诊断未就绪，需要补充信息：")
                for q in initial.get("questions", []):
                    print(f"  - {q}")
                return

            # 如果用户输入了硬件信息，保存到 practice_profile（Agent3 使用）
            if hardware:
                harness.state_manager.update_practice_profile(
                    session_id, {"hardware": hardware}
                )

            stage_info = initial["stage_info"]
            print(f"  阶段   : {stage_info['block_name']} (block_id={stage_info['block_id']})")
            print(f"  子知识点: {stage_info.get('subtopic_name', '')} (subtopic_id={stage_info.get('subtopic_id', '')})")

            # 显示知识盲区（帮助用户了解完整学习路径）
            profile = initial.get("profile", {})
            gaps = profile.get("knowledge_gaps", [])
            if gaps:
                print(f"\n  知识盲区:")
                for gap in gaps:
                    print(f"    - {gap.get('skill', '')} ({gap.get('block_name', '')})")
                    print(f"      目标子知识点: {gap.get('target_subtopic_ids', [])}")

            # === 启动阶段，生成学习包（问答需要 stage 上下文）===
            print("\n[初始化] 启动阶段，生成学习包 ...")
            started = await harness.start_stage(session_id)
            print(f"  学习包审核通过: {started['review_passed']}")
            print(f"  阶段测试题数  : {len(started['learning_package']['questions'])}")

            # === 交互问答循环 ===
            print("\n" + "=" * 72)
            print("进入问答交互模式")
            print("  - 输入问题后，Agent3 回答，Agent4 逐轮审核")
            print("  - next     : 完成当前阶段，推进到下一阶段")
            print("  - status   : 查看当前阶段信息")
            print("  - exit/quit: 结束")
            print("=" * 72)
            print_current_stage(harness, session_id)

            while True:
                print()
                try:
                    question = input("你的问题> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n退出问答测试。")
                    break
                if not question:
                    continue
                if question.lower() in ("exit", "quit", "q", "退出"):
                    print("退出问答测试。")
                    break

                # === 命令：查看当前阶段 ===
                if question.lower() == "status":
                    print_current_stage(harness, session_id)
                    continue

                # === 命令：推进到下一阶段 ===
                if question.lower() == "next":
                    ok = await advance_stage(harness, session_id)
                    if not ok:
                        print("无法继续推进，结束测试。")
                        break
                    continue

                # === 普通问答 ===
                print("\n" + "-" * 72)
                print(f"处理问题: {question}")
                print("-" * 72)

                result = await harness.handle_question(session_id, question)

                question_type = result.get("question_type", "technical")
                iterations = result.get("iterations", 0)
                review_history = result.get("review_history", []) or []

                print(f"\n【问题分类】{question_type}")
                if question_type == "daily":
                    print("  日常类问题 → 跳过 Agent4 审核")
                else:
                    print(f"  技术类问题 → 触发 Agent3↔Agent4 审核辩论，共 {iterations} 轮")
                    for i, round_data in enumerate(review_history, 1):
                        print_review_round(i, len(review_history), round_data)

                # 最终回答与审核状态
                review_summary = result.get("review_summary", {}) or {}
                print("\n" + "=" * 72)
                print("【Agent3 最终回答】")
                print("=" * 72)
                print(result.get("answer", ""))
                print("=" * 72)
                print(f"技术审核: {'✅ 通过' if result.get('tech_pass') else '❌ 未通过'}  |  "
                      f"质量审核: {'✅ 通过' if result.get('quality_pass') else '❌ 未通过'}")
                if review_summary:
                    print(f"审核状态: {review_summary.get('status', '')}  |  "
                          f"需人工复核: {review_summary.get('needs_human_review', False)}")
        finally:
            client.close()


if __name__ == "__main__":
    asyncio.run(main())
