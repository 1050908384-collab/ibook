"""Interactive acceptance check for 学情诊断与阶段规划 Agent with step-by-step foundation selection.

Run from the project root:
    python tests/manual_agent1_stage_planning.py

Flow:
1. User inputs a learning goal (e.g. "掌握 LoRA 微调并部署")
2. 学情诊断与阶段规划 Agent returns matching knowledge blocks
3. User selects mastered blocks
4. User expands each selected block to pick mastered subtopics
5. User confirms → 学情诊断与阶段规划 Agent builds profile + generates first stage
6. Stage-by-stage planning simulation
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
os.environ.setdefault("EXECUTION_MODE", "deterministic")

from agents.deterministic import DeterministicDiagnosticAgent, _shorten_subtopic_name
from db.sqlite_client import SQLiteClient
from models.contracts import StagePlanningInput
from models.profile import PracticeProfile
from scripts.insert_knowledge_data import (
    BLOCKS_FILE_PATH,
    SUBTOPICS_FILE_PATH,
    insert_knowledge_blocks,
    insert_knowledge_subtopics,
    load_required_list,
)


def prompt(label: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{label}{suffix}: ").strip()
    return value or default


def format_stage(stage) -> str:
    prerequisites = ", ".join(stage.prerequisites) if stage.prerequisites else "none"
    subtopic_names = getattr(stage, "subtopic_names", []) or []
    subtopic_ids = getattr(stage, "subtopic_ids", []) or []
    lines = [
        f"Stage {stage.stage_num}: {stage.block_id} / {','.join(subtopic_ids) or stage.subtopic_id or '-'}",
        f"  Block: {stage.block_name}",
    ]
    if subtopic_names:
        lines.append(f"  Subtopics ({len(subtopic_names)}):")
        for i, name in enumerate(subtopic_names, 1):
            lines.append(f"    [{i}] {name}")
    else:
        lines.append(f"  Subtopic: {stage.subtopic_name}")
    lines.extend([
        f"  Estimated hours: {stage.estimated_hours}",
        f"  Daily study time: {stage.recommended_daily_hours}",
        f"  Suggested study days: {stage.suggested_study_days}",
        f"  Direct hard prerequisites: {prerequisites}",
    ])
    return "\n".join(lines)


def select_blocks(blocks: list[dict]) -> list[str]:
    """交互式选择已掌握的知识板块，返回选中的 block_id 列表。"""
    print("\n" + "=" * 72)
    print("第 1 步：选择你已掌握的知识板块")
    print("  输入板块编号（逗号分隔），如 1,3,5")
    print("  直接回车表示跳过（无已掌握板块）")
    print("=" * 72)
    for i, block in enumerate(blocks, 1):
        sub_count = len(block.get("subtopics", []))
        print(f"  [{i}] {block['block_id']}: {block['block_name']}（{sub_count}个知识点）")
        if block.get("description"):
            desc = _shorten_subtopic_name(block["description"].replace("\n", " ").strip())
            print(f"      {desc[:80]}")

    raw = input("\n已掌握的板块编号: ").strip()
    if not raw:
        return []
    # 支持逗号、空格、中文逗号、分号多种分隔
    for sep in [" ", "，", ",", ";", "；", "、"]:
        raw = raw.replace(sep, ",")
    selected_indices: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            idx = int(part)
            if 1 <= idx <= len(blocks):
                selected_indices.append(idx)
    return [blocks[idx - 1]["block_id"] for idx in selected_indices]


def select_subtopics(blocks: list[dict], selected_block_ids: list[str]) -> list[str]:
    """交互式展开选中板块，选择已掌握的知识点，返回 subtopic_id 列表。"""
    if not selected_block_ids:
        return []

    print("\n" + "=" * 72)
    print("第 2 步：展开板块选择已掌握的知识点")
    print("  对每个板块，输入知识点编号（逗号分隔）")
    print("  直接回车表示该板块下无已掌握知识点")
    print("=" * 72)

    mastered: list[str] = []
    blocks_by_id = {b["block_id"]: b for b in blocks}
    for block_id in selected_block_ids:
        block = blocks_by_id.get(block_id)
        if not block:
            continue
        subtopics = block.get("subtopics", [])
        if not subtopics:
            print(f"\n  {block['block_name']}：无知识点数据")
            continue

        print(f"\n  {block['block_id']}: {block['block_name']}")
        for i, sub in enumerate(subtopics, 1):
            name = _shorten_subtopic_name(sub.get("subtopic_name", ""))
            difficulty = sub.get("difficulty", "")
            hours = sub.get("estimated_hours", 0)
            print(f"    [{i}] {sub['subtopic_id']}: {name}（{difficulty}，{hours}h）")

        raw = input(f"  已掌握的知识点编号: ").strip()
        if not raw:
            continue
        # 支持逗号、空格、中文逗号、分号多种分隔
        for sep in [" ", "，", ",", ";", "；", "、"]:
            raw = raw.replace(sep, ",")
        for part in raw.split(","):
            part = part.strip()
            if part.isdigit():
                idx = int(part)
                if 1 <= idx <= len(subtopics):
                    mastered.append(subtopics[idx - 1]["subtopic_id"])

    return mastered


def display_foundation(mastered_subtopic_ids: list[str], blocks: list[dict]) -> None:
    """显示用户确认的基础信息（仅显示完全掌握的板块）。"""
    print("\n" + "=" * 72)
    print("基础信息确认")
    print("=" * 72)

    if not mastered_subtopic_ids:
        print("  无已掌握知识点")
        return

    mastered_set = set(mastered_subtopic_ids)
    fully_mastered_blocks: list[str] = []
    partial_blocks: list[str] = []

    for block in blocks:
        subtopics = block.get("subtopics", [])
        if not subtopics:
            continue
        all_ids = {s["subtopic_id"] for s in subtopics}
        mastered_in_block = all_ids & mastered_set
        if mastered_in_block == all_ids:
            fully_mastered_blocks.append(block["block_name"])
        elif mastered_in_block:
            partial_blocks.append(f"{block['block_name']}（{len(mastered_in_block)}/{len(all_ids)}）")

    if fully_mastered_blocks:
        print("  已掌握板块：")
        for name in fully_mastered_blocks:
            display = name if len(name) <= 30 else name[:27] + "..."
            print(f"    - {display}")
    else:
        print("  已掌握板块：无")

    if partial_blocks:
        print("  部分掌握板块：")
        for info in partial_blocks:
            print(f"    - {info}")

    print(f"  已掌握知识点总数：{len(mastered_subtopic_ids)}")
    print("  （可输入 reset 重新确定基础）")


async def main() -> None:
    print("=" * 72)
    print("学情诊断与阶段规划 Agent stage-planning acceptance check (step-by-step foundation)")
    print("Hardware is collected separately and is not sent to 学情诊断与阶段规划 Agent.")
    print("=" * 72)

    goal = prompt("Learning goal", "掌握 LoRA 微调并部署")
    daily_hours_text = prompt("Daily study hours", "2")
    learning_style = prompt("Learning style", "mixed")
    hardware = prompt("Practice hardware/environment (个性化学习资源生成 Agent-only; optional)")
    stage_limit_text = prompt("Maximum stages to display", "30")

    try:
        daily_hours = float(daily_hours_text)
        stage_limit = max(1, int(stage_limit_text))
    except ValueError:
        raise SystemExit("Daily study hours must be numeric and maximum stages must be an integer.")

    practice_profile = PracticeProfile(hardware=hardware)
    session_id = f"manual-agent1-{uuid.uuid4().hex[:8]}"

    with tempfile.TemporaryDirectory(prefix="agent1-planning-") as temp_dir:
        db_path = Path(temp_dir) / "agent1_planning.db"
        client = SQLiteClient(f"sqlite:///{db_path.as_posix()}")
        try:
            insert_knowledge_blocks(client, load_required_list(BLOCKS_FILE_PATH))
            insert_knowledge_subtopics(client, load_required_list(SUBTOPICS_FILE_PATH))
            client.create_session(session_id)
            client.ensure_subtopic_status(session_id)

            agent = DeterministicDiagnosticAgent(db_client=client)

            # === 第 1 步：按目标获取知识板块 ===
            print("\n[学情诊断与阶段规划 Agent] 根据目标检索知识板块 ...")
            blocks = agent.get_goal_blocks(goal)
            print(f"  匹配到 {len(blocks)} 个知识板块")

            if not blocks:
                print("\n未匹配到任何知识板块，请检查目标描述。")
                return

            # === 第 2 步：选择已掌握板块 ===
            selected_block_ids = select_blocks(blocks)

            # === 第 3 步：展开板块选择知识点 ===
            mastered_subtopic_ids = select_subtopics(blocks, selected_block_ids)

            # === 第 4 步：确认基础 ===
            display_foundation(mastered_subtopic_ids, blocks)

            confirm = prompt("\n确认基础并生成阶段？(yes/no/reset)", "yes")
            if confirm.lower() == "reset":
                # 重新选择
                selected_block_ids = select_blocks(blocks)
                mastered_subtopic_ids = select_subtopics(blocks, selected_block_ids)
                display_foundation(mastered_subtopic_ids, blocks)
                confirm = prompt("\n确认基础并生成阶段？(yes/no)", "yes")

            if confirm.lower() not in ("yes", "y", ""):
                print("已取消。")
                return

            # === 第 5 步：学情诊断与阶段规划 Agent 用 foundation 构建 profile ===
            print("\n[学情诊断与阶段规划 Agent] 构建画像 ...")
            diagnostic = await agent.run_with_foundation(
                session_id=session_id,
                goal=goal,
                mastered_subtopic_ids=mastered_subtopic_ids,
                daily_hours=daily_hours,
                learning_style=learning_style,
            )
            profile = diagnostic.profile

            print("\nLearning profile:")
            print(profile.model_dump_json(indent=2))
            print("\nPractice profile retained outside 学情诊断与阶段规划 Agent:")
            print(practice_profile.model_dump_json(indent=2))
            print("\n学情诊断与阶段规划 Agent knowledge gaps:")
            for gap in profile.knowledge_gaps:
                print(f"- {gap.block_id}: {gap.skill}; targets={gap.target_subtopic_ids}")

            if not profile.knowledge_gaps:
                print("\nNo mapped knowledge gap.")
                return

            target_ids = {
                subtopic_id
                for gap in profile.knowledge_gaps
                for subtopic_id in gap.target_subtopic_ids
            }

            # === 第 6 步：阶段规划 ===
            print("\nPlanned path (each displayed stage is then marked complete for simulation):")

            for stage_num in range(1, stage_limit + 1):
                result = await agent.generate_stage(StagePlanningInput(
                    session_id=session_id,
                    profile=profile,
                    stage_num=stage_num,
                ))
                stage = result.stage
                print("\n" + format_stage(stage))

                if not stage.subtopic_id and not stage.subtopic_ids:
                    print("Stopped: 学情诊断与阶段规划 Agent returned a fallback stage without a catalog subtopic.")
                    break

                # 标记阶段内所有知识点为已完成
                for sub_id in (stage.subtopic_ids or [stage.subtopic_id]):
                    if sub_id:
                        client.set_subtopic_status(session_id, sub_id, 1, 1.0, 1.0, False)
                if stage.subtopic_id in target_ids or any(sid in target_ids for sid in (stage.subtopic_ids or [])):
                    print("\nReached the mapped target knowledge point.")
                    break
            else:
                print("\nStopped at the configured maximum stage count.")
        finally:
            client.close()


if __name__ == "__main__":
    asyncio.run(main())
