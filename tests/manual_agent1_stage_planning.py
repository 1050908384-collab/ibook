"""Interactive acceptance check for Agent1 hard-prerequisite stage planning.

Run from the project root:
    python tests/manual_agent1_stage_planning.py

The script creates an isolated SQLite database, loads the current knowledge
seeds, and simulates completion after each displayed stage.  Hardware is kept
as a PracticeProfile for visibility only; it is intentionally never passed to
Agent1.
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

from agents.deterministic import DeterministicDiagnosticAgent
from db.sqlite_client import SQLiteClient
from models.contracts import DiagnosticInput, StagePlanningInput
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
    return (
        f"Stage {stage.stage_num}: {stage.block_id} / {stage.subtopic_id}\n"
        f"  Block: {stage.block_name}\n"
        f"  Subtopic: {stage.subtopic_name}\n"
        f"  Estimated hours: {stage.estimated_hours}\n"
        f"  Daily study time: {stage.recommended_daily_hours}\n"
        f"  Suggested study days: {stage.suggested_study_days}\n"
        f"  Direct hard prerequisites: {prerequisites}"
    )


async def main() -> None:
    print("=" * 72)
    print("Agent1 stage-planning acceptance check")
    print("Hardware is collected separately and is not sent to Agent1.")
    print("=" * 72)

    goal = prompt("Learning goal", "I want to learn LoRA")
    mastered = prompt("Mastered skills (comma-separated; optional)")
    daily_hours_text = prompt("Daily study hours", "2")
    learning_style = prompt("Learning style", "mixed")
    hardware = prompt("Practice hardware/environment (Agent3-only; optional)")
    stage_limit_text = prompt("Maximum stages to display", "30")

    try:
        daily_hours = float(daily_hours_text)
        stage_limit = max(1, int(stage_limit_text))
    except ValueError:
        raise SystemExit("Daily study hours must be numeric and maximum stages must be an integer.")

    practice_profile = PracticeProfile(hardware=hardware)
    session_id = f"manual-agent1-{uuid.uuid4().hex[:8]}"
    user_input = "\n".join((
        f"学习目标：{goal}",
        f"已掌握基础：{mastered or '无'}",
        f"每日学习时长：{daily_hours}小时",
        f"学习方式：{learning_style}",
    ))

    with tempfile.TemporaryDirectory(prefix="agent1-planning-") as temp_dir:
        db_path = Path(temp_dir) / "agent1_planning.db"
        client = SQLiteClient(f"sqlite:///{db_path.as_posix()}")
        try:
            insert_knowledge_blocks(client, load_required_list(BLOCKS_FILE_PATH))
            insert_knowledge_subtopics(client, load_required_list(SUBTOPICS_FILE_PATH))
            client.create_session(session_id)

            agent = DeterministicDiagnosticAgent(db_client=client)
            diagnostic = await agent.run(DiagnosticInput(session_id=session_id, user_input=user_input))
            profile = diagnostic.profile.model_copy(update={
                "daily_hours": daily_hours,
                "learning_style": learning_style,
            })

            print("\nLearning profile supplied to Agent1:")
            print(profile.model_dump_json(indent=2))
            print("\nPractice profile retained outside Agent1:")
            print(practice_profile.model_dump_json(indent=2))
            print("\nAgent1 knowledge gaps:")
            for gap in profile.knowledge_gaps:
                print(f"- {gap.block_id}: {gap.skill}; targets={gap.target_subtopic_ids}")

            if not profile.knowledge_gaps:
                print("\nNo mapped knowledge gap. Answer Agent1's follow-up questions and rerun.")
                for question in diagnostic.followup_questions:
                    print(f"- {question}")
                return

            target_ids = {
                subtopic_id
                for gap in profile.knowledge_gaps
                for subtopic_id in gap.target_subtopic_ids
            }
            print("\nPlanned path (each displayed stage is then marked complete for simulation):")

            for stage_num in range(1, stage_limit + 1):
                result = await agent.generate_stage(StagePlanningInput(
                    session_id=session_id,
                    profile=profile,
                    stage_num=stage_num,
                ))
                stage = result.stage
                print("\n" + format_stage(stage))

                if not stage.subtopic_id:
                    print("Stopped: Agent1 returned a fallback stage without a catalog subtopic.")
                    break

                client.set_subtopic_status(
                    session_id, stage.subtopic_id, 1, 1.0, 1.0, False
                )
                if stage.subtopic_id in target_ids:
                    print("\nReached the mapped target knowledge point.")
                    break
            else:
                print("\nStopped at the configured maximum stage count.")
        finally:
            client.close()


if __name__ == "__main__":
    asyncio.run(main())
