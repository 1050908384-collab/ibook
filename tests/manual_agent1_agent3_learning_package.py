"""Manual acceptance check: Agent1 plans one stage, then Agent3 builds its package.

Run from the project root:
    python tests/manual_agent1_agent3_learning_package.py

The check uses a temporary database.  It never changes the runtime project.db,
but it loads the same catalog, local Markdown lessons, and video mappings used
by the application.
"""
from __future__ import annotations

import asyncio
import argparse
import os
import sys
import tempfile
import uuid
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path[:0] = [str(PROJECT_ROOT), str(BACKEND_DIR)]
os.environ.setdefault("EXECUTION_MODE", "deterministic")

from agents.deterministic import (
    DeterministicContentAgent,
    DeterministicDiagnosticAgent,
    HybridContentAgent,
)
from config import settings
from db.sqlite_client import SQLiteClient
from models.contracts import ContentGenerationInput, DiagnosticInput, StagePlanningInput
from models.profile import PracticeProfile
from scripts.import_local_learning_materials import import_lessons
from scripts.insert_knowledge_data import (
    BLOCKS_FILE_PATH,
    QUESTIONS_FILE_PATH,
    SUBTOPICS_FILE_PATH,
    VIDEO_CATALOG_PATH,
    insert_block_resources,
    insert_knowledge_blocks,
    insert_knowledge_subtopics,
    insert_learning_resources,
    insert_subtopic_resources,
    insert_test_questions,
    load_required_list,
    load_video_catalog,
)


def prompt(label: str, default: str) -> str:
    value = input(f"{label} [{default}]: ").strip()
    return value or default


def seed_test_database(client: SQLiteClient, database_path: Path) -> None:
    insert_knowledge_blocks(client, load_required_list(BLOCKS_FILE_PATH))
    insert_knowledge_subtopics(client, load_required_list(SUBTOPICS_FILE_PATH))
    insert_test_questions(client, load_required_list(QUESTIONS_FILE_PATH))
    resources, subtopic_mappings, block_mappings = load_video_catalog(VIDEO_CATALOG_PATH)
    insert_learning_resources(client, resources)
    insert_subtopic_resources(client, subtopic_mappings)
    insert_block_resources(client, block_mappings)
    documents, chunks = import_lessons(database_path)
    print(f"Loaded {documents} local lessons and {chunks} chunks into the temporary database.")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Agent1 stage planning -> Agent3 learning package")
    parser.add_argument(
        "--mode", choices=("hybrid", "deterministic"), default="hybrid",
        help="hybrid uses the local LLM; deterministic only validates data/retrieval",
    )
    args = parser.parse_args()
    print("Agent1 -> Agent3 learning-package acceptance check")
    goal = prompt("Learning goal", "掌握 LoRA 微调")
    mastered = prompt("Mastered skills", "Python, PyTorch, HuggingFace")
    daily_hours = float(prompt("Daily study hours", "2"))
    learning_style = prompt("Learning style", "hands-on")
    hardware = input("Practice hardware (optional): ").strip()
    user_input = "\n".join((
        "身份：研究生",
        f"学习目标：{goal}",
        f"已掌握基础：{mastered}",
        f"每日学习时长：{daily_hours}小时",
        f"学习方式：{learning_style}",
    ))

    with tempfile.TemporaryDirectory(
        prefix="agent1-agent3-package-", ignore_cleanup_errors=True
    ) as temp_dir:
        database_path = Path(temp_dir) / "learning_package.db"
        database_url = f"sqlite:///{database_path.as_posix()}"
        original_database_url = settings.DATABASE_URL
        settings.DATABASE_URL = database_url
        client = SQLiteClient(database_url)
        try:
            seed_test_database(client, database_path)
            session_id = f"manual-package-{uuid.uuid4().hex[:8]}"
            client.create_session(session_id)

            agent1 = DeterministicDiagnosticAgent(db_client=client)
            diagnostic = await agent1.run(DiagnosticInput(session_id=session_id, user_input=user_input))
            profile = diagnostic.profile.model_copy(update={
                "daily_hours": daily_hours,
                "learning_style": learning_style,
            })
            if not profile.knowledge_gaps:
                raise RuntimeError("Agent1 未映射出知识盲区，请补充学习目标。")

            stage = (await agent1.generate_stage(StagePlanningInput(
                session_id=session_id, profile=profile, stage_num=1,
            ))).stage
            agent3_class = HybridContentAgent if args.mode == "hybrid" else DeterministicContentAgent
            agent3 = agent3_class(database_url=database_url)
            result = await agent3.run(ContentGenerationInput(
                session_id=session_id,
                stage=stage,
                learning_profile=profile,
                practice_profile=PracticeProfile(hardware=hardware),
            ))
            package = result.learning_package

            print("\nSelected stage:")
            stage_topics = stage.subtopic_names or [stage.subtopic_name]
            print(f"  {stage.block_id} / {', '.join(filter(None, stage_topics))}")
            print(f"  Estimated stage hours: {stage.estimated_hours:g} / daily budget: {daily_hours:g}")
            print("\nLearning package:")
            print(package.tutorial)
            print("\nArticle links:")
            seen_article_urls = set()
            for source in package.sources:
                if source.source_type == "knowledge_doc" and source.locator not in seen_article_urls:
                    print(f"  - {source.title}: {source.locator}")
                    seen_article_urls.add(source.locator)
            print("\nVideos:")
            for video in package.recommended_videos:
                print(f"  - {video['title']}: {video['url']}")
        finally:
            client.close()
            settings.DATABASE_URL = original_database_url


if __name__ == "__main__":
    asyncio.run(main())
