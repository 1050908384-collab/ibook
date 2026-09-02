import sys
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BACKEND_DIR))

from sqlalchemy import text
from backend.db.sqlite_client import SQLiteClient


SEEDS_DIR = Path(__file__).resolve().parent.parent / "data" / "seeds"
BLOCKS_FILE_PATH = SEEDS_DIR / "knowledge_blocks.json"
DOCS_FILE_PATH = SEEDS_DIR / "knowledge_docs.json"
SUBTOPICS_FILE_PATH = SEEDS_DIR / "knowledge_subtopics.json"
RELATIONS_FILE_PATH = SEEDS_DIR / "knowledge_graph_relations.json"
QUESTIONS_FILE_PATH = SEEDS_DIR / "test_questions.json"
VIDEO_CATALOG_PATH = PROJECT_ROOT / "data" / "knowledge_base" / "video_resources.json"
RETIRED_SUBTOPIC_REPLACEMENTS = {
    "KBS-01-05": "KBS-01-04",
    "KBS-02-06": "KBS-02-05",
    "KBS-03-05": "KBS-03-04",
    "KBS-09-06": "KBS-09-05",
    "KBS-10-04": "KBS-10-03",
    "KBS-12-06": "KBS-12-05",
}
LEGACY_BLOCK_REPLACEMENTS = {
    "KB-001": "KB-02",
    "KB-002": "KB-04",
    "KB-003": "KB-08",
    "KB-004": "KB-12",
    "KB-005": "KB-15",
}

def load_seed_data(file_path: Path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_required_list(file_path: Path) -> list:
    if not file_path.exists():
        print("[ERROR] 种子文件不存在: {}".format(file_path))
        sys.exit(1)
    payload = load_seed_data(file_path)
    if not isinstance(payload, list):
        print("[ERROR] 种子文件格式错误，应为 JSON 数组: {}".format(file_path))
        sys.exit(1)
    return payload


def load_video_catalog(file_path: Path) -> tuple[list, list, list]:
    if not file_path.exists():
        print("[ERROR] 视频资源目录不存在: {}".format(file_path))
        sys.exit(1)
    catalog = load_seed_data(file_path)
    resources = catalog.get("resources") if isinstance(catalog, dict) else None
    mappings = catalog.get("subtopic_mappings") if isinstance(catalog, dict) else None
    block_mappings = catalog.get("block_mappings") if isinstance(catalog, dict) else None
    if not isinstance(resources, list) or not isinstance(mappings, list) or not isinstance(block_mappings, list):
        print("[ERROR] 视频资源目录格式错误: {}".format(file_path))
        sys.exit(1)
    return resources, mappings, block_mappings


def insert_knowledge_blocks(db: SQLiteClient, blocks_data: list):
    count = 0
    for block in blocks_data:
        with db.Session() as session:
            session.execute(text("""
                INSERT INTO knowledge_blocks (
                    block_id, block_name, description, estimated_hours,
                    difficulty, prerequisites, corresponding_gaps
                ) VALUES (
                    :block_id, :block_name, :description, :estimated_hours,
                    :difficulty, :prerequisites, :corresponding_gaps
                )
                ON CONFLICT(block_id) DO UPDATE SET
                    block_name=excluded.block_name,
                    description=excluded.description,
                    estimated_hours=excluded.estimated_hours,
                    difficulty=excluded.difficulty,
                    prerequisites=excluded.prerequisites,
                    corresponding_gaps=excluded.corresponding_gaps
            """), {
                "block_id": block["block_id"],
                "block_name": block["block_name"],
                "description": block["description"],
                "estimated_hours": block["estimated_hours"],
                "difficulty": block["difficulty"],
                "prerequisites": json.dumps(block["prerequisites"], ensure_ascii=False),
                "corresponding_gaps": json.dumps(block["corresponding_gaps"], ensure_ascii=False)
            })
            session.commit()
        count += 1
    print("[OK] knowledge_blocks: 已插入 {} 条记录".format(count))


def migrate_legacy_blocks(db: SQLiteClient) -> None:
    """Remove obsolete KB-00x aliases after preserving their linked records."""
    with db.Session() as session:
        for legacy_id, active_id in LEGACY_BLOCK_REPLACEMENTS.items():
            exists = session.execute(
                text("SELECT 1 FROM knowledge_blocks WHERE block_id = :block_id"),
                {"block_id": legacy_id},
            ).scalar()
            if not exists:
                continue
            for table in ("knowledge_docs", "test_questions", "stage_progress", "subtopic_status", "subtopic_resources", "block_resources"):
                session.execute(
                    text(f"UPDATE {table} SET block_id = :active_id WHERE block_id = :legacy_id"),
                    {"active_id": active_id, "legacy_id": legacy_id},
                )
            session.execute(
                text("DELETE FROM knowledge_blocks WHERE block_id = :block_id"),
                {"block_id": legacy_id},
            )
        session.commit()


def insert_knowledge_docs(db: SQLiteClient, docs_data: list):
    count = 0
    for doc in docs_data:
        with db.Session() as session:
            session.execute(text("""
                INSERT INTO knowledge_docs (
                    doc_id, block_id, subtopic_id, title, content, doc_type, source, difficulty
                ) VALUES (
                    :doc_id, :block_id, :subtopic_id, :title, :content, :doc_type, :source, :difficulty
                )
                ON CONFLICT(doc_id) DO UPDATE SET
                    block_id=excluded.block_id,
                    subtopic_id=excluded.subtopic_id,
                    title=excluded.title,
                    content=excluded.content,
                    doc_type=excluded.doc_type,
                    source=excluded.source,
                    difficulty=excluded.difficulty
            """), {
                "doc_id": doc["doc_id"],
                "block_id": doc["block_id"],
                "subtopic_id": doc.get("subtopic_id"),
                "title": doc["title"],
                "content": doc["content"],
                "doc_type": doc["doc_type"],
                "source": doc["source"],
                "difficulty": doc["difficulty"]
            })
            session.commit()
        count += 1
    print("[OK] knowledge_docs: 已插入 {} 条记录".format(count))


def insert_knowledge_subtopics(db: SQLiteClient, subtopics_data: list):
    relationship_data = load_seed_data(RELATIONS_FILE_PATH)
    hard_prerequisites = relationship_data.get("hard_prerequisites", {})
    count = 0
    for subtopic in subtopics_data:
        with db.Session() as session:
            session.execute(text("""
                INSERT INTO knowledge_subtopics (
                    subtopic_id, block_id, subtopic_name, estimated_hours,
                    sequence_index, difficulty, prerequisites, content_summary
                ) VALUES (
                    :subtopic_id, :block_id, :subtopic_name, :estimated_hours,
                    :sequence_index, :difficulty, :prerequisites, :content_summary
                )
                ON CONFLICT(subtopic_id) DO UPDATE SET
                    block_id=excluded.block_id,
                    subtopic_name=excluded.subtopic_name,
                    estimated_hours=excluded.estimated_hours,
                    sequence_index=excluded.sequence_index,
                    difficulty=excluded.difficulty,
                    prerequisites=excluded.prerequisites,
                    content_summary=excluded.content_summary
            """), {
                "subtopic_id": subtopic["subtopic_id"],
                "block_id": subtopic["block_id"],
                "subtopic_name": subtopic["subtopic_name"],
                "estimated_hours": subtopic["estimated_hours"],
                "sequence_index": subtopic["sequence_index"],
                "difficulty": subtopic.get("difficulty"),
                "prerequisites": json.dumps(hard_prerequisites.get(subtopic["subtopic_id"], subtopic.get("prerequisites", [])), ensure_ascii=False),
                "content_summary": subtopic.get("content_summary", ""),
            })
            session.commit()
        count += 1
    print("[OK] knowledge_subtopics: 已插入 {} 条记录".format(count))


def remove_retired_subtopics(db: SQLiteClient, active_subtopic_ids: set[str]) -> None:
    """Migrate references and remove obsolete catalogue nodes."""
    with db.Session() as session:
        rows = session.execute(text("SELECT subtopic_id FROM knowledge_subtopics")).mappings().all()
        retired_ids = [row["subtopic_id"] for row in rows if row["subtopic_id"] not in active_subtopic_ids]
        if retired_ids:
            # Old installations can contain assessment snapshots and other
            # historical tables that reference retired IDs. The authoritative
            # active catalogue has already been upserted above; retaining
            # these legacy nodes is safer than deleting a learner's history.
            print("[WARN] Retaining {} legacy subtopics for historical records.".format(len(retired_ids)))
            return
        removed = 0
        for subtopic_id in retired_ids:
            replacement_id = RETIRED_SUBTOPIC_REPLACEMENTS.get(subtopic_id)
            if replacement_id is None:
                raise ValueError("未定义过期知识点迁移目标: {}".format(subtopic_id))
            session.execute(text("UPDATE knowledge_docs SET subtopic_id = :replacement_id WHERE subtopic_id = :subtopic_id"), {"replacement_id": replacement_id, "subtopic_id": subtopic_id})
            session.execute(text("UPDATE test_questions SET subtopic_id = :replacement_id WHERE subtopic_id = :subtopic_id"), {"replacement_id": replacement_id, "subtopic_id": subtopic_id})
            session.execute(text("UPDATE stage_progress SET subtopic_id = :replacement_id WHERE subtopic_id = :subtopic_id"), {"replacement_id": replacement_id, "subtopic_id": subtopic_id})
            session.execute(text("""
                INSERT INTO subtopic_status (
                    session_id, subtopic_id, block_id, status, mastery_score,
                    last_accuracy, attempt_count, updated_at, started_at, completed_at
                )
                SELECT session_id, :replacement_id, block_id, status, mastery_score,
                       last_accuracy, attempt_count, updated_at, started_at, completed_at
                FROM subtopic_status WHERE subtopic_id = :subtopic_id
                ON CONFLICT(session_id, subtopic_id) DO UPDATE SET
                    status = MAX(subtopic_status.status, excluded.status),
                    mastery_score = MAX(subtopic_status.mastery_score, excluded.mastery_score),
                    attempt_count = subtopic_status.attempt_count + excluded.attempt_count,
                    updated_at = MAX(subtopic_status.updated_at, excluded.updated_at)
            """), {"replacement_id": replacement_id, "subtopic_id": subtopic_id})
            session.execute(text("DELETE FROM subtopic_status WHERE subtopic_id = :subtopic_id"), {"subtopic_id": subtopic_id})
            # user_subtopic_status is keyed by user_id rather than session_id,
            # so it needs the same merge treatment before the legacy node can
            # be removed without violating its foreign key.
            user_status_exists = session.execute(text(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'user_subtopic_status'"
            )).scalar()
            if user_status_exists:
                session.execute(text("""
                    INSERT INTO user_subtopic_status (
                        user_id, subtopic_id, block_id, status, mastery_score,
                        is_initial_foundation, attempt_count, last_accuracy,
                        updated_at, started_at, completed_at
                    )
                    SELECT user_id, :replacement_id, block_id, status, mastery_score,
                           is_initial_foundation, attempt_count, last_accuracy,
                           updated_at, started_at, completed_at
                    FROM user_subtopic_status WHERE subtopic_id = :subtopic_id
                    ON CONFLICT(user_id, subtopic_id) DO UPDATE SET
                        status = MAX(user_subtopic_status.status, excluded.status),
                        mastery_score = MAX(user_subtopic_status.mastery_score, excluded.mastery_score),
                        attempt_count = user_subtopic_status.attempt_count + excluded.attempt_count,
                        last_accuracy = COALESCE(excluded.last_accuracy, user_subtopic_status.last_accuracy),
                        updated_at = MAX(user_subtopic_status.updated_at, excluded.updated_at)
                """), {"replacement_id": replacement_id, "subtopic_id": subtopic_id})
                session.execute(
                    text("DELETE FROM user_subtopic_status WHERE subtopic_id = :subtopic_id"),
                    {"subtopic_id": subtopic_id},
                )
            session.execute(text("DELETE FROM subtopic_resources WHERE subtopic_id = :subtopic_id"), {"subtopic_id": subtopic_id})

            # Assessment snapshots and later schema additions can also point
            # directly at a subtopic. Migrate every declared FK instead of
            # maintaining a fragile, incomplete table-name list.
            table_names = [
                row[0]
                for row in session.execute(
                    text("SELECT name FROM sqlite_master WHERE type = 'table'")
                ).fetchall()
            ]
            skipped_tables = {
                "knowledge_subtopics",
                "subtopic_status",
                "subtopic_resources",
                "user_subtopic_status",
            }
            for table_name in table_names:
                if table_name in skipped_tables:
                    continue
                safe_table = '"' + table_name.replace('"', '""') + '"'
                foreign_keys = session.execute(text(f"PRAGMA foreign_key_list({safe_table})")).fetchall()
                for foreign_key in foreign_keys:
                    _, _, target_table, from_column, to_column, *_ = foreign_key
                    if target_table != "knowledge_subtopics" or to_column != "subtopic_id":
                        continue
                    safe_column = '"' + from_column.replace('"', '""') + '"'
                    session.execute(text(
                        f"UPDATE {safe_table} SET {safe_column} = :replacement_id "
                        f"WHERE {safe_column} = :subtopic_id"
                    ), {"replacement_id": replacement_id, "subtopic_id": subtopic_id})
            session.execute(text("DELETE FROM knowledge_subtopics WHERE subtopic_id = :subtopic_id"), {"subtopic_id": subtopic_id})
            removed += 1
        session.commit()
    print("[OK] 已迁移并删除 {} 个过期知识点".format(removed))


def insert_test_questions(db: SQLiteClient, questions_data: list):
    """Replace the shipped question bank while preserving user-added questions.

    Earlier database initialization inserted a small ``legacy_seed`` bank and
    a few rows without a source or subtopic.  Those rows cannot be traced to
    the current knowledge graph, so they must not remain eligible for runtime
    extraction after the canonical source bank is imported.
    """
    managed_sources = ("测试题目.md", "runtime_supplement", "legacy_seed", "test_questions.json")
    with db.Session() as session:
        placeholders = ", ".join(f":source_{index}" for index in range(len(managed_sources)))
        params = {f"source_{index}": source for index, source in enumerate(managed_sources)}
        session.execute(text(
            f"DELETE FROM test_questions WHERE source IS NULL OR source IN ({placeholders})"
        ), params)
        for question in questions_data:
            session.execute(text("""
                INSERT INTO test_questions (
                    question_id, block_id, subtopic_id, difficulty, question_type,
                    question_text, options, correct_answer, explanation,
                    expected_output, grading_mode, source, hint
                ) VALUES (
                    :question_id, :block_id, :subtopic_id, :difficulty, :question_type,
                    :question_text, :options, :correct_answer, :explanation,
                    :expected_output, :grading_mode, :source, :hint
                )
                ON CONFLICT(question_id) DO UPDATE SET
                    block_id=excluded.block_id,
                    subtopic_id=excluded.subtopic_id,
                    difficulty=excluded.difficulty,
                    question_type=excluded.question_type,
                    question_text=excluded.question_text,
                    options=excluded.options,
                    correct_answer=excluded.correct_answer,
                    explanation=excluded.explanation,
                    expected_output=excluded.expected_output,
                    grading_mode=excluded.grading_mode,
                    source=excluded.source,
                    hint=excluded.hint
            """), {
                "question_id": question["question_id"],
                "block_id": question["block_id"],
                "subtopic_id": question.get("subtopic_id"),
                "difficulty": question.get("difficulty"),
                "question_type": question.get("question_type"),
                "question_text": question["question_text"],
                "options": json.dumps(question.get("options"), ensure_ascii=False) if question.get("options") else None,
                "correct_answer": question.get("correct_answer", ""),
                "explanation": question.get("explanation"),
                "expected_output": question.get("expected_output"),
                "grading_mode": question.get("grading_mode", "exact_match"),
                "source": question.get("source", "test_questions.json"),
            "hint": question.get("hint") or question.get("explanation") or None,
            })
        session.commit()
    print("[OK] test_questions: 已同步 {} 条记录".format(len(questions_data)))


def insert_learning_resources(db: SQLiteClient, resources_data: list):
    for resource in resources_data:
        with db.Session() as session:
            session.execute(text("""
                INSERT INTO learning_resources (
                    resource_id, resource_type, title, raw_url, canonical_url, platform,
                    external_id, external_part, content_summary, recommendation_level,
                    review_status, metadata_json
                ) VALUES (
                    :resource_id, :resource_type, :title, :raw_url, :canonical_url, :platform,
                    :external_id, :external_part, :content_summary, :recommendation_level,
                    :review_status, :metadata_json
                )
                ON CONFLICT(resource_id) DO UPDATE SET
                    title=excluded.title, raw_url=excluded.raw_url,
                    canonical_url=excluded.canonical_url, platform=excluded.platform,
                    external_id=excluded.external_id, external_part=excluded.external_part,
                    content_summary=excluded.content_summary,
                    recommendation_level=excluded.recommendation_level,
                    review_status=excluded.review_status, metadata_json=excluded.metadata_json,
                    updated_at=CURRENT_TIMESTAMP
            """), {**resource, "metadata_json": json.dumps(resource.get("metadata", {}), ensure_ascii=False)})
            session.commit()
    print("[OK] learning_resources: 已插入 {} 条记录".format(len(resources_data)))


def clear_video_catalog(db: SQLiteClient) -> None:
    """Replace video resources and their mappings as one catalogue snapshot."""
    with db.Session() as session:
        session.execute(text("""
            DELETE FROM subtopic_resources
            WHERE resource_id IN (
                SELECT resource_id FROM learning_resources WHERE resource_type = 'video'
            )
        """))
        session.execute(text("""
            DELETE FROM block_resources
            WHERE resource_id IN (
                SELECT resource_id FROM learning_resources WHERE resource_type = 'video'
            )
        """))
        session.execute(text("DELETE FROM learning_resources WHERE resource_type = 'video'"))
        session.commit()
    print("[OK] 已清理旧视频资源目录与关联映射")


def insert_subtopic_resources(db: SQLiteClient, mappings_data: list):
    for mapping in mappings_data:
        with db.Session() as session:
            session.execute(text("""
                INSERT INTO subtopic_resources (
                    resource_id, block_id, subtopic_id, role, sequence_index,
                    relevance_score, mapping_status, mapping_notes
                ) VALUES (
                    :resource_id, :block_id, :subtopic_id, :role, :sequence_index,
                    :relevance_score, :mapping_status, :mapping_notes
                )
                ON CONFLICT(resource_id, subtopic_id) DO UPDATE SET
                    block_id=excluded.block_id, role=excluded.role,
                    sequence_index=excluded.sequence_index,
                    relevance_score=excluded.relevance_score,
                    mapping_status=excluded.mapping_status,
                    mapping_notes=excluded.mapping_notes,
                    updated_at=CURRENT_TIMESTAMP
            """), mapping)
            session.commit()
    print("[OK] subtopic_resources: 已插入 {} 条记录".format(len(mappings_data)))


def insert_block_resources(db: SQLiteClient, mappings_data: list):
    for mapping in mappings_data:
        with db.Session() as session:
            session.execute(text("""
                INSERT INTO block_resources (
                    resource_id, block_id, role, sequence_index,
                    relevance_score, mapping_status, mapping_notes
                ) VALUES (
                    :resource_id, :block_id, :role, :sequence_index,
                    :relevance_score, :mapping_status, :mapping_notes
                )
                ON CONFLICT(resource_id, block_id) DO UPDATE SET
                    role=excluded.role, sequence_index=excluded.sequence_index,
                    relevance_score=excluded.relevance_score,
                    mapping_status=excluded.mapping_status,
                    mapping_notes=excluded.mapping_notes,
                    updated_at=CURRENT_TIMESTAMP
            """), mapping)
            session.commit()
    print("[OK] block_resources: 已插入 {} 条记录".format(len(mappings_data)))


def sync_video_catalog(db: SQLiteClient) -> None:
    """Replace only the video catalogue and its mappings.

    This is intentionally separate from ``main`` so an existing user database
    can be upgraded with newly shipped videos without re-seeding all curriculum
    records.
    """
    resources_data, subtopic_resources_data, block_resources_data = load_video_catalog(VIDEO_CATALOG_PATH)
    clear_video_catalog(db)
    insert_learning_resources(db, resources_data)
    insert_subtopic_resources(db, subtopic_resources_data)
    insert_block_resources(db, block_resources_data)


def main():
    print("[INFO] 正在加载知识种子数据...")
    blocks_data = load_required_list(BLOCKS_FILE_PATH)
    docs_data = load_required_list(DOCS_FILE_PATH)
    subtopics_data = load_required_list(SUBTOPICS_FILE_PATH)
    questions_data = load_required_list(QUESTIONS_FILE_PATH)

    print("[INFO] 正在初始化数据库连接...")
    db = SQLiteClient()

    print("[INFO] 1. 插入知识块 (knowledge_blocks)...")
    insert_knowledge_blocks(db, blocks_data)
    migrate_legacy_blocks(db)

    print("[INFO] 2. 插入知识文档 (knowledge_docs)...")
    insert_knowledge_docs(db, docs_data)

    print("[INFO] 3. 插入子知识点 (knowledge_subtopics)...")
    insert_knowledge_subtopics(db, subtopics_data)
    remove_retired_subtopics(db, {item["subtopic_id"] for item in subtopics_data})

    print("[INFO] 4. 插入测试题 (test_questions)...")
    insert_test_questions(db, questions_data)

    print("[INFO] 5. 插入学习资源数据 (learning_resources)...")
    sync_video_catalog(db)

    db.close()
    print("[DONE] 知识库数据导入完成！")


if __name__ == "__main__":
    main()
