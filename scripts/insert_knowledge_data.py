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
FAQ_FILE_PATH = SEEDS_DIR / "faq_items.json"
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
                INSERT OR REPLACE INTO knowledge_blocks (
                    block_id, block_name, description, estimated_hours,
                    difficulty, prerequisites, corresponding_gaps
                ) VALUES (
                    :block_id, :block_name, :description, :estimated_hours,
                    :difficulty, :prerequisites, :corresponding_gaps
                )
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


def insert_knowledge_docs(db: SQLiteClient, docs_data: list):
    count = 0
    for doc in docs_data:
        with db.Session() as session:
            session.execute(text("""
                INSERT OR REPLACE INTO knowledge_docs (
                    doc_id, block_id, subtopic_id, title, content, doc_type, source, difficulty
                ) VALUES (
                    :doc_id, :block_id, :subtopic_id, :title, :content, :doc_type, :source, :difficulty
                )
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
    count = 0
    for subtopic in subtopics_data:
        with db.Session() as session:
            session.execute(text("""
                INSERT OR REPLACE INTO knowledge_subtopics (
                    subtopic_id, block_id, subtopic_name, estimated_hours,
                    sequence_index, difficulty, prerequisites, content_summary
                ) VALUES (
                    :subtopic_id, :block_id, :subtopic_name, :estimated_hours,
                    :sequence_index, :difficulty, :prerequisites, :content_summary
                )
            """), {
                "subtopic_id": subtopic["subtopic_id"],
                "block_id": subtopic["block_id"],
                "subtopic_name": subtopic["subtopic_name"],
                "estimated_hours": subtopic["estimated_hours"],
                "sequence_index": subtopic["sequence_index"],
                "difficulty": subtopic.get("difficulty"),
                "prerequisites": json.dumps(subtopic.get("prerequisites", []), ensure_ascii=False),
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
            session.execute(text("DELETE FROM subtopic_resources WHERE subtopic_id = :subtopic_id"), {"subtopic_id": subtopic_id})
            session.execute(text("DELETE FROM knowledge_subtopics WHERE subtopic_id = :subtopic_id"), {"subtopic_id": subtopic_id})
            removed += 1
        session.commit()
    print("[OK] 已迁移并删除 {} 个过期知识点".format(removed))


def insert_faq_items(db: SQLiteClient, faq_data: list):
    count = 0
    for faq in faq_data:
        with db.Session() as session:
            session.execute(text("""
                INSERT OR REPLACE INTO faq_items (
                    faq_id, block_id, question, answer, difficulty, tags
                ) VALUES (
                    :faq_id, :block_id, :question, :answer, :difficulty, :tags
                )
            """), {
                "faq_id": faq["faq_id"],
                "block_id": faq["block_id"],
                "question": faq["question"],
                "answer": faq["answer"],
                "difficulty": faq["difficulty"],
                "tags": json.dumps(faq["tags"], ensure_ascii=False)
            })
            session.commit()
        count += 1
    print("[OK] faq_items: 已插入 {} 条记录".format(count))


def insert_test_questions(db: SQLiteClient, questions_data: list):
    count = 0
    for question in questions_data:
        with db.Session() as session:
            session.execute(text("""
                INSERT OR REPLACE INTO test_questions (
                    question_id, block_id, subtopic_id, difficulty, question_type,
                    question_text, options, correct_answer, explanation,
                    expected_output, grading_mode, source, hint
                ) VALUES (
                    :question_id, :block_id, :subtopic_id, :difficulty, :question_type,
                    :question_text, :options, :correct_answer, :explanation,
                    :expected_output, :grading_mode, :source, :hint
                )
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
        count += 1
    print("[OK] test_questions: 已插入 {} 条记录".format(count))


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


def main():
    print("[INFO] 正在加载知识种子数据...")
    blocks_data = load_required_list(BLOCKS_FILE_PATH)
    docs_data = load_required_list(DOCS_FILE_PATH)
    subtopics_data = load_required_list(SUBTOPICS_FILE_PATH)
    faq_data = load_required_list(FAQ_FILE_PATH)
    questions_data = load_required_list(QUESTIONS_FILE_PATH)
    resources_data, subtopic_resources_data, block_resources_data = load_video_catalog(VIDEO_CATALOG_PATH)

    print("[INFO] 正在初始化数据库连接...")
    db = SQLiteClient()

    print("[INFO] 1. 插入知识块 (knowledge_blocks)...")
    insert_knowledge_blocks(db, blocks_data)

    print("[INFO] 2. 插入知识文档 (knowledge_docs)...")
    insert_knowledge_docs(db, docs_data)

    print("[INFO] 3. 插入子知识点 (knowledge_subtopics)...")
    insert_knowledge_subtopics(db, subtopics_data)
    remove_retired_subtopics(db, {item["subtopic_id"] for item in subtopics_data})

    print("[INFO] 4. 插入FAQ数据 (faq_items)...")
    insert_faq_items(db, faq_data)

    print("[INFO] 5. 插入测试题 (test_questions)...")
    insert_test_questions(db, questions_data)

    print("[INFO] 6. 插入学习资源数据 (learning_resources)...")
    clear_video_catalog(db)
    insert_learning_resources(db, resources_data)

    print("[INFO] 7. 插入资源与子知识点关联 (subtopic_resources)...")
    insert_subtopic_resources(db, subtopic_resources_data)

    print("[INFO] 8. 插入资源与知识板块关联 (block_resources)...")
    insert_block_resources(db, block_resources_data)

    db.close()
    print("[DONE] 知识库数据导入完成！")


if __name__ == "__main__":
    main()
