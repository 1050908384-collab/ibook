import sys
import json
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
                    doc_id, block_id, title, content, doc_type, source, difficulty
                ) VALUES (
                    :doc_id, :block_id, :title, :content, :doc_type, :source, :difficulty
                )
            """), {
                "doc_id": doc["doc_id"],
                "block_id": doc["block_id"],
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
                    question_id, block_id, difficulty, question_type,
                    question_text, options, correct_answer, hint
                ) VALUES (
                    :question_id, :block_id, :difficulty, :question_type,
                    :question_text, :options, :correct_answer, :hint
                )
            """), {
                "question_id": question["question_id"],
                "block_id": question["block_id"],
                "difficulty": question["difficulty"],
                "question_type": question["question_type"],
                "question_text": question["question_text"],
                "options": json.dumps(question["options"], ensure_ascii=False) if question["options"] else None,
                "correct_answer": question["correct_answer"],
                "hint": question["hint"]
            })
            session.commit()
        count += 1
    print("[OK] test_questions: 已插入 {} 条记录".format(count))


def main():
    print("[INFO] 正在加载知识种子数据...")
    blocks_data = load_required_list(BLOCKS_FILE_PATH)
    docs_data = load_required_list(DOCS_FILE_PATH)
    subtopics_data = load_required_list(SUBTOPICS_FILE_PATH)
    faq_data = load_required_list(FAQ_FILE_PATH)
    questions_data = load_required_list(QUESTIONS_FILE_PATH)

    print("[INFO] 正在初始化数据库连接...")
    db = SQLiteClient()

    print("[INFO] 1. 插入知识块 (knowledge_blocks)...")
    insert_knowledge_blocks(db, blocks_data)

    print("[INFO] 2. 插入知识文档 (knowledge_docs)...")
    insert_knowledge_docs(db, docs_data)

    print("[INFO] 3. 插入子知识点 (knowledge_subtopics)...")
    insert_knowledge_subtopics(db, subtopics_data)

    print("[INFO] 4. 插入FAQ数据 (faq_items)...")
    insert_faq_items(db, faq_data)

    print("[INFO] 5. 插入测试题 (test_questions)...")
    insert_test_questions(db, questions_data)

    db.close()
    print("[DONE] 知识库数据导入完成！")


if __name__ == "__main__":
    main()
