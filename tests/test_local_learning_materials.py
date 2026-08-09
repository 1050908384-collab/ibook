import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path[:0] = [str(PROJECT_ROOT), str(BACKEND_DIR)]

from scripts.import_local_learning_materials import discover_lessons, import_lessons
from scripts.insert_knowledge_data import (
    BLOCKS_FILE_PATH,
    SUBTOPICS_FILE_PATH,
    insert_knowledge_blocks,
    insert_knowledge_subtopics,
    load_required_list,
)
from db.sqlite_client import SQLiteClient


class LocalLearningMaterialsTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.database_path = Path(self.temp_dir.name) / "materials.db"
        self.client = SQLiteClient(f"sqlite:///{self.database_path.as_posix()}")
        insert_knowledge_blocks(self.client, load_required_list(BLOCKS_FILE_PATH))
        insert_knowledge_subtopics(self.client, load_required_list(SUBTOPICS_FILE_PATH))

    def tearDown(self):
        self.client.close()
        self.temp_dir.cleanup()

    def test_discovers_one_lesson_for_every_seeded_subtopic(self):
        lessons = discover_lessons()
        expected_ids = {
            row["subtopic_id"]
            for row in load_required_list(SUBTOPICS_FILE_PATH)
        }
        self.assertEqual(expected_ids, {lesson.subtopic_id for lesson in lessons})

    def test_import_is_idempotent_and_preserves_subtopic_scope(self):
        self.client.close()
        first_documents, first_chunks = import_lessons(self.database_path)
        second_documents, second_chunks = import_lessons(self.database_path)

        self.assertEqual(73, first_documents)
        self.assertEqual(first_documents, second_documents)
        self.assertGreater(first_chunks, first_documents)
        self.assertEqual(first_chunks, second_chunks)

        with sqlite3.connect(self.database_path) as connection:
            rows = connection.execute(
                "SELECT block_id, subtopic_id, COUNT(*) "
                "FROM knowledge_docs WHERE doc_id LIKE 'LOCAL-%' "
                "GROUP BY block_id, subtopic_id"
            ).fetchall()
            local_count = connection.execute(
                "SELECT COUNT(*) FROM knowledge_docs WHERE doc_id LIKE 'LOCAL-%'"
            ).fetchone()[0]

        self.assertEqual(73, local_count)
        self.assertTrue(all(count == 1 for _, _, count in rows))
        self.assertEqual(
            {"KB-07", "KB-08", "KB-13", "KB-14", "KB-15"},
            {block_id for block_id, _, _ in rows} & {"KB-07", "KB-08", "KB-13", "KB-14", "KB-15"},
        )


if __name__ == "__main__":
    unittest.main()
