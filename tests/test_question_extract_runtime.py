import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from tools.agent3_tools.question_extract_tool import question_extract


DATABASE_URL = f"sqlite:///{(PROJECT_ROOT / 'data' / 'sqlite_data' / 'project.db').as_posix()}"


class RuntimeQuestionExtractionTest(unittest.TestCase):
    def extract(self, block_id, subtopic_ids, count=2):
        result = question_extract(
            {
                "block_id": block_id,
                "subtopic_ids": subtopic_ids,
                "count": count,
            },
            config={"use_mock": False},
            database_url=DATABASE_URL,
        )
        self.assertEqual("success", result["status"])
        self.assertTrue(result["data"]["db_available"])
        return result["data"]["questions"]

    def test_installation_subtopic_returns_real_questions(self):
        questions = self.extract("KB-02", ["KBS-02-00"])
        self.assertGreaterEqual(len(questions), 2)
        self.assertTrue(all(item["block_id"] == "KB-02" for item in questions))

    def test_previously_uncovered_subtopic_has_dedicated_questions(self):
        questions = self.extract("KB-15", ["KBS-15-01"])
        self.assertGreaterEqual(len(questions), 2)
        self.assertTrue(all(item["block_id"] == "KB-15" for item in questions))
        self.assertTrue(all(item["subtopic_id"] == "KBS-15-01" for item in questions))


if __name__ == "__main__":
    unittest.main()
