import json
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class QuestionSeedCoverageTest(unittest.TestCase):
    def test_every_runtime_subtopic_has_three_questions(self):
        subtopics = json.loads((ROOT / "data" / "seeds" / "knowledge_subtopics.json").read_text(encoding="utf-8"))
        questions = json.loads((ROOT / "data" / "seeds" / "test_questions.json").read_text(encoding="utf-8"))
        counts = Counter(question["subtopic_id"] for question in questions)
        missing = [item["subtopic_id"] for item in subtopics if counts[item["subtopic_id"]] < 3]
        self.assertEqual([], missing)


if __name__ == "__main__":
    unittest.main()
