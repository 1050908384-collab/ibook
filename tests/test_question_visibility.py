import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from api.v1.schemas import to_public_learning_package, to_public_question


class QuestionVisibilityTest(unittest.TestCase):
    def test_public_question_removes_answer_fields(self):
        question = to_public_question({
            "question_id": "Q-1",
            "question_text": "2+2?",
            "options": ["3", "4"],
            "correct_answer": "4",
            "explanation": "basic arithmetic",
            "reference_answer": "4",
        })
        self.assertEqual("Q-1", question["question_id"])
        self.assertNotIn("correct_answer", question)
        self.assertNotIn("explanation", question)
        self.assertNotIn("reference_answer", question)

    def test_public_learning_package_contains_questions_without_answers(self):
        package = to_public_learning_package({
            "questions": [{
                "question_id": "Q-1",
                "question_text": "2+2?",
                "correct_answer": "4",
                "explanation": "basic arithmetic",
            }],
        })
        question = package["questions"][0]
        self.assertNotIn("correct_answer", question)
        self.assertNotIn("explanation", question)


if __name__ == "__main__":
    unittest.main()
