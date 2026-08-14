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

    def test_public_learning_package_merges_article_sources_with_videos(self):
        package = to_public_learning_package({
            "recommended_videos": [{"resource_id": "video-1", "title": "Video", "url": "https://example.com/video"}],
            "sources": [{
                "source_id": "doc-1",
                "source_type": "knowledge_doc",
                "title": "Original article",
                "content_ref": "https://example.com/article",
            }],
        })
        self.assertEqual(["video", "article"], [item["resource_type"] for item in package["learning_resources"]])
        self.assertEqual("https://example.com/article", package["learning_resources"][1]["url"])

    def test_resource_backfill_does_not_hide_recommended_videos(self):
        package = to_public_learning_package(
            {"recommended_videos": [{"resource_id": "video-1", "title": "Video", "url": "https://example.com/video"}]},
            learning_resources=[{
                "resource_id": "doc-1",
                "resource_type": "documentation",
                "title": "Document",
                "url": "Official documentation",
            }],
        )
        self.assertEqual(["documentation", "video"], [item["resource_type"] for item in package["learning_resources"]])


if __name__ == "__main__":
    unittest.main()
