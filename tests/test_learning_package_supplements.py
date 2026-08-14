import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from models.material import TestQuestion
from models.stage import ContentBlock, StageInfo
from orchestrator.harness import Harness
from services.assessment_service import AssessmentService


class _AssessmentDb:
    def list_questions_for_stage(self, block_id, subtopic_ids, limit=2):
        return [{
            "question_id": "q-1",
            "block_id": block_id,
            "subtopic_id": "topic-1",
            "difficulty": "easy",
            "question_type": "single_choice",
            "question_text": "Question",
            "options": ["A", "B"],
            "correct_answer": "A",
            "explanation": "Because",
            "hint": None,
        }]


class _AssessmentState:
    def get_learning_package(self, _session_id):
        return {"questions": []}

    def get_current_stage(self, _session_id):
        return {
            "stage_id": "stage-1",
            "stage_num": 1,
            "block_id": "KB-01",
            "subtopic_id": "topic-1",
        }


class LearningPackageSupplementsTest(unittest.TestCase):
    def test_assessment_falls_back_to_database_for_legacy_empty_package(self):
        service = AssessmentService(_AssessmentDb(), _AssessmentState())

        questions, stage = service._collect_current_package_questions("session-1")

        self.assertEqual("stage-1", stage["stage_id"])
        self.assertEqual(["q-1"], [item["question_id"] for item in questions])
        self.assertTrue(questions[0]["is_required"])

    def test_block_aggregation_preserves_questions_and_videos(self):
        harness = Harness.__new__(Harness)
        stage = StageInfo(
            stage_id="stage-1",
            stage_num=1,
            block_id="KB-01",
            block_name="Linux",
            gap_skill="Linux",
            subtopic_id="topic-1",
        )
        block = ContentBlock(
            content_block_id="content-1",
            session_id="session-1",
            stage_id="stage-1",
            block_id="KB-01",
            block_name="Linux",
            block_index=0,
            personalized_content="Learning text",
        )
        question = TestQuestion(
            question_id="q-1",
            block_id="KB-01",
            subtopic_id="topic-1",
            question_text="Question",
            correct_answer="A",
        )
        video = {
            "resource_id": "video-1",
            "title": "Linux video",
            "url": "https://example.com/video",
            "block_id": "KB-01",
            "subtopic_id": "topic-1",
            "review_status": "confirmed",
        }

        package = harness._aggregate_blocks(
            "session-1", stage, [block], "trace-1",
            questions=[question], recommended_videos=[video],
        )

        self.assertEqual(["q-1"], [item.question_id for item in package.questions])
        self.assertEqual(["video-1"], [item["resource_id"] for item in package.recommended_videos])
        self.assertIn("src_video_video-1", [source.source_id for source in package.sources])


if __name__ == "__main__":
    unittest.main()
