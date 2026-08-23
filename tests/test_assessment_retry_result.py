import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from services.assessment_service import AssessmentService


class _AssessmentDb:
    def __init__(self):
        self.stage_lookup_count = 0
        self.stage_num_lookup_count = 0

    def get_submitted_attempt_accuracy(self, session_id, stage_id=None):
        self.stage_lookup_count += 1
        return {"attempt_id": "attempt-old", "accuracy": 0.4}

    def get_assessment_attempt(self, attempt_id):
        return {
            "attempt_id": attempt_id,
            "session_id": "session-1",
            "stage_id": "stage-1",
            "stage_num": 1,
            "status": "submitted",
            "accuracy": 0.4,
            "total_questions": 1,
            "answered_count": 1,
            "correct_count": 0,
            "created_at": "2026-08-22T00:00:00Z",
            "submitted_at": "2026-08-22T00:05:00Z",
        }

    def get_assessment_attempt_questions(self, attempt_id):
        return []

    def get_assessment_answers(self, attempt_id):
        return {}

    def list_assessment_attempts(self, session_id):
        self.stage_num_lookup_count += 1
        return []


class AssessmentRetryResultTest(unittest.TestCase):
    def test_pending_retry_does_not_restore_previous_submitted_attempt(self):
        database = _AssessmentDb()
        service = AssessmentService(database)

        attempt = service.get_submitted_attempt_for_result(
            "session-1",
            {"stage_id": "stage-1", "stage_num": 1, "status": "pending"},
            1,
        )

        self.assertIsNone(attempt)
        self.assertEqual(0, database.stage_lookup_count)
        self.assertEqual(0, database.stage_num_lookup_count)

    def test_ended_stage_still_restores_its_submitted_attempt(self):
        database = _AssessmentDb()
        service = AssessmentService(database)

        attempt = service.get_submitted_attempt_for_result(
            "session-1",
            {"stage_id": "stage-1", "stage_num": 1, "status": "retreat"},
            1,
        )

        self.assertEqual("attempt-old", attempt["attempt_id"])
        self.assertEqual(1, database.stage_lookup_count)


if __name__ == "__main__":
    unittest.main()
