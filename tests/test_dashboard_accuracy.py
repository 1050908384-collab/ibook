import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from api.dashboard import _build_dashboard


class DashboardAccuracyTest(unittest.TestCase):
    def test_average_accuracy_uses_each_stage_result_once(self):
        dashboard = _build_dashboard("goal-a", _DashboardDb([0.6, 0.8, 1.0, None]))

        self.assertEqual(0.8, dashboard["average_accuracy"])
        self.assertEqual(0.8, dashboard["current_accuracy"])

    def test_average_accuracy_is_none_without_stage_results(self):
        dashboard = _build_dashboard("goal-a", _DashboardDb([None]))

        self.assertIsNone(dashboard["average_accuracy"])
        self.assertIsNone(dashboard["current_accuracy"])


class _DashboardDb:
    def __init__(self, accuracies):
        self.stages = [
            {
                "stage_id": f"stage-{index}",
                "stage_num": index,
                "block_id": "KB-01",
                "block_name": "模块",
                "subtopic_id": f"KBS-01-{index:02d}",
                "estimated_hours": 1.0,
                "status": "completed",
                "accuracy": accuracy,
                "strategy_used": None,
                "completed_at": None,
            }
            for index, accuracy in enumerate(accuracies, start=1)
        ]

    def session_exists(self, session_id):
        return True

    def get_latest_profile(self, session_id):
        return {"goal": "目标", "knowledge_gaps": []}

    def list_stage_progress(self, session_id):
        return self.stages

    def get_subtopic_mastery_counts(self, session_id):
        return 0, 0

    def get_block_progress(self, session_id):
        return []

    def list_assessment_attempts(self, session_id):
        return []

    def get_recent_qa(self, session_id, limit=5):
        return []


if __name__ == "__main__":
    unittest.main()
