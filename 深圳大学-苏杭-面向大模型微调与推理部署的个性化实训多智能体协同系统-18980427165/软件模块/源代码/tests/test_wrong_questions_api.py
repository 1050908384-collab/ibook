# -*- coding: utf-8 -*-
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from api import assessment


class WrongQuestionsApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_submitted_wrong_questions_grouped_by_stage(self):
        stages = [
            {
                "stage_id": "stage-1",
                "stage_num": 1,
                "block_name": "Python基础",
                "accuracy": 0.5,
                "wrong_count": 1,
                "questions": [{
                    "attempt_id": "attempt-1",
                    "question_id": "q-1",
                    "sequence_index": 0,
                    "question_type": "single_choice",
                    "question_text": "以下哪项正确？",
                    "user_answer": "A",
                    "correct_answer": "B",
                    "explanation": "B符合题目条件。",
                    "subtopic_id": "ST-01",
                }],
            }
        ]
        fake_service = SimpleNamespace(
            db=SimpleNamespace(get_wrong_questions_by_stage=lambda session_id: stages)
        )
        with patch.object(assessment, "assessment_service", fake_service):
            result = await assessment.get_wrong_questions("session-1", _user_id="user-1")

        self.assertEqual("session-1", result["session_id"])
        self.assertEqual(1, result["wrong_count"])
        self.assertEqual(1, len(result["stages"]))
        self.assertEqual("stage-1", result["stages"][0]["stage_id"])
        self.assertEqual("B", result["stages"][0]["questions"][0]["correct_answer"])

    async def test_empty_stage_result_has_zero_count(self):
        fake_service = SimpleNamespace(
            db=SimpleNamespace(get_wrong_questions_by_stage=lambda session_id: [])
        )
        with patch.object(assessment, "assessment_service", fake_service):
            result = await assessment.get_wrong_questions("session-empty", _user_id="user-1")

        self.assertEqual({"session_id": "session-empty", "wrong_count": 0, "stages": []}, result)


if __name__ == "__main__":
    unittest.main()
