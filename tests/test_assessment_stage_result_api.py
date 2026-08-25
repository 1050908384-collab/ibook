# -*- coding: utf-8 -*-
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from api import assessment


class StageResultApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_existing_pending_stage_returns_empty_result(self):
        state_manager = SimpleNamespace(
            get_stage_result_by_id=lambda session_id, stage_id: None,
            get_stage_snapshot_by_id=lambda session_id, stage_id: {
                "stage_info": {
                    "stage_id": stage_id,
                    "stage_num": 2,
                    "status": "pending",
                }
            },
            get_stage_result=lambda session_id, stage_num: None,
            get_stage_snapshot=lambda session_id, stage_num: None,
        )
        fake_harness = SimpleNamespace(state_manager=state_manager)
        fake_service = SimpleNamespace(
            stage_allows_persisted_result=lambda stage: False,
            get_submitted_attempt_for_result=lambda session_id, stage, stage_num: None,
        )

        with patch.object(assessment, "harness", fake_harness), patch.object(
            assessment, "assessment_service", fake_service
        ):
            result = await assessment.get_stage_result(
                "session-1",
                2,
                stage_id="stage-2",
                _user_id="user-1",
            )

        self.assertEqual(result["session_id"], "session-1")
        self.assertEqual(result["stage_num"], 2)
        self.assertIsNone(result["stage_result"])
        self.assertIsNone(result["assessment_attempt"])

    async def test_missing_stage_still_returns_404(self):
        state_manager = SimpleNamespace(
            get_stage_result_by_id=lambda session_id, stage_id: None,
            get_stage_snapshot_by_id=lambda session_id, stage_id: None,
            get_stage_result=lambda session_id, stage_num: None,
            get_stage_snapshot=lambda session_id, stage_num: None,
        )
        fake_harness = SimpleNamespace(state_manager=state_manager)
        fake_service = SimpleNamespace(
            stage_allows_persisted_result=lambda stage: False,
            get_submitted_attempt_for_result=lambda session_id, stage, stage_num: None,
        )

        with patch.object(assessment, "harness", fake_harness), patch.object(
            assessment, "assessment_service", fake_service
        ):
            with self.assertRaises(HTTPException) as raised:
                await assessment.get_stage_result(
                    "session-1",
                    99,
                    stage_id="missing-stage",
                    _user_id="user-1",
                )

        self.assertEqual(raised.exception.status_code, 404)
        self.assertEqual(raised.exception.detail, "Stage not found")


if __name__ == "__main__":
    unittest.main()
