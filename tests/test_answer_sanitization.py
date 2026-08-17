import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from orchestrator.harness import Harness


class AnswerSanitizationTest(unittest.TestCase):
    def test_removes_internal_review_revision_preamble(self):
        raw = (
            "好的，已根据审核意见对原回答进行修复。主要改动：删除不可靠建议。\n\n"
            "以下是修改后的完整回答：\n\n"
            "在 Windows 上创建虚拟环境可以使用 python -m venv .venv。"
        )
        self.assertEqual(
            "在 Windows 上创建虚拟环境可以使用 python -m venv .venv。",
            Harness._extract_user_facing_answer(raw),
        )


if __name__ == "__main__":
    unittest.main()
