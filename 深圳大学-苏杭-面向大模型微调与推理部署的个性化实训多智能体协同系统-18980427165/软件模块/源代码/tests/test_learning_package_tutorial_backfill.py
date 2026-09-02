import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from api.v1 import learning_resources


class _StubVectorSearchTool:
    def get_local_lessons(self, block_id, subtopic_ids):
        return [{
            "doc_id": "LOCAL-KBS-01-00",
            "subtopic_id": "KBS-01-00",
            "title": "Environment",
            "content": "# Environment\n\nIntro.\n\n## Install\n\n```bash\n# shell comment\necho ok\n```",
        }]


class LearningPackageTutorialBackfillTest(unittest.TestCase):
    def test_placeholder_is_replaced_and_nested_headings_preserve_code(self):
        original_tool = learning_resources.VectorSearchTool
        learning_resources.VectorSearchTool = _StubVectorSearchTool
        try:
            package = learning_resources.backfill_stage_tutorial(
                {"tutorial": "本知识点暂未匹配到本地学习资料。", "block_name": "Linux"},
                {
                    "block_id": "KB-01",
                    "block_name": "Linux",
                    "subtopic_ids": ["KBS-01-00"],
                    "subtopic_names": ["Environment setup"],
                },
            )
        finally:
            learning_resources.VectorSearchTool = original_tool

        self.assertIn("## Environment setup", package["tutorial"])
        self.assertIn("### Install", package["tutorial"])
        self.assertIn("# shell comment", package["tutorial"])
        self.assertNotIn("# Environment\n", package["tutorial"])

    def test_existing_tutorial_is_not_replaced(self):
        package = {"tutorial": "# Existing\n\nReal content."}
        self.assertIs(package, learning_resources.backfill_stage_tutorial(package, {}))


if __name__ == "__main__":
    unittest.main()
