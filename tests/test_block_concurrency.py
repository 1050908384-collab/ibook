import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
sys.path.insert(0, BACKEND_DIR)

from config import settings
from orchestrator.harness import Harness, resolve_block_concurrency


class BlockConcurrencyTests(unittest.TestCase):
    def test_cloud_provider_uses_cloud_limit(self):
        with patch.object(settings, "BLOCK_CONCURRENCY_CLOUD", 5), patch.object(
            settings, "BLOCK_CONCURRENCY_LOCAL", 3
        ):
            self.assertEqual(resolve_block_concurrency("hybrid", "deepseek"), 5)

    def test_local_provider_uses_local_limit(self):
        with patch.object(settings, "BLOCK_CONCURRENCY_CLOUD", 5), patch.object(
            settings, "BLOCK_CONCURRENCY_LOCAL", 3
        ):
            self.assertEqual(resolve_block_concurrency("hybrid", "ollama"), 3)
            self.assertEqual(resolve_block_concurrency("deterministic", "deepseek"), 3)

    def test_missing_cloud_credentials_uses_local_limit(self):
        harness = Harness.__new__(Harness)
        harness.execution_mode = "hybrid"
        harness.agent3 = SimpleNamespace(
            llm_client=SimpleNamespace(_has_deepseek_credentials=lambda: False)
        )
        with patch.object(settings, "LLM_PROVIDER", "deepseek"), patch.object(
            settings, "LLM_FALLBACK_PROVIDER", "ollama"
        ), patch.object(settings, "BLOCK_CONCURRENCY_CLOUD", 5), patch.object(
            settings, "BLOCK_CONCURRENCY_LOCAL", 3
        ):
            self.assertEqual(harness._block_concurrency(), 3)


if __name__ == "__main__":
    unittest.main()