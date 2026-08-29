import os
import sys
import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
sys.path.insert(0, BACKEND_DIR)

from config import settings
from orchestrator.harness import AdaptiveConcurrencyLimiter, Harness, resolve_block_concurrency


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

    def test_runtime_fallback_can_shrink_queued_work(self):
        async def scenario():
            limiter = AdaptiveConcurrencyLimiter(5)
            acquired = []
            gate = asyncio.Event()

            async def worker(index):
                await limiter.acquire()
                acquired.append(index)
                await gate.wait()
                limiter.release()

            tasks = [asyncio.create_task(worker(index)) for index in range(6)]
            await asyncio.sleep(0)
            self.assertEqual(len(acquired), 5)
            limiter.set_limit(3)
            self.assertEqual(len(acquired), 5)
            gate.set()
            await asyncio.gather(*tasks)

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
