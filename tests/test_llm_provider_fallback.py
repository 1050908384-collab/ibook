# -*- coding: utf-8 -*-
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from config import settings
from llm.client import LLMClient
from tools.agent3_tools.code_generation_tool import _llm_generate_code
from tools.agent3_tools.llm_generate_tool import LLMGenerateTool, _generate_answer


class LLMProviderFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.original_values = {
            "LLM_PROVIDER": settings.LLM_PROVIDER,
            "LLM_FALLBACK_PROVIDER": settings.LLM_FALLBACK_PROVIDER,
            "DEEPSEEK_MAX_RETRIES": settings.DEEPSEEK_MAX_RETRIES,
        }
        settings.LLM_PROVIDER = "deepseek"
        settings.LLM_FALLBACK_PROVIDER = "ollama"
        settings.DEEPSEEK_MAX_RETRIES = 0

    async def asyncTearDown(self):
        for name, value in self.original_values.items():
            setattr(settings, name, value)

    async def _client_with_transports(self, cloud_handler, ollama_handler):
        client = LLMClient()
        await client.deepseek_client.aclose()
        await client.client.aclose()
        client.deepseek_api_key = "sk-test"
        client.deepseek_client = httpx.AsyncClient(
            transport=httpx.MockTransport(cloud_handler)
        )
        client.client = httpx.AsyncClient(
            transport=httpx.MockTransport(ollama_handler)
        )
        return client

    async def test_complete_falls_back_after_cloud_http_failure(self):
        def cloud_handler(request):
            return httpx.Response(503, request=request, json={"error": {"message": "busy"}})

        def ollama_handler(request):
            return httpx.Response(
                200,
                request=request,
                json={"message": {"content": "local answer"}, "done_reason": "stop"},
            )

        client = await self._client_with_transports(cloud_handler, ollama_handler)
        try:
            result = await client.complete("hello")
        finally:
            await client.deepseek_client.aclose()
            await client.client.aclose()

        self.assertEqual(result, "local answer")

    async def test_chat_falls_back_when_cloud_returns_empty_content(self):
        def cloud_handler(request):
            return httpx.Response(
                200,
                request=request,
                json={"choices": [{"message": {"content": ""}, "finish_reason": "length"}]},
            )

        def ollama_handler(request):
            return httpx.Response(
                200,
                request=request,
                json={"message": {"content": "local chat"}, "done_reason": "stop"},
            )

        client = await self._client_with_transports(cloud_handler, ollama_handler)
        try:
            result = await client.chat([{"role": "user", "content": "hello"}])
        finally:
            await client.deepseek_client.aclose()
            await client.client.aclose()

        self.assertEqual(result, "local chat")

    async def test_legacy_async_adapter_does_not_block_parallel_tasks(self):
        def slow_generate(input_data, config=None):
            time.sleep(0.15)
            return {"status": "success", "data": {"content": input_data["block_id"]}}

        tool = LLMGenerateTool()
        with patch("tools.agent3_tools.llm_generate_tool.llm_generate", side_effect=slow_generate):
            started = time.perf_counter()
            results = await __import__("asyncio").gather(
                tool.generate_tutorial("block-1", {}),
                tool.generate_tutorial("block-2", {}),
            )
            elapsed = time.perf_counter() - started
        await tool.llm_client.deepseek_client.aclose()
        await tool.llm_client.client.aclose()

        self.assertEqual(results, ["block-1", "block-2"])
        self.assertLess(elapsed, 0.27)

    def test_sync_agent3_tools_use_complete_sync(self):
        with patch("tools.agent3_tools.llm_generate_tool.LLMClient.complete_sync", return_value="answer") as complete:
            result = _generate_answer({"question": "什么是 LoRA？"})
        self.assertEqual(result["content"], "answer")
        complete.assert_called_once()

        code_response = "```python\ndef train():\n    pass\n```"
        with patch("tools.agent3_tools.code_generation_tool.LLMClient.complete_sync", return_value=code_response) as complete:
            examples = _llm_generate_code("LoRA")
        self.assertEqual(examples[0]["language"], "python")
        complete.assert_called_once()


if __name__ == "__main__":
    unittest.main()
