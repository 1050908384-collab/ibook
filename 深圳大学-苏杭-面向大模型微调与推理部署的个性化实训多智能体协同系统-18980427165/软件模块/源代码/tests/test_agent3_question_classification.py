# -*- coding: utf-8 -*-
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path[:0] = [str(PROJECT_ROOT), str(BACKEND_DIR)]

from agents.deterministic import DeterministicContentAgent, HybridContentAgent


class Agent3QuestionClassificationTest(unittest.IsolatedAsyncioTestCase):
    def test_daily_greetings_chitchat_and_emotions(self):
        daily_questions = [
            "你好",
            "谢谢你",
            "今天心情怎么样？",
            "我最近压力很大，学不进去",
            "陪我聊聊天吧",
            "我没有动力了，鼓励我一下",
        ]
        for question in daily_questions:
            with self.subTest(question=question):
                self.assertEqual(
                    "daily",
                    DeterministicContentAgent._classify_question_by_keywords(question),
                )

    def test_technical_intent_wins_over_emotion_words(self):
        technical_questions = [
            "CUDA 报错让我很焦虑，应该怎么排查？",
            "这个 Python 代码看不懂，能解释一下吗？",
            "vLLM 的 PagedAttention 原理是什么？",
            "LoRA 模型微调为什么能减少显存？",
        ]
        for question in technical_questions:
            with self.subTest(question=question):
                self.assertEqual(
                    "technical",
                    DeterministicContentAgent._classify_question_by_keywords(question),
                )

    def test_stage_difficulty_query_is_technical(self):
        self.assertEqual(
            "technical",
            DeterministicContentAgent._classify_question_by_keywords(
                "我这一个阶段的学习难点是什么"
            ),
        )

    async def test_hybrid_daily_rule_skips_llm_classifier(self):
        agent = HybridContentAgent()
        agent.llm_client.complete = AsyncMock(return_value='{"type":"technical"}')

        result = await agent._classify_question_by_llm("你好，今天心情怎么样？")

        self.assertEqual("daily", result)
        agent.llm_client.complete.assert_not_awaited()

    async def test_hybrid_technical_still_uses_llm_classifier(self):
        agent = HybridContentAgent()
        agent.llm_client.complete = AsyncMock(return_value='{"type":"technical"}')

        result = await agent._classify_question_by_llm("vLLM 的 PagedAttention 原理是什么？")

        self.assertEqual("technical", result)
        agent.llm_client.complete.assert_awaited_once()

    async def test_hybrid_stage_difficulty_skips_llm_classifier(self):
        agent = HybridContentAgent()
        agent.llm_client.complete = AsyncMock(return_value='{"type":"daily"}')

        result = await agent._classify_question_by_llm("我这一个阶段的学习难点是什么")

        self.assertEqual("technical", result)
        agent.llm_client.complete.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
