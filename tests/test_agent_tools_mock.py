"""Agent1-3 Tool Mock 驱动示例测试。

演示三种 mock 驱动范式（用薄子类把空壳 tool 方法接到 loader / DB，模拟未来实现）：
  A. 纯 loader 路径      —— Agent1GraphQueryMockTest
  B. seed 进 sqlite 后查 DB —— Agent2KnowledgeBlockMockTest
  C. 多表 seed 后查询      —— Agent3QuestionExtractMockTest
其余空壳（VectorSearchTool / CodeGenerationTool / TemplateRenderTool /
FormatConverterTool）范式同构，按 A 或 B 接到对应 mock 数据即可，此处不逐个展开。
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(TESTS_DIR))

from db.sqlite_client import SQLiteClient  # noqa: E402
from mock_loader import (  # noqa: E402
    get_data_source,
    load_mock,
    load_mock_dict,
    load_mock_named,
    seed_mock_to_sqlite,
)
from tools.agent1_tools.graph_query_tool import GraphQueryTool  # noqa: E402
from tools.agent2_tools.knowledge_block_query_tool import KnowledgeBlockQueryTool  # noqa: E402
from tools.agent3_tools.question_extract_tool import QuestionExtractTool  # noqa: E402


class MockDataShapeTest(unittest.TestCase):
    """数据形状自检：保证 mock 文件存在且字段对齐 schema 的 CHECK 约束。"""

    _DIFFICULTY = {"初级", "中级", "高级"}
    _DOC_TYPE = {"tutorial", "code", "spec", "best_practice", "pitfall"}
    _CHUNK_TYPE = {"text", "code", "table"}
    _QUESTION_TYPE = {"选择", "填空", "判断", "简答"}

    def test_knowledge_blocks_conform(self):
        rows = load_mock("knowledge_blocks")
        self.assertGreater(len(rows), 0)
        for r in rows:
            self.assertIn(r["difficulty"], self._DIFFICULTY)
            self.assertIsInstance(r["prerequisites"], list)
            self.assertIsInstance(r["corresponding_gaps"], list)
            self.assertTrue(r["block_id"] and r["block_name"])

    def test_knowledge_docs_conform(self):
        rows = load_mock("knowledge_docs")
        self.assertGreater(len(rows), 0)
        for r in rows:
            self.assertIn(r["doc_type"], self._DOC_TYPE)
            self.assertIn(r["difficulty"], self._DIFFICULTY)
            self.assertTrue(r["content"])

    def test_knowledge_chunks_conform(self):
        rows = load_mock("knowledge_chunks")
        self.assertGreater(len(rows), 0)
        for r in rows:
            self.assertIn(r["chunk_type"], self._CHUNK_TYPE)
            self.assertTrue(r["content"])

    def test_test_questions_conform(self):
        rows = load_mock("test_questions")
        self.assertGreater(len(rows), 0)
        for r in rows:
            self.assertIn(r["question_type"], self._QUESTION_TYPE)
            self.assertIn(r["difficulty"], self._DIFFICULTY)
            self.assertTrue(r["correct_answer"])
            # options 是 list 或 None
            self.assertTrue(r["options"] is None or isinstance(r["options"], list))

    def test_skill_prerequisites_has_lora_chain(self):
        m = load_mock_named("skill_prerequisites")
        self.assertIn("LoRA", m)
        self.assertIn("Transformer", m["LoRA"]["prerequisites"])
        self.assertIn("QLoRA", m["LoRA"]["dependencies"])


class Agent1GraphQueryMockTest(unittest.IsolatedAsyncioTestCase):
    """模式 A：纯 mock loader 路径。

    演示 agent1 空壳 tool 的测试范式——用一个薄子类把方法接到 loader，
    在真实 GraphClient 实现落地前即可驱动测试。
    """

    class _Mocked(GraphQueryTool):
        async def query_prerequisites(self, skill_name: str) -> list[str]:
            entry = load_mock_named("skill_prerequisites").get(skill_name, {})
            return list(entry.get("prerequisites", []))

        async def query_dependencies(self, skill_name: str) -> list[str]:
            entry = load_mock_named("skill_prerequisites").get(skill_name, {})
            return list(entry.get("dependencies", []))

    async def test_lora_prerequisites(self):
        prereqs = await self._Mocked().query_prerequisites("LoRA")
        self.assertIn("Transformer", prereqs)
        self.assertIn("注意力机制", prereqs)

    async def test_lora_dependencies(self):
        deps = await self._Mocked().query_dependencies("LoRA")
        self.assertIn("QLoRA", deps)

    async def test_unknown_skill_returns_empty(self):
        self.assertEqual(await self._Mocked().query_prerequisites("未知技能"), [])
        self.assertEqual(await self._Mocked().query_dependencies("未知技能"), [])


class _DbTestBase(unittest.IsolatedAsyncioTestCase):
    """模式 B/C 通用基类：起临时 sqlite 文件库。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        db_url = f"sqlite:///{(Path(self.tmp.name) / 'mock.db').as_posix()}"
        self.client = SQLiteClient(db_url)

    def tearDown(self):
        self.client.close()
        self.tmp.cleanup()


class Agent2KnowledgeBlockMockTest(_DbTestBase):
    """模式 B：seed 进临时 sqlite 后 tool 查 DB。"""

    class _DbBacked(KnowledgeBlockQueryTool):
        def __init__(self, client):
            super().__init__()
            self.client = client

        async def query_block(self, skill_name: str) -> dict:
            from sqlalchemy import text

            with self.client.engine.connect() as conn:
                row = conn.execute(text(
                    "SELECT block_id, block_name, description, estimated_hours, "
                    "difficulty, prerequisites, corresponding_gaps "
                    "FROM knowledge_blocks WHERE block_name = :n"
                ), {"n": skill_name}).mappings().first()
            if not row:
                return {}
            d = dict(row)
            for f in ("prerequisites", "corresponding_gaps"):
                if d.get(f):
                    d[f] = json.loads(d[f])
            return d

    async def test_seed_then_query_lora(self):
        seeded = seed_mock_to_sqlite(self.client, "knowledge_blocks")
        self.assertGreater(seeded, 0)

        block = await self._DbBacked(self.client).query_block("LoRA")
        self.assertEqual(block["block_name"], "LoRA")
        self.assertEqual(block["difficulty"], "高级")
        # JSON 字段往返后仍是 list，且内容正确
        self.assertIsInstance(block["prerequisites"], list)
        self.assertIn("Transformer", block["prerequisites"])
        self.assertIn("注意力机制", block["prerequisites"])

    async def test_unknown_returns_empty(self):
        seed_mock_to_sqlite(self.client, "knowledge_blocks")
        self.assertEqual(await self._DbBacked(self.client).query_block("不存在"), {})


class Agent3QuestionExtractMockTest(_DbTestBase):
    """模式 C：多表 seed（blocks + questions）后查询，演示 N 条题目提取。"""

    class _DbBacked(QuestionExtractTool):
        def __init__(self, client):
            super().__init__()
            self.client = client

        async def extract_questions(self, block_id: str, count: int = 5) -> list[dict]:
            from sqlalchemy import text

            with self.client.engine.connect() as conn:
                rows = conn.execute(text(
                    "SELECT question_id, block_id, difficulty, question_type, "
                    "question_text, options, correct_answer, hint "
                    "FROM test_questions WHERE block_id = :b LIMIT :n"
                ), {"b": block_id, "n": count}).mappings().all()
            out = []
            for r in rows:
                d = dict(r)
                if d.get("options"):
                    d["options"] = json.loads(d["options"])
                out.append(d)
            return out

    async def test_extract_questions_for_block(self):
        seed_mock_to_sqlite(self.client, "knowledge_blocks")
        seed_mock_to_sqlite(self.client, "test_questions")

        # 取一个保证有题目的 block_id
        block_id = load_mock("test_questions")[0]["block_id"]
        qs = await self._DbBacked(self.client).extract_questions(block_id, count=3)

        self.assertGreater(len(qs), 0)
        self.assertLessEqual(len(qs), 3)
        self.assertTrue(all(q["block_id"] == block_id for q in qs))
        # options 字段往返正确：选择题为 list，其他为 None
        for q in qs:
            if q["question_type"] == "选择":
                self.assertIsInstance(q["options"], list)
            else:
                self.assertIsNone(q["options"])


class DataSourceSwitchTest(unittest.TestCase):
    """演示 mock / 真实切换 env 入口（不依赖外部环境）。"""

    def test_default_is_mock(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AGENT_TOOL_DATA_SOURCE", None)
            self.assertEqual(get_data_source(), "mock")

    def test_db_source_when_env_set(self):
        with patch.dict(os.environ, {"AGENT_TOOL_DATA_SOURCE": "DB"}):
            self.assertEqual(get_data_source(), "db")

    def test_unknown_value_passes_through(self):
        with patch.dict(os.environ, {"AGENT_TOOL_DATA_SOURCE": "staging"}):
            self.assertEqual(get_data_source(), "staging")


if __name__ == "__main__":
    unittest.main()
