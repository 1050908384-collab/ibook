# -*- coding: utf-8 -*-
"""
专业知识与教学质量审核 Agent 真实知识库核验回归测试（任务 1 + 任务 2）
===============================================
覆盖点：
  ① 真实块核验默认走真实模式（use_mock=False，读 SQLite knowledge_chunks）；
  ② 命中证据含 chunk / heading_path / lines，并附机器可读 content_chunk_id / in_chunk_position；
  ③ 数据侧未入库新列时防御式降级（不崩溃、证据省略 heading/lines 但仍带 chunk_id）；
  ④ 空库安全降级（fallback_reason==knowledge_chunks_empty，不把全文误报为幻觉）；
  ⑤ verify_against_chunk 精确绑定来源块；未知块返回 []（不臆造幻觉）；
  ⑥ Claim 优先核验 vs 无 Claim 整篇回退（专业知识与教学质量审核 Agent.run，issue 是否带 claim_id）。

运行：
  pytest tests/test_agent4_real_verify.py -q
或直接：
  python tests/test_agent4_real_verify.py
"""
import asyncio
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path[:0] = [str(PROJECT_ROOT), str(BACKEND_DIR)]

from config import settings
import tools.agent4_tools.knowledge_verify_tool as kv
from tools.agent4_tools.knowledge_verify_tool import KnowledgeVerifyTool
from agents.agent_04_tech_review import TechnicalReviewAgent


LORA_TEXT = (
    "LoRA 冻结原始预训练权重，只训练低秩增量矩阵 A 和 B，"
    "通过 ΔW=(alpha/r)BA 注入到线性层，从而大幅降低可训练参数量。"
)

# 一段整篇送检必然稀释相似度、并触发全文级幻觉误报的教程正文：
# 前半段与来源块无关（梯度检查点），后半段才提到 LoRA。
UNRELATED_TUTORIAL = (
    "梯度检查点通过在反向传播时重算前向激活值来降低显存峰值，是一种以算力换显存的技巧。\n"
    "此外，混合精度训练把部分张量放到 fp16 以进一步节省显存。\n"
    "回到本节主题：" + LORA_TEXT
)


def _create_base_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE knowledge_docs (
            doc_id VARCHAR(50) PRIMARY KEY,
            block_id VARCHAR(50),
            subtopic_id VARCHAR(50),
            title VARCHAR(200),
            content TEXT,
            source VARCHAR(200)
        )"""
    )


def _seed_doc(conn: sqlite3.Connection) -> None:
    conn.execute(
        "INSERT INTO knowledge_docs(doc_id, block_id, subtopic_id, title, content, source) "
        "VALUES(?,?,?,?,?,?)",
        ("DOC-1", "KB-12", "KBS-12-01", "LoRA 原理", LORA_TEXT, "local"),
    )


class RealVerifyTest(unittest.TestCase):
    def setUp(self):
        self._orig_url = settings.DATABASE_URL
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = Path(self.temp_dir.name) / "kv.db"
        # 每例重置进程内缓存，避免跨库串味。
        kv._CHUNK_CACHE.update(key=None, chunks=[], db_available=False)
        settings.DATABASE_URL = f"sqlite:///{self.db_path.as_posix()}"

    def tearDown(self):
        settings.DATABASE_URL = self._orig_url
        kv._CHUNK_CACHE.update(key=None, chunks=[], db_available=False)
        self.temp_dir.cleanup()

    # --- 建库工具（可选择是否包含新列）---
    def _build_db(self, with_new_columns: bool, with_chunks: bool = True) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            _create_base_schema(conn)
            if with_new_columns:
                conn.execute(
                    """CREATE TABLE knowledge_chunks (
                        chunk_id VARCHAR(50) PRIMARY KEY,
                        doc_id VARCHAR(50),
                        block_id VARCHAR(50),
                        chunk_index INT,
                        content TEXT,
                        chunk_type VARCHAR(20),
                        char_start INT,
                        char_end INT,
                        heading_path TEXT,
                        heading_level INT,
                        source_line_start INT,
                        source_line_end INT,
                        chunk_strategy VARCHAR(30)
                    )"""
                )
            else:
                conn.execute(
                    """CREATE TABLE knowledge_chunks (
                        chunk_id VARCHAR(50) PRIMARY KEY,
                        doc_id VARCHAR(50),
                        block_id VARCHAR(50),
                        chunk_index INT,
                        content TEXT,
                        chunk_type VARCHAR(20),
                        char_start INT,
                        char_end INT
                    )"""
                )
            _seed_doc(conn)
            if with_chunks:
                if with_new_columns:
                    conn.execute(
                        "INSERT INTO knowledge_chunks(chunk_id, doc_id, block_id, chunk_index, content, "
                        "chunk_type, heading_path, heading_level, source_line_start, source_line_end, chunk_strategy) "
                        "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                        ("CHK-1", "DOC-1", "KB-12", 0, LORA_TEXT, "text",
                         "LoRA 微调 > 原理", 2, 10, 24, "heading"),
                    )
                else:
                    conn.execute(
                        "INSERT INTO knowledge_chunks(chunk_id, doc_id, block_id, chunk_index, content, chunk_type) "
                        "VALUES(?,?,?,?,?,?)",
                        ("CHK-1", "DOC-1", "KB-12", 0, LORA_TEXT, "text"),
                    )
            conn.commit()
        finally:
            conn.close()

    # ① + ② 真实核验默认真实模式 + 证据含 chunk/heading/lines + 机器可读字段
    def test_real_verify_with_new_columns(self):
        self._build_db(with_new_columns=True)
        tool = KnowledgeVerifyTool()
        self.assertFalse(tool._default_config["use_mock"])
        errors = asyncio.run(tool.verify_knowledge(LORA_TEXT))
        # 完全匹配的陈述应通过（相似度 >= 0.80），无 error。
        self.assertEqual(errors, [], "完全匹配来源块的陈述不应报幻觉")

        # 用一个部分相关的陈述触发存疑/幻觉，检验证据字段。
        weak = "LoRA 训练低秩矩阵，但会更新全部原始权重。"
        weak_errors = asyncio.run(tool.verify_knowledge(weak))
        self.assertTrue(weak_errors, "与来源块弱相关的陈述应产生 review_needed/hallucination")
        ev = weak_errors[0]["evidence"]
        self.assertIn("chunk=CHK-1", ev)
        self.assertIn("heading_path=", ev)
        self.assertIn("lines=10-24", ev)
        self.assertEqual(weak_errors[0]["content_chunk_id"], "CHK-1")
        self.assertEqual(weak_errors[0]["in_chunk_position"], "lines:10-24")

    # ③ 缺新列降级：不崩溃，证据省略 heading/lines 但仍带 chunk_id
    def test_missing_columns_degrades_gracefully(self):
        self._build_db(with_new_columns=False)
        tool = KnowledgeVerifyTool()
        weak = "LoRA 训练低秩矩阵，但会更新全部原始权重。"
        errors = asyncio.run(tool.verify_knowledge(weak))
        self.assertTrue(errors)
        ev = errors[0]["evidence"]
        self.assertIn("chunk=CHK-1", ev)
        self.assertNotIn("heading_path=", ev)
        self.assertNotIn("lines=", ev)
        # 缺列时无行号，in_chunk_position 退化为 chunk_index 或 None，但不得抛异常。
        self.assertEqual(errors[0]["content_chunk_id"], "CHK-1")

    # ④ 空库安全降级，不把全文误报为幻觉
    def test_empty_db_safe_fallback(self):
        self._build_db(with_new_columns=True, with_chunks=False)
        result = kv.knowledge_verify({"statement": LORA_TEXT}, config={"use_mock": False})
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["data"].get("fallback_reason"), "knowledge_chunks_empty")
        self.assertEqual(result["data"].get("checked_chunks"), 0)
        # 降级为 Mock 判定：对正常陈述不应判为 critical 幻觉。
        self.assertNotEqual(result["data"].get("level"), "potential_hallucination")

    # ⑤ verify_against_chunk 精确绑定 + 未知块返回 []
    def test_verify_against_chunk(self):
        self._build_db(with_new_columns=True)
        tool = KnowledgeVerifyTool()
        # 完全匹配来源块 → 无 error。
        ok = asyncio.run(tool.verify_against_chunk(LORA_TEXT, "CHK-1"))
        self.assertEqual(ok, [])
        # 与来源块无关的陈述 → 幻觉/存疑，且绑定到 CHK-1。
        bad = asyncio.run(tool.verify_against_chunk(
            "梯度检查点通过重算激活值来降低反向传播的显存峰值。", "CHK-1"))
        self.assertTrue(bad)
        self.assertEqual(bad[0]["content_chunk_id"], "CHK-1")
        self.assertIn("chunk=CHK-1", bad[0]["evidence"])
        # 未知块 → []（不臆造幻觉）。
        unknown = asyncio.run(tool.verify_against_chunk(LORA_TEXT, "CHK-UNKNOWN"))
        self.assertEqual(unknown, [])

    # ⑥ Claim 优先核验 vs 无 Claim 整篇回退（专业知识与教学质量审核 Agent.review 层）
    def test_claim_priority_vs_full_tutorial_fallback(self):
        """存在有效关键 Claim 时只核验 Claim（不回退整篇、事实 issue 带 claim_id）；
        无有效关键 Claim 时才回退整篇送检（复现全文级误报，用于对照）。"""
        self._build_db(with_new_columns=True)
        agent = TechnicalReviewAgent()
        # 只关心事实核验分支：难度/负荷/风格类不影响本例结论，按 error_type 过滤。
        def _fact_errors(errors):
            return [
                e for e in errors
                if (getattr(e, "error_type", "") or "") in {"hallucination", "review_needed"}
            ]

        # 关键 Claim 精确绑定来源块 CHK-1，且陈述与来源块完全一致 → 事实核验通过。
        key_claims = [{
            "claim_id": "clm_lora_1",
            "text": LORA_TEXT,
            "is_key_technical": True,
            "section_ref": "tutorial:sec_0",
            "source_ref": "CHK-1",
            "metadata_json": {"source_chunk_id": "CHK-1"},
        }]

        # (a) 有关键 Claim：整篇正文含无关的「梯度检查点」段落，但因走 Claim 核验，
        #     不应因整篇稀释而产生全文级幻觉误报。
        with_claim = asyncio.run(agent.review(
            session_id="t6-a",
            content={"tutorial": UNRELATED_TUTORIAL},
            claims=key_claims,
            sources=[{"source_id": "CHK-1", "source_type": "knowledge_chunk", "title": "LoRA 原理"}],
        ))
        fact_with_claim = _fact_errors(with_claim.errors)
        self.assertEqual(
            fact_with_claim, [],
            "有关键 Claim 且 Claim 与来源块一致时，不应回退整篇、也不应产生全文级幻觉误报",
        )

        # (b) 无关键 Claim：同一段落回退整篇送检 → 复现全文级 hallucination/review_needed 误报，
        #     且这些事实 issue 无法归属到任何 claim_id（claim_id 落空）。
        no_claim = asyncio.run(agent.review(
            session_id="t6-b",
            content={"tutorial": UNRELATED_TUTORIAL},
            claims=[],
        ))
        fact_no_claim = _fact_errors(no_claim.errors)
        self.assertTrue(
            fact_no_claim,
            "无关键 Claim 时应回退整篇送检并复现全文级事实误报（对照组）",
        )
        self.assertTrue(
            all(getattr(e, "claim_id", None) in (None, "") for e in fact_no_claim),
            "整篇回退产出的事实 issue 无法绑定 claim_id",
        )

        # (c) 弱相关关键 Claim：与来源块相悖的关键 Claim 应产出事实 issue，且带 claim_id。
        wrong_claims = [{
            "claim_id": "clm_lora_wrong",
            "text": "LoRA 会更新全部原始预训练权重，并新增大量可训练参数。",
            "is_key_technical": True,
            "section_ref": "tutorial:sec_0",
            "source_ref": "CHK-1",
            "metadata_json": {"source_chunk_id": "CHK-1"},
        }]
        wrong = asyncio.run(agent.review(
            session_id="t6-c",
            content={"tutorial": UNRELATED_TUTORIAL},
            claims=wrong_claims,
            sources=[{"source_id": "CHK-1", "source_type": "knowledge_chunk", "title": "LoRA 原理"}],
        ))
        fact_wrong = _fact_errors(wrong.errors)
        self.assertTrue(fact_wrong, "与来源块相悖的关键 Claim 应产出事实 issue")
        self.assertTrue(
            all(getattr(e, "claim_id", None) == "clm_lora_wrong" for e in fact_wrong),
            "Claim 核验产出的每个事实 issue 必须绑定其 claim_id",
        )


if __name__ == "__main__":
    unittest.main()
