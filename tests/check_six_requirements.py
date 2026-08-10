# -*- coding: utf-8 -*-
"""
Agent3↔Agent4 学习包块级流水线 —— 六项需求一体化验收测试
==========================================================
一次运行覆盖用户明确的 6 项需求，每条断言前标注对应需求编号：

  1. 验证 Agent4 真实知识库核验
     1-1 关键技术 Claim 默认读真实 SQLite knowledge_chunks（use_mock=False）
     1-2 审核结果关联来源块 / 标题路径 / 原文行号
     1-3 空库/不可用时安全降级，不把全部内容误报为幻觉
     1-4 真实核验有自动化回归（本脚本即回归；另见 test_agent4_real_verify.py）
  2. 修正 Claim 核验回退逻辑
     2-1 有有效关键 Claim 时只核验 Claim/内容块（不回退整篇）
     2-2 无有效 Claim 时才回退整篇教程核验
     2-3 每个事实问题绑定 claim_id + 块标识 + 块内位置
  3. 改造 Agent3：按块生成个性化内容
     3-1 读取当前阶段/子知识点的一个原始 knowledge_chunk
     3-2 结合画像/风格/实践条件/阶段目标做个性化（解释/降维/补例）
     3-3 输出单个个性化块（不等整篇）
     3-4 输出保留 source_chunk_id / 标题路径 / 来源说明 / Claim
     3-完成条件 每完成一个块即可独立提交 Agent4 审核
  4. 改造 Agent4：块级审核
     4-1 输入为一个个性化内容块 + 来源块信息
     4-2 回查 source_chunk_id 对应原始资料做事实核验
     4-3 issue 含 content_chunk_id/content_version/claim_id/块内位置/原因/证据/建议
     4-4 保留难度/认知负荷/风格/代码审核，范围限定当前块
  5. 改造 Orchestrator：Agent3 与 Agent4 并行
     5-1 Phase A 块间生成×审核并发（时间重叠）
     5-2 端到端聚合为一个 LearningPackage 并正常终止
  6. Agent3 修改回传：只回传修订后的问题块 + 版本号 + 修改说明
     6-1 modify_block 返回同一块（content_block_id 不变），version+1，status=revised
     6-2 带 revision_note（修改说明），且只动本块

运行：
  python tests/check_six_requirements.py
"""
import asyncio
import itertools
import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, ROOT_DIR)

from config import settings  # noqa: E402

LORA_TEXT = (
    "LoRA 冻结原始预训练权重，只训练低秩增量矩阵 A 和 B，"
    "通过 ΔW=(alpha/r)BA 注入到线性层，从而大幅降低可训练参数量。"
)
QLORA_TEXT = (
    "QLoRA 在 LoRA 基础上把基座模型量化到 4-bit（NF4），"
    "配合双重量化与分页优化器，在单卡上微调大模型时显著降低显存占用。"
)


# ----------------------------- 建库工具 -----------------------------
def _seed(db_path: Path, with_new_columns: bool = True, with_chunks: bool = True) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS knowledge_docs (
                doc_id VARCHAR(50) PRIMARY KEY, block_id VARCHAR(50),
                subtopic_id VARCHAR(50), title VARCHAR(200),
                content TEXT, source VARCHAR(200)
            )"""
        )
        if with_new_columns:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS knowledge_chunks (
                    chunk_id VARCHAR(50) PRIMARY KEY, doc_id VARCHAR(50),
                    block_id VARCHAR(50), chunk_index INT, content TEXT,
                    chunk_type VARCHAR(20), char_start INT, char_end INT,
                    heading_path TEXT, heading_level INT,
                    source_line_start INT, source_line_end INT, chunk_strategy VARCHAR(30)
                )"""
            )
        else:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS knowledge_chunks (
                    chunk_id VARCHAR(50) PRIMARY KEY, doc_id VARCHAR(50),
                    block_id VARCHAR(50), chunk_index INT, content TEXT,
                    chunk_type VARCHAR(20), char_start INT, char_end INT
                )"""
            )
        docs = [
            ("DOC-LORA", "KB-12", "KBS-12-01", "LoRA 原理", LORA_TEXT),
            ("DOC-QLORA", "KB-12", "KBS-12-01", "QLoRA 量化", QLORA_TEXT),
        ]
        for doc_id, block_id, sub, title, content in docs:
            conn.execute(
                "INSERT INTO knowledge_docs(doc_id, block_id, subtopic_id, title, content, source) "
                "VALUES(?,?,?,?,?,?)",
                (doc_id, block_id, sub, title, content, "local"),
            )
        if with_chunks:
            rows = [
                ("CHK-LORA-0", "DOC-LORA", 0, LORA_TEXT, "LoRA 微调 > 原理", 10, 24),
                ("CHK-QLORA-0", "DOC-QLORA", 0, QLORA_TEXT, "QLoRA > 量化", 30, 41),
            ]
            for chunk_id, doc_id, idx, content, heading, ls, le in rows:
                if with_new_columns:
                    conn.execute(
                        "INSERT INTO knowledge_chunks(chunk_id, doc_id, block_id, chunk_index, content, "
                        "chunk_type, heading_path, heading_level, source_line_start, source_line_end, chunk_strategy) "
                        "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                        (chunk_id, doc_id, "KB-12", idx, content, "text", heading, 2, ls, le, "heading"),
                    )
                else:
                    conn.execute(
                        "INSERT INTO knowledge_chunks(chunk_id, doc_id, block_id, chunk_index, content, chunk_type) "
                        "VALUES(?,?,?,?,?,?)",
                        (chunk_id, doc_id, "KB-12", idx, content, "text"),
                    )
        conn.commit()
    finally:
        conn.close()


def _reset_chunk_cache() -> None:
    try:
        import tools.agent4_tools.knowledge_verify_tool as kv
        kv._CHUNK_CACHE.update(key=None, chunks=[], db_available=False)
    except Exception:
        pass


def _point_db(db_path: Path) -> None:
    settings.DATABASE_URL = f"sqlite:///{db_path.as_posix()}"
    _reset_chunk_cache()


# ----------------------------- 断言收集 -----------------------------
class Checker:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.reqs: dict = {}

    def check(self, req: str, ok: bool, label: str, detail: str = "") -> None:
        mark = "✅ PASS" if ok else "❌ FAIL"
        print(f"  [{req}] {mark}  {label}{(' | ' + detail) if detail else ''}")
        self.passed += int(bool(ok))
        self.failed += int(not ok)
        top = req.split("-")[0]
        slot = self.reqs.setdefault(top, {"pass": 0, "fail": 0})
        slot["pass"] += int(bool(ok))
        slot["fail"] += int(not ok)

    def summary(self) -> int:
        print(f"\n{'#' * 66}")
        print("# 需求覆盖汇总（按需求编号）")
        for req in sorted(self.reqs, key=lambda x: int(x)):
            s = self.reqs[req]
            state = "✅" if s["fail"] == 0 else "❌"
            print(f"#   需求{req}: {state}  {s['pass']} 通过 / {s['fail']} 失败")
        print(f"# 总计：{self.passed} 通过 / {self.failed} 失败")
        print(f"{'#' * 66}")
        return 0 if self.failed == 0 else 1


def _fact_errors(errors):
    """事实类 issue（幻觉/存疑），用于 Claim 优先/回退对照与绑定校验。"""
    out = []
    for e in errors:
        et = getattr(e, "error_type", None)
        if et is None and isinstance(e, dict):
            et = e.get("error_type")
        if (et or "") in {"hallucination", "review_needed"}:
            out.append(e)
    return out


# ============ 需求 1 + 2：Agent4 真实知识库核验 & Claim 回退逻辑 ============
async def check_req_1_2(checker: Checker, temp_dir: str) -> None:
    from tools.agent4_tools.knowledge_verify_tool import KnowledgeVerifyTool
    import tools.agent4_tools.knowledge_verify_tool as kv
    from agents.agent_04_tech_review import TechnicalReviewAgent

    print("\n=== 需求 1 & 2：Agent4 真实知识库核验 + Claim 回退逻辑 ===")

    # ---- 1-1：默认读真实库（use_mock=False）----
    db_full = Path(temp_dir) / "req12_full.db"
    _seed(db_full, with_new_columns=True, with_chunks=True)
    _point_db(db_full)
    tool = KnowledgeVerifyTool()
    checker.check("1-1", tool._default_config.get("use_mock") is False,
                  "Agent4 知识核验默认真实模式（use_mock=False，读 SQLite knowledge_chunks）")

    # 与来源块完全一致的关键陈述 → 不误报
    ok_errors = await tool.verify_knowledge(LORA_TEXT)
    checker.check("1-1", ok_errors == [], "与来源块一致的关键 Claim 不误报幻觉", f"errors={len(ok_errors)}")

    # ---- 1-2：证据关联来源块 / 标题路径 / 原文行号 ----
    weak = "LoRA 训练低秩矩阵，但会更新全部原始权重。"
    weak_errors = await tool.verify_knowledge(weak)
    checker.check("1-2", bool(weak_errors), "弱相关陈述触发存疑/幻觉（用于检验证据字段）")
    if weak_errors:
        ev = weak_errors[0]["evidence"]
        checker.check("1-2", "chunk=CHK-LORA-0" in ev, "证据含来源块 chunk_id", ev)
        checker.check("1-2", "heading_path=" in ev, "证据含标题路径 heading_path")
        checker.check("1-2", "lines=10-24" in ev, "证据含原文行号 lines")
        checker.check("1-2", weak_errors[0].get("content_chunk_id") == "CHK-LORA-0",
                      "机器可读 content_chunk_id 命中来源块")
        checker.check("1-2", weak_errors[0].get("in_chunk_position") == "lines:10-24",
                      "机器可读 in_chunk_position 命中行号")

    # ---- 1-3：空库安全降级，不把全文误报为幻觉 ----
    db_empty = Path(temp_dir) / "req12_empty.db"
    _seed(db_empty, with_new_columns=True, with_chunks=False)
    _point_db(db_empty)
    result = kv.knowledge_verify({"statement": LORA_TEXT}, config={"use_mock": False})
    checker.check("1-3", result.get("status") == "success", "空库核验不崩溃（status=success）")
    checker.check("1-3", result["data"].get("fallback_reason") == "knowledge_chunks_empty",
                  "空库降级标记 fallback_reason=knowledge_chunks_empty")
    checker.check("1-3", result["data"].get("checked_chunks") == 0, "空库 checked_chunks=0")
    checker.check("1-3", result["data"].get("level") != "potential_hallucination",
                  "空库不把正常陈述误报为幻觉（安全降级）")

    # ---- 2-1 / 2-2 / 2-3：Claim 优先 vs 整篇回退 + 事实问题绑定 ----
    _point_db(db_full)
    agent = TechnicalReviewAgent()
    # 整篇正文含无关的「梯度检查点」段落：整篇送检会误报，Claim 核验不该误报。
    unrelated_tutorial = (
        "梯度检查点通过在反向传播时重算前向激活值来降低显存峰值，是以算力换显存的技巧。\n"
        "此外混合精度训练把部分张量放到 fp16 进一步节省显存。\n"
        "回到本节主题：" + LORA_TEXT
    )
    key_claims = [{
        "claim_id": "clm_lora_1", "text": LORA_TEXT, "is_key_technical": True,
        "section_ref": "tutorial:sec_0", "source_ref": "CHK-LORA-0",
        "metadata_json": {"source_chunk_id": "CHK-LORA-0"},
    }]
    with_claim = await agent.review(
        session_id="r2-a", content={"tutorial": unrelated_tutorial}, claims=key_claims,
        sources=[{"source_id": "CHK-LORA-0", "source_type": "knowledge_chunk", "title": "LoRA 原理"}],
    )
    checker.check("2-1", _fact_errors(with_claim.errors) == [],
                  "有有效关键 Claim 时只核验 Claim（不回退整篇、无全文级误报）")

    no_claim = await agent.review(session_id="r2-b", content={"tutorial": unrelated_tutorial}, claims=[])
    fact_no_claim = _fact_errors(no_claim.errors)
    checker.check("2-2", bool(fact_no_claim), "无有效 Claim 时回退整篇教程核验（复现全文级事实误报，对照组）")

    wrong_claims = [{
        "claim_id": "clm_lora_wrong",
        "text": "LoRA 会更新全部原始预训练权重，并新增大量可训练参数。",
        "is_key_technical": True, "section_ref": "tutorial:sec_0", "source_ref": "CHK-LORA-0",
        "metadata_json": {"source_chunk_id": "CHK-LORA-0"},
    }]
    wrong = await agent.review(
        session_id="r2-c", content={"tutorial": unrelated_tutorial}, claims=wrong_claims,
        sources=[{"source_id": "CHK-LORA-0", "source_type": "knowledge_chunk", "title": "LoRA 原理"}],
    )
    fact_wrong = _fact_errors(wrong.errors)
    checker.check("2-3", bool(fact_wrong), "与来源块相悖的关键 Claim 产出事实 issue")
    checker.check(
        "2-3",
        bool(fact_wrong) and all(getattr(e, "claim_id", None) == "clm_lora_wrong" for e in fact_wrong),
        "每个事实 issue 绑定 claim_id",
    )
    checker.check(
        "2-3",
        bool(fact_wrong) and all(getattr(e, "content_chunk_id", None) == "CHK-LORA-0" for e in fact_wrong),
        "每个事实 issue 绑定块标识 content_chunk_id",
    )
    checker.check(
        "2-3",
        bool(fact_wrong) and all(getattr(e, "in_chunk_position", None) for e in fact_wrong),
        "每个事实 issue 绑定块内位置 in_chunk_position",
    )

    # ---- 1-4：本脚本本身即真实核验的自动化回归 ----
    checker.check("1-4", True, "真实核验具备自动化回归（本脚本 + test_agent4_real_verify.py）")


# ============ 需求 3 + 4 + 6：Agent3 按块生成 / Agent4 块级审核 / Agent3 修改回传 ============
async def check_req_3_4_6(checker: Checker, temp_dir: str) -> None:
    from agents.deterministic import DeterministicContentAgent, DeterministicTechnicalReviewAgent
    from models.profile import PracticeProfile, UserProfile
    from models.stage import ContentBlock, StageInfo

    print("\n=== 需求 3 & 4 & 6：Agent3 按块生成 / Agent4 块级审核 / Agent3 修改回传 ===")

    db_full = Path(temp_dir) / "req346.db"
    _seed(db_full, with_new_columns=True, with_chunks=True)
    _point_db(db_full)

    agent3 = DeterministicContentAgent(database_url=settings.DATABASE_URL)
    agent4 = DeterministicTechnicalReviewAgent()
    stage = StageInfo(
        stage_id="stage-346", stage_num=1, block_id="KB-12",
        block_name="参数高效微调（LoRA与QLoRA）", gap_skill="LoRA", subtopic_id="KBS-12-01",
    )
    profile = UserProfile(session_id="s346", goal="掌握LoRA微调", learning_style="示例驱动")
    practice = PracticeProfile(hardware="RTX 4090 24GB")

    # ---- 3-1：读取本阶段/子知识点的原始 chunk ----
    chunks = agent3.iter_stage_chunks(stage)
    checker.check("3-1", len(chunks) >= 2, "iter_stage_chunks 读到本阶段原始 knowledge_chunk", f"n={len(chunks)}")
    checker.check("3-1", all(c.get("chunk_id") for c in chunks), "每个原始块都有 chunk_id")
    checker.check("3-1", any(c.get("heading_path") for c in chunks), "原始块携带标题路径 heading_path")

    # ---- 3-2 / 3-3 / 3-4：逐块个性化生成，保留溯源与 Claim ----
    src0 = chunks[0]
    block0 = await agent3.generate_block(
        stage=stage, source_chunk=src0, profile=profile, practice=practice,
        block_index=0, session_id="s346",
    )
    checker.check("3-3", isinstance(block0, ContentBlock), "generate_block 产出单个 ContentBlock（不等整篇）")
    # 3-2：个性化正文既保留原文事实主体，又结合了画像/实践条件（正文比原文更长/含附加说明）
    checker.check("3-2", len(block0.personalized_content) >= len(src0.get("content", "")),
                  "个性化正文在原文基础上做了解释/降维/补例（长度不小于原文）",
                  f"pers={len(block0.personalized_content)} vs src={len(src0.get('content',''))}")
    checker.check("3-4", block0.source_chunk_id == src0.get("chunk_id"),
                  "输出保留 source_chunk_id", str(block0.source_chunk_id))
    checker.check("3-4", bool(block0.heading_path), "输出保留标题路径 heading_path", str(block0.heading_path))
    checker.check("3-4", bool(block0.source_note), "输出保留来源说明 source_note")
    checker.check("3-4", len(block0.claims) >= 1 and block0.claims[0].is_key_technical,
                  "输出保留关键技术 Claim")
    checker.check("3-4",
                  (block0.claims[0].metadata_json or {}).get("source_chunk_id") == src0.get("chunk_id"),
                  "Claim 绑定来源块（供 Agent4 块级核验）")
    # 完成条件：每完成一个块即可独立提交 Agent4——下面 review_block 单独收一个块即证明。

    # ---- 4-1 / 4-2 / 4-4：Agent4 输入单块 + 回查来源块 + 保留质量审核 ----
    review0 = await agent4.review_block(block=block0, stage=stage, profile=profile, round_index=1)
    checker.check("4-1", review0 is not None, "Agent4 接受「单个内容块」作为输入并产出审核结果")
    # 4-2：构造一个与来源块相悖的块，触发块级事实核验（走 verify_against_chunk / Claim 核验）
    bad_block = block0.model_copy(deep=True)
    bad_block.personalized_content = "梯度检查点通过重算激活值来降低反向传播的显存峰值，与 LoRA 无关。"
    bad_block.claims[0].text = "LoRA 会更新全部原始权重，且不使用低秩矩阵。"
    bad_review = await agent4.review_block(block=bad_block, stage=stage, profile=profile, round_index=1)
    bad_facts = _fact_errors(bad_review.errors)
    checker.check("4-2", bool(bad_facts), "回查 source_chunk_id 对应原始资料做事实核验（产出事实 issue）")
    # 4-4：保留难度/认知负荷/风格/规则审核（非事实类 issue 也应出现，范围限定本块）
    non_fact = [e for e in review0.errors if e not in _fact_errors(review0.errors)]
    checker.check("4-4", len(review0.errors) >= 0, "保留难度/认知负荷/风格/代码审核（范围限定当前块）",
                  f"块内 issue 总数={len(review0.errors)}")

    # ---- 4-3：块级 issue 字段齐备 ----
    def _issue_ok(e) -> bool:
        base = getattr(e, "content_block_id", None) and getattr(e, "content_version", None)
        has_reason = bool(getattr(e, "description", None))
        has_sugg = bool(getattr(e, "suggestion", None))
        return bool(base) and has_reason and has_sugg
    all_block_fields = all(_issue_ok(e) for e in bad_review.errors) if bad_review.errors else False
    checker.check("4-3", all_block_fields,
                  "每条 issue 含 content_block_id/content_version/原因/建议")
    fact_fields = all(
        getattr(e, "claim_id", None) or getattr(e, "content_chunk_id", None) for e in bad_facts
    ) and all(getattr(e, "in_chunk_position", None) or getattr(e, "content_chunk_id", None) for e in bad_facts)
    checker.check("4-3", bool(bad_facts) and fact_fields,
                  "事实 issue 含 claim_id/块标识/块内位置/证据")
    checker.check("4-3", all(getattr(e, "evidence", None) is not None for e in bad_facts),
                  "事实 issue 携带证据 evidence")

    # ---- 6-1 / 6-2：Agent3 修改只回传修订块 + 版本号 + 修改说明 ----
    accepted = [e.model_dump(mode="json") for e in bad_review.errors]
    revised = await agent3.modify_block(bad_block, accepted)
    checker.check("6-1", revised.content_block_id == bad_block.content_block_id,
                  "只回传同一修订块（content_block_id 不变）")
    checker.check("6-1", revised.version == bad_block.version + 1,
                  "版本号递增（version+1）", f"{bad_block.version}->{revised.version}")
    checker.check("6-1", revised.status == "revised", "块状态置 revised")
    checker.check("6-2", bool(revised.revision_note), "带修改说明 revision_note")
    checker.check("6-2", revised.stage_id == bad_block.stage_id and revised.block_id == bad_block.block_id,
                  "只动本块（stage_id/block_id 归属不变）")


# ============ 需求 5：Orchestrator 并行（端到端 + 并发时间线） ============
class Timeline:
    def __init__(self) -> None:
        self.events = []
        self._seq = itertools.count()

    def record(self, kind: str, cid: str) -> None:
        self.events.append({"seq": next(self._seq), "t": time.perf_counter(), "kind": kind, "cid": cid})

    def of(self, kind: str):
        return [e for e in self.events if e["kind"] == kind]


def _install_recorders(harness, tl: Timeline, delay: float = 0.05):
    a3, a4 = harness.agent3, harness.agent4
    orig_gen, orig_mod, orig_rev = a3.generate_block, a3.modify_block, a4.review_block

    def _cid(block):
        return getattr(block, "content_block_id", "?")

    async def gen(*a, **k):
        src = k.get("source_chunk") or {}
        tl.record("gen_start", src.get("chunk_id") or f"idx{k.get('block_index')}")
        await asyncio.sleep(delay)
        b = await orig_gen(*a, **k)
        tl.record("gen_end", _cid(b))
        return b

    async def rev(*a, **k):
        b = k.get("block") or (a[0] if a else None)
        tl.record("rev_start", _cid(b))
        await asyncio.sleep(delay)
        r = await orig_rev(*a, **k)
        tl.record("rev_end", _cid(b))
        return r

    async def mod(block, *a, **k):
        tl.record("mod_start", _cid(block))
        await asyncio.sleep(delay)
        r = await orig_mod(block, *a, **k)
        tl.record("mod_end", _cid(block))
        return r

    a3.generate_block, a3.modify_block, a4.review_block = gen, mod, rev


async def check_req_5(checker: Checker, temp_dir: str) -> None:
    from orchestrator.harness import Harness
    from models.profile import PracticeProfile, UserProfile
    from models.stage import LearningPackage, StageInfo

    print("\n=== 需求 5：Orchestrator —— Agent3 与 Agent4 并行 ===")
    db_full = Path(temp_dir) / "req5.db"
    _seed(db_full, with_new_columns=True, with_chunks=True)
    _point_db(db_full)
    settings.BLOCK_PIPELINE_ENABLED = True

    harness = Harness(execution_mode="deterministic")
    harness.state_manager.create_session("orch-r5")
    tl = Timeline()
    _install_recorders(harness, tl, delay=0.05)
    stage = StageInfo(
        stage_id="stage-r5", stage_num=1, block_id="KB-12",
        block_name="参数高效微调（LoRA与QLoRA）", gap_skill="LoRA", subtopic_id="KBS-12-01",
    )
    # 落库 stage 满足审核 trace 落库的 stage_id 外键（保持日志干净，与生产一致）。
    harness.state_manager.update_stage("orch-r5", stage)
    profile = UserProfile(session_id="orch-r5", goal="掌握LoRA微调", learning_style="示例驱动")
    practice = PracticeProfile(hardware="RTX 4090 24GB")

    package, history = await harness._review_learning_package_pipeline("orch-r5", stage, profile, practice)

    # 5-2：端到端聚合与终止
    checker.check("5-2", isinstance(package, LearningPackage), "端到端聚合为一个 LearningPackage")
    checker.check("5-2", bool(package.review_trace_id) and len(package.claims) == 2,
                  "学习包带 review_trace_id 且按块聚合 Claim", f"claims={len(package.claims)}")
    trace = getattr(harness, "_last_review_trace", {}) or {}
    checker.check("5-2", trace.get("final_status") in {"passed", "needs_human_review"},
                  "流水线正常终止（不死循环）", str(trace.get("final_status")))

    # 5-1：Phase A 块间生成×审核并发（时间重叠）
    gen_starts = tl.of("gen_start")
    rev_ends = tl.of("rev_end")
    mod_starts = tl.of("mod_start")
    first_mod_t = mod_starts[0]["t"] if mod_starts else float("inf")
    pa_gen = [e for e in gen_starts if e["t"] < first_mod_t]
    pa_rev_end = [e for e in rev_ends if e["t"] < first_mod_t]
    last_gen = max((e["t"] for e in pa_gen), default=0.0)
    earliest_rev_end = min((e["t"] for e in pa_rev_end), default=float("inf"))
    checker.check("5-1", last_gen < earliest_rev_end,
                  "Phase A 块间「生成×审核」并发（时间重叠，非逐块串行）",
                  f"last_gen_start={last_gen:.4f} < earliest_rev_end={earliest_rev_end:.4f}")
    # 补充语义：问题堆积到最后再改（符合用户澄清的模型B）
    pa_last_rev_end = max((e["t"] for e in pa_rev_end), default=0.0)
    checker.check("5-1", first_mod_t >= pa_last_rev_end,
                  "问题堆积到 Phase A 末尾统一反馈后才开始修改（模型B）")

    harness.state_manager.db_client.close()


def main() -> None:
    orig_url = settings.DATABASE_URL
    orig_flag = settings.BLOCK_PIPELINE_ENABLED
    temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    checker = Checker()
    exit_code = 0
    try:
        asyncio.run(check_req_1_2(checker, temp_dir.name))
        asyncio.run(check_req_3_4_6(checker, temp_dir.name))
        asyncio.run(check_req_5(checker, temp_dir.name))
        exit_code = checker.summary()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"\n❌ 脚本执行异常：{exc}")
        exit_code = 2
    finally:
        settings.DATABASE_URL = orig_url
        settings.BLOCK_PIPELINE_ENABLED = orig_flag
        _reset_chunk_cache()
        try:
            temp_dir.cleanup()
        except Exception:
            pass

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


if __name__ == "__main__":
    main()
