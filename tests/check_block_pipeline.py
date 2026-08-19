# -*- coding: utf-8 -*-
"""
Agent3 ↔ Agent4 块级并行流水线 端到端独立验证脚本
=====================================================
镜像 tests/check_agent34_debate.py 的结构，但验证的是「块级」链路（任务 3~6）：

  iter_stage_chunks  -> Agent3 读取阶段内原始 knowledge_chunk 列表
  generate_block     -> Agent3 逐块生成个性化内容块（保留 source_chunk_id/标题路径/来源说明/Claim）
  review_block       -> Agent4 逐块审核（回查 source_chunk_id 做事实核验，issue 绑定块级字段）
  modify_block       -> Agent3 只回传修订后的问题块 + 版本号 + 修改说明（任务 6）
  review_block(again)-> Agent4 复审修订块

它不实例化 Harness/StateManager（避免全量 DB 依赖），而是把 knowledge_docs +
knowledge_chunks（含《知识库分块新增字段说明.md》的新列）种到一个临时 SQLite，
再把 settings.DATABASE_URL 指向它，让 Agent3/Agent4 的工具走真实核验。

覆盖的验收点（对应用户任务）：
  [任务3] Agent3 逐块产出：content_block_id / source_chunk_id / heading_path / source_note / Claim
  [任务4] Agent4 块级审核：每条 issue 携带 content_chunk_id / content_version / claim_id / in_chunk_position
  [任务2] Claim 优先核验：有关键 Claim 时只对 Claim/来源块核验，不整篇送检
  [任务1] 真实核验 + 安全降级：未知来源块 / 空库不把内容误报为幻觉
  [任务6] modify_block 只回传单块：version+1 / status=revised / 带 revision_note

运行：
  python tests/check_block_pipeline.py
  python tests/check_block_pipeline.py --mode deterministic   # 默认
  python tests/check_block_pipeline.py --mode hybrid          # 需本地 Ollama
"""
import argparse
import asyncio
import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, ROOT_DIR)

from config import settings  # noqa: E402


# ----------------------------------------------------------------------
# 种子数据：真实资料块（与 test_agent4_real_verify 同一 LoRA 语料保持一致）
# ----------------------------------------------------------------------
LORA_TEXT = (
    "LoRA 冻结原始预训练权重，只训练低秩增量矩阵 A 和 B，"
    "通过 ΔW=(alpha/r)BA 注入到线性层，从而大幅降低可训练参数量。"
)
QLORA_TEXT = (
    "QLoRA 在 LoRA 基础上把基座模型量化到 4-bit（NF4），"
    "配合双重量化与分页优化器，在单卡上微调大模型时显著降低显存占用。"
)


def _seed_temp_db(db_path: Path) -> None:
    """建库并种 knowledge_docs + knowledge_chunks（含新列）。

    - knowledge_docs 带 subtopic_id（get_stage_chunks 的 JOIN 依赖它）；
    - knowledge_chunks 带 heading_path/heading_level/source_line_*/chunk_strategy 新列，
      验证防御式列探测在「新列已入库」时能正确带出溯源。
    """
    conn = sqlite3.connect(db_path)
    try:
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
        conn.execute(
            "INSERT INTO knowledge_docs(doc_id, block_id, subtopic_id, title, content, source) "
            "VALUES(?,?,?,?,?,?)",
            ("DOC-LORA", "KB-12", "KBS-12-01", "LoRA 原理", LORA_TEXT, "local"),
        )
        conn.execute(
            "INSERT INTO knowledge_docs(doc_id, block_id, subtopic_id, title, content, source) "
            "VALUES(?,?,?,?,?,?)",
            ("DOC-QLORA", "KB-12", "KBS-12-01", "QLoRA 量化", QLORA_TEXT, "local"),
        )
        conn.execute(
            "INSERT INTO knowledge_chunks(chunk_id, doc_id, block_id, chunk_index, content, "
            "chunk_type, heading_path, heading_level, source_line_start, source_line_end, chunk_strategy) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            ("CHK-LORA-0", "DOC-LORA", "KB-12", 0, LORA_TEXT, "text",
             "LoRA 微调 > 原理", 2, 10, 24, "heading"),
        )
        conn.execute(
            "INSERT INTO knowledge_chunks(chunk_id, doc_id, block_id, chunk_index, content, "
            "chunk_type, heading_path, heading_level, source_line_start, source_line_end, chunk_strategy) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            ("CHK-QLORA-0", "DOC-QLORA", "KB-12", 0, QLORA_TEXT, "text",
             "QLoRA > 量化", 2, 30, 41, "heading"),
        )
        conn.commit()
    finally:
        conn.close()


def pretty(label: str, obj) -> None:
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump(mode="json")
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


class Checker:
    """收集断言结果，最后统一汇总（不用 pytest，方便直接运行）。"""

    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0

    def check(self, ok: bool, label: str, detail: str = "") -> None:
        mark = "✅ PASS" if ok else "❌ FAIL"
        suffix = f" | {detail}" if detail else ""
        print(f"  {mark}  {label}{suffix}")
        if ok:
            self.passed += 1
        else:
            self.failed += 1

    def summary(self) -> int:
        print(f"\n{'#' * 60}")
        print(f"# 断言汇总：{self.passed} 通过 / {self.failed} 失败")
        print(f"{'#' * 60}")
        return 0 if self.failed == 0 else 1


def _issue_has_block_fields(issue: dict) -> tuple[bool, str]:
    """校验单条 issue 是否携带块级溯源字段（任务 4 强约束）。"""
    missing = []
    if not issue.get("content_block_id"):
        missing.append("content_block_id")
    if issue.get("content_version") is None:
        missing.append("content_version")
    # 事实类 issue 必须绑定 claim_id 或 content_chunk_id（任务 2/4）；
    # 非事实类（难度/认知负荷/风格）至少要有 content_block_id + content_version。
    fact_types = {"hallucination", "review_needed"}
    if issue.get("error_type") in fact_types:
        if not issue.get("claim_id") and not issue.get("content_chunk_id"):
            missing.append("claim_id/content_chunk_id(事实类)")
    return (not missing), ("缺少: " + ", ".join(missing) if missing else "块级字段齐备")


async def run(args: argparse.Namespace, checker: Checker) -> None:
    from agents.deterministic import (
        DeterministicContentAgent,
        DeterministicTechnicalReviewAgent,
        HybridContentAgent,
        HybridTechnicalReviewAgent,
    )
    from models.profile import PracticeProfile, UserProfile
    from models.stage import StageInfo

    mode = args.mode
    print("=" * 60)
    print("Agent3 ↔ Agent4 块级并行流水线 端到端验证")
    print(f">>> 执行模式：{mode}")
    print(f">>> 临时知识库：{settings.DATABASE_URL}")
    print("=" * 60)

    if mode == "hybrid":
        agent3 = HybridContentAgent()
        agent4 = HybridTechnicalReviewAgent()
    else:
        agent3 = DeterministicContentAgent()
        agent4 = DeterministicTechnicalReviewAgent()

    session_id = "check-block-pipeline"
    stage = StageInfo(
        stage_id="stage-blk-01", stage_num=1, block_id="KB-12",
        block_name="参数高效微调（LoRA与QLoRA）", gap_skill="LoRA",
        subtopic_id="KBS-12-01",
    )
    profile = UserProfile(session_id=session_id, goal="掌握LoRA微调", learning_style="示例驱动")
    practice = PracticeProfile(hardware="RTX 4090 24GB")

    # ------------------------------------------------------------------
    # 步骤 0：iter_stage_chunks 读取阶段原始块
    # ------------------------------------------------------------------
    print("\n[Step 0] Agent3.iter_stage_chunks（读取阶段原始 knowledge_chunk）...")
    source_chunks = agent3.iter_stage_chunks(stage)
    pretty("原始块列表（iter_stage_chunks）", source_chunks)
    checker.check(len(source_chunks) == 2, "iter_stage_chunks 读到 2 个原始块",
                  f"实际 {len(source_chunks)} 个")
    if source_chunks:
        first = source_chunks[0]
        checker.check(bool(first.get("chunk_id")), "原始块带 chunk_id",
                      str(first.get("chunk_id")))
        checker.check(first.get("heading_path") is not None,
                      "原始块带 heading_path（新列已入库）",
                      str(first.get("heading_path")))
    if not source_chunks:
        checker.check(False, "无原始块，后续块级链路无法验证", "请检查 seed / JOIN")
        return

    # ------------------------------------------------------------------
    # 步骤 1：Agent3 逐块生成个性化内容块（任务 3）
    # ------------------------------------------------------------------
    print("\n[Step 1] Agent3.generate_block（逐块个性化，改一块即可送审）...")
    blocks = []
    for idx, chunk in enumerate(source_chunks):
        block = await agent3.generate_block(
            stage, chunk, profile=profile, practice=practice,
            block_index=idx, session_id=session_id,
        )
        blocks.append(block)
    for i, block in enumerate(blocks):
        pretty(f"个性化内容块[{i}]（generate_block 产出）", {
            "content_block_id": block.content_block_id,
            "block_index": block.block_index,
            "source_chunk_id": block.source_chunk_id,
            "heading_path": block.heading_path,
            "source_line_start": block.source_line_start,
            "source_line_end": block.source_line_end,
            "source_note": block.source_note,
            "version": block.version,
            "status": block.status,
            "personalized_content_preview": (block.personalized_content or "")[:120],
            "personalization_action": block.personalization_action,
            "personalization_reason": block.personalization_reason,
            "fact_source_chunk_ids": block.fact_source_chunk_ids,
            "teaching_tasks": block.teaching_tasks,
            "claims": [c.model_dump(mode="json") for c in block.claims],
        })
    b0 = blocks[0]
    checker.check(bool(b0.content_block_id), "块携带 content_block_id", b0.content_block_id)
    checker.check(b0.source_chunk_id == "CHK-LORA-0", "块保留 source_chunk_id 溯源",
                  str(b0.source_chunk_id))
    checker.check(bool(b0.heading_path), "块保留标题路径 heading_path", str(b0.heading_path))
    checker.check(bool(b0.source_note) and "CHK-LORA-0" in b0.source_note,
                  "块携带来源说明 source_note（含来源块）", b0.source_note)
    checker.check(len(b0.claims) >= 1 and b0.claims[0].is_key_technical,
                  "块携带关键技术 Claim", b0.claims[0].claim_id if b0.claims else "无")
    checker.check(
        b0.claims and (b0.claims[0].metadata_json or {}).get("source_chunk_id") == "CHK-LORA-0",
        "Claim 绑定 source_chunk_id（供 Agent4 块级核验）",
        str((b0.claims[0].metadata_json or {}) if b0.claims else {}),
    )
    checker.check(
        b0.personalization_action == "explain_and_practice"
        and b0.fact_source_chunk_ids == ["CHK-LORA-0"]
        and bool(b0.teaching_tasks),
        "块携带结构化教学层元数据（动作、事实来源、学习任务）",
        str({"action": b0.personalization_action, "tasks": len(b0.teaching_tasks)}),
    )

    # ------------------------------------------------------------------
    # 步骤 2：Agent4 逐块审核（任务 4 + 任务 2 Claim 优先核验）
    # ------------------------------------------------------------------
    print("\n[Step 2] Agent4.review_block（逐块审核，issue 绑定块级字段）...")
    reviews = []
    for block in blocks:
        result = await agent4.review_block(block, stage, profile=profile, round_index=1)
        reviews.append(result)
        issues = [e.model_dump(mode="json") for e in result.errors]
        pretty(f"块 {block.content_block_id} 审核结果", {
            "passed": result.passed,
            "issues": issues,
            "personalization_issues": [e.model_dump(mode="json") for e in result.personalization_errors],
        })
        for iss in issues:
            ok, detail = _issue_has_block_fields(iss)
            checker.check(
                ok,
                f"issue[{iss.get('error_type')}] 携带块级溯源字段",
                detail,
            )
    # 完全匹配来源块的正文（deterministic 保留原文事实主体）不应被误报为幻觉。
    fact_issues = [
        e for r in reviews for e in r.errors
        if e.error_type in {"hallucination"}
    ]
    checker.check(
        len(fact_issues) == 0,
        "真实核验：与来源块一致的正文未被误报为幻觉（任务1/2）",
        f"幻觉 issue 数={len(fact_issues)}",
    )
    checker.check(
        all(review.passed for review in reviews),
        "教学审核建议不阻断已溯源内容块发布",
        f"fact_errors={sum(len(review.errors) for review in reviews)}",
    )

    # 个性化工具只审核教学层元数据，不将其误判为技术事实问题。
    incomplete_block = b0.model_copy(deep=True)
    incomplete_block.personalization_action = ""
    incomplete_block.personalization_reason = ""
    incomplete_block.fact_source_chunk_ids = []
    incomplete_block.teaching_tasks = []
    incomplete_result = await agent4.review_block(
        incomplete_block, stage, profile=profile, round_index=1,
    )
    personalization_types = {
        issue.error_type for issue in incomplete_result.personalization_errors
    }
    checker.check(
        {"personalization_action_missing", "personalization_reason_missing", "teaching_source_missing", "practice_task_missing"}.issubset(personalization_types),
        "个性化检查工具识别缺失的教学层元数据",
        str(sorted(personalization_types)),
    )
    checker.check(
        incomplete_result.passed and not incomplete_result.errors,
        "个性化元数据问题不被误判为技术幻觉",
        f"fact_errors={len(incomplete_result.errors)}",
    )

    # ------------------------------------------------------------------
    # 步骤 3：构造一个「弱相关」块，验证块级事实核验能定位来源块 + claim_id
    # ------------------------------------------------------------------
    print("\n[Step 3] 弱相关块审核（触发块级事实核验，校验 claim_id / content_chunk_id）...")
    weak_block = b0.model_copy(deep=True)
    weak_block.personalized_content = (
        "梯度检查点通过重算激活值来降低反向传播的显存峰值，与 LoRA 的低秩分解无关。"
    )
    # Claim 文本也改成与来源块弱相关，让 verify_against_chunk 命中 review_needed/hallucination
    if weak_block.claims:
        weak_block.claims[0].text = "梯度检查点重算激活值以降低显存。"
    weak_result = await agent4.review_block(weak_block, stage, profile=profile, round_index=1)
    weak_issues = [e.model_dump(mode="json") for e in weak_result.errors]
    pretty("弱相关块审核结果", {"passed": weak_result.passed, "issues": weak_issues})
    fact_weak = [i for i in weak_issues if i.get("error_type") in {"hallucination", "review_needed"}]
    checker.check(len(fact_weak) >= 1, "弱相关块触发块级事实 issue",
                  f"事实 issue 数={len(fact_weak)}")
    if fact_weak:
        i0 = fact_weak[0]
        checker.check(i0.get("content_chunk_id") == "CHK-LORA-0",
                      "事实 issue 绑定正确来源块 content_chunk_id", str(i0.get("content_chunk_id")))
        checker.check(bool(i0.get("claim_id")), "事实 issue 绑定 claim_id", str(i0.get("claim_id")))
        checker.check(bool(i0.get("in_chunk_position")), "事实 issue 携带块内位置 in_chunk_position",
                      str(i0.get("in_chunk_position")))

    # ------------------------------------------------------------------
    # 步骤 4：modify_block 只回传修订块 + 版本号 + 修改说明（任务 6）
    # ------------------------------------------------------------------
    print("\n[Step 4] Agent3.modify_block（只回传修订块 + version + revision_note）...")
    revised = await agent3.modify_block(
        weak_block, weak_issues, revision_note="已按审核建议修正为来源块支持的表述。",
    )
    pretty("修订后的块（modify_block 产出）", {
        "content_block_id": revised.content_block_id,
        "version": revised.version,
        "status": revised.status,
        "revision_note": revised.revision_note,
        "personalized_content_preview": (revised.personalized_content or "")[-160:],
    })
    checker.check(revised.version == weak_block.version + 1, "修订块 version 递增",
                  f"{weak_block.version} -> {revised.version}")
    checker.check(revised.status == "revised", "修订块 status=revised", revised.status)
    checker.check(bool(revised.revision_note), "修订块携带 revision_note（修改说明）", revised.revision_note)
    checker.check(revised.content_block_id == weak_block.content_block_id,
                  "修订块 content_block_id 保持不变（只回传问题块）", revised.content_block_id)

    # ------------------------------------------------------------------
    # 步骤 5：Agent4 复审修订块（改一块即刻送审的闭环）
    # ------------------------------------------------------------------
    print("\n[Step 5] Agent4.review_block（复审修订块，round_index=2）...")
    recheck = await agent4.review_block(revised, stage, profile=profile, round_index=2)
    recheck_issues = [e.model_dump(mode="json") for e in recheck.errors]
    pretty("修订块复审结果", {"passed": recheck.passed, "issues": recheck_issues})
    for iss in recheck_issues:
        ok, detail = _issue_has_block_fields(iss)
        checker.check(ok, f"复审 issue[{iss.get('error_type')}] 携带块级字段", detail)
    checker.check(
        all(i.get("content_version") == revised.version for i in recheck_issues) or not recheck_issues,
        "复审 issue 的 content_version 对齐修订后版本",
        f"版本={revised.version}",
    )


async def run_degrade_check(checker: Checker) -> None:
    """安全降级：未知来源块的 Claim 不应臆造幻觉（任务 1（3））。"""
    from agents.deterministic import DeterministicTechnicalReviewAgent
    from models.profile import UserProfile
    from models.stage import Claim, Source, StageInfo
    from models.stage import ContentBlock

    print("\n[Step 6] 安全降级：未知来源块不臆造幻觉...")
    agent4 = DeterministicTechnicalReviewAgent()
    stage = StageInfo(
        stage_id="stage-blk-01", stage_num=1, block_id="KB-12",
        block_name="参数高效微调（LoRA与QLoRA）", gap_skill="LoRA", subtopic_id="KBS-12-01",
    )
    profile = UserProfile(session_id="check-block-pipeline", goal="掌握LoRA微调")
    unknown_block = ContentBlock(
        content_block_id="cb-unknown", session_id="check-block-pipeline",
        stage_id=stage.stage_id, block_id="KB-12", block_name=stage.block_name,
        block_index=0, source_chunk_id="CHK-DOES-NOT-EXIST",
        personalized_content="这是一段与任何来源块都无法比对的内容。",
        claims=[Claim(
            claim_id="clm-unknown", text="某个无法核验的技术声明。",
            source_ref="src-x", section_ref="block:cb-unknown", claim_type="fact",
            is_key_technical=True, confidence=0.5,
            metadata_json={"source_chunk_id": "CHK-DOES-NOT-EXIST"},
        )],
        sources=[Source(source_id="src-x", source_type="knowledge_doc",
                        title="unknown", locator="CHK-DOES-NOT-EXIST",
                        content_ref="CHK-DOES-NOT-EXIST")],
        version=1,
    )
    result = await agent4.review_block(unknown_block, stage, profile=profile, round_index=1)
    hall = [e for e in result.errors if e.error_type == "hallucination"]
    pretty("未知来源块审核结果", {
        "passed": result.passed,
        "issues": [e.model_dump(mode="json") for e in result.errors],
    })
    checker.check(len(hall) == 0,
                  "未知来源块不臆造幻觉（安全降级）", f"幻觉 issue 数={len(hall)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["deterministic", "hybrid"], default="deterministic")
    args = parser.parse_args()

    orig_url = settings.DATABASE_URL
    temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    db_path = Path(temp_dir.name) / "block_pipeline.db"
    settings.DATABASE_URL = f"sqlite:///{db_path.as_posix()}"
    _seed_temp_db(db_path)

    # 清进程内块缓存，避免跨库串味（knowledge_verify_tool 按 mtime 缓存，这里防御性重置）。
    try:
        import tools.agent4_tools.knowledge_verify_tool as kv
        kv._CHUNK_CACHE.update(key=None, chunks=[], db_available=False)
    except Exception:
        pass

    checker = Checker()
    exit_code = 0
    try:
        asyncio.run(run(args, checker))
        asyncio.run(run_degrade_check(checker))
        exit_code = checker.summary()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"\n❌ 脚本执行异常：{exc}")
        exit_code = 2
    finally:
        settings.DATABASE_URL = orig_url
        try:
            temp_dir.cleanup()
        except Exception:
            pass

    # os._exit 屏蔽事件循环/连接池析构时的噪声，保证退出码干净可判读。
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


if __name__ == "__main__":
    main()
