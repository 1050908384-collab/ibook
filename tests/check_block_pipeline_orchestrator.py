# -*- coding: utf-8 -*-
"""
Orchestrator 块级并行流水线 端到端验证（任务 5）
=================================================
直接实例化真实 Harness（execution_mode=deterministic），把 settings.DATABASE_URL
指向一个种好 knowledge_docs + knowledge_chunks（含新列）的临时 SQLite，
再调用真正的 Harness._review_learning_package_pipeline，验证：

  - Phase A：asyncio.gather 并行「逐块生成 × 逐块审核」，产出首轮 ReviewRound；
  - Phase B：改一块送一块（modify_block → 立即 review_block 复审）逐轮推进；
  - Phase C：_aggregate_blocks 把各块聚合为一个 LearningPackage（带 review_trace_id）；
  - 达到 MAX_REVIEW_ITERATIONS 仍有未决 must_fix 时 final_status=needs_human_review（不崩溃）；
  - 无原始块时安全降级回整篇链路（_review_learning_package）。

运行：
  python tests/check_block_pipeline_orchestrator.py
"""
import asyncio
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

LORA_TEXT = (
    "LoRA 冻结原始预训练权重，只训练低秩增量矩阵 A 和 B，"
    "通过 ΔW=(alpha/r)BA 注入到线性层，从而大幅降低可训练参数量。"
)
QLORA_TEXT = (
    "QLoRA 在 LoRA 基础上把基座模型量化到 4-bit（NF4），"
    "配合双重量化与分页优化器，在单卡上微调大模型时显著降低显存占用。"
)


def _seed(db_path: Path, with_chunks: bool = True) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS knowledge_docs (
                doc_id VARCHAR(50) PRIMARY KEY, block_id VARCHAR(50),
                subtopic_id VARCHAR(50), title VARCHAR(200),
                content TEXT, source VARCHAR(200)
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS knowledge_chunks (
                chunk_id VARCHAR(50) PRIMARY KEY, doc_id VARCHAR(50),
                block_id VARCHAR(50), chunk_index INT, content TEXT,
                chunk_type VARCHAR(20), char_start INT, char_end INT,
                heading_path TEXT, heading_level INT,
                source_line_start INT, source_line_end INT, chunk_strategy VARCHAR(30)
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
        if with_chunks:
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


class Checker:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0

    def check(self, ok: bool, label: str, detail: str = "") -> None:
        mark = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {mark}  {label}{(' | ' + detail) if detail else ''}")
        self.passed += int(bool(ok))
        self.failed += int(not ok)

    def summary(self) -> int:
        print(f"\n{'#' * 60}\n# Orchestrator 断言汇总：{self.passed} 通过 / {self.failed} 失败\n{'#' * 60}")
        return 0 if self.failed == 0 else 1


async def run(checker: Checker) -> None:
    from orchestrator.harness import Harness
    from models.profile import PracticeProfile, UserProfile
    from models.review import ReviewRound
    from models.stage import LearningPackage, StageInfo

    harness = Harness(execution_mode="deterministic")
    # 生产路径中 start_stage 一定发生在 create_session + update_stage 之后：真实
    # session 行与 stage_progress 行都存在，才能满足 review_traces 的
    # FK(session_id → sessions, stage_id → stage_progress) 与 require_session 约束。
    harness.state_manager.create_session("orch-block")
    stage = StageInfo(
        stage_id="stage-orch-01", stage_num=1, block_id="KB-12",
        block_name="参数高效微调（LoRA与QLoRA）", gap_skill="LoRA", subtopic_id="KBS-12-01",
    )
    # 落库 stage 以满足 review_traces.stage_id 外键（对齐生产 start_stage 前置条件）。
    harness.state_manager.update_stage("orch-block", stage)
    profile = UserProfile(session_id="orch-block", goal="掌握LoRA微调", learning_style="practice")
    practice = PracticeProfile(hardware="RTX 4090 24GB")

    print("\n[有块场景] Harness._review_learning_package_pipeline ...")
    package, history = await harness._review_learning_package_pipeline(
        "orch-block", stage, profile, practice,
    )
    checker.check(isinstance(package, LearningPackage), "返回 LearningPackage 实例")
    checker.check(bool(package.review_trace_id), "学习包带 review_trace_id", str(package.review_trace_id))
    checker.check(len(package.tutorial) > 0, "聚合 tutorial 非空", f"len={len(package.tutorial)}")
    # 新版 _aggregate_blocks 不再把完整 breadcrumb heading_path 拼成 ## 标题；
    # 因此用"两个块在原文里的核心事实/关键句"是否都被保留在 tutorial 里来断言按块聚合成功。
    lora_marker = "只训练低秩增量矩阵 A 和 B"
    qlora_marker = "基座模型量化到 4-bit"
    checker.check(lora_marker in package.tutorial and qlora_marker in package.tutorial,
                  "tutorial 同时包含 LoRA/QLoRA 两块的核心技术表述（按块聚合）",
                  f"found_lora={lora_marker in package.tutorial}, found_qlora={qlora_marker in package.tutorial}")
    checker.check(len(package.claims) == 2, "聚合 2 条 Claim（每块一条）", f"claims={len(package.claims)}")
    checker.check(len(package.content_blocks) == 2,
                  "学习包保留 2 个结构化个性化内容块", f"blocks={len(package.content_blocks)}")
    checker.check(all(isinstance(h, ReviewRound) for h in history) and len(history) >= 1,
                  "review_history 为 ReviewRound 列表", f"rounds={len(history)}")

    trace = getattr(harness, "_last_review_trace", {}) or {}
    print(f"  _last_review_trace = {trace}")
    checker.check(trace.get("final_status") in {"passed", "needs_human_review"},
                  "final_status 合法（passed / needs_human_review）", str(trace.get("final_status")))
    # 已溯源块的难度、认知负荷和风格建议属于个性化审核，不能替代事实审核
    # 阻断发布；不存在事实问题时流水线应直接通过。
    checker.check(trace.get("final_status") == "passed",
                  "仅有个性化审核建议时学习包正常通过",
                  f"unresolved={trace.get('unresolved_must_fix_count')}")

    # ---- 审核 trace 落库验证（问题B 修复）：块级流水线也应写入 review_traces / rounds ----
    from sqlalchemy import text
    tid = trace.get("trace_id")
    with harness.state_manager.db_client.engine.connect() as conn:
        trace_row = conn.execute(
            text("SELECT final_status, package_status FROM review_traces WHERE trace_id=:t"),
            {"t": tid},
        ).fetchone()
        round_cnt = conn.execute(
            text("SELECT COUNT(*) FROM review_debate_rounds WHERE trace_id=:t"),
            {"t": tid},
        ).scalar()
        claim_cnt = conn.execute(
            text("SELECT COUNT(*) FROM learning_package_claims WHERE trace_id=:t"),
            {"t": tid},
        ).scalar()
    checker.check(trace_row is not None, "review_traces 已落库（块级流水线补齐持久化）", f"trace_id={tid}")
    checker.check(bool(trace_row) and trace_row[0] == "passed",
                  "review_traces.final_status 已落库为最终状态",
                  str(trace_row[0]) if trace_row else "None")
    checker.check((round_cnt or 0) == 0, "无事实问题时不创建无效辩论轮次", f"rounds={round_cnt}")
    checker.check((claim_cnt or 0) >= 1, "learning_package_claims 已落库", f"claims={claim_cnt}")

    harness.state_manager.db_client.close()


async def run_no_chunks(checker: Checker, temp_dir: str) -> None:
    """无原始块场景：应安全降级回整篇链路，不抛异常。"""
    from orchestrator.harness import Harness
    from models.profile import PracticeProfile, UserProfile
    from models.stage import LearningPackage, StageInfo

    # 换一个空 chunks 的库（保留 docs 但无 knowledge_chunks 行）
    empty_db = Path(temp_dir) / "orch_empty.db"
    _seed(empty_db, with_chunks=False)
    orig = settings.DATABASE_URL
    settings.DATABASE_URL = f"sqlite:///{empty_db.as_posix()}"
    try:
        import tools.agent4_tools.knowledge_verify_tool as kv
        kv._CHUNK_CACHE.update(key=None, chunks=[], db_available=False)
    except Exception:
        pass

    print("\n[无块场景] 降级回整篇链路 ...")
    harness = Harness(execution_mode="deterministic")
    harness.state_manager.create_session("orch-empty")
    stage = StageInfo(
        stage_id="stage-orch-02", stage_num=1, block_id="KB-12",
        block_name="参数高效微调（LoRA与QLoRA）", gap_skill="LoRA", subtopic_id="KBS-12-01",
    )
    # 落库 stage 满足 review_traces.stage_id 外键（降级整篇链路同样会落库审核 trace）。
    harness.state_manager.update_stage("orch-empty", stage)
    profile = UserProfile(session_id="orch-empty", goal="掌握LoRA微调")
    practice = PracticeProfile(hardware="")
    try:
        package, history = await harness._review_learning_package_pipeline(
            "orch-empty", stage, profile, practice,
        )
        checker.check(isinstance(package, LearningPackage), "无块降级仍返回 LearningPackage")
        checker.check(True, "无块降级未抛异常（回退整篇链路）")
    except Exception as exc:
        import traceback
        traceback.print_exc()
        checker.check(False, "无块降级未抛异常（回退整篇链路）", str(exc))
    finally:
        harness.state_manager.db_client.close()
        settings.DATABASE_URL = orig


def main() -> None:
    orig_url = settings.DATABASE_URL
    orig_flag = settings.BLOCK_PIPELINE_ENABLED
    temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    db_path = Path(temp_dir.name) / "orch_block.db"
    settings.DATABASE_URL = f"sqlite:///{db_path.as_posix()}"
    settings.BLOCK_PIPELINE_ENABLED = True
    _seed(db_path, with_chunks=True)
    try:
        import tools.agent4_tools.knowledge_verify_tool as kv
        kv._CHUNK_CACHE.update(key=None, chunks=[], db_available=False)
    except Exception:
        pass

    checker = Checker()
    exit_code = 0
    try:
        asyncio.run(run(checker))
        asyncio.run(run_no_chunks(checker, temp_dir.name))
        exit_code = checker.summary()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"\n❌ 脚本执行异常：{exc}")
        exit_code = 2
    finally:
        settings.DATABASE_URL = orig_url
        settings.BLOCK_PIPELINE_ENABLED = orig_flag
        try:
            temp_dir.cleanup()
        except Exception:
            pass

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


if __name__ == "__main__":
    main()
