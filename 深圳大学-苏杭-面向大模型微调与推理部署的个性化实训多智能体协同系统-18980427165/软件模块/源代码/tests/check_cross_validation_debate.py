# -*- coding: utf-8 -*-
"""
交叉验证辩论方案 —— 分级审核 + 缓存 + 强制溯源 + 类型路由 断言回归
====================================================================
针对用户「辩论方案」四点直接、可断言地验证核心构件（不依赖 LLM / 网络）：

  方案 1  强制溯源      deterministic._build_sourced_claim
          有来源 → is_key_technical=True / source_ref 非空；
          无来源 → is_key_technical=False / source_ref=None / metadata.sourcing=unsupported，
                   且 confidence 被下压（不进入辩论）。

  方案 2  三级相似度漏斗 knowledge_verify_tool._classify / _fuse_with_semantic
          ≥0.80 → pass（lexical，不查向量）；<0.65 → potential_hallucination（lexical）；
          0.65~0.80 灰区且向量不可用 → review_needed + need_llm_arbitration=True。
          两个明确档「根本不需要 LLM 辩论」。

  方案 3  按类型路由     agent_04_tech_review._classify_claim_route
          有 KB 来源 → kb_fact（走全库/块核验）；
          无来源但引用「我的设备/本机」→ profile_fact（豁免）；
          无来源、非关键技术 → personalized（豁免）；
          无来源但 is_key_technical → 仍 kb_fact（交 must_fix 处理缺依据）。

  方案 4  主张级缓存     sqlite_client.save/get/get_batch_verified_claim + _claim_fingerprint
          (claim_fp+chunk_id) round-trip；重复写累加 hit_count；批量命中；
          指纹对「空白/大小写等价改写」稳定、对不同 chunk_id 天然区分。

  集成    专业知识与教学质量审核 Agent 核验接入缓存/路由 _verify_claims_hallucination
          命中 pass → 跳过送检（0 issue）；命中 hallucination → 复用裁决产出 issue（from_cache=True，不再送检）；
          profile_fact / personalized 路由 → 豁免核验（0 issue）。

运行：
  python tests/check_cross_validation_debate.py
退出码 0 表示全部断言通过，非 0 表示有失败。
"""
import asyncio
import os
import sys
import tempfile
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, ROOT_DIR)

# 纯规则/工具链路，屏蔽 LLM。
os.environ.setdefault("EXECUTION_MODE", "deterministic")

from db.sqlite_client import SQLiteClient  # noqa: E402
from agents.deterministic import _build_sourced_claim  # noqa: E402
import tools.agent4_tools.knowledge_verify_tool as kv  # noqa: E402
from agents.agent_04_tech_review import (  # noqa: E402
    TechnicalReviewAgent,
    _claim_fingerprint,
    _classify_claim_route,
    _ROUTE_KB_FACT,
    _ROUTE_PROFILE_FACT,
    _ROUTE_PERSONALIZED,
    _CACHEABLE_LEVELS,
)


# ----------------------------- 断言汇总器 -----------------------------
_passed = 0
_failed = 0
# 待释放的 SQLite 客户端（Windows 下需先 dispose 引擎再删临时库文件，否则文件被占用）
_clients: list = []


def _track(client):
    _clients.append(client)
    return client


def check(cond: bool, label: str, detail: str = "") -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  \u2705 PASS  {label}" + (f" | {detail}" if detail else ""))
    else:
        _failed += 1
        print(f"  \u274c FAIL  {label}" + (f" | {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n{'#' * 60}\n# {title}\n{'#' * 60}")


# ============================================================
# 方案 1：强制溯源
# ============================================================
def test_forced_sourcing() -> None:
    section("方案 1：个性化学习资源生成 Agent 生成侧强制溯源（_build_sourced_claim）")

    sourced = _build_sourced_claim(
        claim_id="clm_s1",
        text="LoRA 冻结原始预训练权重，只训练低秩增量矩阵 A 和 B",
        section_ref="tutorial:sec_1",
        source_id="src_kb12_0",
        chunk_id="CHK-LORA-1",
        claim_type="concept",
        confidence=0.85,
    )
    check(sourced.is_key_technical is True, "1-1 有来源 → is_key_technical=True",
          f"is_key_technical={sourced.is_key_technical}")
    check(sourced.source_ref == "src_kb12_0", "1-2 有来源 → source_ref 指向来源",
          f"source_ref={sourced.source_ref}")
    check((sourced.metadata_json or {}).get("source_chunk_id") == "CHK-LORA-1",
          "1-3 metadata 保留 source_chunk_id",
          f"source_chunk_id={(sourced.metadata_json or {}).get('source_chunk_id')}")
    check(abs(sourced.confidence - 0.85) < 1e-6, "1-4 有来源 → confidence 保留原值",
          f"confidence={sourced.confidence}")

    unsupported = _build_sourced_claim(
        claim_id="clm_u1",
        text="某些人认为该结论一定成立",  # 无引用的客观断言
        section_ref="tutorial:sec_2",
        source_id=None,
        chunk_id="",
        claim_type="fact",
        confidence=0.9,
    )
    check(unsupported.is_key_technical is False,
          "1-5 无来源 → is_key_technical=False（不进入辩论）",
          f"is_key_technical={unsupported.is_key_technical}")
    check(unsupported.source_ref is None, "1-6 无来源 → source_ref=None",
          f"source_ref={unsupported.source_ref}")
    check((unsupported.metadata_json or {}).get("sourcing") == "unsupported",
          "1-7 无来源 → metadata.sourcing=unsupported",
          f"sourcing={(unsupported.metadata_json or {}).get('sourcing')}")
    check(unsupported.confidence <= 0.3,
          "1-8 无来源 → confidence 被下压（<=0.3）",
          f"confidence={unsupported.confidence}")


# ============================================================
# 方案 2：三级相似度漏斗（明确档不辩论；灰区才升级）
# ============================================================
def test_similarity_funnel() -> None:
    section("方案 2：三级相似度漏斗（词面→向量→LLM）")

    # _classify 阈值边界（THRESHOLD_CRITICAL=0.65 / THRESHOLD_PASS=0.80）
    check(kv._classify(0.80) == "pass", "2-1 相似度 0.80 → pass（不辩论）")
    check(kv._classify(0.90) == "pass", "2-1b 相似度 0.90 → pass")
    check(kv._classify(0.64) == "potential_hallucination",
          "2-2 相似度 0.64 → potential_hallucination（不辩论）")
    check(kv._classify(0.70) == "review_needed", "2-3 相似度 0.70 → review_needed（灰区）")

    # 明确档：直接采信词面，verify_method=lexical，绝不查向量/LLM
    clear_pass = kv._fuse_with_semantic("0.85", "任意陈述", "KB-12")
    check(clear_pass["level"] == "pass" and clear_pass["verify_method"] == "lexical"
          and clear_pass["need_llm_arbitration"] is False,
          "2-4 明确 pass 档直接采信词面（零 LLM）", f"{clear_pass}")

    clear_hallu = kv._fuse_with_semantic("0.50", "任意陈述", "KB-12")
    check(clear_hallu["level"] == "potential_hallucination"
          and clear_hallu["verify_method"] == "lexical"
          and clear_hallu["need_llm_arbitration"] is False,
          "2-5 明确幻觉档直接采信词面（零 LLM）", f"{clear_hallu}")

    # 灰区 + 向量层不可用：桩掉 _semantic_similarity → None，应保持 review_needed 且请求 LLM 仲裁
    orig = kv._semantic_similarity
    try:
        kv._semantic_similarity = lambda statement, block_id="", top_k=3: None
        gray_no_vec = kv._fuse_with_semantic("0.72", "灰区陈述", "KB-12")
        check(gray_no_vec["level"] == "review_needed"
              and gray_no_vec["need_llm_arbitration"] is True,
              "2-6 灰区且向量不可用 → 保持 review_needed + 请求 LLM 仲裁", f"{gray_no_vec}")

        # 灰区 + 向量语义足够高 → 提升为 pass（同义改写救回，无需 LLM）
        kv._semantic_similarity = lambda statement, block_id="", top_k=3: 0.86
        gray_up = kv._fuse_with_semantic("0.72", "同义改写陈述", "KB-12")
        check(gray_up["level"] == "pass" and gray_up["verify_method"] == "semantic"
              and gray_up["need_llm_arbitration"] is False,
              "2-7 灰区+向量语义高 → 提升为 pass（同义改写救回）", f"{gray_up}")

        # 灰区 + 向量语义也低 → 下沉为潜在幻觉（无需 LLM）
        kv._semantic_similarity = lambda statement, block_id="", top_k=3: 0.40
        gray_down = kv._fuse_with_semantic("0.72", "语义也低的陈述", "KB-12")
        check(gray_down["level"] == "potential_hallucination"
              and gray_down["verify_method"] == "semantic"
              and gray_down["need_llm_arbitration"] is False,
              "2-8 灰区+向量语义低 → 下沉为潜在幻觉", f"{gray_down}")

        # 灰区 + 向量语义中间 → 仍需 LLM 仲裁（第三级）
        kv._semantic_similarity = lambda statement, block_id="", top_k=3: 0.65
        gray_mid = kv._fuse_with_semantic("0.72", "语义中间的陈述", "KB-12")
        check(gray_mid["level"] == "review_needed" and gray_mid["need_llm_arbitration"] is True,
              "2-9 灰区+向量语义中间 → 升级到 LLM 仲裁", f"{gray_mid}")
    finally:
        kv._semantic_similarity = orig


# ============================================================
# 方案 3：按类型路由
# ============================================================
def test_claim_routing() -> None:
    section("方案 3：按 claim_type + 溯源标记路由（_classify_claim_route）")

    kb = {"claim_type": "concept", "source_ref": "src_kb", "is_key_technical": True,
          "text": "LoRA 冻结原始权重"}
    check(_classify_claim_route(kb, "CHK-1") == _ROUTE_KB_FACT,
          "3-1 有 KB 来源 → kb_fact（走核验）")
    check(_classify_claim_route(kb, None) == _ROUTE_KB_FACT,
          "3-1b 仅 source_ref 也 → kb_fact")

    profile = {"claim_type": "fact", "is_key_technical": False,
               "text": "我的设备显存是 24GB，建议 batch_size=8"}
    check(_classify_claim_route(profile, None) == _ROUTE_PROFILE_FACT,
          "3-2 无来源 + 引用「我的设备」→ profile_fact（豁免）")

    personalized = {"claim_type": "explanation", "is_key_technical": False,
                    "text": "为帮助你理解，这里用一个通俗类比来讲解"}
    check(_classify_claim_route(personalized, None) == _ROUTE_PERSONALIZED,
          "3-3 无来源 + 非关键技术 → personalized（豁免）")

    key_no_src = {"claim_type": "fact", "is_key_technical": True,
                  "text": "QLoRA 必须使用 4-bit 量化"}
    check(_classify_claim_route(key_no_src, None) == _ROUTE_KB_FACT,
          "3-4 无来源但关键技术 → 仍 kb_fact（交 must_fix 处理缺依据）")


# ============================================================
# 方案 4：主张级缓存 round-trip
# ============================================================
def test_claim_cache(db_url: str) -> None:
    section("方案 4：跨会话主张级缓存（save/get/get_batch + 指纹）")

    client = _track(SQLiteClient(db_url))

    # 指纹稳定性：空白/大小写等价改写落同一指纹；不同 chunk_id 天然区分
    fp_a = _claim_fingerprint("LoRA 冻结 权重", "CHK-1")
    fp_b = _claim_fingerprint("  lora 冻结权重  ", "CHK-1")
    fp_c = _claim_fingerprint("LoRA 冻结 权重", "CHK-2")
    check(fp_a == fp_b, "4-1 指纹对空白/大小写等价改写稳定", f"{fp_a} == {fp_b}")
    check(fp_a != fp_c, "4-2 不同 chunk_id → 不同指纹（换来源即新主张）", f"{fp_a} != {fp_c}")

    # save → get round-trip（pass 档）
    client.save_verified_claim(fp_a, "CHK-1", {
        "level": "pass", "confidence": 0.88, "verify_method": "semantic",
        "claim_text": "LoRA 冻结权重", "evidence": {"chunk": "CHK-1", "sim": 0.88},
    })
    got = client.get_verified_claim(fp_a, "CHK-1")
    check(got is not None and got["level"] == "pass", "4-3 save→get round-trip 命中 pass",
          f"level={got and got.get('level')}")
    check(got is not None and abs(float(got["confidence"]) - 0.88) < 1e-6,
          "4-4 缓存保留 confidence", f"confidence={got and got.get('confidence')}")
    check(got is not None and got["hit_count"] == 0, "4-5 首次写入 hit_count=0",
          f"hit_count={got and got.get('hit_count')}")

    # 重复写同 key → hit_count 累加（同一幻觉/主张不重复辩论的计数证据）
    client.save_verified_claim(fp_a, "CHK-1", {
        "level": "pass", "confidence": 0.90, "verify_method": "semantic",
        "claim_text": "LoRA 冻结权重", "evidence": None,
    })
    got2 = client.get_verified_claim(fp_a, "CHK-1")
    check(got2 is not None and got2["hit_count"] == 1, "4-6 重复写 → hit_count 累加",
          f"hit_count={got2 and got2.get('hit_count')}")

    # hallucination 档缓存
    fp_h = _claim_fingerprint("LoRA 对 W 执行 SVD 分解", "CHK-1")
    client.save_verified_claim(fp_h, "CHK-1", {
        "level": "hallucination", "confidence": 0.42, "verify_method": "chunk",
        "claim_text": "LoRA 对 W 执行 SVD 分解", "evidence": "来源块无 SVD 依据",
    })

    # miss：未写入的 key 返回 None
    check(client.get_verified_claim(_claim_fingerprint("从未出现的主张", "CHK-9"), "CHK-9") is None,
          "4-7 未写入的 key → miss（None）")

    # 批量查询：命中已写入的两条，miss 项不返回
    batch = client.get_verified_claims_batch([
        (fp_a, "CHK-1"), (fp_h, "CHK-1"), (fp_a, "CHK-2"),  # 最后一项 chunk 不同 → miss
    ])
    check((fp_a, "CHK-1") in batch and (fp_h, "CHK-1") in batch,
          "4-8 批量查询命中已写入项", f"hits={list(batch.keys())}")
    check((fp_a, "CHK-2") not in batch, "4-9 批量查询不返回 miss 项")

    # _CACHEABLE_LEVELS 只含确定档（灰区不入缓存）
    check(_CACHEABLE_LEVELS == {"pass", "hallucination"},
          "4-10 仅 pass/hallucination 可缓存（灰区 review_needed 不入缓存）",
          f"{sorted(_CACHEABLE_LEVELS)}")


# ============================================================
# 集成：专业知识与教学质量审核 Agent 核验接入缓存 / 路由
# ============================================================
async def test_agent4_cache_integration(db_url: str) -> None:
    section("集成：专业知识与教学质量审核 Agent._verify_claims_hallucination 接入缓存 + 路由")

    agent = TechnicalReviewAgent()
    # 用临时库替换懒加载 DB 客户端，避免污染真实 project.db
    agent._db_client = _track(SQLiteClient(db_url))

    # 预置缓存：pass 档命中 → 应跳过送检
    pass_text = "LoRA 冻结原始预训练权重，只训练低秩增量矩阵"
    pass_fp = _claim_fingerprint(pass_text, "CHK-PASS")
    agent._db_client.save_verified_claim(pass_fp, "CHK-PASS", {
        "level": "pass", "confidence": 0.9, "verify_method": "semantic",
        "claim_text": pass_text, "evidence": None,
    })
    # 预置缓存：hallucination 档命中 → 应复用裁决产出 issue（from_cache=True），不再送检
    hallu_text = "LoRA 对原始权重执行 SVD 分解"
    hallu_fp = _claim_fingerprint(hallu_text, "CHK-HALLU")
    agent._db_client.save_verified_claim(hallu_fp, "CHK-HALLU", {
        "level": "hallucination", "confidence": 0.4, "verify_method": "chunk",
        "claim_text": hallu_text, "evidence": "来源块无 SVD 依据",
    })

    # 哨兵：若真的送检说明缓存没生效
    verify_calls = {"chunk": 0, "kb": 0}
    orig_chunk = agent.knowledge_verify_tool.verify_against_chunk
    orig_kb = agent.knowledge_verify_tool.verify_knowledge

    async def _spy_chunk(statement, chunk_id, *a, **kw):
        verify_calls["chunk"] += 1
        return await orig_chunk(statement, chunk_id, *a, **kw)

    async def _spy_kb(statement, *a, **kw):
        verify_calls["kb"] += 1
        return await orig_kb(statement, *a, **kw)

    agent.knowledge_verify_tool.verify_against_chunk = _spy_chunk
    agent.knowledge_verify_tool.verify_knowledge = _spy_kb

    try:
        claims = [
            # 缓存命中 pass → 跳过送检、无 issue
            {"claim_id": "c_pass", "text": pass_text, "is_key_technical": True,
             "section_ref": "sec_1", "source_ref": "src_pass",
             "metadata_json": {"source_chunk_id": "CHK-PASS"}},
            # 缓存命中 hallucination → 复用裁决产出 issue（from_cache=True）
            {"claim_id": "c_hallu", "text": hallu_text, "is_key_technical": True,
             "section_ref": "sec_2", "source_ref": "src_hallu",
             "metadata_json": {"source_chunk_id": "CHK-HALLU"}},
            # profile_fact 路由 → 豁免核验，无 issue
            {"claim_id": "c_profile", "text": "我的设备显存是 24GB", "is_key_technical": False,
             "section_ref": "sec_3"},
            # personalized 路由 → 豁免核验，无 issue
            {"claim_id": "c_pers", "text": "这里用一个通俗类比来帮助你理解", "is_key_technical": False,
             "section_ref": "sec_4"},
        ]

        errors = await agent._verify_claims_hallucination(claims)

        by_claim = {e.get("claim_id"): e for e in errors}
        check("c_pass" not in by_claim, "I-1 缓存命中 pass → 跳过送检，无 issue",
              f"issue_claims={list(by_claim.keys())}")
        check("c_hallu" in by_claim and by_claim["c_hallu"].get("from_cache") is True,
              "I-2 缓存命中 hallucination → 复用裁决产出 issue（from_cache=True）",
              f"from_cache={by_claim.get('c_hallu', {}).get('from_cache')}")
        check(by_claim.get("c_hallu", {}).get("error_type") == "hallucination",
              "I-3 复用的缓存 issue error_type=hallucination")
        check("c_profile" not in by_claim and "c_pers" not in by_claim,
              "I-4 profile_fact / personalized 路由 → 豁免核验，无 issue")
        check(verify_calls["chunk"] == 0 and verify_calls["kb"] == 0,
              "I-5 全部命中缓存/豁免 → 零真实送检调用",
              f"verify_calls={verify_calls}")
    finally:
        agent.knowledge_verify_tool.verify_against_chunk = orig_chunk
        agent.knowledge_verify_tool.verify_knowledge = orig_kb


# ----------------------------- 主流程 -----------------------------
async def _amain() -> int:
    print("=" * 60)
    print("交叉验证辩论方案 —— 分级审核 + 缓存 + 强制溯源 + 类型路由")
    print("=" * 60)

    test_forced_sourcing()
    test_similarity_funnel()
    test_claim_routing()

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db_path = Path(tmp) / "cross_validation.db"
        db_url = f"sqlite:///{db_path.as_posix()}"
        try:
            test_claim_cache(db_url)
            await test_agent4_cache_integration(db_url)
        finally:
            # Windows：删临时库前必须先释放 SQLAlchemy 引擎持有的文件句柄。
            for c in _clients:
                try:
                    c.close()
                except Exception:
                    pass

    print(f"\n{'#' * 60}")
    print(f"# 断言汇总：{_passed} 通过 / {_failed} 失败")
    print(f"{'#' * 60}")
    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    code = asyncio.run(_amain())
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
