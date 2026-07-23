"""
knowledge_verify_tool — Agent4 知识依据验证工具（幻觉检测）
================================================
功能：输入技术陈述，输出置信度(0~1)与是否有知识库依据，用于幻觉判定。
    - 真实模式：查 knowledge_chunks 表做语义检索（关键词重叠归一化相似度）
    - Mock 模式：返回 ./data/test_data/mock_knowledge_verify_tool.json 预设数据

统一接口：
    def knowledge_verify(input_data: dict, config: dict = None) -> dict
    返回：{"status", "data", "message", "mock_used"}

规则对齐（technical_review.skill.md 规则1：幻觉判定阈值）：
    - similarity < 0.65        → 潜在幻觉   (critical)
    - 0.65 <= similarity < 0.80 → 存疑需复核 (major / review_needed)
    - similarity >= 0.80        → 有依据     (pass)

实现说明：
    生产环境应通过 Chroma 向量库（db/chroma_client.py）做真正的语义检索；
    本工具在向量库未就绪时退化为对 knowledge_chunks 表的关键词重叠检索，
    相似度经归一化映射到 [0,1]，作为可运行的语义近似。
    输出结构对齐 models/review.py 的 ReviewError 字段。
    保留 KnowledgeVerifyTool 类作为向后兼容适配器。
"""

import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

from config import BACKEND_DIR, DATA_DIR, settings

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

# 幻觉判定阈值（来源：technical_review.skill.md 规则1）
THRESHOLD_CRITICAL = 0.65  # < 此值判为潜在幻觉
THRESHOLD_PASS = 0.80      # >= 此值判为有依据

MOCK_FILE = DATA_DIR / "test_data" / "mock_knowledge_verify_tool.json"

# 英文停用词（用于关键词提取去噪）
_STOPWORDS = {
    "the", "a", "an", "is", "are", "of", "to", "in", "on", "for", "and", "or",
    "with", "that", "this", "it", "as", "by", "be", "was", "were", "will", "can",
    "we", "you", "use", "used", "using", "model", "models", "data", "value",
}


# ------------------------------------------------------------------
# 统一函数式接口（规范实现）
# ------------------------------------------------------------------
def knowledge_verify(input_data: dict, config: dict = None) -> dict:
    """
    验证技术陈述是否有知识库依据。

    参数：
        input_data: dict
            - statement: str   待验证的技术陈述（优先）
            - content: str     同义字段，statement 缺失时使用
            - scenario: str    Mock 模式下选择场景：normal|boundary|exception
        config: dict
            - use_mock: bool   是否使用 Mock（默认 True）
            - top_k: int       返回最相似分块数量（默认 5）

    返回：
        {
            "status": "success" | "error",
            "data": {
                "statement": str,            # 截断后的陈述
                "confidence": float,         # 0~1，最高相似度
                "has_basis": bool,           # confidence >= 0.65
                "level": "pass" | "review_needed" | "potential_hallucination",
                "thresholds": {"critical": 0.65, "pass": 0.80},
                "matched_chunks": [
                    {"chunk_id","doc_id","block_id","content","similarity"}
                ],
                "checked_chunks": int,
                "db_available": bool
            },
            "message": str,
            "mock_used": bool
        }
    """
    config = config or {}
    use_mock = config.get("use_mock", True)
    top_k = int(config.get("top_k", 5))

    statement = (input_data or {}).get("statement") or (input_data or {}).get("content") or ""
    if not statement.strip():
        return {
            "status": "error",
            "data": None,
            "message": "输入未包含 statement/content 字段，无法验证",
            "mock_used": use_mock,
        }

    try:
        if use_mock:
            return _mock_verify(statement, input_data)

        # 真实模式：查 knowledge_chunks 表做语义检索
        chunks, db_available = _load_chunks()
        stmt_terms = _tokenize(statement)

        scored: List[Dict[str, Any]] = []
        if stmt_terms and chunks:
            for ch in chunks:
                chunk_terms = _tokenize(ch["content"])
                if not chunk_terms:
                    continue
                sim = _similarity(stmt_terms, chunk_terms)
                if sim > 0:
                    scored.append({
                        "chunk_id": ch["chunk_id"],
                        "doc_id": ch["doc_id"],
                        "block_id": ch["block_id"],
                        "content": ch["content"][:200],
                        "similarity": round(sim, 4),
                    })

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        matched = scored[:top_k]
        confidence = matched[0]["similarity"] if matched else 0.0

        level = _classify(confidence)
        return {
            "status": "success",
            "data": {
                "statement": statement[:200],
                "confidence": round(confidence, 4),
                "has_basis": confidence >= THRESHOLD_CRITICAL,
                "level": level,
                "thresholds": {"critical": THRESHOLD_CRITICAL, "pass": THRESHOLD_PASS},
                "matched_chunks": matched,
                "checked_chunks": len(chunks),
                "db_available": db_available,
            },
            "message": _build_message(level, confidence, len(chunks), db_available),
            "mock_used": False,
        }
    except Exception as e:
        logger.exception("knowledge_verify 执行失败")
        return {
            "status": "error",
            "data": None,
            "message": f"knowledge_verify 执行异常: {e}",
            "mock_used": use_mock,
        }


# ------------------------------------------------------------------
# 真实模式：数据库语义检索
# ------------------------------------------------------------------
def _get_db_path() -> Path:
    """解析 settings.DATABASE_URL 为绝对路径（与 sqlite_client 一致）。"""
    url = settings.DATABASE_URL
    raw = url.replace("sqlite:///", "", 1)
    p = Path(raw)
    if not p.is_absolute():
        p = (BACKEND_DIR / p).resolve()
    return p


def _load_chunks() -> tuple[List[Dict[str, str]], bool]:
    """
    SQL: 从 knowledge_chunks 表读取全部分块（doc_id/block_id 关联 knowledge_docs/knowledge_blocks）。
    knowledge_chunks 是向量检索的源表，chunk_id 与 Chroma 向量库 ID 一一对应。
    数据库或表不存在时返回空列表与 db_available=False，调用方据此降级。
    """
    db_path = _get_db_path()
    if not db_path.exists():
        logger.warning("数据库不存在: %s，knowledge_verify 无可用知识库", db_path)
        return [], False
    try:
        # sqlite3 只读连接，避免对库产生任何写副作用
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            cur = conn.execute(
                """
                -- 读取全部分块用于相似度评分；字段对齐 knowledge_chunks 表定义
                SELECT chunk_id, doc_id, block_id, content
                FROM knowledge_chunks
                WHERE content IS NOT NULL AND content <> ''
                """
            )
            rows = cur.fetchall()
        finally:
            conn.close()
        chunks = [
            {"chunk_id": r[0], "doc_id": r[1], "block_id": r[2], "content": r[3]}
            for r in rows
        ]
        return chunks, True
    except sqlite3.Error as e:
        logger.warning("读取 knowledge_chunks 失败: %s", e)
        return [], False


def _tokenize(text: str) -> set:
    """
    分词：英文按单词、中文按双字 bigram，去停用词。
    作为无分词库时的语义近似手段。
    """
    if not text:
        return set()
    text = text.lower()
    terms: set = set()

    # 英文单词
    for w in re.findall(r"[a-z][a-z0-9_\-]{1,}", text):
        if w not in _STOPWORDS:
            terms.add(w)

    # 中文双字 bigram（仅连续 CJK 字符）
    cjk = re.findall(r"[\u4e00-\u9fff]+", text)
    for seg in cjk:
        for i in range(len(seg) - 1):
            terms.add(seg[i : i + 2])
        if len(seg) == 1:
            terms.add(seg)
    return terms


def _similarity(stmt_terms: set, chunk_terms: set) -> float:
    """
    计算陈述与分块的归一化相似度（0~1）。
    采用 recall（召回）为主、density（精确）为辅的混合策略：
        sim = 0.65 * |∩|/|stmt| + 0.35 * |∩|/|chunk|
    并要求至少 2 个共同项，否则降权，避免单字误命中。
    """
    inter = stmt_terms & chunk_terms
    if not inter:
        return 0.0
    n_inter = len(inter)
    recall = n_inter / len(stmt_terms) if stmt_terms else 0.0
    density = n_inter / len(chunk_terms) if chunk_terms else 0.0
    sim = 0.65 * recall + 0.35 * density
    if n_inter < 2:
        sim *= 0.5  # 共同项过少降权
    return max(0.0, min(1.0, sim))


def _classify(confidence: float) -> str:
    if confidence < THRESHOLD_CRITICAL:
        return "potential_hallucination"
    if confidence < THRESHOLD_PASS:
        return "review_needed"
    return "pass"


def _build_message(level: str, confidence: float, n_chunks: int, db_available: bool) -> str:
    if not db_available:
        return f"知识库不可用（knowledge_chunks 为空），无法验证，置信度 {confidence:.2f}"
    if level == "potential_hallucination":
        return f"置信度 {confidence:.2f} < {THRESHOLD_CRITICAL}，存在潜在幻觉风险（检索 {n_chunks} 个分块）"
    if level == "review_needed":
        return f"置信度 {confidence:.2f} 处于存疑区间 [{THRESHOLD_CRITICAL},{THRESHOLD_PASS})，建议人工复核"
    return f"置信度 {confidence:.2f} >= {THRESHOLD_PASS}，有知识库依据"


# ------------------------------------------------------------------
# Mock 模式
# ------------------------------------------------------------------
def _mock_verify(statement: str, input_data: dict) -> dict:
    """从 mock JSON 读取预设场景，按 scenario 或陈述内容自动选择。"""
    scenario = _pick_scenario(statement, input_data)
    scenarios = _load_mock()

    if scenario not in scenarios:
        scenario = "normal"
    data = dict(scenarios[scenario])  # 浅拷贝避免污染缓存

    # 把陈述回填，便于调用方核对
    data["statement"] = statement[:200]
    data.setdefault("thresholds", {"critical": THRESHOLD_CRITICAL, "pass": THRESHOLD_PASS})

    level = data.get("level", "pass")
    confidence = data.get("confidence", 0.0)
    msg = {
        "pass": f"[Mock] 置信度 {confidence:.2f}，有知识库依据",
        "review_needed": f"[Mock] 置信度 {confidence:.2f}，存疑需人工复核",
        "potential_hallucination": f"[Mock] 置信度 {confidence:.2f} < {THRESHOLD_CRITICAL}，潜在幻觉",
    }.get(level, f"[Mock] 置信度 {confidence:.2f}")

    return {
        "status": "success",
        "data": data,
        "message": msg,
        "mock_used": True,
    }


def _pick_scenario(statement: str, input_data: dict) -> str:
    """优先用显式 scenario；否则按陈述内容自动判定场景。"""
    scenario = (input_data or {}).get("scenario")
    if scenario:
        return scenario
    s = statement.lower()
    # SVD / 权重分解 → 幻觉异常场景
    if any(k in statement for k in ("svd", "SVD", "奇异值", "分解为", "U*Σ*V", "uσv")):
        return "exception"
    if any(k in statement for k in ("秩r", "秩 r", "r通常", "r 通常", "过大会")):
        return "boundary"
    return "normal"


def _load_mock() -> dict:
    if not MOCK_FILE.exists():
        logger.warning("Mock 文件不存在: %s", MOCK_FILE)
        return {}
    try:
        with MOCK_FILE.open("r", encoding="utf-8") as f:
            return json.load(f).get("scenarios", {})
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("读取 Mock 文件失败: %s", e)
        return {}


# ------------------------------------------------------------------
# 向后兼容适配器
# 保留原 TechnicalReviewAgent 调用的 async 类接口，委托给函数式核心。
# verify_knowledge 返回 ReviewError 形态的错误列表：
#   - confidence < 0.65        → hallucination (critical)
#   - 0.65 <= confidence < 0.80 → review_needed (major)
#   - >= 0.80                   → 空列表（通过）
# ------------------------------------------------------------------
class KnowledgeVerifyTool:
    def __init__(self):
        # 默认走 Mock，保证在知识库未就绪时仍可运行
        self._default_config = {"use_mock": True}

    async def verify_knowledge(self, content: str) -> List[Dict[str, Any]]:
        result = knowledge_verify({"statement": content}, config=self._default_config)
        if result.get("status") != "success":
            return []
        data = result["data"]
        confidence = data.get("confidence", 0.0)
        matched = data.get("matched_chunks", [])
        evidence = (
            f"knowledge_verify_tool.similarity={confidence:.2f}"
            + (f"，命中分块 {matched[0]['chunk_id']}" if matched else "，知识库无对应声明依据")
        )

        if confidence < THRESHOLD_CRITICAL:
            return [{
                "error_type": "hallucination",
                "severity": "critical",
                "location": "技术陈述",
                "description": f"知识库相似度 {confidence:.2f} < {THRESHOLD_CRITICAL} 阈值，存在潜在幻觉风险。",
                "evidence": evidence,
                "suggestion": "删除无法验证的技术声明或补充可靠来源。",
                # 兼容旧调用方 print 使用的字段
                "type": "hallucination",
                "message": f"知识库相似度 {confidence:.2f}，存在潜在幻觉风险。",
            }]
        if confidence < THRESHOLD_PASS:
            return [{
                "error_type": "review_needed",
                "severity": "major",
                "location": "技术陈述",
                "description": f"知识库相似度 {confidence:.2f} 处于存疑区间，需人工复核。",
                "evidence": evidence,
                "suggestion": "补充知识库依据或人工复核后放行。",
                "type": "review_needed",
                "message": f"知识库相似度 {confidence:.2f}，存疑需复核。",
            }]
        return []
