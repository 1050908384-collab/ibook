# -*- coding: utf-8 -*-
"""
knowledge_verify_tool — 专业知识与教学质量审核 Agent 知识依据验证工具（幻觉检测 + 代码错误匹配）
======================================================================
功能：
  模式1（陈述验证）：输入技术陈述，输出置信度(0~1)与是否有知识库依据，用于幻觉判定。
  模式2（代码错误验证）：输入 code_executor_tool 返回的报错信息，匹配知识库中已知错误模式，
                       返回 is_known_error（布尔）和 known_solution（字符串）。

    - 真实模式：查 knowledge_chunks 表做语义检索（关键词重叠归一化相似度）
    - Mock 模式：返回 ./data/test_data/mock_knowledge_verify_tool.json 预设数据

统一接口：
    def knowledge_verify(input_data: dict, config: dict = None) -> dict
    返回：{"status", "data", "message", "mock_used"}

规则对齐（technical_review.skill.md 规则1：幻觉判定阈值）:
    - similarity < 0.65        → 潜在幻觉   (critical)
    - 0.65 <= similarity < 0.80 → 存疑需复核 (major / review_needed)
    - similarity >= 0.80        → 有依据     (pass)

实现说明：
    本工具不依赖 Chroma。块级审核优先按 source_chunk_id 精确回查 SQLite
    knowledge_chunks；没有明确来源块的陈述审核，才使用全库关键词重叠相似度
    作为候选依据强度参考。该分数不是严格的语义等价或事实正确性证明。
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

# 「库外/超纲」判定阈值：相似度极低且几乎零命中时，说明知识库根本没有覆盖该主题，
# 与「库内但表述错误」的幻觉本质不同——不应逼 个性化学习资源生成 Agent 反复改写，而应放行 个性化学习资源生成 Agent 内容。
# 见问答链路库外诊断（tests/check_qa_out_of_kb_diagnostic.py）：库外提问相似度普遍 <= 0.02。
THRESHOLD_OUT_OF_SCOPE = 0.10

MOCK_FILE = DATA_DIR / "test_data" / "mock_knowledge_verify_tool.json"

# knowledge_chunks 进程内缓存（按 db 路径 + mtime 失效），见 _load_chunks
_CHUNK_CACHE: Dict[str, Any] = {"key": None, "chunks": [], "db_available": False}

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
    知识依据验证，支持两种模式：

    模式1 — 陈述验证（默认）：
        输入技术陈述，输出置信度(0~1)与是否有知识库依据，用于幻觉判定。
        input_data:
            - statement: str   待验证的技术陈述（优先）
            - content: str     同义字段，statement 缺失时使用
            - scenario: str    Mock 模式下选择场景：normal|boundary|exception

    模式2 — 代码错误验证：
        输入 code_executor_tool 返回的报错信息，匹配知识库中已知错误模式。
        input_data:
            - error_info: str   报错信息字符串（code_executor_tool 返回）
            - code: str         可选，出错的代码片段

    返回（两种模式均返回统一结构）：
        {
            "status": "success" | "error",
            "data": { ... },
            "message": str,
            "mock_used": bool
        }
    """
    config = config or {}
    use_mock = config.get("use_mock", True)

    error_info = (input_data or {}).get("error_info") or ""
    if error_info.strip():
        return _verify_code_error(input_data, config)

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

        # 安全兜底：知识库为空/不可用时，真实检索会对每条陈述判 similarity=0.0，
        # 从而把所有内容误报为 hallucination（critical）。此时缺乏可比对语料，
        # 只能标记为“未核验”，不能用 Mock 或低置信度替代事实判断。
        # （知识库导入分块后，chunks 非空即自动走真实检索。）
        if not chunks:
            logger.warning(
                "knowledge_chunks 为空或不可用（db_available=%s），跳过幻觉判定",
                db_available,
            )
            return {
                "status": "success",
                "data": {
                    "statement": statement[:200],
                    "confidence": 0.0,
                    "has_basis": None,
                    "level": "unverifiable",
                    "thresholds": {"critical": THRESHOLD_CRITICAL, "pass": THRESHOLD_PASS},
                    "matched_chunks": [],
                    "checked_chunks": 0,
                    "db_available": db_available,
                    "fallback_reason": "knowledge_chunks_empty",
                    "verification_skipped": True,
                },
                "message": "knowledge_chunks 为空或不可用，已跳过事实核验，不产生幻觉结论。",
                "mock_used": False,
            }

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
                        "heading_path": ch.get("heading_path"),
                        "source_line_start": ch.get("source_line_start"),
                        "source_line_end": ch.get("source_line_end"),
                        "chunk_index": ch.get("chunk_index"),
                    })

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        matched = scored[:top_k]
        lexical_confidence = matched[0]["similarity"] if matched else 0.0

        # 三级漏斗：明确档直接采信词面；灰区 [0.65,0.80) 用向量语义复判，
        # 仍模糊才标记 need_llm_arbitration 交上层做一次 LLM 蕴含仲裁。
        block_hint = matched[0]["block_id"] if matched else ""
        fused = _fuse_with_semantic(lexical_confidence, statement, block_hint)
        level = fused["level"]
        confidence = fused["confidence"]
        return {
            "status": "success",
            "data": {
                "statement": statement[:200],
                "confidence": round(confidence, 4),
                "has_basis": confidence >= THRESHOLD_CRITICAL,
                "level": level,
                "verify_method": fused["verify_method"],
                "need_llm_arbitration": fused["need_llm_arbitration"],
                "lexical_confidence": round(lexical_confidence, 4),
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
    knowledge_chunks 是 专业知识与教学质量审核 Agent 的事实核验来源表，chunk_id 用于精确绑定 个性化学习资源生成 Agent 来源块。
    数据库或表不存在时返回空列表与 db_available=False，调用方据此降级。

    带进程内缓存：按 (db路径, 文件mtime) 缓存结果，避免同一次审核对多条 Claim 逐条送检时
    反复全表读取。知识库文件更新（mtime 变化）后缓存自动失效。
    """
    db_path = _get_db_path()
    if not db_path.exists():
        logger.warning("数据库不存在: %s，knowledge_verify 无可用知识库", db_path)
        return [], False

    try:
        mtime = db_path.stat().st_mtime
    except OSError:
        mtime = None
    cache_key = (str(db_path), mtime)
    if _CHUNK_CACHE.get("key") == cache_key:
        return _CHUNK_CACHE["chunks"], _CHUNK_CACHE["db_available"]

    try:
        # sqlite3 只读连接，避免对库产生任何写副作用
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            # 防御式探测新列（heading_path/source_line_*/chunk_index），缺失即以 NULL 降级，
            # 保证「数据侧尚未入库新字段」时不崩溃（见《知识库分块新增字段说明.md》）。
            try:
                cols = {r[1] for r in conn.execute("PRAGMA table_info(knowledge_chunks)").fetchall()}
            except sqlite3.Error:
                cols = set()
            optional = ["heading_path", "source_line_start", "source_line_end", "chunk_index"]
            select_parts = ["chunk_id", "doc_id", "block_id", "content"]
            for col in optional:
                select_parts.append(col if col in cols else f"NULL AS {col}")
            cur = conn.execute(
                f"""
                -- 读取全部分块用于相似度评分；字段对齐 knowledge_chunks 表定义
                SELECT {', '.join(select_parts)}
                FROM knowledge_chunks
                WHERE content IS NOT NULL AND content <> ''
                """
            )
            rows = cur.fetchall()
        finally:
            conn.close()
        chunks = [
            {
                "chunk_id": r[0], "doc_id": r[1], "block_id": r[2], "content": r[3],
                "heading_path": r[4], "source_line_start": r[5],
                "source_line_end": r[6], "chunk_index": r[7],
            }
            for r in rows
        ]
        _CHUNK_CACHE.update(key=cache_key, chunks=chunks, db_available=True)
        return chunks, True
    except sqlite3.Error as e:
        logger.warning("读取 knowledge_chunks 失败: %s", e)
        return [], False


def _load_chunk_by_id(chunk_id: str) -> Dict[str, Any] | None:
    """按 chunk_id 从缓存后的全量分块中定位单个来源块（供 verify_against_chunk）。

    复用 _load_chunks 的进程内缓存，避免重复全表读取；库不可用/未找到返回 None。
    """
    if not chunk_id:
        return None
    try:
        chunks, _ = _load_chunks()
    except Exception:
        return None
    for ch in chunks:
        if ch.get("chunk_id") == chunk_id:
            return ch
    return None


def _format_in_chunk_position(match: Dict[str, Any] | None) -> str | None:
    """根据命中块的原文行号构造块内定位串（缺列则返回 None）。"""
    if not match:
        return None
    start = match.get("source_line_start")
    end = match.get("source_line_end")
    if start is not None and end is not None:
        return f"lines:{start}-{end}"
    if start is not None:
        return f"lines:{start}"
    idx = match.get("chunk_index")
    if idx is not None:
        return f"chunk_index:{idx}"
    return None


def _build_chunk_evidence(confidence: float, match: Dict[str, Any] | None) -> str:
    """构造带来源块/标题路径/原文行号的证据串（字段缺失优雅省略）。"""
    if not match:
        return f"knowledge_verify_tool.similarity={confidence:.2f}，知识库无对应声明依据"
    parts = [f"knowledge_verify_tool.similarity={confidence:.2f}"]
    if match.get("chunk_id"):
        parts.append(f"chunk={match['chunk_id']}")
    if match.get("heading_path"):
        parts.append(f"heading_path={match['heading_path']}")
    pos = _format_in_chunk_position(match)
    if pos and pos.startswith("lines:"):
        parts.append(pos.replace("lines:", "lines="))
    return "，".join(parts)


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


# ------------------------------------------------------------------
# 三级漏斗第二级：向量语义复判（仅在词面法落入灰区时启用）
# ------------------------------------------------------------------
# 词面重叠（_similarity）无法识别「同义改写」（如"冻结基座权重" vs "锁定预训练参数"），
# 也可能被「只差一个否定词」误判为高相似。因此对落入灰区 [0.65,0.80) 的陈述，用
# qwen3-embedding 语义向量再判一次：语义足够高 → 提升为 pass；语义也低 → 下沉为潜在
# 幻觉；仍处于中间地带 → 标记 need_llm_arbitration，交上层做一次 LLM 蕴含仲裁。
# 明确档（>=0.80 / <0.65）不进入本级，零额外开销。
SEMANTIC_PASS = 0.80       # 语义相似度 >= 此值：视为有依据
SEMANTIC_CRITICAL = 0.55   # 语义相似度 < 此值：视为潜在幻觉


def _semantic_similarity(statement: str, block_id: str = "", top_k: int = 3) -> float | None:
    """用 Chroma(qwen3-embedding) 计算陈述与知识库的最高语义相似度。

    返回 0~1；Chroma 不可用/无结果/未安装时返回 None（由调用方回退到词面结论，
    绝不因向量层缺失而改判）。复用 vector_search_tool 的检索管线，保持同一向量空间。
    """
    statement = (statement or "").strip()
    if not statement:
        return None
    try:
        # 延迟导入：向量检索是可选增强，缺 chromadb 时不影响词面主链路。
        from tools.agent3_tools.vector_search_tool import _search_chroma
    except Exception:
        return None
    try:
        block_ids = [block_id] if block_id else []
        results = _search_chroma(statement, block_ids, [], top_k)
    except Exception:
        return None
    if not results:
        return None
    return max((r.get("similarity") or 0.0) for r in results)


def _fuse_with_semantic(lexical_confidence: str, statement: str, block_id: str = "") -> Dict[str, Any]:
    """三级漏斗核心：给词面置信度做语义复判，产出最终 level 与判定方式。

    返回 {"level", "confidence", "verify_method", "need_llm_arbitration"}。
    - 明确档（>=0.80 / <0.65）直接采信词面，verify_method="lexical"，不查向量。
    - 灰区 [0.65,0.80)：查向量语义；语义高→pass、语义低→潜在幻觉、语义中间→need_llm。
    """
    lex = float(lexical_confidence)
    lex_level = _classify(lex)
    if lex_level != "review_needed":
        # 明确档：词面已足够，不额外花向量/LLM。
        return {
            "level": lex_level,
            "confidence": round(lex, 4),
            "verify_method": "lexical",
            "need_llm_arbitration": False,
        }

    sem = _semantic_similarity(statement, block_id)
    if sem is None:
        # 向量层不可用：保持灰区结论，交由上层决定是否 LLM 仲裁。
        return {
            "level": "review_needed",
            "confidence": round(lex, 4),
            "verify_method": "lexical",
            "need_llm_arbitration": True,
        }
    if sem >= SEMANTIC_PASS:
        return {"level": "pass", "confidence": round(sem, 4),
                "verify_method": "semantic", "need_llm_arbitration": False}
    if sem < SEMANTIC_CRITICAL:
        return {"level": "potential_hallucination", "confidence": round(sem, 4),
                "verify_method": "semantic", "need_llm_arbitration": False}
    # 语义仍处中间地带：词面/向量都不敢下结论 → 交 LLM 蕴含仲裁（第三级）。
    return {"level": "review_needed", "confidence": round(sem, 4),
            "verify_method": "semantic", "need_llm_arbitration": True}


def _build_message(level: str, confidence: float, n_chunks: int, db_available: bool) -> str:
    if not db_available:
        return f"知识库不可用（knowledge_chunks 为空），无法验证，置信度 {confidence:.2f}"
    if level == "potential_hallucination":
        return f"置信度 {confidence:.2f} < {THRESHOLD_CRITICAL}，存在潜在幻觉风险（检索 {n_chunks} 个分块）"
    if level == "review_needed":
        return f"置信度 {confidence:.2f} 处于存疑区间 [{THRESHOLD_CRITICAL},{THRESHOLD_PASS})，建议人工复核"
    return f"置信度 {confidence:.2f} >= {THRESHOLD_PASS}，有知识库依据"


# ------------------------------------------------------------------
# 代码错误验证模式：匹配 code_executor_tool 返回的报错信息
# ------------------------------------------------------------------

# 常见错误类型映射（错误信息 → 标准化错误类别）
_ERROR_TYPE_MAP = {
    "typeerror": "TypeError",
    "valueerror": "ValueError",
    "keyerror": "KeyError",
    "indexerror": "IndexError",
    "attributeerror": "AttributeError",
    "nameerror": "NameError",
    "syntaxerror": "SyntaxError",
    "indentationerror": "IndentationError",
    "taberror": "TabError",
    "zerodivisionerror": "ZeroDivisionError",
    "memoryerror": "MemoryError",
    "notimplementederror": "NotImplementedError",
    "importerror": "ImportError",
    "modulenotfounderror": "ModuleNotFoundError",
    "filenotfounderror": "FileNotFoundError",
    "permissionerror": "PermissionError",
    "connectionerror": "ConnectionError",
    "timeouterror": "TimeoutError",
    "jsondecodeerror": "JSONDecodeError",
    "jsonencodeerror": "JSONEncodeError",
    "httpx": "HTTPError",
    "requestexception": "RequestException",
    "cuda": "CUDAError",
    "pytorch": "PyTorchError",
    "torcherror": "PyTorchError",
    "torch": "PyTorchError",
    "tensorflow": "TensorFlowError",
    "runtimeerror": "RuntimeError",
    "jsondecode": "JSONDecodeError",
}


# 错误严重等级映射（code_error_patterns.skill.md 规则1）
_SEVERITY_MAP: Dict[str, str] = {}
for _et in ("SyntaxError", "IndentationError", "TabError"):
    _SEVERITY_MAP[_et] = "critical"
for _et in (
    "TypeError", "ValueError", "KeyError", "IndexError", "AttributeError",
    "NameError", "ZeroDivisionError", "MemoryError",
    "ImportError", "ModuleNotFoundError", "FileNotFoundError", "PermissionError",
    "ConnectionError", "TimeoutError", "OSError", "IOError",
    "CUDAError", "PyTorchError", "TensorFlowError", "TorchError",
    "JSONDecodeError", "JSONEncodeError", "HTTPError", "RequestException",
    "RuntimeError", "RecursionError", "OverflowError", "UnboundLocalError",
):
    _SEVERITY_MAP[_et] = "major"
for _et in ("AssertionError", "NotImplementedError", "StopIteration", "GeneratorExit", "Exception"):
    _SEVERITY_MAP[_et] = "minor"


# 错误归因映射（code_error_patterns.skill.md 规则2）
_ENVIRONMENT_ERRORS = {
    "ModuleNotFoundError", "ImportError", "FileNotFoundError", "PermissionError",
    "IsADirectoryError", "NotADirectoryError", "ConnectionError", "TimeoutError",
    "OSError", "IOError", "CUDAError", "PyTorchError", "TensorFlowError", "TorchError",
    "MemoryError",
}

_DEPENDENCY_ERRORS = {"AttributeError", "ImportError", "TypeError"}

_DATA_ERRORS = {"JSONDecodeError", "JSONEncodeError", "ValueError", "KeyError"}

_CODE_ERRORS = {
    "SyntaxError", "IndentationError", "TabError",
    "KeyError", "IndexError", "TypeError", "AttributeError",
    "NameError", "ZeroDivisionError", "AssertionError",
    "RecursionError", "OverflowError", "UnboundLocalError",
}


def classify_error_severity(error_type: str) -> str:
    """
    按 code_error_patterns.skill.md 规则1 将错误类型映射为严重等级。

    返回: "critical" / "major" / "minor"
    """
    return _SEVERITY_MAP.get(error_type, "major")


def classify_error_root_cause(error_type: str, error_info: str = "") -> str:
    """
    按 code_error_patterns.skill.md 规则2 判定错误归因类型。

    返回: "code_issue" / "environment_issue" / "dependency_issue" / "data_issue" / "code_unknown"
    """
    if not error_type:
        return "code_unknown"

    if error_type in _ENVIRONMENT_ERRORS:
        return "environment_issue"

    info_lower = (error_info or "").lower()
    dep_keywords = ("deprecated", "removed", "not found in module", "version", "incompatible", "api")
    if error_type in _DEPENDENCY_ERRORS and any(kw in info_lower for kw in dep_keywords):
        return "dependency_issue"

    data_keywords = ("parse", "format", "invalid", "unexpected value", "cannot parse", "missing field")
    if error_type in _DATA_ERRORS and any(kw in info_lower for kw in data_keywords):
        return "data_issue"

    if error_type in _CODE_ERRORS:
        return "code_issue"

    return "code_unknown"


def _extract_error_type(error_info: str) -> str:
    """从报错信息中提取标准化的错误类型。"""
    if not error_info:
        return ""
    text_lower = error_info.lower()
    for keyword, error_type in _ERROR_TYPE_MAP.items():
        if keyword in text_lower:
            return error_type
    return ""


def _extract_error_keywords(error_info: str) -> set:
    """从报错信息中提取关键词（去除错误类型名和停用词后）。"""
    if not error_info:
        return set()
    text = error_info.lower()
    terms: set = set()

    for w in re.findall(r"[a-z][a-z0-9_\-]{1,}", text):
        if w not in _STOPWORDS and len(w) > 2:
            terms.add(w)

    cjk = re.findall(r"[\u4e00-\u9fff]+", text)
    for seg in cjk:
        for i in range(len(seg) - 1):
            terms.add(seg[i : i + 2])
        if len(seg) == 1:
            terms.add(seg)

    return terms


def _match_error_pattern(error_info: str, code: str = "") -> Dict[str, Any]:
    """
    将报错信息与知识库中的错误模式进行匹配。

    策略：
    1. 提取错误类型（如 TypeError）
    2. 提取报错关键词
    3. 在 knowledge_chunks 中搜索包含错误类型且与关键词匹配的分块
    4. 从匹配分块中提取解决方案（包含"解决"、"fix"、"修复"、"建议"等关键词的段落）

    返回：
        {
            "is_known_error": bool,
            "known_solution": str,
            "error_type": str,
            "matched_chunk": dict or None,
            "confidence": float,
        }
    """
    error_type = _extract_error_type(error_info)
    error_terms = _extract_error_keywords(error_info)

    if not error_type and not error_terms:
        return {
            "is_known_error": False,
            "known_solution": "",
            "error_type": "",
            "matched_chunk": None,
            "confidence": 0.0,
        }

    try:
        chunks, db_available = _load_chunks()
    except Exception:
        chunks, db_available = [], False

    if not chunks:
        return {
            "is_known_error": False,
            "known_solution": "",
            "error_type": error_type,
            "matched_chunk": None,
            "confidence": 0.0,
        }

    scored: List[Dict[str, Any]] = []
    for ch in chunks:
        content_lower = ch["content"].lower()
        chunk_terms = _tokenize(ch["content"])

        has_error_type = error_type.lower() in content_lower if error_type else False
        sim = _similarity(error_terms, chunk_terms) if error_terms else 0.0

        if has_error_type:
            sim = max(sim, 0.5)

        if sim > 0:
            solution = _extract_solution(ch["content"])
            scored.append({
                "chunk_id": ch["chunk_id"],
                "doc_id": ch["doc_id"],
                "block_id": ch["block_id"],
                "content": ch["content"][:300],
                "similarity": round(sim, 4),
                "has_error_type": has_error_type,
                "solution": solution,
            })

    scored.sort(key=lambda x: x["similarity"], reverse=True)

    if scored:
        best = scored[0]
        confidence = best["similarity"]
        is_known = confidence >= 0.5 or best["has_error_type"]
        solution = best["solution"] or (best["content"][:200] if is_known else "")

        return {
            "is_known_error": is_known,
            "known_solution": solution,
            "error_type": error_type,
            "severity": classify_error_severity(error_type),
            "root_cause": classify_error_root_cause(error_type, error_info),
            "matched_chunk": {
                "chunk_id": best["chunk_id"],
                "doc_id": best["doc_id"],
                "block_id": best["block_id"],
                "content": best["content"],
            },
            "confidence": round(confidence, 4),
        }

    return {
        "is_known_error": False,
        "known_solution": "",
        "error_type": error_type,
        "severity": classify_error_severity(error_type),
        "root_cause": classify_error_root_cause(error_type, error_info),
        "matched_chunk": None,
        "confidence": 0.0,
    }


# 解决方案提取关键词
_SOLUTION_KEYWORDS = ["解决", "fix", "修复", "建议", "如何", "处理", "避免", "改用", "使用", "try", "instead"]


def _extract_solution(content: str) -> str:
    """从知识库分块内容中提取解决方案段落。"""
    if not content:
        return ""

    sentences = re.split(r'[。\n！？!?]', content)
    solution_parts = []

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        sent_lower = sent.lower()
        if any(kw in sent_lower for kw in _SOLUTION_KEYWORDS):
            solution_parts.append(sent)

    if solution_parts:
        return "；".join(solution_parts)
    return ""


def _verify_code_error(input_data: dict, config: dict) -> dict:
    """
    代码错误验证模式核心实现。

    参数：
        input_data:
            - error_info: str   报错信息（必填）
            - code: str         可选，出错的代码片段
            - scenario: str     Mock 模式下选择场景
        config:
            - use_mock: bool    是否使用 Mock

    返回：
        {
            "status": "success" | "error",
            "data": {
                "mode": "code_error_verify",
                "error_info": str,
                "error_type": str,
                "is_known_error": bool,
                "known_solution": str,
                "confidence": float,
                "matched_chunk": dict or None,
                "suggested_code": str
            },
            "message": str,
            "mock_used": bool
        }
    """
    use_mock = config.get("use_mock", True)
    error_info = (input_data or {}).get("error_info", "")
    code = (input_data or {}).get("code", "")

    if not error_info.strip():
        return {
            "status": "error",
            "data": None,
            "message": "代码错误验证模式需要 error_info 字段",
            "mock_used": use_mock,
        }

    try:
        if use_mock:
            return _mock_verify_code_error(error_info, code, input_data)

        result = _match_error_pattern(error_info, code)

        return {
            "status": "success",
            "data": {
                "mode": "code_error_verify",
                "error_info": error_info[:300],
                "error_type": result["error_type"],
                "is_known_error": result["is_known_error"],
                "known_solution": result["known_solution"],
                "severity": result.get("severity", "major"),
                "root_cause": result.get("root_cause", "code_unknown"),
                "confidence": result["confidence"],
                "matched_chunk": result["matched_chunk"],
                "suggested_code": "",
            },
            "message": (
                f"匹配到已知错误 {result['error_type']}，置信度 {result['confidence']:.2f}"
                if result["is_known_error"]
                else f"未在知识库中找到匹配的错误模式（类型：{result['error_type']}）"
            ),
            "mock_used": False,
        }
    except Exception as e:
        logger.exception("_verify_code_error 执行失败")
        return {
            "status": "error",
            "data": None,
            "message": f"代码错误验证执行异常: {e}",
            "mock_used": use_mock,
        }


def _mock_verify_code_error(error_info: str, code: str, input_data: dict) -> dict:
    """Mock 模式：按错误类型或 scenario 返回预设的错误匹配结果。"""
    mock_data = _load_mock()
    scenario = (input_data or {}).get("scenario", "")

    if not scenario:
        error_type = _extract_error_type(error_info)
        type_to_scenario = {
            "TypeError": "type_error",
            "ValueError": "value_error",
            "KeyError": "key_error",
            "IndexError": "key_error",
            "AttributeError": "attribute_error",
            "SyntaxError": "syntax_error",
            "IndentationError": "syntax_error",
            "TabError": "syntax_error",
            "ImportError": "import_error",
            "ModuleNotFoundError": "import_error",
            "ZeroDivisionError": "zero_division",
            "PyTorchError": "pytorch_error",
            "CUDAError": "cuda_error",
            "TorchError": "pytorch_error",
            "RuntimeError": "pytorch_error",
        }
        scenario = type_to_scenario.get(error_type, "unknown_error")

    error_scenarios = mock_data.get("code_error_scenarios", {})
    if scenario not in error_scenarios:
        scenario = "unknown_error"

    data = dict(error_scenarios.get(scenario, error_scenarios.get("unknown_error", {})))
    data["mode"] = "code_error_verify"
    data["error_info"] = error_info[:300]

    error_type = data.get("error_type", _extract_error_type(error_info))
    data["severity"] = classify_error_severity(error_type)
    data["root_cause"] = classify_error_root_cause(error_type, error_info)

    return {
        "status": "success",
        "data": data,
        "message": (
            f"[Mock] 匹配到已知错误，置信度 {data.get('confidence', 0):.2f}"
            if data.get("is_known_error", False)
            else f"[Mock] 未找到匹配的错误模式"
        ),
        "mock_used": True,
    }


# ------------------------------------------------------------------
# Mock 模式
# ------------------------------------------------------------------
def _mock_verify(statement: str, input_data: dict) -> dict:
    """从 mock JSON 读取预设场景，按 scenario 或陈述内容自动选择。"""
    scenario = _pick_scenario(statement, input_data)
    scenarios = _load_mock().get("scenarios", {})

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
            full_data = json.load(f)
        scenarios = full_data.get("scenarios", {})
        code_error_scenarios = full_data.get("code_error_scenarios", {})
        return {
            "scenarios": scenarios,
            "code_error_scenarios": code_error_scenarios,
        }
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
        # 真实模式：查 knowledge_chunks 表做语义近似检索（关键词重叠归一化相似度）。
        # 与 rule_check_tool / 学情诊断与阶段规划 Agent 各工具一致，默认走真实检索而非 Mock，
        # 否则 专业知识与教学质量审核 Agent 的“交叉验证/幻觉检测”名存实亡（见《个性化学习资源生成 专业知识与教学质量审核 Agent学习包三问题检查报告》问题①）。
        self._default_config = {"use_mock": False}

    def retrieve_evidence(self, content: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return candidate KB chunks for semantic review.

        Retrieval is deliberately separate from ``verify_knowledge``: interactive
        answers may explain a concept in their own words, so lexical overlap is
        useful for finding evidence but must not itself become a hallucination
        verdict.
        """
        try:
            config = {**self._default_config, "top_k": max(1, int(top_k))}
            result = knowledge_verify({"statement": content}, config=config)
            if result.get("status") != "success":
                return []
            return list((result.get("data") or {}).get("matched_chunks") or [])
        except Exception:
            return []

    async def verify_knowledge(
        self,
        content: str,
        *,
        strict: bool = True,
    ) -> List[Dict[str, Any]]:
        """Verify content against the KB.

        ``strict=True`` is the learning-package policy.  Interactive answers
        use ``strict=False`` because a whole answer is not an atomic claim and
        its lexical score is diluted by explanations and examples.
        """
        result = knowledge_verify({"statement": content}, config=self._default_config)
        if result.get("status") != "success":
            return []
        data = result["data"]
        if data.get("verification_skipped"):
            return []
        confidence = data.get("confidence", 0.0)
        matched = data.get("matched_chunks", [])
        top = matched[0] if matched else None
        evidence = _build_chunk_evidence(confidence, top)
        # 机器可读溯源字段：命中来源块 ID + 原文行号（缺列时为 None，不影响流程）
        content_chunk_id = top.get("chunk_id") if top else None
        in_chunk_position = _format_in_chunk_position(top) if top else None

        if confidence < THRESHOLD_CRITICAL:
            # 库外/超纲：相似度极低且无有效命中 → 知识库未覆盖该主题（而非表述错误的幻觉）。
            # 标记为 out_of_scope（非 critical、must_fix=False），上层据此直接放行 个性化学习资源生成 Agent 内容，
            # 不再逼迫反复改写。区分依据：几乎零命中（top 相似度 < THRESHOLD_OUT_OF_SCOPE）。
            top_similarity = float(top.get("similarity", 0.0)) if top else 0.0
            if confidence < THRESHOLD_OUT_OF_SCOPE and top_similarity < THRESHOLD_OUT_OF_SCOPE:
                return [{
                    "error_type": "out_of_scope",
                    "severity": "info",
                    "must_fix": False,
                    "location": "技术陈述",
                    "description": (
                        f"知识库相似度 {confidence:.2f} 远低于阈值且无有效命中，"
                        f"该主题不在当前知识库覆盖范围内。"
                    ),
                    "evidence": evidence,
                    "suggestion": "该问题超出知识库覆盖范围，直接输出 个性化学习资源生成 Agent 内容，不做幻觉改写。",
                    "type": "out_of_scope",
                    "message": f"知识库相似度 {confidence:.2f}，主题超出覆盖范围。",
                    "confidence": confidence,
                    "content_chunk_id": content_chunk_id,
                    "in_chunk_position": in_chunk_position,
                }]
            if not strict:
                return [{
                    "error_type": "review_needed",
                    "severity": "major",
                    "location": "技术陈述",
                    "description": (
                        f"知识库词面相似度 {confidence:.2f} 偏低；整段问答需要复核，"
                        "该分数不等同于回答错误。"
                    ),
                    "evidence": evidence,
                    "suggestion": "仅核对关键技术结论；不要因整段回答的词面相似度直接判定幻觉。",
                    "type": "review_needed",
                    "message": f"问答相似度 {confidence:.2f}，非阻断复核提示。",
                    "confidence": confidence,
                    "content_chunk_id": content_chunk_id,
                    "in_chunk_position": in_chunk_position,
                }]
            return [{
                "error_type": "hallucination",
                "severity": "critical",
                "location": "技术陈述",
                "description": f"知识库相似度 {confidence:.2f} < {THRESHOLD_CRITICAL} 阈值，存在潜在幻觉风险。",
                "evidence": evidence,
                "suggestion": "删除无法验证的技术声明或补充可靠来源。",
                "type": "hallucination",
                "message": f"知识库相似度 {confidence:.2f}，存在潜在幻觉风险。",
                "confidence": confidence,
                "content_chunk_id": content_chunk_id,
                "in_chunk_position": in_chunk_position,
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
                "confidence": confidence,
                "content_chunk_id": content_chunk_id,
                "in_chunk_position": in_chunk_position,
            }]
        return []

    async def verify_against_chunk(self, statement: str, chunk_id: str) -> List[Dict[str, Any]]:
        """对单个已知来源块（chunk_id）核验某陈述（块级审核精确绑定来源）。

        - 回查该 chunk 原文，用同一相似度打分（recall+density）与阈值分类；
        - 块缺失 / 库不可用 / chunk 不存在时返回 []（**不产生假幻觉**，延续空库降级原则）；
        - 命中时 evidence/字段绑定到该确切来源块。
        """
        if not statement or not statement.strip() or not chunk_id:
            return []
        chunk = _load_chunk_by_id(chunk_id)
        if not chunk:
            # 未知来源块或库不可用：无可比对语料，不臆造幻觉。
            return []
        stmt_terms = _tokenize(statement)
        chunk_terms = _tokenize(chunk.get("content") or "")
        if not stmt_terms or not chunk_terms:
            return []
        confidence = round(_similarity(stmt_terms, chunk_terms), 4)
        top = {
            "chunk_id": chunk.get("chunk_id"),
            "similarity": confidence,
            "heading_path": chunk.get("heading_path"),
            "source_line_start": chunk.get("source_line_start"),
            "source_line_end": chunk.get("source_line_end"),
            "chunk_index": chunk.get("chunk_index"),
        }
        evidence = _build_chunk_evidence(confidence, top)
        in_chunk_position = _format_in_chunk_position(top)
        if confidence < THRESHOLD_CRITICAL:
            return [{
                "error_type": "hallucination",
                "severity": "critical",
                "location": "技术陈述",
                "description": f"与来源块 {chunk_id} 相似度 {confidence:.2f} < {THRESHOLD_CRITICAL}，该陈述在来源块中缺乏依据。",
                "evidence": evidence,
                "suggestion": "修正为来源块支持的表述或补充可靠来源。",
                "type": "hallucination",
                "message": f"来源块相似度 {confidence:.2f}，存在潜在幻觉风险。",
                "confidence": confidence,
                "content_chunk_id": chunk.get("chunk_id"),
                "in_chunk_position": in_chunk_position,
            }]
        if confidence < THRESHOLD_PASS:
            return [{
                "error_type": "review_needed",
                "severity": "major",
                "location": "技术陈述",
                "description": f"与来源块 {chunk_id} 相似度 {confidence:.2f} 处于存疑区间，需人工复核。",
                "evidence": evidence,
                "suggestion": "补充来源块依据或人工复核后放行。",
                "type": "review_needed",
                "message": f"来源块相似度 {confidence:.2f}，存疑需复核。",
                "confidence": confidence,
                "content_chunk_id": chunk.get("chunk_id"),
                "in_chunk_position": in_chunk_position,
            }]
        return []

    async def verify_code_error(
        self,
        error_info: str,
        code: str = "",
        scenario: str = "",
    ) -> Dict[str, Any]:
        """
        验证代码错误是否与知识库中已知错误模式匹配。

        参数：
            error_info: str      code_executor_tool 返回的报错信息
            code: str            可选，出错的代码片段
            scenario: str        可选，Mock 场景选择

        返回：
            {
                "is_known_error": bool,
                "known_solution": str,
                "error_type": str,
                "confidence": float,
                "matched_chunk": dict or None,
                "suggested_code": str,
                "mock_used": bool
            }
        """
        input_data: Dict[str, Any] = {"error_info": error_info}
        if code:
            input_data["code"] = code
        if scenario:
            input_data["scenario"] = scenario

        result = knowledge_verify(input_data, config=self._default_config)

        if result.get("status") != "success":
            return {
                "is_known_error": False,
                "known_solution": "",
                "error_type": "",
                "severity": "major",
                "root_cause": "code_unknown",
                "confidence": 0.0,
                "matched_chunk": None,
                "suggested_code": "",
                "mock_used": result.get("mock_used", True),
            }

        data = result.get("data", {})
        return {
            "is_known_error": data.get("is_known_error", False),
            "known_solution": data.get("known_solution", ""),
            "error_type": data.get("error_type", ""),
            "severity": data.get("severity", "major"),
            "root_cause": data.get("root_cause", "code_unknown"),
            "confidence": data.get("confidence", 0.0),
            "matched_chunk": data.get("matched_chunk"),
            "suggested_code": data.get("suggested_code", ""),
            "mock_used": result.get("mock_used", True),
        }
