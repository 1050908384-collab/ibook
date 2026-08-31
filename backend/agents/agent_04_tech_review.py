# -*- coding: utf-8 -*-
import hashlib
import re
from typing import Dict, Any, List, Optional
from agents.base_agent import BaseAgent
from models.review import ReviewError, TechnicalReviewResult
from tools.agent4_tools.code_executor_tool import CodeExecutorTool
from tools.agent4_tools.code_parser_tool import CodeParserTool
from tools.agent4_tools.knowledge_verify_tool import (
    KnowledgeVerifyTool,
    classify_error_severity,
    classify_error_root_cause,
)
from tools.agent4_tools.rule_check_tool import RuleCheckTool
from tools.agent4_tools.difficulty_analyzer_tool import DifficultyAnalyzerTool
from tools.agent4_tools.cognitive_load_tool import CognitiveLoadTool
from tools.agent4_tools.style_match_tool import StyleMatchTool
from tools.learning_resource_tool import LearningResourceTool

_ERROR_CLASSIFICATION_MAP: Dict[str, Dict[str, str]] = {
    "SyntaxError": {"category": "SyntaxError", "severity": "critical"},
    "IndentationError": {"category": "SyntaxError", "severity": "critical"},
    "TabError": {"category": "SyntaxError", "severity": "critical"},
    "RuntimeError": {"category": "RuntimeError", "severity": "major"},
    "TypeError": {"category": "RuntimeError", "severity": "major"},
    "ValueError": {"category": "RuntimeError", "severity": "major"},
    "KeyError": {"category": "RuntimeError", "severity": "major"},
    "IndexError": {"category": "RuntimeError", "severity": "major"},
    "AttributeError": {"category": "RuntimeError", "severity": "major"},
    "NameError": {"category": "RuntimeError", "severity": "major"},
    "UnboundLocalError": {"category": "RuntimeError", "severity": "major"},
    "ZeroDivisionError": {"category": "RuntimeError", "severity": "major"},
    "MemoryError": {"category": "RuntimeError", "severity": "major"},
    "RecursionError": {"category": "RuntimeError", "severity": "major"},
    "OverflowError": {"category": "RuntimeError", "severity": "major"},
    "ImportError": {"category": "RuntimeError", "severity": "major"},
    "ModuleNotFoundError": {"category": "RuntimeError", "severity": "major"},
    "FileNotFoundError": {"category": "RuntimeError", "severity": "major"},
    "PermissionError": {"category": "RuntimeError", "severity": "major"},
    "IsADirectoryError": {"category": "RuntimeError", "severity": "major"},
    "NotADirectoryError": {"category": "RuntimeError", "severity": "major"},
    "ConnectionError": {"category": "RuntimeError", "severity": "major"},
    "TimeoutError": {"category": "RuntimeError", "severity": "major"},
    "OSError": {"category": "RuntimeError", "severity": "major"},
    "IOError": {"category": "RuntimeError", "severity": "major"},
    "CUDAError": {"category": "RuntimeError", "severity": "major"},
    "TorchError": {"category": "RuntimeError", "severity": "major"},
    "PyTorchError": {"category": "RuntimeError", "severity": "major"},
    "TensorFlowError": {"category": "RuntimeError", "severity": "major"},
    "JSONDecodeError": {"category": "RuntimeError", "severity": "major"},
    "JSONEncodeError": {"category": "RuntimeError", "severity": "major"},
    "HTTPError": {"category": "RuntimeError", "severity": "major"},
    "RequestException": {"category": "RuntimeError", "severity": "major"},
    "URLError": {"category": "RuntimeError", "severity": "major"},
    "AssertionError": {"category": "LogicError", "severity": "minor"},
    "NotImplementedError": {"category": "LogicError", "severity": "minor"},
    "StopIteration": {"category": "LogicError", "severity": "minor"},
    "GeneratorExit": {"category": "LogicError", "severity": "minor"},
    "Exception": {"category": "LogicError", "severity": "minor"},
}

_ERROR_PATTERN_MAP: List[tuple] = [
    (r'SyntaxError|IndentationError|TabError', "SyntaxError", "critical"),
    (r'ModuleNotFoundError|ImportError|FileNotFoundError|PermissionError|IsADirectoryError|NotADirectoryError', "RuntimeError", "major"),
    (r'ConnectionError|TimeoutError|OSError|IOError', "RuntimeError", "major"),
    (r'CUDAError|TorchError|PyTorchError|TensorFlowError', "RuntimeError", "major"),
    (r'JSONDecodeError|JSONEncodeError', "RuntimeError", "major"),
    (r'HTTPError|RequestException|URLError', "RuntimeError", "major"),
    (r'KeyError|IndexError|TypeError|ValueError|AttributeError|NameError|UnboundLocalError|ZeroDivisionError|MemoryError|RecursionError|OverflowError', "RuntimeError", "major"),
    (r'RuntimeError', "RuntimeError", "major"),
    (r'AssertionError|NotImplementedError|StopIteration|GeneratorExit', "LogicError", "minor"),
    (r'Exception', "LogicError", "minor"),
]

_ENVIRONMENT_ERROR_TYPES = {
    "ModuleNotFoundError", "ImportError", "FileNotFoundError",
    "PermissionError", "ConnectionError", "TimeoutError",
    "CUDAError", "TorchError", "PyTorchError", "MemoryError",
    "IsADirectoryError", "NotADirectoryError",
}

_DEPENDENCY_KEYWORDS = ["deprecated", "removed", "not found in module", "version", "incompatible", "API"]
_DATA_KEYWORDS = ["cannot parse", "invalid format", "unexpected value", "parse error", "malformed"]


def _classify_error_by_patterns(error_info: str) -> Dict[str, str]:
    if not error_info:
        return {"category": "unknown", "severity": "major"}
    text_lower = error_info.lower()
    for pattern, category, severity in _ERROR_PATTERN_MAP:
        if re.search(pattern, error_info, re.IGNORECASE):
            return {"category": category, "severity": severity}
    return {"category": "unknown", "severity": "major"}


def _determine_root_cause(error_type: str, error_info: str) -> str:
    if error_type in _ENVIRONMENT_ERROR_TYPES:
        return "environment_issue"
    info_lower = error_info.lower()
    if any(kw in info_lower for kw in _DEPENDENCY_KEYWORDS):
        return "dependency_issue"
    if any(kw in info_lower for kw in _DATA_KEYWORDS):
        return "data_issue"
    if error_type and error_type not in _ENVIRONMENT_ERROR_TYPES:
        return "code_issue"
    if error_type == "unknown":
        if any(kw in info_lower for kw in ("install", "pip ", "package", "module", "import", "环境", "依赖")):
            return "environment_issue"
        if any(kw in info_lower for kw in ("parse", "format", "invalid", "json", "数据")):
            return "data_issue"
    return "code_unknown"


# ============================================================================
# 审核辩论协议 — Error ID 生成与归一化（协议 2.4 节，规则必须钉死）
# ============================================================================

# normalized_description：去中英文标点。覆盖 ASCII 标点 + 常见中文标点。
_PUNCTUATION_RE = re.compile(
    r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~"
    r"。，、；：？！“”‘’（）《》【】…—～·「」『』〈〉﹏]"
)
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_location(location: Optional[str]) -> str:
    """协议 2.4：normalized_location = 小写 + 去首尾空格。"""
    return (location or "").strip().lower()


def _normalize_description(description: Optional[str]) -> str:
    """协议 2.4：normalized_description = 小写 + 去中英文标点 + 折叠连续空白。"""
    text = (description or "").lower()
    text = _PUNCTUATION_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def compute_canonical_error_id(
    claim_id: Optional[str],
    error_type: Optional[str],
    location: Optional[str],
    description: Optional[str],
) -> str:
    """canonical_error_id = sha1(claim_id + error_type + normalized_location + normalized_description)[:12]

    跨轮稳定 ID，同一问题跨轮保留同一个值；在单个 trace 语义下唯一。
    专业知识与教学质量审核 Agent 不自行发明跨轮逻辑 ID（协议 2.4）。
    """
    seed = "".join([
        (claim_id or ""),
        (error_type or ""),
        _normalize_location(location),
        _normalize_description(description),
    ])
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]


def compute_error_id(canonical_error_id: str, round_index: int) -> str:
    """error_id = canonical_error_id + "_r" + round_index（协议 2.4）。"""
    return f"{canonical_error_id}_r{round_index}"


# ============================================================================
# 审核辩论协议 — source_refs / change_request / must_fix 派生（协议第 6 节）
# ============================================================================

# error_type 命中以下语义（子串匹配，忽略大小写）自动进入 must_fix
_MUST_FIX_ERROR_TYPE_KEYWORDS = (
    # 事实错误 / 幻觉
    "hallucination", "fact", "factual", "事实", "幻觉",
    # 来源缺失
    "source_missing", "missing_source", "no_source", "来源缺失", "缺少来源",
    # 代码不可执行
    "syntaxerror", "runtimeerror", "code_error", "不可执行", "无法运行", "无法复现",
    # 违反安全 / 规范
    "security", "compliance", "violation", "安全", "规范", "违反", "deploy",
    # ★ Agent4 教学层元数据硬性缺失（PersonalizationCheckTool → rule_errors 通道）
    # 实操缺动手任务、个性化 action/reason/source 声明缺失、补救缺证据/任务
    # 全部都是"块外壳未按协议构建"，属于阻断发布的形式红线，必须修复。
    "personalization_action_missing", "personalization_reason_missing",
    "teaching_source_missing", "remediation_evidence_missing",
    "remediation_task_missing", "practice_task_missing",
    # ★ Agent4 grade/skill_tags 形式红线（★★ 新规则）
    "grade_terminology_density_mismatch", "skill_tags_mastered_intro_filler",
    # ★ Agent4 块设计红线（★★ _verify_paragraph_order）：段落顺序被改动
    "paragraph_order_violation",
    # ★ Agent4 style_mismatch(major/critical) 结构性不匹配（learner_model.skill.md 规则3）。
    # 注：StyleMatchTool 仅在 match<0.5 时产出 style_mismatch(severity=major)，无 minor 版本，
    # 因此此处加入关键字白名单即等价于"凡 style_mismatch 均 must_fix=True"，不导致过度阻断。
    "style_mismatch",
)


def downgrade_severity(severity: str) -> str:
    """降一级：critical→major→minor（minor 已最低则保持）。供 downgrade 裁决落库时参考。"""
    order = {"critical": "major", "major": "minor", "minor": "minor"}
    return order.get(str(severity).lower(), "minor")


def _matches_must_fix_error_type(error_type: Optional[str]) -> bool:
    et = (error_type or "").lower()
    return any(kw in et for kw in _MUST_FIX_ERROR_TYPE_KEYWORDS)


# 「质量红线」类错误：其成因是物理度量（认知负荷/难度超载），只能靠改写正文
# （patched_sections）真正缓解——补一条来源引用（source_refs）并不能降低概念密度或
# 难度不匹配。rebut 复核时这类问题的「部分整改」必须以正文 patch 为证据，不能仅凭
# 来源引用就降级（否则关键红线会被无实质整改的「部分接受」放松）。
_QUALITY_REDLINE_ERROR_TYPE_KEYWORDS = (
    "cognitive_load", "认知负荷", "负荷超载", "overload",
    "difficulty_mismatch", "难度不匹配", "难度", "too_hard", "too_easy",
    "concept_density", "概念密度",
)


def _is_quality_redline_error_type(error_type: Optional[str]) -> bool:
    """判断 error_type 是否属于「只能靠改写正文缓解」的质量红线类问题。"""
    et = (error_type or "").lower()
    return any(kw in et for kw in _QUALITY_REDLINE_ERROR_TYPE_KEYWORDS)


def _build_source_index(sources: Optional[List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """把 sources 列表转为 {source_id: source} 索引，供命中匹配。"""
    index: Dict[str, Dict[str, Any]] = {}
    for s in sources or []:
        if isinstance(s, dict):
            sid = s.get("source_id")
            if sid:
                index[str(sid)] = s
    return index


def _select_key_claims(claims: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """筛出「有效关键 Claim」：is_key_technical=true、text 非空、且未被丢弃（status!=dropped）。

    Claim 回退修正（任务 2）的判定依据：存在有效关键 Claim 时只核验 Claim，
    不再回退整篇；无有效关键 Claim 时才回退整篇。
    """
    result: List[Dict[str, Any]] = []
    for c in claims or []:
        if not isinstance(c, dict):
            continue
        if not c.get("is_key_technical"):
            continue
        if c.get("status") == "dropped":
            continue
        if not (c.get("text") or "").strip():
            continue
        result.append(c)
    return result


def _resolve_source_refs(
    raw: Dict[str, Any],
    source_index: Dict[str, Dict[str, Any]],
) -> List[str]:
    """强制为每个 error 解析 source_refs：命中 sources 里的某条则返回其 source_id 列表；
    未命中则返回空列表（空列表本身即"未命中来源"，是 must_fix 的触发条件之一）。
    """
    refs: List[str] = []
    # 1. 工具已显式提供 source_refs
    candidates = raw.get("source_refs")
    if isinstance(candidates, str):
        candidates = [candidates]
    if isinstance(candidates, list):
        for c in candidates:
            if c and str(c) in source_index and str(c) not in refs:
                refs.append(str(c))
    # 2. 单个 source_ref 字段
    single = raw.get("source_ref")
    if single and str(single) in source_index and str(single) not in refs:
        refs.append(str(single))
    return refs


def _extract_change_request_required(raw: Dict[str, Any]) -> bool:
    """读取工具原始 change_request.required（若已显式提供），供 must_fix 派生使用。"""
    existing = raw.get("change_request")
    if isinstance(existing, dict) and existing.get("required") is True:
        return True
    return False


def _build_change_request(raw: Dict[str, Any], must_fix: bool, source_refs: List[str]) -> Dict[str, Any]:
    """强制为每个 error 生成 change_request：{"required": bool, "action": str}。

    - 若工具已显式给出完整 change_request，则沿用（仅做类型收敛）
    - 否则依据 must_fix 与是否命中来源推导默认 action
    - must_fix=true 时 required 必为 true（明确告诉 个性化学习资源生成 Agent 要改什么）
    """
    existing = raw.get("change_request")
    if isinstance(existing, dict) and "required" in existing and "action" in existing:
        return {
            "required": bool(existing["required"]) or must_fix,
            "action": str(existing["action"]),
        }
    if must_fix:
        action = "rewrite_with_supported_source" if not source_refs else "rewrite"
    else:
        action = "review_suggestion"
    return {"required": bool(must_fix), "action": action}


def _extract_source_supports_claim(raw: Dict[str, Any]) -> bool:
    """读取工具原始信号：来源是否支持该 Claim（协议第 6 节规则3）。

    默认 True（假定来源支持）；仅当工具显式给出以下任一信号时判定为「不支持」：
      - raw["source_supports_claim"] is False
      - raw["source_supported"] is False
      - error_type 属于「来源不支持/来源不相关/来源矛盾」一类
    """
    for key in ("source_supports_claim", "source_supported"):
        if raw.get(key) is False:
            return False
    et = str(raw.get("error_type") or raw.get("type") or "").lower()
    unsupported_markers = (
        "source_mismatch", "source_unsupported", "unsupported_source",
        "source_irrelevant", "source_conflict", "来源不支持", "来源不相关", "来源矛盾",
    )
    if any(m in et for m in unsupported_markers):
        return False
    return True


def derive_must_fix(
    error_type: Optional[str],
    severity: str,
    source_refs: List[str],
    is_key_technical: bool,
    change_request_required: bool,
    source_supports_claim: bool = True,
) -> tuple[bool, Optional[str]]:
    """must_fix 派生规则（协议第 6 节 / 专业知识与教学质量审核 Agent任务说明 第 5 点）。

    满足任一条件即自动进入 must_fix：
      - severity = critical
      - error_type 属于事实错误/来源缺失/代码不可执行/违反安全或规范
      - 关键技术陈述缺少来源（source_refs 为空且 is_key_technical=true）
      - 关键技术陈述的来源存在但不支持该 Claim（协议第 6 节规则3）
      - change_request.required = true
    返回 (must_fix, must_fix_reason)。
    """
    if severity == "critical":
        return True, "severity=critical，风险等级最高，强制修复"
    if _matches_must_fix_error_type(error_type):
        return True, f"error_type={error_type} 属于事实错误/来源缺失/代码不可执行/违反安全规范"
    if not source_refs and is_key_technical:
        return True, "关键技术陈述（is_key_technical=true）缺少有效来源支撑"
    # 协议第 6 节规则(3)：来源存在（source_refs 非空）但不支持该关键技术 Claim
    if source_refs and is_key_technical and not source_supports_claim:
        return True, "关键技术陈述的来源存在但不支持该 Claim（协议第 6 节规则3）"
    if change_request_required:
        return True, "change_request.required=true，明确要求 个性化学习资源生成 Agent 修改"
    return False, None


# ---------------------------------------------------------------------------
# 主张级缓存 + 按类型路由（辩论方案 第 2、4 点）
# ---------------------------------------------------------------------------
# 缓存策略（用户确认）：仅沉淀两个「确定档」——pass / hallucination——的裁决结果，
# 供跨会话复用（同一个幻觉不辩论第二次）。review_needed / 灰区 需 LLM 仲裁的档
# 不入缓存，避免把不稳定的中间裁决固化。
_CACHEABLE_LEVELS = {"pass", "hallucination"}

# 路由类型（用户确认）：按 claim_type + 溯源标记决定核验策略。
#   kb_fact      → 知识库事实：有 source_chunk_id / source_ref，走 KB 相似度核验
#   profile_fact → 用户画像事实：引用学习者自身设备/配置等，无 KB chunk，
#                  豁免全库幻觉判定，只保留「声称有来源却缺来源」的形式校验
#   personalized → 个性化推理讲解：无 KB 来源、非关键技术，豁免全库检索
_ROUTE_KB_FACT = "kb_fact"
_ROUTE_PROFILE_FACT = "profile_fact"
_ROUTE_PERSONALIZED = "personalized"

# 用户画像/自身配置类主张的判别词（无 KB chunk 时用于区分「个人事实」与「客观技术事实」）
_PROFILE_HINT_TERMS = (
    "我的", "你的", "您的", "本机", "本地", "当前设备", "我们的设备",
    "我的显卡", "我的显存", "我的设备", "我的配置", "你的显卡", "你的设备",
    "your gpu", "your device", "my gpu", "my device", "my setup", "local machine",
)


def _claim_fingerprint(text: str, chunk_id: str = "") -> str:
    """主张文本指纹：normalize(text) + chunk_id 的 sha1[:16]。

    归一化去除空白/大小写差异，使「同一主张的等价改写」尽量落到同一指纹。
    chunk_id 参与指纹：换了来源块即视为新主张，天然 miss（与 verified_claims 主键一致）。
    """
    normalized = re.sub(r"\s+", "", (text or "").strip().lower())
    raw = f"{normalized}||{chunk_id or ''}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _classify_claim_route(claim: Dict[str, Any], source_chunk_id: Optional[str]) -> str:
    """按 claim_type + 溯源标记判定核验路由（见上方常量说明）。

    - 有明确 KB 来源（source_chunk_id 或 source_ref）→ kb_fact
    - 无 KB 来源但文本引用「我的/本机/设备配置」等 → profile_fact
    - 其余无来源、非关键技术的陈述 → personalized
    - 无来源但被标为关键技术的客观事实 → 仍按 kb_fact 走全库检索（缺来源会被判缺依据）
    """
    if source_chunk_id or claim.get("source_ref"):
        return _ROUTE_KB_FACT
    text = (claim.get("text") or "").lower()
    if any(term.lower() in text for term in _PROFILE_HINT_TERMS):
        return _ROUTE_PROFILE_FACT
    if claim.get("is_key_technical"):
        # 关键技术客观事实必须有依据：无来源时仍送全库检索，交由 must_fix 规则处理缺来源。
        return _ROUTE_KB_FACT
    return _ROUTE_PERSONALIZED


class TechnicalReviewAgent(BaseAgent):
    def __init__(self):
        super().__init__("technical_review_agent", [
            "technical_review", "industry_standards", "code_error_patterns",
            "pedagogy", "learner_model", "quality_metrics"
        ])
        self.code_executor_tool = CodeExecutorTool()
        self.code_parser_tool = CodeParserTool()
        self.knowledge_verify_tool = KnowledgeVerifyTool()
        self.rule_check_tool = RuleCheckTool()
        self.difficulty_analyzer_tool = DifficultyAnalyzerTool()
        self.cognitive_load_tool = CognitiveLoadTool()
        self.style_match_tool = StyleMatchTool()
        self.learning_resource_tool = LearningResourceTool()
        self.load_tools([
            self.code_executor_tool,
            self.code_parser_tool,
            self.knowledge_verify_tool,
            self.rule_check_tool,
            self.difficulty_analyzer_tool,
            self.cognitive_load_tool,
            self.style_match_tool,
        ])
        # 主张级缓存客户端：懒加载 SQLiteClient，失败时降级为 None（缓存不可用不阻断核验）。
        self._db_client = None

    @property
    def db_client(self):
        """懒加载 SQLiteClient，供主张级缓存读写；任何创建失败都不抛出，返回 None 走无缓存路径。"""
        if self._db_client is None:
            try:
                from db.sqlite_client import SQLiteClient
                self._db_client = SQLiteClient()
            except Exception:
                self._db_client = None
        return self._db_client

    def _cache_lookup_claim(self, claim_fp: str, chunk_id: str) -> Optional[Dict[str, Any]]:
        """查主张级缓存；命中返回行 dict，miss / 缓存不可用返回 None（不抛出）。"""
        client = self.db_client
        if client is None or not claim_fp:
            return None
        try:
            return client.get_verified_claim(claim_fp, chunk_id or "")
        except Exception:
            return None

    def _cache_store_claim(self, claim_fp: str, chunk_id: str, level: str, data: Dict[str, Any]) -> None:
        """把确定档（pass/hallucination）裁决沉淀到主张级缓存；非确定档或缓存不可用则跳过。"""
        if level not in _CACHEABLE_LEVELS:
            return
        client = self.db_client
        if client is None or not claim_fp:
            return
        try:
            client.save_verified_claim(claim_fp, chunk_id or "", {
                "level": level,
                "confidence": data.get("confidence", 0.0),
                "verify_method": data.get("verify_method", "lexical"),
                "claim_text": data.get("claim_text"),
                "evidence": data.get("evidence"),
            })
        except Exception:
            pass

    def _profile_to_dict(self, profile: Any) -> Dict[str, Any]:
        if hasattr(profile, "model_dump"):
            return profile.model_dump(mode="json")
        if hasattr(profile, "dict"):
            return profile.dict()
        if isinstance(profile, dict):
            return profile
        return {}

    @staticmethod
    def _extract_stage_scope(stage: Any) -> tuple:
        """从 stage（dict / 含同名属性的对象 / None）提取 (block_id, subtopic_id)，供视频归属校验。

        兼容 harness 传 StageInfo 对象与测试传 dict 两种形态；缺失时返回 (None, None)，
        learning_resource_tool 会据此安全降级（空 block_id -> 不查询/空基线）。
        """
        if stage is None:
            return None, None
        if isinstance(stage, dict):
            return stage.get("block_id"), stage.get("subtopic_id")
        return getattr(stage, "block_id", None), getattr(stage, "subtopic_id", None)

    async def review(
        self,
        session_id: str,
        content: Dict[str, Any],
        profile: Optional[Dict[str, Any]] = None,
        sources: Optional[List[Dict[str, Any]]] = None,
        claims: Optional[List[Dict[str, Any]]] = None,
        round_index: int = 1,
        recommended_videos: Optional[List[Dict[str, Any]]] = None,
        stage: Optional[Dict[str, Any]] = None,
    ) -> TechnicalReviewResult:
        """审核学习包 / 回答，产出带证据可追溯的 issues（审核辩论协议 专业知识与教学质量审核 Agent→个性化学习资源生成 Agent）。

        参数（协议 2.1 / 2.2）：
        - sources：学习包来源清单（个性化学习资源生成 Agent 产出），每条含 source_id 等字段
        - claims：个性化学习资源生成 Agent 的可验证陈述声明，每条含 claim_id / section_ref / is_key_technical 等
        - round_index：当前审核轮次（用于生成 error_id 后缀 _r{round_index}），跨轮稳定 canonical 不变
        - recommended_videos：个性化学习资源生成 Agent 从资源库挑选的推荐视频列表（仅学习包场景，
          对应 technical_review.skill.md 规则5「推荐视频资源审核」）
        - stage：当前阶段信息（含 block_id / subtopic_id），供视频归属校验；dict 或含同名属性的对象均可

        每个 error 强制携带：canonical_error_id、error_id、source_refs、change_request、
        must_fix、must_fix_reason、claim_id（协议 2.3 / 2.4 / 第 6 节）。
        视频错误（video_resource_mismatch/video_url_mismatch/video_content_unverified）
        由 learning_resource_tool.validate_recommended_videos 产出后一并补全协议字段。
        """
        raw_errors = []
        video_resource_errors: List[Dict[str, Any]] = []
        print(f"=== 专业知识与教学质量审核 Agent(合并后) TechnicalReviewAgent 开始审核 ===")

        profile_dict = self._profile_to_dict(profile) if profile else {}

        # 入口 guard：content 必须包含至少一个已知分支
        known_branches = ("tutorial", "code_examples", "answer")
        if not any(k in content for k in known_branches):
            print(f"=== 专业知识与教学质量审核 Agent(合并后) content 缺少已知分支（tutorial/code_examples/answer）===")
            guard_error = ReviewError(
                error_type="invalid_input",
                severity="critical",
                description=f"待审核内容缺少已知分支字段（需包含 {known_branches} 之一）",
                suggestion="请补充 tutorial / code_examples / answer 字段后重新审核",
            )
            guard_errors = self._enrich_debate_fields(
                [guard_error], sources=sources, claims=claims, round_index=round_index
            )
            return TechnicalReviewResult(
                passed=False,
                errors=guard_errors,
                review_details={"summary": "输入内容缺少可审核分支"},
            )

        if "tutorial" in content:
            print(f"=== 审核教程内容（前200字符）===\n{content['tutorial'][:200]}")
            # 步骤1：硬性检查（规则检查）
            try:
                rule_errors = await self.rule_check_tool.check_rules(content["tutorial"])
                raw_errors.extend(rule_errors)
            except Exception:
                pass
            # 步骤2：事实检查（幻觉检测）
            # Claim 回退修正（任务 2）：存在有效关键 Claim 时，只对 Claim 核验（结果可为空，
            # 不再回退整篇），避免整篇送检重新产生全文级误报；没有有效关键 Claim 时才回退整篇。
            try:
                key_claims = _select_key_claims(claims)
                if key_claims:
                    claim_hallucinations = await self._verify_claims_hallucination(key_claims)
                    raw_errors.extend(claim_hallucinations)
                else:
                    hallucination_errors = await self.knowledge_verify_tool.verify_knowledge(content["tutorial"])
                    raw_errors.extend(hallucination_errors)
            except Exception:
                pass
            # 步骤3-4：质量审核（难度/认知负荷/风格匹配）
            print(f"=== 质量审核：分析教程难度 ===")
            try:
                difficulty_errors = await self.difficulty_analyzer_tool.analyze(
                    content["tutorial"], profile=profile_dict
                )
                raw_errors.extend(difficulty_errors)
            except Exception:
                pass
            try:
                load_errors = await self.cognitive_load_tool.analyze(content["tutorial"])
                raw_errors.extend(load_errors)
            except Exception:
                pass
            try:
                style_errors = await self.style_match_tool.check(
                    content["tutorial"], profile=profile_dict
                )
                raw_errors.extend(style_errors)
            except Exception:
                pass

        if "code_examples" in content:
            print(f"=== 审核代码示例 ===")
            for code in content["code_examples"]:
                try:
                    syntax_errors = await self.code_parser_tool.parse_code(code.get("code", ""))
                    raw_errors.extend(syntax_errors)
                except Exception:
                    pass
                try:
                    code_rule_errors = await self.rule_check_tool.check_code_rules(code.get("code", ""))
                    raw_errors.extend(code_rule_errors)
                except Exception:
                    pass

        if "answer" in content:
            print(f"=== 审核回答内容（前200字符）===\n{content['answer'][:200]}")
            # Q&A does not use whole-answer lexical similarity as a verdict.
            # HybridTechnicalReviewAgent supplies candidate chunks to its
            # semantic LLM review; this protocol path keeps the same policy.
            print("=== 问答事实核验：跳过整段词面相似度硬判定 ===")
            # 步骤3-4：质量审核
            print(f"=== 质量审核：分析回答质量 ===")
            try:
                style_errors = await self.style_match_tool.check(
                    content["answer"], profile=profile_dict
                )
                raw_errors.extend(style_errors)
            except Exception:
                pass
            try:
                difficulty_errors = await self.difficulty_analyzer_tool.analyze(
                    content["answer"], profile=profile_dict
                )
                raw_errors.extend(difficulty_errors)
            except Exception:
                pass

        # 步骤5：推荐视频资源审核（仅学习包场景，technical_review.skill.md 规则5）
        # 视频归属越界/URL不一致/内容未审核，比对资源库权威记录 learning_resources。
        # answer 场景不涉及视频；video 分支不构成"已知分支"guard（tutorial/code_examples/answer 仍需存在）。
        if "answer" not in content and recommended_videos:
            block_id, subtopic_id = self._extract_stage_scope(stage)
            print(f"=== 审核推荐视频资源（block_id={block_id}, subtopic_id={subtopic_id}, "
                  f"共{len(recommended_videos)}个）===")
            try:
                video_resource_errors = await self.learning_resource_tool.validate_recommended_videos(
                    recommended_videos, block_id, subtopic_id
                )
                raw_errors.extend(video_resource_errors)
            except Exception:
                pass

        # 步骤6：综合裁决 — 将原始dict转为ReviewError，按规则4裁决优先级
        errors = self._normalize_errors(raw_errors)
        # 步骤7：审核辩论协议增强 — 为每个 error 强制补全
        #   source_refs / change_request / canonical_error_id / error_id / must_fix / claim_id
        #   raw_errors 已含视频错误（携带 source_ref=src_video_{id} 与 change_request）
        errors = self._enrich_debate_fields(
            errors, sources=sources, claims=claims, round_index=round_index, raw_errors=raw_errors
        )
        # 审核通过 = 未发现任何问题（len==0 时必然无 critical，无需重复判定 critical）
        passed = len(errors) == 0
        must_fix_count = sum(1 for e in errors if e.must_fix)
        critical_count = sum(1 for e in errors if e.severity == "critical")
        summary = (
            f"审核完成：发现{len(errors)}个问题"
            + (f"（其中{critical_count}个critical）" if critical_count else "")
            + (f"，must_fix={must_fix_count}个" if must_fix_count else "")
        )
        print(f"=== 专业知识与教学质量审核 Agent(合并后) TechnicalReviewAgent 审核完成 ===")
        print(f"=== 发现问题: {len(errors)}个, must_fix={must_fix_count}个, passed={passed} ===")
        if errors:
            for i, e in enumerate(errors, 1):
                print(
                    f"  {i}. [{e.severity}] {e.error_type} "
                    f"(cid={e.canonical_error_id}, must_fix={e.must_fix}, "
                    f"source_refs={e.source_refs or '未命中来源'}): {e.description}"
                )

        return TechnicalReviewResult(passed=passed, errors=errors, review_details={"summary": summary})

    def build_answer_review_summary(
        self,
        technical_result: TechnicalReviewResult,
        iterations: int = 1,
        max_iterations: Optional[int] = None,
    ) -> Dict[str, Any]:
        """产出答疑链路的审核摘要（专业知识与教学质量审核 Agent6_API接口任务说明.md 第 1 节）。

        专业知识与教学质量审核 Agent 在答疑链路中的职责仅是审核 `answer`，并返回审核摘要，字段固定为：
            technical_passed / quality_passed / iterations / status / needs_human_review

        说明（与合并后 专业知识与教学质量审核 Agent 事实一致）：
        - 质量审核已合并进 专业知识与教学质量审核 Agent，`quality_passed` 与 `technical_passed` 共享同一份
          审核结论（technical_result.passed），不再存在独立的质量审核体。
        - status 取值：
            "passed"              —— 审核通过（无遗留问题）
            "passed_with_warnings"—— 未达上限即结束但仍有非 must_fix 的 minor 问题
            "needs_human_review"  —— 达到最大审核轮次仍有未决 must_fix（协议第 7 节）
        - needs_human_review：仅当达到 max_iterations 且仍存在 must_fix 的问题时为 True，
          与学习包审核链路的降级判定口径一致（不阻塞用户，交由人工复核）。

        参数：
        - technical_result：专业知识与教学质量审核 Agent 最后一轮审核结果（含 errors / passed）
        - iterations：答疑审核实际迭代轮次
        - max_iterations：允许的最大审核轮次；缺省时从 config.settings 读取，
          读取失败兜底为 iterations（即视为「已到上限」，保守判定）
        """
        errors = technical_result.errors or []
        passed = bool(technical_result.passed)
        must_fix_count = sum(1 for e in errors if getattr(e, "must_fix", False))

        if max_iterations is None:
            try:
                from config import settings
                max_iterations = int(getattr(settings, "MAX_REVIEW_ITERATIONS", iterations))
            except Exception:
                max_iterations = iterations

        reached_limit = iterations >= max_iterations
        needs_human_review = (not passed) and reached_limit and must_fix_count > 0

        if passed:
            status = "passed"
        elif needs_human_review:
            status = "needs_human_review"
        elif must_fix_count == 0:
            # 仍有问题但无未决 must_fix（仅 minor）→ 允许降级放行，标记警告
            status = "passed_with_warnings"
        else:
            # 未通过、未到上限、仍有 must_fix：审核链路尚未收敛
            status = "in_progress"

        return {
            "technical_passed": passed,
            "quality_passed": passed,
            "iterations": int(iterations),
            "status": status,
            "needs_human_review": needs_human_review,
        }

    def _normalize_errors(self, raw_errors: List[Dict[str, Any]]) -> List[ReviewError]:
        """将工具返回的原始dict列表转为ReviewError对象列表。

        复用 deterministic.normalize_review_errors 公共实现，消除两处重复逻辑。
        采用延迟 import 避免顶层引入 deterministic 的重依赖（db/llm 等）。
        """
        from agents.deterministic import normalize_review_errors
        return normalize_review_errors(raw_errors)

    @staticmethod
    def _build_claim_index(
        claims: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Dict[str, Any]]:
        """把 claims 按 section_ref（归一化后）建索引，用于按 location 反查所属 claim。"""
        by_section: Dict[str, Dict[str, Any]] = {}
        for c in claims or []:
            if not isinstance(c, dict):
                continue
            section_ref = c.get("section_ref")
            if section_ref:
                by_section[_normalize_location(section_ref)] = c
        return by_section

    def _match_claim(
        self,
        error_location: Optional[str],
        raw_claim_id: Optional[str],
        claims: Optional[List[Dict[str, Any]]],
        claim_by_section: Dict[str, Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """为一个 error 关联 claim：优先用工具给出的 claim_id，其次按 location 命中 section_ref。"""
        # 1. 工具已显式给出 claim_id
        if raw_claim_id:
            for c in claims or []:
                if isinstance(c, dict) and str(c.get("claim_id")) == str(raw_claim_id):
                    return c
        # 2. 按 location 命中 section_ref（前缀匹配，兼容 tutorial:sec_2 与 tutorial:sec_2:p_1）
        loc = _normalize_location(error_location)
        if loc:
            if loc in claim_by_section:
                return claim_by_section[loc]
            for section_norm, claim in claim_by_section.items():
                if loc.startswith(section_norm) or section_norm.startswith(loc):
                    return claim
        return None

    async def _verify_claims_hallucination(
        self,
        claims: Optional[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """按 Claim 粒度做幻觉检测，使幻觉 issue 能精确归属到具体 claim_id / section_ref。

        整篇 tutorial 作为单条陈述送检时，知识库相似度会被无关文字稀释，且产出的
        hallucination issue 无法归属到某条 Claim（claim_id/location 只能落空）。
        这里对每条「关键技术 Claim」单独送检：命中幻觉/存疑时，直接带出该 Claim 的
        claim_id 与 section_ref（作为 location），供 _enrich_debate_fields 关联来源与派生 must_fix。

        仅对 is_key_technical=true 的 Claim 送检（非关键陈述不因相似度低而误报），
        无 claims 时返回空列表，由调用方回退到整篇送检。

        分级审核 + 缓存（辩论方案 第 2、4 点）：
          1. 按 claim_type + 溯源标记路由：profile_fact / personalized 豁免全库幻觉判定；
             仅 kb_fact 走 KB 相似度核验。
          2. kb_fact 先查主张级缓存（claim 文本指纹 + chunk_id）：命中 pass → 直接放行不送检；
             命中 hallucination → 复用裁决直接产出幻觉 issue；miss → 送检后把 pass/hallucination
             写回缓存（review_needed 灰区不入缓存）。
        """
        claim_errors: List[Dict[str, Any]] = []
        for c in claims or []:
            if not isinstance(c, dict) or not c.get("is_key_technical"):
                continue
            # 已被 个性化学习资源生成 Agent 修改/丢弃的 Claim 不再送检（status=dropped 语义上已作废）。
            if c.get("status") == "dropped":
                continue
            text = c.get("text") or ""
            if not text.strip():
                continue
            source_chunk_id = c.get("source_chunk_id") or (c.get("metadata_json") or {}).get("source_chunk_id")

            # --- 按类型路由：非知识库事实豁免全库幻觉判定 ---
            route = _classify_claim_route(c, source_chunk_id)
            if route in (_ROUTE_PROFILE_FACT, _ROUTE_PERSONALIZED):
                # 用户画像事实 / 个性化推理：无 KB chunk，不做相似度幻觉判定，避免误报。
                # 关键技术缺来源的「缺依据」由下游 derive_must_fix 统一处理，此处不产 issue。
                continue

            # --- 主张级缓存：同一主张不辩论第二次 ---
            claim_fp = _claim_fingerprint(text, source_chunk_id or "")
            cached = self._cache_lookup_claim(claim_fp, source_chunk_id or "")
            if cached is not None:
                cached_level = cached.get("level")
                if cached_level == "pass":
                    # 命中「已核验通过」：直接放行，跳过整轮送检。
                    continue
                if cached_level == "hallucination":
                    # 命中「已核验幻觉」：复用裁决，重建幻觉 issue（不重新送检）。
                    claim_errors.append(self._build_cached_hallucination(c, cached, source_chunk_id))
                    continue

            try:
                # 有精确来源块时按块核验（绑定确切 chunk），否则退回全库相似度检索。
                if source_chunk_id:
                    errs = await self.knowledge_verify_tool.verify_against_chunk(text, source_chunk_id)
                else:
                    errs = await self.knowledge_verify_tool.verify_knowledge(text)
            except Exception:
                continue

            # 送检结论落缓存：空列表=pass；否则取首条错误的档位/置信度（pass/hallucination 入缓存）。
            level, confidence, verify_method, evidence = self._summarize_verify_result(errs, source_chunk_id)
            self._cache_store_claim(claim_fp, source_chunk_id or "", level, {
                "confidence": confidence,
                "verify_method": verify_method,
                "claim_text": text[:500],
                "evidence": evidence,
            })

            for e in errs:
                # 把幻觉 issue 锚定到该 Claim：claim_id 供 _match_claim 直接命中，
                # location=section_ref 供按位置回退匹配，两者任一都能建立归属。
                e["claim_id"] = c.get("claim_id")
                if c.get("section_ref"):
                    e["location"] = c.get("section_ref")
                # 块级溯源：来源块 ID 与块内位置随 issue 带出（供 _enrich_debate_fields 回填）。
                if source_chunk_id and not e.get("content_chunk_id"):
                    e["content_chunk_id"] = source_chunk_id
                claim_errors.append(e)
        return claim_errors

    @staticmethod
    def _summarize_verify_result(
        errs: List[Dict[str, Any]],
        source_chunk_id: Optional[str],
    ) -> tuple:
        """把 verify_* 返回的 errs 归纳为 (level, confidence, verify_method, evidence)。

        - 空列表 → pass（相似度 >= 0.80）
        - error_type=hallucination → hallucination（相似度 < 0.65）
        - error_type=review_needed → review_needed（0.65~0.80 灰区，不入缓存）
        verify_method 依来源块存在与否记 chunk/lexical（当前适配器用词面相似度）。
        """
        verify_method = "chunk" if source_chunk_id else "lexical"
        if not errs:
            return "pass", 1.0, verify_method, None
        first = errs[0]
        etype = first.get("error_type")
        confidence = first.get("confidence", 0.0)
        level = "hallucination" if etype == "hallucination" else "review_needed"
        return level, confidence, verify_method, first.get("evidence")

    @staticmethod
    def _build_cached_hallucination(
        claim: Dict[str, Any],
        cached: Dict[str, Any],
        source_chunk_id: Optional[str],
    ) -> Dict[str, Any]:
        """用缓存的幻觉裁决重建一条 hallucination issue（跳过重新送检）。"""
        confidence = cached.get("confidence", 0.0)
        e: Dict[str, Any] = {
            "error_type": "hallucination",
            "severity": "critical",
            "location": claim.get("section_ref") or "技术陈述",
            "description": f"[缓存命中] 已核验主张相似度 {confidence:.2f}，判定为潜在幻觉。",
            "evidence": cached.get("evidence"),
            "suggestion": "删除无法验证的技术声明或补充可靠来源。",
            "type": "hallucination",
            "message": f"[缓存命中] 相似度 {confidence:.2f}，潜在幻觉。",
            "confidence": confidence,
            "claim_id": claim.get("claim_id"),
            "from_cache": True,
        }
        if source_chunk_id:
            e["content_chunk_id"] = source_chunk_id
        return e

    def _enrich_debate_fields(
        self,
        errors: List[ReviewError],
        sources: Optional[List[Dict[str, Any]]] = None,
        claims: Optional[List[Dict[str, Any]]] = None,
        round_index: int = 1,
        raw_errors: Optional[List[Any]] = None,
    ) -> List[ReviewError]:
        """为每个 ReviewError 强制补全审核辩论协议字段（协议 2.3 / 2.4 / 第 6 节）。

        - source_refs：命中 sources 则填 source_id 列表，否则为空（=未命中来源）
        - change_request：{"required": bool, "action": str}
        - canonical_error_id / error_id：按 2.4 哈希规则生成，跨轮稳定
        - claim_id：关联的 Claim
        - must_fix / must_fix_reason：按第 6 节派生规则计算
        canonical_error_id 在单个 trace 语义下唯一：同一轮内若出现重复则给 error_id 加序号后缀区分实例。
        """
        source_index = _build_source_index(sources)
        claim_by_section = self._build_claim_index(claims)

        # 建立 raw dict 的查找表（按 error_type + description），用于取回工具原始协议提示字段
        raw_lookup: Dict[tuple, Dict[str, Any]] = {}
        for r in raw_errors or []:
            if isinstance(r, dict):
                key = (
                    str(r.get("error_type") or r.get("type") or "unknown"),
                    str(r.get("description") or r.get("message") or "未提供问题说明"),
                )
                raw_lookup.setdefault(key, r)

        seen_error_ids: Dict[str, int] = {}
        for err in errors:
            raw = raw_lookup.get((err.error_type, err.description), {})

            # 1. 关联 claim
            claim = self._match_claim(err.location, raw.get("claim_id"), claims, claim_by_section)
            claim_id = str(claim.get("claim_id")) if claim else raw.get("claim_id")
            is_key_technical = bool(claim.get("is_key_technical")) if claim else False
            err.claim_id = claim_id

            # 2. source_refs（命中 sources，否则空=未命中来源）
            source_refs = _resolve_source_refs(raw, source_index)
            # 若 error 未显式指定来源，但其 claim 已绑定 source_ref 且命中 sources，则继承
            if not source_refs and claim:
                claim_source = claim.get("source_ref")
                if claim_source and str(claim_source) in source_index:
                    source_refs = [str(claim_source)]
            err.source_refs = source_refs

            # 3. must_fix 派生（先看工具是否已提示 change_request.required）
            change_request_required = _extract_change_request_required(raw)
            source_supports_claim = _extract_source_supports_claim(raw)
            must_fix, must_fix_reason = derive_must_fix(
                error_type=err.error_type,
                severity=err.severity,
                source_refs=source_refs,
                is_key_technical=is_key_technical,
                change_request_required=change_request_required,
                source_supports_claim=source_supports_claim,
            )
            err.must_fix = must_fix
            err.must_fix_reason = must_fix_reason

            # 4. change_request（must_fix=true 时 required 必为 true）
            err.change_request = _build_change_request(raw, must_fix, source_refs)

            # 5. canonical_error_id / error_id（协议 2.4 哈希规则，钉死不可发明）
            canonical = compute_canonical_error_id(
                claim_id=err.claim_id,
                error_type=err.error_type,
                location=err.location,
                description=err.description,
            )
            err.canonical_error_id = canonical
            error_id = compute_error_id(canonical, round_index)
            # 同一轮内 canonical 冲突（重复实例）时追加序号，保证 error_id 实例唯一
            if error_id in seen_error_ids:
                seen_error_ids[error_id] += 1
                error_id = f"{error_id}_{seen_error_ids[error_id]}"
            else:
                seen_error_ids[error_id] = 0
            err.error_id = error_id

            # 6. 块级溯源字段回填（块级并行流水线）：normalize_review_errors 会丢弃未知键，
            #    故从 raw（工具/Claim 输出）回填到 ReviewError 的可选字段。
            if raw.get("content_chunk_id") and not err.content_chunk_id:
                err.content_chunk_id = raw.get("content_chunk_id")
            if raw.get("in_chunk_position") and not err.in_chunk_position:
                err.in_chunk_position = raw.get("in_chunk_position")
            if raw.get("content_block_id") and not err.content_block_id:
                err.content_block_id = raw.get("content_block_id")
            if raw.get("content_version") is not None and err.content_version is None:
                err.content_version = raw.get("content_version")

        return errors

    async def rebut(
        self,
        session_id: str,
        issues: List[Dict[str, Any]],
        author_replies: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        """对 个性化学习资源生成 Agent 的每条回复做二次裁决（复核，协议第 4~5 节 / 专业知识与教学质量审核 Agent任务说明 第 5 点）。

        参数：
        - issues：专业知识与教学质量审核 Agent 上一轮出具的 issues（每条含 canonical_error_id / must_fix / severity /
          source_refs / change_request 等）
        - author_replies：个性化学习资源生成 Agent 的逐条回复，每条含
          canonical_error_id、disposition(accepted|rejected|partially_accepted)、
          reason、patched_sections、source_refs

        返回 {canonical_error_id: "persist" | "withdraw" | "downgrade"}：
        - persist   坚持原 issue（个性化学习资源生成 Agent 反证不成立，或对 must_fix issue 直接 rejected）
        - withdraw  撤销该 issue（个性化学习资源生成 Agent 反证成立，专业知识与教学质量审核 Agent 认错）
        - downgrade 降级（如 critical→major，或 must_fix→false）

        约束：must_fix=true 时 个性化学习资源生成 Agent 不允许直接 rejected；若仍 rejected，专业知识与教学质量审核 Agent 一律 persist，
        交由 Orchestrator 最终裁决（协议第 4 节）。
        """
        print(f"=== 专业知识与教学质量审核 Agent 复核裁决 rebut 开始（session={session_id}）===")
        # 建立 issue 索引（按 canonical_error_id）
        issue_index: Dict[str, Dict[str, Any]] = {}
        for iss in issues or []:
            if isinstance(iss, dict):
                cid = iss.get("canonical_error_id")
                if cid:
                    issue_index[str(cid)] = iss

        verdicts: Dict[str, str] = {}
        for reply in author_replies or []:
            if not isinstance(reply, dict):
                continue
            cid = reply.get("canonical_error_id")
            if not cid:
                continue
            cid = str(cid)
            issue = issue_index.get(cid, {})
            try:
                verdicts[cid] = self._adjudicate_reply(issue, reply)
            except Exception:
                # 单条裁决异常兜底：保守坚持原 issue，避免误撤销
                verdicts[cid] = "persist"

        print(f"=== 专业知识与教学质量审核 Agent 复核裁决 rebut 完成：{verdicts} ===")
        return verdicts

    def _adjudicate_reply(
        self,
        issue: Dict[str, Any],
        reply: Dict[str, Any],
    ) -> str:
        """对单条 个性化学习资源生成 Agent 回复裁决，返回 persist / withdraw / downgrade。"""
        disposition = str(reply.get("disposition", "")).lower()
        must_fix = bool(issue.get("must_fix", False))
        severity = str(issue.get("severity", "major")).lower()
        error_type = issue.get("error_type") or issue.get("type")
        reply_source_refs = reply.get("source_refs") or []
        if isinstance(reply_source_refs, str):
            reply_source_refs = [reply_source_refs]
        patched = reply.get("patched_sections") or []
        if isinstance(patched, str):
            patched = [patched]

        # must_fix=true 且 个性化学习资源生成 Agent 直接 rejected：违反协议约束，一律 persist 交 Orchestrator
        if must_fix and disposition == "rejected":
            return "persist"

        if disposition == "accepted":
            # 个性化学习资源生成 Agent 已按要求整改：若补充了新来源或完成 patch，专业知识与教学质量审核 Agent 认可并撤销
            if reply_source_refs or patched:
                return "withdraw"
            # 声称接受但无任何整改证据 → 保守坚持
            return "persist"

        if disposition == "partially_accepted":
            # 部分整改：风险下降但未完全消除 → 降级（critical→major，或解除 must_fix）。
            # 但「降级」会经 Orchestrator 解除 must_fix 并降 severity，一轮即放松红线。
            #
            # 关键：rebut 发生在 个性化学习资源生成 Agent.modify_package 之前，此时正文尚未真正修改，
            # patched_sections 往往只是 个性化学习资源生成 Agent 对 issue.location 的回显（其 modify_package
            # 目前仅在文末追加「## 辩论修订」说明，并不定点改正原文，见检查报告问题③.2）。
            # 因此 patched_sections 不能单独作为「已整改」的证据，否则会「口头部分接受即一轮放行」。
            #
            # 收紧规则（P1）：
            #   1) 无任何证据（既无新来源也无 patch）→ persist；
            #   2) must_fix issue：必须提供「新来源」source_refs 作为可核验证据才降级；
            #      仅有 patched_sections（无新来源）→ persist，交由下一轮重新审核核验实际改动；
            #   3) 质量红线类（认知负荷/难度超载）：靠改写正文缓解，且需下一轮复核 →
            #      一律 persist，仅补来源不算整改；
            #   4) 非 must_fix 的部分整改（有新来源或 patch）→ 按原规则降级。
            has_evidence = bool(reply_source_refs or patched)
            if not has_evidence:
                return "persist"
            if _is_quality_redline_error_type(error_type):
                # 质量红线：改动效果需下一轮实测，rebut 阶段不预先降级
                return "persist"
            if must_fix and not reply_source_refs:
                # must_fix 且仅回显 patched_sections、无可核验新来源 →
                # 不预先降级，persist 交下一轮 re-review 核验正文是否真的改对
                return "persist"
            return "downgrade"

        if disposition == "rejected":
            # 非 must_fix 的驳回：若 个性化学习资源生成 Agent 给出有效反证（新来源或说明），专业知识与教学质量审核 Agent 认错撤销
            if reply_source_refs:
                return "withdraw"
            # 仅 minor 且给出理由，可降级；否则坚持
            if severity == "minor" and reply.get("reason"):
                return "downgrade"
            return "persist"

        # disposition 缺失或非法值：保守坚持
        return "persist"

    async def analyze_code_error(
        self,
        session_id: str,
        error_info: str,
        code: str = "",
        question: str = "",
        profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        分析用户代码错误，返回结构化的诊断结果。

        支持两种输入场景：
        1. 用户提供问题代码 + 报错信息（error_info + code）
        2. 用户仅描述问题，无代码（error_info 或 question）
        3. 用户提供代码但无报错 → 自动执行代码获取 error_info

        6步流程（对齐 technical_review.skill.md 2.2节）：
        Step 1: 解析报错信息 → classify_error_severity() + classify_error_root_cause()
        Step 2: 若有代码无报错 → code_executor_tool.execute() 获取 error_info
        Step 3: knowledge_verify_tool.verify_code_error() 知识库匹配
        Step 4: code_parser_tool.parse_code() 语法检查
        Step 5: rule_check_tool.check_code_error_patterns() 错误模式检测
        Step 6: 组装 CodeErrorClassification 输出

        返回字段（对齐 code_error_patterns.skill.md 输出规范）：
            {
                "error_category": str,
                "severity": str,
                "root_cause": str,
                "fix_suggestion": {
                    "summary": str,
                    "detail": str,
                    "code_fix": str,
                    "environment_checks": list
                },
                "confidence": float,
                "classification_method": str,
                "evidence": str,
                "error_type": str,
                "is_known_error": bool,
                "known_solution": str,
                "code_syntax_errors": list,
                "code_rule_violations": list,
                "mode": "code_error_analysis"
            }
        """
        print(f"=== 专业知识与教学质量审核 Agent 代码错误分析 开始 ===")
        print(f"  错误信息: {error_info[:200]}")
        if code:
            print(f"  代码片段: {code[:200]}")
        if question:
            print(f"  用户问题: {question[:200]}")

        profile_dict = self._profile_to_dict(profile) if profile else {}

        result: Dict[str, Any] = {
            "error_category": "unknown",
            "severity": "major",
            "root_cause": "code_unknown",
            "fix_suggestion": {
                "summary": "",
                "detail": "",
                "code_fix": "",
                "environment_checks": [],
            },
            "confidence": 0.0,
            "classification_method": "rule_based",
            "evidence": "",
            "error_type": "",
            "is_known_error": False,
            "known_solution": "",
            "code_syntax_errors": [],
            "code_rule_violations": [],
            "mode": "code_error_analysis",
        }

        # Step 2: 若有代码无报错 → 执行代码获取 error_info
        exec_result: Dict[str, Any] = {}
        if code and not error_info.strip():
            print(f"  [Step2] 用户提供代码但无报错，执行代码...")
            try:
                exec_result = await self.code_executor_tool.execute(code=code)
                if exec_result.get("has_error", False):
                    error_info = exec_result.get("error_info", "")
                    print(f"  [Step2] 执行捕获错误: {error_info[:100]}")
                else:
                    print(f"  [Step2] 代码执行成功，无运行时错误")
            except Exception:
                exec_result = {}
                print(f"  [Step2] 代码执行异常，跳过")

        # Step 1: 解析报错 → 错误分类与归因
        error_type = ""
        if error_info.strip():
            classification = _classify_error_by_patterns(error_info)
            result["error_category"] = classification["category"]
            result["severity"] = classification["severity"]

            error_type = self._extract_error_type_from_info(error_info)
            if error_type:
                result["error_type"] = error_type
                result["severity"] = classify_error_severity(error_type)
                result["root_cause"] = classify_error_root_cause(error_type, error_info)
            else:
                result["root_cause"] = classify_error_root_cause("unknown", error_info)
        else:
            result["root_cause"] = "code_unknown"
            if question:
                result["root_cause"] = self._infer_root_cause_from_question(question)

        print(f"  [Step1] 错误分类: {result['error_category']}, 严重等级: {result['severity']}, 归因: {result['root_cause']}")

        # Step 3: 知识库匹配
        if error_info.strip():
            try:
                error_verification = await self.knowledge_verify_tool.verify_code_error(
                    error_info=error_info,
                    code=code,
                )
            except Exception:
                error_verification = {}
            result["is_known_error"] = error_verification.get("is_known_error", False)
            result["known_solution"] = error_verification.get("known_solution", "")
            kb_severity = error_verification.get("severity", "")
            kb_root_cause = error_verification.get("root_cause", "")
            kb_error_type = error_verification.get("error_type", "")
            kb_confidence = error_verification.get("confidence", 0.0)

            if kb_error_type and not result["error_type"]:
                result["error_type"] = kb_error_type
            if kb_severity and kb_severity != "major":
                result["severity"] = kb_severity
            if kb_root_cause and kb_root_cause != "code_unknown":
                result["root_cause"] = kb_root_cause
        else:
            kb_confidence = 0.0

        # Step 4: 代码语法检查
        syntax_errors: List[Dict[str, Any]] = []
        if code:
            try:
                syntax_errors = await self.code_parser_tool.parse_code(code)
            except Exception:
                syntax_errors = []
            result["code_syntax_errors"] = syntax_errors

        # Step 5: 代码错误模式检测
        rule_violations: List[Dict[str, Any]] = []
        if code:
            try:
                rule_violations = await self.rule_check_tool.check_code_error_patterns(code)
            except Exception:
                rule_violations = []
            result["code_rule_violations"] = rule_violations

        # Step 6: 组装输出
        code_fix = ""
        env_checks: List[str] = []
        fix_detail = ""

        if result["is_known_error"] and result["known_solution"]:
            fix_detail = result["known_solution"]
            code_fix = result["known_solution"]
        elif result["root_cause"] == "environment_issue":
            fix_detail = f"检测到环境问题（{result['error_type']}），建议检查运行环境配置。"
            env_checks = [
                "确认依赖包是否已正确安装",
                "检查Python版本是否满足要求",
                "确认虚拟环境是否已激活",
                "检查文件路径和权限设置",
            ]
            if result["error_type"] in ("CUDAError", "PyTorchError", "TorchError"):
                env_checks.append("确认GPU资源是否充足（nvidia-smi）")
                env_checks.append("检查CUDA版本与PyTorch版本是否匹配")
            elif result["error_type"] in ("ModuleNotFoundError", "ImportError"):
                env_checks.append("执行 pip install 安装缺失依赖")
        elif result["root_cause"] == "dependency_issue":
            fix_detail = f"检测到外部依赖问题（{result['error_type']}），可能是第三方库版本不兼容。"
            env_checks = [
                "检查第三方库版本是否兼容",
                "查看库的API变更文档",
                "尝试升级或降级相关依赖",
            ]
        elif result["root_cause"] == "data_issue":
            fix_detail = "检测到数据问题，输入数据格式或内容可能不符合预期。"
            env_checks = [
                "检查输入数据格式是否正确",
                "验证数据编码和字符集",
                "确认数据字段完整性",
            ]
        elif result["error_category"] == "SyntaxError":
            fix_detail = f"代码存在语法错误（{result['error_type']}），需要修复语法问题后才能运行。"
            code_fix = "请检查代码的语法正确性，特别是缩进、括号配对和关键字使用。"
        elif result["error_category"] == "RuntimeError":
            fix_detail = f"代码存在运行时错误（{result['error_type']}），代码可运行但会产生异常结果。"
            code_fix = f"建议检查{result['error_type']}相关的逻辑，确保操作的对象存在且类型正确。"
        elif result["error_category"] == "LogicError":
            fix_detail = "代码存在逻辑错误，代码可运行但结果可能不符合预期。"
        else:
            if question:
                fix_detail = f"未能精确识别错误类型。用户问题：{question}"
            else:
                fix_detail = "未能精确识别错误类型，建议提供更详细的报错信息和代码上下文。"

        result["fix_suggestion"] = {
            "summary": f"{result['error_category']}({result['severity']}): {fix_detail[:80]}",
            "detail": fix_detail,
            "code_fix": code_fix,
            "environment_checks": env_checks,
        }

        if result["is_known_error"] and kb_confidence > 0:
            result["confidence"] = max(0.6, kb_confidence)
            result["classification_method"] = "hybrid"
        elif kb_confidence > 0.5:
            result["confidence"] = kb_confidence
            result["classification_method"] = "knowledge_based"
        elif result["error_category"] != "unknown":
            result["confidence"] = 0.7
            result["classification_method"] = "rule_based"
        else:
            result["confidence"] = 0.3
            result["classification_method"] = "rule_based"

        evidence_parts = [f"规则分类: {result['error_category']}({result['severity']})"]
        if result["error_type"]:
            evidence_parts.append(f"错误类型: {result['error_type']}")
        evidence_parts.append(f"归因: {result['root_cause']}")
        if result["is_known_error"]:
            evidence_parts.append(f"知识库匹配度: {kb_confidence:.2f}")
        if result["code_syntax_errors"]:
            evidence_parts.append(f"语法检查: {len(result['code_syntax_errors'])}个问题")
        if result["code_rule_violations"]:
            evidence_parts.append(f"规范检查: {len(result['code_rule_violations'])}个问题")
        if exec_result:
            evidence_parts.append(f"代码执行: {'成功' if not exec_result.get('has_error') else '出错'}")
        result["evidence"] = "; ".join(evidence_parts)

        print(f"=== 专业知识与教学质量审核 Agent 代码错误分析 完成 ===")
        print(f"  错误分类: {result['error_category']}")
        print(f"  严重等级: {result['severity']}")
        print(f"  归因: {result['root_cause']}")
        print(f"  错误类型: {result['error_type']}")
        print(f"  已知错误: {result['is_known_error']}")
        print(f"  置信度: {result['confidence']:.2f}")
        print(f"  分类方法: {result['classification_method']}")

        return result

    @staticmethod
    def _extract_error_type_from_info(error_info: str) -> str:
        match = re.match(r'^(\w+Error|\w+Exception|SyntaxError|IndentationError|TabError)(?:[:\s])', error_info or "")
        if match:
            return match.group(1)
        type_patterns = [
            "SyntaxError", "IndentationError", "TabError",
            "TypeError", "ValueError", "KeyError", "IndexError",
            "AttributeError", "NameError", "UnboundLocalError",
            "ZeroDivisionError", "MemoryError", "RecursionError", "OverflowError",
            "ImportError", "ModuleNotFoundError", "FileNotFoundError", "PermissionError",
            "IsADirectoryError", "NotADirectoryError",
            "ConnectionError", "TimeoutError", "OSError", "IOError",
            "CUDAError", "TorchError", "PyTorchError", "TensorFlowError",
            "JSONDecodeError", "JSONEncodeError", "HTTPError", "RequestException",
            "AssertionError", "NotImplementedError", "StopIteration", "GeneratorExit",
            "RuntimeError", "Exception",
        ]
        for tp in type_patterns:
            if tp.lower() in (error_info or "").lower():
                return tp
        return ""

    @staticmethod
    def _infer_root_cause_from_question(question: str) -> str:
        text_lower = (question or "").lower()
        env_kw = ("pip install", "import", "module", "环境", "依赖", "package", "not found", "找不到")
        data_kw = ("parse", "format", "json", "数据", "invalid", "格式")
        if any(kw in text_lower for kw in env_kw):
            return "environment_issue"
        if any(kw in text_lower for kw in data_kw):
            return "data_issue"
        return "code_unknown"
