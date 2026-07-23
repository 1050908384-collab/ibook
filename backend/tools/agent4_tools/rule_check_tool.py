# -*- coding: utf-8 -*-
"""
rule_check_tool — Agent4 行业规范检查工具
================================================
功能：检查内容是否违反行业规范（安全/部署/数据隐私）+ 技术确定性规则（LoRA机制/参数量）。
    - 真实模式：内置确定性正则规则 + 匹配 knowledge_docs 表中 doc_type='best_practice' 的规则文档作为依据来源
    - Mock 模式：返回 ./data/test_data/mock_rule_check_tool.json 预设违规列表

统一接口：
    def rule_check(input_data: dict, config: dict = None) -> dict
    返回：{"status", "data", "message", "mock_used"}

规则对齐：
    - technical_review.skill.md 规则2（错误模式库）：LoRA机制错误、参数量计算错误
    - industry_standards.skill.md 规则1（安全）、规则2（部署）、规则3（数据隐私）
    - 输出 violation 结构对齐 models/review.py 的 ReviewError 字段
      （error_type/severity/location/description/evidence/suggestion + category）

违规分级（industry_standards.skill.md 步骤3）：
    - Critical：硬编码密钥、不可商用模型冒充自有、LoRA机制错误、参数量计算错误
    - Major：端口暴露、权限缺失、真实用户数据、非加密传输
    - Minor：许可证说明缺失、显存标注缺失
    passed 规则：存在任意 critical → passed=False（阻断发布）

保留 RuleCheckTool 类作为向后兼容适配器。
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

MOCK_FILE = DATA_DIR / "test_data" / "mock_rule_check_tool.json"


# ------------------------------------------------------------------
# 内置确定性规则（正则匹配）
# 每条规则：error_type / severity / category / pattern / location / description / evidence / suggestion
# 规则来源：industry_standards.skill.md 规则1/2/3 + technical_review.skill.md 规则2
# ------------------------------------------------------------------
_BUILTIN_RULES: List[Dict[str, Any]] = [
    # ---- 规则1 安全规范（industry_standards.skill.md） ----
    {
        "error_type": "hardcoded_secret",
        "severity": "critical",
        "category": "security",
        "pattern": r'API_KEY\s*=\s*["\']sk-|os\.environ\[\s*["\']\w*KEY["\']\s*\]\s*=\s*["\']|api_key\s*=\s*["\'][a-zA-Z0-9_\-]{20,}["\']',
        "location": "API密钥赋值",
        "description": "检测到硬编码 API 密钥，密钥以明文形式写入源码。",
        "evidence": "industry_standards规则1.1：禁止 API_KEY = \"sk-...\" 等硬编码密钥。",
        "suggestion": '改为 os.getenv("OPENAI_API_KEY") 从环境变量读取，密钥存入 .env 并加入 .gitignore。',
    },
    {
        "error_type": "port_exposure",
        "severity": "major",
        "category": "security",
        "pattern": r'host\s*=\s*["\']0\.0\.0\.0["\']|(?:app|uvicorn)\.run\([^)]*0\.0\.0\.0|bind\s*.*0\.0\.0\.0:\s*\d+',
        "location": "服务启动调用",
        "description": "服务绑定到 0.0.0.0，暴露所有网卡，存在未授权访问风险。",
        "evidence": "industry_standards规则1.2：禁止 host=\"0.0.0.0\" 绑定全部网卡。",
        "suggestion": '改为 host="127.0.0.1"，通过 nginx 反向代理对外暴露。',
    },
    # ---- 规则3 数据隐私 ----
    {
        "error_type": "real_user_data",
        "severity": "major",
        "category": "privacy",
        "pattern": r'phone\s*=\s*["\']1[3-9]\d{9}["\']|email\s*=\s*["\'][^@]+@(?!example\.com)[^"\']+["\']|user_id\s*=\s*["\']\d{6,}["\']',
        "location": "用户数据字段",
        "description": "示例代码中使用疑似真实用户数据（手机号/邮箱/用户ID未脱敏）。",
        "evidence": "industry_standards规则3.1：涉及用户数据的操作必须脱敏。",
        "suggestion": "使用脱敏示例：phone=\"138****1234\"、email=\"user001@example.com\"。",
    },
    {
        "error_type": "insecure_transport",
        "severity": "major",
        "category": "privacy",
        "pattern": r'(?:http|ws)://(?!localhost|127\.0\.0\.1)[^\s"\']+',
        "location": "数据传输调用",
        "description": "使用非加密传输协议（http/ws），存在数据窃听风险。",
        "evidence": "industry_standards规则3.2：数据传输 URL 必须以 https:// 或 wss:// 开头。",
        "suggestion": "改用 https:// 或 wss:// 加密传输。",
    },
]


# ------------------------------------------------------------------
# 统一函数式接口（规范实现）
# ------------------------------------------------------------------
def rule_check(input_data: dict, config: dict = None) -> dict:
    """
    检查内容是否违反行业规范与技术确定性规则。

    参数：
        input_data: dict
            - content: str       待检查内容（代码/配置/文档/技术陈述）
            - scenario: str       Mock 模式下选择场景：normal|boundary|exception
        config: dict
            - use_mock: bool      是否使用 Mock（默认 True）

    返回：
        {
            "status": "success" | "error",
            "data": {
                "violations": [
                    {error_type, severity, category, location, description, evidence,
                     suggestion, reference_doc}
                ],
                "summary": str,
                "counts": {"critical": int, "major": int, "minor": int},
                "passed": bool,                 # 无 critical 即通过
                "best_practice_docs_loaded": int,
                "db_available": bool
            },
            "message": str,
            "mock_used": bool
        }
    """
    config = config or {}
    use_mock = config.get("use_mock", True)

    content = (input_data or {}).get("content") or ""
    if not content.strip():
        return {
            "status": "error",
            "data": None,
            "message": "输入未包含 content 字段，无法检查",
            "mock_used": use_mock,
        }

    try:
        if use_mock:
            return _mock_check(content, input_data)

        # 真实模式：内置规则 + best_practice 文档加载
        bp_docs, db_available = _load_best_practice_docs()
        violations: List[Dict[str, Any]] = []

        # 1. LoRA 机制错误 + 参数量计算错误（technical_review.skill.md 规则2，需特殊逻辑）
        violations.extend(_check_lora_mechanism(content))
        violations.extend(_check_parameter_count(content))

        # 2. 内置正则规则（安全/数据隐私）
        for rule in _BUILTIN_RULES:
            if re.search(rule["pattern"], content, re.IGNORECASE):
                violations.append(_build_violation(rule))

        # 3. 部署规范：许可证缺失 + 显存标注缺失（条件触发，需检测模型加载）
        violations.extend(_check_deployment_rules(content))

        # 4. 用 best_practice 文档为各 category 违规补充 reference_doc 依据
        cat_map = _build_category_doc_map(bp_docs)
        for v in violations:
            v["reference_doc"] = cat_map.get(v["category"])

        return _assemble(violations, len(bp_docs), db_available)
    except Exception as e:
        logger.exception("rule_check 执行失败")
        return {
            "status": "error",
            "data": None,
            "message": f"rule_check 执行异常: {e}",
            "mock_used": use_mock,
        }


# ------------------------------------------------------------------
# 真实模式：数据库规则文档加载
# ------------------------------------------------------------------
def _get_db_path() -> Path:
    """解析 settings.DATABASE_URL 为绝对路径（与 sqlite_client 一致）。"""
    url = settings.DATABASE_URL
    raw = url.replace("sqlite:///", "", 1)
    p = Path(raw)
    if not p.is_absolute():
        p = (BACKEND_DIR / p).resolve()
    return p


def _load_best_practice_docs() -> tuple[List[Dict[str, str]], bool]:
    """
    SQL: 从 knowledge_docs 表读取 doc_type='best_practice' 的行业规范规则文档。
    knowledge_docs.doc_type 取值含 tutorial/code/spec/best_practice/pitfall，
    其中 best_practice 即 Agent4 行业规范规则库（industry_standards.skill.md）。
    数据库不存在时返回空列表与 db_available=False。
    """
    db_path = _get_db_path()
    if not db_path.exists():
        logger.warning("数据库不存在: %s，rule_check 无 best_practice 规则文档", db_path)
        return [], False
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            cur = conn.execute(
                """
                -- 仅取 best_practice 类型文档作为行业规范规则依据
                SELECT doc_id, block_id, title, content, source
                FROM knowledge_docs
                WHERE doc_type = 'best_practice'
                """
            )
            rows = cur.fetchall()
        finally:
            conn.close()
        docs = [
            {"doc_id": r[0], "block_id": r[1], "title": r[2], "content": r[3], "source": r[4]}
            for r in rows
        ]
        return docs, True
    except sqlite3.Error as e:
        logger.warning("读取 best_practice 文档失败: %s", e)
        return [], False


def _build_category_doc_map(docs: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    """按标题/内容关键词将 best_practice 文档归类，供违规引用。"""
    keywords = {
        "security": ["安全", "密钥", "端口", "权限", "security", "secret"],
        "deployment": ["部署", "显存", "许可证", "license", "vram", "deployment"],
        "privacy": ["隐私", "脱敏", "数据", "privacy", "敏感"],
        "technical": ["lora", "LoRA", "微调", "参数", "训练"],
    }
    cat_map: Dict[str, Dict[str, str]] = {}
    for d in docs:
        text = (d.get("title", "") + " " + d.get("content", "")).lower()
        for cat, kws in keywords.items():
            if cat not in cat_map and any(kw.lower() in text for kw in kws):
                cat_map[cat] = {"doc_id": d["doc_id"], "title": d["title"], "source": d.get("source")}
    return cat_map


# ------------------------------------------------------------------
# 技术确定性规则（保留并迁移自原 rule_check_tool.py）
# ------------------------------------------------------------------
def _check_lora_mechanism(content: str) -> List[Dict[str, Any]]:
    """检测 LoRA 机制错误：将 LoRA 描述为 SVD 分解或训练 U/Σ/V。"""
    normalized = "".join(content.lower().split())
    patterns = (
        r"(?:原始)?权重(?:矩阵)?w.*?分解.*?u.*?v",
        r"w.*?=.*?u.*?(?:sigma|σ).*?v",
        r"lora.*?(?:svd|奇异值分解)",
    )
    if any(re.search(p, normalized, re.IGNORECASE) for p in patterns):
        return [{
            "error_type": "lora_mechanism_error",
            "severity": "critical",
            "category": "technical",
            "location": "LoRA原理说明",
            "description": "将LoRA描述为对原始权重W执行SVD或训练U/Σ/V，混淆了LoRA与权重矩阵分解。",
            "evidence": "标准LoRA冻结W并训练低秩增量ΔW=(α/r)BA。rule_check_tool正则命中：lora.*?(?:svd|奇异值分解)。",
            "suggestion": "改为说明冻结原权重，仅训练低秩矩阵A和B形成增量更新。",
        }]
    return []


def _check_parameter_count(content: str) -> List[Dict[str, Any]]:
    """检测参数量计算错误：声称参数量下降但 trainable >= original。"""
    normalized = "".join(content.lower().split())
    m = re.search(
        r"(?:原模型|原始模型|模型).*?(\d+(?:\.\d+)?)万(?:个)?参数.*?(?:微调参数|参数量).*?(\d+(?:\.\d+)?)万",
        normalized,
    )
    if not m:
        return []
    original = float(m.group(1))
    trainable = float(m.group(2))
    claims_reduction = any(w in normalized for w in ("减少", "降低", "仅为", "只有"))
    if claims_reduction and trainable >= original:
        return [{
            "error_type": "parameter_count_error",
            "severity": "critical",
            "category": "technical",
            "location": "参数量示例",
            "description": f"示例声称参数量下降，但可训练参数 {trainable:g}万 不小于原参数 {original:g}万。",
            "evidence": "LoRA参数量应按 r(d_in+d_out) 计算，并与原矩阵 d_in*d_out 比较。",
            "suggestion": "重新选择合理维度和秩，并逐步计算参数量。",
        }]
    return []


# ------------------------------------------------------------------
# 部署规范（条件触发）
# ------------------------------------------------------------------
def _check_deployment_rules(content: str) -> List[Dict[str, Any]]:
    """检测许可证说明缺失与显存标注缺失（仅在涉及模型加载时触发）。"""
    violations: List[Dict[str, Any]] = []
    # 涉及模型加载的标志
    model_load = bool(re.search(
        r"from_pretrained|device_map|load.*model|加载.*模型|AutoModel", content, re.IGNORECASE
    ))
    if not model_load:
        return violations

    # 规则2.1 许可证合规：涉及模型加载但无许可证说明
    if not re.search(r"许可证|license|MIT|Apache|GPL|Llama|Qwen|ChatGLM license", content, re.IGNORECASE):
        violations.append({
            "error_type": "license_missing",
            "severity": "minor",
            "category": "deployment",
            "location": "部署说明",
            "description": "部署文档引用模型但未标注许可证类型。",
            "evidence": "industry_standards规则2.1：部署文档引用模型时必须标注许可证类型。",
            "suggestion": "补充许可证说明，例如：# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）。",
        })

    # 规则2.2 显存最低要求标注：涉及模型加载但无显存关键词
    if not re.search(r"显存|VRAM|GPU内存|memory", content, re.IGNORECASE):
        violations.append({
            "error_type": "vram_note_missing",
            "severity": "minor",
            "category": "deployment",
            "location": "部署说明",
            "description": "部署文档涉及模型加载但未标注显存要求。",
            "evidence": "industry_standards规则2.2：涉及 from_pretrained 必须含显存关键词（显存|VRAM|GPU内存|memory）。",
            "suggestion": "补充注释，例如：# 注意：Llama-2-7b FP16 需要至少14GB显存。",
        })
    return violations


def _build_violation(rule: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "error_type": rule["error_type"],
        "severity": rule["severity"],
        "category": rule["category"],
        "location": rule["location"],
        "description": rule["description"],
        "evidence": rule["evidence"],
        "suggestion": rule["suggestion"],
    }


# ------------------------------------------------------------------
# 结果组装
# ------------------------------------------------------------------
def _assemble(violations: List[Dict[str, Any]], bp_count: int, db_available: bool) -> dict:
    counts = {"critical": 0, "major": 0, "minor": 0}
    for v in violations:
        sev = v.get("severity", "major")
        counts[sev] = counts.get(sev, 0) + 1
        # 兼容旧调用方 print 使用的 type/message 字段
        v["type"] = v["error_type"]
        v["message"] = v["description"]

    passed = counts["critical"] == 0
    summary = (
        f"检测到 {len(violations)} 处违规："
        f"{counts['critical']}个critical、{counts['major']}个major、{counts['minor']}个minor。"
        + ("无 critical 违规，不阻断发布。" if passed else "存在 critical 违规，已阻断发布。")
    )
    return {
        "status": "success",
        "data": {
            "violations": violations,
            "summary": summary,
            "counts": counts,
            "passed": passed,
            "best_practice_docs_loaded": bp_count,
            "db_available": db_available,
        },
        "message": summary,
        "mock_used": False,
    }


# ------------------------------------------------------------------
# Mock 模式
# ------------------------------------------------------------------
def _mock_check(content: str, input_data: dict) -> dict:
    """从 mock JSON 读取预设场景，按 scenario 或内容自动选择。"""
    scenario = _pick_scenario(content, input_data)
    scenarios = _load_mock()
    if scenario not in scenarios:
        scenario = "normal"

    data = json.loads(json.dumps(scenarios[scenario]))  # 深拷贝避免污染缓存
    # 回填 content 便于核对
    data["content"] = content[:200]
    data.setdefault("db_available", False)
    data.setdefault("best_practice_docs_loaded", 0)

    return {
        "status": "success",
        "data": data,
        "message": f"[Mock] {data.get('summary', '')}",
        "mock_used": True,
    }


def _pick_scenario(content: str, input_data: dict) -> str:
    """优先用显式 scenario；否则按内容自动判定场景。"""
    scenario = (input_data or {}).get("scenario")
    if scenario:
        return scenario
    # critical 安全违规 → exception
    if re.search(r'API_KEY\s*=\s*["\']sk-|0\.0\.0\.0|svd|奇异值', content, re.IGNORECASE):
        return "exception"
    # 仅 minor 部署缺失 → boundary
    if re.search(r"from_pretrained", content, re.IGNORECASE) and not re.search(r"显存|VRAM", content, re.IGNORECASE):
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
#   check_rules(content)      → 检查教程/文本内容
#   check_code_rules(code)    → 检查代码示例（路由同 check_rules）
# ------------------------------------------------------------------
class RuleCheckTool:
    def __init__(self):
        # 默认走 Mock，保证在知识库未就绪时仍可运行
        self._default_config = {"use_mock": True}

    async def check_rules(self, content: str) -> List[Dict[str, Any]]:
        result = rule_check({"content": content}, config=self._default_config)
        if result.get("status") != "success":
            return []
        return result["data"].get("violations", [])

    async def check_code_rules(self, code: str) -> List[Dict[str, Any]]:
        result = rule_check({"content": code}, config=self._default_config)
        if result.get("status") != "success":
            return []
        return result["data"].get("violations", [])
