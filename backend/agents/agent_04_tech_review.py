# -*- coding: utf-8 -*-
import re
from typing import Dict, Any, List, Optional
from agents.base_agent import BaseAgent
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
        self.load_tools([
            self.code_executor_tool,
            self.code_parser_tool,
            self.knowledge_verify_tool,
            self.rule_check_tool,
            self.difficulty_analyzer_tool,
            self.cognitive_load_tool,
            self.style_match_tool,
        ])

    def _profile_to_dict(self, profile: Any) -> Dict[str, Any]:
        if hasattr(profile, "model_dump"):
            return profile.model_dump(mode="json")
        if hasattr(profile, "dict"):
            return profile.dict()
        if isinstance(profile, dict):
            return profile
        return {}

    async def review(
        self,
        session_id: str,
        content: Dict[str, Any],
        profile: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        errors = []
        print(f"=== Agent4(合并后) TechnicalReviewAgent 开始审核 ===")

        profile_dict = self._profile_to_dict(profile) if profile else {}

        if "tutorial" in content:
            print(f"=== 审核教程内容（前200字符）===\n{content['tutorial'][:200]}")
            # 技术审核
            hallucination_errors = await self.knowledge_verify_tool.verify_knowledge(content["tutorial"])
            errors.extend(hallucination_errors)

            rule_errors = await self.rule_check_tool.check_rules(content["tutorial"])
            errors.extend(rule_errors)

            # 质量审核（原Agent5功能）
            print(f"=== 质量审核：分析教程难度 ===")
            difficulty_errors = await self.difficulty_analyzer_tool.analyze(
                content["tutorial"], profile=profile_dict
            )
            errors.extend(difficulty_errors)

            load_errors = await self.cognitive_load_tool.analyze(content["tutorial"])
            errors.extend(load_errors)

            style_errors = await self.style_match_tool.check(
                content["tutorial"], profile=profile_dict
            )
            errors.extend(style_errors)

        if "code_examples" in content:
            print(f"=== 审核代码示例 ===")
            for code in content["code_examples"]:
                syntax_errors = await self.code_parser_tool.parse_code(code.get("code", ""))
                errors.extend(syntax_errors)

                code_rule_errors = await self.rule_check_tool.check_code_rules(code.get("code", ""))
                errors.extend(code_rule_errors)

        if "answer" in content:
            print(f"=== 审核回答内容（前200字符）===\n{content['answer'][:200]}")
            # 技术审核
            hallucination_errors = await self.knowledge_verify_tool.verify_knowledge(content["answer"])
            errors.extend(hallucination_errors)

            # 质量审核（原Agent5功能）
            print(f"=== 质量审核：分析回答质量 ===")
            style_errors = await self.style_match_tool.check(
                content["answer"], profile=profile_dict
            )
            errors.extend(style_errors)

            difficulty_errors = await self.difficulty_analyzer_tool.analyze(
                content["answer"], profile=profile_dict
            )
            errors.extend(difficulty_errors)

        print(f"=== Agent4(合并后) TechnicalReviewAgent 审核完成 ===")
        print(f"=== 发现问题: {len(errors)}个 ===")
        if errors:
            for i, e in enumerate(errors, 1):
                print(f"  {i}. [{e.get('type', 'unknown')}] {e.get('message', '')}")

        return errors

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
        print(f"=== Agent4 代码错误分析 开始 ===")
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
            exec_result = await self.code_executor_tool.execute(code=code)
            if exec_result.get("has_error", False):
                error_info = exec_result.get("error_info", "")
                print(f"  [Step2] 执行捕获错误: {error_info[:100]}")
            else:
                print(f"  [Step2] 代码执行成功，无运行时错误")

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
            error_verification = await self.knowledge_verify_tool.verify_code_error(
                error_info=error_info,
                code=code,
            )
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
            syntax_errors = await self.code_parser_tool.parse_code(code)
            result["code_syntax_errors"] = syntax_errors

        # Step 5: 代码错误模式检测
        rule_violations: List[Dict[str, Any]] = []
        if code:
            rule_violations = await self.rule_check_tool.check_code_error_patterns(code)
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

        print(f"=== Agent4 代码错误分析 完成 ===")
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