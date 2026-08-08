# -*- coding: utf-8 -*-
import re
import uuid
from typing import Any, Dict, List, Optional

from agents.base_agent import BaseAgent
from models.stage import Claim, LearningPackage, Source
from tools.agent3_tools.vector_search_tool import VectorSearchTool
from tools.agent3_tools.video_search_tool import VideoSearchTool
from tools.agent3_tools.llm_generate_tool import LLMGenerateTool
from tools.agent3_tools.code_generation_tool import CodeGenerationTool
from tools.agent3_tools.question_extract_tool import QuestionExtractTool
from tools.agent3_tools.template_render_tool import TemplateRenderTool
from tools.agent3_tools.format_converter_tool import FormatConverterTool
from tools.agent3_tools.code_executor_tool import CodeExecutorTool


def _extract_claims_from_tutorial(tutorial: str, sources: List[Source]) -> List[Claim]:
    """从教程中抽取可验证陈述，绑定到 source 和 section_ref."""
    claims: List[Claim] = []
    lines = tutorial.split("\n")
    current_section = "sec_0"
    section_counter = 0
    para_counter = 0

    for line_idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            section_counter += 1
            current_section = f"sec_{section_counter}"
            para_counter = 0
            continue
        if stripped.startswith("```"):
            continue

        para_counter += 1
        section_ref = f"tutorial:{current_section}:p_{para_counter}"

        sentences = re.split(r'[。！？.!?\n]+', stripped)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

        for sent_idx, sentence in enumerate(sentences):
            is_key = any(kw in sentence for kw in [
                "原理", "公式", "架构", "核心", "关键", "必须",
                "algorithm", "function", "method", "class",
                "必须", "只能", "不可", "不能", "should", "must",
            ])
            claim_id = f"clm_{uuid.uuid4().hex[:12]}"

            source_ref = None
            for src in sources:
                if src.source_type == "knowledge_doc" and src.locator:
                    source_ref = src.source_id
                    break

            claims.append(Claim(
                claim_id=claim_id,
                text=sentence[:200],
                source_ref=source_ref,
                source_span=section_ref,
                section_ref=section_ref,
                claim_type="concept" if is_key else "fact",
                is_key_technical=is_key,
                confidence=0.85 if is_key else 0.6,
                status="active",
            ))

    return claims


def _generate_sources(block_id: str, related_docs: List[Dict[str, Any]]) -> List[Source]:
    """根据检索到的文档生成来源清单。"""
    sources: List[Source] = []
    for idx, doc in enumerate(related_docs[:5]):
        src_id = f"src_{block_id}_{idx}"
        sources.append(Source(
            source_id=src_id,
            source_type="knowledge_doc",
            title=doc.get("title", f"文档{idx+1}"),
            locator=doc.get("source") or doc.get("doc_id", f"doc_{idx}"),
            content_ref=f"chunk:{doc.get('chunk_id', f'chk_{idx}')}",
            source_version=doc.get("version", "2026-07-31"),
            metadata_json={"block_id": block_id},
        ))
    return sources


class ContentGeneratorAgent(BaseAgent):
    def __init__(self, database_url: str = None):
        super().__init__("content_generator_agent", [
            "knowledge_graph", "content_strategy", "language_models"
        ])
        self.vector_search_tool = VectorSearchTool()
        self.video_search_tool = VideoSearchTool()
        self.llm_generate_tool = LLMGenerateTool()
        self.code_generation_tool = CodeGenerationTool()
        self.question_extract_tool = QuestionExtractTool(database_url=database_url)
        self.template_render_tool = TemplateRenderTool()
        self.format_converter_tool = FormatConverterTool()
        self.code_executor_tool = CodeExecutorTool()
        self.load_tools([
            self.vector_search_tool, self.video_search_tool, self.llm_generate_tool,
            self.code_generation_tool, self.question_extract_tool,
            self.template_render_tool, self.format_converter_tool,
            self.code_executor_tool,
        ])

    async def generate_learning_package(
        self, session_id: str, stage_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        block_id = stage_info.get("block_id", "")
        block_name = stage_info.get("block_name", "")
        subtopic_id = stage_info.get("subtopic_id", "")
        gap = stage_info.get("gap", {})

        related_docs = await self.vector_search_tool.search_by_block(block_id, subtopic_id)
        tutorial = await self.llm_generate_tool.generate_tutorial(block_id, gap, related_docs)
        code_examples = await self.code_generation_tool.generate_code(block_id, stage_info)
        questions = await self.question_extract_tool.extract_questions(tutorial, stage_info)

        sources = _generate_sources(block_id, related_docs)

        # 视频资源检索
        try:
            video_results = await self.video_search_tool.search_by_block(
                block_id=block_id, block_name=block_name, top_k=5
            )
            for v in video_results:
                sources.append(Source(
                    source_id=v["video_id"],
                    source_type="video",
                    title=v["title"],
                    locator=v["url"],
                    content_ref=v.get("topic", ""),
                    metadata_json={
                        "topic": v.get("topic", ""),
                        "duration_hours": v.get("duration_hours", 0),
                        "block_id": v.get("block_id", ""),
                    },
                ))
        except Exception:
            pass

        claims = _extract_claims_from_tutorial(tutorial, sources)

        return {
            "tutorial": tutorial,
            "code_examples": code_examples,
            "questions": questions,
            "format": "markdown",
            "sources": [s.model_dump(mode="json") for s in sources],
            "claims": [c.model_dump(mode="json") for c in claims],
        }

    async def reply_to_review(
        self,
        session_id: str,
        issues: List[Dict[str, Any]],
        package: Dict[str, Any],
        debate_history: Optional[List[Dict]] = None,
    ) -> List[Dict[str, Any]]:
        """对 Agent4 的每条 issue 逐条回复。

        边界规则（协议强制）：
        - must_fix=true 的 issue：只能 accepted 或 partially_accepted，不能 rejected
        - rejected 必须带 source_refs 反证，否则无效
        """
        replies: List[Dict[str, Any]] = []
        sources = package.get("sources", [])

        for issue in issues:
            must_fix = issue.get("must_fix", False)
            canonical_id = issue.get("canonical_error_id", "")

            if must_fix:
                # must_fix=true 不能 rejected，只能 accepted 或 partially_accepted
                disposition = "partially_accepted"
                reason = f"[{issue.get('error_type', '')}] 根据审核建议修正: {issue.get('suggestion') or issue.get('description', '')}"
                source_refs = issue.get("source_refs") or [
                    s.get("source_id") for s in sources[:1] if isinstance(s, dict)
                ] or [s.source_id for s in sources[:1]]
                patched_sections = [issue.get("location") or "tutorial:sec_0"]
            else:
                disposition = "accepted"
                reason = f"接受审核意见: {issue.get('description', '')}"
                source_refs = issue.get("source_refs") or []
                patched_sections = [issue.get("location") or "tutorial:sec_0"]

            replies.append({
                "canonical_error_id": canonical_id,
                "disposition": disposition,
                "reason": reason,
                "source_refs": source_refs,
                "patched_sections": patched_sections,
            })

        return replies

    async def modify_package(
        self,
        session_id: str,
        package: Dict[str, Any],
        errors: Dict[str, Any],
        accepted_issues: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """修改学习包内容，只处理 accepted/partially_accepted 的 issue.

        - 内容改写 → claim status="revised"
        - 内容删除 → claim status="dropped"
        - 新增的 claim → status="active"
        - 保留 review_trace_id 不变（同一 trace 内迭代）
        - version 递增
        """
        tech_errors = errors.get("tech", [])
        quality_errors = errors.get("quality", [])

        def _get_attr(item: Any, key: str, default: Any = None) -> Any:
            if hasattr(item, key):
                return getattr(item, key)
            if isinstance(item, dict):
                return item.get(key, default)
            return default

        if accepted_issues is not None:
            accepted_canonical_ids = {
                _get_attr(i, "canonical_error_id") for i in accepted_issues
            }
            tech_errors = [
                e for e in tech_errors
                if _get_attr(e, "canonical_error_id") in accepted_canonical_ids
                or not _get_attr(e, "must_fix", False)
            ]

        if tech_errors and "tutorial" in package:
            package["tutorial"] = await self.llm_generate_tool.fix_tutorial(
                package["tutorial"], tech_errors
            )

        if quality_errors and "tutorial" in package:
            package["tutorial"] = await self.llm_generate_tool.adjust_tutorial_difficulty(
                package["tutorial"], quality_errors
            )

        # 更新 claims 的 status
        if "claims" in package and isinstance(package["claims"], list):
            accepted_claim_ids = {
                _get_attr(i, "claim_id") for i in (accepted_issues or [])
                if _get_attr(i, "claim_id")
            }
            for claim in package["claims"]:
                claim_id = _get_attr(claim, "claim_id")
                if claim_id in accepted_claim_ids:
                    # 内容改写 → revised
                    if isinstance(claim, dict):
                        claim["status"] = "revised"
                    elif hasattr(claim, "status"):
                        claim.status = "revised"
                elif any(
                    _get_attr(e, "claim_id") == claim_id
                    for e in tech_errors
                ):
                    # 内容删除 → dropped
                    if isinstance(claim, dict):
                        claim["status"] = "dropped"
                    elif hasattr(claim, "status"):
                        claim.status = "dropped"

        # version 递增
        if "version" in package:
            package["version"] = package.get("version", 1) + 1

        return package

    async def answer_question(self, session_id: str, question: str,
                              block_id: str = "", subtopic_id: str = "", qa_history: list = None,
                              profile_summary: str = "") -> dict:
        """回答用户问题：先分类（technical/daily），再走对应流程。

        返回 dict：{"answer": str, "question_type": "technical" | "daily"}
        - 技术类：问题分析 + 本阶段相关知识文档检索 + 历史 memory + LLM 综合
        - 日常类：历史对话 memory + 画像 + LLM 直接回答
        """
        # 问题分类
        question_type = await self.llm_generate_tool.classify_question(question)

        if question_type == "daily":
            # 日常类：只用历史对话 memory + 画像，跳过 Agent4
            answer = await self.llm_generate_tool.generate_daily_answer(
                question, "", qa_history or [], profile_summary
            )
            return {"answer": answer, "question_type": "daily"}

        # 技术类：问题分析 + 本阶段相关知识文档检索 + 历史 memory
        code_snippets = self._extract_code_from_question(question)
        if code_snippets:
            answer = await self._handle_code_correction(session_id, question, code_snippets)
            if qa_history or profile_summary:
                memory_ctx = self._build_memory_text(qa_history, profile_summary)
                if memory_ctx:
                    answer = f"{answer}\n\n{memory_ctx}"
            return {"answer": answer, "question_type": "technical"}

        # 本阶段相关知识文档检索（用 block_id 过滤）
        if block_id:
            related_docs = await self.vector_search_tool.search_by_block(block_id, subtopic_id)
        else:
            related_docs = await self.vector_search_tool.search_by_query(question, top_k=3)

        answer = await self.llm_generate_tool.generate_answer(
            question, related_docs,
            tutorial="", qa_history=qa_history, profile_summary=profile_summary
        )
        return {"answer": answer, "question_type": "technical"}

    @staticmethod
    def _build_memory_text(qa_history: list, profile_summary: str) -> str:
        """构建历史 memory 上下文文本。"""
        parts = []
        if qa_history:
            recent = qa_history[-3:]
            history_text = "\n".join([
                f"- 问：{item.get('question', '')[:80]}\n  答：{item.get('answer', '')[:100]}"
                for item in recent if isinstance(item, dict)
            ])
            if history_text:
                parts.append(f"💬 最近对话记录：\n{history_text}")
        if profile_summary:
            parts.append(f"👤 学习画像：{profile_summary}")
        return "\n\n".join(parts)

    def _extract_code_from_question(self, question: str) -> List[str]:
        patterns = [
            r'```(\w*)\n(.*?)```',
            r'```\n(.*?)```',
            r'`([^`]+)`',
        ]
        code_snippets = []
        for pattern in patterns:
            matches = re.findall(pattern, question, re.DOTALL)
            for match in matches:
                if isinstance(match, tuple):
                    code = match[1].strip()
                else:
                    code = match.strip()
                if code and len(code) > 5:
                    code_snippets.append(code)
        return code_snippets

    async def _handle_code_correction(
        self, session_id: str, question: str, code_snippets: List[str]
    ) -> str:
        execution_results = []
        for code in code_snippets[:3]:
            result = await self.code_executor_tool.execute_code(code)
            execution_results.append({
                "code": code,
                "output": result.get("output", ""),
                "error": result.get("error", ""),
                "return_code": result.get("return_code", 0),
                "timeout": result.get("timeout", False),
            })

        execution_context = "\n\n".join([
            f"代码片段{i+1}执行结果:\n输出: {r['output']}\n错误: {r['error']}\n返回码: {r['return_code']}"
            for i, r in enumerate(execution_results)
        ])

        prompt = f"""你是一个专业的代码纠错老师。请按照以下结构回答：

## 错误分析
分析代码中的错误原因

## 修改步骤
分步骤说明修改方案

## 修改后完整代码
提供修改后的完整代码

## 修改说明
总结修改要点和学习建议

用户问题：{question}

代码执行结果：
{execution_context}

请先肯定用户的尝试，再指出问题，最后给出修改方案。"""

        return await self.llm_generate_tool.generate_answer(prompt, [])

    async def modify_answer(
        self,
        session_id: str,
        answer: str,
        errors: Dict[str, Any],
        question: str = "",
    ) -> str:
        tech_errors = errors.get("tech", [])
        quality_errors = errors.get("quality", [])
        if tech_errors or quality_errors:
            answer = await self.llm_generate_tool._modify_answer(
                answer, tech_errors, quality_errors, question
            )
        return answer
