from typing import Dict, Any, List, Set
from agents.base_agent import BaseAgent
from tools.agent1_tools.extract_skills_tool import ExtractSkillsTool
from tools.agent1_tools.calculate_confidence_tool import CalculateConfidenceTool
from tools.agent1_tools.graph_query_tool import GraphQueryTool
from tools.agent1_tools.knowledge_block_query_tool import KnowledgeBlockQueryTool
from tools.agent1_tools.knowledge_catalog_tool import KnowledgeCatalogTool
from tools.agent1_tools.stage_generator_tool import StageGeneratorTool


class DiagnosticAgent(BaseAgent):
    def __init__(self):
        super().__init__("diagnostic_agent", ["learner_profiling", "knowledge_graph_mapping", "stage_planning"])
        self.extract_skills_tool = ExtractSkillsTool()
        self.calculate_confidence_tool = CalculateConfidenceTool()
        self.graph_query_tool = GraphQueryTool()
        self.knowledge_block_query_tool = KnowledgeBlockQueryTool()
        self.knowledge_catalog_tool = KnowledgeCatalogTool()
        self.stage_generator_tool = StageGeneratorTool()
        self.load_tools([
            self.extract_skills_tool,
            self.calculate_confidence_tool,
            self.graph_query_tool,
            self.knowledge_block_query_tool,
            self.knowledge_catalog_tool,
            self.stage_generator_tool,
        ])

    @staticmethod
    def _parse_block_list(raw: str) -> List[str]:
        """Parse existing_skills / expected_target_blocks from free-text.

        Accepted separators: Chinese/English comma, slash, pipe, semicolon,
        space, newline, Chinese enumeration comma "、".  Returns an ordered
        de-duplicated list of block IDs / skill tags.
        """
        if not raw:
            return []
        if isinstance(raw, list):
            parts = [str(p).strip() for p in raw]
            return [p for p in dict.fromkeys(parts) if p]
        text = str(raw).strip()
        # Collapse known zero-foundation placeholders to empty.
        if text.casefold() in {"none", "无", "无基础", "零基础", "0基础", "未提供", "null", "[]"}:
            return []
        parts = __import__("re").split(r"[、,，/|；;\s\n]+", text)
        return [p for p in dict.fromkeys(part.strip(" \t\"'()（）") for part in parts) if p]

    def _resolve_mastered_block_ids(self, declared_tags: List[str]) -> Set[str]:
        """Resolve a mixed list (KB-ids + skill names + subtopic ids) to a
        stable set of catalog block IDs.
        """
        block_ids: Set[str] = set()
        for tag in declared_tags:
            if not tag:
                continue
            # KB-ID style shortcut.
            if tag.upper().startswith("KB-") and self.knowledge_catalog_tool.get_block(tag):
                block_ids.add(tag)
                continue
            resolved = self.knowledge_catalog_tool.resolve_block(tag)
            if resolved and resolved.get("block_id"):
                block_ids.add(str(resolved["block_id"]))
                continue
            candidates = self.knowledge_catalog_tool.search(tag, top_k=1)
            if candidates and candidates[0].get("matched_terms"):
                block_ids.add(str(candidates[0].get("block_id", "")))
        return {bid for bid in block_ids if bid}

    async def generate_profile(self, session_id: str, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        user_input = raw_data.get("user_input", "")

        # ── T1 根因A 修复位点：解析 existing_skills 声明 并落 mastered_block_ids ──
        declared_skills = self._parse_block_list(
            raw_data.get("existing_skills")
            or raw_data.get("mastered_skills")
            or raw_data.get("skill_tags")
            or ""
        )
        declared_mastered_block_ids: Set[str] = self._resolve_mastered_block_ids(declared_skills)

        skill_tags = await self.extract_skills_tool.extract_skills(user_input)
        knowledge_gaps = await self.extract_skills_tool.extract_gaps(user_input)

        # Seed each gap with catalog metadata we will need for filtering
        # and topological gap selection below.
        for gap in knowledge_gaps:
            prerequisites = await self.graph_query_tool.query_prerequisites(gap.get("skill", ""))
            gap["prerequisites"] = prerequisites
            # Resolve block_id when the gap lacks it (matches deterministic path).
            if not gap.get("block_id"):
                resolved = self.knowledge_catalog_tool.resolve_block(gap.get("skill", "")) or {}
                if resolved.get("block_id"):
                    gap["block_id"] = str(resolved["block_id"])
                    gap.setdefault("block_name", str(resolved.get("block_name", gap.get("skill", ""))))
            # Block-level prerequisite names used by downstream summary/review.
            if declared_mastered_block_ids and isinstance(gap.get("prerequisites"), list):
                mastered_names: Set[str] = set()
                for pid in declared_mastered_block_ids:
                    blk = self.knowledge_catalog_tool.get_block(pid) or {}
                    if blk.get("block_name"):
                        mastered_names.add(str(blk["block_name"]))
                gap["prerequisites"] = [
                    pre for pre in gap["prerequisites"]
                    if isinstance(pre, str) and pre not in mastered_names
                ]

        confidence = await self.calculate_confidence_tool.calculate(user_input, skill_tags, knowledge_gaps)

        # ── T1 根因A 修复位点：用用户声明的已掌握块过滤 knowledge_gaps ──
        # Rules:
        #  1. gap 的 block_id 在 declared_mastered_block_ids 中 → 整 gap 标记已掌握（不重复教学）
        #  2. 声明过的 skill 名称追加到 skill_tags，保证前端/审查时"已掌握"可观察
        if declared_mastered_block_ids:
            for gap in knowledge_gaps:
                gap_bid = str(gap.get("block_id", "") or "")
                if gap_bid in declared_mastered_block_ids:
                    gap["status"] = "已掌握"
                    gap["progress"] = 1.0
            for bid in declared_mastered_block_ids:
                blk = self.knowledge_catalog_tool.get_block(bid) or {}
                label = str(blk.get("block_name") or bid)
                if label and label not in skill_tags:
                    skill_tags.append(label)
            # Preserve declared IDs in profile (used by downstream alignments
            # in harness.py and by future stages when reconstructing mastery).
            declared_mastered_list = sorted(declared_mastered_block_ids)
        else:
            declared_mastered_list = []

        profile = {
            "session_id": session_id,
            "skill_tags": skill_tags,
            "mastered_block_ids": declared_mastered_list,
            "knowledge_gaps": knowledge_gaps,
            "goal": raw_data.get("goal", ""),
            "daily_hours": raw_data.get("daily_hours", 2.0),
            "learning_style": raw_data.get("learning_style", "mixed"),
            "learning_rhythm": raw_data.get("learning_rhythm", "medium"),
            "confidence_index": raw_data.get("confidence_index", 0.5),
            "confusion_markers": [],
            "confidence": confidence,
        }

        return profile

    async def update_profile(self, session_id: str, profile: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in updates.items():
            if key in profile:
                profile[key] = value
        
        return profile

    async def generate_stage(self, session_id: str, profile: Dict[str, Any], stage_num: int = 1) -> Dict[str, Any]:
        """生成下一学习阶段计划（原Agent2功能，已合并至学情诊断与阶段规划 Agent）。"""
        knowledge_gaps = profile.get("knowledge_gaps", [])
        unmastered_gaps = [g for g in knowledge_gaps if g.get("status") != "已掌握" and float(g.get("progress", 0.0)) < 1.0]

        if not unmastered_gaps:
            raise ValueError("No unmastered knowledge gap remains")

        # ── T1 根因B 修复位点：选 gap 改成拓扑最小优先 + expected_target 平局优先 ──
        #
        # 1) mastered_set = 用户声明已掌握 block_id（existing_skills 解析出来的）
        #    ∪ 当前 profile.knowledge_gaps 里 status=="已掌握" 的 block_id
        # 2) 对每个 unmastered_gap，数"它自身 prerequisites 块名能解析出的 block_id
        #    中还有多少不在 mastered_set 里"——这个就是未掌握前置数（unmet_prereq_count）
        # 3) 排序优先级：unmet_prereq_count ASC
        #                 → expected_target_blocks 命中优先（平局按用户写的目标里出现的顺序）
        #                 → gap 在 knowledge_gaps 中的原顺序（稳定兜底）
        declared_mastered: Set[str] = set(
            str(b) for b in (profile.get("mastered_block_ids") or []) if b
        )
        for g in knowledge_gaps:
            if g.get("status") == "已掌握" or float(g.get("progress", 0.0)) >= 1.0:
                bid = str(g.get("block_id", "") or "")
                if bid:
                    declared_mastered.add(bid)

        expected_target_raw = (
            profile.get("expected_target_blocks")
            or profile.get("target_blocks")
            or ""
        )
        expected_targets = [
            t for t in self._parse_block_list(str(expected_target_raw)) if t
        ]
        expected_idx = {tid: i for i, tid in enumerate(expected_targets)}

        def _resolve_prereq_block_ids(pre_list) -> Set[str]:
            ids: Set[str] = set()
            if not pre_list:
                return ids
            for item in pre_list:
                if not isinstance(item, str) or not item:
                    continue
                # Already looks like KB-xxx.
                if item.upper().startswith("KB-") and self.knowledge_catalog_tool.get_block(item):
                    ids.add(item)
                    continue
                resolved = self.knowledge_catalog_tool.resolve_block(item) or {}
                if resolved.get("block_id"):
                    ids.add(str(resolved["block_id"]))
            return ids

        def _expected_rank(gap_block_id: str) -> int:
            if gap_block_id in expected_idx:
                return expected_idx[gap_block_id]
            if not expected_targets:
                return 0
            # Try match by canonical skill label / block name against the
            # expected_target list, which may contain user-written names.
            blk = self.knowledge_catalog_tool.get_block(gap_block_id) or {}
            label = str(blk.get("block_name", "")).casefold()
            labels = {label} if label else set()
            labels.update({str(sk).casefold() for sk in expected_targets})
            for i, tgt in enumerate(expected_targets):
                t_fold = tgt.casefold()
                if t_fold and (t_fold == label or t_fold in labels):
                    return i
            return len(expected_targets)  # Not a target = lowest priority

        orig_rank = {id(g): idx for idx, g in enumerate(unmastered_gaps)}
        ranked = sorted(
            unmastered_gaps,
            key=lambda g: (
                sum(
                    1 for pre in _resolve_prereq_block_ids(g.get("prerequisites") or [])
                    if pre and pre not in declared_mastered
                ),
                _expected_rank(str(g.get("block_id", "") or "")),
                orig_rank.get(id(g), 0),
            ),
        )
        gap = ranked[0]
        skill_name = gap.get("skill", "")

        block_info = {}
        try:
            block_info = await self.knowledge_block_query_tool.query_block(skill_name)
        except Exception:
            block_info = {}

        generated_stage = {}
        try:
            generated_stage = await self.stage_generator_tool.generate_stage(
                session_id, gap, block_info, profile.get("goal", ""), stage_num,
                profile=profile
            )
        except Exception:
            generated_stage = {}

        # 从 block_info 中提取本阶段目标 subtopic 列表（支持多知识点分页）
        block_subtopics = block_info.get("subtopics", []) or []
        # gap 中记录的"目标 subtopic"（用户未掌握、本阶段需要学习的）
        target_ids = gap.get("target_subtopic_ids", []) or []
        target_subtopics = gap.get("target_subtopics", []) or []

        subtopic_ids: list[str] = []
        subtopic_names: list[str] = []

        if target_ids:
            # 优先按 gap 的 target_subtopic_ids 顺序，从 block_subtopics 匹配名称
            id_to_name = {
                str(s.get("subtopic_id", "")): s.get("subtopic_name", "") or s.get("name", "") or s.get("title", "")
                for s in block_subtopics if isinstance(s, dict)
            }
            for sid in target_ids:
                sid_str = str(sid)
                if sid_str and sid_str not in subtopic_ids:
                    subtopic_ids.append(sid_str)
                    subtopic_names.append(id_to_name.get(sid_str, sid_str))
        elif target_subtopics:
            # 回退到 target_subtopics 列表
            for sub in target_subtopics:
                if isinstance(sub, dict):
                    sid = str(sub.get("subtopic_id", "") or sub.get("id", ""))
                    name = sub.get("subtopic_name", "") or sub.get("name", "") or sub.get("title", "")
                    if sid and sid not in subtopic_ids:
                        subtopic_ids.append(sid)
                        subtopic_names.append(name)
                elif isinstance(sub, str):
                    if sub not in subtopic_names:
                        subtopic_names.append(sub)
                        subtopic_ids.append("")

        # 如果 target 为空，回退到 block 的全部 subtopics（首个阶段兜底）
        if not subtopic_ids and not subtopic_names and block_subtopics:
            for sub in block_subtopics:
                if isinstance(sub, dict):
                    sid = str(sub.get("subtopic_id", "") or sub.get("id", ""))
                    name = sub.get("subtopic_name", "") or sub.get("name", "") or sub.get("title", "")
                    if sid:
                        subtopic_ids.append(sid)
                        subtopic_names.append(name)

        # 单值字段取首个，向后兼容
        subtopic_id = subtopic_ids[0] if subtopic_ids else ""
        subtopic_name = subtopic_names[0] if subtopic_names else ""

        return {
            "stage_id": generated_stage.get("stage_id", ""),
            "stage_num": stage_num,
            "block_id": block_info.get("block_id", ""),
            "block_name": block_info.get("block_name", skill_name),
            "subtopic_id": subtopic_id,
            "subtopic_name": subtopic_name,
            "subtopic_ids": subtopic_ids,
            "subtopic_names": subtopic_names,
            "estimated_hours": generated_stage.get("estimated_hours", 2.0),
            "gap_skill": skill_name,
            "prerequisites": block_info.get("prerequisites", []),
            "status": "pending",
            "title": generated_stage.get("title"),
            "description": generated_stage.get("description"),
            "objectives": generated_stage.get("objectives", []),
            "difficulty": generated_stage.get("difficulty"),
            "recommended_daily_hours": generated_stage.get("recommended_daily_hours"),
            "suggested_study_days": generated_stage.get("suggested_study_days"),
            "learning_style_notes": generated_stage.get("learning_style_notes"),
            "hardware_notes": generated_stage.get("hardware_notes"),
            "personalization_summary": generated_stage.get("personalization_summary"),
        }
