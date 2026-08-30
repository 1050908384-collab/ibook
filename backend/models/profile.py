# -*- coding: utf-8 -*-
import re
from typing import List, Optional, Literal

from pydantic import BaseModel, Field

from models.device_spec import resolve_device


# ── 学习风格权威枚举（与前端 FoundationPickerModal.styleOptions 一一对应）
#    任何新增/改名必须同步前端 styleOptions 和此处，否则默认回落到 MIXED。
#    单一真相源（Single Source of Truth）：
#      - UserProfile.learning_style
#      - FoundationSubmitRequest.learning_style
#      - _extract_learning_style() 返回值
#      - stage_generator_tool style_map keys
#      - Agent3 / handle_question 的风格判断
LEARNING_STYLE_THEORY: Literal["theory"] = "theory"
LEARNING_STYLE_PRACTICE: Literal["practice"] = "practice"
LEARNING_STYLE_MIXED: Literal["mixed"] = "mixed"
LEARNING_STYLES = (LEARNING_STYLE_THEORY, LEARNING_STYLE_PRACTICE, LEARNING_STYLE_MIXED)
LearningStyle = Literal["theory", "practice", "mixed"]

# 自然语言关键词 → 枚举值。保持前后端一致：只识别三类风格。
LEARNING_STYLE_KEYWORDS: dict[str, tuple[str, ...]] = {
    LEARNING_STYLE_THEORY:   ("theory",   "理论", "原理", "概念", "推导"),
    LEARNING_STYLE_PRACTICE: ("practice", "实操", "实践", "动手", "代码", "实验", "部署"),
    LEARNING_STYLE_MIXED:    ("mixed",    "混合", "结合", "均衡"),
}
LEARNING_STYLE_DEFAULT: LearningStyle = LEARNING_STYLE_MIXED


def normalize_learning_style(value: object) -> LearningStyle:
    """把任意输入安全地收敛到权威枚举三值之一。
    非法值一律回落到 MIXED，保证下游 style_map/get 永远命中不漂移。"""
    if value is None:
        return LEARNING_STYLE_DEFAULT
    text = str(value).strip().lower()
    if not text:
        return LEARNING_STYLE_DEFAULT
    # 1) 精确匹配枚举字面
    if text in LEARNING_STYLES:
        return text  # type: ignore[return-value]
    # 2) 组合优先：同时命中 theory+practice → mixed（如"理论结合实操"）
    has_t = any(kw in text for kw in LEARNING_STYLE_KEYWORDS[LEARNING_STYLE_THEORY])
    has_p = any(kw in text for kw in LEARNING_STYLE_KEYWORDS[LEARNING_STYLE_PRACTICE])
    has_m = any(kw in text for kw in LEARNING_STYLE_KEYWORDS[LEARNING_STYLE_MIXED])
    if has_m or (has_t and has_p):
        return LEARNING_STYLE_MIXED
    if has_p:
        return LEARNING_STYLE_PRACTICE
    if has_t:
        return LEARNING_STYLE_THEORY
    return LEARNING_STYLE_DEFAULT


class KnowledgeGap(BaseModel):
    skill: str
    block_id: str = ""
    block_name: str = ""
    target_subtopic_ids: List[str] = Field(default_factory=list)
    mapping_confidence: float = 0.0
    level: str = "beginner"
    status: str = "未掌握"
    progress: float = 0.0
    prerequisites: List[str] = Field(default_factory=list)


class UserProfile(BaseModel):
    session_id: str
    user_id: str = "local-user"
    grade: str = ""
    skill_tags: List[str] = Field(default_factory=list)
    mastered_block_ids: List[str] = Field(default_factory=list)
    expected_target_blocks: List[str] = Field(default_factory=list)
    goal_scope: str = "focused"
    target_block_ids: List[str] = Field(default_factory=list)
    target_subtopic_ids: List[str] = Field(default_factory=list)
    planned_block_order: List[str] = Field(default_factory=list)
    active_stage_block_id: str = ""
    foundation_confirmed: bool = False
    knowledge_gaps: List[KnowledgeGap] = Field(default_factory=list)
    goal: str = ""
    daily_hours: float = 2.0
    learning_style: LearningStyle = LEARNING_STYLE_DEFAULT
    learning_rhythm: str = "medium"
    confidence_index: float = 0.5
    confusion_markers: List[str] = Field(default_factory=list)
    confidence: float = 0.7
    version: int = 1


class PracticeProfile(BaseModel):
    """Execution context supplied by orchestration to 个性化学习资源生成 Agent only.

    ``hardware`` keeps the user-facing free-text device string for backward
    compatibility. The structured ``gpu_model`` / ``vram_gb`` /
    ``gpu_architecture`` / ``device_type`` fields are auto-derived from that
    text (and auto-completed from a device-model lookup) so 个性化学习资源生成 Agent can reason
    about VRAM without re-parsing the string every time.
    """
    hardware: str = ""
    device_type: str = ""
    gpu_model: str = ""
    vram_gb: Optional[int] = None
    gpu_architecture: str = ""

    @classmethod
    def from_user_input(cls, text: str) -> "PracticeProfile":
        labeled_match = re.search(
            r"(?:我的\s*)?(?:设备条件|硬件条件|显卡|hardware|device)\s*(?:是|为|:|：)\s*([^\n，。；;]+)",
            text,
            re.IGNORECASE,
        )
        if labeled_match:
            return cls.from_hardware_text(labeled_match.group(1).strip())

        # Initial free-text diagnostics commonly describe a GPU without a
        # label, for example "I have an RTX 4090 with 24GB VRAM".
        gpu_match = re.search(
            r"\b(?:nvidia\s+)?(?:rtx|gtx)\s*\d{3,4}(?:\s+with\s+\d+\s*gb(?:\s+vram)?)?\b",
            text,
            re.IGNORECASE,
        )
        if gpu_match:
            return cls.from_hardware_text(gpu_match.group(0).strip())
        return cls()

    @classmethod
    def from_hardware_text(cls, hardware: str) -> "PracticeProfile":
        """Build a profile from a device string, auto-completing GPU specs.

        Given just a model (e.g. "RTX 4090") this fills in VRAM/architecture
        from the device-spec lookup; an explicitly stated VRAM always wins.
        """
        hardware = (hardware or "").strip()
        if not hardware:
            return cls()
        spec = resolve_device(hardware)
        return cls(
            hardware=hardware,
            device_type=spec.device_type,
            gpu_model=spec.gpu_model,
            vram_gb=spec.vram_gb,
            gpu_architecture=spec.gpu_architecture,
        )
