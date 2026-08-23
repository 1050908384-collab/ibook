# -*- coding: utf-8 -*-
import re
from typing import List, Optional

from pydantic import BaseModel, Field

from models.device_spec import resolve_device


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
    foundation_confirmed: bool = False
    knowledge_gaps: List[KnowledgeGap] = Field(default_factory=list)
    goal: str = ""
    daily_hours: float = 2.0
    learning_style: str = "mixed"
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
