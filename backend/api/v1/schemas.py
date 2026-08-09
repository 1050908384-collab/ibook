"""Shared public request and response contracts for the v1 frontend API."""
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class PublicQuestion(BaseModel):
    model_config = ConfigDict(extra="allow")

    question_id: str
    question_text: str
    question_type: Optional[str] = None
    options: Optional[Any] = None
    difficulty: Optional[str] = None
    hint: Optional[str] = None


class LearningResource(BaseModel):
    resource_id: str
    resource_type: Literal["video", "article", "documentation", "code"]
    title: str
    url: str = ""
    platform: str = ""
    description: str = ""
    recommendation_level: str = "supplementary"
    role: str = "supplementary"
    order: int = 1


class PublicLearningPackage(BaseModel):
    stage_id: Optional[str] = None
    stage_num: int = 1
    block_name: str = ""
    subtopic_name: str = ""
    estimated_hours: float = 0
    tutorial: Optional[str] = None
    code_examples: list[dict[str, Any]] = Field(default_factory=list)
    learning_resources: list[LearningResource] = Field(default_factory=list)
    questions: list[PublicQuestion] = Field(default_factory=list)
    review_summary: dict[str, Any] = Field(default_factory=dict)
    recommended_daily_hours: Optional[float] = None
    suggested_study_days: Optional[int] = None
    learning_style_notes: Optional[str] = None
    hardware_notes: Optional[str] = None
    personalization_summary: Optional[str] = None


class ProfileResponse(BaseModel):
    profile_id: str
    session_id: str
    goal: str = ""
    grade: str = ""
    skill_tags: list[str] = Field(default_factory=list)
    daily_hours: float = 2.0
    learning_style: str = "mixed"
    hardware: str = ""
    knowledge_gaps: list[dict[str, Any]] = Field(default_factory=list)
    learning_rhythm: str = "medium"
    confidence_index: float = 0.5
    confusion_markers: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    version: int = 1
    confirmed: bool = False
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class ProfileUpdateResponse(BaseModel):
    profile_id: str
    session_id: str
    updated_fields: dict[str, Any] = Field(default_factory=dict)
    profile_modified: bool = False
    message: str = ""


def to_public_question(question: dict[str, Any]) -> dict[str, Any]:
    """Remove fields that must not reach clients before assessment submission."""
    internal_fields = {
        "correct_answer", "explanation", "expected_output", "reference_answer",
        "grading_mode", "grading_rule", "judge_rule",
    }
    return {key: value for key, value in question.items() if key not in internal_fields}


def to_public_learning_package(
    package: Optional[dict[str, Any]],
    stage: Optional[dict[str, Any]] = None,
    learning_resources: Optional[list[dict[str, Any]]] = None,
) -> Optional[dict[str, Any]]:
    """Build the single safe learning-package representation used by all v1 APIs."""
    if not package:
        return None

    stage = stage or {}
    resources = learning_resources if learning_resources is not None else package.get("learning_resources", [])
    if not resources:
        # Agent3 persists videos under recommended_videos.  Normalise them at
        # the public API boundary so a restored session matches a newly
        # generated package from /stages/current/package.
        resources = [
            {
                "resource_id": video.get("resource_id", f"video_{index}"),
                "resource_type": "video",
                "title": video.get("title", ""),
                "url": video.get("url", video.get("canonical_url", "")),
                "platform": video.get("platform", ""),
                "description": video.get("description", video.get("content_summary", "")),
                "recommendation_level": video.get("recommendation_level", video.get("role", "supplementary")),
                "role": video.get("role", "supplementary"),
                "order": video.get("order", video.get("sequence_index", index + 1)),
            }
            for index, video in enumerate(package.get("recommended_videos", []), start=1)
            if isinstance(video, dict)
        ]
    # Fallback stages can intentionally omit a catalog subtopic.  The public
    # contract still requires a displayable string, so use the diagnosed gap
    # or block title instead of passing None into Pydantic validation.
    public_subtopic_name = (
        stage.get("subtopic_name")
        or package.get("subtopic_name")
        or stage.get("gap_skill")
        or package.get("gap_skill")
        or stage.get("block_name")
        or package.get("block_name", "")
    )

    return PublicLearningPackage(
        stage_id=stage.get("stage_id", package.get("stage_id")),
        stage_num=stage.get("stage_num", 1),
        block_name=stage.get("block_name", package.get("block_name", "")),
        subtopic_name=public_subtopic_name,
        estimated_hours=stage.get("estimated_hours", 0),
        tutorial=package.get("tutorial"),
        code_examples=package.get("code_examples", []),
        learning_resources=resources or [],
        questions=[to_public_question(question) for question in package.get("questions", [])],
        review_summary={
            "status": package.get("status", "draft"),
            "reviewed_by": "Agent4",
            "review_notes": package.get("review_notes", ""),
        },
        recommended_daily_hours=stage.get("recommended_daily_hours"),
        suggested_study_days=stage.get("suggested_study_days"),
        learning_style_notes=stage.get("learning_style_notes"),
        hardware_notes=stage.get("hardware_notes"),
        personalization_summary=stage.get("personalization_summary"),
    ).model_dump(mode="json")
