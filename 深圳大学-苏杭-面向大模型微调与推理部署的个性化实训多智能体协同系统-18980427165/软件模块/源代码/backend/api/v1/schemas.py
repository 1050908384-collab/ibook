"""Shared public request and response contracts for the v1 frontend API."""
import logging
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


class PublicQuestion(BaseModel):
    model_config = ConfigDict(extra="allow")

    question_id: str
    question_text: str
    question_type: Optional[str] = None
    options: Optional[Any] = None
    difficulty: Optional[str] = None
    hint: Optional[str] = None


def _norm_str(value: Any) -> str:
    """把 None / 非字符串安全地规范化成空字符串，避免 Pydantic string_type 校验报错。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


class LearningResource(BaseModel):
    model_config = ConfigDict(extra="allow")

    resource_id: str
    resource_type: Literal["video", "article", "documentation", "code"]
    title: str
    url: str = ""
    platform: str = ""
    description: str = ""
    recommendation_level: str = "supplementary"
    role: str = "supplementary"
    order: int = 1
    subtopic_id: str = ""


class PublicLearningPackage(BaseModel):
    stage_id: Optional[str] = None
    stage_num: int = 1
    block_name: str = ""
    subtopic_name: str = ""
    subtopic_ids: list[str] = Field(default_factory=list)
    subtopic_names: list[str] = Field(default_factory=list)
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
    device_type: str = ""
    gpu_model: str = ""
    vram_gb: Optional[int] = None
    gpu_architecture: str = ""
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

    # 1. 动态注入本地学习资料的"多参考链接"文章资源（优先于 snapshot 里的旧 learning_resources）。
    #    这样旧会话/老阶段也能看到每知识点多条文章网址，无需重新生成。
    try:
        from api.v1.learning_resources import build_stage_document_resources

        dynamic_articles = build_stage_document_resources(stage)
    except Exception as exc:  # noqa: BLE001 - 容错：失败时退回 snapshot 数据
        logger.warning("to_public_learning_package: build_stage_document_resources failed: %s", exc)
        dynamic_articles = []
    logger.info(
        "to_public_learning_package: stage.block_id=%s stage.subtopic_ids=%s dynamic_articles=%d",
        stage.get("block_id"), stage.get("subtopic_ids"), len(dynamic_articles),
    )
    resources = list(dynamic_articles)
    dynamic_ids = {
        resource.get("resource_id")
        for resource in resources
        if isinstance(resource, dict) and resource.get("resource_id")
    }
    dynamic_urls = {
        str(resource.get("url", "")).strip()
        for resource in resources
        if isinstance(resource, dict) and str(resource.get("url", "")).startswith(("http://", "https://"))
    }
    dynamic_subtopics = {
        str(resource.get("subtopic_id", ""))
        for resource in resources
        if isinstance(resource, dict) and resource.get("subtopic_id")
    }

    # 2. 合并快照里原有的 learning_resources（视频类保留，article/documentation 类若与动态资源 URL 或
    #    subtopic_id 重叠则跳过，避免重复）。
    legacy_resources = list(
        learning_resources if learning_resources is not None else package.get("learning_resources", [])
    )
    for resource in legacy_resources:
        if not isinstance(resource, dict):
            continue
        rtype = str(resource.get("resource_type") or "")
        rid = resource.get("resource_id")
        if rid and rid in dynamic_ids:
            continue
        if rtype in {"article", "documentation", "code", ""}:
            rurl = str(resource.get("url", resource.get("canonical_url", "")) or "").strip()
            rsub = str(resource.get("subtopic_id") or "")
            # 与动态 URL 相同 → 重复，跳过
            if rurl and rurl in dynamic_urls:
                continue
            # 该 subtopic 已经有动态 LOCAL 资源 → snapshot 里的旧 article 记录不再需要
            if rsub and rsub in dynamic_subtopics:
                continue
        resources.append(resource)

    existing_resource_ids = {
        resource.get("resource_id")
        for resource in resources
        if isinstance(resource, dict) and resource.get("resource_id")
    }
    # 个性化学习资源生成 Agent persists videos under recommended_videos. Normalise and merge
    # them at the API boundary so article backfills never hide videos.
    resources.extend(
        [
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
                "subtopic_id": video.get("subtopic_id", ""),
            }
            for index, video in enumerate(package.get("recommended_videos", []), start=1)
            if isinstance(video, dict) and video.get("resource_id", f"video_{index}") not in existing_resource_ids
        ]
    )
    existing_resource_ids = {
        resource.get("resource_id")
        for resource in resources
        if isinstance(resource, dict) and resource.get("resource_id")
    }
    for index, source in enumerate(package.get("sources", []), start=1):
        if not isinstance(source, dict) or source.get("source_type") not in {
            "knowledge_doc", "best_practice", "manual", "external_note"
        }:
            continue
        resource_id = source.get("source_id") or f"source_{index}"
        if resource_id in existing_resource_ids:
            continue
        metadata = source.get("metadata_json") if isinstance(source.get("metadata_json"), dict) else {}
        subtopic_id = metadata.get("subtopic_id")
        if subtopic_id and f"LOCAL-{subtopic_id}" in existing_resource_ids:
            continue
        candidates = [source.get("content_ref"), source.get("locator")]
        locator = next(
            (value for value in candidates if str(value or "").startswith(("http://", "https://"))),
            source.get("locator") or source.get("content_ref") or "",
        )
        resources.append({
            "resource_id": resource_id,
            "resource_type": "article" if str(locator).startswith(("http://", "https://")) else "documentation",
            "title": source.get("title") or metadata.get("title") or f"文章来源 {index}",
            "url": locator,
            "platform": metadata.get("platform", ""),
            "description": metadata.get("description", ""),
            "recommendation_level": "primary",
            "role": "source",
            "order": len(resources) + 1,
            "subtopic_id": subtopic_id or "",
        })
        existing_resource_ids.add(resource_id)
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

    # 关键修复：对所有 learning_resources 字典里的字符串字段（尤其 subtopic_id / url /
    # title / platform / description）做 None → "" 规范化。老快照里 recommended_videos
    # 和 sources 的 subtopic_id 经常是 None，会触发 Pydantic string_type 校验错误，
    # 进而导致整个 GET /sessions/{id} / stages/current/package 接口 500。
    normalized_resources: list[dict[str, Any]] = []
    for r in resources or []:
        if not isinstance(r, dict):
            continue
        normalized_resources.append({
            **r,
            "resource_id": _norm_str(r.get("resource_id")) or f"res_{len(normalized_resources) + 1}",
            "resource_type": _norm_str(r.get("resource_type")) or "article",
            "title": _norm_str(r.get("title")),
            "url": _norm_str(r.get("url", r.get("canonical_url"))),
            "platform": _norm_str(r.get("platform")),
            "description": _norm_str(r.get("description", r.get("content_summary"))),
            "recommendation_level": _norm_str(r.get("recommendation_level"))
                                   or _norm_str(r.get("role"))
                                   or "supplementary",
            "role": _norm_str(r.get("role")) or "supplementary",
            "order": int(r.get("order") or r.get("sequence_index") or 1),
            "subtopic_id": _norm_str(r.get("subtopic_id")),
        })

    return PublicLearningPackage(
        stage_id=stage.get("stage_id", package.get("stage_id")),
        stage_num=stage.get("stage_num", 1),
        block_name=stage.get("block_name", package.get("block_name", "")),
        subtopic_name=public_subtopic_name,
        subtopic_ids=list(stage.get("subtopic_ids") or []),
        subtopic_names=list(stage.get("subtopic_names") or []),
        estimated_hours=stage.get("estimated_hours", 0),
        tutorial=package.get("tutorial"),
        code_examples=package.get("code_examples", []),
        learning_resources=normalized_resources,
        questions=[to_public_question(question) for question in package.get("questions", [])],
        review_summary={
            "status": package.get("status", "draft"),
            "reviewed_by": "专业知识与教学质量审核 Agent",
            "review_notes": package.get("review_notes", ""),
        },
        recommended_daily_hours=stage.get("recommended_daily_hours"),
        suggested_study_days=stage.get("suggested_study_days"),
        learning_style_notes=stage.get("learning_style_notes"),
        hardware_notes=stage.get("hardware_notes"),
        personalization_summary=stage.get("personalization_summary"),
    ).model_dump(mode="json")
