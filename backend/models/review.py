from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, computed_field

class ReviewError(BaseModel):
    error_type: str
    severity: Literal["critical", "major", "minor"] = "major"
    location: Optional[str] = None
    description: str
    evidence: Optional[str] = None
    suggestion: Optional[str] = None
    # === Agent3/4 审核辩论协议扩展字段（协议 2.3 / 2.4）===
    error_id: Optional[str] = None                          # 当前轮实例 ID：canonical_error_id + "_r" + round_index
    canonical_error_id: Optional[str] = None                # 跨轮稳定 ID（sha1 前12位，见协议 2.4）
    source_refs: List[str] = Field(default_factory=list)    # 证据来源，指向 sources[].source_id；空表示"未命中来源"
    change_request: Optional[Dict[str, Any]] = None         # {"required": bool, "action": str}
    must_fix: bool = False                                  # 流程约束级别，非 severity（协议第 6 节派生）
    must_fix_reason: Optional[str] = None
    claim_id: Optional[str] = None                          # 关联的 Claim（协议 2.2）

class ReviewResult(BaseModel):
    reviewer: Literal["technical", "quality"]
    passed: bool
    errors: List[ReviewError] = Field(default_factory=list)
    review_details: Optional[Any] = None


class TechnicalReviewResult(ReviewResult):
    reviewer: Literal["technical"] = "technical"


class QualityReviewResult(ReviewResult):
    reviewer: Literal["quality"] = "quality"
    scores: Optional[dict] = None


class ReviewRound(BaseModel):
    iteration: int
    technical: TechnicalReviewResult
    quality: QualityReviewResult

    @computed_field
    @property
    def passed(self) -> bool:
        return self.technical.passed and self.quality.passed


# ============================================================================
# Agent3/4 审核辩论协议模型（协议 2.3 / 第 3~5 节；映射数据库框架第九节 3 张表）
# 字段与约束严格对齐 review_debate_issues / review_debate_rounds /
# review_debate_responses 的 DDL（backend/db/sqlite_client.py）。
# 注意：source_id / claim_id / canonical_error_id 均为「单个 trace 内唯一」，
# 跨表引用必须携带 trace_id（协议 8.1）。
# ============================================================================

class DebateIssue(BaseModel):
    """争议单：围绕一个 canonical_error_id 展开的完整生命周期。

    对应表 review_debate_issues（主键 issue_id，UNIQUE(trace_id, canonical_error_id)）。
    """
    issue_id: str                                           # 全局唯一
    trace_id: str
    canonical_error_id: str                                 # 单个 trace 语义下唯一
    error_id: str                                           # 当前轮实例 ID
    claim_id: Optional[str] = None
    error_type: Optional[str] = None
    severity: Literal["critical", "major", "minor"] = "major"
    must_fix: bool = False
    must_fix_reason: Optional[str] = None
    description: Optional[str] = None
    location: Optional[str] = None
    evidence: Optional[str] = None
    suggestion: Optional[str] = None
    source_refs: List[str] = Field(default_factory=list)
    change_request: Optional[Dict[str, Any]] = None
    current_status: Literal[
        "open", "accepted", "rejected", "partially_accepted",
        "deferred", "overruled", "withdrawn", "needs_human_review",
    ] = "open"
    final_disposition: Optional[Literal[
        "accepted", "rejected", "partially_accepted",
        "deferred", "overruled", "withdrawn", "needs_human_review",
    ]] = None
    opened_in_round_num: Optional[int] = None
    resolved_in_round_num: Optional[int] = None


class DebateResponse(BaseModel):
    """每轮里对某 Issue 的一条回复（作者答辩 / 审核复核 / 编排裁决）。

    对应表 review_debate_responses
    （主键 response_id，UNIQUE(round_id, issue_id, actor, response_type)）。
    """
    response_id: str                                        # 全局唯一
    round_id: str
    issue_id: str
    actor: Literal["agent3", "agent4", "orchestrator"]
    response_type: Literal["author_reply", "reviewer_verdict", "orchestrator_ruling"]
    disposition: Literal[
        "accepted", "rejected", "partially_accepted",
        "persist", "withdraw", "downgrade", "overruled", "deferred",
    ]
    rationale: Optional[str] = None
    source_refs: List[str] = Field(default_factory=list)
    patched_sections: List[str] = Field(default_factory=list)


class DebateRound(BaseModel):
    """答辩复核轮次（一轮 = Agent3 回复 + Agent4 复核 + Orchestrator 裁决）。

    对应表 review_debate_rounds（主键 round_id，UNIQUE(trace_id, round_num)）。
    """
    round_id: str                                           # 全局唯一
    trace_id: str
    round_num: int
    review_iteration: Optional[int] = None
    round_status: Literal["open", "closed"] = "open"
    orchestrator_verdict: Literal[
        "pending", "all_resolved", "partial_resolved",
        "deadlocked", "needs_human_review",
    ] = "pending"
    verdict_summary: Optional[str] = None
