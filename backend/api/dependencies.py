# -*- coding: utf-8 -*-
"""
api/dependencies.py - 共享依赖注入
"""

import re

from fastapi import Depends, Header, HTTPException

from orchestrator.harness import Harness
from db.sqlite_client import SQLiteClient


_USER_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$")

_harness_instance = None
_assessment_service_instance = None


def get_harness() -> Harness:
    """获取 Harness 单例"""
    global _harness_instance
    if _harness_instance is None:
        _harness_instance = Harness()
    return _harness_instance


def get_current_user_id(x_user_id: str = Header(alias="X-User-ID")) -> str:
    """Resolve the caller identity. The frontend persists one anonymous ID per browser."""
    user_id = x_user_id.strip()
    if not _USER_ID_PATTERN.fullmatch(user_id):
        raise HTTPException(status_code=400, detail="Invalid X-User-ID")
    return user_id


def require_owned_session(
    session_id: str,
    user_id: str = Depends(get_current_user_id),
) -> str:
    """Reject cross-user access without revealing whether the session exists."""
    try:
        owner_id = get_harness().state_manager.db_client.get_session_user_id(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc
    if owner_id != user_id:
        raise HTTPException(status_code=404, detail="Session not found")
    return user_id


def require_owned_attempt(
    attempt_id: str,
    user_id: str = Depends(get_current_user_id),
) -> str:
    attempt = get_harness().state_manager.db_client.get_assessment_attempt(attempt_id)
    if not attempt:
        raise HTTPException(status_code=404, detail="Assessment attempt not found")
    return require_owned_session(attempt["session_id"], user_id)


def get_assessment_service():
    """获取 AssessmentService 单例"""
    global _assessment_service_instance
    if _assessment_service_instance is None:
        from services.assessment_service import AssessmentService
        db_client = SQLiteClient()
        _assessment_service_instance = AssessmentService(
            db_client=db_client,
            state_manager=get_harness().state_manager,
        )
    return _assessment_service_instance
