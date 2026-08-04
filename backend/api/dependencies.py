# -*- coding: utf-8 -*-
"""
api/dependencies.py - 共享依赖注入
"""

from orchestrator.harness import Harness
from db.sqlite_client import SQLiteClient

_harness_instance = None
_assessment_service_instance = None


def get_harness() -> Harness:
    """获取 Harness 单例"""
    global _harness_instance
    if _harness_instance is None:
        _harness_instance = Harness()
    return _harness_instance


def get_assessment_service():
    """获取 AssessmentService 单例"""
    global _assessment_service_instance
    if _assessment_service_instance is None:
        from services.assessment_service import AssessmentService
        db_client = SQLiteClient()
        _assessment_service_instance = AssessmentService(db_client=db_client)
    return _assessment_service_instance
