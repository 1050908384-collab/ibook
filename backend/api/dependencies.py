from functools import lru_cache

from orchestrator.harness import Harness


@lru_cache(maxsize=1)
def get_harness() -> Harness:
    return Harness()


@lru_cache(maxsize=1)
def get_assessment_service():
    """测试服务（Agent46_API接口任务说明.md 第 2 节）单例。

    复用全局 Harness 的 state_manager / db_client，保证读到“当前学习包”的题目，
    并与 submit_answer 共享同一套判题口径。
    """
    from services.assessment_service import AssessmentService

    harness = get_harness()
    return AssessmentService(
        db_client=harness.state_manager.db_client,
        state_manager=harness.state_manager,
    )
