from .chat import router as chat_router
from .session import router as session_router
from .feedback import router as feedback_router
from .submit import router as submit_router
from .report import router as report_router
from .status import router as status_router

chat = chat_router
session = session_router
feedback = feedback_router
submit = submit_router
report = report_router
status = status_router

__all__ = ["chat", "session", "feedback", "submit", "report", "status"]