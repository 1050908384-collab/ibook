# -*- coding: utf-8 -*-
import sys
import os
import logging
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.chat import router as chat_router
from api.session import router as session_router
from api.stage import router as stage_router
from api.visualize import router as visualize_router
from api.assessment import router as assessment_router
from api.dashboard import router as dashboard_router
from api.v1.sessions import router as v1_sessions_router
from api.v1.diagnosis import router as v1_diagnosis_router
from api.v1.profile import router as v1_profile_router
from api.v1.stages import router as v1_stages_router
from api.v1.qa import router as v1_qa_router
from config import settings

# Reuse Uvicorn's configured logger so request diagnostics are visible in the
# same terminal that runs the backend.
logger = logging.getLogger("uvicorn.error")

app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    description="AI Agent Trainer - 多Agent个性化辅导系统"
)

@app.middleware("http")
async def add_utf8_json_charset(request, call_next):
    started_at = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        if request.url.path.startswith("/api/"):
            logger.exception(
                "api_request_failed method=%s path=%s origin=%s",
                request.method,
                request.url.path,
                request.headers.get("origin", "-"),
            )
        raise

    content_type = response.headers.get("content-type", "")
    if content_type.startswith("application/json") and "charset=" not in content_type:
        response.headers["content-type"] = "application/json; charset=utf-8"
    if request.url.path.startswith("/api/"):
        logger.info(
            "api_request method=%s path=%s status=%s duration_ms=%.1f origin=%s",
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - started_at) * 1000,
            request.headers.get("origin", "-"),
        )
    return response


# Register CORS after the diagnostic middleware so it also decorates error
# responses. The regex permits private-LAN frontend hosts without hard-coding
# an individual machine address.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.CORS_ORIGINS.split(",") if origin.strip()],
    allow_origin_regex=r"^https?://(?:localhost|127\.0\.0\.1|192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[0-1])\.\d{1,3}\.\d{1,3})(?::\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/api/chat", tags=["Chat"])
app.include_router(session_router, prefix="/api/session", tags=["Session"])
app.include_router(stage_router, prefix="/api/stage", tags=["Stage"])
app.include_router(visualize_router, prefix="/api/visualize", tags=["Visualize"])
# Agent46_API接口任务说明.md 第 2/4 节：测试服务与仪表盘，路径遵循文档 /api/v1 前缀
app.include_router(assessment_router, prefix="/api/v1", tags=["Assessment"])
app.include_router(dashboard_router, prefix="/api/v1", tags=["Dashboard"])
# Agent13_API接口任务说明.md：会话/诊断/画像/阶段管理接口
app.include_router(v1_sessions_router, prefix="/api/v1", tags=["Sessions"])
app.include_router(v1_diagnosis_router, prefix="/api/v1", tags=["Diagnosis"])
app.include_router(v1_profile_router, prefix="/api/v1", tags=["Profile"])
app.include_router(v1_stages_router, prefix="/api/v1", tags=["Stages"])
app.include_router(v1_qa_router, prefix="/api/v1", tags=["Q&A"])

@app.get("/")
def root():
    return {"message": f"{settings.APP_NAME} is running"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.DEBUG,
        access_log=True,
        log_level=settings.LOG_LEVEL.lower(),
    )
