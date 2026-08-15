# -*- coding: utf-8 -*-
import sys
import os
import logging
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
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
from api.v1.user_scope import router as v1_user_scope_router
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
            print(
                f"[API][500] {request.method} {request.url.path} origin={request.headers.get('origin', '-')}",
                flush=True,
            )
        raise

    content_type = response.headers.get("content-type", "")
    if content_type.startswith("application/json") and "charset=" not in content_type:
        response.headers["content-type"] = "application/json; charset=utf-8"
    if request.url.path.startswith("/api/"):
        message = (
            f"[API][{response.status_code}] {request.method} {request.url.path} "
            f"duration_ms={(time.perf_counter() - started_at) * 1000:.1f} "
            f"origin={request.headers.get('origin', '-')}"
        )
        logger.info(
            "api_request method=%s path=%s status=%s duration_ms=%.1f origin=%s",
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - started_at) * 1000,
            request.headers.get("origin", "-"),
        )
        print(message, flush=True)
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
app.include_router(v1_user_scope_router, prefix="/api/v1", tags=["User Scope"])


@app.get("/health")
def health_check():
    return {"status": "healthy"}

# =====================================================================
# 生产模式前端静态资源托管 (单端口 8500 一键部署)
# 当执行过 npm run build 生成 frontend/dist 后，后端直接托管前端，
# 启动后浏览器访问 http://localhost:8500 就是完整 UI，无需再跑 Vite。
# 支持 React BrowserRouter 的 SPA 回退：任何非 /api/ 路径都回退 index.html
# =====================================================================
BACKEND_DIR = Path(__file__).resolve().parent
FRONTEND_DIST = (BACKEND_DIR.parent / "frontend" / "dist").resolve()
FRONTEND_INDEX = FRONTEND_DIST / "index.html"

if FRONTEND_DIST.is_dir() and FRONTEND_INDEX.is_file():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="frontend-assets")
    # favicon / manifest 等根目录静态文件（逐个挂载，避免和 SPA 回退冲突）
    for child in FRONTEND_DIST.iterdir():
        if child.is_file() and child.name != "index.html":
            safe_name = child.name.strip("/")
            if not safe_name or safe_name == "assets":
                continue
            try:
                app.mount(f"/{safe_name}", StaticFiles(directory=str(FRONTEND_DIST), html=False), name=f"fe-{safe_name}")
            except Exception:
                pass

    _index_bytes: bytes | None = None
    _index_mtime: float = 0.0

    def read_index_html() -> bytes:
        global _index_bytes, _index_mtime
        try:
            mtime = FRONTEND_INDEX.stat().st_mtime
        except OSError:
            return b""
        if _index_bytes is None or mtime != _index_mtime:
            _index_bytes = FRONTEND_INDEX.read_bytes()
            _index_mtime = mtime
        return _index_bytes

    @app.get("/", include_in_schema=False)
    def root_prod() -> HTMLResponse:
        return HTMLResponse(read_index_html())

    @app.api_route("/{path:path}", methods=["GET", "HEAD"], include_in_schema=False)
    async def spa_fallback(path: str, request: Request) -> HTMLResponse:
        # /api/* 和 /docs /openapi.json /health 等已知路径不进入 SPA 回退
        if path.startswith("api/") or path in {"docs", "openapi.json", "health", "redoc"}:
            return HTMLResponse(status_code=404, content="Not Found")
        data = read_index_html()
        if not data:
            return HTMLResponse(status_code=404, content="Frontend dist not found. Run `npm run build` in frontend/.")
        return HTMLResponse(data)
else:
    @app.get("/")
    def root():
        return {"message": f"{settings.APP_NAME} is running", "tip": "Run `npm run build` in frontend/ to host UI here on the same port."}

if __name__ == "__main__":
    import uvicorn
    from api.dependencies import get_harness

    resolved_database_url = get_harness().state_manager.db_client.database_url
    print(
        f"[STARTUP] backend listening on http://{settings.APP_HOST}:{settings.APP_PORT} "
        f"log_level={settings.LOG_LEVEL.upper()} database={resolved_database_url}",
        flush=True,
    )
    uvicorn.run(
        "main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.DEBUG,
        access_log=True,
        log_level=settings.LOG_LEVEL.lower(),
    )
