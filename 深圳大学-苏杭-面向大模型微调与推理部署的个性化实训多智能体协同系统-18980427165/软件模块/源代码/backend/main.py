# -*- coding: utf-8 -*-
import sys
import os
import logging
import time
from pathlib import Path

BACKEND_DIR_HERE = Path(__file__).resolve().parent
PROJECT_ROOT_HERE = BACKEND_DIR_HERE.parent
SCRIPTS_DIR = PROJECT_ROOT_HERE / "scripts"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 让启动时可以直接 import scripts 下的初始化脚本（真正做到一键启动）
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
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
from config import DATA_DIR, settings

# Reuse Uvicorn's configured logger so request diagnostics are visible in the
# same terminal that runs the backend.
logger = logging.getLogger("uvicorn.error")

app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    description="AI Agent Trainer - 多Agent个性化辅导系统"
)


# =====================================================================
# 一键启动 · 初始化兜底
# 当 project.db 是新建空库（克隆后首次启动）时：
#   1. 从 data/seeds/*.json 导入 块/知识点/题库/视频资源
#   2. 从 data/知识库原始数据/*.md 导入本地学习资料（+分块）
#   3. 生成 knowledge_graph.gpickle / nodes.jsonl 等
# 有任意数据存在则跳过，不影响已有环境启动速度。
# =====================================================================
_BOOTSTRAP_DONE = False


def _count_rows(db_client, table_name: str) -> int:
    with db_client.Session() as session:
        return session.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar() or 0


def _count_video_catalog_rows(db_client, mapping_table: str | None = None) -> int:
    """Count video resources or video mappings in the runtime database."""
    with db_client.Session() as session:
        if mapping_table:
            statement = text(
                f"""
                SELECT COUNT(*)
                FROM {mapping_table} AS mapping
                JOIN learning_resources AS resource
                  ON resource.resource_id = mapping.resource_id
                WHERE resource.resource_type = 'video'
                """
            )
        else:
            statement = text("SELECT COUNT(*) FROM learning_resources WHERE resource_type = 'video'")
        return session.execute(statement).scalar() or 0


def _resolve_runtime_graph_path() -> Path:
    raw = Path(settings.GRAPH_PATH)
    return raw if raw.is_absolute() else (BACKEND_DIR_HERE / raw).resolve()


def _run_bootstrap_once() -> None:
    global _BOOTSTRAP_DONE
    if _BOOTSTRAP_DONE:
        return
    _BOOTSTRAP_DONE = True

    from api.dependencies import get_harness
    harness = get_harness()
    db_client = harness.state_manager.db_client

    blocks = _count_rows(db_client, "knowledge_blocks")
    subtopics = _count_rows(db_client, "knowledge_subtopics")
    questions = _count_rows(db_client, "test_questions")
    docs = _count_rows(db_client, "knowledge_docs")
    chunks = _count_rows(db_client, "knowledge_chunks")
    videos = _count_video_catalog_rows(db_client)
    block_video_mappings = _count_video_catalog_rows(db_client, "block_resources")
    subtopic_video_mappings = _count_video_catalog_rows(db_client, "subtopic_resources")
    graph_path = _resolve_runtime_graph_path()

    need_seeds = blocks == 0 or subtopics == 0 or questions == 0
    # Earlier installations can have curriculum and article data but predate
    # the video catalogue.  Treat that as an incomplete bootstrap state.
    need_video_catalog = videos == 0 or block_video_mappings == 0 or subtopic_video_mappings == 0
    need_lessons = chunks == 0
    need_graph = not graph_path.is_file()

    if not (need_seeds or need_video_catalog or need_lessons or need_graph):
        return

    print("[BOOTSTRAP] 首次启动检测到素材未就绪，开始自动导入（这是正常的，只会跑一次）...", flush=True)

    if need_seeds:
        print(f"[BOOTSTRAP 1/3] 从 data/seeds/*.json 导入基础素材：blocks={blocks} subtopics={subtopics} questions={questions}", flush=True)
        try:
            import insert_knowledge_data  # scripts/insert_knowledge_data.py
            insert_knowledge_data.main()
        except Exception as exc:
            logger.exception("bootstrap seeds import failed")
            print(f"[WARN] seeds 导入失败（不会阻塞主程序）：{exc}", flush=True)

    elif need_video_catalog:
        print(
            "[BOOTSTRAP] 检测到视频资源目录不完整，开始同步视频资源"
            f"：videos={videos} block_mappings={block_video_mappings} "
            f"subtopic_mappings={subtopic_video_mappings}",
            flush=True,
        )
        try:
            import insert_knowledge_data  # scripts/insert_knowledge_data.py
            insert_knowledge_data.sync_video_catalog(db_client)
        except Exception as exc:
            logger.exception("bootstrap video catalog import failed")
            print(f"[WARN] 视频资源同步失败（不会阻塞主程序）：{exc}", flush=True)

    if need_lessons:
        print(f"[BOOTSTRAP 2/3] 从 data/知识库原始数据 导入本地学习资料并分块：docs={docs} chunks={chunks}", flush=True)
        try:
            import import_local_learning_materials  # scripts/import_local_learning_materials.py
            doc_count, chunk_count = import_local_learning_materials.import_lessons()
            print(f"[OK] 本地资料导入完成：{doc_count} 文档 / {chunk_count} 分块", flush=True)
        except Exception as exc:
            logger.exception("bootstrap local lessons import failed")
            print(f"[WARN] 本地学习资料导入失败（不会阻塞主程序）：{exc}", flush=True)

    if need_graph:
        print(f"[BOOTSTRAP 3/3] 构建知识图谱 runtime/knowledge_graph.gpickle 及衍生 JSON", flush=True)
        try:
            import build_knowledge_graph  # scripts/build_knowledge_graph.py
            summary = build_knowledge_graph.build_knowledge_graph()
            print(f"[OK] 知识图谱构建完成：nodes={summary.get('nodes')} edges={summary.get('edges')}", flush=True)
        except Exception as exc:
            logger.exception("bootstrap knowledge graph build failed")
            print(f"[WARN] 知识图谱构建失败（不会阻塞主程序，懒构建仍可触发）：{exc}", flush=True)

    print("[BOOTSTRAP] 初始化结束，继续启动后端服务。", flush=True)


@app.on_event("startup")
async def _on_startup_bootstrap():
    # 放在 startup 事件里，保证：
    # 1. 无论是 `python main.py` 还是 `uvicorn main:app` 都会执行
    # 2. SQLiteClient 建表已完成（get_harness 内部依赖注入才第一次实例化）
    _run_bootstrap_once()



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
# 专业知识与教学质量审核 Agent6_API接口任务说明.md 第 2/4 节：测试服务与仪表盘，路径遵循文档 /api/v1 前缀
app.include_router(assessment_router, prefix="/api/v1", tags=["Assessment"])
app.include_router(dashboard_router, prefix="/api/v1", tags=["Dashboard"])
# 学情诊断与阶段规划 Agent3_API接口任务说明.md：会话/诊断/画像/阶段管理接口
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
