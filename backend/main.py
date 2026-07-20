import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.chat import router as chat_router
from api.session import router as session_router
from api.stage import router as stage_router
from api.visualize import router as visualize_router
from config import settings

app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    description="AI Agent Trainer - 多Agent个性化辅导系统"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_utf8_json_charset(request, call_next):
    response = await call_next(request)
    content_type = response.headers.get("content-type", "")
    if content_type.startswith("application/json") and "charset=" not in content_type:
        response.headers["content-type"] = "application/json; charset=utf-8"
    return response

app.include_router(chat_router, prefix="/api/chat", tags=["Chat"])
app.include_router(session_router, prefix="/api/session", tags=["Session"])
app.include_router(stage_router, prefix="/api/stage", tags=["Stage"])
app.include_router(visualize_router, prefix="/api/visualize", tags=["Visualize"])

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
        reload=True
    )
