from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.config import settings
from backend.api import chat, session, feedback, submit, report, status

app = FastAPI(title="AI Agent Trainer", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat, prefix="/api/chat", tags=["chat"])
app.include_router(session, prefix="/api/session", tags=["session"])
app.include_router(feedback, prefix="/api/feedback", tags=["feedback"])
app.include_router(submit, prefix="/api/submit", tags=["submit"])
app.include_router(report, prefix="/api/report", tags=["report"])
app.include_router(status, prefix="/api/status", tags=["status"])


@app.get("/")
def root():
    return {"message": "AI Agent Trainer API", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.PORT)