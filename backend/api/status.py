from fastapi import APIRouter
from typing import Dict

router = APIRouter()


@router.get("/")
async def get_status():
    return {
        "status": "running",
        "version": "1.0.0",
        "agents": {
            "agent_01_diagnostic": "online",
            "agent_02_retrieval": "online",
            "agent_03_generation": "online",
            "agent_04_tech_review": "online",
            "agent_05_quality_review": "online",
            "agent_06_iteration": "online"
        },
        "llm_provider": "ollama",
        "database": "mock"
    }