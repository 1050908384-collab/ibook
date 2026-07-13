from pydantic_settings import BaseSettings
from typing import Dict, Optional


class Settings(BaseSettings):
    LLM_PROVIDER: str = "ollama"
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "deepseek-7b"
    
    DEEPSEEK_API_KEY: Optional[str] = None
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    
    MAX_DEBATE_ROUNDS: int = 3
    MAX_ITERATIONS: int = 10
    ACCURACY_THRESHOLD: float = 0.85
    COVERAGE_THRESHOLD: float = 0.90
    STAGNATION_THRESHOLD: float = 0.03
    
    CHROMA_DB_PATH: str = "data/chroma_db"
    SQLITE_PATH: str = "data/sqlite_data"
    GRAPH_DATA_PATH: str = "data/graph_data"
    
    PORT: int = 8000
    
    AGENT_MODEL_ROUTING: Dict[str, str] = {
        "agent_01_diagnostic": "deepseek-7b",
        "agent_02_retrieval": "deepseek-7b",
        "agent_03_generation": "deepseek-7b",
        "agent_04_tech_review": "deepseek-7b",
        "agent_05_quality_review": "deepseek-7b",
        "agent_06_iteration": "deepseek-7b",
    }
    
    class Config:
        env_file = ".env"


settings = Settings()