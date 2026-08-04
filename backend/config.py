from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

BACKEND_DIR = Path(__file__).parent
PROJECT_ROOT = BACKEND_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
SKILLS_DIR = BACKEND_DIR / "skills"


class Settings(BaseSettings):
    APP_NAME: str = "AI Agent Trainer"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8500
    # Comma-separated browser origins.  Keep this configurable for LAN and
    # production deployments instead of relying on a wildcard with cookies.
    CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:5174,http://127.0.0.1:5174"

    OLLMA_BASE_URL: str = "http://localhost:11434"
    OLLMA_MODEL: str = "deepseek-r1:7b"

    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_API_BASE: str = "https://api.deepseek.com"

    DATABASE_URL: str = "sqlite:///../data/sqlite_data/project.db"
    CHROMA_PATH: str = "../data/chroma_db"
    GRAPH_PATH: str = "../data/graph_data/runtime/knowledge_graph.gpickle"

    LOG_LEVEL: str = "INFO"
    DEBUG: bool = False

    MAX_REVIEW_ITERATIONS: int = 3
    CONFIDENCE_THRESHOLD: float = 0.7
    EXECUTION_MODE: str = "hybrid"

    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
