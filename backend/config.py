from pydantic import AliasChoices, Field
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

    # Accept the historic misspelled names so existing local .env files keep
    # working, while new deployments use the documented OLLAMA_* names.
    OLLAMA_BASE_URL: str = Field(
        default="http://localhost:11434",
        validation_alias=AliasChoices("OLLAMA_BASE_URL", "OLLMA_BASE_URL"),
    )
    OLLAMA_MODEL: str = Field(
        default="deepseek-r1:7b",
        validation_alias=AliasChoices("OLLAMA_MODEL", "OLLMA_MODEL"),
    )
    # Ordered local generation fallbacks. An empty value keeps the historic
    # single-model behavior through OLLAMA_MODEL.
    OLLAMA_FALLBACK_MODELS: str = ""
    # Chroma 向量库使用的 Ollama embedding 模型（中文语义检索）
    EMBEDDING_MODEL: str = "qwen3-embedding:0.6b"

    # DeepSeek uses an OpenAI-compatible Chat Completions API.  Keep the key
    # in backend/.env (or the process environment), never in source control.
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_API_BASE: str = "https://api.deepseek.com"
    DEEPSEEK_MODEL: str = "deepseek-chat"
    DEEPSEEK_TEMPERATURE: float = 0.7
    DEEPSEEK_TOP_P: float = 0.95
    DEEPSEEK_MAX_TOKENS: int = 8192
    DEEPSEEK_PROXY_URL: str = ""
    DEEPSEEK_FALLBACK_IPS: str = ""
    # 云端推理偶发长时间排队或先输出 reasoning_content；答疑需要比默认 HTTP
    # 超时更长的读取窗口，并由客户端对可恢复错误做有限重试。
    DEEPSEEK_TIMEOUT_SECONDS: float = 180.0
    DEEPSEEK_MAX_RETRIES: int = 2
    # Ollama 本地生成参数（用于可替换性测试时对比云端模型参数）
    OLLAMA_TEMPERATURE: float = 0.7
    OLLAMA_TOP_P: float = 0.95
    OLLAMA_MAX_TOKENS: int = 8192
    # Use the configured cloud model by default. When the cloud DeepSeek call
    # fails, automatically fall back to the local Ollama model
    # (deepseek-r1:7b by default, see OLLAMA_MODEL). Set this to "" to disable
    # the automatic local fallback for a deployment.
    LLM_PROVIDER: str = "deepseek"
    LLM_FALLBACK_PROVIDER: str = "ollama"

    DATABASE_URL: str = "sqlite:///../data/sqlite_data/project.db"
    CHROMA_PATH: str = "../data/chroma_db"
    GRAPH_PATH: str = "../data/graph_data/runtime/knowledge_graph.gpickle"

    LOG_LEVEL: str = "INFO"
    DEBUG: bool = False

    MAX_REVIEW_ITERATIONS: int = 3
    CONFIDENCE_THRESHOLD: float = 0.7
    EXECUTION_MODE: str = "hybrid"
    # 块级并行流水线开关（个性化学习资源生成 Agent/专业知识与教学质量审核 Agent 按块并行生成+审核）。
    # 默认开启：start_stage 走块级真并发流水线，并按云端/本地模型选择并发上限；
    # 如需回退旧整篇审核链路，可在 .env 覆盖为 False。
    BLOCK_PIPELINE_ENABLED: bool = True
    BLOCK_CONCURRENCY_CLOUD: int = 5
    BLOCK_CONCURRENCY_LOCAL: int = 3

    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
