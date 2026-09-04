from pydantic import AliasChoices, Field

# BaseSettings 在 pydantic v2 中从 pydantic-settings 拆出；兼容 v1/v2。
# 若评委机没装 pydantic-settings（v2 附加包），回退到 pydantic v1 自带 BaseSettings。
try:  # pydantic v2
    from pydantic_settings import BaseSettings, SettingsConfigDict  # type: ignore
except Exception:  # pragma: no cover - pydantic v1 回退
    from pydantic import BaseSettings  # type: ignore

    class _Config:
        env_file = ".env"
        extra = "ignore"

    SettingsConfigDict = lambda **kw: _Config  # type: ignore
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
    CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5174,http://localhost:5173,http://127.0.0.1:5174"

    # ---- LLM provider runtime override (written by launcher on startup) ----
    # "一键启动" 检测到 DeepSeek API Key 未正确配置（占位符 / 空值）时，会
    # 把此变量设为 "ollama" 并传入后端进程。此时 LLMClient 会**完全跳过云端**，
    # 一次都不尝试连接 DeepSeek，直接使用本地 Ollama。
    # 可用值：
    #   - ""       (默认/空)：尊重 LLM_PROVIDER + LLM_FALLBACK_PROVIDER 原有配置
    #   - "ollama"：强制所有调用走本地 Ollama（云跳过）
    #   - "deepseek"：强制走云端（调试用，不推荐）
    LLM_PROVIDER_OVERRIDE: str = ""


    # Accept the historic misspelled names so existing local .env files keep
    # working, while new deployments use the documented OLLAMA_* names.
    OLLAMA_BASE_URL: str = Field(
        default="http://localhost:11434",
        validation_alias=AliasChoices("OLLAMA_BASE_URL", "OLLMA_BASE_URL"),
    )
    OLLAMA_MODEL: str = Field(
        default="qwen2.5:7b",
        validation_alias=AliasChoices("OLLAMA_MODEL", "OLLMA_MODEL"),
    )
    # Ordered local generation fallbacks (comma-separated list, e.g.
    # "qwen2.5:7b,deepseek-r1:7b").  An empty value keeps the historic
    # single-model behavior through OLLAMA_MODEL.  In ordered fallback the
    # first entry SHOULD equal OLLAMA_MODEL so the two sources stay
    # consistent when a downstream consumer checks either.
    OLLAMA_FALLBACK_MODELS: str = "qwen2.5:7b,deepseek-r1:7b"
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
    # When no explicit proxy is configured, honor standard HTTPS_PROXY/
    # HTTP_PROXY environment variables (useful on networks that block direct
    # TCP 443). Set false to force direct connections.
    DEEPSEEK_USE_SYSTEM_PROXY: bool = True
    DEEPSEEK_FALLBACK_IPS: str = ""
    # 云端推理偶发长时间排队或先输出 reasoning_content；答疑需要比默认 HTTP
    # 超时更长的读取窗口，并由客户端对可恢复错误做有限重试。
    DEEPSEEK_TIMEOUT_SECONDS: float = 180.0
    DEEPSEEK_MAX_RETRIES: int = 2
    # Any provider exposing the OpenAI Chat Completions contract can be used
    # without changing application code.  The historic DEEPSEEK_* variables
    # remain the defaults for backwards compatibility.
    OPENAI_COMPATIBLE_API_KEY: str = ""
    OPENAI_COMPATIBLE_API_BASE: str = ""
    OPENAI_COMPATIBLE_MODEL: str = ""
    OPENAI_COMPATIBLE_TIMEOUT_SECONDS: float = 180.0
    OPENAI_COMPATIBLE_MAX_RETRIES: int = 2
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
