from typing import Callable, Any
from backend.config import settings


def with_fallback(func: Callable, fallback_provider: str = "ollama") -> Callable:
    def wrapper(*args, **kwargs):
        provider = kwargs.get("provider", settings.LLM_PROVIDER)
        
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Primary provider {provider} failed: {str(e)}")
            print(f"Falling back to {fallback_provider}")
            
            kwargs["provider"] = fallback_provider
            return func(*args, **kwargs)
    
    return wrapper