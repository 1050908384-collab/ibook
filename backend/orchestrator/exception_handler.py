from typing import Dict, Any, Callable
import logging
from backend.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExceptionHandler:
    def __init__(self):
        self.fallback_provider = "ollama"
        self.circuit_breaker_tripped = False
        self.error_count = 0
        self.max_errors_before_circuit = 3
    
    def handle(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Agent execution error: {str(e)}")
            self.error_count += 1
            
            if self.error_count >= self.max_errors_before_circuit:
                self.circuit_breaker_tripped = True
                logger.warning("Circuit breaker tripped!")
                return self._circuit_breaker_response(e)
            
            return self._fallback_response(e)
    
    def _fallback_response(self, error: Exception) -> Dict[str, Any]:
        return {
            "success": False,
            "error": str(error),
            "fallback_used": True,
            "message": "服务暂时异常，已切换到备用方案"
        }
    
    def _circuit_breaker_response(self, error: Exception) -> Dict[str, Any]:
        return {
            "success": False,
            "error": str(error),
            "circuit_breaker_tripped": True,
            "message": "服务过载，已触发熔断保护，请稍后重试"
        }
    
    def reset(self):
        self.error_count = 0
        self.circuit_breaker_tripped = False