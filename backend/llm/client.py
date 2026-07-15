from typing import Optional
import asyncio
import httpx
from config import settings
import logging

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self):
        self.ollma_base_url = settings.OLLMA_BASE_URL
        self.ollma_model = settings.OLLMA_MODEL
        self.client = httpx.AsyncClient(timeout=120)

    async def complete(self, prompt: str, system_prompt: str = "", max_tokens: int = 4096, temperature: float = 0.7) -> str:
        try:
            response = await self.client.post(
                f"{self.ollma_base_url}/api/generate",
                json={
                    "model": self.ollma_model,
                    "prompt": prompt,
                    "system": system_prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": False
                }
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except httpx.RequestError as e:
            logger.error(f"Ollama request failed: {e}")
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"LLM complete error: {e}")
            return f"Error: {str(e)}"

    async def chat(self, messages: list, max_tokens: int = 4096, temperature: float = 0.7) -> str:
        try:
            response = await self.client.post(
                f"{self.ollma_base_url}/api/chat",
                json={
                    "model": self.ollma_model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": False
                }
            )
            response.raise_for_status()
            result = response.json()
            return result.get("message", {}).get("content", "")
        except httpx.RequestError as e:
            logger.error(f"Ollama chat request failed: {e}")
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"LLM chat error: {e}")
            return f"Error: {str(e)}"