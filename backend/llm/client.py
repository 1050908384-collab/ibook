from typing import Optional
import asyncio
import httpx
import json
import re
from config import settings
import logging

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self):
        self.ollma_base_url = settings.OLLMA_BASE_URL
        self.ollma_model = settings.OLLMA_MODEL
        self.client = httpx.AsyncClient(timeout=120)

    def _clean_response(self, response_text: str) -> str:
        response_text = response_text.replace("<think>", "").replace("</think>", "")
        response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
        return response_text.strip()

    async def complete(self, prompt: str, system_prompt: str = "", max_tokens: int = 8192, temperature: float = 0.7) -> str:
        try:
            print(f"=== 发送给Ollama的prompt（前200字符）=== {prompt[:200]}")
            
            payload = {
                "model": self.ollma_model,
                "prompt": prompt,
                "system": system_prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False
            }
            
            response = await self.client.post(
                f"{self.ollma_base_url}/api/generate",
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            response_text = result.get("response", "")
            cleaned_response = self._clean_response(response_text)
            return cleaned_response
        except httpx.RequestError as e:
            logger.error(f"Ollama request failed: {e}")
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"LLM complete error: {e}")
            return f"Error: {str(e)}"

    async def chat(self, messages: list, max_tokens: int = 8192, temperature: float = 0.7) -> str:
        try:
            payload = {
                "model": self.ollma_model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False
            }
            
            response = await self.client.post(
                f"{self.ollma_base_url}/api/chat",
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            response_text = result.get("message", {}).get("content", "")
            cleaned_response = self._clean_response(response_text)
            return cleaned_response
        except httpx.RequestError as e:
            logger.error(f"Ollama chat request failed: {e}")
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"LLM chat error: {e}")
            return f"Error: {str(e)}"