from typing import Optional
import requests
from config import settings

class LLMClient:
    def __init__(self):
        self.ollma_base_url = settings.OLLMA_BASE_URL
        self.ollma_model = settings.OLLMA_MODEL

    async def complete(self, prompt: str, system_prompt: str = "", max_tokens: int = 4096, temperature: float = 0.7) -> str:
        try:
            response = requests.post(
                f"{self.ollma_base_url}/api/generate",
                json={
                    "model": self.ollma_model,
                    "prompt": prompt,
                    "system": system_prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": False
                },
                timeout=120
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            return f"Error: {str(e)}"

    async def chat(self, messages: list, max_tokens: int = 4096, temperature: float = 0.7) -> str:
        try:
            response = requests.post(
                f"{self.ollma_base_url}/api/chat",
                json={
                    "model": self.ollma_model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": False
                },
                timeout=120
            )
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "")
        except Exception as e:
            return f"Error: {str(e)}"
