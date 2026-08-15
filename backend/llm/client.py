# -*- coding: utf-8 -*-
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
        self.ollama_base_url = settings.OLLAMA_BASE_URL
        self.ollama_model = settings.OLLAMA_MODEL
        self.client = httpx.AsyncClient(timeout=120)
        self.last_done_reason: Optional[str] = None

    def _clean_response(self, response_text: str) -> str:
        response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
        return response_text.replace("<think>", "").replace("</think>", "").strip()

    async def complete(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 8192,
        temperature: float = 0.7,
        json_mode: bool = False,
    ) -> str:
        self.last_done_reason = None
        try:
            logger.debug("Sending Ollama prompt: %s", prompt[:200])
            
            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "system": system_prompt,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                },
                "stream": False
            }
            if json_mode:
                payload["format"] = "json"
            
            response = await self.client.post(
                f"{self.ollama_base_url}/api/generate",
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            self.last_done_reason = result.get("done_reason")
            response_text = result.get("response", "")
            cleaned_response = self._clean_response(response_text)
            if self.last_done_reason == "length":
                logger.warning("Ollama stopped generation because num_predict was exhausted")
            logger.debug("Ollama response: %s", cleaned_response[:500])
            return cleaned_response
        except httpx.RequestError as e:
            logger.error(f"Ollama request failed: {e}")
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"LLM complete error: {e}")
            return f"Error: {str(e)}"

    async def chat(self, messages: list, max_tokens: int = 8192, temperature: float = 0.7) -> str:
        self.last_done_reason = None
        try:
            payload = {
                "model": self.ollama_model,
                "messages": messages,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                },
                "stream": False
            }
            
            response = await self.client.post(
                f"{self.ollama_base_url}/api/chat",
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            self.last_done_reason = result.get("done_reason")
            response_text = result.get("message", {}).get("content", "")
            cleaned_response = self._clean_response(response_text)
            logger.debug("Ollama chat response: %s", cleaned_response[:500])
            return cleaned_response
        except httpx.RequestError as e:
            logger.error(f"Ollama chat request failed: {e}")
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"LLM chat error: {e}")
            return f"Error: {str(e)}"
