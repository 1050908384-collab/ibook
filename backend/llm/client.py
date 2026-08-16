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
        self.deepseek_api_key = settings.DEEPSEEK_API_KEY
        self.deepseek_api_base = settings.DEEPSEEK_API_BASE.rstrip("/")
        self.deepseek_model = settings.DEEPSEEK_MODEL
        self.ollama_base_url = settings.OLLAMA_BASE_URL.rstrip("/")
        self.ollama_model = settings.OLLAMA_MODEL
        self.client = httpx.AsyncClient(timeout=120)
        self.last_done_reason: Optional[str] = None

    @property
    def _chat_completions_url(self) -> str:
        if self.deepseek_api_base.endswith("/v1"):
            return f"{self.deepseek_api_base}/chat/completions"
        return f"{self.deepseek_api_base}/v1/chat/completions"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.deepseek_api_key}",
            "Content-Type": "application/json",
        }

    def _has_deepseek_credentials(self) -> bool:
        # 兼容多个常见占位符写法，避免未替换的模板 key 被当作真实凭证发请求
        placeholder_tokens = {
            "",
            "your-api-key-here",
            "your-deepseek-api-key",
            "your_api_key_here",
            "your_deepseek_api_key",
            "sk-yourkey",
            "sk-your-key",
            "placeholder",
            "xxx",
            "xxxx",
        }
        normalized = (self.deepseek_api_key or "").strip().strip('"').strip("'").lower()
        return (
            settings.LLM_PROVIDER.lower() == "deepseek"
            and bool(self.deepseek_api_key)
            and normalized not in placeholder_tokens
            and not normalized.startswith("your-")
            and not normalized.startswith("your_")
        )

    def _can_fallback_to_ollama(self) -> bool:
        return settings.LLM_FALLBACK_PROVIDER.lower() == "ollama"

    def _clean_response(self, response_text: str) -> str:
        response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
        return response_text.replace("<think>", "").replace("</think>", "").strip()

    @staticmethod
    def _extract_error_message(response: httpx.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            return response.text.strip()
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                message = error.get("message")
                if isinstance(message, str) and message.strip():
                    return message.strip()
        return response.text.strip()

    def _format_deepseek_status_error(self, error: httpx.HTTPStatusError) -> str:
        response = error.response
        status_code = response.status_code
        detail = self._extract_error_message(response)
        if status_code == 401:
            return "Error: DeepSeek API authentication failed (HTTP 401). Check DEEPSEEK_API_KEY."
        if status_code == 403:
            return "Error: DeepSeek API access denied (HTTP 403). Check account or endpoint permissions."
        if status_code == 404:
            return (
                f"Error: DeepSeek API endpoint or model is invalid (HTTP 404). "
                f"Check DEEPSEEK_API_BASE/DEEPSEEK_MODEL. Detail: {detail}"
            )
        if status_code == 429:
            return "Error: DeepSeek API rate limit exceeded (HTTP 429). Retry later."
        return f"Error: DeepSeek API request failed (HTTP {status_code}). Detail: {detail}"

    async def complete(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 8192,
        temperature: float = 0.7,
        json_mode: bool = False,
    ) -> str:
        self.last_done_reason = None
        if not self._has_deepseek_credentials():
            return await self._complete_with_ollama(prompt, system_prompt, max_tokens, temperature, json_mode)
        try:
            logger.debug("Sending DeepSeek prompt: %s", prompt[:200])
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            payload = {
                "model": self.deepseek_model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if json_mode:
                payload["response_format"] = {"type": "json_object"}

            response = await self.client.post(
                self._chat_completions_url,
                json=payload,
                headers=self._headers(),
            )
            response.raise_for_status()
            result = response.json()
            choice = result.get("choices", [{}])[0]
            self.last_done_reason = choice.get("finish_reason")
            response_text = choice.get("message", {}).get("content", "")
            cleaned_response = self._clean_response(response_text)
            if self.last_done_reason == "length":
                logger.warning("DeepSeek stopped generation because max_tokens was exhausted")
            logger.debug("DeepSeek response: %s", cleaned_response[:500])
            return cleaned_response
        except httpx.HTTPStatusError as e:
            logger.error("DeepSeek returned HTTP %s: %s", e.response.status_code, self._extract_error_message(e.response))
            if self._can_fallback_to_ollama():
                logger.info("Falling back to local Ollama after DeepSeek HTTP error")
                return await self._complete_with_ollama(prompt, system_prompt, max_tokens, temperature, json_mode)
            return self._format_deepseek_status_error(e)
        except httpx.RequestError as e:
            logger.error("DeepSeek request failed: %s", e)
            if self._can_fallback_to_ollama():
                logger.info("Falling back to local Ollama after DeepSeek request failure")
                return await self._complete_with_ollama(prompt, system_prompt, max_tokens, temperature, json_mode)
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"LLM complete error: {e}")
            return f"Error: {str(e)}"

    async def _complete_with_ollama(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
        json_mode: bool,
    ) -> str:
        """Use the project's local Ollama model when no cloud API key is configured."""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            payload = {
                "model": self.ollama_model,
                "messages": messages,
                "stream": False,
                "think": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            }
            if json_mode:
                payload["format"] = "json"

            response = await self.client.post(f"{self.ollama_base_url}/api/chat", json=payload)
            response.raise_for_status()
            result = response.json()
            self.last_done_reason = result.get("done_reason")
            return self._clean_response((result.get("message") or {}).get("content", ""))
        except httpx.RequestError as error:
            logger.error("Ollama request failed: %s", error)
            return f"Error: {str(error)}"
        except Exception as error:
            logger.error("Ollama completion failed: %s", error)
            return f"Error: {str(error)}"

    async def chat(self, messages: list, max_tokens: int = 8192, temperature: float = 0.7) -> str:
        self.last_done_reason = None
        if not self._has_deepseek_credentials():
            try:
                response = await self.client.post(
                    f"{self.ollama_base_url}/api/chat",
                    json={
                        "model": self.ollama_model,
                        "messages": messages,
                        "stream": False,
                        "options": {"temperature": temperature, "num_predict": max_tokens},
                    },
                )
                response.raise_for_status()
                result = response.json()
                self.last_done_reason = result.get("done_reason")
                return self._clean_response((result.get("message") or {}).get("content", ""))
            except httpx.RequestError as error:
                logger.error("Ollama chat request failed: %s", error)
                return f"Error: {str(error)}"
            except Exception as error:
                logger.error("Ollama chat failed: %s", error)
                return f"Error: {str(error)}"
        try:
            payload = {
                "model": self.deepseek_model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

            response = await self.client.post(
                self._chat_completions_url,
                json=payload,
                headers=self._headers(),
            )
            response.raise_for_status()
            result = response.json()
            choice = result.get("choices", [{}])[0]
            self.last_done_reason = choice.get("finish_reason")
            response_text = choice.get("message", {}).get("content", "")
            cleaned_response = self._clean_response(response_text)
            logger.debug("DeepSeek chat response: %s", cleaned_response[:500])
            return cleaned_response
        except httpx.HTTPStatusError as e:
            logger.error("DeepSeek chat returned HTTP %s: %s", e.response.status_code, self._extract_error_message(e.response))
            if not self._can_fallback_to_ollama():
                return self._format_deepseek_status_error(e)
            try:
                response = await self.client.post(
                    f"{self.ollama_base_url}/api/chat",
                    json={
                        "model": self.ollama_model,
                        "messages": messages,
                        "stream": False,
                        "think": False,
                        "options": {"temperature": temperature, "num_predict": max_tokens},
                    },
                )
                response.raise_for_status()
                result = response.json()
                self.last_done_reason = result.get("done_reason")
                return self._clean_response((result.get("message") or {}).get("content", ""))
            except Exception as fallback_error:
                logger.error("Ollama chat fallback failed after DeepSeek HTTP error: %s", fallback_error)
                return self._format_deepseek_status_error(e)
        except httpx.RequestError as e:
            logger.error("DeepSeek chat request failed: %s", e)
            if not self._can_fallback_to_ollama():
                return f"Error: {str(e)}"
            try:
                response = await self.client.post(
                    f"{self.ollama_base_url}/api/chat",
                    json={
                        "model": self.ollama_model,
                        "messages": messages,
                        "stream": False,
                        "think": False,
                        "options": {"temperature": temperature, "num_predict": max_tokens},
                    },
                )
                response.raise_for_status()
                result = response.json()
                self.last_done_reason = result.get("done_reason")
                return self._clean_response((result.get("message") or {}).get("content", ""))
            except Exception as fallback_error:
                logger.error("Ollama chat fallback failed: %s", fallback_error)
                return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"LLM chat error: {e}")
            return f"Error: {str(e)}"
