# -*- coding: utf-8 -*-
from typing import Any, Optional
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
        self.ollama_fallback_models = settings.OLLAMA_FALLBACK_MODELS
        timeout = httpx.Timeout(
            settings.DEEPSEEK_TIMEOUT_SECONDS,
            connect=min(20.0, settings.DEEPSEEK_TIMEOUT_SECONDS),
        )
        self.client = httpx.AsyncClient(timeout=timeout)
        proxy_url = settings.DEEPSEEK_PROXY_URL.strip() or None
        self.deepseek_client = httpx.AsyncClient(
            timeout=timeout,
            proxy=proxy_url,
            trust_env=False,
        )
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

    def _ordered_ollama_models(self) -> list[str]:
        configured = [
            model.strip()
            for model in self.ollama_fallback_models.split(",")
            if model.strip()
        ]
        return configured or [self.ollama_model]

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

    @staticmethod
    def _is_retryable_status(status_code: int) -> bool:
        return status_code in {408, 429, 500, 502, 503, 504}

    async def _complete_deepseek(self, payload: dict[str, Any]) -> str:
        """Request DeepSeek with bounded retries for transient transport failures.

        A response that contains reasoning but no user-facing content is not a
        usable answer. Retry it once with an explicit concise-answer instruction
        instead of allowing Agent3's legacy fallback to expose prompt context.
        """
        attempts = max(1, settings.DEEPSEEK_MAX_RETRIES + 1)
        last_error = ""
        retry_payload = payload

        for attempt in range(1, attempts + 1):
            try:
                response = await self.deepseek_client.post(
                    self._chat_completions_url,
                    json=retry_payload,
                    headers=self._headers(),
                )
                response.raise_for_status()
                result = response.json()
                choice = result.get("choices", [{}])[0]
                self.last_done_reason = choice.get("finish_reason")
                message = choice.get("message") or {}
                cleaned_response = self._clean_response(str(message.get("content") or ""))
                reasoning_length = len(str(message.get("reasoning_content") or ""))

                if cleaned_response:
                    logger.info(
                        "DeepSeek response accepted: attempt=%d finish_reason=%s content_chars=%d reasoning_chars=%d",
                        attempt, self.last_done_reason, len(cleaned_response), reasoning_length,
                    )
                    return cleaned_response

                last_error = (
                    "DeepSeek returned empty content "
                    f"(finish_reason={self.last_done_reason or 'unknown'}, reasoning_chars={reasoning_length})"
                )
                logger.warning("%s; attempt=%d/%d", last_error, attempt, attempts)
                # A concise retry is especially important for reasoning models
                # that spend their entire budget before emitting message.content.
                retry_instruction = "请直接输出完整、简洁的最终答案，不要展示推理过程；控制在 180 字以内。"
                if payload.get("response_format"):
                    retry_instruction = "请直接输出符合请求格式的完整 JSON，不要展示推理过程。"
                retry_payload = {
                    **payload,
                    "messages": [
                        *payload["messages"],
                        {
                            "role": "user",
                            "content": retry_instruction,
                        },
                    ],
                    "max_tokens": max(512, min(int(payload["max_tokens"]), 2048)),
                }
            except httpx.HTTPStatusError as error:
                status_code = error.response.status_code
                last_error = self._format_deepseek_status_error(error)
                logger.warning("DeepSeek HTTP failure: attempt=%d/%d status=%d", attempt, attempts, status_code)
                if not self._is_retryable_status(status_code):
                    return last_error
            except httpx.RequestError as error:
                last_error = f"Error: DeepSeek API transport failure: {error}"
                logger.warning("DeepSeek transport failure: attempt=%d/%d error=%s", attempt, attempts, error)
            except (ValueError, KeyError, TypeError) as error:
                last_error = f"Error: DeepSeek API response format invalid: {error}"
                logger.warning("DeepSeek response parse failure: attempt=%d/%d error=%s", attempt, attempts, error)
                break

            if attempt < attempts:
                await asyncio.sleep(min(2 ** (attempt - 1), 3))

        if last_error.startswith("Error:"):
            return last_error
        return f"Error: DeepSeek {last_error or 'API returned no usable answer'}"

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
        logger.debug("Sending DeepSeek prompt: %s", prompt[:200])
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "model": self.deepseek_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        response = await self._complete_deepseek(payload)
        if response.startswith("Error:") and self._can_fallback_to_ollama():
            logger.info("Falling back to local Ollama after DeepSeek failure")
            return await self._complete_with_ollama(prompt, system_prompt, max_tokens, temperature, json_mode)
        return response

    async def _complete_with_ollama(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
        json_mode: bool,
    ) -> str:
        """Use ordered local Ollama models when cloud generation is unavailable."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return await self._complete_with_ollama_messages(
            messages, max_tokens, temperature, json_mode,
        )

    async def _complete_with_ollama_messages(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        json_mode: bool = False,
    ) -> str:
        """Try local models in configured order; an empty final answer is unusable."""
        last_error = ""
        for model in self._ordered_ollama_models():
            try:
                payload: dict[str, Any] = {
                    "model": model,
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
                message = result.get("message") or {}
                content = self._clean_response(str(message.get("content") or ""))
                if content:
                    logger.info("Ollama response accepted from model=%s", model)
                    return content
                thinking = str(message.get("thinking") or "")
                last_error = (
                    f"Ollama model {model} returned no final content "
                    f"(done_reason={self.last_done_reason or 'unknown'}, thinking_chars={len(thinking)})"
                )
                logger.warning("%s; trying next local model", last_error)
            except httpx.RequestError as error:
                last_error = f"Ollama model {model} request failed: {error}"
                logger.warning("%s; trying next local model", last_error)
            except Exception as error:
                last_error = f"Ollama model {model} completion failed: {error}"
                logger.warning("%s; trying next local model", last_error)
        return f"Error: {last_error or 'no configured Ollama model produced a usable answer'}"

    async def chat(self, messages: list, max_tokens: int = 8192, temperature: float = 0.7) -> str:
        self.last_done_reason = None
        if not self._has_deepseek_credentials():
            return await self._complete_with_ollama_messages(messages, max_tokens, temperature)
        try:
            payload = {
                "model": self.deepseek_model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

            response = await self.deepseek_client.post(
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
            return await self._complete_with_ollama_messages(messages, max_tokens, temperature)
        except httpx.RequestError as e:
            logger.error("DeepSeek chat request failed: %s", e)
            if not self._can_fallback_to_ollama():
                return f"Error: {str(e)}"
            return await self._complete_with_ollama_messages(messages, max_tokens, temperature)
        except Exception as e:
            logger.error(f"LLM chat error: {e}")
            return f"Error: {str(e)}"

    def complete_sync(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        json_mode: bool = False,
    ) -> str:
        """Synchronous cloud-then-local completion for tools that cannot await.

        Mirrors :meth:`complete`: prefers cloud DeepSeek, and on failure falls
        back to the local Ollama model (deepseek-r1:7b by default) when
        ``LLM_FALLBACK_PROVIDER=ollama``. Returns the answer text, or
        ``"Error: ..."`` when no provider yields a usable answer.
        """
        # No cloud credentials configured: go straight to the local model,
        # matching the async complete() behavior (no fallback flag required).
        if not self._has_deepseek_credentials():
            return self._complete_with_ollama_sync(messages, max_tokens, temperature, json_mode)

        payload: dict[str, Any] = {
            "model": self.deepseek_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            with httpx.Client(
                timeout=settings.DEEPSEEK_TIMEOUT_SECONDS,
                proxy=(settings.DEEPSEEK_PROXY_URL.strip() or None),
                trust_env=False,
            ) as client:
                resp = client.post(
                    self._chat_completions_url,
                    json=payload,
                    headers=self._headers(),
                )
                resp.raise_for_status()
                choice = (resp.json().get("choices") or [{}])[0]
                content = self._clean_response(
                    str((choice.get("message") or {}).get("content") or "")
                )
            if content:
                logger.info(
                    "DeepSeek sync response accepted: finish_reason=%s content_chars=%d",
                    choice.get("finish_reason"), len(content),
                )
                return content
            logger.warning(
                "DeepSeek sync returned empty content (finish_reason=%s)",
                choice.get("finish_reason"),
            )
        except httpx.HTTPStatusError as error:
            status_code = error.response.status_code
            logger.warning("DeepSeek sync HTTP failure: status=%d", status_code)
            if not self._can_fallback_to_ollama():
                return self._format_deepseek_status_error(error)
        except (httpx.RequestError, ValueError, KeyError, TypeError) as error:
            logger.warning("DeepSeek sync failure: %s", error)
            if not self._can_fallback_to_ollama():
                return f"Error: DeepSeek sync failure: {error}"

        # Cloud failed (HTTP error, transport error, or empty content) and the
        # local fallback is enabled: switch to the local deepseek-r1:7b model.
        if self._can_fallback_to_ollama():
            logger.info("Falling back to local Ollama after DeepSeek sync failure")
            return self._complete_with_ollama_sync(messages, max_tokens, temperature, json_mode)
        return "Error: DeepSeek sync returned no usable answer"

    def _complete_with_ollama_sync(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        json_mode: bool,
    ) -> str:
        """Synchronous local Ollama call mirroring the async fallback path."""
        last_error = ""
        for model in self._ordered_ollama_models():
            try:
                payload: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "think": False,
                    "options": {"temperature": temperature, "num_predict": max_tokens},
                }
                if json_mode:
                    payload["format"] = "json"
                with httpx.Client(timeout=settings.DEEPSEEK_TIMEOUT_SECONDS, trust_env=False) as client:
                    resp = client.post(f"{self.ollama_base_url}/api/chat", json=payload)
                    resp.raise_for_status()
                    result = resp.json()
                message = result.get("message") or {}
                content = self._clean_response(str(message.get("content") or ""))
                if content:
                    logger.info("Ollama sync response accepted from model=%s", model)
                    return content
                last_error = (
                    f"Ollama model {model} returned no final content "
                    f"(done_reason={result.get('done_reason') or 'unknown'})"
                )
                logger.warning("%s; trying next local model", last_error)
            except httpx.RequestError as error:
                last_error = f"Ollama model {model} request failed: {error}"
                logger.warning("%s; trying next local model", last_error)
            except Exception as error:
                last_error = f"Ollama model {model} completion failed: {error}"
                logger.warning("%s; trying next local model", last_error)
        return f"Error: {last_error or 'no configured Ollama model produced a usable answer'}"
