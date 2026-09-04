# -*- coding: utf-8 -*-
from typing import Any, Callable, Optional
import asyncio
import httpcore
import httpx
import json
import re
from config import settings
import logging

logger = logging.getLogger(__name__)


class LLMUnavailableError(RuntimeError):
    """Raised when the configured primary LLM fails and fallback is disabled."""


class _DeepSeekAsyncNetworkBackend(httpcore.AsyncNetworkBackend):
    def __init__(self, hostname: str, fallback_ips: list[str]):
        self.hostname = hostname
        self.fallback_ips = fallback_ips
        self.backend = httpcore.AnyIOBackend()

    async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        candidates = [*self.fallback_ips, host] if host == self.hostname else [host]
        last_error = None
        candidate_timeout = min(timeout or 10.0, 8.0)
        for candidate in candidates:
            try:
                return await self.backend.connect_tcp(
                    candidate,
                    port,
                    timeout=candidate_timeout,
                    local_address=local_address,
                    socket_options=socket_options,
                )
            except (httpcore.ConnectError, httpcore.ConnectTimeout) as error:
                last_error = error
                logger.warning(
                    "DeepSeek TCP candidate failed: hostname=%s candidate=%s error=%r",
                    host,
                    candidate,
                    error,
                )
        if last_error is not None:
            raise last_error
        raise httpcore.ConnectError(f"No TCP candidate available for {host}")

    async def connect_unix_socket(self, path, timeout=None, socket_options=None):
        return await self.backend.connect_unix_socket(path, timeout, socket_options)

    async def sleep(self, seconds):
        await self.backend.sleep(seconds)


class _DeepSeekSyncNetworkBackend(httpcore.NetworkBackend):
    def __init__(self, hostname: str, fallback_ips: list[str]):
        self.hostname = hostname
        self.fallback_ips = fallback_ips
        self.backend = httpcore.SyncBackend()

    def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        candidates = [*self.fallback_ips, host] if host == self.hostname else [host]
        last_error = None
        candidate_timeout = min(timeout or 10.0, 8.0)
        for candidate in candidates:
            try:
                return self.backend.connect_tcp(
                    candidate,
                    port,
                    timeout=candidate_timeout,
                    local_address=local_address,
                    socket_options=socket_options,
                )
            except (httpcore.ConnectError, httpcore.ConnectTimeout) as error:
                last_error = error
                logger.warning(
                    "DeepSeek sync TCP candidate failed: hostname=%s candidate=%s error=%r",
                    host,
                    candidate,
                    error,
                )
        if last_error is not None:
            raise last_error
        raise httpcore.ConnectError(f"No TCP candidate available for {host}")

    def connect_unix_socket(self, path, timeout=None, socket_options=None):
        return self.backend.connect_unix_socket(path, timeout, socket_options)


class LLMClient:
    """Hybrid LLM entrypoint: cloud-first (DeepSeek / OpenAI-compatible) with a
    local Ollama ordered fallback.

    Cloud-skipping behaviour:
      * When ``LLM_PROVIDER_OVERRIDE`` is set to ``"ollama"`` (typically written
        by the launcher script after detecting that DeepSeek credentials are
        placeholders or missing), this client **never** attempts a cloud call
        and routes every request straight to the local Ollama ordered list.
        This is a *persistent, once-for-all* decision for the process lifetime.
      * When credentials are missing / still templated placeholders, this client
        skips cloud for the remainder of this process run (credentials cannot
        heal themselves at runtime without a backend restart).
      * When a cloud call fails due to HTTP/network/empty-body issues, the
        client falls back locally once, but does **not** remember the failure
        as a permanent "cloud is down" state.  Subsequent calls still try cloud
        first — transient network flaps / momentary 408/5xx should not prevent
        a working key from being used the next time.
    """

    def __init__(self):
        # 1) Apply launcher-side provider override BEFORE any provider selection.
        override = (settings.model_dump().get("LLM_PROVIDER_OVERRIDE") or getattr(settings, "LLM_PROVIDER_OVERRIDE", "")).lower().strip()
        self._provider_override: Optional[str] = override or None

        base_provider = settings.LLM_PROVIDER.lower().strip()
        # If override forces a specific provider for the entire backend run,
        # use it as cloud_provider.  "" means "use LLM_PROVIDER as usual".
        effective_cloud_provider = self._provider_override or base_provider
        self.cloud_provider = effective_cloud_provider
        self._uses_openai_compatible_cloud = self.cloud_provider in {
            "openai", "openai_compatible", "custom_openai_compatible",
        }
        # Keep the historic attribute names because integrations and tests use
        # them, but bind them to the selected cloud provider at runtime.
        self.deepseek_api_key = (
            settings.OPENAI_COMPATIBLE_API_KEY if self._uses_openai_compatible_cloud
            else settings.DEEPSEEK_API_KEY
        )
        self.deepseek_api_base = (
            settings.OPENAI_COMPATIBLE_API_BASE if self._uses_openai_compatible_cloud
            else settings.DEEPSEEK_API_BASE
        ).rstrip("/")
        self.deepseek_model = (
            settings.OPENAI_COMPATIBLE_MODEL if self._uses_openai_compatible_cloud
            else settings.DEEPSEEK_MODEL
        )
        self.ollama_base_url = settings.OLLAMA_BASE_URL.rstrip("/")
        self.ollama_model = settings.OLLAMA_MODEL
        self.ollama_fallback_models = settings.OLLAMA_FALLBACK_MODELS
        timeout = httpx.Timeout(
            self._cloud_timeout_seconds(),
            connect=min(20.0, self._cloud_timeout_seconds()),
        )
        self.client = httpx.AsyncClient(timeout=timeout, trust_env=False)
        proxy_url = settings.DEEPSEEK_PROXY_URL.strip() or None
        self.deepseek_client = self._build_async_deepseek_client(timeout, proxy_url)
        self.last_done_reason: Optional[str] = None
        self.last_provider: Optional[str] = None
        self.last_model: Optional[str] = None
        self.last_fallback_reason: Optional[str] = None

        # ================= Cloud-skipping caches (process-lifetime) ==============
        # Latch 1 – FORCE-LOCAL set by launcher override.  If True, not a single
        #           cloud call is attempted for the entire lifetime of this process.
        self._cloud_force_local: bool = (self._provider_override == "ollama")
        # Latch 2 – credentials missing.  Evaluated lazily once and cached so we
        #           don't scan the key + placeholder tokens on every call.
        #           Note: credentials never "self-heal" at runtime — you must
        #           restart the backend after editing .env, so caching is safe.
        self._cloud_credentials_checked: bool = False
        self._cloud_credentials_ok: bool = False
        # NOTE: We intentionally do NOT keep a "runtime cloud failure latch".
        #       HTTP/408/5xx/network flaps must not cause a good key to be
        #       locked out for the entire process run.  Per-call fallback plus
        #       the harness concurrency shrink via _notify_fallback is enough.

        if self._cloud_force_local:
            logger.info(
                "LLM_PROVIDER_OVERRIDE=ollama is active: this backend will route "
                "ALL LLM calls directly to local Ollama and never attempt a "
                "cloud DeepSeek connection for this process lifetime."
            )

        # Optional orchestration hook. It is intentionally process-local and
        # is used to adapt concurrency after a runtime cloud failure.
        self._fallback_listener: Optional[Callable[[str], None]] = None

    def set_fallback_listener(self, listener: Optional[Callable[[str], None]]) -> None:
        """Notify the owning pipeline when a cloud call falls back locally."""
        self._fallback_listener = listener

    def _notify_fallback(self, reason: str) -> None:
        listener = self._fallback_listener
        if listener is None:
            return
        try:
            listener(reason)
        except Exception:
            logger.exception("Fallback listener failed")

    def _deepseek_fallback_ips(self) -> list[str]:
        return [
            value.strip()
            for value in settings.DEEPSEEK_FALLBACK_IPS.split(",")
            if value.strip()
        ]

    def _cloud_timeout_seconds(self) -> float:
        return (
            settings.OPENAI_COMPATIBLE_TIMEOUT_SECONDS
            if self._uses_openai_compatible_cloud
            else settings.DEEPSEEK_TIMEOUT_SECONDS
        )

    def _cloud_max_retries(self) -> int:
        return (
            settings.OPENAI_COMPATIBLE_MAX_RETRIES
            if self._uses_openai_compatible_cloud
            else settings.DEEPSEEK_MAX_RETRIES
        )

    def _build_async_deepseek_client(
        self, timeout: httpx.Timeout, proxy_url: Optional[str]
    ) -> httpx.AsyncClient:
        if proxy_url:
            return httpx.AsyncClient(timeout=timeout, proxy=proxy_url, trust_env=False)
        # Allow each machine's standard HTTPS_PROXY/HTTP_PROXY settings when
        # no project-specific proxy is supplied. This keeps deployments
        # portable and is required on networks where direct 443 is blocked.
        use_system_proxy = bool(settings.DEEPSEEK_USE_SYSTEM_PROXY)
        transport = httpx.AsyncHTTPTransport(trust_env=use_system_proxy)
        fallback_ips = self._deepseek_fallback_ips()
        if fallback_ips:
            hostname = httpx.URL(self.deepseek_api_base).host
            transport._pool._network_backend = _DeepSeekAsyncNetworkBackend(
                hostname, fallback_ips
            )
        return httpx.AsyncClient(timeout=timeout, transport=transport)

    def _build_sync_deepseek_client(self) -> httpx.Client:
        proxy_url = settings.DEEPSEEK_PROXY_URL.strip() or None
        if proxy_url:
            return httpx.Client(
                timeout=self._cloud_timeout_seconds(),
                proxy=proxy_url,
                trust_env=False,
            )
        transport = httpx.HTTPTransport(trust_env=bool(settings.DEEPSEEK_USE_SYSTEM_PROXY))
        fallback_ips = self._deepseek_fallback_ips()
        if fallback_ips:
            hostname = httpx.URL(self.deepseek_api_base).host
            transport._pool._network_backend = _DeepSeekSyncNetworkBackend(
                hostname, fallback_ips
            )
        return httpx.Client(
            timeout=self._cloud_timeout_seconds(),
            transport=transport,
        )

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
        """Cached credential check.  Runs once per process lifetime."""
        if self._cloud_credentials_checked:
            return self._cloud_credentials_ok
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
        result = (
            self.cloud_provider in {"deepseek", "openai", "openai_compatible", "custom_openai_compatible"}
            and bool(self.deepseek_api_key)
            and normalized not in placeholder_tokens
            and not normalized.startswith("your-")
            and not normalized.startswith("your_")
        )
        self._cloud_credentials_checked = True
        self._cloud_credentials_ok = result
        if not result:
            logger.info(
                "Cloud credentials missing/placeholder — skipping cloud for "
                "the rest of this process run (cached until restart)."
            )
        return result

    def _should_skip_cloud(self) -> tuple[bool, Optional[str]]:
        """Central check — only two latches (both deterministic, runtime-safe):

        - ``LLM_PROVIDER_OVERRIDE=ollama`` from the launcher (config is known to
          be missing / not replaced at launch time).
        - credentials check found placeholders / empty (credentials don't heal
          mid-process, so once bad, always bad).

        Returns (skip, reason_or_None).  Importantly, this intentionally does
        **not** latch on transient HTTP/network failures — those must only
        trigger a one-shot local fallback for the current call.
        """
        if self._cloud_force_local:
            return True, "force_local_via_LLM_PROVIDER_OVERRIDE"
        # Force a credential refresh on the first call; it caches itself.
        if self._cloud_credentials_checked and not self._cloud_credentials_ok:
            return True, "deepseek_credentials_missing (cached)"
        return False, None

    # (Historical helper retained for grep-ability – not used anymore, because
    #  we never permanently disable cloud on a single runtime failure.  The
    #  equivalent helper is just the per-call fallback + _notify_fallback.)
    def _mark_runtime_fallback(self, reason: str) -> None:
        """Single-shot fallback: log, shrink concurrency via listener, done."""
        logger.info("Cloud call fell back to local Ollama: %s", reason)
        self._notify_fallback(reason)

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
        instead of allowing 个性化学习资源生成 Agent's legacy fallback to expose prompt context.
        """
        attempts = max(1, self._cloud_max_retries() + 1)
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
                    self.last_provider = self.cloud_provider
                    self.last_model = self.deepseek_model
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
        self.last_provider = None
        self.last_model = None
        self.last_fallback_reason = None

        # ------------------------------------------------------------------
        # Latch-based cloud skipping.  Two deterministic latches are checked:
        #   1) launcher forced LLM_PROVIDER_OVERRIDE=ollama → never try cloud
        #   2) credentials known missing/placeholder (cached once)
        # HTTP/network failures never flip a permanent latch: subsequent calls
        # always retry cloud first, then fall back locally if unreachable.
        # ------------------------------------------------------------------
        skip, reason = self._should_skip_cloud()
        if skip:
            if self._can_fallback_to_ollama():
                self.last_fallback_reason = reason
                return await self._complete_with_ollama(prompt, system_prompt, max_tokens, temperature, json_mode)
            # Local fallback also disabled: surface a clear error.
            return f"Error: Cloud provider skipped ({reason}) and Ollama fallback is not enabled."

        if not self._has_deepseek_credentials():
            if self._can_fallback_to_ollama():
                self.last_fallback_reason = "deepseek_credentials_missing"
                return await self._complete_with_ollama(prompt, system_prompt, max_tokens, temperature, json_mode)
            return "Error: DeepSeek API credentials are not configured."
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
            "thinking": {"type": "disabled"},
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        response = await self._complete_deepseek(payload)
        if response.startswith("Error:") and self._can_fallback_to_ollama():
            self.last_fallback_reason = response
            # One-shot fallback for this call only.  We do NOT permanently lock
            # out cloud here, because a 408/5xx/network flap this time might
            # resolve on the next call when the key itself is valid.
            self._notify_fallback(response)
            logger.info("Falling back to local Ollama after DeepSeek failure: %s", response)
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
                    self.last_provider = "ollama"
                    self.last_model = model
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
        self.last_provider = None
        self.last_model = None
        self.last_fallback_reason = None
        # ---- Latch-based cloud skipping (mirrors `complete`) ----
        skip, reason = self._should_skip_cloud()
        if skip:
            if self._can_fallback_to_ollama():
                self.last_fallback_reason = reason
                return await self._complete_with_ollama_messages(messages, max_tokens, temperature)
            return f"Error: Cloud provider skipped ({reason}) and Ollama fallback is not enabled."

        if not self._has_deepseek_credentials():
            if self._can_fallback_to_ollama():
                self.last_fallback_reason = "deepseek_credentials_missing"
                return await self._complete_with_ollama_messages(messages, max_tokens, temperature)
            return "Error: DeepSeek API credentials are not configured."
        payload = {
            "model": self.deepseek_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        response = await self._complete_deepseek(payload)
        if response.startswith("Error:") and self._can_fallback_to_ollama():
            self.last_fallback_reason = response
            # One-shot fallback (same rationale as complete) – transient failures
            # must not permanently lock out a working key for the remaining run.
            self._notify_fallback(response)
            logger.info("Falling back to local Ollama after DeepSeek chat failure: %s", response)
            return await self._complete_with_ollama_messages(messages, max_tokens, temperature)
        return response

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
        back to the local Ollama model (qwen2.5:7b by default with deepseek-r1:7b
        as the ordered backup) when ``LLM_FALLBACK_PROVIDER=ollama``.  Runtime
        failures trigger a *single-shot* local fallback — we intentionally do
        NOT remember the failure as permanent, because transient network /
        408/5xx issues must not kill a correctly configured cloud key for the
        remainder of the backend process.
        """
        self.last_done_reason = None
        self.last_provider = None
        self.last_fallback_reason = None

        # ---- Latch-based cloud skipping (same as async paths) ----
        skip, reason = self._should_skip_cloud()
        if skip:
            if self._can_fallback_to_ollama():
                self.last_fallback_reason = reason
                return self._complete_with_ollama_sync(messages, max_tokens, temperature, json_mode)
            return f"Error: Cloud provider skipped ({reason}) and Ollama fallback is not enabled."

        if not self._has_deepseek_credentials():
            if self._can_fallback_to_ollama():
                self.last_fallback_reason = "deepseek_credentials_missing"
                return self._complete_with_ollama_sync(messages, max_tokens, temperature, json_mode)
            return "Error: DeepSeek API credentials are not configured."

        payload: dict[str, Any] = {
            "model": self.deepseek_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "thinking": {"type": "disabled"},
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        cloud_failed_reason: Optional[str] = None
        try:
            with self._build_sync_deepseek_client() as client:
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
                self.last_provider = self.cloud_provider
                logger.info(
                    "DeepSeek sync response accepted: finish_reason=%s content_chars=%d",
                    choice.get("finish_reason"), len(content),
                )
                return content
            logger.warning(
                "DeepSeek sync returned empty content (finish_reason=%s)",
                choice.get("finish_reason"),
            )
            cloud_failed_reason = "Error: DeepSeek returned empty content"
        except httpx.HTTPStatusError as error:
            status_code = error.response.status_code
            logger.warning("DeepSeek sync HTTP failure: status=%d", status_code)
            formatted_error = self._format_deepseek_status_error(error)
            if not self._can_fallback_to_ollama():
                return formatted_error
            cloud_failed_reason = formatted_error
        except (httpx.RequestError, ValueError, KeyError, TypeError) as error:
            logger.warning("DeepSeek sync failure: %s", error)
            formatted_error = f"Error: DeepSeek sync failure: {error}"
            if not self._can_fallback_to_ollama():
                return formatted_error
            cloud_failed_reason = formatted_error

        # Cloud failed (HTTP error, transport error, or empty content) and the
        # local fallback is enabled: do a single-shot local fallback.
        # Intentionally no permanent latch — the next call still tries cloud.
        if self._can_fallback_to_ollama():
            self.last_fallback_reason = cloud_failed_reason or "deepseek_empty_response"
            if cloud_failed_reason:
                # Notify harness once so it can shrink concurrency, but keep
                # cloud reachable for the next attempt.
                self._notify_fallback(cloud_failed_reason)
            logger.info("Falling back to local Ollama after DeepSeek sync failure: %s", self.last_fallback_reason)
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
                with httpx.Client(timeout=self._cloud_timeout_seconds(), trust_env=False) as client:
                    resp = client.post(f"{self.ollama_base_url}/api/chat", json=payload)
                    resp.raise_for_status()
                    result = resp.json()
                message = result.get("message") or {}
                content = self._clean_response(str(message.get("content") or ""))
                if content:
                    self.last_provider = "ollama"
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
