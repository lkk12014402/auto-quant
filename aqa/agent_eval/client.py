"""OpenAI-compatible async model client for endpoint-centric (agentic) evaluation.

Ports the role of NeMo Evaluator's ``ModelClient``: a thin async client that talks
to an OpenAI-compatible ``/v1/chat/completions`` (+ ``/v1/completions``) endpoint --
e.g. a quantized model served by ``vllm serve``.  Unlike the lm_eval subprocess
path, agentic/terminal benchmarks drive the model as a *served endpoint*, so a
solver can hold a multi-turn conversation.

The HTTP POST is delegated to an injectable ``transport`` coroutine so the client
(and everything above it) is unit-testable with no server and no httpx.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from ..logging_util import AqaError, get_logger

_log = get_logger("agent_eval.client")

# transport(url, payload, headers) -> decoded JSON response dict.
Transport = Callable[[str, dict, dict], Awaitable[dict]]


async def _httpx_transport(url: str, payload: dict, headers: dict) -> dict:
    """Default transport: POST JSON via httpx (lazily imported)."""
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - exercised only without httpx
        raise AqaError("httpx is required for the live endpoint transport; "
                       "`pip install autoquant-agent[agent]`") from exc
    async with httpx.AsyncClient(timeout=None) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()


@dataclass
class ModelClient:
    """Async OpenAI-compatible client (chat + completion) for a served model."""

    base_url: str
    model: str
    api_key: str | None = None
    timeout: float = 300.0
    max_retries: int = 2
    default_params: dict[str, Any] = field(default_factory=dict)
    transport: Transport | None = None

    def _url(self, path: str) -> str:
        base = self.base_url.rstrip("/")
        if not base.endswith("/v1"):
            base = base + "/v1"
        return f"{base}/{path.lstrip('/')}"

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    async def _post(self, url: str, payload: dict) -> dict:
        transport = self.transport or _httpx_transport
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return await asyncio.wait_for(
                    transport(url, payload, self._headers()), timeout=self.timeout)
            except Exception as exc:  # noqa: BLE001 - retry any transport error
                last = exc
                if attempt < self.max_retries:
                    await asyncio.sleep(0.5 * (attempt + 1))
                    _log.warning("endpoint POST retry %d: %s", attempt + 1, exc)
        raise AqaError(f"endpoint request failed after {self.max_retries + 1} tries: {last}")

    async def chat(self, messages: list[dict[str, str]], **params: Any) -> str:
        """Call ``/chat/completions`` and return the assistant message content."""
        payload = {"model": self.model, "messages": messages,
                   **self.default_params, **params}
        data = await self._post(self._url("chat/completions"), payload)
        return data["choices"][0]["message"]["content"]

    async def complete(self, prompt: str, **params: Any) -> str:
        """Call ``/completions`` and return the generated text."""
        payload = {"model": self.model, "prompt": prompt,
                   **self.default_params, **params}
        data = await self._post(self._url("completions"), payload)
        return data["choices"][0]["text"]
