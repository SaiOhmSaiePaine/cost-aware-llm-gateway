"""Async client for any OpenAI-compatible server: Ollama (dev) or vLLM (GPU prod)."""
from collections.abc import Sequence
from typing import Any

import httpx

from src.client.base import ChatTurn, LLMClient, LLMResult, UpstreamError


class OpenAICompatibleClient(LLMClient):
    """Calls POST {base_url}/chat/completions over a pooled httpx.AsyncClient."""

    def __init__(self, base_url: str, model: str, timeout_seconds: float) -> None:
        self.name = f"local:{model}"
        self._model = model
        self._http = httpx.AsyncClient(base_url=base_url, timeout=timeout_seconds)

    async def complete(
        self,
        messages: Sequence[ChatTurn],
        *,
        max_tokens: int | None = 512,
        temperature: float | None = None,
    ) -> LLMResult:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        try:
            response = await self._http.post("/chat/completions", json=payload)
            response.raise_for_status()
            body: dict[str, Any] = response.json()
            text: str = body["choices"][0]["message"]["content"]
        except httpx.TimeoutException as exc:
            raise UpstreamError(self.name, f"timeout: {exc!r}") from exc
        except httpx.HTTPError as exc:
            raise UpstreamError(self.name, f"http error: {exc!r}") from exc
        except (KeyError, IndexError, ValueError) as exc:
            raise UpstreamError(self.name, f"malformed response: {exc!r}") from exc

        usage: dict[str, Any] = body.get("usage") or {}
        return LLMResult(
            text=text,
            model=self._model,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
        )

    async def aclose(self) -> None:
        await self._http.aclose()
