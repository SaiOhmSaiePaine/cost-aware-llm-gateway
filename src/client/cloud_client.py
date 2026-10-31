"""Async Google Gemini client with hard timeout enforcement."""
import asyncio
from collections.abc import Sequence

from google import genai
from google.genai import types

from src.client.base import ChatTurn, LLMClient, LLMResult, UpstreamError


class GeminiClient(LLMClient):
    """Wraps google-genai's native async API (`client.aio`)."""

    def __init__(self, api_key: str | None, model: str, timeout_seconds: float) -> None:
        self.name = f"cloud:{model}"
        self._model = model
        self._timeout = timeout_seconds
        self._client: genai.Client | None = genai.Client(api_key=api_key) if api_key else None

    @staticmethod
    def _to_gemini(messages: Sequence[ChatTurn]) -> tuple[str | None, list[types.Content]]:
        """Split OpenAI-style turns into Gemini's system_instruction + contents."""
        system_parts = [m.content for m in messages if m.role == "system"]
        contents = [
            types.Content(role="model" if m.role == "assistant" else "user", parts=[types.Part(text=m.content)])
            for m in messages
            if m.role != "system"
        ]
        return ("\n".join(system_parts) or None), contents

    async def complete(
        self,
        messages: Sequence[ChatTurn],
        *,
        max_tokens: int | None = 512,
        temperature: float | None = None,
    ) -> LLMResult:
        if self._client is None:
            raise UpstreamError(self.name, "GEMINI_API_KEY is not configured")

        system_instruction, contents = self._to_gemini(messages)
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        try:
            response = await asyncio.wait_for(
                self._client.aio.models.generate_content(model=self._model, contents=contents, config=config),
                timeout=self._timeout,
            )
        except TimeoutError as exc:
            raise UpstreamError(self.name, f"timeout after {self._timeout}s") from exc
        except Exception as exc:  # SDK raises many types (APIError, ClientError, httpx errors)
            raise UpstreamError(self.name, f"{type(exc).__name__}: {exc}") from exc

        if not response.text:
            raise UpstreamError(self.name, "empty response (possibly safety-blocked)")

        usage = response.usage_metadata
        return LLMResult(
            text=response.text,
            model=self._model,
            prompt_tokens=(usage.prompt_token_count or 0) if usage else 0,
            completion_tokens=(usage.candidates_token_count or 0) if usage else 0,
        )
