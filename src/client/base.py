"""Abstract async LLM client interface and shared types."""
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class ChatTurn:
    """Provider-agnostic chat message."""

    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class LLMResult:
    """Normalized completion returned by every client."""

    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class UpstreamError(Exception):
    """Raised when an upstream model call fails (timeout, HTTP error, bad payload)."""

    def __init__(self, provider: str, message: str) -> None:
        super().__init__(f"[{provider}] {message}")
        self.provider = provider


class LLMClient(ABC):
    """Every model backend implements this contract."""

    name: str

    @abstractmethod
    async def complete(
        self,
        messages: Sequence[ChatTurn],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResult:
        """Return a completion or raise UpstreamError."""

    async def aclose(self) -> None:  # noqa: B027 - optional hook
        """Release network resources (override when needed)."""
