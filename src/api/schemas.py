"""OpenAI-compatible request/response schemas plus gateway telemetry metadata."""
import time
import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.client.base import ChatTurn
from src.routing.classifier import ComplexityBreakdown

_SWAGGER_PLACEHOLDER = "string"


def _is_meaningful(text: str | None) -> bool:
    """Return True if text is non-empty and not Swagger UI's default 'string' placeholder."""
    if text is None:
        return False
    stripped = text.strip()
    return bool(stripped) and stripped != _SWAGGER_PLACEHOLDER


class TextContentPart(BaseModel):
    type: Literal["text"]
    text: str


class ChatMessage(BaseModel):
    role: Literal["user", "system", "developer", "assistant"] = "user"
    content: str | list[TextContentPart]

    @property
    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return "\n".join(part.text for part in self.content)


class ChatCompletionRequest(BaseModel):
    """Accepts the OpenAI shape (`messages`) or the legacy `/docs`-friendly `prompt` field."""

    model_config = ConfigDict(
        extra="ignore",  # tolerate temperature, max_tokens, etc.
        json_schema_extra={
            "examples": [
                {
                    "model": "auto-router",
                    "messages": [
                        {"role": "user", "content": "string"}
                    ],
                    "prompt": "string",
                    "simulate_cloud_outage": False,
                }
            ]
        },
    )

    model: str = "auto-router"
    messages: list[ChatMessage] | None = None
    prompt: str | None = Field(default=None, description="Legacy shortcut for a single user message.")
    max_tokens: int | None = Field(default=None, ge=1, description="Maximum tokens to generate (defaults to 512 on local 1B model).")
    temperature: float | None = Field(default=None, ge=0.0, le=2.0, description="Sampling temperature.")
    stream: bool = False
    simulate_cloud_outage: bool = Field(default=False, description="Force a cloud failure to test failover.")

    @model_validator(mode="after")
    def _require_user_input(self) -> "ChatCompletionRequest":
        # Strip untouched Swagger UI "string" placeholders so `prompt` and `messages` don't collide
        clean_prompt = self.prompt.strip() if _is_meaningful(self.prompt) and self.prompt else None
        clean_messages = [m for m in (self.messages or []) if _is_meaningful(m.text)]

        # If a meaningful `prompt` is provided, use it as the active user turn (preserving any system turns)
        if clean_prompt:
            clean_messages = [m for m in clean_messages if m.role != "user"]
            clean_messages.append(ChatMessage(role="user", content=clean_prompt))

        if not clean_messages and not clean_prompt:
            raise ValueError("Provide a non-empty 'prompt' or 'messages' (replace the default 'string' placeholder).")
        if clean_messages and not any(m.role == "user" for m in clean_messages):
            raise ValueError("'messages' must contain at least one non-empty user message.")

        self.prompt = clean_prompt
        self.messages = clean_messages or None
        return self

    def to_turns(self) -> list[ChatTurn]:
        """Normalize to provider-agnostic turns ('developer' is OpenAI's new name for 'system')."""
        if not self.messages:
            return [ChatTurn(role="user", content=(self.prompt or "").strip())]
        return [
            ChatTurn(role="system" if m.role == "developer" else m.role, content=m.text.strip())
            for m in self.messages
        ]

    @property
    def last_user_text(self) -> str:
        return next(t.content for t in reversed(self.to_turns()) if t.role == "user").strip()

    @property
    def is_single_turn(self) -> bool:
        """Only single-turn requests are cacheable; multi-turn answers depend on history."""
        return not any(t.role == "assistant" for t in self.to_turns())


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class Choice(BaseModel):
    index: int = 0
    message: AssistantMessage
    finish_reason: Literal["stop"] = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class GatewayMetadata(BaseModel):
    routed_to: str
    latency_ms: float
    cost_usd: float
    cache_hit: bool
    failover_triggered: bool = False
    guardrail_blocked: bool | None = None
    circuit_breaker_state: str
    onnx_embed_ms: float | None = None
    matched_prompt: str | None = None
    similarity_score: float | None = None
    complexity_score: float | None = None
    complexity_breakdown: ComplexityBreakdown | None = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    usage: Usage = Field(default_factory=Usage)
    gateway_metadata: GatewayMetadata
