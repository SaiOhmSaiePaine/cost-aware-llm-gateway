"""POST /v1/chat/completions — the 4-stage gateway pipeline."""
import logging
import time
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Response

from src.api.dependencies import GatewayServices, get_services
from src.api.schemas import (
    AssistantMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    GatewayMetadata,
    Usage,
)
from src.cache.semantic_cache import CacheLookupResult
from src.routing.router import RouteTarget

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])

ServicesDep = Annotated[GatewayServices, Depends(get_services)]


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


def _set_trace_headers(response: Response, route: str, latency_ms: float, cache_hit: bool) -> None:
    response.headers["X-Gateway-Route"] = route
    response.headers["X-Gateway-Latency-Ms"] = str(latency_ms)
    response.headers["X-Gateway-Cache-Hit"] = str(cache_hit).lower()


async def _try_cache(
    services: GatewayServices, prompt: str, system_prompt: str, use_cache: bool
) -> tuple[bytes | None, float | None, CacheLookupResult | None]:
    """Embed + look up. Any failure degrades to a cache bypass, never a 500."""
    embed_start = time.perf_counter()
    try:
        vector = await services.embedder.aembed(prompt)
    except Exception as exc:  # ONNX failures must not take the gateway down
        logger.warning("embedding_failed_cache_bypassed", extra={"error": str(exc)})
        return None, None, None
    embed_ms = _elapsed_ms(embed_start)
    lookup = await services.cache.lookup(prompt, vector, system_prompt=system_prompt) if use_cache else None
    return vector, embed_ms, lookup


@router.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    response_model_exclude_none=True,
)
async def chat_completions(
    body: ChatCompletionRequest,
    response: Response,
    background_tasks: BackgroundTasks,
    services: ServicesDep,
) -> ChatCompletionResponse:
    if body.stream:
        raise HTTPException(status_code=400, detail="stream=true is not supported yet.")

    start = time.perf_counter()
    stats, settings, breaker = services.stats, services.settings, services.router.breaker
    stats.total_requests += 1
    prompt = body.last_user_text
    turns = body.to_turns()
    system_prompt = "\n".join(t.content for t in turns if t.role == "system")

    # --- Stage 1: semantic cache (skipped for outage drills and multi-turn chats) ---
    use_cache = body.is_single_turn and not body.simulate_cloud_outage
    vector, embed_ms, lookup = await _try_cache(services, prompt, system_prompt, use_cache)

    if lookup is not None and lookup.guardrail_blocked:
        stats.guardrail_blocks += 1
    if lookup is not None and lookup.hit and lookup.answer is not None:
        stats.cache_hits += 1
        stats.add_savings(settings.cloud_cost_per_request_usd)
        latency_ms = _elapsed_ms(start)
        _set_trace_headers(response, RouteTarget.CACHE, latency_ms, cache_hit=True)
        return ChatCompletionResponse(
            model=RouteTarget.CACHE,
            choices=[Choice(message=AssistantMessage(content=lookup.answer))],
            gateway_metadata=GatewayMetadata(
                routed_to=RouteTarget.CACHE,
                latency_ms=latency_ms,
                cost_usd=0.0,
                cache_hit=True,
                circuit_breaker_state=breaker.state,
                onnx_embed_ms=embed_ms,
                matched_prompt=lookup.matched_prompt,
                similarity_score=round(lookup.similarity or 0.0, 3),
            ),
        )

    # --- Stage 2: hybrid complexity scoring + explicit model override ---
    complexity = services.router.classify(prompt, vector=vector, model=body.model)

    # --- Stage 3: dispatch with circuit breaker (UpstreamError -> 503 via exception handler) ---
    dispatch = await services.router.dispatch(
        turns,
        complexity.tier,
        body.simulate_cloud_outage,
        max_tokens=body.max_tokens,
        temperature=body.temperature,
    )

    if dispatch.route is RouteTarget.LOCAL:
        stats.local_routed_calls += 1
        stats.add_savings(settings.cloud_cost_per_request_usd - dispatch.cost_usd)
    else:
        stats.cloud_routed_calls += 1
        if dispatch.failover_triggered:
            stats.failover_saves += 1

    # --- Stage 4: background cache write (never cache degraded failover answers) ---
    if use_cache and vector is not None and not dispatch.failover_triggered:
        background_tasks.add_task(
            services.cache.store, prompt, dispatch.result.text, vector, system_prompt
        )

    latency_ms = _elapsed_ms(start)
    _set_trace_headers(response, dispatch.route, latency_ms, cache_hit=False)
    usage = Usage(
        prompt_tokens=dispatch.result.prompt_tokens,
        completion_tokens=dispatch.result.completion_tokens,
        total_tokens=dispatch.result.prompt_tokens + dispatch.result.completion_tokens,
    )
    return ChatCompletionResponse(
        model=dispatch.result.model,
        choices=[Choice(message=AssistantMessage(content=dispatch.result.text))],
        usage=usage,
        gateway_metadata=GatewayMetadata(
            routed_to=dispatch.route,
            latency_ms=latency_ms,
            cost_usd=dispatch.cost_usd,
            cache_hit=False,
            failover_triggered=dispatch.failover_triggered,
            guardrail_blocked=True if lookup is not None and lookup.guardrail_blocked else None,
            circuit_breaker_state=breaker.state,
            onnx_embed_ms=embed_ms,
            matched_prompt=lookup.matched_prompt if lookup is not None and lookup.guardrail_blocked else None,
            similarity_score=round(lookup.similarity, 3) if lookup is not None and lookup.similarity else None,
            complexity_score=complexity.score,
            complexity_breakdown=complexity.breakdown,
        ),
    )
