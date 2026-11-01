# Statement of Intent: Phase 2.5 — Core Routing & Performance Upgrade

**Date:** 2026-10-30  
**Status:** Confirmed via `/grill-me` interview  

---

## 1. Core Intent

- **Outcome:** Upgrade the LLM Gateway with a **Hybrid Semantic + Lexical Router** (reusing Stage-1 ONNX vectors at `<0.1ms` extra latency), explicit `model` override support (`auto-router`, `local`, `cloud`), an in-memory LRU embedding cache (`<2ms` repeat cache hits), `KNN 3` semantic search with system-prompt isolation, and bounded generation (`max_tokens=512`).
- **User:** Both portfolio reviewers testing live in Swagger / Playground and developers pointing the official `openai` Python SDK at `http://localhost:8000/v1` for local coding/chat workflows.
- **Why now:** Before adding Phase 3 Docker/Grafana wrappers, the core routing engine must stop misrouting real coding/debugging/architecture prompts (e.g., `"Write a Python binary search function"` or `"Why does my React useEffect loop?"`) to the weak 1B local model and eliminate `78–116ms` ONNX repeat latency spikes and `14–28s` unbounded local rambling.
- **Success:**
  1. Coding, debugging, system design, and math prompts reliably route to **Cloud (`gemini-3.5-flash-lite`)** while greetings, facts, and short translations stay on **Local (`llama3.2:1b`)**.
  2. Explicit `model` values (`"local"`, `"cloud"`, `"auto-router"`) are respected.
  3. Repeated cache hits drop from `~100ms` to `<5ms` end-to-end (`onnx_embed_ms <= 1ms`).
  4. `KNN 3` prevents false cache misses when rank #1 is blocked by a numeric/negation guardrail, and different `system` prompts no longer collide in cache.
- **Constraint:** Zero extra ML model downloads or external API calls for routing—reuse the existing 384-dim `all-MiniLM-L6-v2` ONNX embedder already running in Stage 1.
- **Out of scope:** Fine-tuning a custom neural classifier, training external router models, or modifying Phase 3/4 Docker/Grafana deliverables before Phase 2.5 is verified.

---

## 2. Root-Cause Audit & Phase 2.5 Specification

| # | Area | Root Cause Identified in `src/` | Phase 2.5 Solution | Files Touched |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **Hybrid Semantic + Lexical Router** | [`src/routing/classifier.py`](../../src/routing/classifier.py) relies on a 16-word `REASONING_KEYWORDS` set (`0.15` per hit) and literal `"def "`/`"class "` markers. Prompts like `"Write a Python binary search function"` (`0.008`) or `"Why does my React useEffect cause an infinite loop?"` (`0.012`) score `< 0.35` and misroute to the 1B local model. | Combine expanded lexical/code/math signals with **Semantic Prototype Routing**: compute cosine similarity between the already-computed 384-d prompt embedding and pre-computed Complex vs. Simple reference anchor centroids (`<0.1ms` numpy dot product). | `src/routing/classifier.py`, `src/routing/router.py`, `src/api/router.py` |
| **2** | **Explicit `model` Override Support** | [`src/api/router.py`](../../src/api/router.py) ignores `body.model` and always auto-routes, even when an OpenAI SDK caller explicitly passes `model="local"` or `model="cloud"`. | Default `model="auto-router"` to the hybrid classifier, while honoring explicit local (`"local"`, `"llama3.2:1b"`) or cloud (`"cloud"`, `"gemini-3.5-flash-lite"`) overrides. | `src/api/schemas.py`, `src/routing/router.py`, `src/api/router.py` |
| **3** | **Sub-2ms LRU Embedding Cache** | [`src/cache/embedder.py`](../../src/cache/embedder.py) runs `asyncio.to_thread(self.embed, text)` on every request, causing `78–116ms` ONNX thread-wakeup spikes even on repeated prompts. | Add an in-memory `@lru_cache(maxsize=1024)` on normalized prompt text so repeated prompts return their 384-d `bytes` vector in `<0.1ms`, cutting repeat Redis cache hits from `~100ms` to `<5ms`. | `src/cache/embedder.py` |
| **4** | **`KNN 3` Semantic Search After Guardrail Block** | [`src/cache/semantic_cache.py`](../../src/cache/semantic_cache.py) queries `@embedding:[VECTOR_RANGE $radius $vec]=>{$YIELD_DISTANCE_AS: dist}` with `LIMIT 0 1`. If candidate #1 is blocked by `passes_guardrails` (e.g., `"tip on $60"`), candidate #2 (`"tip on $90"`) is never inspected. | Fetch up to 3 nearest neighbors (`LIMIT 0 3`) sorted by `@dist` and return the first candidate that meets the similarity threshold and passes `passes_guardrails`. | `src/cache/semantic_cache.py` |
| **5** | **System-Prompt Cache Isolation** | [`src/api/router.py`](../../src/api/router.py) embeds and caches only `last_user_text`, ignoring `system`/`developer` instructions. Different system personas asking the same user question collide in cache. | Incorporate normalized `system`/`developer` context into the cache lookup/guardrail check so distinct system personas never return each other's cached responses. | `src/api/router.py`, `src/cache/semantic_cache.py` |
| **6** | **Bounded Generation (`max_tokens` & `temperature`)** | [`src/client/vllm_client.py`](../../src/client/vllm_client.py) and [`src/client/cloud_client.py`](../../src/client/cloud_client.py) do not set `max_tokens`, allowing `llama3.2:1b` to ramble for `14–28s` during failover. | Add `max_tokens: int | None = 512` and `temperature: float | None = None` to `ChatCompletionRequest` and forward them to both Ollama/vLLM and Gemini clients. | `src/api/schemas.py`, `src/client/base.py`, `src/client/vllm_client.py`, `src/client/cloud_client.py` |

