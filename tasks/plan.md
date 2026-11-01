# Implementation Plan: Cost-Aware LLM Gateway (Phases 2.5, 3 & 4)

## Current Codebase Audit & Stage Assessment

| Phase | Milestone | Status | Evidence in Codebase |
| :--- | :--- | :--- | :--- |
| **Phase 1** | ONNX Embeddings + OpenAI SDK Parity | **Completed** | `src/cache/embedder.py` (`FastEmbed` ONNX), `src/api/schemas.py`, `test_openai_sdk.py` |
| **Phase 2** | Modular `src/` Architecture + Async I/O | **Completed** | `src/{config,api,cache,client,routing,telemetry}`, `redis.asyncio`, `BackgroundTasks`, 31/31 unit tests in `tests/unit/` |
| **Phase 2.5** | Core Routing & Performance Upgrade | **Completed** | Confirmed in `docs/intent/routing-performance.md` (Hybrid Semantic + Lexical Router, `model` override, LRU embed cache, `KNN 3` + system isolation, `max_tokens=512`) |
| **Phase 3** | Production Observability & Containerization | **Planned** | Needs `prometheus-client`, `/metrics`, `config/prometheus.yml`, `config/grafana_dashboard.json`, `Dockerfile`, `docker-compose.yml`, `Makefile` |
| **Phase 4** | Control Plane Dashboard & Automated Benchmarks | **Planned** | Needs `GET /dashboard` UI, `benchmarks/benchmark_gateway.py`, `benchmarks/fixtures/prompts.json`, `tests/integration/test_chat_endpoint.py` |

---

## Overview

We have completed Phases 1 and 2, and completed our `/grill-me` architecture audit (`docs/intent/routing-performance.md`). Before wrapping the system in Phase 3 (Prometheus/Grafana/Docker) and Phase 4 (Control Plane Dashboard & Benchmarks), we will execute **Phase 2.5 (Core Routing & Performance Upgrade)** so routing accuracy and latency are rock-solid.

---

## Dependency Graph

```text
Task 0: Phase 2.5 — Core Routing & Performance Upgrade (classifier, embedder LRU, KNN 3 + system isolation, max_tokens)
    │
    └── Task 1: Prometheus Metrics Module (src/telemetry/metrics.py)
            │
            ├── Task 2: Wire Metrics into Pipeline & Expose GET /metrics + Integration Test
            │       │
            │       ├── Task 3: Prometheus & Grafana Provisioning (config/prometheus.yml + grafana_dashboard.json)
            │       │       │
            │       │       └── Task 4: Dockerfile, docker-compose.yml & Makefile
            │       │
            │       └── Task 5: Interactive Control Plane UI (GET /dashboard)
            │
            └── Task 6: 100-Prompt Benchmark Suite & README Results Table
```

---

## Architecture Decisions

1. **Low-Cardinality Prometheus Labels (`RouteTarget`):**
   - `src/routing/router.py` already defines `RouteTarget` (`REDIS_VECTOR_CACHE`, `LOCAL`, `CLOUD`, `LOCAL_AUTO_FAILOVER_FROM_CLOUD`, `LOCAL_CIRCUIT_OPEN_FALLBACK`). We will use these bounded enum values for `route_target` labels on `gateway_requests_total` and `gateway_request_latency_seconds` to prevent Prometheus label cardinality explosion.
2. **Pre-Warming ONNX Model in Multi-Stage `Dockerfile`:**
   - The `Dockerfile` will pre-download `sentence-transformers/all-MiniLM-L6-v2` during `docker build` so container startups are instantaneous and work without outbound HuggingFace downloads at runtime.
3. **Zero-Build Single-Page Control Plane at `GET /dashboard`:**
   - Mounting the interactive Control Plane & Playground directly on FastAPI (`src/api/dashboard.py` served at `GET /dashboard`) keeps deployment to a single Python service while providing a rich dark-mode UI (live prompt playground, `simulate_cloud_outage` toggle, route trace badges, and real-time cost/latency telemetry polling `/v1/admin/stats`).
4. **Clean Git Hygiene for Draw.io Backup Files:**
   - Ignore `.$*.bkp` swap files in `.gitignore` while keeping `architecture.drawio` tracked.

---

## Task List

### Phase 2.5: Core Routing & Performance Upgrade (`docs/intent/routing-performance.md`)
- [x] **Task 0:** Implement Hybrid Semantic + Lexical Router, explicit `model` override, LRU embedding cache (`<2ms`), `KNN 3` + system-prompt cache isolation, and bounded generation (`max_tokens=512`)

### Phase 3: Production Observability (Prometheus, Grafana & Docker Compose)
- [ ] **Task 1:** Implement Prometheus metrics definitions in `src/telemetry/metrics.py`
- [ ] **Task 2:** Instrument `/v1/chat/completions`, expose `GET /metrics`, and add `tests/integration/test_chat_endpoint.py`
- [ ] **Task 3:** Create Prometheus scrape config (`config/prometheus.yml`) and provisioned Grafana dashboard (`config/grafana_dashboard.json`)
- [ ] **Task 4:** Author multi-stage `Dockerfile`, `docker-compose.yml`, and `Makefile` (plus ignore `.$*.bkp` in `.gitignore`)

### Checkpoint: After Phase 3 (Tasks 1–4)
- [ ] All unit and integration tests pass (`pytest`)
- [ ] `GET /metrics` returns valid Prometheus text exposition format with request counters, latency histograms, and USD saved
- [ ] `docker compose config` validates cleanly and container stack starts
- [ ] Review with human before proceeding to Phase 4

### Phase 4: Startup MVP Control Plane Dashboard & Automated Benchmarks
- [ ] **Task 5:** Build interactive dark-mode Control Plane & Playground at `GET /dashboard`
- [ ] **Task 6:** Build 100-prompt benchmark suite (`benchmarks/fixtures/prompts.json` & `benchmarks/benchmark_gateway.py`) and update `README.md`

### Checkpoint: Complete (After Tasks 5–6)
- [ ] `GET /dashboard` renders live Playground + Cost/Latency telemetry in browser
- [ ] Benchmark script executes and outputs P50/P95 latency, cache hit rate, and cost savings table in `README.md`
- [ ] Ready for final review & GitHub push

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
| :--- | :--- | :--- |
| Containerized gateway cannot reach host Ollama (`localhost:11434`) | High | Configure `LOCAL_BASE_URL=http://host.docker.internal:11434/v1` in `docker-compose.yml` so the gateway container reaches the macOS host Ollama daemon seamlessly. |
| Cold-start ONNX download inside Docker container | Medium | Pre-fetch the `fastembed` ONNX model during the `Dockerfile` build step. |
| Draw.io temporary backup file (`.$architecture.drawio.bkp`) committed by mistake | Low | Add `.$*.bkp` to `.gitignore` in Task 4. |

---

## Open Questions

1. **Control Plane UI (Phase 4, Task 5):** Do you prefer the zero-build single-page dashboard mounted directly at `http://localhost:8000/dashboard` (recommended: instant startup, no `node_modules` overhead), or a separate Next.js container in `frontend/`?
2. **Start Execution:** Shall we begin executing **Phase 3 (Tasks 1–4)** now?

