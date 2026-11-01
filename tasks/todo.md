# Task Breakdown & Checklist (Phases 2.5, 3 & 4)

## Phase 2.5: Core Routing & Performance Upgrade (`docs/intent/routing-performance.md`)

## Task 0: Upgrade Routing Classifier, Cache Lookup, and Bounded Generation

**Description:** Implement the 6 confirmed upgrades from `docs/intent/routing-performance.md`:
1. **Hybrid Semantic + Lexical Router (`src/routing/classifier.py`):** Combine expanded lexical/code/math triggers with cosine similarity between the Stage-1 384-d ONNX vector and Complex vs. Simple reference anchor centroids (`<0.1ms` dot product).
2. **Explicit `model` Override (`src/routing/router.py`, `src/api/router.py`):** Respect explicit `model="local"` / `"llama3.2:1b"` and `model="cloud"` / `"gemini-3.5-flash-lite"` while defaulting `model="auto-router"` to the hybrid classifier.
3. **Sub-2ms LRU Embedding Cache (`src/cache/embedder.py`):** Cache normalized prompt embeddings in memory (`@lru_cache(maxsize=1024)`) to eliminate `78–116ms` ONNX thread spikes on repeated prompts.
4. **`KNN 3` Semantic Search (`src/cache/semantic_cache.py`):** Query `LIMIT 0 3` and return the first neighbor above threshold that passes `passes_guardrails`.
5. **System-Prompt Cache Isolation (`src/api/router.py`, `src/cache/semantic_cache.py`):** Include normalized `system`/`developer` prompt context in cache lookups/guardrails so different system personas never collide.
6. **Bounded Generation (`src/api/schemas.py`, `src/client/*.py`):** Add `max_tokens: int | None = 512` and `temperature: float | None = None` to `ChatCompletionRequest` and forward to local and cloud LLM clients.

**Acceptance criteria:**
- [x] Coding, debugging, system design, and math prompts score `>= 0.35` and route to `CLOUD`, while greetings, facts, and short translations score `< 0.35` and route to `LOCAL`.
- [x] Explicit `model="local"` and `model="cloud"` override the classifier.
- [x] Repeated prompt embeddings take `<1ms` (`onnx_embed_ms <= 1.0`).
- [x] `KNN 3` finds a valid second neighbor when neighbor #1 is blocked by `passes_guardrails`, and mismatched `system` prompts do not collide.
- [x] All unit tests in `tests/unit/` pass.

**Verification:**
- [x] Tests pass: `./.venv/bin/python -m pytest -v`

---

## Phase 3: Production Observability (Prometheus, Grafana & Docker Compose)

## Task 1: Implement Prometheus Metrics Module (`src/telemetry/metrics.py`)

**Description:** Install `prometheus-client` into `.venv` (and `pyproject.toml` / `requirements.txt`) and create `src/telemetry/metrics.py` defining the core Prometheus metrics for request routing, end-to-end latency, ONNX embedding latency, guardrail blocks, circuit breaker state, and cumulative USD saved.

**Acceptance criteria:**
- [ ] `prometheus-client` is installed in `.venv` and listed in `pyproject.toml` and `requirements.txt`.
- [ ] `src/telemetry/metrics.py` defines `gateway_requests_total` (Counter), `gateway_request_latency_seconds` (Histogram with buckets from `0.002` to `10.0`s), `gateway_onnx_embed_latency_seconds` (Histogram), `gateway_guardrail_blocks_total` (Counter), `gateway_cost_saved_usd_total` (Counter), and `gateway_circuit_breaker_state` (Gauge).
- [ ] Unit test `tests/unit/test_metrics.py` verifies metric recording and Prometheus payload generation.

**Verification:**
- [ ] Tests pass: `./.venv/bin/python -m pytest tests/unit/test_metrics.py -v`
- [ ] Build succeeds: `./.venv/bin/python -c "from src.telemetry.metrics import render_metrics; print(render_metrics()[0][:200])"`
- [ ] Manual check: Confirm no high-cardinality labels (like raw prompts) are attached to any metric.

**Dependencies:** None

**Files likely touched:**
- `src/telemetry/metrics.py`
- `tests/unit/test_metrics.py`
- `pyproject.toml`
- `requirements.txt`

**Estimated scope:** Small (4 files)

---

## Task 2: Instrument Gateway Pipeline, Expose `GET /metrics`, and Add Integration Tests

**Description:** Wire `src/telemetry/metrics.py` into `POST /v1/chat/completions` (`src/api/router.py`), expose `GET /metrics` in `src/api/health.py`, and create `tests/integration/test_chat_endpoint.py` to test the full HTTP pipeline (cache hit, guardrail block, local route, cloud route, circuit breaker failover, `/health`, `/v1/admin/stats`, and `/metrics`) using mocked Redis/LLM clients.

**Acceptance criteria:**
- [ ] Every `/v1/chat/completions` request records latency, route target, cache hit status, ONNX embedding duration, and cost savings in Prometheus metrics.
- [ ] `GET /metrics` returns HTTP 200 with `text/plain; version=0.0.4` Prometheus exposition format.
- [ ] `tests/integration/test_chat_endpoint.py` covers end-to-end request flows without requiring external network access.

**Verification:**
- [ ] Tests pass: `./.venv/bin/python -m pytest -v`
- [ ] Build succeeds: `./.venv/bin/python -c "from src.main import app"`
- [ ] Manual check: `curl http://localhost:8000/metrics` displays `gateway_requests_total` and `gateway_request_latency_seconds_bucket`.

**Dependencies:** Task 1

**Files likely touched:**
- `src/api/router.py`
- `src/api/health.py`
- `tests/integration/test_chat_endpoint.py`

**Estimated scope:** Medium (3 files)

---

## Task 3: Prometheus Scrape Config & Provisioned Grafana Dashboard

**Description:** Create `config/prometheus.yml` to scrape `gateway:8000/metrics` every 5 seconds, along with Grafana datasource/dashboard provisioning files and `config/grafana_dashboard.json` visualizing P50/P95 Latency, Cache Hit Ratio, Route Distribution, and Cumulative USD Saved.

**Acceptance criteria:**
- [ ] `config/prometheus.yml` configures a 5s scrape interval targeting `gateway:8000`.
- [ ] `config/grafana_dashboard.json` and provisioning configs define panels for request rate by route, P50/P95 latency, cache hit %, and total USD saved.

**Verification:**
- [ ] Tests pass: `./.venv/bin/python -m json.tool config/grafana_dashboard.json > /dev/null`
- [ ] Build succeeds: Validate YAML/JSON syntax of all files in `config/`.
- [ ] Manual check: Inspect panel queries in `config/grafana_dashboard.json` to ensure metric names match `src/telemetry/metrics.py`.

**Dependencies:** Task 2

**Files likely touched:**
- `config/prometheus.yml`
- `config/grafana_datasources.yml`
- `config/grafana_dashboards.yml`
- `config/grafana_dashboard.json`

**Estimated scope:** Medium (4 files)

---

## Task 4: Containerization (`Dockerfile`, `docker-compose.yml`, `Makefile`)

**Description:** Create a multi-stage `Dockerfile` (with pre-cached ONNX embedding weights), `docker-compose.yml` orchestrating `gateway`, `redis-stack`, `prometheus`, and `grafana`, a developer `Makefile` (`make up`, `make down`, `make test`, `make benchmark`), and update `.gitignore` to ignore draw.io backup files (`.$*.bkp`).

**Acceptance criteria:**
- [ ] `Dockerfile` builds a slim Python 3.11 image without PyTorch bloat and pre-warms the `fastembed` ONNX model.
- [ ] `docker-compose.yml` defines `gateway` (port 8000), `redis` (ports 6379/8001), `prometheus` (port 9090), and `grafana` (port 3000).
- [ ] `Makefile` provides `up`, `down`, `dev`, `test`, and `benchmark` targets.
- [ ] `.gitignore` excludes `.$*.bkp`.

**Verification:**
- [ ] Tests pass: `./.venv/bin/python -m pytest -v`
- [ ] Build succeeds: `docker compose config -q`
- [ ] Manual check: Verify `git status` no longer shows `.$architecture.drawio.bkp` as untracked.

**Dependencies:** Task 3

**Files likely touched:**
- `Dockerfile`
- `.dockerignore`
- `docker-compose.yml`
- `Makefile`
- `.gitignore`

**Estimated scope:** Medium (5 files)

---

## Checkpoint: After Tasks 1–4 (End of Phase 3)
- [ ] All unit and integration tests pass (`pytest`)
- [ ] `GET /metrics` exposes live Prometheus counters and histograms
- [ ] `docker compose config` validates without errors
- [ ] Review with human before proceeding to Phase 4

---

## Phase 4: Startup MVP Control Plane Dashboard & Automated Benchmarks

## Task 5: Interactive Control Plane & Live Playground (`GET /dashboard`)

**Description:** Build a dark-mode single-page Control Plane & Playground mounted at `GET /dashboard` (`src/api/dashboard.py`) where users can send prompts, toggle `simulate_cloud_outage`, inspect route badges (`REDIS_VECTOR_CACHE`, `LOCAL`, `CLOUD`, `FAILOVER`), view complexity score breakdowns, and watch live KPI cards and charts update from `/v1/admin/stats`.

**Acceptance criteria:**
- [ ] `GET /dashboard` returns an interactive HTML/Tailwind/Chart.js dashboard.
- [ ] Users can submit prompts, toggle `simulate_cloud_outage`, and inspect latency, similarity score, complexity breakdown, and response headers.
- [ ] Live telemetry cards and charts auto-refresh from `GET /v1/admin/stats` and `GET /health`.

**Verification:**
- [ ] Tests pass: `./.venv/bin/python -m pytest -v`
- [ ] Build succeeds: `curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/dashboard` returns `200`.
- [ ] Manual check: Open `http://localhost:8000/dashboard` in Chrome DevTools MCP and test sending a prompt and viewing live stats.

**Dependencies:** Task 2

**Files likely touched:**
- `src/api/dashboard.py`
- `src/main.py`
- `tests/integration/test_chat_endpoint.py`

**Estimated scope:** Small (3 files)

---

## Task 6: 100-Prompt Benchmark Suite & README Results

**Description:** Create `benchmarks/fixtures/prompts.json` (100 prompts across paraphrased cache pairs, numeric/negation guardrail traps, simple local tasks, and complex cloud tasks) and `benchmarks/benchmark_gateway.py` to measure P50/P95 latency, cache hit rate, guardrail blocks, and total USD saved, then embed the benchmark results table into `README.md`.

**Acceptance criteria:**
- [ ] `benchmarks/fixtures/prompts.json` contains 100 categorized test prompts.
- [ ] `benchmarks/benchmark_gateway.py` runs asynchronously against the gateway and prints a Markdown summary table (P50/P95 Latency, Cache Hit %, Cost Saved %).
- [ ] `README.md` is updated with the benchmark results table and dashboard instructions.

**Verification:**
- [ ] Tests pass: `./.venv/bin/python -m pytest -v`
- [ ] Build succeeds: `./.venv/bin/python benchmarks/benchmark_gateway.py --help`
- [ ] Manual check: Run benchmark against the local gateway and verify the Markdown table in `README.md`.

**Dependencies:** Task 2

**Files likely touched:**
- `benchmarks/fixtures/prompts.json`
- `benchmarks/benchmark_gateway.py`
- `README.md`

**Estimated scope:** Medium (3 files)

---

## Checkpoint: Complete (After Tasks 5–6)
- [ ] All acceptance criteria met across Phases 3 & 4
- [ ] Full `pytest` suite passes (unit + integration)
- [ ] Ready for final review and `git push`

