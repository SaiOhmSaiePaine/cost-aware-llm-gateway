.PHONY: up build down dev test

up:
	-docker stop redis-vector 2>/dev/null || true
	docker compose up -d

build:
	-docker stop redis-vector 2>/dev/null || true
	docker compose up -d --build

down:
	docker compose down

dev:
	./.venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

test:
	./.venv/bin/python -m pytest -v
