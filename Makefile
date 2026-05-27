SHELL := /bin/bash
PYTHON ?= python
CONFIG ?= configs/baseline.yaml

.PHONY: help install dev-install test lint format docker-build docker-run smoke baseline clean

help:
	@echo "Targets:"
	@echo "  install        uv pip install --system -e ."
	@echo "  dev-install    uv pip install --system -e .[dev]"
	@echo "  test           run the pytest suite"
	@echo "  lint           ruff check + ruff format --check"
	@echo "  format         ruff format (write changes)"
	@echo "  docker-build   build the reproducible Docker image"
	@echo "  docker-run     open an interactive shell inside the Docker container"
	@echo "  smoke          run the pipeline on a single image (--limit 1)"
	@echo "  baseline       run the full 30-image baseline end-to-end"
	@echo "  clean          remove caches (does NOT touch results/ or data/)"

install:
	uv pip install --system -e .

dev-install:
	uv pip install --system -e ".[dev]"

test:
	pytest tests/

lint:
	ruff check src tests scripts
	ruff format --check src tests scripts

format:
	ruff format src tests scripts
	ruff check --fix src tests scripts

docker-build:
	docker compose -f docker/docker-compose.yml build

docker-run:
	docker compose -f docker/docker-compose.yml run --rm baseline

smoke:
	$(PYTHON) scripts/03_generate_refs.py        --config $(CONFIG) --limit 1
	$(PYTHON) scripts/04_collect_hidden_states.py --config $(CONFIG) --limit 1
	$(PYTHON) scripts/05_compute_metrics.py      --config $(CONFIG) --limit 1

baseline:
	$(PYTHON) scripts/01_prepare_data.py          --config $(CONFIG)
	$(PYTHON) scripts/02_build_manifest.py        --config $(CONFIG)
	$(PYTHON) scripts/03_generate_refs.py         --config $(CONFIG)
	$(PYTHON) scripts/04_collect_hidden_states.py --config $(CONFIG)
	$(PYTHON) scripts/05_compute_metrics.py       --config $(CONFIG)
	$(PYTHON) scripts/06_summarize.py             --config $(CONFIG)
	$(PYTHON) scripts/07_make_plots.py            --config $(CONFIG)

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
