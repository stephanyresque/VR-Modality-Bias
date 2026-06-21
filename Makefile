SHELL := /bin/bash
PYTHON ?= python
CONFIG ?= configs/baseline.yaml

.PHONY: help install dev-install test lint format docker-build docker-run smoke baseline phase2 phase2-smoke phase3 phase3-smoke phase3-coherence chair-report block5-validate run-all run-all-smoke clean

help:
	@echo "Targets:"
	@echo "  install        uv pip install --system -e ."
	@echo "  dev-install    uv pip install --system -e .[dev]"
	@echo "  test           run the pytest suite"
	@echo "  lint           ruff check + ruff format --check"
	@echo "  format         ruff format (write changes)"
	@echo "  docker-build   build the reproducible Docker image"
	@echo "  docker-run     open an interactive shell inside the Docker container"
	@echo "  smoke          run the baseline pipeline on a single image (--limit 1)"
	@echo "  baseline       run the full N-image baseline end-to-end (N from configs/baseline.yaml)"
	@echo "  phase2-smoke   quick Phase-2 smoke (1 img, short, alpha=1.3) — confirms entrypoint + IO + log path"
	@echo "  phase2         full Phase-2 sweep (50 imgs * 3 lengths * (OFF + 5 alphas)). Resumable; safe under tmux."
	@echo "  phase3-smoke   quick Phase-3 smoke (1 img, short, OFF + SPARC alpha=1.1) — confirms entrypoint + IO"
	@echo "  phase3-coherence Phase-3 coherence smoke (2 imgs, long, prints captions to stdout for eyeball check)"
	@echo "  phase3         full Phase-3 generation (50 imgs * 3 lengths * (OFF + SPARC alpha=1.1)). Resumable."
	@echo "  chair-report   compute CHAIR + degeneration + pair samples from a phase3 run (stdout)"
	@echo "  block5-validate Block-5 LLaVA end-to-end validation (3 imgs, audit + share_tail + captions)"
	@echo "  run-all-smoke  Block-6 smoke (1 img long, baseline+SPARC) — confirms entrypoint + IO + log path"
	@echo "  run-all        Block-6 full orchestrator (50 imgs × 3 lengths × 2 conds). Resumable, tmux-safe."
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
	$(PYTHON) scripts/02_build_manifest.py        --config $(CONFIG) --overwrite
	$(PYTHON) scripts/03_generate_refs.py         --config $(CONFIG)
	$(PYTHON) scripts/04_collect_hidden_states.py --config $(CONFIG)
	$(PYTHON) scripts/05_compute_metrics.py       --config $(CONFIG)
	$(PYTHON) scripts/06_summarize.py             --config $(CONFIG)
	$(PYTHON) scripts/07_make_plots.py            --config $(CONFIG)
	$(PYTHON) scripts/08_unit_example.py          --config $(CONFIG)

# Phase 2 — resumable alpha sweep, safe under tmux.
#   * Logs to results/runs/<run-name>/logs/phase2.log (no terminal dep)
#   * Re-running picks up where it stopped (skips done cells)
#   * Add OVERWRITE=1 to force recompute
# Override the run dir name via: make phase2 PHASE2_RUN_NAME=my_run
PHASE2_RUN_NAME ?= phase2_alpha_sweep
PHASE2_FLAGS ?=
ifeq ($(OVERWRITE),1)
    PHASE2_FLAGS += --overwrite
endif

phase2-smoke:
	$(PYTHON) scripts/15_phase2_sweep.py --run-name $(PHASE2_RUN_NAME)_smoke --smoke

phase2:
	$(PYTHON) scripts/15_phase2_sweep.py --run-name $(PHASE2_RUN_NAME) $(PHASE2_FLAGS)

# Phase 3 — free caption generation for CHAIR evaluation (baseline vs SPARC α=1.1).
#   * Logs to results/runs/<run-name>/logs/phase3.log
#   * Re-running skips already-generated (image, length, condition) cells
#   * Add OVERWRITE=1 to force regenerate
# Override the run dir name via: make phase3 PHASE3_RUN_NAME=my_run
PHASE3_RUN_NAME ?= phase3
PHASE3_FLAGS ?=
ifeq ($(OVERWRITE),1)
    PHASE3_FLAGS += --overwrite
endif

phase3-smoke:
	$(PYTHON) scripts/18_phase3_generate.py --run-name $(PHASE3_RUN_NAME)_smoke --smoke

# Coherence smoke — 2 imgs on `long`, captions printed to stdout. Use this
# to eyeball whether SPARC (with the official COCO hparams + greedy) stays
# coherent on long captions BEFORE launching the full sweep.
phase3-coherence:
	$(PYTHON) scripts/18_phase3_generate.py --run-name $(PHASE3_RUN_NAME)_coherence --coherence-smoke

phase3:
	$(PYTHON) scripts/18_phase3_generate.py --run-name $(PHASE3_RUN_NAME) $(PHASE3_FLAGS)

# CHAIR report — auto-downloads COCO val2017 annotations if missing.
chair-report:
	$(PYTHON) scripts/17_chair_report.py --run-dir results/runs/$(PHASE3_RUN_NAME) --auto-download

# Block-5 functional validation — LLaVA-1.5-7B end-to-end on 3 long images.
# Audit + baseline (FD-OFF) + SPARC (FD-ON), paired per image, writes to
# results/{diagnostico,avaliacao}/llava-1.5-7b/long/.... Prints captions
# + share_tail + audit verdict. NOT the full 50-image run.
block5-validate:
	$(PYTHON) scripts/22_block5_validate.py --length long --limit 3

# Block-6 — single resumable orchestrator for the full diagnostic + SPARC
# run. 50 imgs × 3 lengths × 2 conditions = 300 cells, written to
# results/{diagnostico,avaliacao}/llava-1.5-7b/<length>/<run-name>/.
# Resume by relaunching the SAME command — done cells are skipped.
# Add OVERWRITE=1 to force recompute, RUN_NAME=foo to use a different run.
RUN_ALL_NAME ?= run_all_v1
RUN_ALL_FLAGS ?=
ifeq ($(OVERWRITE),1)
    RUN_ALL_FLAGS += --overwrite
endif

run-all-smoke:
	$(PYTHON) scripts/23_run_all.py --run-name $(RUN_ALL_NAME) --smoke $(RUN_ALL_FLAGS)

run-all:
	$(PYTHON) scripts/23_run_all.py --run-name $(RUN_ALL_NAME) $(RUN_ALL_FLAGS)

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
