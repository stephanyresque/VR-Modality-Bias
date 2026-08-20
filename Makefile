SHELL := /bin/bash
PYTHON ?= python
CONFIG ?= configs/baseline.yaml

.PHONY: help install dev-install test lint format docker-build docker-run smoke baseline phase2 phase2-smoke phase3 phase3-smoke phase3-coherence phase4-smolvlm-smoke clean

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
	@echo "  phase3         full Phase-3 generation. Needs SELECTED_LAYER= and SE_LAYERS="
	@echo "  phase4-smolvlm-smoke  Phase-4 coherence smoke for SmolVLM-2.2B + SPARC (Llama variant of attn forward)"
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
	$(PYTHON) scripts/generate_refs.py        --config $(CONFIG) --limit 1
	$(PYTHON) scripts/collect_hidden_states.py --config $(CONFIG) --limit 1
	$(PYTHON) scripts/compute_metrics.py      --config $(CONFIG) --limit 1

baseline:
	$(PYTHON) scripts/prepare_data.py          --config $(CONFIG)
	$(PYTHON) scripts/build_manifest.py        --config $(CONFIG) --overwrite
	$(PYTHON) scripts/generate_refs.py         --config $(CONFIG)
	$(PYTHON) scripts/collect_hidden_states.py --config $(CONFIG)
	$(PYTHON) scripts/compute_metrics.py       --config $(CONFIG)
	$(PYTHON) scripts/summarize.py             --config $(CONFIG)
	$(PYTHON) scripts/make_plots.py            --config $(CONFIG)
	$(PYTHON) scripts/unit_example.py          --config $(CONFIG)

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
	$(PYTHON) scripts/phase2_sweep.py --run-name $(PHASE2_RUN_NAME)_smoke --smoke

phase2:
	$(PYTHON) scripts/phase2_sweep.py --run-name $(PHASE2_RUN_NAME) $(PHASE2_FLAGS)

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

# SELECTED_LAYER and SE_LAYERS deliberately have no defaults. The reference
# layer is an experimental result for one model, derived with
# scripts/select_reference_layer.py; baking a number in here would freeze an
# experiment decision into the build file, and a value carried over from a
# deeper backbone produces a plausible run that is silently not the one
# configured.
define require_sparc_layers
	@if [ -z "$(SELECTED_LAYER)" ] || [ -z "$(SE_LAYERS)" ]; then \
		echo "ERROR: SELECTED_LAYER and SE_LAYERS are required and have no defaults." >&2; \
		echo "  SELECTED_LAYER  SPARC reference layer for THIS model" >&2; \
		echo "  SE_LAYERS       inclusive window, two integers, e.g. \"0 23\"" >&2; \
		echo "  Derive the reference layer with scripts/select_reference_layer.py." >&2; \
		echo "  e.g. make $@ SELECTED_LAYER=15 SE_LAYERS=\"0 23\"" >&2; \
		exit 1; \
	fi
endef

phase3-smoke:
	$(require_sparc_layers)
	$(PYTHON) scripts/phase3_generate.py --run-name $(PHASE3_RUN_NAME)_smoke --smoke \
		--selected-layer $(SELECTED_LAYER) --se-layers $(SE_LAYERS)

# Coherence smoke — 2 imgs on `long`, captions printed to stdout. Use this
# to eyeball whether SPARC (with the official COCO hparams + greedy) stays
# coherent on long captions BEFORE launching the full sweep.
phase3-coherence:
	$(require_sparc_layers)
	$(PYTHON) scripts/phase3_generate.py --run-name $(PHASE3_RUN_NAME)_coherence --coherence-smoke \
		--selected-layer $(SELECTED_LAYER) --se-layers $(SE_LAYERS)

phase3:
	$(require_sparc_layers)
	$(PYTHON) scripts/phase3_generate.py --run-name $(PHASE3_RUN_NAME) $(PHASE3_FLAGS) \
		--selected-layer $(SELECTED_LAYER) --se-layers $(SE_LAYERS)

# Phase 4 — SmolVLM-2.2B coherence smoke. Confirms the Llama variant of
# add_custom_attention_layers (and the SmolVLM decoder-path lookup) work
# before committing to a full Phase-1/2/3 run on SmolVLM.
phase4-smolvlm-smoke:
	$(require_sparc_layers)
	$(PYTHON) scripts/phase3_generate.py \
		--run-name phase4_smolvlm_coherence \
		--coherence-smoke \
		--length-config-pattern configs/run_smolvlm22_{length}.yaml \
		--selected-layer $(SELECTED_LAYER) --se-layers $(SE_LAYERS)

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
