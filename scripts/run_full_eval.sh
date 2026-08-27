#!/usr/bin/env bash
# Full-evaluation orchestrator: the three conditions (no mitigation, SPARC
# original L15, full method on the derived layer) over the two datasets of the
# 26/08 selection — ODI-Bench (composed questions, 10 types) and OmniCoT-Real
# (original format, anchors in the prompt, 6 types). Per dataset it delegates
# to run_composed_matrix.sh with ARMS="arm1_sparc arm5_reflayer": the paired
# OFF inside each arm is the third condition, identical between arms under
# greedy. For OmniCoT the reference layer is derived from its own diagnostic
# before the matrix (Ponto 4 discipline), unless SKIP_DIAG=1.
# Ingestion is a precondition, not a stage: this script checks the processed
# dirs exist and points at the ingest command when they do not.
# Run: bash scripts/run_full_eval.sh [--smoke]

set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage: bash scripts/run_full_eval.sh [--smoke]

Preconditions (run once, before this script):
  python scripts/ingest_odibench.py --raw data/raw/odibench \
      --out data/processed/odibench --limit 200
  python scripts/ingest_omnicot.py --raw data/raw/omnicot \
      --out data/processed/omnicot

Optional env:
  SKIP_ODIBENCH=1    skip the ODI-Bench block
  SKIP_OMNICOT=1     skip the OmniCoT block
  SKIP_DIAG=1        reuse an existing derived_layer.txt for OmniCoT instead
                     of running its diagnostic
  ODIBENCH_LAYER     arm5 layer for ODI-Bench (default 17, derived 20/08)
  DIAG_LIMIT         images in the OmniCoT diagnostic (default 100)
  NUM_ITEMS_ODI      images in the ODI matrix (default 200)
  NUM_ITEMS_OMNI     images in the OmniCoT matrix (default 200 = all)
  OUTPUT_ROOT        parent of the two run roots (default results/runs)
  JUDGE_MODEL, JUDGE_REVISION, PYTHON   forwarded to the matrix script
  --smoke            2 items everywhere, end-to-end plumbing check
EOF
}

SMOKE=0
SMOKE_FLAG=()
for arg in "$@"; do
    case "$arg" in
        --smoke) SMOKE=1; SMOKE_FLAG=(--smoke) ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $arg" >&2; usage; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT" || exit 1

PYTHON="${PYTHON:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/runs}"
ODIBENCH_LAYER="${ODIBENCH_LAYER:-17}"
DIAG_LIMIT="${DIAG_LIMIT:-100}"
NUM_ITEMS_ODI="${NUM_ITEMS_ODI:-200}"
NUM_ITEMS_OMNI="${NUM_ITEMS_OMNI:-200}"
SKIP_ODIBENCH="${SKIP_ODIBENCH:-0}"
SKIP_OMNICOT="${SKIP_OMNICOT:-0}"
SKIP_DIAG="${SKIP_DIAG:-0}"
ARMS="arm1_sparc arm5_reflayer"

if [[ "$SMOKE" -eq 1 ]]; then
    DIAG_LIMIT=2
fi

require_processed() {
    local processed_dir="$1"; local ingest_hint="$2"
    if [[ ! -f "$processed_dir/questions.jsonl" || ! -f "$processed_dir/manifest.jsonl" ]]; then
        echo "ERROR: $processed_dir is not ingested (questions.jsonl/manifest.jsonl missing)." >&2
        echo "  run first: $ingest_hint" >&2
        exit 1
    fi
}

echo "=================================================================="
echo "FULL EVALUATION — 3 conditions x 2 datasets"
echo "  arms            : $ARMS (+ paired OFF = no mitigation)"
echo "  odibench        : skip=$SKIP_ODIBENCH items=$NUM_ITEMS_ODI layer=$ODIBENCH_LAYER"
echo "  omnicot         : skip=$SKIP_OMNICOT items=$NUM_ITEMS_OMNI diag_limit=$DIAG_LIMIT skip_diag=$SKIP_DIAG"
echo "  output root     : $OUTPUT_ROOT"
echo "  smoke           : $SMOKE"
echo "=================================================================="

# ---- block 1: ODI-Bench -----------------------------------------------------
if [[ "$SKIP_ODIBENCH" -ne 1 ]]; then
    require_processed "data/processed/odibench" \
        "python scripts/ingest_odibench.py --raw data/raw/odibench --out data/processed/odibench --limit 200"
    echo ""
    echo "[full_eval] ODI-Bench matrix -> $OUTPUT_ROOT/full_eval_odibench"
    DERIVED_LAYER="$ODIBENCH_LAYER" \
    CONFIG="configs/run_smolvlm22_odibench_composed.yaml" \
    QUESTIONS="data/processed/odibench/questions.jsonl" \
    ARMS="$ARMS" \
    NUM_ITEMS="$NUM_ITEMS_ODI" \
    OUTPUT_ROOT="$OUTPUT_ROOT/full_eval_odibench" \
    bash scripts/run_composed_matrix.sh "${SMOKE_FLAG[@]}"
fi

# ---- block 2: OmniCoT-Real --------------------------------------------------
if [[ "$SKIP_OMNICOT" -ne 1 ]]; then
    require_processed "data/processed/omnicot" \
        "python scripts/ingest_omnicot.py --raw data/raw/omnicot --out data/processed/omnicot"

    LAYER_FILE="$OUTPUT_ROOT/full_eval_omnicot/derived_layer.txt"
    if [[ "$SKIP_DIAG" -ne 1 ]]; then
        echo ""
        echo "[full_eval] OmniCoT diagnostic ($DIAG_LIMIT images) -> derived layer"
        "$PYTHON" scripts/build_manifest.py \
            --config configs/run_smolvlm22_omnicot_medium.yaml --overwrite
        # generate_refs creates the timestamped run dir (and its LATEST
        # pointer) that the two next stages resolve; without it,
        # collect_hidden_states has nowhere to run. Same stage order as the
        # ODI diagnostic of 20/08.
        "$PYTHON" scripts/generate_refs.py \
            --config configs/run_smolvlm22_omnicot_medium.yaml --limit "$DIAG_LIMIT"
        "$PYTHON" scripts/collect_hidden_states.py \
            --config configs/run_smolvlm22_omnicot_medium.yaml --limit "$DIAG_LIMIT"
        "$PYTHON" scripts/compute_metrics.py \
            --config configs/run_smolvlm22_omnicot_medium.yaml --limit "$DIAG_LIMIT"
        # Resolve the CURRENT diagnostic run through the LATEST pointer
        # instead of a name wildcard: a wildcard would also swallow the
        # parquet of an earlier smoke run and contaminate the curve.
        DIAG_DIR="$(tr -d '[:space:]' < results/runs/diag_smolvlm22_omnicot_medium_LATEST.txt)"
        "$PYTHON" scripts/select_reference_layer.py \
            --metrics-glob "$DIAG_DIR/*.parquet" \
            --out "$LAYER_FILE"
    fi
    if [[ ! -f "$LAYER_FILE" ]]; then
        echo "ERROR: $LAYER_FILE not found. Run the diagnostic (unset SKIP_DIAG)" >&2
        echo "  or write the layer there by hand if it was derived elsewhere." >&2
        exit 1
    fi
    OMNICOT_LAYER="$(tr -d '[:space:]' < "$LAYER_FILE")"
    echo "[full_eval] OmniCoT derived layer: $OMNICOT_LAYER"

    echo ""
    echo "[full_eval] OmniCoT matrix -> $OUTPUT_ROOT/full_eval_omnicot"
    DERIVED_LAYER="$OMNICOT_LAYER" \
    CONFIG="configs/run_smolvlm22_omnicot_composed.yaml" \
    QUESTIONS="data/processed/omnicot/questions.jsonl" \
    ARMS="$ARMS" \
    NUM_ITEMS="$NUM_ITEMS_OMNI" \
    OUTPUT_ROOT="$OUTPUT_ROOT/full_eval_omnicot" \
    bash scripts/run_composed_matrix.sh "${SMOKE_FLAG[@]}"
fi

echo ""
echo "=================================================================="
echo "FULL EVALUATION DONE"
echo "  odibench runs : $OUTPUT_ROOT/full_eval_odibench"
echo "  omnicot runs  : $OUTPUT_ROOT/full_eval_omnicot"
echo "=================================================================="
