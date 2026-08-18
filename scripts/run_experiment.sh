#!/usr/bin/env bash
# End-to-end run for ONE dataset: preflight, diagnostic, description track,
# composed-question track. One dataset per invocation on purpose -- each has its
# own vocabulary, ground truth and reference layer, and looping over datasets
# here would let one failure contaminate the next.
# Run: see usage() below, or bash scripts/run_experiment.sh --help

set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage: DATASET=<name> CONFIG_PATTERN=<configs/run_x_{length}.yaml> \
       COMPOSED_CONFIG=<configs/run_x_long.yaml> \
       VOCABULARY=<vocab.json> ANNOTATIONS=<annotations.jsonl> \
       QUESTIONS=<questions.jsonl> DIRECTION_TERMS=<direction_terms.json> \
       bash scripts/run_experiment.sh [--smoke]

Stages run in order and each one can be skipped:
  preflight    validate every path and cross the manifest/annotation ids
  diagnostic   generate_refs -> collect_hidden_states -> compute_metrics ->
               summarize -> select_reference_layer, writing the derived layer
               to a file the later stages read
  description  scripts/run_sparc_matrix.sh
  composed     scripts/run_composed_matrix.sh

Required env: DATASET          short name, used for the run and log directories
              CONFIG_PATTERN   description-track config pattern with {length}
              COMPOSED_CONFIG  single config for the composed track
              VOCABULARY       object vocabulary JSON
              ANNOTATIONS      object annotations JSONL
              QUESTIONS        question annotations JSONL
              DIRECTION_TERMS  direction table JSON
Optional env: STAGES        default "preflight diagnostic description composed"
              ARMS          arm subset, passed to both matrices
              NUM_ITEMS     default 100
              DIAG_CONFIG   config for the diagnostic (default: COMPOSED_CONFIG)
              DERIVED_LAYER skip deriving and force this layer
              OUTPUT_ROOT   default results/runs
              PYTHON
  --smoke   2 items everywhere, and the matrices run in their own smoke mode.
EOF
}

# ---- CLI --------------------------------------------------------------------
SMOKE=0
for arg in "$@"; do
    case "$arg" in
        --smoke) SMOKE=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $arg" >&2; usage; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT" || exit 1

PYTHON="${PYTHON:-python}"

for required in DATASET CONFIG_PATTERN COMPOSED_CONFIG VOCABULARY ANNOTATIONS \
                QUESTIONS DIRECTION_TERMS; do
    if [[ -z "${!required:-}" ]]; then
        echo "ERROR: $required is not set." >&2
        usage
        exit 1
    fi
done

STAGES="${STAGES:-preflight diagnostic description composed}"
read -r -a SELECTED_STAGES <<< "$STAGES"
KNOWN_STAGES=(preflight diagnostic description composed)
for stage in "${SELECTED_STAGES[@]}"; do
    found=0
    for known in "${KNOWN_STAGES[@]}"; do
        [[ "$stage" == "$known" ]] && found=1 && break
    done
    if [[ "$found" -eq 0 ]]; then
        echo "ERROR: unknown stage '$stage'. Known: ${KNOWN_STAGES[*]}" >&2
        exit 1
    fi
done

has_stage() {
    local wanted="$1"
    for stage in "${SELECTED_STAGES[@]}"; do
        [[ "$stage" == "$wanted" ]] && return 0
    done
    return 1
}

NUM_ITEMS="${NUM_ITEMS:-100}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/runs}"
DIAG_CONFIG="${DIAG_CONFIG:-$COMPOSED_CONFIG}"
SMOKE_FLAG=()
if [[ "$SMOKE" -eq 1 ]]; then
    NUM_ITEMS=2
    SMOKE_FLAG=(--smoke)
fi

EXPERIMENT_DIR="$OUTPUT_ROOT/experiment_$DATASET"
mkdir -p "$EXPERIMENT_DIR"
STARTED="$(date +%Y-%m-%d_%H%M%S)"
LOG_FILE="$EXPERIMENT_DIR/run_${STARTED}.log"
DERIVED_LAYER_FILE="$EXPERIMENT_DIR/derived_layer.txt"

# Everything from here on is teed to the log so the run can be followed from
# outside the tmux with `tail -f`.
exec > >(while IFS= read -r line; do
             printf '%s %s\n' "$(date +%H:%M:%S)" "$line"
         done | tee -a "$LOG_FILE") 2>&1

echo "=================================================================="
echo "EXPERIMENT  dataset=$DATASET"
echo "=================================================================="
echo "  stages           : ${SELECTED_STAGES[*]}"
echo "  arms             : ${ARMS:-<matrix default>}"
echo "  items            : $NUM_ITEMS   smoke=$SMOKE"
echo "  config pattern   : $CONFIG_PATTERN"
echo "  composed config  : $COMPOSED_CONFIG"
echo "  diagnostic config: $DIAG_CONFIG"
echo "  vocabulary       : $VOCABULARY"
echo "  annotations      : $ANNOTATIONS"
echo "  questions        : $QUESTIONS"
echo "  direction terms  : $DIRECTION_TERMS"
echo "  output root      : $OUTPUT_ROOT"
echo "  log file         : $LOG_FILE"
echo "=================================================================="

STAGE_RESULTS=()
record() { STAGE_RESULTS+=("$1"); }

# ---- preflight --------------------------------------------------------------
# Runs before anything else and aborts the whole run: a wrong path costs seconds
# here and three hours anywhere later.
if has_stage preflight; then
    echo ""
    echo "---- STAGE preflight ----------------------------------------------"
    PREFLIGHT_CONFIGS=()
    for length in short medium long; do
        PREFLIGHT_CONFIGS+=("${CONFIG_PATTERN/\{length\}/$length}")
    done
    PREFLIGHT_CONFIGS+=("$COMPOSED_CONFIG")

    PREFLIGHT_ARMS=()
    if [[ -n "${ARMS:-}" ]]; then
        read -r -a PREFLIGHT_ARMS <<< "$ARMS"
    fi

    preflight_cmd=(
        "$PYTHON" scripts/preflight.py
        --config "${PREFLIGHT_CONFIGS[@]}"
        --limit "$NUM_ITEMS"
        --annotations "$ANNOTATIONS"
        --vocabulary "$VOCABULARY"
        --questions "$QUESTIONS"
        --direction-terms "$DIRECTION_TERMS"
        --output-root "$OUTPUT_ROOT"
    )
    if [[ "${#PREFLIGHT_ARMS[@]}" -gt 0 ]]; then
        preflight_cmd+=(--arms "${PREFLIGHT_ARMS[@]}")
    fi

    if ! "${preflight_cmd[@]}"; then
        echo ""
        echo "ABORTED: preflight found problems. Nothing was generated."
        exit 1
    fi
    record "preflight OK"
fi

# ---- diagnostic -------------------------------------------------------------
if has_stage diagnostic; then
    echo ""
    echo "---- STAGE diagnostic ---------------------------------------------"
    for script in generate_refs collect_hidden_states compute_metrics summarize; do
        echo "[diagnostic] scripts/$script.py"
        "$PYTHON" "scripts/$script.py" --config "$DIAG_CONFIG" --limit "$NUM_ITEMS"
    done

    diag_run_name="$("$PYTHON" -c \
        "import sys,yaml;print(yaml.safe_load(open(sys.argv[1]))['run']['name'])" \
        "$DIAG_CONFIG")"
    echo "[diagnostic] deriving the reference layer from ${diag_run_name}_* runs"
    "$PYTHON" scripts/select_reference_layer.py \
        --metrics-glob "$OUTPUT_ROOT/${diag_run_name}_*/metrics.parquet" \
        --out "$DERIVED_LAYER_FILE"
    record "diagnostic OK (layer -> $DERIVED_LAYER_FILE)"
fi

# ---- the reference layer the later stages use -------------------------------
if [[ -z "${DERIVED_LAYER:-}" ]]; then
    if [[ -f "$DERIVED_LAYER_FILE" ]]; then
        DERIVED_LAYER="$(tr -d '[:space:]' < "$DERIVED_LAYER_FILE")"
        echo ""
        echo "reference layer $DERIVED_LAYER read from $DERIVED_LAYER_FILE"
    elif has_stage description || has_stage composed; then
        echo ""
        echo "ERROR: no reference layer. Run the diagnostic stage first, or set" >&2
        echo "  DERIVED_LAYER=<layer> explicitly." >&2
        exit 1
    fi
fi
export DERIVED_LAYER

# ---- description track ------------------------------------------------------
if has_stage description; then
    echo ""
    echo "---- STAGE description --------------------------------------------"
    if LENGTH_CONFIG_PATTERN="$CONFIG_PATTERN" \
       NUM_IMAGES="$NUM_ITEMS" \
       OUTPUT_ROOT="$OUTPUT_ROOT" \
       VOCABULARY="$VOCABULARY" ANNOTATIONS="$ANNOTATIONS" \
       PYTHON="$PYTHON" \
       bash scripts/run_sparc_matrix.sh "${SMOKE_FLAG[@]}"; then
        record "description OK"
    else
        record "description FAILED"
        echo "STAGE description FAILED -- later stages still run so a partial"
        echo "run is not lost; check the summary below."
    fi
fi

# ---- composed track ---------------------------------------------------------
if has_stage composed; then
    echo ""
    echo "---- STAGE composed -----------------------------------------------"
    if CONFIG="$COMPOSED_CONFIG" \
       QUESTIONS="$QUESTIONS" DIRECTION_TERMS="$DIRECTION_TERMS" \
       NUM_ITEMS="$NUM_ITEMS" \
       OUTPUT_ROOT="$OUTPUT_ROOT" \
       PYTHON="$PYTHON" \
       bash scripts/run_composed_matrix.sh "${SMOKE_FLAG[@]}"; then
        record "composed OK"
    else
        record "composed FAILED"
        echo "STAGE composed FAILED -- see the summary below."
    fi
fi

# ---- summary ----------------------------------------------------------------
echo ""
echo "=================================================================="
echo "EXPERIMENT SUMMARY  dataset=$DATASET"
echo "=================================================================="
for result in "${STAGE_RESULTS[@]}"; do
    echo "  $result"
done
echo ""
echo "  results under : $OUTPUT_ROOT"
echo "  experiment dir: $EXPERIMENT_DIR"
echo "  log file      : $LOG_FILE"
if [[ -f "$DERIVED_LAYER_FILE" ]]; then
    echo "  reference layer: $(tr -d '[:space:]' < "$DERIVED_LAYER_FILE")"
fi
echo "=================================================================="

for result in "${STAGE_RESULTS[@]}"; do
    [[ "$result" == *FAILED* ]] && exit 1
done
exit 0
