#!/usr/bin/env bash
# Run the SPARC matrix on the COMPOSED-QUESTION track, sequentially and
# unattended: per arm, composed_generate.py writes answers.jsonl. Scoring is a
# separate offline stage against a judge model, so nothing here grades anything.
# Sibling of scripts/run_sparc_matrix.sh; the difference is that this track has
# no length regime, so it takes ONE config instead of a {length} pattern.
# Run: DERIVED_LAYER=<layer> CONFIG=<cfg.yaml> QUESTIONS=<q.jsonl> \
#      bash scripts/run_composed_matrix.sh [--smoke]

set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage: DERIVED_LAYER=<layer> CONFIG=<cfg.yaml> QUESTIONS=<questions.jsonl> \
       bash scripts/run_composed_matrix.sh [--smoke]

Runs the selected arms in sequence, each into its own results/runs/<arm>_q dir.
A failing arm stops the script; a re-run resumes (composed_generate.py skips
done cells, the arm guard protects dirs).

Required env: DERIVED_LAYER    arm5's reference layer, from select_reference_layer.py
              CONFIG           family config (model block, seed, max_new_tokens)
              QUESTIONS        JSON Lines question annotations
Optional env: ARMS             space-separated subset of
                               "baseline arm1_sparc arm2_adaptive arm3_qcond
                                arm4_conserve arm5_reflayer"
                               (default: the five arm* ones)
              NUM_ITEMS (default 100), SEED (default 0),
              OUTPUT_ROOT (default results/runs), PYTHON.
  --smoke   run the whole sequence with 2 items for an end-to-end check.
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

# ---- required env, no defaults ----------------------------------------------
for required in DERIVED_LAYER CONFIG QUESTIONS; do
    if [[ -z "${!required:-}" ]]; then
        echo "ERROR: $required is not set; the composed matrix needs it." >&2
        usage
        exit 1
    fi
done

# ---- overridable run config -------------------------------------------------
NUM_ITEMS="${NUM_ITEMS:-100}"
SEED="${SEED:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/runs}"

# ---- fixed SPARC hparams, common to every arm (the SmolVLM recipe) ----------
ALPHA=1.05
BETA=0.1
TAU=3.0
SELECTED_LAYER=15
SE_LAYERS_LO=0
SE_LAYERS_HI=24
REPETITION_PENALTY=1.2

if [[ "$SMOKE" -eq 1 ]]; then
    NUM_ITEMS=2
    SUFFIX="_smoke"
else
    SUFFIX=""
fi

# ---- improvement-flag groups (explicit; no reliance on CLI defaults) --------
ADAPTIVE_FLAGS=(--adaptive --lam 0.5 --ceiling 2.0)
QCOND_FLAGS=(--qcond --qtop-frac 0.05)
CONSERVE_FLAGS=(--conserve --rho 0.5 --sink-frac 0.05)

# ---- arm selection ----------------------------------------------------------
KNOWN_ARMS=(baseline arm1_sparc arm2_adaptive arm3_qcond arm4_conserve arm5_reflayer)
DEFAULT_ARMS="arm1_sparc arm2_adaptive arm3_qcond arm4_conserve arm5_reflayer"
read -r -a SELECTED_ARMS <<< "${ARMS:-$DEFAULT_ARMS}"

for arm in "${SELECTED_ARMS[@]}"; do
    found=0
    for known in "${KNOWN_ARMS[@]}"; do
        [[ "$arm" == "$known" ]] && found=1 && break
    done
    if [[ "$found" -eq 0 ]]; then
        echo "ERROR: unknown arm '$arm'." >&2
        echo "  known arms: ${KNOWN_ARMS[*]}" >&2
        exit 1
    fi
done

# The composed run dirs are suffixed so they never collide with the description
# track's dirs under the same output root.
RUN_NAMES=()
for arm in "${SELECTED_ARMS[@]}"; do
    RUN_NAMES+=("${arm}_q${SUFFIX}")
done

echo "=================================================================="
echo "SPARC composed-question matrix"
echo "  config           : $CONFIG"
echo "  questions        : $QUESTIONS"
echo "  items/arm        : $NUM_ITEMS"
echo "  common hparams   : alpha=$ALPHA beta=$BETA tau=$TAU selected_layer=$SELECTED_LAYER se_layers=($SE_LAYERS_LO,$SE_LAYERS_HI) rep_penalty=$REPETITION_PENALTY (greedy)"
echo "  arms             : ${SELECTED_ARMS[*]}"
echo "  arm5 ref layer   : $DERIVED_LAYER"
echo "  seed (provenance): $SEED"
echo "  output root      : $OUTPUT_ROOT"
echo "  smoke            : $SMOKE"
echo "=================================================================="

run_arm() {
    local run_name="$1"; shift
    local sel_layer="$1"; shift
    local extra_flags=("$@")
    local run_dir="$OUTPUT_ROOT/$run_name"
    mkdir -p "$run_dir"

    echo ""
    echo "------------------------------------------------------------------"
    echo "[composed] ARM $run_name (selected_layer=$sel_layer, flags: ${extra_flags[*]:-none})"
    echo "------------------------------------------------------------------"

    "$PYTHON" scripts/composed_generate.py \
        --run-name "$run_name" \
        --output-root "$OUTPUT_ROOT" \
        --config "$CONFIG" \
        --questions "$QUESTIONS" \
        --limit "$NUM_ITEMS" \
        --alpha "$ALPHA" \
        --beta "$BETA" \
        --tau "$TAU" \
        --selected-layer "$sel_layer" \
        --se-layers "$SE_LAYERS_LO" "$SE_LAYERS_HI" \
        --repetition-penalty "$REPETITION_PENALTY" \
        "${extra_flags[@]}" \
        2>&1 | tee -a "$run_dir/console.log"
}

dispatch_arm() {
    local arm="$1"
    local run_name="${arm}_q${SUFFIX}"
    case "$arm" in
        baseline)
            run_arm "$run_name" "$SELECTED_LAYER" --baseline-only ;;
        arm1_sparc)
            run_arm "$run_name" "$SELECTED_LAYER" ;;
        arm2_adaptive)
            run_arm "$run_name" "$SELECTED_LAYER" "${ADAPTIVE_FLAGS[@]}" ;;
        arm3_qcond)
            run_arm "$run_name" "$SELECTED_LAYER" "${ADAPTIVE_FLAGS[@]}" "${QCOND_FLAGS[@]}" ;;
        arm4_conserve)
            run_arm "$run_name" "$SELECTED_LAYER" "${ADAPTIVE_FLAGS[@]}" "${QCOND_FLAGS[@]}" "${CONSERVE_FLAGS[@]}" ;;
        arm5_reflayer)
            run_arm "$run_name" "$DERIVED_LAYER" "${ADAPTIVE_FLAGS[@]}" "${QCOND_FLAGS[@]}" "${CONSERVE_FLAGS[@]}" ;;
    esac
}

for arm in "${SELECTED_ARMS[@]}"; do
    dispatch_arm "$arm"
done

# ---- summary (reached only if every arm succeeded; set -e stops on failure) --
echo ""
echo "=================================================================="
echo "COMPOSED MATRIX SUMMARY"
echo "=================================================================="
for run_name in "${RUN_NAMES[@]}"; do
    run_dir="$OUTPUT_ROOT/$run_name"
    answers="$run_dir/answers.jsonl"
    if [[ -f "$answers" ]]; then
        cells="$(wc -l < "$answers" | tr -d '[:space:]')"
        status="done"
    else
        cells=0
        status="NO_ANSWERS"
    fi
    printf '  %-22s %-10s cells=%-6s %s\n' "$run_name" "$status" "$cells" "$answers"
done
echo "=================================================================="
