#!/usr/bin/env bash
# Run the incremental SPARC evaluation matrix (5 arms) sequentially and
# unattended: per arm, phase3_generate.py (captions) then chair_report.py.
# Run: DERIVED_LAYER=<layer> bash scripts/run_sparc_matrix.sh          # full, 100 imgs
#      DERIVED_LAYER=<layer> bash scripts/run_sparc_matrix.sh --smoke  # 2 imgs, e2e

set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage: DERIVED_LAYER=<layer> bash scripts/run_sparc_matrix.sh [--smoke]

Runs the five-arm SPARC matrix in sequence (arm1_sparc .. arm5_reflayer),
each into its own results/runs/<arm> dir. A failing arm stops the script; a
re-run resumes (phase3_generate.py skips done cells, the arm guard protects dirs).

Required env: DERIVED_LAYER  arm5's reference layer, from select_reference_layer.py
              VOCABULARY    JSON vocabulary (name, categories, synonyms)
              ANNOTATIONS   JSON Lines object annotations (image_id, objects)
Optional env: ARMS          space-separated subset of
                            "baseline arm1_sparc arm2_adaptive arm3_qcond
                             arm4_conserve arm5_reflayer"
                            (default: the five arm* ones)
              MODEL_ID, LENGTH_CONFIG_PATTERN, NUM_IMAGES (default 100),
              SEED (default 0), OUTPUT_ROOT (default results/runs), PYTHON.
  --smoke   run the whole sequence with 2 images for an end-to-end check.
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

# ---- run from the repo root so scripts/ and configs/ resolve ----------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT" || exit 1

PYTHON="${PYTHON:-python}"

# ---- model ------------------------------------------------------------------
# phase3_generate.py picks the model from the length config, not a CLI id, so the
# model knob here is the config pattern. Default: SmolVLM-2.2B, whose config
# declares model_id HuggingFaceTB/SmolVLM-Instruct (key smolvlm-2.2b).
MODEL_ID="${MODEL_ID:-HuggingFaceTB/SmolVLM-Instruct}"
# Set in two steps, not via ${VAR:-default}: the literal {length} placeholder
# would otherwise be swallowed as the closing brace of the parameter expansion.
LENGTH_CONFIG_PATTERN="${LENGTH_CONFIG_PATTERN:-}"
if [[ -z "$LENGTH_CONFIG_PATTERN" ]]; then
    LENGTH_CONFIG_PATTERN='configs/run_smolvlm22_{length}.yaml'
fi

# ---- overridable run config -------------------------------------------------
NUM_IMAGES="${NUM_IMAGES:-100}"
# phase3_generate.py has no --seed flag: it derives per-image seeds from the
# length config's seed_global, and the matrix decodes greedily (deterministic),
# so SEED is recorded for provenance only, never passed to phase3.
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
LENGTHS=(short medium long)

# ---- arm5 needs the diagnostic-derived reference layer ----------------------
if [[ -z "${DERIVED_LAYER:-}" ]]; then
    echo "ERROR: DERIVED_LAYER is not set; arm5_reflayer needs it." >&2
    echo "Derive it first with scripts/select_reference_layer.py, then" >&2
    echo "  export DERIVED_LAYER=<layer>  and re-run." >&2
    exit 1
fi

# No defaults on purpose: the vocabulary and the ground truth belong to the
# dataset, not to the matrix. Same contract as DERIVED_LAYER above.
if [[ -z "${VOCABULARY:-}" ]]; then
    echo "ERROR: VOCABULARY is not set; chair_report.py needs it." >&2
    echo "  export VOCABULARY=<vocab.json>  and re-run." >&2
    exit 1
fi
if [[ -z "${ANNOTATIONS:-}" ]]; then
    echo "ERROR: ANNOTATIONS is not set; chair_report.py needs it." >&2
    echo "  export ANNOTATIONS=<annotations.jsonl>  and re-run." >&2
    exit 1
fi

if [[ "$SMOKE" -eq 1 ]]; then
    NUM_IMAGES=2
    SUFFIX="_smoke"
else
    SUFFIX=""
fi

# ---- improvement-flag groups (explicit; no reliance on CLI defaults) --------
ADAPTIVE_FLAGS=(--adaptive --lam 0.5 --ceiling 2.0)
QCOND_FLAGS=(--qcond --qtop-frac 0.05)
CONSERVE_FLAGS=(--conserve --rho 0.5 --sink-frac 0.05)

# ---- arm selection ----------------------------------------------------------
# ARMS picks which arms run; the five incremental ones stay the default. The
# extra 'baseline' arm generates the SPARC-OFF condition only, which is what a
# three-arm pilot (baseline / published method / our full version) needs.
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

RUN_NAMES=()
for arm in "${SELECTED_ARMS[@]}"; do
    RUN_NAMES+=("${arm}${SUFFIX}")
done

echo "=================================================================="
echo "SPARC evaluation matrix"
echo "  model_id         : $MODEL_ID"
echo "  config pattern   : $LENGTH_CONFIG_PATTERN"
echo "  images/arm       : $NUM_IMAGES   lengths: ${LENGTHS[*]}"
echo "  common hparams   : alpha=$ALPHA beta=$BETA tau=$TAU selected_layer=$SELECTED_LAYER se_layers=($SE_LAYERS_LO,$SE_LAYERS_HI) rep_penalty=$REPETITION_PENALTY (greedy)"
echo "  arms             : ${SELECTED_ARMS[*]}"
echo "  arm5 ref layer   : $DERIVED_LAYER"
echo "  vocabulary       : $VOCABULARY"
echo "  annotations      : $ANNOTATIONS"
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
    echo "[matrix] ARM $run_name (selected_layer=$sel_layer, flags: ${extra_flags[*]:-none})"
    echo "------------------------------------------------------------------"

    "$PYTHON" scripts/phase3_generate.py \
        --run-name "$run_name" \
        --output-root "$OUTPUT_ROOT" \
        --limit "$NUM_IMAGES" \
        --lengths "${LENGTHS[@]}" \
        --length-config-pattern "$LENGTH_CONFIG_PATTERN" \
        --alpha "$ALPHA" \
        --beta "$BETA" \
        --tau "$TAU" \
        --selected-layer "$sel_layer" \
        --se-layers "$SE_LAYERS_LO" "$SE_LAYERS_HI" \
        --repetition-penalty "$REPETITION_PENALTY" \
        "${extra_flags[@]}" \
        2>&1 | tee -a "$run_dir/console.log"

    echo "[matrix] CHAIR report for $run_name -> $run_dir/chair_report.txt"
    "$PYTHON" scripts/chair_report.py --run-dir "$run_dir" \
        --vocabulary "$VOCABULARY" --annotations "$ANNOTATIONS" \
        2>&1 | tee "$run_dir/chair_report.txt"
}

dispatch_arm() {
    local arm="$1"
    local run_name="${arm}${SUFFIX}"
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
echo "MATRIX SUMMARY"
echo "=================================================================="
for run_name in "${RUN_NAMES[@]}"; do
    run_dir="$OUTPUT_ROOT/$run_name"
    report="$run_dir/chair_report.txt"
    captions="$run_dir/captions.jsonl"
    if [[ -f "$captions" ]]; then
        cells="$(wc -l < "$captions" | tr -d '[:space:]')"
    else
        cells=0
    fi
    if [[ -f "$report" ]]; then
        status="done"
    else
        status="NO_REPORT"
    fi
    printf '  %-22s %-10s cells=%-6s %s\n' "$run_name" "$status" "$cells" "$report"
done
echo "=================================================================="
