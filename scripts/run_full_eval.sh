#!/usr/bin/env bash
# Full-evaluation orchestrator over the two datasets of the 26/08 selection
# (ODI-Bench composed, 10 types; OmniCoT-Real original format, 6 types) and
# over one or more model families. Per family and dataset it delegates to
# run_composed_matrix.sh; the paired OFF inside each arm is the no-mitigation
# condition, identical between arms under greedy.
#   smolvlm-2.2b     : arm1_sparc (L15) + arm5_reflayer (full method on the
#                      derived layer; OmniCoT derives its own from a diagnostic)
#   llava / qwen /
#   internvl3        : arm1_sparc only, on the depth-scaled heuristic layer,
#                      images capped at 1536 px on the longer side (the SmolVLM
#                      processor applies the same cap by itself)
# THINK_SAMPLE=1 appends, per family and dataset, a <think> run of arm1_sparc
# (OFF + ON) into *_q_think dirs. ROPE_SAMPLE=1 appends the SpheRoPE run of
# arm1_sparc into *_q_rope dirs: Qwen gets the spherical width RoPE (vit+llm)
# plus circular padding; the families without 2D RoPE (SmolVLM, LLaVA) get
# the circular padding only.
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
  FAMILIES           space-separated subset of
                     "smolvlm-2.2b llava-1.5-7b qwen2.5-vl-7b internvl3-8b-hf"
                     (default: smolvlm-2.2b)
  ARMS               override the per-family arm list for every family
  SKIP_ODIBENCH=1    skip the ODI-Bench block
  SKIP_OMNICOT=1     skip the OmniCoT block
  SKIP_DIAG=1        reuse an existing derived_layer.txt for OmniCoT instead
                     of running its diagnostic (SmolVLM only)
  ODIBENCH_LAYER     arm5 layer for ODI-Bench (default 17, derived 20/08)
  DIAG_LIMIT         images in the OmniCoT diagnostic (default 100)
  NUM_ITEMS_ODI      images in the ODI matrix (default 50)
  NUM_ITEMS_OMNI     images in the OmniCoT matrix (default 50)
  THINK_SAMPLE=1     also run the <think> condition per family and dataset
  THINK_ITEMS        images in each think run (default 50)
  THINK_MAX_NEW_TOKENS  token budget of the think runs (default 512)
  ROPE_SAMPLE=1      also run the SpheRoPE / circular-padding condition
  ROPE_PAD           circular padding in pixels per side (default 112)
  ROPE_MODE          SpheRoPE mode for Qwen: vit|llm|both (default both)
  MAX_EDGE           longer-side cap for the non-SmolVLM families (default 1536)
  OUTPUT_ROOT        parent of the run roots (default results/runs); runs land
                     in $OUTPUT_ROOT/full_eval_<dataset>/<family>/
  JUDGE_MODEL, JUDGE_REVISION, PYTHON   forwarded to the matrix script
  SKIP_JUDGE, SKIP_GENERATION           forwarded to the matrix script
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
NUM_ITEMS_ODI="${NUM_ITEMS_ODI:-50}"
NUM_ITEMS_OMNI="${NUM_ITEMS_OMNI:-50}"
SKIP_ODIBENCH="${SKIP_ODIBENCH:-0}"
SKIP_OMNICOT="${SKIP_OMNICOT:-0}"
SKIP_DIAG="${SKIP_DIAG:-0}"
THINK_SAMPLE="${THINK_SAMPLE:-0}"
THINK_ITEMS="${THINK_ITEMS:-50}"
THINK_MAX_NEW_TOKENS="${THINK_MAX_NEW_TOKENS:-512}"
ROPE_SAMPLE="${ROPE_SAMPLE:-0}"
ROPE_PAD="${ROPE_PAD:-112}"
ROPE_MODE="${ROPE_MODE:-both}"
MAX_EDGE="${MAX_EDGE:-1536}"
FAMILIES="${FAMILIES:-smolvlm-2.2b}"
ARMS_OVERRIDE="${ARMS:-}"

KNOWN_FAMILIES=(smolvlm-2.2b llava-1.5-7b qwen2.5-vl-7b internvl3-8b-hf)
read -r -a SELECTED_FAMILIES <<< "$FAMILIES"
for family in "${SELECTED_FAMILIES[@]}"; do
    found=0
    for known in "${KNOWN_FAMILIES[@]}"; do
        [[ "$family" == "$known" ]] && found=1 && break
    done
    if [[ "$found" -eq 0 ]]; then
        echo "ERROR: unknown family '$family'. Known: ${KNOWN_FAMILIES[*]}" >&2
        exit 1
    fi
done

if [[ "$SMOKE" -eq 1 ]]; then
    DIAG_LIMIT=2
fi

# ---- per-family table --------------------------------------------------------
# config tag, reference layer of the heuristic arms, se_layers upper bound,
# longer-side cap (empty = hand the image over as is), default arms.
family_tag()      { case "$1" in smolvlm-2.2b) echo smolvlm22 ;; llava-1.5-7b) echo llava ;; qwen2.5-vl-7b) echo qwen7b ;; internvl3-8b-hf) echo internvl3 ;; esac; }
family_layer()    { case "$1" in smolvlm-2.2b) echo 15 ;; llava-1.5-7b) echo 20 ;; qwen2.5-vl-7b) echo 18 ;; internvl3-8b-hf) echo 18 ;; esac; }
family_se_hi()    { case "$1" in smolvlm-2.2b) echo 24 ;; llava-1.5-7b) echo 32 ;; qwen2.5-vl-7b) echo 28 ;; internvl3-8b-hf) echo 28 ;; esac; }
family_max_edge() { case "$1" in smolvlm-2.2b) echo "" ;; *) echo "$MAX_EDGE" ;; esac; }
family_arms() {
    if [[ -n "$ARMS_OVERRIDE" ]]; then echo "$ARMS_OVERRIDE"; return; fi
    case "$1" in smolvlm-2.2b) echo "arm1_sparc arm5_reflayer" ;; *) echo "arm1_sparc" ;; esac
}
family_spherope() { case "$1" in qwen2.5-vl-7b) echo "$ROPE_MODE" ;; *) echo off ;; esac; }
family_config()   { echo "configs/run_$(family_tag "$1")_$2_composed.yaml"; }

require_processed() {
    local processed_dir="$1"; local ingest_hint="$2"
    if [[ ! -f "$processed_dir/questions.jsonl" || ! -f "$processed_dir/manifest.jsonl" ]]; then
        echo "ERROR: $processed_dir is not ingested (questions.jsonl/manifest.jsonl missing)." >&2
        echo "  run first: $ingest_hint" >&2
        exit 1
    fi
}

echo "=================================================================="
echo "FULL EVALUATION — conditions x 2 datasets x families"
echo "  families        : ${SELECTED_FAMILIES[*]}"
echo "  arms override   : ${ARMS_OVERRIDE:-<per family>}"
echo "  odibench        : skip=$SKIP_ODIBENCH items=$NUM_ITEMS_ODI smolvlm_arm5_layer=$ODIBENCH_LAYER"
echo "  omnicot         : skip=$SKIP_OMNICOT items=$NUM_ITEMS_OMNI diag_limit=$DIAG_LIMIT skip_diag=$SKIP_DIAG"
echo "  think sample    : $THINK_SAMPLE (items=$THINK_ITEMS max_new_tokens=$THINK_MAX_NEW_TOKENS)"
echo "  rope sample     : $ROPE_SAMPLE (pad=$ROPE_PAD px, qwen mode=$ROPE_MODE, others: padding only)"
echo "  max_edge        : $MAX_EDGE (non-SmolVLM families)"
echo "  output root     : $OUTPUT_ROOT"
echo "  smoke           : $SMOKE"
echo "=================================================================="

if [[ "$SKIP_ODIBENCH" -ne 1 ]]; then
    require_processed "data/processed/odibench" \
        "python scripts/ingest_odibench.py --raw data/raw/odibench --out data/processed/odibench --limit 200"
fi
if [[ "$SKIP_OMNICOT" -ne 1 ]]; then
    require_processed "data/processed/omnicot" \
        "python scripts/ingest_omnicot.py --raw data/raw/omnicot --out data/processed/omnicot"
fi

run_matrix() {
    local family="$1" dataset="$2" derived_layer="$3" num_items="$4" out_root="$5"
    DERIVED_LAYER="$derived_layer" \
    CONFIG="$(family_config "$family" "$dataset")" \
    QUESTIONS="data/processed/$dataset/questions.jsonl" \
    ARMS="$(family_arms "$family")" \
    SELECTED_LAYER="$(family_layer "$family")" \
    SE_LAYERS_HI="$(family_se_hi "$family")" \
    MAX_EDGE="$(family_max_edge "$family")" \
    NUM_ITEMS="$num_items" \
    OUTPUT_ROOT="$out_root" \
    bash scripts/run_composed_matrix.sh "${SMOKE_FLAG[@]}"
}

run_think_sample() {
    local family="$1" dataset="$2" out_root="$3"
    echo ""
    echo "[full_eval] $family / $dataset THINK sample ($THINK_ITEMS images) -> $out_root"
    DERIVED_LAYER="$(family_layer "$family")" \
    CONFIG="$(family_config "$family" "$dataset")" \
    QUESTIONS="data/processed/$dataset/questions.jsonl" \
    ARMS="arm1_sparc" \
    SELECTED_LAYER="$(family_layer "$family")" \
    SE_LAYERS_HI="$(family_se_hi "$family")" \
    MAX_EDGE="$(family_max_edge "$family")" \
    THINK=1 \
    MAX_NEW_TOKENS="$THINK_MAX_NEW_TOKENS" \
    NUM_ITEMS="$THINK_ITEMS" \
    OUTPUT_ROOT="$out_root" \
    bash scripts/run_composed_matrix.sh "${SMOKE_FLAG[@]}"
}

run_rope_sample() {
    local family="$1" dataset="$2" num_items="$3" out_root="$4"
    echo ""
    echo "[full_eval] $family / $dataset ROPE run (spherope=$(family_spherope "$family") pad=$ROPE_PAD) -> $out_root"
    DERIVED_LAYER="$(family_layer "$family")" \
    CONFIG="$(family_config "$family" "$dataset")" \
    QUESTIONS="data/processed/$dataset/questions.jsonl" \
    ARMS="arm1_sparc" \
    SELECTED_LAYER="$(family_layer "$family")" \
    SE_LAYERS_HI="$(family_se_hi "$family")" \
    MAX_EDGE="$(family_max_edge "$family")" \
    SPHEROPE="$(family_spherope "$family")" \
    CIRCULAR_PAD="$ROPE_PAD" \
    NUM_ITEMS="$num_items" \
    OUTPUT_ROOT="$out_root" \
    bash scripts/run_composed_matrix.sh "${SMOKE_FLAG[@]}"
}

for family in "${SELECTED_FAMILIES[@]}"; do
    arms="$(family_arms "$family")"

    # ---- block 1: ODI-Bench -------------------------------------------------
    if [[ "$SKIP_ODIBENCH" -ne 1 ]]; then
        out_root="$OUTPUT_ROOT/full_eval_odibench/$family"
        echo ""
        echo "[full_eval] $family / ODI-Bench matrix (arms: $arms) -> $out_root"
        run_matrix "$family" odibench "$ODIBENCH_LAYER" "$NUM_ITEMS_ODI" "$out_root"
        if [[ "$THINK_SAMPLE" -eq 1 ]]; then
            run_think_sample "$family" odibench "$out_root"
        fi
        if [[ "$ROPE_SAMPLE" -eq 1 ]]; then
            run_rope_sample "$family" odibench "$NUM_ITEMS_ODI" "$out_root"
        fi
    fi

    # ---- block 2: OmniCoT-Real ----------------------------------------------
    if [[ "$SKIP_OMNICOT" -ne 1 ]]; then
        out_root="$OUTPUT_ROOT/full_eval_omnicot/$family"
        omnicot_layer="$(family_layer "$family")"
        if [[ " $arms " == *" arm5_reflayer "* ]]; then
            LAYER_FILE="$out_root/derived_layer.txt"
            if [[ "$SKIP_DIAG" -ne 1 ]]; then
                echo ""
                echo "[full_eval] $family / OmniCoT diagnostic ($DIAG_LIMIT images) -> derived layer"
                diag_config="configs/run_$(family_tag "$family")_omnicot_medium.yaml"
                "$PYTHON" scripts/build_manifest.py --config "$diag_config" --overwrite
                # generate_refs creates the timestamped run dir (and its LATEST
                # pointer) that the two next stages resolve.
                "$PYTHON" scripts/generate_refs.py --config "$diag_config" --limit "$DIAG_LIMIT"
                "$PYTHON" scripts/collect_hidden_states.py --config "$diag_config" --limit "$DIAG_LIMIT"
                "$PYTHON" scripts/compute_metrics.py --config "$diag_config" --limit "$DIAG_LIMIT"
                # Resolve the CURRENT diagnostic run through the LATEST pointer,
                # never a name wildcard: a wildcard would also swallow the
                # parquet of an earlier smoke run and contaminate the curve.
                DIAG_DIR="$(tr -d '[:space:]' < "results/runs/diag_$(family_tag "$family")_omnicot_medium_LATEST.txt")"
                mkdir -p "$out_root"
                "$PYTHON" scripts/select_reference_layer.py \
                    --metrics-glob "$DIAG_DIR/*.parquet" \
                    --out "$LAYER_FILE"
            fi
            if [[ ! -f "$LAYER_FILE" ]]; then
                echo "ERROR: $LAYER_FILE not found. Run the diagnostic (unset SKIP_DIAG)" >&2
                echo "  or write the layer there by hand if it was derived elsewhere." >&2
                exit 1
            fi
            omnicot_layer="$(tr -d '[:space:]' < "$LAYER_FILE")"
            echo "[full_eval] $family / OmniCoT derived layer: $omnicot_layer"
        fi
        echo ""
        echo "[full_eval] $family / OmniCoT matrix (arms: $arms) -> $out_root"
        run_matrix "$family" omnicot "$omnicot_layer" "$NUM_ITEMS_OMNI" "$out_root"
        if [[ "$THINK_SAMPLE" -eq 1 ]]; then
            run_think_sample "$family" omnicot "$out_root"
        fi
        if [[ "$ROPE_SAMPLE" -eq 1 ]]; then
            run_rope_sample "$family" omnicot "$NUM_ITEMS_OMNI" "$out_root"
        fi
    fi
done

echo ""
echo "=================================================================="
echo "FULL EVALUATION DONE"
echo "  families      : ${SELECTED_FAMILIES[*]}"
echo "  odibench runs : $OUTPUT_ROOT/full_eval_odibench/<family>"
echo "  omnicot runs  : $OUTPUT_ROOT/full_eval_omnicot/<family>"
echo "=================================================================="
