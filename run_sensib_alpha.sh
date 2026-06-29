#!/usr/bin/env bash
set -euo pipefail

REPO="/aluno/aluno_stephany/repos/VR-Modality-Bias"
cd "$REPO"

LIMIT=50
LENGTHS="short medium long"
BETA=0.1
REP_PEN=1.2

# Familias -> (config_pattern, selected_layer, se_hi)  [se_lo sempre 0]
declare -A CFG=(
  [smolvlm-2.2b]="configs/run_smolvlm22_{length}.yaml 15 24"
  [llava-1.5-7b]="configs/run_llava_{length}.yaml 20 32"
  [qwen2.5-vl-7b]="configs/run_qwen7b_{length}.yaml 18 28"
)

# Configs de intensidade NOVAS:  nome  alpha  tau
CONFIGS=(
  "a110_oficial 1.1 1.5"
  "a115_agressiva 1.15 1.5"
)

ROOT="results/sensibilidade_alpha"
mkdir -p "$ROOT"

for cfg in "${CONFIGS[@]}"; do
  read -r CNAME ALPHA TAU <<< "$cfg"
  OUTDIR="$ROOT/$CNAME"
  mkdir -p "$OUTDIR"
  echo "############################################################"
  echo "# CONFIG: $CNAME  (alpha=$ALPHA tau=$TAU)"
  echo "############################################################"

  for FAM in smolvlm-2.2b llava-1.5-7b qwen2.5-vl-7b; do
    read -r PATTERN SEL SEHI <<< "${CFG[$FAM]}"
    RUN_NAME="chair_${FAM}"
    RUN_DIR="$OUTDIR/$RUN_NAME"

    # Resume: se ja tem resultado CHAIR, pula
    if [[ -f "$RUN_DIR/chair_results.json" ]]; then
      echo ">>> [$CNAME/$FAM] ja tem chair_results.json -- pulando."
      continue
    fi

    echo ">>> [$CNAME/$FAM] gerando legendas (alpha=$ALPHA tau=$TAU sel=$SEL se=0-$SEHI)..."
    python scripts/18_phase3_generate.py \
      --run-name "$RUN_NAME" \
      --output-root "$OUTDIR" \
      --limit "$LIMIT" \
      --lengths $LENGTHS \
      --length-config-pattern "$PATTERN" \
      --alpha "$ALPHA" --beta "$BETA" --tau "$TAU" \
      --selected-layer "$SEL" --se-layers 0 "$SEHI" \
      --repetition-penalty "$REP_PEN"

    echo ">>> [$CNAME/$FAM] rodando CHAIR..."
    python scripts/17_chair_report.py \
      --run-dir "$RUN_DIR" \
      --auto-download

    echo ">>> [$CNAME/$FAM] OK -- chair_results em $RUN_DIR"
  done
done

echo ""
echo "############################################################"
echo "# SENSIBILIDADE AO ALPHA -- CONCLUIDO"
echo "# Resultados em: $ROOT/"
echo "############################################################"