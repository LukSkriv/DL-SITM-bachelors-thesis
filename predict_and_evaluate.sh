#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 -m MODEL  -t TRACES  [-c CM_PNG]  [-p PREDS_H5]  [-T DATASET_TYPE]

  -m  cesta k .h5 modelu
  -t  cesta k náměrům .hdf5
  -c  výsledná matice záměn (default: confusion_matrix.png)
  -p  (nepovinné) cesta k .hdf5 pro uložení předpovědí
  -T  typ datasetu: default|ARM  (default: default) -> k použití správného okna ořezu
EOF
  exit 1
}

# defaults
CM_FILE="confusion_matrix.png"
DATASET_TYPE="default"

while getopts "m:t:c:p:T:" opt; do
  case $opt in
    m) MODEL=$OPTARG ;;
    t) TRACES=$OPTARG ;;
    c) CM_FILE=$OPTARG ;;
    p) PRED_OUT=$OPTARG ;;
    T) DATASET_TYPE=$OPTARG ;;
    *) usage ;;
  esac
done

# required checks
: "${MODEL:?missing -m}" "${TRACES:?missing -t}"

cmd=( python3 predict_and_evaluate.py
      --model        "$MODEL"
      --traces       "$TRACES"
      --confusion_matrix_file "$CM_FILE"
      --dataset_type "$DATASET_TYPE" )

[ -n "${PRED_OUT-}" ] && cmd+=( --output_predicts "$PRED_OUT" )

echo "Running: ${cmd[*]}"
"${cmd[@]}"
