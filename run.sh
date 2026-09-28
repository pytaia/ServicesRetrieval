#!/usr/bin/env bash
# Финальный ответ:   bash run.sh
# С валидацией:      bash run.sh val
# DATA_DIR — папка с train.parquet, benchmark_queries.parquet, benchmark_items.parquet
set -euo pipefail
export DATA_DIR=${DATA_DIR:-./} WORK_DIR=${WORK_DIR:-./work}

python prepare.py test
python dssm.py test
python predict.py

if [[ "${1:-}" == "val" ]]; then
  python prepare.py val
  python validate.py corpus
  python dssm.py val
  python validate.py eval
fi
