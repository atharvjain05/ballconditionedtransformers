#!/usr/bin/env bash
# Train all four models on the synthetic dataset and dump metrics to results/.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=".venv/bin/python"

for cfg in configs/synthetic_kalman.yaml \
           configs/synthetic_lstm.yaml \
           configs/synthetic_symmetric.yaml \
           configs/synthetic_ballcond.yaml; do
  echo
  echo "=== $cfg ==="
  $PY -m ballcond.train --config "$cfg" --out results
done

echo
echo "Sweep complete. See results/ for per-run metrics.json."
