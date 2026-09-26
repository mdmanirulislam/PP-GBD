#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

for seed in 11 23 47 59 83; do
  python code/experiment.py models "$seed"
  python code/experiment.py dp "$seed"
  python code/experiment.py fed "$seed"
  python code/experiment.py ablations "$seed"
done
python code/experiment.py cost 11
python code/experiment.py agg
python code/figures_topologies.py
python code/figures_results.py
