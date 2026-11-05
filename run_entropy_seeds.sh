#!/usr/bin/env bash
# Corre las configuraciones de g=256 y g=1024 de test_entropy.py en tres semillas.
# Retoma solo: test_entropy.py saltea las combinaciones (config, semilla) ya guardadas.
set -e
cd "$(dirname "$0")"
KEY=("Q b3 g128" "Q b3 g256" "Q b3 g1024" "Q b4 g256" "Q b4 g1024" "Q b5 g1024" "tubo b3 g256" "tubo b4 g256" "tubo b3 g1024" "tubo b4 g1024" "tubo b5 g1024" "Q kfac 3.5 g256" "Q kfac 4.0 g256" "Q kfac 4.5 g256" "tubo kfac 3.5 g256" "tubo kfac 4.0 g256" "tubo kfac 4.5 g256")
for s in 42 0 1; do
  python -u test_entropy.py --seed "$s" --only "${KEY[@]}" 2>&1 | grep --line-buffered -E "tasa fija|Error|Traceback" | tee -a results/entropy_comparison.log
done
