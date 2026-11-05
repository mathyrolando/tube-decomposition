#!/usr/bin/env bash
# Entropía de índices con ADMM-Q, semilla 42. Retoma solo.
cd "$(dirname "$0")"
python -u test_shift.py --seed 42 --only "Q admmH kfac 3.0 g128 ent" "tubo admmH kfac 3.0 g256 ent" \
  "Q admmH kfac 3.0 g256 ent" "Q admmH kfac 3.5 g128 ent" "tubo admmH kfac 3.5 g256 ent" \
  "Q admmH kfac 3.5 g256 ent" "Q admmH kfac 3.5 g1024 ent" "Q admmH kfac 4.0 g1024 ent"
