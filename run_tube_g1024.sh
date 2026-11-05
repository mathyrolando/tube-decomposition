#!/usr/bin/env bash
# Tubo ADMM-Q con g=1024, rival justo de Q g1024 en el eje de entropía. Retoma solo.
cd "$(dirname "$0")"
python -u test_shift.py --seed 42 --only "tubo admmH kfac 3.0 g1024 ent" "tubo admmH kfac 3.5 g1024 ent" \
  "tubo admmH kfac 4.0 g1024 ent"
