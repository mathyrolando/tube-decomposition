#!/usr/bin/env bash
# Tríada con tubo vs Q sola, ambos con ADMM-Q y escala Hessiana. Retoma solo.
cd "$(dirname "$0")"
python -u test_shift.py --seed 42 --only "Q admmH kfac 2.5 g256 γ0.0" "tubo admmH kfac 2.5 g256 γ0.0"
for s in 0 1; do
  python -u test_shift.py --seed "$s" --only "Q admmH kfac 3.0 g256 γ0.0" "tubo admmH kfac 3.0 g256 γ0.0" \
    "Q admmH kfac 3.5 g256 γ0.0" "tubo admmH kfac 3.5 g256 γ0.0"
done
