#!/usr/bin/env bash
# Q sola a g=128, mismo costo que el tubo a g=256. Retoma solo.
cd "$(dirname "$0")"
python -u test_shift.py --seed 42 --only "Q admmH kfac 2.5 g128 γ0.0" "Q admmH kfac 3.0 g128 γ0.0" \
  "Q admmH kfac 3.5 g128 γ0.0" "Q admmH kfac 4.0 g128 γ0.0" "Q admmH kfac 4.5 g128 γ0.0"
for s in 0 1; do
  python -u test_shift.py --seed "$s" --only "Q admmH kfac 3.0 g128 γ0.0" "Q admmH kfac 3.5 g128 γ0.0"
done
python -u test_shift.py --seed 42 --only "Q kfac 2.5 g128 γ0.0" "Q kfac 3.0 g128 γ0.0" \
  "Q kfac 3.5 g128 γ0.0" "Q kfac 4.0 g128 γ0.0" "Q kfac 4.5 g128 γ0.0"
