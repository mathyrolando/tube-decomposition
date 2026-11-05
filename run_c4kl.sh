#!/usr/bin/env bash
# PPL en C4 y KL respecto de FP16 en los puntos principales (OPT-125M, ADMM-H): tubo r=8 g256 contra Q g128
# a igual costo, presupuestos 3.0, 3.5 y 4.0, semillas 42, 0 y 1. Además re-mide esos puntos con git limpio.
# Retoma solo.
cd "$(dirname "$0")"
PY=${PYTHON:-python}
$PY -u test_shift.py --seed 42 --extra --only fp16
for s in 42 0 1; do
  for b in 3.0 3.5 4.0; do
    $PY -u test_shift.py --seed $s --extra --only "tubo r8 admmH kfac $b g256" "Q admmH kfac $b g128 γ0.0"
  done
done
echo "C4 Y KL TERMINADO"
