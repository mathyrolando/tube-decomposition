#!/usr/bin/env bash
# Rival SRR (código oficial: P se descarta, L de rango r tras Q), ADMM-H, OPT-125M.
# Brazos a igual costo que L+S del tubo: r=8 con L en 8 bits y r=4 con L en 16 bits; r=16 en 16 bits es
# su configuración nativa (~0.4 bits de L). Solo semilla 42: la diferencia con el tubo supera el ruido.
cd "$(dirname "$0")"
PY=${PYTHON:-python}
for b in 3.0 3.5 2.5 4.0 4.5; do
  $PY -u test_shift.py --seed 42 --only "srr admmH kfac $b g256 r8 L8" "srr admmH kfac $b g256 r4 L16" "srr admmH kfac $b g256 r16 L16"
done
echo "SRR TERMINADO"
