#!/usr/bin/env bash
# Referencias FP16, ruido de corrida del tubo y r=8 contra r=16 en validación. Retoma solo.
cd "$(dirname "$0")"
PY=${PYTHON:-python}
$PY -u test_shift.py --seed 42 --only fp16
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 $PY -u test_shift.py --seed 42 --model facebook/opt-1.3b --only fp16
for b in 3.0 3.5; do
  $PY -u test_shift.py --seed 42 --only "tubo admmH kfac $b g256 rep1" "tubo admmH kfac $b g256 rep2"
done
for s in 0 1; do
  for b in 3.0 3.5; do
    $PY -u test_shift.py --seed $s --only "tubo admmH kfac $b g256 r8 a0.7 val" "tubo admmH kfac $b g256 r16 a0.7 val"
  done
done
echo "DÍA 1 TERMINADO"
