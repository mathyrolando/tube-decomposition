#!/usr/bin/env bash
# Revalida con ppl_val r, g y alpha_min del tubo (ADMM-H, OPT-125M, semilla 42). Retoma solo.
cd "$(dirname "$0")"
for b in 3.0 3.5; do
  for v in "g256 r8 a0.7" "g256 r32 a0.7" "g128 r16 a0.7" "g64 r16 a0.7" "g256 r16 a0.4" "g256 r16 a0.7"; do
    python -u test_shift.py --seed 42 --only "tubo admmH kfac $b $v val"
  done
done
echo "REVALIDACIÓN TERMINADA"
