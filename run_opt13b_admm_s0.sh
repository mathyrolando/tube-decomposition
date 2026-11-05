#!/usr/bin/env bash
# OPT-1.3B con ADMM-Q y escala Hessiana, semilla 0: pares Q g128 vs tubo g256 (r=16, como la semilla 42)
# en 3.0 y 3.5, con PPL en C4 y KL respecto de FP16 (--extra). Retoma solo.
cd "$(dirname "$0")"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
M=facebook/opt-1.3b
${PYTHON:-python} -u test_shift.py --model $M --seed 0 --extra --only fp16
for job in "Q admmH kfac 3.0 g128 γ0.0" "tubo admmH kfac 3.0 g256 γ0.0" \
           "Q admmH kfac 3.5 g128 γ0.0" "tubo admmH kfac 3.5 g256 γ0.0"; do
  ${PYTHON:-python} -u test_shift.py --model $M --seed 0 --extra --only "$job"
done
echo "TANDA 1.3B ADMM SEMILLA 0 TERMINADA"
