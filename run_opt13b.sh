#!/usr/bin/env bash
# OPT-1.3B, semilla 42: curvas y mapas K-FAC, pares GPTQ (Q g128 vs tubo g256) y un par ADMM-H. Retoma solo.
cd "$(dirname "$0")"
M=facebook/opt-1.3b
if [ ! -f results/rd_alloc_opt1.3b.json ]; then
  python -u test_rd_alloc.py --model $M --variants kfac --budgets 2.5 3.0 3.5 4.0 4.5 --plan_only || exit 1
fi
for job in "Q kfac 3.0 g128 γ0.0" "tubo kfac 3.0 g256 γ0.0" "Q kfac 3.5 g128 γ0.0" "tubo kfac 3.5 g256 γ0.0" \
           "Q kfac 4.0 g128 γ0.0" "tubo kfac 4.0 g256 γ0.0" "Q admmH kfac 3.0 g128 γ0.0" "tubo admmH kfac 3.0 g256 γ0.0"; do
  python -u test_shift.py --model $M --seed 42 --only "$job"
done
echo "TANDA 1.3B TERMINADA"
