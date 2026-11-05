#!/usr/bin/env bash
# OPT-1.3B con GPTQ: presupuesto 2.5 y semillas 0 y 1 en 3.0, semilla 0 en 3.5. Retoma solo.
cd "$(dirname "$0")"
M=facebook/opt-1.3b
run() { python -u test_shift.py --model $M --seed "$1" --only "$2"; }
run 42 "Q kfac 2.5 g128 γ0.0"; run 42 "tubo kfac 2.5 g256 γ0.0"
for s in 0 1; do run $s "Q kfac 3.0 g128 γ0.0"; run $s "tubo kfac 3.0 g256 γ0.0"; done
run 0 "Q kfac 3.5 g128 γ0.0"; run 0 "tubo kfac 3.5 g256 γ0.0"
echo "TANDA 1.3B-B TERMINADA"
