#!/usr/bin/env bash
# Corre test_shift.py en tres semillas; retoma solo, salteando lo ya guardado.
set -e
cd "$(dirname "$0")"
for s in 42 0 1; do
  python -u test_shift.py --seed "$s" "$@" 2>&1 | grep --line-buffered -E "^\[seed|Error|Traceback|error" | tee -a results/shift_comparison.log
done
