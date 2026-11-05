#!/usr/bin/env python3
"""
Traduce la PPL de punta a punta a Δbits, la misma unidad que test_components.py.

Referencia: la línea log-lineal de PPL de Q sola (triad_quant_only, g=64) entre
w y w+1 bits. Para un modelo con PPL p y bits efectivos b:
    b_eq  = (w + 0.5) + (ln PPL_Qw - ln p) / (ln PPL_Qw - ln PPL_Q{w+1})
    Δbits = b_eq - b
Δbits > 0: el modelo le gana a la línea de Q sola en PPL.

Uso:
    uv run compare_ppl.py                                   # familias por defecto, w=3
    uv run compare_ppl.py --families qs_all_s025_4b alloc_qs_s025_4b --wbits 3
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

DATA = Path(__file__).resolve().parent / "results" / "pareto_curves_data.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", default=["qs_all_s025_4b", "alloc_qs_s025_4b"])
    ap.add_argument("--wbits", type=int, default=3)
    args = ap.parse_args()

    data = json.load(open(DATA))
    q = {e["wbits"]: e for e in data["triad_quant_only"]}
    w = args.wbits
    p0, p1 = q[w]["ppl"], q[w + 1]["ppl"]
    b0 = q[w]["bits"]
    slope = math.log(p0) - math.log(p1)
    print(f"Referencia Q sola: Q{w} {b0:.2f}b PPL {p0:.3f} | Q{w + 1} {q[w + 1]['bits']:.2f}b PPL {p1:.3f}\n")
    print(f"{'familia':<22}{'bits':>8}{'PPL':>9}{'b_eq':>8}{'Δbits PPL':>11}{'Δbits pred':>12}  elecciones")

    for fam in args.families:
        entries = data.get(fam, [])
        entries = [entries] if isinstance(entries, dict) else entries
        for e in entries:
            if e.get("wbits") != w:
                continue
            bits = e.get("bits_csr") or e["bits"]
            b_eq = b0 + (math.log(p0) - math.log(e["ppl"])) / slope
            al = e.get("alloc_summary") or {}
            pred = al.get("pred_delta_bits")
            pred_s = f"{pred:>+12.4f}" if pred is not None else f"{'-':>12}"
            print(f"{fam:<22}{bits:>8.4f}{e['ppl']:>9.3f}{b_eq:>8.4f}{b_eq - bits:>+11.4f}{pred_s}  {al.get('choices', '')}")


if __name__ == "__main__":
    main()