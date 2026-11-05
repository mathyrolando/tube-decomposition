#!/usr/bin/env python3
"""
Prueba por matriz de la descomposición por tubo contra Q sola y el BCD actual.

Mismos bloques y misma contabilidad que components_diagnostic.json. Δbits > 0 significa
que la configuración le gana a Q sola con grilla estándar (ver test_components.delta_bits).

    Qa     Q sola con la grilla ampliada a alpha_min: aísla el efecto de la grilla
    tL     tubo con L
    tS     tubo con S
    tLS    tubo con L y S
    bL, bS, bLS   BCD actual, leído de components_diagnostic.json

Uso:
    python test_tube.py
    python test_tube.py --layers 0 --rank 8 --out tube_r8.json
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

from pipeline import DEV
from data_loader import get_wikitext2, set_seed
from test_components import delta_bits
from test_l_quant import capture_block_engines
from tube import tube_compress
from provenance import GIT

RESULTS_DIR = Path(__file__).resolve().parent / "results"
BCD_PATH = RESULTS_DIR / "components_diagnostic.json"
GROUPSIZE = 64
CONFIGS = ["Qa", "tL", "tS", "tLS"]


def main():
    ap = argparse.ArgumentParser(description="Tubo por matriz contra Q sola y BCD")
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 6, 11])
    ap.add_argument("--wbits", type=int, default=3)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--sparsity", type=float, default=0.005)
    ap.add_argument("--l_bits", type=int, default=3)
    ap.add_argument("--s_val_bits", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--alpha_min", type=float, default=0.40)
    ap.add_argument("--init", default="hessian", choices=["hessian", "pivot"])
    ap.add_argument("--nsamples", type=int, default=128)
    ap.add_argument("--out", default="tube_diagnostic.json")
    args = ap.parse_args()

    bcd = {}
    if BCD_PATH.exists():
        bcd = {(r["layer"], r["name"], r["wbits"]): r for r in json.loads(BCD_PATH.read_text())["per_matrix"]}

    set_seed(42)
    dataloader, _ = get_wikitext2(nsamples=args.nsamples, seed=42, model="facebook/opt-125m")
    w = args.wbits
    tube_args = dict(wbits=w, groupsize=GROUPSIZE, l_bits=args.l_bits, s_val_bits=args.s_val_bits,
                     rounds=args.rounds, steps=args.steps, lr=args.lr,
                     alpha_min=args.alpha_min, init=args.init)

    rows: List[Dict[str, Any]] = []
    print(f"{'bl':<3}{'matriz':<10}{'Qa':>7}{'bL':>7}{'tL':>7}{'bS':>7}{'tS':>7}{'bLS':>7}{'tLS':>7}"
          f"{'rangoL':>8}{'modelo':>16}", flush=True)
    for layer in args.layers:
        t0 = time.time()
        engines = capture_block_engines(layer, dataloader, args.nsamples, DEV)
        for name, eng in engines.items():
            with torch.no_grad():
                qerr = {b: eng.run_triad_quant_only(wbits=b, groupsize=GROUPSIZE)["functional_error"]
                        for b in (w, w + 1)}
                wide = eng.run_triad_quant_only(wbits=w, groupsize=GROUPSIZE, alpha_min=args.alpha_min)
            runs = {
                "tL": tube_compress(eng, rank=args.rank, sparsity=0.0, **tube_args),
                "tS": tube_compress(eng, rank=0, sparsity=args.sparsity, **tube_args),
                "tLS": tube_compress(eng, rank=args.rank, sparsity=args.sparsity, **tube_args),
            }
            entry: Dict[str, Any] = {"layer": layer, "name": name, "m": eng.rows, "n": eng.columns,
                                     "err_Q": qerr[w], "configs": {}}
            bits_q = w + 32.0 / GROUPSIZE
            entry["configs"]["Qa"] = {"bits": bits_q, "err": wide["functional_error"],
                                      "delta_bits": delta_bits(wide["functional_error"], bits_q, w, qerr)[0]}
            for key, res in runs.items():
                entry["configs"][key] = {
                    "bits": res["bits"], "err": res["functional_error"],
                    "delta_bits": delta_bits(res["functional_error"], res["bits"], w, qerr)[0],
                    "bits_L": res["bits_L"], "bits_S": res["bits_S"], "k_sparse": res["k_sparse"],
                    "range_ratio": res["range_ratio"], "model_history": res["model_history"],
                }
            prev = bcd.get((layer, name, w), {}).get("configs", {})
            entry["bcd"] = {k: prev[k]["delta_bits"] for k in ("Q+L", "Q+S", "Q+L+S") if k in prev}
            rows.append(entry)

            c, b = entry["configs"], entry["bcd"]
            hist = c["tL"]["model_history"]
            short = name.split(".")[-1]
            print(f"{layer:<3}{short:<10}{c['Qa']['delta_bits']:>+7.3f}{b.get('Q+L', float('nan')):>+7.3f}"
                  f"{c['tL']['delta_bits']:>+7.3f}{b.get('Q+S', float('nan')):>+7.3f}{c['tS']['delta_bits']:>+7.3f}"
                  f"{b.get('Q+L+S', float('nan')):>+7.3f}{c['tLS']['delta_bits']:>+7.3f}"
                  f"{c['tL']['range_ratio']:>8.3f}{hist[0]:>8.3g}{hist[-1]:>8.3g}", flush=True)
        del engines
        torch.cuda.empty_cache()
        print(f"[OK] bloque {layer} en {time.time() - t0:.1f}s", flush=True)

    P = sum(r["m"] * r["n"] for r in rows)
    summary = {k: sum(r["configs"][k]["delta_bits"] * r["m"] * r["n"] for r in rows) / P for k in CONFIGS}
    for k in ("Q+L", "Q+S", "Q+L+S"):
        have = [r for r in rows if k in r["bcd"]]
        if have:
            Pk = sum(r["m"] * r["n"] for r in have)
            summary["bcd " + k] = sum(r["bcd"][k] * r["m"] * r["n"] for r in have) / Pk
    print("\nΔbits ponderado por parámetros: " + " | ".join(f"{k} {v:+.3f}" for k, v in summary.items()))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / args.out).write_text(json.dumps({"git": GIT, "args": vars(args), "per_matrix": rows,
                                                    "summary": summary}, indent=2))
    print(f"[OK] Resultados en {RESULTS_DIR / args.out}")


if __name__ == "__main__":
    main()
