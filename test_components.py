#!/usr/bin/env python3
"""
Ablación por componente y por matriz: ¿dónde paga L, dónde paga S?

Para cada sublayer lineal de los bloques pedidos y para cada base Q de w bits:
  - Q sola a w y w+1 bits (triad_quant_only, g=64)  ->  línea de referencia
  - Q + L      (sparsity = 0)
  - Q + S      (rank = 0)
  - Q + L + S

Métrica principal: bits equivalentes ganados ("Δbits").
  Interpolando log-linealmente el error de Q sola entre w y w+1 bits,
      b_eq(err) = (w + 0.5) + (ln Q_w - ln err) / (ln Q_w - ln Q_{w+1})
  es la cantidad de bits que necesitaría Q sola para igualar ese error.
      Δbits = b_eq(err_config) - bits_config
  Δbits > 0  -> el componente paga más de lo que cuesta (le gana a Q sola).
  Δbits < 0  -> esos bits rinden más gastados en Q.
  Fuera de [w, w+1] bits la línea se extrapola; se marca con '*'.

Resumen: Δbits promedio ponderado por parámetros para cada configuración, y
el de un "oráculo" que elige por matriz la mejor opción entre
{Q sola, Q+L, Q+S, Q+L+S}. Es una cota de lo que daría asignar por matriz.

Uso:
    uv run test_components.py                        # bloque 0, bases 2 y 3 bits
    uv run test_components.py --layers 0 6 11
    uv run test_components.py --wbits 3 --rank 32 --sparsity 0.005
    # S no estructurada barata: valores int8 y conteo por fila
    uv run test_components.py --layers 0 6 11 --wbits 3 --sparsity 0.0025 --s_val_bits 8 --s_index rowcount
    # S por columnas (OWQ): 4 canales de entrada completos en 8 bits
    uv run test_components.py --layers 0 6 11 --wbits 3 --s_struct column --s_cols 4 --s_val_bits 8
"""
from __future__ import annotations
import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Any, List

import torch

from pipeline import DEV
from data_loader import get_wikitext2, set_seed
from test_l_quant import capture_block_engines
from provenance import GIT

CURRENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = CURRENT_DIR / "results"
OUT_PATH = RESULTS_DIR / "components_diagnostic.json"

GROUPSIZE = 64
Q_OVERHEAD = 32.0 / GROUPSIZE   # misma contabilidad que triad_quant_only
CONFIGS = ["Q+L", "Q+S", "Q+L+S"]


def delta_bits(err: float, bits: float, w: int, qerr: Dict[int, float]):
    """Bits equivalentes ganados frente a la línea log-lineal Q_w -> Q_{w+1}."""
    lw, lw1 = math.log(qerr[w]), math.log(qerr[w + 1])
    slope = lw - lw1  # ln-error ganado por bit de Q
    b_eq = (w + Q_OVERHEAD) + (lw - math.log(max(err, 1e-30))) / slope
    extrap = not (w + Q_OVERHEAD <= b_eq <= w + 1 + Q_OVERHEAD)
    return b_eq - bits, b_eq, extrap


def main():
    ap = argparse.ArgumentParser(description="Ablación Q / Q+L / Q+S / Q+L+S por matriz")
    ap.add_argument("--layers", type=int, nargs="+", default=[0])
    ap.add_argument("--wbits", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--sparsity", type=float, default=0.01)
    ap.add_argument("--l_bits", type=int, default=3)
    ap.add_argument("--scale_dim_l", default="col")
    ap.add_argument("--l_quant", default="hessian")
    ap.add_argument("--cycles", type=int, default=2)
    ap.add_argument("--s_struct", default="unstructured", choices=["unstructured", "column"])
    ap.add_argument("--s_val_bits", type=int, default=16)
    ap.add_argument("--s_index", default="csr", choices=["csr", "rowcount"])
    ap.add_argument("--s_cols", type=int, default=4)
    ap.add_argument("--nsamples", type=int, default=128)
    args = ap.parse_args()

    set_seed(42)
    dataloader, _ = get_wikitext2(nsamples=args.nsamples, seed=42, model="facebook/opt-125m")
    q_levels = sorted(set(args.wbits) | {w + 1 for w in args.wbits})

    rows: List[Dict[str, Any]] = []
    for layer in args.layers:
        t0 = time.time()
        engines = capture_block_engines(layer, dataloader, args.nsamples, DEV)
        print(f"\n[OK] Hessianas del bloque {layer} capturadas en {time.time() - t0:.1f}s")

        for name, eng in engines.items():
            m, n = eng.rows, eng.columns
            qerr = {w: eng.run_triad_quant_only(wbits=w, groupsize=GROUPSIZE)["functional_error"]
                    for w in q_levels}

            for w in args.wbits:
                entry: Dict[str, Any] = {
                    "layer": layer, "name": name, "m": m, "n": n, "wbits": w,
                    "err_Q": qerr[w], "err_Q_next": qerr[w + 1],
                    "bits_Q": w + Q_OVERHEAD, "configs": {},
                }
                for cfg in CONFIGS:
                    rank = args.rank if "L" in cfg else 0
                    use_s = "S" in cfg
                    res = eng.run_qls_triad(
                        wbits=w, rank=rank, groupsize=GROUPSIZE,
                        sparsity=args.sparsity if use_s else 0.0,
                        cycles=args.cycles, l_bits=args.l_bits,
                        scale_dim_l=args.scale_dim_l, l_quant=args.l_quant,
                        s_struct=args.s_struct, s_val_bits=args.s_val_bits,
                        s_index=args.s_index, s_cols=args.s_cols if use_s else 0,
                    )
                    dg = res.get("diagnostics", {})
                    bits = res["effective_bits_csr"]
                    err = res["functional_error"]
                    db, b_eq, extrap = delta_bits(err, bits, w, qerr)
                    entry["configs"][cfg] = {
                        "bits": bits, "extra_bits": bits - (w + Q_OVERHEAD), "err": err,
                        "b_eq": b_eq, "delta_bits": db, "extrapolated": extrap,
                        "bits_S": dg.get("triad_bits_S"),
                        "bits_S_index": dg.get("triad_bits_S_index"),
                        "bits_S_entropy_bound": dg.get("triad_bits_S_entropy_bound"),
                        "k_sparse": dg.get("triad_k_sparse"),
                    }
                # Oráculo por matriz: la mejor opción incluyendo "no agregar nada" (Δbits = 0)
                best = max(CONFIGS, key=lambda c: entry["configs"][c]["delta_bits"])
                if entry["configs"][best]["delta_bits"] <= 0:
                    best = "Q"
                entry["best"] = best
                rows.append(entry)

                print(f"\n== bloque {layer} / {name} ({m}x{n}) | base Q{w}: "
                      f"err Q{w} = {qerr[w]:.4g} ({w + Q_OVERHEAD:.2f}b), "
                      f"err Q{w + 1} = {qerr[w + 1]:.4g} ({w + 1 + Q_OVERHEAD:.2f}b)")
                print(f"   {'config':<8}{'bits':>8}{'+bits':>8}{'err':>12}{'b_eq':>8}{'Δbits':>9}")
                for cfg in CONFIGS:
                    c = entry["configs"][cfg]
                    flag = "*" if c["extrapolated"] else " "
                    print(f"   {cfg:<8}{c['bits']:>8.3f}{c['extra_bits']:>8.3f}{c['err']:>12.4g}"
                          f"{c['b_eq']:>8.3f}{c['delta_bits']:>+9.3f}{flag}")
                cs = entry["configs"]["Q+S"]
                if cs["k_sparse"]:
                    print(f"   S: {cs['k_sparse']} no-ceros | {cs['bits_S']:.4f} b/p "
                          f"({cs['bits_S'] * m * n / cs['k_sparse']:.1f} bits por no-cero; "
                          f"índice {cs['bits_S_index'] * m * n / cs['k_sparse']:.1f}, "
                          f"cota de entropía {cs['bits_S_entropy_bound'] * m * n / cs['k_sparse']:.1f})")
                print(f"   -> mejor opción: {best}")

    # ---- Resumen ponderado por parámetros ----
    print("\n== Resumen: Δbits promedio ponderado por parámetros (> 0 le gana a Q sola) ==")
    summary: Dict[str, Any] = {}
    for w in args.wbits:
        sub = [r for r in rows if r["wbits"] == w]
        P = sum(r["m"] * r["n"] for r in sub)
        s = {cfg: sum(r["configs"][cfg]["delta_bits"] * r["m"] * r["n"] for r in sub) / P
             for cfg in CONFIGS}
        s["oraculo"] = sum(
            (0.0 if r["best"] == "Q" else r["configs"][r["best"]]["delta_bits"]) * r["m"] * r["n"]
            for r in sub) / P
        counts = {k: sum(1 for r in sub if r["best"] == k) for k in ["Q"] + CONFIGS}
        s["eleccion_oraculo"] = counts
        summary[f"Q{w}"] = s
        print(f"   base Q{w}: " + " | ".join(f"{c} {s[c]:+.3f}" for c in CONFIGS)
              + f" | oráculo {s['oraculo']:+.3f}   elecciones {counts}")

    # Por tipo de matriz (agregando bloques), para ver patrones como el de k_proj
    print("\n== Δbits por tipo de matriz (promedio sobre bloques) ==")
    by_type: Dict[str, Any] = {}
    for w in args.wbits:
        for name in sorted({r["name"] for r in rows}):
            sub = [r for r in rows if r["wbits"] == w and r["name"] == name]
            by_type[f"Q{w}/{name}"] = {
                cfg: sum(r["configs"][cfg]["delta_bits"] for r in sub) / len(sub) for cfg in CONFIGS
            }
            d = by_type[f"Q{w}/{name}"]
            print(f"   Q{w} {name:<20}" + " | ".join(f"{c} {d[c]:+.3f}" for c in CONFIGS))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({"git": GIT, "args": vars(args), "per_matrix": rows, "summary": summary,
                   "by_type": by_type}, f, indent=2)
    print(f"\n[OK] Resultados guardados en {OUT_PATH}")


if __name__ == "__main__":
    main()