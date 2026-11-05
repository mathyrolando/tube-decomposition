#!/usr/bin/env python3
"""
Asignación de tasa-distorsión de b por matriz, solo con Q (g fijo).

Resuelve, sobre b_l enteros,
    min sum_l alpha_l err_l(b_l)   s.a.   sum_l p_l (b_l + 32/g) <= B sum_l p_l
con err_l(b) medido en cada matriz y alpha_l la sensibilidad de salida K-FAC de la
salida calculada por sensitivities/extract_output_sensitivities.py. Esa alpha suma sobre las m salidas y err_l es por fila, así que su producto
es la contribución a la pérdida bajo G_l isotrópica.

El lagrangiano da puntos de la envolvente convexa. Lo que queda de presupuesto se
reparte con el mejor cociente de mejora por bit.

Variantes:
  control  b uniforme: tiene que reproducir triad_quant_only
  kfac     pondera por alpha_l
  plano    alpha_l = 1, para aislar cuánto aporta la sensibilidad

Uso:
    python test_rd_alloc.py
    python test_rd_alloc.py --budgets 3.5 --variants kfac
"""
from __future__ import annotations
import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Any, List

import torch

from pipeline import DEV, evaluate_ppl, get_opt, run_compression_mode
from data_loader import get_wikitext2, set_seed
from test_l_quant import capture_block_engines
from provenance import GIT

CURRENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = CURRENT_DIR / "results"
ALPHA_DIR = CURRENT_DIR / "sensitivities"
MODEL = "facebook/opt-125m"
CURVES_PATH = RESULTS_DIR / "rd_curves.json"
OUT_PATH = RESULTS_DIR / "rd_alloc.json"
ALPHA_PATH = ALPHA_DIR / "opt125m_output_sensitivities.pt"
N_BLOCKS = 12


def configure(model: str):
    """Rutas y bloques por modelo; OPT-125M conserva los nombres originales."""
    global MODEL, CURVES_PATH, OUT_PATH, ALPHA_PATH, N_BLOCKS
    from transformers import AutoConfig
    MODEL = model
    N_BLOCKS = AutoConfig.from_pretrained(model).num_hidden_layers
    tag = model.split("/")[-1].replace("-", "")
    if tag != "opt125m":
        CURVES_PATH = RESULTS_DIR / f"rd_curves_{tag}.json"
        OUT_PATH = RESULTS_DIR / f"rd_alloc_{tag}.json"
        ALPHA_PATH = ALPHA_DIR / f"{tag}_output_sensitivities.pt"


def measure_curves(dataloader, nsamples: int, levels: List[int], g: int) -> Dict[str, Any]:
    if CURVES_PATH.exists():
        data = json.loads(CURVES_PATH.read_text())
        if data["levels"] == levels and data["groupsize"] == g:
            print(f"[OK] Curvas reutilizadas de {CURVES_PATH}", flush=True)
            return data

    alpha = torch.load(ALPHA_PATH, map_location="cpu", weights_only=False)
    mats: Dict[str, Any] = {}
    for layer in range(N_BLOCKS):
        t0 = time.time()
        engines = capture_block_engines(layer, dataloader, nsamples, DEV, MODEL)
        for name, eng in engines.items():
            with torch.no_grad():
                errs = {str(b): eng.run_triad_quant_only(wbits=b, groupsize=g)["functional_error"]
                        for b in levels}
            mats[f"{layer}.{name}"] = {
                "p": eng.rows * eng.columns,
                "alpha": alpha[f"model.decoder.layers.{layer}.{name}"],
                "err": errs,
            }
        del engines
        torch.cuda.empty_cache()
        print(f"[OK] curvas del bloque {layer} en {time.time() - t0:.1f}s", flush=True)

    data = {"git": GIT, "levels": levels, "groupsize": g, "matrices": mats}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CURVES_PATH.write_text(json.dumps(data, indent=2))
    return data


def solve(curves: Dict[str, Any], budget: float, weighted: bool) -> Dict[str, int]:
    levels, g, mats = curves["levels"], curves["groupsize"], curves["matrices"]
    over = 32.0 / g
    P = sum(v["p"] for v in mats.values())
    w = {k: (v["alpha"] if weighted else 1.0) for k, v in mats.items()}

    def cost(k: str, b: int) -> float:
        return w[k] * mats[k]["err"][str(b)]

    def mean_bits(bm: Dict[str, int]) -> float:
        return sum(mats[k]["p"] * (bm[k] + over) for k in mats) / P

    def at(lam: float) -> Dict[str, int]:
        return {k: min(levels, key=lambda b: cost(k, b) + lam * mats[k]["p"] * b) for k in mats}

    # Bisección geométrica: lam alto gasta poco
    lo, hi = 1e-16, 1e6
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if mean_bits(at(mid)) > budget:
            lo = mid
        else:
            hi = mid
    bm = at(hi)

    # Repartir el presupuesto restante
    while True:
        best, best_gain = None, 0.0
        spent = mean_bits(bm)
        for k in mats:
            higher = [b for b in levels if b > bm[k]]
            if not higher:
                continue
            nb = min(higher)
            step = mats[k]["p"] * (nb - bm[k]) / P
            if spent + step > budget + 1e-9:
                continue
            gain = (cost(k, bm[k]) - cost(k, nb)) / step
            if gain > best_gain:
                best, best_gain = (k, nb), gain
        if best is None:
            return bm
        bm[best[0]] = best[1]


def summarize(curves: Dict[str, Any], bm: Dict[str, int]) -> Dict[str, Any]:
    mats, over = curves["matrices"], 32.0 / curves["groupsize"]
    P = sum(v["p"] for v in mats.values())
    by_type: Dict[str, List[int]] = {}
    for k, b in bm.items():
        by_type.setdefault(k.split(".")[-1], []).append(b)
    return {
        "bits": sum(mats[k]["p"] * (bm[k] + over) for k in mats) / P,
        "pred_cost": sum(mats[k]["alpha"] * mats[k]["err"][str(bm[k])] for k in mats),
        "mean_b_by_type": {t: sum(v) / len(v) for t, v in sorted(by_type.items())},
    }


def main():
    ap = argparse.ArgumentParser(description="Asignación R-D de b por matriz con Q sola")
    ap.add_argument("--budgets", type=float, nargs="+", default=[2.5, 3.0, 3.25, 3.5, 4.0, 4.5])
    ap.add_argument("--variants", nargs="+", default=["control", "kfac", "plano"])
    ap.add_argument("--levels", type=int, nargs="+", default=[2, 3, 4, 5, 6, 8])
    ap.add_argument("--groupsize", type=int, default=64)
    ap.add_argument("--nsamples", type=int, default=128)
    ap.add_argument("--plan_only", action="store_true", help="resolver e imprimir, sin evaluar PPL")
    ap.add_argument("--model", default="facebook/opt-125m")
    args = ap.parse_args()
    configure(args.model)

    set_seed(42)
    dataloader, testenc = get_wikitext2(nsamples=args.nsamples, seed=42, model=MODEL)
    curves = measure_curves(dataloader, args.nsamples, args.levels, args.groupsize)
    keys = list(curves["matrices"])

    jobs = []
    if "control" in args.variants:
        jobs.append(("control", 3.5, {k: 3 for k in keys}))
    for budget in args.budgets:
        if "kfac" in args.variants:
            jobs.append(("kfac", budget, solve(curves, budget, True)))
        if "plano" in args.variants and budget == 3.5:
            jobs.append(("plano", budget, solve(curves, budget, False)))

    results = json.loads(OUT_PATH.read_text()) if OUT_PATH.exists() else []
    for variant, budget, bm in jobs:
        info = summarize(curves, bm)
        print(f"\n[{variant}] presupuesto {budget}: bits {info['bits']:.3f}, costo predicho "
              f"{info['pred_cost']:.4g}, b medio {info['mean_b_by_type']}", flush=True)
        if args.plan_only:
            # Guarda el mapa sin evaluar, para que otros scripts (test_shift) lo usen.
            results.append({"variant": variant, "budget": budget, "ppl": None, **info, "bit_map": bm, "git": GIT})
            OUT_PATH.write_text(json.dumps(results, indent=2))
            continue
        model = get_opt(MODEL)
        out = run_compression_mode(model, dataloader, DEV, mode="q_bitmap", groupsize=args.groupsize,
                                   nsamples=args.nsamples, bit_map=bm)
        ppl = evaluate_ppl(model, testenc, DEV)
        print(f"[{variant}] bits_pipeline {out.get('effective_bits')} PPL {ppl:.3f}", flush=True)
        results.append({"variant": variant, "budget": budget, "ppl": ppl,
                        "bits_pipeline": out.get("effective_bits"), **info, "bit_map": bm, "git": GIT})
        OUT_PATH.write_text(json.dumps(results, indent=2))
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
