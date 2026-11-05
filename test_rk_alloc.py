#!/usr/bin/env python3
"""
Reparto conjunto de (b, r, rho) por matriz con un precio único por bit, para el tubo.

1. Mide, en cada matriz, el error funcional y los bits del tubo para cada opción.
2. Elige una opción por matriz minimizando sum_l alpha_l err_l + lam * sum_l p_l bits_l,
   con lam ajustado para igualar el presupuesto del tubo de referencia.
3. Evalúa PPL de validación y de test con ese reparto.

Uso:
    python test_rk_alloc.py --seed 42
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import torch

from pipeline import DEV, evaluate_ppl, get_opt, run_compression_mode
from data_loader import get_wikitext2, set_seed
from test_shift import get_wikitext2_val
from provenance import GIT

CURRENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = CURRENT_DIR / "results"
ALPHA_PATH = CURRENT_DIR / "sensitivities" / "opt125m_output_sensitivities.pt"
TUBE = dict(wbits=3, r=16, sparsity=0.0025, groupsize=256, l_bits=3, s_val_bits=4,
            s_index="rowcount", alpha_min=0.70)
OPTIONS = [dict(db=db, r=r, sparsity=sp, base=(db == 0 and r == 16 and sp == 0.0025))
           for db in (-1, 0, 1) for r, sp in ((0, 0.0), (0, 0.0025), (16, 0.0025), (32, 0.005))]


def solve(mats: Dict[str, Any], budget: float) -> Dict[str, Dict[str, Any]]:
    P = sum(v["p"] for v in mats.values())

    def at(lam: float) -> Dict[str, Dict[str, Any]]:
        return {k: min(v["options"], key=lambda o: v["alpha"] * o["err"] + lam * v["p"] * o["bits"])
                for k, v in mats.items()}

    def mean_bits(choice) -> float:
        return sum(mats[k]["p"] * choice[k]["bits"] for k in mats) / P

    lo, hi = 1e-16, 1e6
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if mean_bits(at(mid)) > budget:
            lo = mid
        else:
            hi = mid
    choice = at(hi)
    # Gastar lo que sobra con el mejor cociente de mejora por bit
    while True:
        spent, best, best_gain = mean_bits(choice), None, 0.0
        for k, v in mats.items():
            cur = choice[k]
            for o in v["options"]:
                step = v["p"] * (o["bits"] - cur["bits"]) / P
                if step <= 0 or spent + step > budget + 1e-9:
                    continue
                gain = v["alpha"] * (cur["err"] - o["err"]) / step
                if gain > best_gain:
                    best, best_gain = (k, o), gain
        if best is None:
            return choice
        choice[best[0]] = best[1]


def main():
    ap = argparse.ArgumentParser(description="Reparto de (b, r, rho) por matriz para el tubo")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--budget_map", type=float, default=3.5, help="Mapa K-FAC de b de partida")
    args = ap.parse_args()

    maps = {x["budget"]: x["bit_map"] for x in json.loads((RESULTS_DIR / "rd_alloc.json").read_text())
            if x["variant"] == "kfac"}
    bit_map = maps[args.budget_map]
    alpha = torch.load(ALPHA_PATH, map_location="cpu", weights_only=False)
    set_seed(args.seed)
    dataloader, testenc = get_wikitext2(nsamples=128, seed=args.seed, model="facebook/opt-125m")
    valenc = get_wikitext2_val()

    curves_path = RESULTS_DIR / f"rk_curves_s{args.seed}_m{args.budget_map}.json"
    if curves_path.exists():
        mats = json.loads(curves_path.read_text())
    else:
        set_seed(args.seed)
        model = get_opt()
        out = run_compression_mode(model, dataloader, DEV, mode="tube_options", nsamples=128,
                                   bit_map=bit_map, options=OPTIONS, **TUBE)
        mats = {}
        for s in out["layer_stats"]:
            k = f"{s['layer_idx']}.{s['layer_name']}"
            mats[k] = {"p": s["numel"], "options": s["options"],
                       "alpha": float(alpha[f"model.decoder.layers.{k}"])}
        curves_path.write_text(json.dumps(mats, indent=2))
        del model
        torch.cuda.empty_cache()

    P = sum(v["p"] for v in mats.values())
    base_bits = sum(v["p"] * next(o["bits"] for o in v["options"]
                                  if o["wbits"] == bit_map[k] and o["r"] == 16 and o["sparsity"] == 0.0025)
                    for k, v in mats.items()) / P
    choice = solve(mats, base_bits)
    bits = sum(mats[k]["p"] * choice[k]["bits"] for k in mats) / P
    summary: Dict[str, int] = {}
    for c in choice.values():
        tag = f"r{c['r']} rho{c['sparsity']}"
        summary[tag] = summary.get(tag, 0) + 1
    db = sum(mats[k]["p"] * (choice[k]["wbits"] - bit_map[k]) for k in mats) / P
    print(f"presupuesto {base_bits:.3f} -> elegido {bits:.3f}; opciones {summary}; Δb medio {db:+.3f}", flush=True)

    option_map = {k: {"wbits": c["wbits"], "r": c["r"], "sparsity": c["sparsity"]} for k, c in choice.items()}
    set_seed(args.seed)
    model = get_opt()
    out = run_compression_mode(model, dataloader, DEV, mode="tube", nsamples=128, option_map=option_map, **TUBE)
    ppl_val = evaluate_ppl(model, valenc, DEV)
    ppl = evaluate_ppl(model, testenc, DEV)
    print(f"RESULTADO bits {out['avg_effective_bits']:.3f} val {ppl_val:.3f} test {ppl:.3f}", flush=True)
    path = RESULTS_DIR / "rk_alloc.json"
    results = json.loads(path.read_text()) if path.exists() else []
    results.append({"seed": args.seed, "budget_map": args.budget_map, "bits": out["avg_effective_bits"],
                    "ppl_val": ppl_val, "ppl": ppl, "summary": summary, "mean_db": db,
                    "option_map": option_map, "git": GIT})
    path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
