#!/usr/bin/env python3
"""
Combina el tubo con b por matriz: aplica tube.py sobre los mapas K-FAC ya resueltos en
rd_alloc.json, para comparar contra el mismo mapa con Q sola.

Uso:
    python test_tube_alloc.py
    python test_tube_alloc.py --seed 0 --budgets 3.5 --out tube_alloc_s0.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch

from pipeline import DEV, evaluate_ppl, get_opt, run_compression_mode
from data_loader import get_wikitext2, set_seed
from provenance import GIT

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main():
    ap = argparse.ArgumentParser(description="Tubo con b por matriz")
    ap.add_argument("--budgets", type=float, nargs="+", default=[3.25, 3.5, 4.0, 4.5])
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--sparsity", type=float, default=0.0025)
    ap.add_argument("--alpha_min", type=float, default=0.70)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_control", action="store_true")
    ap.add_argument("--out", default="tube_alloc.json")
    args = ap.parse_args()

    maps = {x["budget"]: x["bit_map"] for x in json.loads((RESULTS_DIR / "rd_alloc.json").read_text())
            if x["variant"] == "kfac"}
    jobs: List[Dict[str, Any]] = []
    if not args.no_control:
        jobs.append({"variant": "control_b3", "budget": 3.5, "bit_map": None})
    for b in args.budgets:
        jobs.append({"variant": "tube_kfac", "budget": b, "bit_map": maps[b]})

    set_seed(args.seed)
    dataloader, testenc = get_wikitext2(nsamples=128, seed=args.seed, model="facebook/opt-125m")
    path = RESULTS_DIR / args.out
    results = json.loads(path.read_text()) if path.exists() else []
    for job in jobs:
        set_seed(args.seed)
        model = get_opt()
        out = run_compression_mode(model, dataloader, DEV, mode="tube", wbits=3, r=args.rank,
                                   sparsity=args.sparsity, groupsize=64, l_bits=3, s_val_bits=4,
                                   s_index="rowcount", nsamples=128, bit_map=job["bit_map"],
                                   alpha_min=args.alpha_min)
        ppl = evaluate_ppl(model, testenc, DEV)
        bits = out.get("avg_bits_csr") or out.get("bits_csr") or out.get("bits")
        print(f"[{job['variant']}] presupuesto Q {job['budget']}: bits {bits} PPL {ppl:.3f}", flush=True)
        results.append({"variant": job["variant"], "budget": job["budget"], "seed": args.seed,
                        "alpha_min": args.alpha_min, "bits": bits, "ppl": ppl, "git": GIT})
        path.write_text(json.dumps(results, indent=2))
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
