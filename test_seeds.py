#!/usr/bin/env python3
"""
Comparación pareada por semilla de calibración entre Q sola y el tubo con b por matriz.
Los mapas K-FAC se toman de rd_alloc.json (resueltos con semilla 42) y no se reoptimizan.

Uso:
    python test_seeds.py --seeds 42 0 1
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch

from pipeline import DEV, evaluate_ppl, get_opt, run_compression_mode
from data_loader import get_wikitext2, set_seed
from provenance import GIT

RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUT_PATH = RESULTS_DIR / "seeds_comparison.json"
TUBE = dict(mode="tube", wbits=3, r=16, sparsity=0.0025, groupsize=64, l_bits=3, s_val_bits=4,
            s_index="rowcount", alpha_min=0.70)


def main():
    ap = argparse.ArgumentParser(description="Q sola contra tubo con b por matriz, varias semillas")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 0, 1])
    args = ap.parse_args()

    maps = {x["budget"]: x["bit_map"] for x in json.loads((RESULTS_DIR / "rd_alloc.json").read_text())
            if x["variant"] == "kfac"}
    configs: Dict[str, Dict[str, Any]] = {
        "Q b3 g64": dict(mode="triad_quant_only", wbits=3, groupsize=64),
        "Q b4 g256": dict(mode="triad_quant_only", wbits=4, groupsize=256),
        "Q b4 g64": dict(mode="triad_quant_only", wbits=4, groupsize=64),
        "Q kfac 3.5": dict(mode="q_bitmap", groupsize=64, bit_map=maps[3.5]),
        "Q kfac 4.0": dict(mode="q_bitmap", groupsize=64, bit_map=maps[4.0]),
        "Q kfac 4.5": dict(mode="q_bitmap", groupsize=64, bit_map=maps[4.5]),
        "tubo kfac 3.5": dict(TUBE, bit_map=maps[3.5]),
        "tubo kfac 4.0": dict(TUBE, bit_map=maps[4.0]),
        "tubo kfac 4.5": dict(TUBE, bit_map=maps[4.5]),
    }

    results = json.loads(OUT_PATH.read_text()) if OUT_PATH.exists() else []
    done = {(r["config"], r["seed"]) for r in results}
    for seed in args.seeds:
        set_seed(seed)
        dataloader, testenc = get_wikitext2(nsamples=128, seed=seed, model="facebook/opt-125m")
        for name, cfg in configs.items():
            if (name, seed) in done:
                continue
            set_seed(seed)
            model = get_opt()
            out = run_compression_mode(model, dataloader, DEV, nsamples=128, **cfg)
            ppl = evaluate_ppl(model, testenc, DEV)
            bits = out["avg_effective_bits"]
            print(f"[seed {seed}] {name}: bits {bits:.3f} PPL {ppl:.3f}", flush=True)
            results.append({"config": name, "seed": seed, "bits": bits, "ppl": ppl, "git": GIT})
            OUT_PATH.write_text(json.dumps(results, indent=2))
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
