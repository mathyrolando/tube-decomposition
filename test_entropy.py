#!/usr/bin/env python3
"""
Entropía real de los índices de Q, con y sin tubo. Compara PPL contra bits en dos ejes:
tasa fija y entropía, reemplazando b por H(índices). L y S conservan su costo de tasa fija.

Uso:
    python test_entropy.py --seed 42
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
OUT_PATH = RESULTS_DIR / "entropy_comparison.json"
TUBE = dict(mode="tube", wbits=3, r=16, sparsity=0.0025, groupsize=64, l_bits=3, s_val_bits=4,
            s_index="rowcount", alpha_min=0.70)


def main():
    ap = argparse.ArgumentParser(description="Entropía de índices de Q")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", nargs="*", default=None, help="Nombres de configuración a correr")
    args = ap.parse_args()

    maps = {x["budget"]: x["bit_map"] for x in json.loads((RESULTS_DIR / "rd_alloc.json").read_text())
            if x["variant"] == "kfac"}
    configs: Dict[str, Dict[str, Any]] = {}
    for b, g in [(3, 64), (3, 128), (3, 256), (3, 1024), (4, 64), (4, 128), (4, 256), (4, 1024), (5, 1024)]:
        configs[f"Q b{b} g{g}"] = dict(mode="triad_quant_only", wbits=b, groupsize=g)
    for budget in (3.5, 4.0, 4.5):
        configs[f"Q kfac {budget}"] = dict(mode="q_bitmap", groupsize=64, bit_map=maps[budget])
        configs[f"tubo kfac {budget}"] = dict(TUBE, bit_map=maps[budget])
    configs["tubo b3 g64"] = dict(TUBE)
    configs["tubo b3 g256"] = dict(TUBE, groupsize=256)
    configs["tubo b4 g256"] = dict(TUBE, wbits=4, groupsize=256)
    configs["tubo solo S b3 g64"] = dict(TUBE, r=0)
    configs["tubo solo L b3 g64"] = dict(TUBE, sparsity=0.0)
    for budget in (3.5, 4.0, 4.5):
        configs[f"Q kfac {budget} g256"] = dict(mode="q_bitmap", groupsize=256, bit_map=maps[budget])
        configs[f"tubo kfac {budget} g256"] = dict(TUBE, groupsize=256, bit_map=maps[budget])
    for b in (3, 4, 5):
        configs[f"tubo b{b} g1024"] = dict(TUBE, wbits=b, groupsize=1024)
    if args.only:
        configs = {k: v for k, v in configs.items() if k in args.only}

    results = json.loads(OUT_PATH.read_text()) if OUT_PATH.exists() else []
    done = {(r["config"], r["seed"]) for r in results}
    set_seed(args.seed)
    dataloader, testenc = get_wikitext2(nsamples=128, seed=args.seed, model="facebook/opt-125m")
    for name, cfg in configs.items():
        if (name, args.seed) in done:
            continue
        set_seed(args.seed)
        model = get_opt()
        out = run_compression_mode(model, dataloader, DEV, nsamples=128, **cfg)
        ppl = evaluate_ppl(model, testenc, DEV)
        stats = out["layer_stats"]
        total = sum(s["numel"] for s in stats)
        mean = lambda f: sum(f(s) * s["numel"] for s in stats) / total
        rec = {"config": name, "seed": args.seed, "ppl": ppl, "bits": out["avg_effective_bits"],
               "wbits": mean(lambda s: s["wbits"]), "git": GIT}
        for kind in ("rel", "zero", "grp"):
            rec[f"H_{kind}"] = mean(lambda s: s["entropy"][kind])
            rec[f"bits_{kind}"] = rec["bits"] - rec["wbits"] + rec[f"H_{kind}"]
        print(f"{name}: tasa fija {rec['bits']:.3f} H_zero {rec['H_zero']:.3f} H_grp {rec['H_grp']:.3f} "
              f"bits_ent {rec['bits_zero']:.3f} PPL {ppl:.3f}", flush=True)
        results.append(rec)
        OUT_PATH.write_text(json.dumps(results, indent=2))
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
