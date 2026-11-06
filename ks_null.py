#!/usr/bin/env python3
"""
Nulo del KS por fila (Lilliefors: media y desvío estimados de la propia fila) y fracción de filas de
OPT-125M que lo superan. Mismo estadístico que outlier_stats.ks_normal. Solo pesos, CPU.

Uso (desde la raíz del repositorio):
    HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= python ks_null.py
"""
from __future__ import annotations
import json
from pathlib import Path

import torch

from outlier_stats import ks_normal
from pipeline import get_opt, find_sublayers
from provenance import GIT

OUT = Path(__file__).resolve().parent / "results" / "ks_null.json"
REPS, SEED = 10_000, 0


def main():
    torch.set_num_threads(4)
    gen = torch.Generator().manual_seed(SEED)
    null = {}
    for n in (768, 3072):
        ks = torch.cat([ks_normal(torch.randn(1000, n, generator=gen, dtype=torch.float64))
                        for _ in range(REPS // 1000)])
        q = torch.quantile(ks, torch.tensor([0.05, 0.5, 0.95, 0.99], dtype=torch.float64))
        null[n] = {"p5": float(q[0]), "median": float(q[1]), "p95": float(q[2]), "p99": float(q[3]), "reps": REPS}
        print(n, null[n], flush=True)

    model = get_opt("facebook/opt-125m")
    per_matrix = []
    for li, blk in enumerate(model.model.decoder.layers):
        for name, lin in find_sublayers(blk).items():
            W = lin.weight.data.double()
            n = W.shape[1]
            ks = ks_normal(W)
            per_matrix.append({"layer": li, "matrix": name, "n": n, "rows": W.shape[0],
                               "frac_rows_gt_p95": float((ks > null[n]["p95"]).double().mean()),
                               "frac_rows_gt_p99": float((ks > null[n]["p99"]).double().mean()),
                               "ks_row_median": float(ks.median())})
    by_type = {}
    for m in sorted({r["matrix"] for r in per_matrix}):
        rows = [r for r in per_matrix if r["matrix"] == m]
        tot = sum(r["rows"] for r in rows)
        by_type[m] = {"n": rows[0]["n"],
                      "frac_rows_gt_p95_pooled": sum(r["frac_rows_gt_p95"] * r["rows"] for r in rows) / tot,
                      "frac_rows_gt_p99_pooled": sum(r["frac_rows_gt_p99"] * r["rows"] for r in rows) / tot,
                      "frac_rows_gt_p95_min": min(r["frac_rows_gt_p95"] for r in rows),
                      "frac_rows_gt_p95_max": max(r["frac_rows_gt_p95"] for r in rows)}
        print(m, by_type[m], flush=True)
    OUT.write_text(json.dumps({"model": "facebook/opt-125m", "statistic": "KS contra N(media, desvío) de la fila",
                               "null": {str(k): v for k, v in null.items()}, "seed": SEED,
                               "by_type": by_type, "per_matrix": per_matrix, "git": GIT}, indent=2))


if __name__ == "__main__":
    main()
