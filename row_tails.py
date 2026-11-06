#!/usr/bin/env python3
"""
Colas de los pesos de OPT-125M dentro de cada fila y de cada grupo de cuantización (solo pesos, CPU).

outlier_stats.weight_stats estandariza con media y desvío de la matriz entera: una mezcla de filas
gaussianas con escalas distintas ya da colas pesadas. Acá se estandariza por fila y por grupo.

1. Por fila (media y desvío de la fila): fracción |w-mu| > 6 sigma, curtosis en exceso (mediana y p90 por
   matriz) y fracción de filas con algún peso a más de 6 sigma; nulo simulado con filas gaussianas del mismo n.
2. Por grupo de g columnas consecutivas de una fila (g = 64, 128 de Q sola, 256 del tubo):
   (max - min) / desvío del grupo, mediana y p95, contra el mismo cociente para g gaussianas iid.

Uso (desde la raíz del repositorio):
    HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= python row_tails.py
"""
from __future__ import annotations
import json
from pathlib import Path

import torch

from pipeline import get_opt, find_sublayers
from provenance import GIT

OUT = Path(__file__).resolve().parent / "results" / "row_tails.json"
SEED, NULL_ROWS, NULL_GROUPS, GROUPS = 0, 100_000, 200_000, (64, 128, 256)


def row_stats(W: torch.Tensor) -> dict:
    """W double [filas, n]; estadísticos por fila."""
    mu, sd = W.mean(1, keepdim=True), W.std(1, keepdim=True)
    z = (W - mu) / sd
    big = z.abs() > 6
    kurt = (z ** 4).mean(1) - 3
    return {"frac_gt_6sigma": float(big.double().mean()), "frac_rows_any_gt_6sigma": float(big.any(1).double().mean()),
            "kurt": kurt}


def group_ratio(W: torch.Tensor, g: int) -> torch.Tensor:
    G = W.reshape(W.shape[0], -1, g)
    return ((G.amax(-1) - G.amin(-1)) / G.std(-1)).flatten()


def q(v: torch.Tensor, p: float) -> float:
    v = v.double().flatten().sort().values
    return float(v[min(int(p * (v.numel() - 1) + 0.5), v.numel() - 1)])


def main():
    torch.set_num_threads(4)
    gen = torch.Generator().manual_seed(SEED)
    null = {}
    for n in (768, 3072):
        fr, anyr, ks = [], [], []
        for _ in range(NULL_ROWS // 10_000 if n == 768 else NULL_ROWS // 40_000):
            s = row_stats(torch.randn(10_000, n, generator=gen, dtype=torch.float64))
            fr.append(s["frac_gt_6sigma"]); anyr.append(s["frac_rows_any_gt_6sigma"]); ks.append(s["kurt"])
        k = torch.cat(ks)
        null[f"row_n{n}"] = {"frac_gt_6sigma": sum(fr) / len(fr), "frac_rows_any_gt_6sigma": sum(anyr) / len(anyr),
                             "kurt_median": q(k, 0.5), "kurt_p90": q(k, 0.9), "rows": int(k.numel()),
                             "frac_rows_any_gt_6sigma_analytic_iid": float(1 - (1 - 1.973e-9) ** n)}
        print(n, null[f"row_n{n}"], flush=True)
    for g in GROUPS:
        r = group_ratio(torch.randn(NULL_GROUPS, g, generator=gen, dtype=torch.float64), g)
        null[f"group_g{g}"] = {"median": q(r, 0.5), "p95": q(r, 0.95), "p99": q(r, 0.99), "groups": NULL_GROUPS}
        print(g, null[f"group_g{g}"], flush=True)

    model = get_opt("facebook/opt-125m")
    per_matrix, pooled = [], {}
    for li, blk in enumerate(model.model.decoder.layers):
        for name, lin in find_sublayers(blk).items():
            W = lin.weight.data.double()
            s = row_stats(W)
            rec = {"layer": li, "matrix": name, "n": W.shape[1], "rows": W.shape[0],
                   "frac_gt_6sigma_row": s["frac_gt_6sigma"], "frac_rows_any_gt_6sigma": s["frac_rows_any_gt_6sigma"],
                   "kurt_row_median": q(s["kurt"], 0.5), "kurt_row_p90": q(s["kurt"], 0.9)}
            P = pooled.setdefault(name, {"kurt": [], **{g: [] for g in GROUPS}, "w": []})
            P["kurt"].append(s["kurt"]); P["w"].append(rec)
            for g in GROUPS:
                r = group_ratio(W, g)
                rec[f"g{g}_ratio_median"], rec[f"g{g}_ratio_p95"] = q(r, 0.5), q(r, 0.95)
                rec[f"g{g}_frac_gt_null_p99"] = float((r > null[f"group_g{g}"]["p99"]).double().mean())
                P[g].append(r)
            per_matrix.append(rec)
    by_type = {}
    for m, P in sorted(pooled.items()):
        tot = sum(r["rows"] for r in P["w"])
        k = torch.cat(P["kurt"])
        d = {"n": P["w"][0]["n"],
             "frac_gt_6sigma_row": sum(r["frac_gt_6sigma_row"] * r["rows"] for r in P["w"]) / tot,
             "frac_rows_any_gt_6sigma": sum(r["frac_rows_any_gt_6sigma"] * r["rows"] for r in P["w"]) / tot,
             "frac_rows_any_gt_6sigma_max_matrix": max(r["frac_rows_any_gt_6sigma"] for r in P["w"]),
             "kurt_row_median": q(k, 0.5), "kurt_row_p90": q(k, 0.9)}
        for g in GROUPS:
            r = torch.cat(P[g])
            d[f"g{g}_ratio_median"], d[f"g{g}_ratio_p95"] = q(r, 0.5), q(r, 0.95)
            d[f"g{g}_frac_gt_null_p99"] = float((r > null[f"group_g{g}"]["p99"]).double().mean())
        by_type[m] = d
        print(m, d, flush=True)
    OUT.write_text(json.dumps({"model": "facebook/opt-125m", "seed": SEED, "null": null, "by_type": by_type,
                               "per_matrix": per_matrix, "git": GIT}, indent=2))


if __name__ == "__main__":
    main()
