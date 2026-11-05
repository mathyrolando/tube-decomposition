#!/usr/bin/env python3
"""
Prueba espectral: ¿el error de cuantización de Q tiene estructura de rango bajo?

GPTQ cumple exactamente E = W - Q = Err U, con H^{-1} = U^T U (U triangular superior)
y Err_ij = e_ij / U_jj, donde e_ij es el error de redondeo de la columna j ya compensada.
Por lo tanto M = E U^{-1} es el ruido de redondeo en la base donde es entrada a entrada,
y tiene los mismos valores singulares que E L_H.

L puede capturar a lo sumo la energía de los r mayores valores singulares de M. Si M es
ruido sin estructura, esa fracción la fija su perfil de varianzas. Se mide contra:
  - borde MP homogéneo: r (sqrt(m) + sqrt(n))^2 / (m n)
  - nulo por signos: M_ij -> +-M_ij al azar. Conserva |M_ij| y destruye solo la coherencia.

Conversión a bits: capturar una fracción eps ahorra -ln(1 - eps) / pendiente bits de Q, con
pendiente = ln(err_b / err_{b+1}) medida en esa matriz. Entonces
    c* = bits_ahorrados / (r (m + n) / (m n))
son los bits por entrada de factor a los que L empata con gastar en Q. Es una cota
optimista: supone E fijo, sin recuantizar.

También mide qué parte de la energía viene de entradas recortadas (|e_ij| > Delta/2).

Uso:
    python test_spectrum.py --layers 0 6 11
"""
from __future__ import annotations
import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

from pipeline import DEV
from data_loader import get_wikitext2, set_seed
from test_l_quant import capture_block_engines
from provenance import GIT

CURRENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = CURRENT_DIR / "results"
OUT_PATH = RESULTS_DIR / "spectrum_diagnostic.json"


def rounding_stats(Q: torch.Tensor, e: torch.Tensor, M: torch.Tensor, g: int) -> Dict[str, float]:
    """Separa redondeo interno de recorte, estimando Delta por fila y grupo desde Q."""
    n = Q.shape[1]
    tol = 1e-6 * Q.abs().max().clamp(min=1e-12)
    clipped = torch.zeros_like(Q, dtype=torch.bool)
    delta = torch.zeros_like(Q)
    for start in range(0, n, g):
        end = min(start + g, n)
        gaps = torch.sort(Q[:, start:end], dim=1).values.diff(dim=1)
        gaps = gaps.masked_fill(gaps <= tol, float("inf"))
        d = gaps.min(dim=1).values if gaps.shape[1] else torch.full((Q.shape[0],), float("inf"), device=Q.device)
        d = d.unsqueeze(1).expand(-1, end - start)
        delta[:, start:end] = d
        clipped[:, start:end] = e[:, start:end].abs() > 0.5 * d * 1.001

    valid = torch.isfinite(delta)
    inner = valid & ~clipped
    total = M.pow(2).sum().clamp(min=1e-30)
    uniform_ratio = (e[inner].pow(2).sum() / (delta[inner].pow(2).sum() / 12).clamp(min=1e-30)).item()
    return {
        "clip_frac_entries": clipped.float().mean().item(),
        "clip_frac_energy": (M[clipped].pow(2).sum() / total).item(),
        # 1.0 significa que dentro del rango vale exactamente Delta^2 / 12
        "inner_var_over_uniform": uniform_ratio,
    }


def analyze(eng, b: int, g: int, ranks: List[int], n_null: int, slope: float, res: Dict[str, Any]) -> Dict[str, Any]:
    L_H, _ = eng.get_cholesky()
    U = torch.linalg.cholesky(torch.cholesky_inverse(L_H), upper=True)
    Q = res["W_approx"]
    E = eng.W_orig - Q
    M = torch.linalg.solve_triangular(U, E, upper=True, left=False)
    e = M * torch.diag(U).unsqueeze(0)

    m, n = eng.rows, eng.columns
    sv2 = torch.linalg.svdvals(M).pow(2)
    total = sv2.sum().item()

    nulls = []
    for _ in range(n_null):
        signs = torch.randint(0, 2, M.shape, device=M.device, dtype=M.dtype) * 2 - 1
        nulls.append(torch.linalg.svdvals(M * signs).pow(2))
    nulls = torch.stack(nulls)
    null_total = nulls.sum(dim=1, keepdim=True)
    null_cum = (nulls.cumsum(dim=1) / null_total)

    per_rank = {}
    cum = (sv2.cumsum(0) / total)
    for r in ranks:
        cap = cum[r - 1].item()
        null_vals = null_cum[:, r - 1]
        null_mean = null_vals.mean().item()
        null_std = null_vals.std().item() if n_null > 1 else 0.0
        factor_ratio = r * (m + n) / (m * n)
        saved = -math.log(max(1 - cap, 1e-12)) / slope
        saved_null = -math.log(max(1 - null_mean, 1e-12)) / slope
        per_rank[r] = {
            "captured": cap,
            "null_mean": null_mean,
            "null_std": null_std,
            "excess_ratio": cap / null_mean,
            "mp_bound": r * (math.sqrt(m) + math.sqrt(n)) ** 2 / (m * n),
            "bits_saved": saved,
            "c_star": saved / factor_ratio,
            "c_star_null": saved_null / factor_ratio,
        }

    return {
        "err": res["functional_error"],
        "err_whitened": total / m,
        "slope_per_bit": slope,
        "sigma1_over_null": (sv2[0] / nulls[:, 0].mean()).sqrt().item(),
        "ranks": per_rank,
        **rounding_stats(Q, e, M, g),
    }


def main():
    ap = argparse.ArgumentParser(description="Espectro de E L_H contra un nulo sin estructura")
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 6, 11])
    ap.add_argument("--wbits", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--groupsize", type=int, default=64)
    ap.add_argument("--ranks", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--null_draws", type=int, default=5)
    ap.add_argument("--nsamples", type=int, default=128)
    args = ap.parse_args()

    set_seed(42)
    dataloader, _ = get_wikitext2(nsamples=args.nsamples, seed=42, model="facebook/opt-125m")
    levels = sorted(set(args.wbits) | {w + 1 for w in args.wbits})

    rows: List[Dict[str, Any]] = []
    print(f"{'bloque':<7}{'matriz':<20}{'b':>2}{'err':>11}{'%rec':>7}{'E rec':>7}{'var/U':>7}"
          f"{'r16':>8}{'nulo':>8}{'MP':>8}{'exc':>7}{'bits':>8}{'c*':>7}{'c*nulo':>8}", flush=True)
    for layer in args.layers:
        t0 = time.time()
        engines = capture_block_engines(layer, dataloader, args.nsamples, DEV)
        for name, eng in engines.items():
            with torch.no_grad():
                runs = {b: eng.run_triad_quant_only(wbits=b, groupsize=args.groupsize) for b in levels}
                for b in args.wbits:
                    slope = math.log(runs[b]["functional_error"] / runs[b + 1]["functional_error"])
                    entry = analyze(eng, b, args.groupsize, args.ranks, args.null_draws, slope, runs[b])
                    entry.update(layer=layer, name=name, m=eng.rows, n=eng.columns, wbits=b)
                    rows.append(entry)
                    r = entry["ranks"][16]
                    print(f"{layer:<7}{name:<20}{b:>2}{entry['err']:>11.4g}"
                          f"{100 * entry['clip_frac_entries']:>7.2f}{100 * entry['clip_frac_energy']:>7.1f}"
                          f"{entry['inner_var_over_uniform']:>7.2f}"
                          f"{r['captured']:>8.4f}{r['null_mean']:>8.4f}{r['mp_bound']:>8.4f}"
                          f"{r['excess_ratio']:>7.2f}{r['bits_saved']:>8.3f}{r['c_star']:>7.2f}"
                          f"{r['c_star_null']:>8.2f}", flush=True)
        del engines
        torch.cuda.empty_cache()
        print(f"[OK] bloque {layer} en {time.time() - t0:.1f}s", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({"git": GIT, "args": vars(args), "per_matrix": rows}, f, indent=2)
    print(f"[OK] Resultados guardados en {OUT_PATH}")


if __name__ == "__main__":
    main()
