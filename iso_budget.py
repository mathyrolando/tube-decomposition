#!/usr/bin/env python3
"""
Diagnóstico de Compensación Iso-Presupuesto (R vs K):
Analiza el trade-off entre rango bajo y esparsidad en esquemas bipartitos.
Ejecuta:
1. Barrido a rho fijo = 0.50 variando kappa in [0.10, 0.85].
2. Barrido iso-bits estricto: fija un presupuesto de bits constante (target_bits, ej: 4.0b)
   despejando rho(kappa) mediante bisección sobre hardware real (min(csr, bitmap)).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn

from pipeline import get_opt, find_sublayers, DEV
from engine import BipartiteQuantEngine, compute_effective_bits_bipartite
from data_loader import get_wikitext2, set_seed
from provenance import GIT

CURRENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = CURRENT_DIR / "results"
ISO_RESULTS_PATH = RESULTS_DIR / "iso_budget_sweep_diagnostic.json"


def run_iso_budget_sweep_diagnostic(
    dataloader,
    dev=DEV,
    nsamples: int = 128,
    target_bits: Optional[float] = None
) -> Dict[str, Any]:
    print("\n[DIAGNÓSTICO ISO-PRESUPUESTO (R vs K)]")
    print("Evaluando combinaciones de asignación bajo presupuesto de memoria fijo...")

    model = get_opt()
    layers = model.model.decoder.layers
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    layers[0] = layers[0].to(dev)

    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=torch.float16, device=dev)
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader[:nsamples]:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    subset = find_sublayers(layers[0])
    target_layer = subset["self_attn.out_proj"]
    eng = BipartiteQuantEngine(target_layer)

    def add_b(_, inp, out):
        eng.add_batch(inp[0].data, out.data)

    h = target_layer.register_forward_hook(add_b)
    for j in range(nsamples):
        layers[0](inps[j].unsqueeze(0), attention_mask=cache["attention_mask"])
    h.remove()

    kappas = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.85]
    fixed_rho_results = []

    print("--- 1. Barrido con rho = 0.50 (parámetros retenidos fijos) ---")
    for kap in kappas:
        res_fp16 = eng.run_oats_fp16(rho=0.50, kappa=kap, iterations=30, use_cholesky=False)
        res_q = eng.run_oats_quantized_sequential(rho=0.50, kappa=kap, u_bits=4, v_bits=4, s_bits=8, iterations=30)

        entry = {
            "kappa": kap,
            "rho": 0.50,
            "r": res_fp16["r"],
            "k": res_fp16["k"],
            "sparsity_pct": 100.0 * res_fp16["k"] / (eng.rows * eng.columns),
            "err_fp16": res_fp16["functional_error"],
            "err_quantized": res_q["functional_error"],
            "effective_bits_csr": res_q["effective_bits"],
            "effective_bits_bitmap": res_q.get("effective_bits_bitmap")
        }
        fixed_rho_results.append(entry)
        print(f"  kappa={kap:.2f} | r={res_fp16['r']:3d} | k_density={entry['sparsity_pct']:.1f}% | "
              f"Err FP16={res_fp16['functional_error']:.2f} | Err Quant={res_q['functional_error']:.2f} | "
              f"Bits={res_q['effective_bits']:.2f}")

    # 2. Barrido Iso-Bits Estricto
    b_target = target_bits if target_bits is not None else 4.0
    print(f"\n--- 2. Barrido Iso-Bits Estricto (Presupuesto objetivo = {b_target:.2f} bits) ---")
    strict_isobits_results = []
    u_b, s_b = 4, 8

    for kap in kappas:
        def bits_for_rho(r_val):
            m, n = eng.rows, eng.columns
            d_geom = (m * n) / (m + n)
            r_cur = max(1, int(kap * (1.0 - r_val) * d_geom))
            k_cur = max(1, int((1.0 - kap) * (1.0 - r_val) * m * n))
            b_csr, b_bmp = compute_effective_bits_bipartite(
                m, n, r_cur, k_cur, u_bits=u_b, v_bits=u_b, s_bits=s_b, s_quant_mode="rowwise"
            )
            return min(b_csr, b_bmp)

        lo_r, hi_r = 0.01, 0.98
        for _ in range(16):
            mid_r = 0.5 * (lo_r + hi_r)
            if bits_for_rho(mid_r) > b_target:
                lo_r = mid_r
            else:
                hi_r = mid_r
        rho_solved = float(0.5 * (lo_r + hi_r))

        res_fp16 = eng.run_oats_fp16(rho=rho_solved, kappa=kap, iterations=30, use_cholesky=False)
        res_q = eng.run_oats_quantized_sequential(rho=rho_solved, kappa=kap, u_bits=u_b, v_bits=u_b, s_bits=s_b, iterations=30)

        entry = {
            "kappa": kap,
            "rho_solved": rho_solved,
            "r": res_fp16["r"],
            "k": res_fp16["k"],
            "target_bits": b_target,
            "sparsity_pct": 100.0 * res_fp16["k"] / (eng.rows * eng.columns),
            "err_fp16": res_fp16["functional_error"],
            "err_quantized": res_q["functional_error"],
            "effective_bits_csr": res_q["effective_bits"],
            "effective_bits_bitmap": res_q.get("effective_bits_bitmap"),
            "bits_paper": res_q.get("bits_paper")
        }
        strict_isobits_results.append(entry)
        print(f"  kappa={kap:.2f} | rho={rho_solved:.3f} | r={res_fp16['r']:3d} | "
              f"Err FP16={res_fp16['functional_error']:.2f} | Err Quant={res_q['functional_error']:.2f} | "
              f"Bits Paper={entry['bits_paper']:.2f} | Bits Bmp={res_q.get('effective_bits_bitmap', 0):.2f}")

    return {
        "fixed_rho_results": fixed_rho_results,
        "strict_isobits_results": strict_isobits_results
    }


def main():
    parser = argparse.ArgumentParser(description="Diagnóstico Iso-Presupuesto R vs K")
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--target_bits", type=float, default=4.0)
    args = parser.parse_args()

    set_seed(42)
    dataloader, _ = get_wikitext2(nsamples=args.nsamples, seed=42, model="facebook/opt-125m")
    res = run_iso_budget_sweep_diagnostic(dataloader, dev=DEV, nsamples=args.nsamples, target_bits=args.target_bits)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ISO_RESULTS_PATH, "w") as f:
        json.dump({"git": GIT, **res}, f, indent=2)
    print(f"\n[OK] Resultados guardados en {ISO_RESULTS_PATH}")


if __name__ == "__main__":
    main()
