#!/usr/bin/env python3
"""
Prueba de una capa: ¿cuánto error funcional introduce cuantizar L = U V^T a 4 bits?

Para cada sublayer lineal de un bloque de OPT (q, k, v, out_proj, fc1, fc2):
  1. Construye el L continuo como el paso inicial de la Tríada:
       S = top-k(|W| * sqrt(diag H))  (k = sparsity),  L = SVD_r((W - S) L_H) L_H^-1
  2. Cuantiza L con 6 variantes: {minmax, hessian} x {col, row, mixed}
     y mide  loss_L = tr(dL H dL^T) / m   (misma normalización que functional_error).
  3. Referencias de escala:
       - energy_L: tr(L H L^T) / m  (cuánto "pesa" L)
       - err_Q{w}: error funcional de la base Q sola a w in {2,3,4} bits (triad_quant_only)
  4. (Opcional) Tríada completa por variante: error funcional final de Q+Lq+S.

Uso:
    uv run test_l_quant.py                       # bloque 0, r=16, l_bits=4, tríada con wbits=3
    uv run test_l_quant.py --layer 6 --triad_wbits 2
    uv run test_l_quant.py --skip_triad          # solo la medición de L (más rápido)
"""
from __future__ import annotations
import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.nn as nn

from pipeline import get_opt, find_sublayers, DEV
from engine import BipartiteQuantEngine, quantize_lowrank_factors, lowrank_bits
from data_loader import get_wikitext2, set_seed
from provenance import GIT

CURRENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = CURRENT_DIR / "results"
OUT_PATH = RESULTS_DIR / "l_quant_diagnostic.json"

VARIANTS = [(meth, sd) for meth in ("minmax", "hessian") for sd in ("col", "row", "mixed")]
BASELINE = ("minmax", "col")  # comportamiento actual de canonical_qls_l4


@torch.no_grad()
def capture_block_engines(layer_idx: int, dataloader, nsamples: int, dev,
                          model_id: str = "facebook/opt-125m") -> Dict[str, BipartiteQuantEngine]:
    """Propaga la calibración (sin cuantizar) hasta el bloque `layer_idx` y acumula H por sublayer."""
    model = get_opt(model_id)
    model.config.use_cache = False
    dec = model.model.decoder
    layers = dec.layers

    dec.embed_tokens = dec.embed_tokens.to(dev)
    dec.embed_positions = dec.embed_positions.to(dev)
    if getattr(dec, "project_in", None) is not None:
        dec.project_in = dec.project_in.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
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
    attention_mask = cache["attention_mask"]

    # Avanzar por los bloques previos sin cuantizar
    for i in range(layer_idx):
        blk = layers[i].to(dev)
        for j in range(nsamples):
            inps[j] = blk(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = blk.cpu()
        torch.cuda.empty_cache()

    blk = layers[layer_idx].to(dev)
    subset = find_sublayers(blk)
    engines = {name: BipartiteQuantEngine(subset[name]) for name in subset}

    def hook(name):
        def tmp(_, inp, out):
            engines[name].add_batch(inp[0].data, out.data)
        return tmp

    handles = [subset[n].register_forward_hook(hook(n)) for n in subset]
    for j in range(nsamples):
        blk(inps[j].unsqueeze(0), attention_mask=attention_mask)
    for h in handles:
        h.remove()

    del inps
    torch.cuda.empty_cache()
    return engines


@torch.no_grad()
def continuous_lowrank(eng: BipartiteQuantEngine, rank: int, sparsity: float):
    """Reproduce el paso inicial S -> L de run_qls_triad y devuelve (U_cont, V_cont)."""
    W = eng.W_orig
    m, n = eng.rows, eng.columns
    diag_H_sqrt = torch.sqrt(torch.diag(eng.H)).unsqueeze(0)
    k = int(m * n * sparsity)
    S = torch.zeros_like(W)
    if k > 0:
        sal = torch.abs(W) * diag_H_sqrt
        thresh = torch.topk(sal.flatten(), k, largest=True)[0][-1]
        mask = sal >= thresh
        S[mask] = W[mask]

    L_H, L_H_inv = eng.get_cholesky(percdamp=0.01)
    U, s, Vh = torch.linalg.svd((W - S) @ L_H, full_matrices=False)
    r_cur = min(rank, s.shape[0])
    sqrt_s = torch.sqrt(s[:r_cur])
    U_cont = U[:, :r_cur] * sqrt_s.unsqueeze(0)
    V_cont = (Vh[:r_cur, :] @ L_H_inv).t() * sqrt_s.unsqueeze(0)
    return U_cont, V_cont, r_cur


def hloss(eng: BipartiteQuantEngine, D: torch.Tensor) -> float:
    return torch.trace(D @ eng.H @ D.t()).item() / eng.rows


def main():
    ap = argparse.ArgumentParser(description="Diagnóstico de cuantización de L en una capa")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--sparsity", type=float, default=0.01)
    ap.add_argument("--l_bits", type=int, default=4)
    ap.add_argument("--nsamples", type=int, default=128)
    ap.add_argument("--triad_wbits", type=int, default=3)
    ap.add_argument("--skip_triad", action="store_true")
    args = ap.parse_args()

    set_seed(42)
    dataloader, _ = get_wikitext2(nsamples=args.nsamples, seed=42, model="facebook/opt-125m")
    t0 = time.time()
    engines = capture_block_engines(args.layer, dataloader, args.nsamples, DEV)
    print(f"[OK] Hessianas del bloque {args.layer} capturadas en {time.time() - t0:.1f}s")

    rows: List[Dict[str, Any]] = []
    for name, eng in engines.items():
        m, n = eng.rows, eng.columns
        U_c, V_c, r = continuous_lowrank(eng, args.rank, args.sparsity)
        L_c = U_c @ V_c.t()
        entry: Dict[str, Any] = {
            "layer": args.layer, "name": name, "m": m, "n": n, "r": r,
            "energy_L": hloss(eng, L_c),
            "variants": {},
        }

        # Referencias de escala: error de la base Q sola
        for w in (2, 3, 4):
            entry[f"err_Q{w}"] = eng.run_triad_quant_only(wbits=w, groupsize=64)["functional_error"]

        for meth, sd in VARIANTS:
            U_q, V_q = quantize_lowrank_factors(U_c, V_c, eng.H, args.l_bits, scale_dim=sd, method=meth)
            loss_L = hloss(eng, L_c - U_q @ V_q.t())
            v = {
                "loss_L": loss_L,
                "loss_L_over_energy": loss_L / max(entry["energy_L"], 1e-12),
                "bits_L_per_param": lowrank_bits(m, n, r, args.l_bits, sd) / (m * n),
            }
            if not args.skip_triad:
                res = eng.run_qls_triad(
                    wbits=args.triad_wbits, rank=args.rank, sparsity=args.sparsity,
                    groupsize=64, cycles=2, l_bits=args.l_bits,
                    scale_dim_l=sd, l_quant=meth,
                )
                v["triad_err"] = res["functional_error"]
                v["triad_bits_csr"] = res["effective_bits_csr"]
            entry["variants"][f"{meth}-{sd}"] = v

        rows.append(entry)

        # ---- Impresión por matriz ----
        print(f"\n== bloque {args.layer} / {name}  ({m}x{n}, r={r}) ==")
        print(f"   energía de L = {entry['energy_L']:.4g} | err Q2 = {entry['err_Q2']:.4g} | err Q3 = {entry['err_Q3']:.4g} | err Q4 = {entry['err_Q4']:.4g}")
        base = entry["variants"][f"{BASELINE[0]}-{BASELINE[1]}"]
        hdr = f"   {'variante':<16}{'loss_L':>12}{'/energía':>10}{'vs base':>9}{'bits_L':>9}"
        if not args.skip_triad:
            hdr += f"{'err tríada':>13}{'vs base':>9}"
        print(hdr)
        for key, v in entry["variants"].items():
            line = (f"   {key:<16}{v['loss_L']:>12.4g}{v['loss_L_over_energy']:>10.3%}"
                    f"{v['loss_L'] / max(base['loss_L'], 1e-12):>9.3f}{v['bits_L_per_param']:>9.4f}")
            if not args.skip_triad:
                line += (f"{v['triad_err']:>13.4g}"
                         f"{v['triad_err'] / max(base['triad_err'], 1e-12):>9.3f}")
            print(line)

    # ---- Resumen: media geométrica de la razón contra la base, sobre las matrices ----
    print("\n== Resumen (media geométrica sobre las matrices; < 1 es mejor que minmax-col) ==")
    summary = {}
    for meth, sd in VARIANTS:
        key = f"{meth}-{sd}"
        def gmean(field):
            vals = [r["variants"][key][field] / max(r["variants"]["minmax-col"][field], 1e-12) for r in rows]
            return math.exp(sum(math.log(max(x, 1e-12)) for x in vals) / len(vals))
        tot_params = sum(r["m"] * r["n"] for r in rows)
        bits = sum(r["variants"][key]["bits_L_per_param"] * r["m"] * r["n"] for r in rows) / tot_params
        s = {"loss_L_ratio": gmean("loss_L"), "bits_L_per_param": bits}
        if not args.skip_triad:
            s["triad_err_ratio"] = gmean("triad_err")
        summary[key] = s
        line = f"   {key:<16} loss_L x{s['loss_L_ratio']:.3f} | bits_L {bits:.4f}"
        if not args.skip_triad:
            line += f" | err tríada x{s['triad_err_ratio']:.3f}"
        print(line)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({"git": GIT, "args": vars(args), "per_matrix": rows, "summary": summary}, f, indent=2)
    print(f"\n[OK] Resultados guardados en {OUT_PATH}")


if __name__ == "__main__":
    main()