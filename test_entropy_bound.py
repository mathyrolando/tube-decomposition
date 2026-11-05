#!/usr/bin/env python3
"""
Cota rápida: ¿puede pagar L cuando los índices de Q se comprimen por entropía?

Con paso Δ fijo y sin límite de niveles, el error de redondeo es ≈ Δ²/12 por peso,
sea cual sea el residuo T = W - L. Entonces, a igual error, L gana solo si baja la
entropía de los índices round(T/Δ) más de lo que cuesta guardarla:

    ganancia_neta = H(round(W/Δ)) - H(round(T/Δ)) - bits_L

Δ se elige por matriz para que H(round(W/Δ)) = objetivo (2.5 o 3 bits). L = SVD_r de W
(Frobenius), que no es la L óptima para la entropía: es una referencia, no una cota exacta.
bits_L = l_bits · r(m+n) / (mn), con L cuantizada a l_bits como en el tubo.
Solo usa los pesos (no hace falta calibración).

Uso:
    python test_entropy_bound.py --model facebook/opt-125m
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch
from transformers import OPTForCausalLM

from provenance import GIT

RESULTS = Path(__file__).resolve().parent / "results"
OUT = RESULTS / "entropy_bound.json"
NAMES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj", "fc1", "fc2")


def entropy(x: torch.Tensor, delta: float) -> float:
    _, c = torch.unique(torch.round(x / delta), return_counts=True)
    p = c.double() / c.sum()
    return float(-(p * p.log2()).sum())


def delta_for(x: torch.Tensor, target: float) -> float:
    lo, hi = 1e-6 * x.abs().max().item(), x.abs().max().item()
    for _ in range(40):
        mid = (lo * hi) ** 0.5
        if entropy(x, mid) > target:
            lo = mid
        else:
            hi = mid
    return (lo * hi) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/opt-125m")
    ap.add_argument("--targets", type=float, nargs="+", default=[2.5, 3.0])
    ap.add_argument("--ranks", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--l_bits", type=int, default=3)
    args = ap.parse_args()

    model = OPTForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
    rows = []
    for i, layer in enumerate(model.model.decoder.layers):
        for name in NAMES:
            W = layer.get_submodule(name).weight.detach().float()
            m, n = W.shape
            U, s, Vh = torch.linalg.svd(W, full_matrices=False)
            for t in args.targets:
                d = delta_for(W, t)
                hw = entropy(W, d)
                for r in args.ranks:
                    T = W - (U[:, :r] * s[:r]) @ Vh[:r]
                    ht = entropy(T, d)
                    bl = args.l_bits * r * (m + n) / (m * n)
                    rows.append({"layer": i, "name": name, "m": m, "n": n, "target": t, "r": r,
                                 "H_W": hw, "H_T": ht, "bits_L": bl, "net": hw - ht - bl,
                                 "energy_L": float((s[:r] ** 2).sum() / (s ** 2).sum())})
        print(f"[OK] bloque {i}", flush=True)

    P = lambda x: x["m"] * x["n"]
    summary = {}
    for t in args.targets:
        for r in args.ranks:
            sel = [x for x in rows if x["target"] == t and x["r"] == r]
            tot = sum(P(x) for x in sel)
            mean = lambda k: sum(x[k] * P(x) for x in sel) / tot
            summary[f"{t}_r{r}"] = {"saving": mean("H_W") - mean("H_T"), "bits_L": mean("bits_L"),
                                    "net": mean("net"), "energy_L": mean("energy_L"),
                                    "frac_pos": sum(P(x) for x in sel if x["net"] > 0) / tot}
            s_ = summary[f"{t}_r{r}"]
            print(f"objetivo {t} r{r}: ahorro {s_['saving']:.4f}  costo L {s_['bits_L']:.4f}  neto {s_['net']:+.4f}  "
                  f"energía L {s_['energy_L']:.3f}  fracción de parámetros con neto>0 {s_['frac_pos']:.2f}", flush=True)
        by = {}
        for x in rows:
            if x["target"] == t and x["r"] == 16:
                by.setdefault(x["name"], []).append(x["net"])
        print(f"  neto r16 por tipo (objetivo {t}): " +
              "  ".join(f"{k.split('.')[-1]} {sum(v)/len(v):+.3f}" for k, v in by.items()), flush=True)

    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    out[args.model] = {"args": vars(args), "git": GIT, "summary": summary, "rows": rows}
    OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
