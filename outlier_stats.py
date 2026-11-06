#!/usr/bin/env python3
"""
Activaciones atípicas, concentración de energía de H = X X^T y gaussianidad de los pesos (OPT), en CPU.

Calibración: la del pipeline (WikiText-2 train, 128 secuencias de 2048 tokens, semilla 42), modelo SIN comprimir
en float16 (como en el pipeline). Las entradas X se capturan con hooks en las 6 capas lineales de cada bloque;
q/k/v comparten entrada (salida de self_attn_layer_norm) y fc1 recibe la salida de final_layer_norm.

A1. Criterio de LLM.int8() (Dettmers et al. 2022, Tabla 4): medido sobre las entradas de atención (q/k/v)
    y de FFN (fc1), es decir las salidas de LayerNorm: 2 "capas" por bloque. Una coordenada j califica en
    una capa si |x_j| >= 6 en >= 6% de las posiciones; es atípica si califica en >= 25% de las capas.
A2. Por capa lineal: max|x|, curtosis (Pearson, gaussiana = 3) de todas las entradas, y
    max_j max|x_j| / mediana_j max|x_j|.
A3. Por capa lineal: fracción de tr(H) en las 1/4/16 coordenadas con mayor H_jj y en los 1/4/16 autovalores
    mayores; número de condición de H + lambda I con lambda = 0.01 * media(diag H) (percdamp de GPTQ,
    columnas muertas con H_jj = 1, como en engine.get_cholesky).
B.  Por matriz W: KS contra N(media, desvío) de la matriz (y por fila: mediana y máximo), curtosis en
    exceso, fracción de |w - media| > 6 desvío.

Uso (desde la raíz del repositorio):
    HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= python outlier_stats.py [--model facebook/opt-125m]
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn

from pipeline import get_opt, find_sublayers
from data_loader import get_wikitext2, set_seed
from provenance import GIT

RESULTS = Path(__file__).resolve().parent / "results"
NSAMPLES, SEED, THR = 128, 42, 6.0
# Histograma de |x| >= 6 por coordenada (para cuartiles): bins log-espaciados de 6 a 1e4.
EDGES = torch.logspace(math.log10(THR), 4, 241, dtype=torch.float64)
INPUT_OF = {"self_attn.q_proj": "attn_in", "self_attn.k_proj": "attn_in", "self_attn.v_proj": "attn_in",
            "self_attn.out_proj": "out_in", "fc1": "ffn_in", "fc2": "fc2_in"}


class Acc:
    """Estadísticos acumulados de las entradas de una capa (una por tipo de entrada y bloque)."""

    def __init__(self, d: int):
        self.n = 0
        self.m = torch.zeros(5, dtype=torch.float64)  # sumas de x^0..x^4 sobre todas las entradas
        self.cmax = torch.zeros(d, dtype=torch.float64)
        self.cnt6 = torch.zeros(d, dtype=torch.float64)  # posiciones con |x_j| >= 6
        self.hist = torch.zeros(d, len(EDGES), dtype=torch.float64)
        self.H = torch.zeros(d, d, dtype=torch.float64)

    def add(self, x: torch.Tensor):
        x = x.reshape(-1, x.shape[-1]).double()
        a = x.abs()
        self.n += x.shape[0]
        for p in range(5):
            self.m[p] += (x ** p).sum()
        self.cmax = torch.maximum(self.cmax, a.max(0).values)
        big = a >= THR
        self.cnt6 += big.sum(0)
        if big.any():
            rows, cols = big.nonzero(as_tuple=True)
            b = torch.bucketize(a[rows, cols], EDGES, right=True) - 1
            self.hist.index_put_((cols, b.clamp(0, len(EDGES) - 1)), torch.ones_like(b, dtype=torch.float64),
                                 accumulate=True)
        self.H += x.t() @ x

    def kurtosis(self) -> float:
        n, s1, s2, s3, s4 = self.m.tolist()
        mu = s1 / n
        var = s2 / n - mu ** 2
        m4 = s4 / n - 4 * mu * s3 / n + 6 * mu ** 2 * s2 / n - 3 * mu ** 4
        return m4 / var ** 2


def h_stats(H: torch.Tensor) -> dict:
    H = H.clone()
    d = torch.diag(H)
    tr = float(d.sum())
    dsort = d.sort(descending=True).values
    ev = torch.linalg.eigvalsh(H).flip(0).clamp_min(0)
    Hd = H.clone()
    dead = d == 0
    Hd[dead, dead] = 1
    lam = 0.01 * float(torch.diag(Hd).mean())
    evd = torch.linalg.eigvalsh(Hd + lam * torch.eye(len(d), dtype=H.dtype))
    return {**{f"diag_top{k}": float(dsort[:k].sum()) / tr for k in (1, 4, 16)},
            **{f"eig_top{k}": float(ev[:k].sum()) / tr for k in (1, 4, 16)},
            "cond_damped": float(evd[-1] / evd[0]), "damping": "0.01*mean(diag H)", "dead_cols": int(dead.sum())}


def ks_normal(v: torch.Tensor) -> float:
    """Estadístico KS de v contra N(media(v), desvío(v)) (vectorizado sobre la última dimensión)."""
    v = v.double()
    mu, sd = v.mean(-1, keepdim=True), v.std(-1, keepdim=True)
    z = ((v - mu) / sd).sort(-1).values
    n = z.shape[-1]
    F = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
    i = torch.arange(1, n + 1, dtype=torch.float64)
    return torch.maximum(i / n - F, F - (i - 1) / n).max(-1).values


def weight_stats(W: torch.Tensor) -> dict:
    w = W.double().flatten()
    mu, sd = w.mean(), w.std()
    z = (w - mu) / sd
    ks_rows = ks_normal(W.double())
    return {"ks": float(ks_normal(w)), "ks_row_median": float(ks_rows.median()), "ks_row_max": float(ks_rows.max()),
            "excess_kurtosis": float((z ** 4).mean() - 3), "frac_gt_6sigma": float((z.abs() > 6).double().mean()),
            "shape": list(W.shape)}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/opt-125m")
    args = ap.parse_args()
    torch.set_num_threads(4)
    set_seed(SEED)
    dataloader, _ = get_wikitext2(nsamples=NSAMPLES, seed=SEED, model=args.model)
    model = get_opt(args.model).cpu()
    model.config.use_cache = False
    layers = model.model.decoder.layers
    inps = torch.zeros((NSAMPLES, model.seqlen, model.config.hidden_size), dtype=next(model.parameters()).dtype)
    cache = {"i": 0, "mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kw):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["mask"] = kw["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader[:NSAMPLES]:
        try:
            model(batch[0])
        except ValueError:
            pass
    layers[0] = layers[0].module
    mask = cache["mask"]

    per_layer, int8_layers = [], []  # int8_layers: (bloque, tipo, cnt6/n por coordenada, hist)
    for li, blk in enumerate(layers):
        subset = find_sublayers(blk)
        accs = {}
        for name, lin in subset.items():
            key = INPUT_OF[name]
            if key not in accs:
                accs[key] = Acc(lin.in_features)
        hooked = {}
        for name, lin in subset.items():
            key = INPUT_OF[name]
            if key not in hooked:  # q/k/v comparten entrada: se registra una vez
                hooked[key] = lin.register_forward_hook(
                    (lambda k: lambda _, i, o: accs[k].add(i[0].data))(key))
        for j in range(NSAMPLES):
            inps[j] = blk(inps[j].unsqueeze(0), attention_mask=mask)[0]
        for h in hooked.values():
            h.remove()
        hs = {k: h_stats(a.H) for k, a in accs.items()}
        for name, lin in subset.items():
            a = accs[INPUT_OF[name]]
            cm = a.cmax
            per_layer.append({"layer": li, "matrix": name, "input": INPUT_OF[name],
                              "act_absmax": float(cm.max()), "act_kurtosis": a.kurtosis(),
                              "act_max_over_median_colmax": float(cm.max() / cm.median()),
                              **{f"H_{k}": v for k, v in hs[INPUT_OF[name]].items()},
                              **{f"W_{k}": v for k, v in weight_stats(lin.weight.data).items()}})
            print(li, name, {k: per_layer[-1][k] for k in ("act_absmax", "act_kurtosis", "H_eig_top1", "W_ks")},
                  flush=True)
        for k in ("attn_in", "ffn_in"):
            int8_layers.append((li, k, accs[k].cnt6 / accs[k].n, accs[k].hist, accs[k].cmax))
        del accs

    # A1: criterio LLM.int8()
    L = len(int8_layers)
    frac = torch.stack([f for _, _, f, _, _ in int8_layers])  # [capas, d]
    qual = frac >= 0.06
    outl = (qual.double().mean(0) >= 0.25).nonzero().flatten().tolist()
    coords = []
    for j in outl:
        h = sum(hh[j] for _, _, _, hh, _ in int8_layers)
        c = h.cumsum(0) / h.sum()
        q = [float(EDGES[int((c >= p).nonzero()[0])]) for p in (0.25, 0.5, 0.75)]
        coords.append({"coord": j, "pct_layers": float(qual[:, j].double().mean() * 100),
                       "pct_positions_mean_in_qualifying_layers": float(frac[qual[:, j], j].mean() * 100),
                       "magnitude_quartiles_ge6": q, "magnitude_max": max(float(cm[j]) for *_, cm in int8_layers)})
    int8 = {"where": "entradas de atención (q/k/v, salida de self_attn_layer_norm) y de FFN (fc1, salida de "
                     "final_layer_norm); 2 capas por bloque",
            "n_layers": L, "n_outlier_coords": len(outl), "coords": coords,
            "pct_layers_with_any_qualifying_coord": float(qual.any(1).double().mean() * 100),
            "note": "cuartiles de |x| sobre las posiciones con |x|>=6, histograma log (241 bordes de 6 a 1e4)"}

    def agg(rows, key):
        v = torch.tensor([r[key] for r in rows], dtype=torch.float64)
        return {"median": float(v.median()), "min": float(v.min()), "max": float(v.max())}

    keys = [k for k in per_layer[0] if k.startswith(("act_", "H_", "W_")) and isinstance(per_layer[0][k], float)]
    summary = {m: {k: agg([r for r in per_layer if r["matrix"] == m], k) for k in keys} for m in INPUT_OF}
    tag = args.model.split("/")[-1].replace("-", "")
    out = RESULTS / ("outlier_stats.json" if tag == "opt125m" else f"outlier_stats_{tag}.json")
    out.write_text(json.dumps({"model": args.model, "nsamples": NSAMPLES, "seed": SEED, "dtype": "float16 (CPU)",
                               "int8_criterion": int8, "summary_by_matrix_type": summary, "per_layer": per_layer,
                               "git": GIT}, indent=2))
    print(json.dumps(int8, indent=1)[:3000], flush=True)


if __name__ == "__main__":
    main()
