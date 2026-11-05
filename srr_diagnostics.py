#!/usr/bin/env python3
"""
Diagnósticos de SRR (srr.py) en OPT-125M, solo CPU (no compite con corridas en la GPU).

1. k* por matriz para r = 4, 8, 16: cuántas direcciones preserva SRR antes de cuantizar.
   k* depende solo de W, H y r (no del presupuesto de bits). H es la del modelo sin comprimir
   (una sola pasada de calibración), no la del modelo cuantizado en cascada de la corrida real.
2. Bloque 3, self_attn.k_proj, Q a 3 bits g256 (ADMM-H), r=16: error funcional de SRR con L en
   16, 8, 4 y 3 bits, y de Q sola. Muestra que L después de Q necesita bits altos.

Uso (desde lab/):
    HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= python srr_diagnostics.py
"""
from __future__ import annotations
import json
from pathlib import Path

import torch
import torch.nn as nn

from pipeline import get_opt, find_sublayers
from engine import BipartiteQuantEngine
from data_loader import get_wikitext2, set_seed
from srr import _tail_ratio, srr_compress
from provenance import GIT

OUT = Path(__file__).resolve().parent / "results" / "srr_diagnostics.json"
NSAMPLES, SEED, RANKS = 128, 42, (4, 8, 16)
PROBE = (3, "self_attn.k_proj")


def kstar(eng: BipartiteQuantEngine, rank: int, seed: int = 0) -> int:
    """Igual que en srr_compress (sonda E uniforme de n x n con semilla 0, la de la corrida)."""
    W = eng.W_orig.float()
    n = eng.columns
    L_H, _ = eng.get_cholesky()
    gen = torch.Generator(device=W.device).manual_seed(seed)
    E = torch.rand((n, n), generator=gen, device=W.device) - 0.5
    rw = _tail_ratio(torch.linalg.svdvals(W @ L_H))
    re = _tail_ratio(torch.linalg.svdvals(E @ L_H))
    return int(torch.stack([rw[k] * re[rank - k] for k in range(rank + 1)]).argmin())


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    set_seed(SEED)
    dataloader, _ = get_wikitext2(nsamples=NSAMPLES, seed=SEED, model="facebook/opt-125m")
    model = get_opt("facebook/opt-125m").cpu()
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

    kstars, probe = [], {}
    for li, blk in enumerate(layers):
        subset = find_sublayers(blk)
        engines = {name: BipartiteQuantEngine(subset[name]) for name in subset}
        handles = [subset[n].register_forward_hook(
            (lambda name: lambda _, i, o: engines[name].add_batch(i[0].data, o.data))(n)) for n in subset]
        for j in range(NSAMPLES):
            blk(inps[j].unsqueeze(0), attention_mask=mask)
        for h in handles:
            h.remove()
        for name, eng in engines.items():
            ks = {f"r{r}": kstar(eng, r) for r in RANKS}
            kstars.append({"layer": li, "matrix": name, **ks})
            print(li, name, ks, flush=True)
            if (li, name) == PROBE:
                q = eng.run_admmq(wbits=3, groupsize=256, alpha_min=0.70, scale="hessian")
                probe["Q sola"] = eng.evaluate_functional_error(q["W_approx"].float())
                for lb in (16, 8, 4, 3):
                    res = srr_compress(eng, wbits=3, rank=16, groupsize=256, l_bits=lb, alpha_min=0.70)
                    probe[f"SRR r16 L{lb}"] = res["functional_error"]
                print("probe", probe, flush=True)
        for j in range(NSAMPLES):
            inps[j] = blk(inps[j].unsqueeze(0), attention_mask=mask)[0]
        del engines

    summary = {}
    for r in RANKS:
        v = [x[f"r{r}"] for x in kstars]
        summary[f"r{r}"] = {"mean": sum(v) / len(v), "min": min(v), "max": max(v),
                            "zeros": sum(k == 0 for k in v), "n": len(v)}
    OUT.write_text(json.dumps({"model": "facebook/opt-125m", "nsamples": NSAMPLES, "seed": SEED,
                               "H": "modelo sin comprimir", "kstar_summary": summary, "kstar": kstars,
                               "probe": {"layer": PROBE[0], "matrix": PROBE[1], "wbits": 3, "groupsize": 256,
                                         "quantizer": "admm_h", "rank": 16, "functional_error": probe},
                               "git": GIT}, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
