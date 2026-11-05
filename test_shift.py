#!/usr/bin/env python3
"""
Q y tubo con corrimiento por gradiente W~ = W + gamma W C H^-1 (forma GPTAQ capa a capa).
gamma se elige con el PPL del split de validación de WikiText-2, nunca con el de test.

Uso:
    python test_shift.py --seed 42
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from pipeline import DEV, evaluate_ppl, get_opt, run_compression_mode
from data_loader import TokenizerWrapper, get_c4, get_wikitext2, set_seed
from provenance import GIT

RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUT_PATH = RESULTS_DIR / "shift_comparison.json"
TUBE = dict(mode="tube", wbits=3, r=16, sparsity=0.0025, groupsize=256, l_bits=3, s_val_bits=4,
            s_index="rowcount", alpha_min=0.70)
GAMMAS = (0.0, 0.25, 0.5)


def get_wikitext2_val(model: str = "facebook/opt-125m") -> TokenizerWrapper:
    data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    tok = AutoTokenizer.from_pretrained(model, use_fast=False)
    return TokenizerWrapper(tok("\n\n".join(data["text"]), return_tensors="pt").input_ids)


@torch.no_grad()
def kl_to_reference(model, ref, testenc: TokenizerWrapper, dev) -> float:
    """KL(p_FP16 || p_comprimido) media por token sobre WikiText-2 test, en nats."""
    ids = testenc.input_ids
    n = ids.numel() // model.seqlen
    model, ref = model.to(dev).eval(), ref.to(dev).eval()
    total, count = 0.0, 0
    for i in range(n):
        batch = ids[:, i * model.seqlen:(i + 1) * model.seqlen].to(dev)
        lp = torch.log_softmax(model(batch).logits.float(), -1)
        lr = torch.log_softmax(ref(batch).logits.float(), -1)
        total += float((lr.exp() * (lr - lp)).sum())
        count += batch.numel()
    return total / count


def main():
    ap = argparse.ArgumentParser(description="Corrimiento por gradiente en Q y tubo")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--model", default="facebook/opt-125m")
    ap.add_argument("--extra", action="store_true",
                    help="además mide PPL en C4 y KL respecto de FP16; guarda con sufijo ' +c4kl'")
    args = ap.parse_args()
    tag = args.model.split("/")[-1].replace("-", "")
    alloc = RESULTS_DIR / ("rd_alloc.json" if tag == "opt125m" else f"rd_alloc_{tag}.json")
    maps = {x["budget"]: x["bit_map"] for x in json.loads(alloc.read_text())
            if x["variant"] == "kfac"}
    jobs: Dict[str, Dict[str, Any]] = {}
    for budget in (2.5, 3.0, 3.5, 4.0):
        for gamma in GAMMAS:
            jobs[f"Q kfac {budget} g256 γ{gamma}"] = dict(mode="q_bitmap", groupsize=256,
                                                          bit_map=maps[budget], shift_gamma=gamma)
            jobs[f"tubo kfac {budget} g256 γ{gamma}"] = dict(TUBE, bit_map=maps[budget], shift_gamma=gamma)
    for budget in (3.5, 4.0, 4.5):
        jobs[f"Q kfac {budget} g1024 γ0.0"] = dict(mode="q_bitmap", groupsize=1024, bit_map=maps[budget])
    for budget in (3.5, 4.0):
        jobs[f"Q admm kfac {budget} g256 γ0.0"] = dict(mode="q_bitmap", groupsize=256, bit_map=maps[budget],
                                                       quantizer="admm")
        jobs[f"tubo admm kfac {budget} g256 γ0.0"] = dict(TUBE, bit_map=maps[budget], quantizer="admm")
        jobs[f"Q admmH kfac {budget} g256 γ0.0"] = dict(mode="q_bitmap", groupsize=256, bit_map=maps[budget],
                                                        quantizer="admm_h")
        jobs[f"tubo admmH kfac {budget} g256 γ0.0"] = dict(TUBE, bit_map=maps[budget], quantizer="admm_h")
    for budget in (2.5, 3.0, 4.5):
        jobs[f"tubo admmH kfac {budget} g256 γ0.0"] = dict(TUBE, bit_map=maps[budget], quantizer="admm_h")
    for budget in (2.5, 3.0):
        jobs[f"Q admmH kfac {budget} g256 γ0.0"] = dict(mode="q_bitmap", groupsize=256, bit_map=maps[budget],
                                                        quantizer="admm_h")
    jobs["Q admmH kfac 4.5 g256 γ0.0"] = dict(mode="q_bitmap", groupsize=256, bit_map=maps[4.5],
                                               quantizer="admm_h")
    # Q sola al costo del tubo: g=128 cuesta 0.125 bits más que g=256 (queda 0.011 debajo del tubo).
    for budget in (2.5, 3.0, 3.5, 4.0, 4.5):
        jobs[f"Q admmH kfac {budget} g128 γ0.0"] = dict(mode="q_bitmap", groupsize=128, bit_map=maps[budget],
                                                        quantizer="admm_h")
        jobs[f"Q kfac {budget} g128 γ0.0"] = dict(mode="q_bitmap", groupsize=128, bit_map=maps[budget])
    # Entropía de índices con ADMM-Q (run_admmq ya devuelve Q_idx); nombres nuevos para no saltear.
    for budget in (3.0, 3.5):
        for g in (128, 256):
            jobs[f"Q admmH kfac {budget} g{g} ent"] = dict(mode="q_bitmap", groupsize=g, bit_map=maps[budget],
                                                           quantizer="admm_h")
        jobs[f"tubo admmH kfac {budget} g256 ent"] = dict(TUBE, bit_map=maps[budget], quantizer="admm_h")
    for budget in (3.5, 4.0):
        jobs[f"Q admmH kfac {budget} g1024 ent"] = dict(mode="q_bitmap", groupsize=1024, bit_map=maps[budget],
                                                        quantizer="admm_h")
    for budget in (3.0, 3.5, 4.0):
        jobs[f"tubo admmH kfac {budget} g1024 ent"] = dict(TUBE, groupsize=1024, bit_map=maps[budget],
                                                           quantizer="admm_h")
    # Revalidación con ppl_val de elecciones hechas antes mirando test: r, g del tubo y alpha_min.
    for budget in (3.0, 3.5):
        for r, g, am in ((8, 256, 0.70), (32, 256, 0.70), (16, 128, 0.70), (16, 64, 0.70), (16, 256, 0.40),
                         (16, 256, 0.70)):
            jobs[f"tubo admmH kfac {budget} g{g} r{r} a{am} val"] = dict(TUBE, r=r, groupsize=g, alpha_min=am,
                                                                        bit_map=maps[budget], quantizer="admm_h")
    # Ruido de corrida: misma semilla, mismo código; el nombre distinto evita que se saltee.
    for budget in (3.0, 3.5):
        for k in (1, 2):
            jobs[f"tubo admmH kfac {budget} g256 rep{k}"] = dict(TUBE, bit_map=maps[budget], quantizer="admm_h")
    # Rival SRR (código oficial, L en 16 bits) con el mismo cuantizador final que el tubo.
    # r=16 en 16 bits es su configuración (~0.5 bits de L); r=4 en 16 bits y r=8 en 8 bits cuestan
    # lo mismo que L+S del tubo. L en 8 bits se cuantiza con el mismo cuantizador de L que el tubo.
    for budget in (2.5, 3.0, 3.5, 4.0, 4.5):
        for r, lb in ((4, 16), (8, 8), (16, 16)):
            jobs[f"srr admmH kfac {budget} g256 r{r} L{lb}"] = dict(mode="srr", r=r, groupsize=256, l_bits=lb,
                                                                  alpha_min=0.70, bit_map=maps[budget],
                                                                  quantizer="admm_h")
    # r=8 elegido por validación (run_day1.sh: margen medio -1.61 contra -1.07 de r=16, 3 semillas).
    for budget in (2.5, 3.0, 3.5, 4.0, 4.5):
        jobs[f"tubo r8 admmH kfac {budget} g256"] = dict(TUBE, r=8, bit_map=maps[budget], quantizer="admm_h")
    jobs["fp16"] = None  # referencia sin comprimir
    if args.only:
        jobs = {k: v for k, v in jobs.items() if k in args.only}
    if args.extra:
        jobs = {f"{k} +c4kl": v for k, v in jobs.items()}

    results = json.loads(OUT_PATH.read_text()) if OUT_PATH.exists() else []
    done = {(r["config"], r["seed"], r.get("model", "facebook/opt-125m")) for r in results}
    set_seed(args.seed)
    dataloader, testenc = get_wikitext2(nsamples=128, seed=args.seed, model=args.model)
    valenc = get_wikitext2_val(args.model)
    c4enc = get_c4(nsamples=1, seed=0, model=args.model)[1] if args.extra else None
    for name, cfg in jobs.items():
        if (name, args.seed, args.model) in done:
            continue
        set_seed(args.seed)
        model = get_opt(args.model)
        if cfg is None:
            ppl_val, ppl = evaluate_ppl(model, valenc, DEV), evaluate_ppl(model, testenc, DEV)
            rec = {"config": name, "seed": args.seed, "model": args.model, "ppl_val": ppl_val,
                   "ppl": ppl, "bits": 16.0, "git": GIT}
            if args.extra:
                rec["ppl_c4"], rec["kl"] = evaluate_ppl(model, c4enc, DEV), 0.0
            results.append(rec)
            print(f"[seed {args.seed}] {name}: val {ppl_val:.3f} test {ppl:.3f} "
                  f"c4 {rec.get('ppl_c4', float('nan')):.3f}", flush=True)
            OUT_PATH.write_text(json.dumps(results, indent=2))
            del model
            torch.cuda.empty_cache()
            continue
        out = run_compression_mode(model, dataloader, DEV, nsamples=128, **cfg)
        ppl_val = evaluate_ppl(model, valenc, DEV)
        ppl = evaluate_ppl(model, testenc, DEV)
        extra = {}
        if args.extra:
            extra["ppl_c4"] = evaluate_ppl(model, c4enc, DEV)
            ref = get_opt(args.model)
            extra["kl"] = kl_to_reference(model, ref, testenc, DEV)
            del ref
            torch.cuda.empty_cache()
        stats = out["layer_stats"]
        total = sum(s["numel"] for s in stats)
        mean = lambda f: sum(f(s) * s["numel"] for s in stats) / total
        rec = {"config": name, "seed": args.seed, "model": args.model, "gamma": cfg.get("shift_gamma", 0.0),
               "ppl_val": ppl_val, "ppl": ppl, "bits": out["avg_effective_bits"],
               "wbits": mean(lambda s: s.get("wbits", 0)), **extra, "git": GIT}
        if all("entropy" in s for s in stats):
            rec["H_zero"] = mean(lambda s: s["entropy"]["zero"])
            rec["bits_zero"] = rec["bits"] - rec["wbits"] + rec["H_zero"]
        print(f"[seed {args.seed}] {name}: bits {rec['bits']:.3f} ent {rec.get('bits_zero', float('nan')):.3f} "
              f"val {ppl_val:.3f} test {ppl:.3f} c4 {rec.get('ppl_c4', float('nan')):.3f} "
              f"kl {rec.get('kl', float('nan')):.4f}", flush=True)
        results.append(rec)
        OUT_PATH.write_text(json.dumps(results, indent=2))
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
