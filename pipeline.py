#!/usr/bin/env python3
"""
Pipeline central de compresión y evaluación:
Contiene la lógica de ejecución capa por capa sobre OPT, recolección de Hessianas,
aplicación de los modos de cuantización/descomposición de engine.py y cómputo de PPL.
"""
from __future__ import annotations
import math
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import statistics
import copy

import torch
import torch.nn as nn
from transformers import OPTForCausalLM

CURRENT_DIR = Path(__file__).resolve().parent
REPO_DIR = CURRENT_DIR / "vendor" / "gptq"
COMMON_DIR = CURRENT_DIR / "common"

if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from data_loader import get_wikitext2, set_seed
from engine import BipartiteQuantEngine, compute_effective_bits_bipartite, index_entropy
from tube import tube_compress
from srr import srr_compress

DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def find_sublayers(module: nn.Module, layers=(nn.Linear,)) -> Dict[str, nn.Linear]:
    res = {}
    for name, layer in module.named_modules():
        if isinstance(layer, layers):
            res[name] = layer
    return res


def get_opt(model_id: str = "facebook/opt-125m"):
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip  # type: ignore
    torch.nn.init.uniform_ = skip  # type: ignore
    torch.nn.init.normal_ = skip  # type: ignore
    model = OPTForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
    model.seqlen = model.config.max_position_embeddings  # type: ignore
    return model


@torch.no_grad()
def evaluate_ppl(model, testenc, dev) -> float:
    print("Evaluando perplejidad en WikiText-2...")
    model.eval()
    testenc = testenc.input_ids.to(dev)
    nsamples = testenc.numel() // model.seqlen
    model = model.to(dev)

    nlls = []
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    return float(ppl.item())


@torch.no_grad()
def run_compression_mode(
    model,
    dataloader,
    dev,
    mode: str,
    rho: Optional[float] = None,
    kappa: Optional[float] = None,
    r: int = 64,
    sparsity: float = 0.05,
    u_bits: int = 4,
    v_bits: int = 4,
    s_bits: int = 8,
    wbits: int = 3,
    groupsize: int = 64,
    iterations: int = 30,
    nsamples: int = 128,
    l_bits: int = 16,
    scale_dim_l: str = "col",
    l_quant: str = "minmax",
    s_struct: str = "unstructured",
    s_val_bits: int = 16,
    s_index: str = "csr",
    s_cols: int = 0,
    alloc_candidates: str = "Q+S",
    per_column_uv: bool = False,
    use_qslr_quantizer: bool = True,
    bit_map: Optional[Dict[str, int]] = None,
    alpha_min: Optional[float] = None,
    shift_gamma: float = 0.0,
    option_map: Optional[Dict[str, Dict[str, Any]]] = None,
    options: Optional[List[Dict[str, Any]]] = None,
    quantizer: str = "gptq"
) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f" Ejecutando modo: {mode}")
    print(f" rho={rho}, kappa={kappa}, r={r}, sp={sparsity}, u_bits={u_bits}, s_bits={s_bits}, wbits={wbits}, l_bits={l_bits}, groupsize={groupsize}")
    print(f"=======================================================")
    
    # Resolver 0.50 y 0.30 solo para OATS, QSLR y 3BASiL si llegan como None
    rho_eff = 0.50 if rho is None else rho
    kappa_eff = 0.30 if kappa is None else kappa
    
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
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
    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    fp_inps = inps.clone() if shift_gamma else None
    fp_outs = torch.zeros_like(inps) if shift_gamma else None
    attention_mask = cache["attention_mask"]

    total_bits_weighted = 0.0
    total_bits_bitmap_weighted = 0.0
    total_bits_csr_weighted = 0.0
    total_bits_paper_weighted = 0.0
    total_err_weighted = 0.0
    total_params = 0

    layer_stats = []

    for i in range(len(layers)):
        layer = layers[i].to(dev)
        subset = find_sublayers(layer)
        engines = {name: BipartiteQuantEngine(subset[name]) for name in subset}

        def add_batch(name):
            def tmp(_, inp, out):
                engines[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = [subset[name].register_forward_hook(add_batch(name)) for name in subset]
        if shift_gamma:
            # Flujo original en paralelo: x_fp entra a la copia sin cuantizar del bloque
            fp_layer = copy.deepcopy(layer)
            fp_subset = find_sublayers(fp_layer)
            x_fp: Dict[str, torch.Tensor] = {}

            def keep_fp(name):
                def tmp(_, inp, out):
                    x_fp[name] = inp[0].data
                return tmp

            def cross(name):
                def tmp(_, inp, out):
                    engines[name].add_cross(inp[0].data, x_fp[name])
                return tmp

            handles += [fp_subset[name].register_forward_hook(keep_fp(name)) for name in fp_subset]
            handles += [subset[name].register_forward_hook(cross(name)) for name in subset]
        for j in range(nsamples):
            if shift_gamma:
                fp_outs[j] = fp_layer(fp_inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()
        if shift_gamma:
            for name in subset:
                engines[name].apply_gradient_shift(shift_gamma)

        for name in subset:
            eng = engines[name]
            if mode == "oats_fp16_wanda":
                res = eng.run_oats_fp16(rho=rho_eff, kappa=kappa_eff, iterations=iterations, use_cholesky=False)
            elif mode == "oats_fp16_cholesky":
                res = eng.run_oats_fp16(rho=rho_eff, kappa=kappa_eff, iterations=iterations, use_cholesky=True)
            elif mode == "oats_quantized_qslr":
                res = eng.run_oats_quantized_sequential(
                    rho=rho_eff, kappa=kappa_eff, u_bits=u_bits, v_bits=v_bits, s_bits=s_bits,
                    groupsize=groupsize, iterations=iterations, per_column_uv=per_column_uv
                )
            elif mode == "bipartite_bcd_quantized":
                res = eng.run_bipartite_bcd_quantized(
                    r=r, sparsity=sparsity, rho=rho, kappa=kappa,
                    u_bits=u_bits, v_bits=v_bits, s_bits=s_bits,
                    groupsize=groupsize, cycles=max(2, iterations // 5),
                    use_qslr_quantizer=use_qslr_quantizer, per_column_uv=per_column_uv
                )
            elif mode == "3basil_fp16":
                res = eng.run_3basil_fp16(
                    rho_budget=rho_eff, kappa=kappa_eff, iterations=iterations
                )
            elif mode == "gptq":
                res = eng.run_gptq(
                    wbits=wbits,
                    groupsize=groupsize
                )
            elif mode == "triad_quant_only":
                res = eng.run_triad_quant_only(
                    wbits=wbits,
                    groupsize=groupsize,
                    alpha_min=0.70 if alpha_min is None else alpha_min
                )
            elif mode == "q_bitmap":
                # Q sola con b por matriz; clave "<bloque>.<sublayer>"
                if quantizer.startswith("admm"):
                    res = eng.run_admmq(wbits=bit_map[f"{i}.{name}"], groupsize=groupsize,
                                        scale="hessian" if quantizer == "admm_h" else "admm")
                else:
                    res = eng.run_triad_quant_only(
                        wbits=bit_map[f"{i}.{name}"],
                        groupsize=groupsize
                    )
            elif mode == "tube":
                # bit_map opcional: b por matriz; option_map: (b, r, sparsity) por matriz.
                # Clave "<bloque>.<sublayer>"
                key = f"{i}.{name}"
                opt_l = option_map[key] if option_map else {}
                res = tube_compress(
                    eng, wbits=opt_l.get("wbits", bit_map[key] if bit_map else wbits),
                    rank=opt_l.get("r", r), sparsity=opt_l.get("sparsity", sparsity),
                    groupsize=groupsize,
                    l_bits=l_bits, s_val_bits=s_val_bits, s_index=s_index,
                    scale_dim_l=scale_dim_l,
                    alpha_min=0.40 if alpha_min is None else alpha_min,
                    quantizer=quantizer
                )
            elif mode == "srr":
                # Rival SRR (arXiv 2602.02001): L1 antes de Q y L2 sobre el error; b por matriz
                res = srr_compress(eng, wbits=bit_map[f"{i}.{name}"] if bit_map else wbits, rank=r,
                                   groupsize=groupsize, l_bits=l_bits, scale_dim_l=scale_dim_l,
                                   alpha_min=0.70 if alpha_min is None else alpha_min, quantizer=quantizer)
            elif mode == "tube_options":
                # Mide error y bits del tubo para cada opción; aplica la opción "base"
                key = f"{i}.{name}"
                b0 = bit_map[key] if bit_map else wbits
                measured, res = [], None
                for opt_l in options:
                    wb = b0 + opt_l["db"]
                    if wb < 2 or wb > 8:
                        continue
                    out_l = tube_compress(
                        eng, wbits=wb, rank=opt_l["r"], sparsity=opt_l["sparsity"],
                        groupsize=groupsize, l_bits=l_bits, s_val_bits=s_val_bits,
                        s_index=s_index, scale_dim_l=scale_dim_l,
                        alpha_min=0.70 if alpha_min is None else alpha_min
                    )
                    measured.append({"wbits": wb, "r": opt_l["r"], "sparsity": opt_l["sparsity"],
                                     "err": out_l["functional_error"], "bits": out_l["bits"]})
                    if opt_l.get("base"):
                        res = out_l
                res["options"] = measured
            elif mode == "qls_triad":
                res = eng.run_qls_triad(
                    wbits=wbits,
                    rank=r,
                    sparsity=sparsity,
                    groupsize=groupsize,
                    cycles=2,
                    l_bits=l_bits,
                    scale_dim_l=scale_dim_l,
                    l_quant=l_quant,
                    s_struct=s_struct,
                    s_val_bits=s_val_bits,
                    s_index=s_index,
                    s_cols=s_cols
                )
            elif mode == "triad_alloc":
                res = eng.run_triad_alloc(
                    wbits=wbits,
                    groupsize=groupsize,
                    candidates=alloc_candidates,
                    rank=r,
                    sparsity=sparsity,
                    cycles=2,
                    l_bits=l_bits,
                    scale_dim_l=scale_dim_l,
                    l_quant=l_quant,
                    s_struct=s_struct,
                    s_val_bits=s_val_bits,
                    s_index=s_index,
                    s_cols=s_cols
                )
            else:
                raise ValueError(f"Modo desconocido: {mode}")

            subset[name].weight.data = res["W_approx"].half()
            _numel = subset[name].weight.numel()
            paper_b = res.get("effective_bits_paper") if res.get("effective_bits_paper") is not None else res.get("bits_paper")
            stat_entry = {
                "layer_idx": i,
                "layer_name": name,
                "numel": _numel,
                "functional_error": res["functional_error"],
                "effective_bits": res["effective_bits"],
                "effective_bits_bitmap": res.get("effective_bits_bitmap"),
                "effective_bits_csr": res.get("effective_bits_csr"),
                "effective_bits_paper": paper_b,
            }
            if "options" in res:
                stat_entry["options"] = res["options"]
            if res.get("Q_idx") is not None:
                stat_entry["entropy"] = index_entropy(res["Q_idx"], res["Q_zero"], res["groupsize"])
                stat_entry["wbits"] = res["wbits"]
            if "diagnostics" in res:
                stat_entry.update(res["diagnostics"])
            layer_stats.append(stat_entry)

            total_bits_weighted += res["effective_bits"] * _numel
            if res.get("effective_bits_bitmap") is not None:
                total_bits_bitmap_weighted += res["effective_bits_bitmap"] * _numel
            if res.get("effective_bits_csr") is not None:
                total_bits_csr_weighted += res["effective_bits_csr"] * _numel
            if paper_b is not None:
                total_bits_paper_weighted += paper_b * _numel
            total_err_weighted += res["functional_error"] * _numel
            total_params += _numel

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps
        if shift_gamma:
            del fp_layer
            fp_inps, fp_outs = fp_outs, fp_inps

    model.config.use_cache = use_cache
    avg_bits = float(total_bits_weighted / total_params)
    avg_bits_bmp = float(total_bits_bitmap_weighted / total_params) if total_bits_bitmap_weighted > 0 else None
    avg_bits_csr = float(total_bits_csr_weighted / total_params) if total_bits_csr_weighted > 0 else None
    avg_bits_paper = float(total_bits_paper_weighted / total_params) if total_bits_paper_weighted > 0 else None
    avg_err = float(total_err_weighted / total_params)

    last_diag = next(
        (s for s in reversed(layer_stats) if s.get("loss_decomp_oats") is not None),
        {}
    )

    diag_entries = [s for s in layer_stats if s.get("loss_decomp_oats") is not None]
    if diag_entries:
        r1_list = []
        r2_list = []
        rel_errs = []
        for s in diag_entries:
            ql = s.get("loss_quant_L", 0.0) or 0.0
            qs = s.get("loss_quant_S", 0.0) or 0.0
            denom_ls = ql + qs
            decomp = s.get("loss_decomp_oats", 0.0) or 0.0
            cross = s.get("cross_LS", 0.0) or 0.0
            closure = s.get("closure_rel_err", 0.0) or 0.0
            rel_errs.append(closure)
            if denom_ls > 1e-12:
                r1_list.append(cross / denom_ls)
            if decomp > 1e-12:
                r2_list.append(denom_ls / decomp)
        diag_summary = {
            "cross_over_quant_median": statistics.median(r1_list) if r1_list else None,
            "cross_over_quant_max": max(r1_list, key=abs) if r1_list else None,
            "quant_over_decomp_median": statistics.median(r2_list) if r2_list else None,
            "closure_rel_err_max": max(rel_errs) if rel_errs else None,
        }
    else:
        diag_summary = {}

    alloc_summary = None
    alloc_entries = [s for s in layer_stats if s.get("alloc_choice") is not None]
    if alloc_entries:
        P = sum(s["numel"] for s in alloc_entries)
        counts: Dict[str, int] = {}
        by_type: Dict[str, Dict[str, int]] = {}
        for s in alloc_entries:
            c = s["alloc_choice"]
            counts[c] = counts.get(c, 0) + 1
            t = by_type.setdefault(s["layer_name"], {})
            t[c] = t.get(c, 0) + 1
        alloc_summary = {
            "choices": counts,
            "choices_by_type": by_type,
            "pred_delta_bits": sum(s["alloc_delta_bits"] * s["numel"] for s in alloc_entries) / P,
        }
        print(f"[ALLOC] elecciones: {counts} | Δbits predicho (ponderado): {alloc_summary['pred_delta_bits']:+.4f}")

    return {
        "mode": mode,
        "alloc_summary": alloc_summary,
        "avg_effective_bits": avg_bits,
        "avg_effective_bits_bitmap": avg_bits_bmp,
        "avg_effective_bits_csr": avg_bits_csr,
        "avg_effective_bits_paper": avg_bits_paper,
        "avg_functional_error": avg_err,
        "diagnostics": last_diag,
        "diagnostics_summary": diag_summary,
        "layer_stats": layer_stats
    }