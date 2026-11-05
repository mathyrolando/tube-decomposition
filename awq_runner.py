#!/usr/bin/env python3
"""
Módulo runner para AWQ (Activation-aware Weight Quantization, MLSys 2024).
Importación perezosa de la implementación oficial de MIT HAN Lab (external/llm-awq).
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn

CURRENT_DIR = Path(__file__).resolve().parent
REPO_AWQ = CURRENT_DIR / "external" / "llm-awq"
COMMON_DIR = CURRENT_DIR / "common"

if str(REPO_AWQ) not in sys.path:
    sys.path.insert(0, str(REPO_AWQ))
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from data_loader import get_wikitext2, set_seed

DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def apply_awq_official(
    model: nn.Module,
    wbits: int = 4,
    groupsize: int = 64,
    nsamples: int = 128,
    seqlen: int = 512,
    dev=DEV,
    **kwargs
) -> Dict[str, Any]:
    """
    Aplica AWQ oficial al modelo OPT completo y lo cuantiza (simulado).
    Devuelve diccionario con las métricas de bits efectivos equivalentes a pipeline.py.
    """
    try:
        from awq.quantize.pre_quant import run_awq
        from awq.quantize.quantizer import pseudo_quantize_model_weight
    except ImportError as e:
        raise ImportError(
            f"No se pudo importar 'awq'. Verifica que exista '{REPO_AWQ}': {e}"
        )

    print(f"\n=======================================================")
    print(f" Ejecutando AWQ 1-a-1 Oficial: wbits={wbits}, groupsize={groupsize}")
    print(f"=======================================================")

    dataloader, _ = get_wikitext2(nsamples=nsamples, seed=42, seqlen=seqlen, model="facebook/opt-125m")
    calib_samples = [inp for inp, _ in dataloader]

    q_config = {
        "zero_point": True,
        "q_group_size": groupsize
    }

    # 1. Búsqueda y aplicación de Auto-Scale y Auto-Clip según el paper oficial
    run_awq(
        model=model,
        enc=None,
        w_bit=wbits,
        q_config=q_config,
        n_samples=nsamples,
        seqlen=seqlen,
        auto_scale=True,
        mse_range=True,
        calib_data=calib_samples
    )

    # 2. Pseudo-cuantización uniforme simulada
    pseudo_quantize_model_weight(model, w_bit=wbits, q_config=q_config)

    # Overhead de escala FP16 + zero-point FP16 por grupo (32 bits / g)
    overhead = 32.0 / groupsize if groupsize > 0 else 0.0
    eff_bits = float(wbits + overhead)

    return {
        "mode": "awq",
        "avg_effective_bits": eff_bits,
        "avg_effective_bits_csr": None,
        "avg_effective_bits_bitmap": None,
        "avg_effective_bits_paper": None,
        "avg_functional_error": 0.0,
        "diagnostics": {},
        "diagnostics_summary": None,
    }
