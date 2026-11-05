#!/usr/bin/env python3
"""
Orquestador Unificado de Barridos de Pareto:
Fuente de verdad única para todas las familias evaluadas en la tesis:
- gptq, gptq_g64, triad_quant_only, awq
- oats_fp16, basil_fp16
- qslr_quantized, qslr_quantized_s4
- bipartite_bcd, bipartite_bcd_s4
- canonical_qls, canonical_qls_l4, canonical_qls_l3_hq, record_ptq

Uso:
    uv run sweeps.py                    # Corre todas las familias pendientes
    uv run sweeps.py bipartite_bcd_s4   # Corre solo la familia especificada
    uv run sweeps.py --dry-run          # Muestra qué configuraciones se correrían sin ejecutar nada
    uv run sweeps.py --force            # Recalcula forzosamente
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch

CURRENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = CURRENT_DIR / "results"
SWEEP_DATA_PATH = RESULTS_DIR / "pareto_curves_data.json"

from pipeline import get_opt, evaluate_ppl, run_compression_mode, DEV
from data_loader import get_wikitext2, set_seed
from provenance import GIT

RHOS = (0.30, 0.50, 0.70, 0.80, 0.90)
WB = (2, 3, 4, 6, 8)

QSLR = dict(mode="oats_quantized_qslr", kappa=0.30, u_bits=4, v_bits=4, iterations=30)
BCD  = dict(mode="bipartite_bcd_quantized", kappa=0.30, u_bits=4, v_bits=4, iterations=30)
TUBE = dict(mode="tube", groupsize=64, l_bits=3, s_val_bits=4, s_index="rowcount")

FAMILIES: Dict[str, List[Dict[str, Any]]] = {
    "gptq":                   [dict(mode="gptq", wbits=w, groupsize=-1) for w in WB],
    "gptq_g64":               [dict(mode="gptq", wbits=w, groupsize=64) for w in WB],
    "triad_quant_only":       [dict(mode="triad_quant_only", wbits=w, groupsize=64) for w in WB],
    # Frontera real de Q sola: puntos intermedios variando g, sin interpolar.
    "q_only_g_sweep":         [dict(mode="triad_quant_only", wbits=w, groupsize=g)
                               for w, g in ((2, 32), (3, 256), (3, 128), (3, 32), (4, 256), (4, 128))],
    "awq":                    [dict(mode="awq", wbits=w, groupsize=64) for w in WB],
    "oats_fp16":              [dict(mode="oats_fp16_wanda", rho=r, kappa=0.30, iterations=30)
                                for r in (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)],
    "basil_fp16":             [dict(mode="3basil_fp16", rho=r, kappa=0.30, iterations=30)
                                for r in (0.50, 0.60, 0.70, 0.80)],
    "qslr_quantized":         [dict(QSLR, rho=r, s_bits=8) for r in RHOS],
    "qslr_quantized_s4":      [dict(QSLR, rho=r, s_bits=4) for r in RHOS],
    "bipartite_bcd":          [dict(BCD, rho=r, s_bits=8) for r in RHOS],
    "bipartite_bcd_s4":       [dict(BCD, rho=r, s_bits=4) for r in RHOS],
    "canonical_qls":          [dict(mode="qls_triad", wbits=w, r=16, sparsity=0.01, groupsize=64) for w in WB],
    "canonical_qls_l4":       [dict(mode="qls_triad", wbits=w, r=16, sparsity=0.01, groupsize=64,
                                    l_bits=4, scale_dim_l="col") for w in WB],
    # L cuantizado Hessian-aware a 3 bits, escalas por columna (punto de quiebre en test_l_quant.py, bloque 0).
    "canonical_qls_l3_hq":    [dict(mode="qls_triad", wbits=w, r=16, sparsity=0.01, groupsize=64,
                                    l_bits=3, scale_dim_l="col", l_quant="hessian") for w in WB],
    # ---- Validación de punta a punta del asignador (punto 2) ----
    # S en todas las matrices: Q3 + S no estructurada 0.25%, valores 4b, conteo por fila
    "qs_all_s025_4b":         [dict(mode="qls_triad", wbits=w, r=0, sparsity=0.0025, groupsize=64,
                                    s_val_bits=4, s_index="rowcount") for w in (2, 3)],
    # Asignador v0: por matriz elige Q3 sola o Q3 + S (la misma S), según Δbits
    "alloc_qs_s025_4b":       [dict(mode="triad_alloc", wbits=w, sparsity=0.0025, groupsize=64,
                                    s_val_bits=4, s_index="rowcount", alloc_candidates="Q+S") for w in (2, 3)],
    "record_ptq":             [dict(mode="qls_triad", wbits=3, r=20, sparsity=0.012, groupsize=64)],
    # Descomposición por tubo (tube.py): L y S preparan el residuo para Q
    "tube_r16":               [dict(TUBE, wbits=w, r=16, sparsity=0.0025) for w in (2, 3, 4)],
    "tube_r8":                [dict(TUBE, wbits=w, r=8, sparsity=0.0025) for w in (2, 3, 4)],
    # Diagnóstico de b=2: grilla ampliada en Q sola y tubo con grilla estándar
    "tube_b2_diag":           [dict(mode="triad_quant_only", wbits=2, groupsize=64, alpha_min=0.40),
                               dict(TUBE, wbits=2, r=16, sparsity=0.0025, alpha_min=0.70),
                               dict(TUBE, wbits=2, r=0, sparsity=0.0025),
                               dict(TUBE, wbits=2, r=16, sparsity=0.0)],
    "tube_ablation":          [dict(TUBE, wbits=3, r=16, sparsity=0.0), dict(TUBE, wbits=3, r=0, sparsity=0.0025)],
}


def normalize_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in sorted(cfg.items()):
        if isinstance(v, float):
            out[k] = round(v, 4)
        else:
            out[k] = v
    if "groupsize" not in out and out.get("mode") in ("triad_quant_only", "qls_triad", "awq", "triad_alloc"):
        out["groupsize"] = 64
    if out.get("mode") == "triad_alloc":
        out.setdefault("alloc_candidates", "Q+S")
    if out.get("mode") in ("qls_triad", "triad_alloc"):
        out.setdefault("scale_dim_l", "col")
        out.setdefault("l_quant", "minmax")
        out.setdefault("s_struct", "unstructured")
        out.setdefault("s_val_bits", 16)
        out.setdefault("s_index", "csr")
        out.setdefault("s_cols", 0)
    if "per_column_uv" not in out and ("qslr" in out.get("mode", "") or "bcd" in out.get("mode", "")):
        out["per_column_uv"] = False
    return out


def configs_match(c1: Dict[str, Any], c2: Dict[str, Any]) -> bool:
    return normalize_config(c1) == normalize_config(c2)


def entry_matches_config(entry: Dict[str, Any], target_cfg: Dict[str, Any]) -> bool:
    if not isinstance(entry, dict):
        return False
    if "config" in entry and isinstance(entry["config"], dict):
        return configs_match(entry["config"], target_cfg)
    
    # Comparación semántica sobre campos sueltos de entradas existentes
    if target_cfg.get("iterations", 30) != 30:
        return False   # las entradas viejas se calcularon con 30

    if "wbits" in target_cfg and entry.get("wbits") != target_cfg["wbits"]:
        return False
    if "rho" in target_cfg:
        e_rho = entry.get("rho")
        if e_rho is None or abs(float(e_rho) - float(target_cfg["rho"])) > 0.01:
            return False
    if "kappa" in target_cfg:
        e_kap = entry.get("kappa")
        if e_kap is not None and abs(float(e_kap) - float(target_cfg["kappa"])) > 0.01:
            return False
    if "s_bits" in target_cfg:
        e_sb = entry.get("s_bits", 8)  # antiguas entradas sin s_bits usaban default=8
        if e_sb != target_cfg["s_bits"]:
            return False
    if "per_column_uv" in target_cfg and entry.get("per_column_uv", False) != target_cfg["per_column_uv"]:
        return False
    if "r" in target_cfg and entry.get("r") != target_cfg["r"]:
        return False
    if "sparsity" in target_cfg:
        e_sp = entry.get("sparsity") or entry.get("sp")
        if e_sp is None or abs(float(e_sp) - float(target_cfg["sparsity"])) > 0.005:
            return False
    if "l_bits" in target_cfg and entry.get("l_bits") != target_cfg["l_bits"]:
        return False
    if "scale_dim_l" in target_cfg and entry.get("scale_dim_l", "col") != target_cfg["scale_dim_l"]:
        return False
    if "l_quant" in target_cfg and entry.get("l_quant", "minmax") != target_cfg["l_quant"]:
        return False
    for key, default in (("s_struct", "unstructured"), ("s_val_bits", 16),
                         ("s_index", "csr"), ("s_cols", 0)):
        if key in target_cfg and entry.get(key, default) != target_cfg[key]:
            return False
    return True


def make_record(cfg: Dict[str, Any], res: Dict[str, Any], ppl: float) -> Dict[str, Any]:
    norm_cfg = normalize_config(cfg)
    return {
        **norm_cfg,
        "config": norm_cfg,
        "ppl": ppl,
        "bits": res["avg_effective_bits"],
        "bits_csr": res.get("avg_effective_bits_csr"),
        "bits_bitmap": res.get("avg_effective_bits_bitmap"),
        "bits_paper": res.get("avg_effective_bits_paper"),
        **({"alloc_summary": res["alloc_summary"]} if res.get("alloc_summary") else {}),
    }


def run_point(cfg: Dict[str, Any], dataloader, testenc, dev=DEV):
    set_seed(42)
    model = get_opt()
    if cfg.get("mode") == "awq":
        from awq_runner import apply_awq_official
        res = apply_awq_official(model, dev=dev, **cfg)
    else:
        res = run_compression_mode(
            model=model, dataloader=dataloader, dev=dev, nsamples=128, **cfg
        )
    ppl = evaluate_ppl(model, testenc, dev)
    del model
    torch.cuda.empty_cache()
    return res, ppl


def load_sweep_data() -> Dict[str, Any]:
    if SWEEP_DATA_PATH.exists():
        try:
            with open(SWEEP_DATA_PATH, "r") as f:
                d = json.load(f)
                return d
        except Exception as e:
            print(f"[WARN] Error leyendo {SWEEP_DATA_PATH}: {e}")
    return {}


def save_sweep_data(data: Dict[str, Any]):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SWEEP_DATA_PATH, "w") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Orquestador Unificado de Barridos de Pareto")
    parser.add_argument("families", nargs="*", default=list(FAMILIES.keys()),
                        help="Familias a ejecutar (por defecto todas).")
    parser.add_argument("--force", action="store_true",
                        help="Fuerza el re-cómputo de puntos ya existentes.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Lista qué configuraciones se ejecutarían sin correr inferencia ni tocar GPU.")
    args = parser.parse_args()

    # Validar familias solicitadas
    for fam in args.families:
        if fam not in FAMILIES:
            print(f"[ERROR] Familia '{fam}' desconocida. Disponibles: {list(FAMILIES.keys())}")
            sys.exit(1)

    sweep_data = load_sweep_data()

    if args.dry_run:
        print("="*65)
        print(" MODO DRY-RUN: Verificación de Puntos a Ejecutar")
        print("="*65)
        total_pending = 0
        total_cached = 0
        for fam in args.families:
            entries = sweep_data.get(fam, [])
            if isinstance(entries, dict):
                entries = [entries]
            print(f"\n[Familia: {fam}]")
            for cfg in FAMILIES[fam]:
                norm_cfg = normalize_config(cfg)
                already = not args.force and any(entry_matches_config(e, norm_cfg) for e in entries)
                if already:
                    matching = next(e for e in entries if entry_matches_config(e, norm_cfg))
                    b = matching.get("bits_csr") or matching.get("bits")
                    p = matching.get("ppl")
                    print(f"  [CACHE] {norm_cfg} -> PPL={p:.2f} | Bits={b:.2f}")
                    total_cached += 1
                else:
                    print(f"  [RUN  ] {norm_cfg} (PENDIENTE)")
                    total_pending += 1
        print("\n" + "="*65)
        print(f" Resumen Dry-Run: {total_cached} en cache, {total_pending} pendientes por correr.")
        print("="*65)
        return

    print("="*65)
    print(" ORQUESTADOR DE BARRIDOS DE PARETO (OPT-125M / WikiText-2)")
    print(f" Familias a procesar: {args.families}")
    print(f" Force: {args.force} | Device: {DEV}")
    print("="*65)

    dataloader, testenc = get_wikitext2(nsamples=128, seed=42, model="facebook/opt-125m")

    for fam in args.families:
        print(f"\n>>> Procesando familia: {fam}...")
        raw_entries = sweep_data.get(fam, [])
        if isinstance(raw_entries, dict):
            # Normalizar record_ptq si venía como dict
            raw_entries = [raw_entries]
        entries: List[Dict[str, Any]] = list(raw_entries)
        sweep_data[fam] = entries

        for cfg in FAMILIES[fam]:
            norm_cfg = normalize_config(cfg)
            if not args.force and any(entry_matches_config(e, norm_cfg) for e in entries):
                print(f"  [Skip] {fam} {norm_cfg} ya computado.")
                continue

            print(f"  -> Ejecutando {fam} {norm_cfg}...")
            res, ppl = run_point(norm_cfg, dataloader, testenc, dev=DEV)
            record = make_record(norm_cfg, res, ppl)

            # Reemplazar si existía
            entries[:] = [e for e in entries if not entry_matches_config(e, norm_cfg)]
            entries.append({**record, "git": GIT})
            sweep_data[fam] = entries
            save_sweep_data(sweep_data)
            print(f"  [OK] {fam} {norm_cfg} -> PPL={ppl:.2f} | Bits={record['bits']:.2f}")

    print("\n" + "="*65)
    print(f" ¡Barridos completados exitosamente! Datos consolidados en: {SWEEP_DATA_PATH}")
    print("="*65)


if __name__ == "__main__":
    main()