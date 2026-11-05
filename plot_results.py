#!/usr/bin/env python3
"""
Generador de Figuras Científicas para Experimento 43.
1. Frontera de Pareto: Bits Efectivos vs. Perplejidad WikiText-2.
2. Curvas de Compensación Iso-Presupuesto.

Estilo declarativo: agregar/quitar curvas solo requiere editar CURVE_SPECS.
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

CURRENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = CURRENT_DIR / "results"

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

FP16_PPL = 27.66


# ---------------------------------------------------------------------------
# Contabilidad de bits
# ---------------------------------------------------------------------------

def get_real_bits(p: dict) -> float:
    """min(bits_csr, bits_bitmap) cuando ambos existen; si no, el que haya."""
    csr = p.get("bits_csr")
    bmp = p.get("bits_bitmap")
    if csr is not None and bmp is not None:
        return min(csr, bmp)
    return bmp if bmp is not None else (csr if csr is not None else p.get("bits", float("nan")))


def get_dense_bits(p: dict) -> float:
    return p.get("bits", float("nan"))


def get_paper_bits(p: dict, default_sb: int = 8, default_ub: int = 4) -> float:
    """Contabilidad teorica del paper sin indices ni overhead de escalas."""
    if p.get("bits_paper") is not None:
        return p["bits_paper"]
    rho = p.get("rho")
    if rho is None:
        return float("nan")
    if "s_bits" in p or "u_bits" in p:
        kappa = p.get("kappa", 0.30)
        sb = p.get("s_bits", default_sb)
        ub = p.get("u_bits", default_ub)
        return (1.0 - rho) * ((1.0 - kappa) * sb + kappa * ub)
    return 16.0 * (1.0 - rho)


# ---------------------------------------------------------------------------
# Registro declarativo de curvas
# ---------------------------------------------------------------------------
# Campos obligatorios: family, label, color, linestyle, marker
# Opcionales: bits_fn (default get_real_bits), paper_label, paper_sb (8),
#             paper_ub (4), lw (1.8), ms (35), zorder (6)

CURVE_SPECS = [
    # ---- Triadas de la tesis ----
    dict(
        family="canonical_qls",
        label=r"Triada Q+L+S [bits reales] (Base Q en {2,3,4,6,8}b, r=16, k=1%, g=64)",
        color="#ff7f0e", linestyle="-", marker="*", lw=2.6, ms=55, zorder=8,
    ),
    dict(
        family="canonical_qls_l4",
        label=r"Triada Q+Lq+S [bits reales] (L cuantizado 4b, r=16, k=1%)",
        color="#d84315", linestyle="-", marker="P", lw=2.4, ms=45, zorder=9,
    ),
    dict(
        family="canonical_qls_l3_hq",
        label=r"Triada Q+Lq+S [bits reales] (L 3b Hessian-aware, escalas por columna, r=16, k=1%)",
        color="#6a1b9a", linestyle="-", marker="X", lw=2.4, ms=45, zorder=10,
    ),
    dict(
        family="triad_quant_only",
        label="Base Q sola [densa] (Hessian Scale Grid, g=64)",
        color="#f57c00", linestyle="--", marker="h", lw=2.0, ms=35, zorder=7,
        bits_fn=get_dense_bits,
    ),
    # ---- Cuantizadores densos de literatura ----
    dict(
        family="gptq",
        label="GPTQ [densa] (per-channel)",
        color="#8c564b", linestyle="-", marker="v", lw=1.8, ms=35, zorder=6,
        bits_fn=get_dense_bits,
    ),
    dict(
        family="gptq_g64",
        label="GPTQ [densa] (g=64, sin scale search)",
        color="#a0522d", linestyle=":", marker="^", lw=1.6, ms=30, zorder=6,
        bits_fn=get_dense_bits,
    ),
    dict(
        family="awq",
        label="AWQ [densa] (MLSys 2024, Auto-Scale+Clip, g=64)",
        color="#00897b", linestyle="--", marker="p", lw=1.8, ms=35, zorder=6,
        bits_fn=get_dense_bits,
    ),
    # ---- Poda bipartita FP16 ----
    dict(
        family="oats_fp16",
        label="OATS FP16 [bits reales: min(CSR, bitmap)] (ICLR 2025)",
        color="#1f77b4", linestyle="-.", marker="o", lw=1.8, ms=35, zorder=5,
        paper_label="OATS FP16 [paper: 16*(1-rho)]",
    ),
    dict(
        family="basil_fp16",
        label="3BASiL FP16 [bits reales: min(CSR, bitmap)] (NeurIPS 2025)",
        color="#17becf", linestyle=":", marker="s", lw=2.0, ms=35, zorder=5,
        paper_label="3BASiL FP16 [paper: 16*(1-rho)]",
    ),
    # ---- QSLR cuantizado ----
    dict(
        family="qslr_quantized",
        label="QSLR (U4/V4+S8) [bits reales: min(CSR, bitmap)]",
        color="#d62728", linestyle="--", marker="^", lw=1.8, ms=35, zorder=6,
        paper_label="QSLR (U4/V4+S8) [paper: sin indices]",
    ),
    dict(
        family="qslr_quantized_s4",
        label="QSLR (U4/V4+S4) [bits reales: min(CSR, bitmap)]",
        color="#e07b00", linestyle="--", marker="^", lw=1.8, ms=35, zorder=6,
        paper_label="QSLR (U4/V4+S4) [paper: sin indices]",
        paper_sb=4,
    ),
    dict(
        family="bipartite_bcd",
        label="BCD Bipartito (Sq+Lq, S8) [bits reales: min(CSR, bitmap)]",
        color="#9467bd", linestyle="--", marker="x", lw=1.8, ms=35, zorder=5,
    ),
    dict(
        family="bipartite_bcd_s4",
        label="BCD Bipartito (Sq+Lq, S4) [bits reales: min(CSR, bitmap)]",
        color="#7b1fa2", linestyle="-.", marker="P", lw=1.8, ms=35, zorder=5,
    ),
]


# ---------------------------------------------------------------------------
# Graficos
# ---------------------------------------------------------------------------

def _load_sweep_data() -> dict:
    path = RESULTS_DIR / "pareto_curves_data.json"
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARN] Error leyendo {path}: {e}")
    return {}


def plot_pareto_frontier(zoomed: bool = False):
    sweep_data = _load_sweep_data()
    fig, ax = plt.subplots(figsize=(11.5, 7.2), dpi=300)

    for spec in CURVE_SPECS:
        fam = spec["family"]
        entries = sweep_data.get(fam, [])
        if isinstance(entries, dict):
            entries = [entries]
        if not entries:
            print(f"[INFO] '{fam}' no encontrado en JSON.")
            continue

        bits_fn = spec.get("bits_fn", get_real_bits)
        color   = spec["color"]
        lw      = spec.get("lw", 1.8)
        ms      = spec.get("ms", 35)
        zo      = spec.get("zorder", 6)

        pts = sorted(entries, key=bits_fn)
        xb  = [bits_fn(p) for p in pts]
        yp  = [p["ppl"] for p in pts]

        ax.plot(xb, yp, color=color, linestyle=spec["linestyle"], linewidth=lw,
                zorder=zo, label=spec["label"])
        edge_col = None if spec["marker"] in ("x", "+") else "none"
        ax.scatter(xb, yp, color=color, marker=spec["marker"], s=ms,
                   edgecolors=edge_col, linewidth=1.0, zorder=zo + 1)

        if "paper_label" in spec:
            psb = spec.get("paper_sb", 8)
            pub = spec.get("paper_ub", 4)
            pts_p = sorted(entries, key=lambda p: get_paper_bits(p, psb, pub))
            xbp   = [get_paper_bits(p, psb, pub) for p in pts_p]
            ypp   = [p["ppl"] for p in pts_p]
            ax.plot(xbp, ypp, color=color, linestyle=":", linewidth=lw * 0.78,
                    alpha=0.55, zorder=zo - 1, label=spec["paper_label"])
            ax.scatter(xbp, ypp, color=color, marker="v", s=ms * 0.8,
                       alpha=0.55, zorder=zo - 1)

    # Punto Record PTQ
    record = sweep_data.get("record_ptq")
    if isinstance(record, list):
        record = record[0] if record else None
    if isinstance(record, dict):
        ptq_b = get_real_bits(record)
        ptq_p = record["ppl"]
        ax.scatter([ptq_b], [ptq_p], color="#d95f02", marker="D", s=55,
                   edgecolors="#7f2704", linewidth=1.2, zorder=12,
                   label=f"Record PTQ [bits reales] (PPL {ptq_p:.2f}, {ptq_b:.2f}b)")

    ax.axhline(FP16_PPL, color="#2ca02c", linestyle=":", linewidth=1.8,
               label=f"Base FP16 densa (PPL = {FP16_PPL})")
    ax.scatter([16.0], [FP16_PPL], color="#2ca02c", marker="D", s=40, zorder=5)

    ax.set_xlabel("Bits efectivos por parametro", fontsize=11, fontweight="bold")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.45)

    if zoomed:
        ax.set_ylim(25, 160)
        ax.set_xlim(1.5, 10.5)
        ax.yaxis.set_major_locator(ticker.FixedLocator([27, 30, 35, 40, 50, 60, 80, 100, 140]))
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_ylabel("Perplejidad WikiText-2 (Zoom, Menor es Mejor)", fontsize=11, fontweight="bold")
        ax.set_title("Frontera de Pareto: Detalle Regimen Alta Calidad (PPL <= 160, OPT-125M)",
                     fontsize=12, fontweight="bold", pad=12)
        ax.axvspan(1.5, 4.5, color="#e8f5e9", alpha=0.45,
                   label="Regimen objetivo de compresion extrema (< 4.5 bits)")
        ax.legend(loc="upper right", frameon=True, fontsize=7.5, framealpha=0.92)
        out_path = RESULTS_DIR / "pareto_bipartite_vs_qls_zoomed.png"
    else:
        ax.set_ylim(18, 14000)
        ax.set_xlim(0.8, 17.5)
        ax.set_ylabel("Perplejidad WikiText-2 (Escala Logaritmica, Menor es Mejor)",
                      fontsize=11, fontweight="bold")
        ax.set_title("Frontera de Pareto: Esquemas Bipartitos vs. Cuantizacion Densa vs. Triada Q+L+S (OPT-125M)",
                     fontsize=12, fontweight="bold", pad=12)
        ax.axvspan(1.0, 4.5, color="#e8f5e9", alpha=0.45,
                   label="Regimen objetivo de compresion extrema (< 4.5 bits)")
        ax.legend(loc="upper right", frameon=True, fontsize=7.0, framealpha=0.92)
        out_path = RESULTS_DIR / "pareto_bipartite_vs_qls.png"

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Grafico guardado en {out_path}")


def plot_iso_budget():
    diag_file = RESULTS_DIR / "iso_budget_sweep_diagnostic.json"
    if not diag_file.exists():
        print(f"[WARN] No se encontro {diag_file}")
        return

    with open(diag_file) as f:
        data = json.load(f)

    fig, ax1 = plt.subplots(figsize=(8.5, 5.5), dpi=300)

    fixed_data  = data.get("fixed_rho_results", []) if isinstance(data, dict) else data
    strict_data = data.get("strict_isobits_results", []) if isinstance(data, dict) else []

    kappas    = [d["kappa"] for d in fixed_data]
    err_fp16  = [d["err_fp16"] for d in fixed_data]
    err_quant = [d["err_quantized"] for d in fixed_data]
    bits_vals = [d.get("effective_bits_csr") or d.get("effective_bits_quant", 0) for d in fixed_data]

    color_err  = "#d62728"
    color_bits = "#1f77b4"

    ax1.set_xlabel(r"Ratio de Rango kappa", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Error Funcional Ponderado de Fisher", color=color_err, fontsize=11, fontweight="bold")
    l1 = ax1.plot(kappas, err_quant, color=color_err, marker="o", linewidth=2.2,
                  label="Error Cuantizado QSLR (rho=0.50 fija)")
    l2 = ax1.plot(kappas, err_fp16, color="#ff9896", marker="s", linestyle="--", linewidth=1.8,
                  label="Error FP16 OATS (rho=0.50 fija)")

    lines = l1 + l2
    if strict_data:
        ks = [d["kappa"] for d in strict_data]
        es = [d["err_quantized"] for d in strict_data]
        ls = ax1.plot(ks, es, color="#7b1fa2", marker="D", linewidth=2.0,
                      label="Error Cuantizado (Iso-Bits Estricto 4.0b)")
        lines += ls

    ax1.tick_params(axis="y", labelcolor=color_err)
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Bits Efectivos por Parametro (CSR)", color=color_bits, fontsize=11, fontweight="bold")
    l3 = ax2.plot(kappas, bits_vals, color=color_bits, marker="^", linewidth=2.0,
                  linestyle="-.", label="Bits Efectivos Reales (rho=0.50)")
    lines += l3
    if strict_data:
        tb = strict_data[0].get("target_bits", 4.0)
        lt = ax2.axhline(tb, color="#7b1fa2", linestyle=":", linewidth=1.5,
                         label=f"Presupuesto Iso-Bits ({tb:.1f}b)")
        lines.append(lt)

    ax2.tick_params(axis="y", labelcolor=color_bits)
    ax2.grid(False)

    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper center", frameon=True, fontsize=8.5)
    plt.title("Dilema Iso-Presupuesto en Esquemas Bipartitos: Error vs. Bits",
              fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()

    out_path = RESULTS_DIR / "iso_budget_tradeoff.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Grafico iso-presupuesto guardado en {out_path}")


if __name__ == "__main__":
    plot_pareto_frontier(zoomed=False)
    plot_pareto_frontier(zoomed=True)
    plot_iso_budget()