#!/usr/bin/env python3
"""
Tríada con tubo vs Q sola a igual costo, desde results/shift_comparison.json.
  Panel 1: OPT-125M con ADMM-Q y escala Hessiana (Q g128 vs tubo g256).
  Panel 2: OPT-1.3B con GPTQ (Q g128 vs tubo g256).
Línea = semilla 42; puntos claros = otras semillas.

Uso:
    python plot_tube_vs_q.py
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parent / "results"
OUT = RESULTS / "tube_vs_q.png"

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
Q_COLOR, TUBE_COLOR = "#2a78d6", "#eb6834"   # slots 1 y 2 de la paleta validada

PANELS = [
    dict(title="OPT-125M · ADMM-Q con escala Hessiana", model="facebook/opt-125m",
         q="Q admmH kfac {b} g128 γ0.0", tube="tubo admmH kfac {b} g256 γ0.0"),
    dict(title="OPT-1.3B · GPTQ", model="facebook/opt-1.3b",
         q="Q kfac {b} g128 γ0.0", tube="tubo kfac {b} g256 γ0.0"),
]
BUDGETS = (2.5, 3.0, 3.5, 4.0, 4.5)


def points(records, model, pattern):
    """{semilla: [(bits, ppl), ...]} para los presupuestos con resultado."""
    out = {}
    for b in BUDGETS:
        name = pattern.format(b=b)
        for r in records:
            if r["config"] == name and r.get("model", "facebook/opt-125m") == model:
                out.setdefault(r["seed"], []).append((r["bits"], r["ppl"]))
    return {s: sorted(v) for s, v in out.items()}


def main():
    records = json.loads((RESULTS / "shift_comparison.json").read_text())
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE)
    for ax, p in zip(axes, PANELS):
        ax.set_facecolor(SURFACE)
        q, t = points(records, p["model"], p["q"]), points(records, p["model"], p["tube"])
        for series, color, label in ((q, Q_COLOR, "Q sola"), (t, TUBE_COLOR, "Tríada con tubo")):
            main_pts = series.get(42, [])
            xs, ys = zip(*main_pts)
            ax.plot(xs, ys, color=color, lw=2, marker="o", ms=7, mec=SURFACE, mew=2,
                    label=f"{label} (semilla 42)", zorder=3)
            extra = [pt for s, v in series.items() if s != 42 for pt in v]
            if extra:
                ex, ey = zip(*extra)
                ax.scatter(ex, ey, s=90, facecolor="none", edgecolor=color, lw=1.5, zorder=4,
                           label=f"{label} (semilla{'s' if len(series) > 2 else ''} "
                                 f"{' y '.join(str(s) for s in sorted(series) if s != 42)})")
        # Diferencia medida por presupuesto (semilla 42)
        for (qb, qp), (tb, tp) in zip(q.get(42, []), t.get(42, [])):
            ax.annotate(f"{tp - qp:+.2f}", ((qb + tb) / 2, min(qp, tp)), xytext=(0, -16),
                        textcoords="offset points", ha="center", color=INK2, fontsize=8.5)
        ax.set_title(p["title"], loc="left", color=INK, fontsize=11)
        ax.set_xlabel("bits efectivos por peso (tasa fija)")
        ax.set_ylabel("perplejidad WikiText-2 (test)")
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.margins(x=0.08, y=0.12)
        ax.legend(frameon=False, loc="upper right", fontsize=8.5)
    fig.suptitle("Tríada con tubo vs Q sola a igual costo (menor es mejor). "
                 "Números: diferencia de perplejidad, semilla 42.",
                 x=0.01, ha="left", fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT, dpi=150, facecolor=SURFACE)
    print(f"[OK] {OUT}")


if __name__ == "__main__":
    main()
