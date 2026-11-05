#!/usr/bin/env python3
"""
Experimento 43 - Motor de Descomposición Bipartita Cuantizada y OATS / QSLR:
Implementa:
1. OATS Original (ICLR 2025): Alternating Thresholding con ponderación diagonal de Wanda D = diag(sqrt(X^T X)).
2. OATS Cuantizado Secuencial (Paradigma QSLR, IEEE Access 2025):
   W -> S + U V^T, luego cuantización independiente de S, U, V.
3. Closed-Loop BCD Bipartito Cuantizado (Propuesta del Director):
   Alternar en bucle cerrado entre S cuantizado y L = U V^T cuantizado (sin base densa Q).
4. Ponderación Wanda (D) vs Cholesky (L_H).
5. Barrido iso-presupuesto (r, k) para optimizar la asignación bajo restricción de memoria.
"""
from __future__ import annotations
import math
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import torch
import torch.nn as nn
import transformers

REPO_DIR = Path(__file__).resolve().parent / "vendor" / "gptq"
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from quant import Quantizer, quantize


def discrete_entropy(symbols: torch.Tensor) -> float:
    """Entropía empírica en bits por símbolo."""
    counts = torch.unique(symbols, return_counts=True)[1].double()
    prob = counts / counts.sum()
    return float(-(prob * prob.log2()).sum())


def index_entropy(idx: torch.Tensor, zero: torch.Tensor, groupsize: int) -> Dict[str, float]:
    """
    Entropía de los índices de Q, en bits por peso, sin costo de tablas.
      rel:  índice relativo al cero, una tabla por matriz.
      zero: índice absoluto condicionado al punto cero, que el decodificador conoce.
      grp:  índice relativo, una tabla por grupo de columnas.
    """
    n = idx.shape[1]
    idx, zero = idx.cpu(), zero.cpu()
    cond = 0.0
    abs_idx = idx + zero
    for z in torch.unique(zero):
        sel = zero == z
        cond += sel.double().mean().item() * discrete_entropy(abs_idx[sel])
    grp = sum(discrete_entropy(idx[:, s:s + groupsize]) * (min(s + groupsize, n) - s)
              for s in range(0, n, groupsize)) / n
    return {"rel": discrete_entropy(idx), "zero": cond, "grp": grp}


def quantize_matrix_uniform(
    tensor: torch.Tensor,
    n_bits: int = 4,
    groupsize: int = 64,
    symmetric: bool = False
) -> Tuple[torch.Tensor, float]:
    """
    Cuantiza un tensor 2D de forma uniforme por grupos de columnas.
    Retorna el tensor reconstruido y el overhead de bits por parámetro de escalas/ceros.
    """
    # Si el número de bits es mayor o igual a 16, no hacemos nada
    if n_bits >= 16:
        return tensor.clone(), 0.0

    # Guardamos la forma original
    orig_shape = tensor.shape
    m, n = orig_shape
    # Si groupsize es <= 0 o mayor a n, usamos n
    g = n if (groupsize <= 0 or groupsize > n) else groupsize
    
    # Pad n to multiple of g if needed
    pad = (g - (n % g)) % g
    if pad > 0:
        tensor = torch.nn.functional.pad(tensor, (0, pad))
    
    m_p, n_p = tensor.shape
    tensor_reshaped = tensor.view(-1, g)
    
    # Si es simétrica
    if symmetric:
        # Tomo el máximo absoluto
        max_val = torch.max(torch.abs(tensor_reshaped), dim=-1, keepdim=True)[0].clamp(min=1e-8)
        # Valor máximo entero positivo que se puede representar con n_bits
        qmax = (1 << (n_bits - 1)) - 1
        # Calculo la escala
        scale = max_val / qmax
        # Cero es cero
        zero = torch.zeros_like(scale)
        # Redondeo
        q = torch.clamp(torch.round(tensor_reshaped / scale), -qmax, qmax)
        # Reconstruyo el tensor
        rec = q * scale
        # escala en fp16 por grupo
        overhead_bits = (16.0 / g)  
    else:
        min_val = torch.min(tensor_reshaped, dim=-1, keepdim=True)[0]
        max_val = torch.max(tensor_reshaped, dim=-1, keepdim=True)[0]
        qmax = (1 << n_bits) - 1
        scale = (max_val - min_val).clamp(min=1e-8) / qmax
        zero = torch.round(-min_val / scale)
        q = torch.clamp(torch.round(tensor_reshaped / scale) + zero, 0, qmax)
        rec = (q - zero) * scale
        overhead_bits = (32.0 / g)  # escala + zero en fp16 por grupo

    rec = rec.view(m_p, n_p)
    if pad > 0:
        rec = rec[:, :n]

    return rec, overhead_bits


def compute_effective_bits_bipartite(
    m: int,
    n: int,
    r: int,
    k: int,
    u_bits: int = 4,
    v_bits: int = 4,
    s_bits: int = 8,
    groupsize: int = -1,
    sym_u: bool = False,
    sym_v: bool = False,
    sym_s: bool = False,
    scale_dim_uv: str = "row",
    s_quant_mode: str = "rowwise"
) -> Tuple[float, float]:
    """
    Calcula los bits efectivos por parámetro de W = S + U V^T bajo dos modelos de índice:

    Modelo A (CSR):
      - Índices de columna: ceil(log2(n)) bits por no-cero de S.
      - Punteros de fila: (m + 1) elementos de ceil(log2(k + 1)) bits.
      - Valores de S: k * s_bits.
      - Overhead de escalas de S.
    Modelo B (Bitmap):
      - 1 bit por posición de toda la matriz (m * n bits) para la máscara binaria de soporte.
      - Valores de S: k * s_bits.
      - Overhead de escalas de S.

    Factores de rango bajo U (m x r) y V (n x r):
      - Si bits == 16: FP16 puro, overhead de escala = 0.
      - Si bits < 16:
        - Escala por fila: (16 si sym sino 32) bits por fila.
        - Escala por columna: (16 si sym sino 32) bits por columna (r escalas en total).

    Escalas de S:
      - Si s_bits == 16 o s_quant_mode == "none": FP16 puro, overhead = 0.
      - Si s_quant_mode == "rowwise": m * (16 si sym_s sino 32) bits.
      - Si s_quant_mode == "grouped" y groupsize > 0: ceil(m*n/groupsize) * (16 si sym_s sino 32) bits.

    Retorna: (bits_csr, bits_bitmap) como par.
    """
    total_weights = m * n

    # --- Overhead de U ---
    if u_bits >= 16:
        bits_u = m * r * 16.0
    elif r > 0:
        cost_per_scale = 16.0 if sym_u else 32.0
        if scale_dim_uv == "col":
            bits_u = (m * r * u_bits) + (r * cost_per_scale)
        else:
            bits_u = (m * r * u_bits) + (m * cost_per_scale)
    else:
        bits_u = 0.0

    # --- Overhead de V ---
    if v_bits >= 16:
        bits_v = n * r * 16.0
    elif r > 0:
        cost_per_scale = 16.0 if sym_v else 32.0
        if scale_dim_uv == "col":
            bits_v = (n * r * v_bits) + (r * cost_per_scale)
        else:
            bits_v = (n * r * v_bits) + (n * cost_per_scale)
    else:
        bits_v = 0.0

    # --- Overhead de escalas de S ---
    if s_bits >= 16 or s_quant_mode == "none":
        bits_s_scales = 0.0
    else:
        cost_s_scale = 16.0 if sym_s else 32.0
        if s_quant_mode == "grouped" and groupsize > 0:
            n_groups = math.ceil(total_weights / groupsize)
            bits_s_scales = n_groups * cost_s_scale
        else:  # rowwise
            bits_s_scales = m * cost_s_scale

    # --- Modelo A: CSR ---
    idx_bits = max(1, int(math.ceil(math.log2(n + 1)))) if n > 1 else 1
    ptr_bits = max(1, int(math.ceil(math.log2(k + 1)))) if k > 0 else 1
    bits_s_csr = (k * (s_bits + idx_bits)) + ((m + 1) * ptr_bits) + bits_s_scales
    eff_bits_csr = (bits_u + bits_v + bits_s_csr) / total_weights

    # --- Modelo B: Bitmap (1 bit por posición para soporte global) ---
    bits_s_bitmap = (total_weights * 1.0) + (k * s_bits) + bits_s_scales
    eff_bits_bitmap = (bits_u + bits_v + bits_s_bitmap) / total_weights

    return float(eff_bits_csr), float(eff_bits_bitmap)


def grid_search_rowwise(
    T: torch.Tensor,
    H_proj: torch.Tensor,
    bits: int,
    alphas: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float]:
    """
    Cuantización independiente por fila con búsqueda de escala Hessian-aware
    y clipping centrado: lo = center - a * half_width, hi = center + a * half_width.
    """
    qmax = (1 << bits) - 1
    rows = T.shape[0]
    if mask is not None:
        _pinf = torch.full_like(T, float("inf"))
        _ninf = torch.full_like(T, float("-inf"))
        t_min = torch.where(mask, T, _pinf).min(dim=-1, keepdim=True)[0]
        t_max = torch.where(mask, T, _ninf).max(dim=-1, keepdim=True)[0]
        empty = ~mask.any(dim=-1, keepdim=True)
        t_min = torch.where(empty, torch.zeros_like(t_min), t_min)
        t_max = torch.where(empty, torch.zeros_like(t_max), t_max)
    else:
        t_min = T.min(dim=-1, keepdim=True)[0]
        t_max = T.max(dim=-1, keepdim=True)[0]

    center = 0.5 * (t_max + t_min)
    half_width = 0.5 * (t_max - t_min).clamp(min=1e-8)

    best_Tq = T.clone()
    best_loss_row = torch.full((rows, 1), float("inf"), device=T.device)

    for a in alphas:
        lo = center - a * half_width
        hi = center + a * half_width
        scale = (hi - lo).clamp(min=1e-8) / qmax
        zero = torch.round(-lo / scale)
        Tq = (torch.clamp(torch.round(T / scale) + zero, 0, qmax) - zero) * scale
        if mask is not None:
            Tq = torch.where(mask, Tq, torch.zeros_like(Tq))
        D = T - Tq
        loss_row = ((D @ H_proj) * D).sum(dim=-1, keepdim=True)
        upd = loss_row < best_loss_row
        best_loss_row = torch.where(upd, loss_row, best_loss_row)
        best_Tq = torch.where(upd, Tq, best_Tq)

    return best_Tq, best_loss_row.sum().item()


def grid_search_colwise(
    T: torch.Tensor,
    H_proj: torch.Tensor,
    bits: int,
    alphas: torch.Tensor,
    H_rows: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float]:
    """
    Cuantización independiente por columna (dim=0) con búsqueda de escala
    Hessian-aware y clipping centrado. Si H_rows se especifica (p. ej. H_S para V),
    calcula la pérdida de Fisher exacta: Tr(D^T H_rows D H_proj) = ((H_rows @ D) @ H_proj * D).sum().
    """
    qmax = (1 << bits) - 1
    t_min = T.min(dim=0, keepdim=True)[0]
    t_max = T.max(dim=0, keepdim=True)[0]
    center = 0.5 * (t_max + t_min)
    half_width = 0.5 * (t_max - t_min).clamp(min=1e-8)

    best_Tq = T.clone()
    best_loss = float("inf")

    for a in alphas:
        lo = center - a * half_width
        hi = center + a * half_width
        scale = (hi - lo).clamp(min=1e-8) / qmax
        zero = torch.round(-lo / scale)
        Tq = (torch.clamp(torch.round(T / scale) + zero, 0, qmax) - zero) * scale
        D = T - Tq
        if H_rows is not None:
            loss = (((H_rows @ D) @ H_proj) * D).sum().item()
        else:
            loss = ((D @ H_proj) * D).sum().item()
        if loss < best_loss:
            best_loss = loss
            best_Tq = Tq

    return best_Tq, best_loss


# ---------------------------------------------------------------------------
# Cuantización de factores de bajo rango L = U V^T (compartida por la Tríada
# y por test_l_quant.py)
# ---------------------------------------------------------------------------

_SCALE_LAYOUTS = {
    # scale_dim_l -> (eje de escalas de U, eje de escalas de V)
    "col":   ("col", "col"),
    "row":   ("row", "row"),
    "mixed": ("col", "row"),   # U por columna (r escalas), V por fila (n escalas)
}


def _grid_quant_axis(
    T: torch.Tensor,
    bits: int,
    alphas: torch.Tensor,
    axis: str,
    slice_loss,
) -> Tuple[torch.Tensor, float]:
    """
    Cuantización asimétrica con una escala/zero por fila (axis="row") o por
    columna (axis="col"), con clipping centrado y alpha elegido POR REBANADA.

    slice_loss(D) debe devolver la pérdida de cada rebanada con forma
    (rows, 1) si axis="row" o (1, cols) si axis="col".
    Con alphas = [1.0] se reduce exactamente a min-max.
    """
    qmax = (1 << bits) - 1
    red = -1 if axis == "row" else 0
    t_min = T.min(dim=red, keepdim=True)[0]
    t_max = T.max(dim=red, keepdim=True)[0]
    center = 0.5 * (t_max + t_min)
    half_width = 0.5 * (t_max - t_min).clamp(min=1e-8)

    best_Tq = T.clone()
    best_loss = torch.full_like(t_min, float("inf"))

    for a in alphas:
        lo = center - a * half_width
        hi = center + a * half_width
        scale = (hi - lo).clamp(min=1e-8) / qmax
        zero = torch.round(-lo / scale)
        Tq = (torch.clamp(torch.round(T / scale) + zero, 0, qmax) - zero) * scale
        loss = slice_loss(T - Tq)
        upd = loss < best_loss
        best_loss = torch.where(upd, loss, best_loss)
        best_Tq = torch.where(upd, Tq, best_Tq)

    return best_Tq, float(best_loss.sum().item())


def quantize_lowrank_factors(
    U_cont: torch.Tensor,
    V_cont: torch.Tensor,
    H: torch.Tensor,
    bits: int,
    scale_dim: str = "col",
    method: str = "minmax",
    grid_steps: int = 15,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Cuantiza L = U V^T (U: m x r, V: n x r) a `bits` bits.

    method="minmax":  reproduce exactamente el comportamiento previo de la
                      Tríada (min-max puro, sin re-target).
    method="hessian": búsqueda de clipping por rebanada minimizando el error
                      funcional tr(dL H dL^T), con re-target analítico de V
                      tras cuantizar U (como en QSLR / BCD bipartito).

    Pérdidas por rebanada usadas (dL = dU V^T  ó  U_q dV^T):
      U por fila i:     dU_i H_U dU_i^T                 con H_U = V^T H V
      U por columna j:  H_U[jj] * ||dU_:j||^2           (aprox. diagonal en r)
      V por fila k:     dV_k G dV_k^T                   con G = U_q^T U_q
                        (el factor H_kk no cambia el argmin de la fila)
      V por columna j:  G[jj] * dV_:j^T H dV_:j         (H completa entre canales)
    """
    if scale_dim not in _SCALE_LAYOUTS:
        raise ValueError(f"scale_dim desconocido: {scale_dim}")
    if method not in ("minmax", "hessian"):
        raise ValueError(f"method desconocido: {method}")
    ax_u, ax_v = _SCALE_LAYOUTS[scale_dim]
    dev = U_cont.device
    r = U_cont.shape[1]

    if method == "minmax":
        alphas = torch.ones(1, device=dev)
    else:
        alphas = torch.linspace(0.70, 1.05, grid_steps, device=dev)

    # ---------------- U ----------------
    H_U = V_cont.t() @ H @ V_cont  # r x r
    if ax_u == "row":
        loss_u = lambda D: ((D @ H_U) * D).sum(dim=-1, keepdim=True)
    else:
        diag_hu = torch.diag(H_U).unsqueeze(0)  # (1, r)
        loss_u = lambda D: (D * D).sum(dim=0, keepdim=True) * diag_hu
    U_q, _ = _grid_quant_axis(U_cont, bits, alphas, ax_u, loss_u)

    # ---------------- V ----------------
    if method == "hessian":
        G = U_q.t() @ U_q
        G_inv = torch.linalg.inv(G + 1e-6 * torch.eye(r, device=dev))
        V_tgt = V_cont @ (U_cont.t() @ U_q) @ G_inv
    else:
        G = U_q.t() @ U_q
        V_tgt = V_cont

    if ax_v == "row":
        loss_v = lambda D: ((D @ G) * D).sum(dim=-1, keepdim=True)
    else:
        diag_g = torch.diag(G).unsqueeze(0)  # (1, r)
        loss_v = lambda D: ((H @ D) * D).sum(dim=0, keepdim=True) * diag_g
    V_q, _ = _grid_quant_axis(V_tgt, bits, alphas, ax_v, loss_v)

    return U_q, V_q


def lowrank_bits(m: int, n: int, r: int, l_bits: int, scale_dim: str = "col") -> float:
    """Bits totales (no por parámetro) para almacenar L = U V^T."""
    if r <= 0:
        return 0.0
    if l_bits >= 16:
        return 16.0 * r * (m + n)
    cost_scale = 32.0  # escala + zero en FP16
    payload = (m + n) * r * l_bits
    ax_u, ax_v = _SCALE_LAYOUTS[scale_dim]
    n_scales_u = r if ax_u == "col" else m
    n_scales_v = r if ax_v == "col" else n
    return payload + (n_scales_u + n_scales_v) * cost_scale


# ---------------------------------------------------------------------------
# Componente S de la Tríada: valores cuantizados, índices y variante por columnas
# ---------------------------------------------------------------------------

def quantize_sparse_unstructured(
    S: torch.Tensor,
    mask: torch.Tensor,
    H: torch.Tensor,
    bits: int,
    grid_steps: int = 8,
) -> torch.Tensor:
    """
    Cuantiza los no-ceros de S con UNA escala simétrica por matriz (costo ~16 bits
    totales, despreciable), eligiendo el clipping alpha que minimiza tr(dS H dS^T).
    Las escalas por fila costarían 32/n b/p (~0.04 para n=768), lo mismo que
    todo S al 0.25%, por eso se descartan.
    El error de S lo absorbe Q, que en la Tríada siempre se recuantiza al final.
    """
    if bits >= 16 or not mask.any():
        return S.clone()
    qmax = (1 << (bits - 1)) - 1
    amax = S.abs().max().clamp(min=1e-12)
    best, best_loss = S.clone(), float("inf")
    for a in torch.linspace(0.5, 1.0, grid_steps, device=S.device):
        scale = a * amax / qmax
        Sq = torch.clamp(torch.round(S / scale), -qmax, qmax) * scale
        Sq = torch.where(mask, Sq, torch.zeros_like(Sq))
        D = S - Sq
        loss = ((D @ H) * D).sum().item()
        if loss < best_loss:
            best_loss, best = loss, Sq
    return best


def quantize_sparse_columns(S_cols: torch.Tensor, bits: int, grid_steps: int = 8) -> torch.Tensor:
    """
    Cuantiza columnas densas de S (m x c) con escala/zero asimétricos POR COLUMNA.
    En la métrica tr(dS H dS^T) restringida a una columna j, la pérdida es
    H_jj ||dS_:j||^2 más términos cruzados; la búsqueda por columna usa MSE
    (H_jj no cambia el argmin de la columna).
    """
    if bits >= 16:
        return S_cols.clone()
    alphas = torch.linspace(0.70, 1.0, grid_steps, device=S_cols.device)
    Sq, _ = _grid_quant_axis(S_cols, bits, alphas, "col", lambda D: (D * D).sum(dim=0, keepdim=True))
    return Sq


def sparse_bits(
    m: int,
    n: int,
    S_mat: torch.Tensor,
    s_struct: str,
    s_val_bits: int,
    s_index: str,
    s_cols: int,
) -> Dict[str, float]:
    """
    Bits totales (no por parámetro) para almacenar S.

    unstructured:
      valores:  k * s_val_bits  (+16 de la escala global si s_val_bits < 16)
      índice:
        "csr"      -> k * ceil(log2(n+1)) + (m+1) * ceil(log2(k+1))   (histórico)
        "rowcount" -> k * ceil(log2(n+1)) + m * ceil(log2(max_k_fila+1))
      "entropy_bound": log2 C(mn, k), cota inferior de cualquier índice no
                       estructurado (solo informativa).
    column:
      c columnas densas: c*m*s_val_bits + c*32 (escala+zero por columna)
                         + c*ceil(log2(n+1)) (qué columnas son).
    """
    if s_struct == "column":
        if s_cols <= 0:
            return {"total": 0.0, "values": 0.0, "index": 0.0, "entropy_bound": 0.0, "k": 0}
        idx = max(1, math.ceil(math.log2(n + 1)))
        vals = s_cols * m * s_val_bits + (s_cols * 32.0 if s_val_bits < 16 else 0.0)
        index = s_cols * idx
        return {"total": vals + index, "values": vals, "index": index,
                "entropy_bound": float(index), "k": s_cols * m}

    nz = S_mat != 0
    k = int(nz.sum().item())
    if k == 0:
        return {"total": 0.0, "values": 0.0, "index": 0.0, "entropy_bound": 0.0, "k": 0}
    vals = k * s_val_bits + (16.0 if s_val_bits < 16 else 0.0)
    idx_bits = max(1, math.ceil(math.log2(n + 1)))
    if s_index == "rowcount":
        max_row = int(nz.sum(dim=1).max().item())
        index = k * idx_bits + m * max(1, math.ceil(math.log2(max_row + 1)))
    else:  # "csr"
        index = k * idx_bits + (m + 1) * max(1, math.ceil(math.log2(k + 1)))
    N = m * n
    log2_binom = (math.lgamma(N + 1) - math.lgamma(k + 1) - math.lgamma(N - k + 1)) / math.log(2)
    return {"total": vals + index, "values": vals, "index": float(index),
            "entropy_bound": log2_binom, "k": k}


class BipartiteQuantEngine:
    def __init__(self, layer: Any):
        self.layer = layer
        self.dev = layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.W_orig = W.float()
        self.rows = W.shape[0]     # d_out (m)
        self.columns = W.shape[1]  # d_in  (n)
        
        # Estadísticas de activación
        self.H = torch.zeros((self.columns, self.columns), device=self.dev, dtype=torch.float32)
        self.nsamples = 0

    def add_batch(self, inp: torch.Tensor, out: torch.Tensor):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def add_cross(self, inp: torch.Tensor, inp_fp: torch.Tensor):
        """Acumula C = (2/N) sum (x_fp - x) x^T, con la misma normalización que H."""
        inp = inp.reshape((-1, inp.shape[-1])).t().float()
        inp_fp = inp_fp.reshape((-1, inp_fp.shape[-1])).t().float()
        if not hasattr(self, "C"):
            self.C = torch.zeros_like(self.H)
            self.nc = 0
        tmp = 1
        self.C *= self.nc / (self.nc + tmp)
        self.nc += tmp
        self.C += (2 / self.nc) * (inp_fp - inp).matmul(inp.t())

    def apply_gradient_shift(self, gamma: float, percdamp: float = 0.01):
        """Reemplaza W por W + gamma W C H^-1: GPTQ sobre este objetivo equivale a la
        compensación de primer orden de GPTAQ con gradiente actualizado."""
        if gamma == 0 or not hasattr(self, "C"):
            return
        L_H, _ = self.get_cholesky(percdamp=percdamp)
        H_inv = torch.cholesky_inverse(L_H)
        self.W_orig = self.W_orig + gamma * (self.W_orig @ self.C @ H_inv)

    def get_wanda_d(self) -> torch.Tensor:
        """Matriz diagonal de activación D = diag(sqrt(X^T X))."""
        d = torch.sqrt(torch.diag(self.H)).clamp(min=1e-8)
        return d

    def get_cholesky(self, percdamp: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor]:
        H = self.H.clone()
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp

        L_H = torch.linalg.cholesky(H)
        L_H_inv = torch.linalg.inv(L_H)
        return L_H, L_H_inv

    def evaluate_functional_error(self, W_approx: torch.Tensor) -> float:
        """Calcula el error cuadrático medio ponderado por la Hessiana de Fisher."""
        diff = (self.W_orig - W_approx).float()
        err = torch.trace(diff @ self.H @ diff.t()).item() / self.rows
        return float(err)

    # -------------------------------------------------------------
    # 1. OATS Original (ICLR 2025) en FP16
    # -------------------------------------------------------------
    def run_oats_fp16(
        self,
        rho: float = 0.50,
        kappa: float = 0.30,
        iterations: int = 80,
        use_cholesky: bool = False,
        per_row: bool = False
    ) -> Dict[str, Any]:
        """
        Ejecuta el algoritmo OATS original: Alternating Thresholding sobre W D.
        Retorna S y L en FP16 puro.
        Calcula tanto los bits teóricos del paper (16 * (1 - rho)) como los
        bits efectivos reales con máscara binaria (bitmap) o índices CSR.
        """
        m, n = self.rows, self.columns
        d_geom = (m * n) / (m + n)
        r = max(1, int(kappa * (1.0 - rho) * d_geom))
        k = max(1, int((1.0 - kappa) * (1.0 - rho) * m * n))
        
        if use_cholesky:
            L_H, L_H_inv = self.get_cholesky()
            W_weighted = self.W_orig @ L_H
        else:
            d = self.get_wanda_d()
            W_weighted = self.W_orig * d.unsqueeze(0)  # W @ D

        # Alternating Thresholding
        L = torch.zeros_like(W_weighted)
        S = torch.zeros_like(W_weighted)
        
        for it in range(iterations):
            # Paso Low-Rank: SVT sobre (W_weighted - S)
            R_L = W_weighted - S
            U, s_vals, Vh = torch.linalg.svd(R_L, full_matrices=False)
            r_cur = min(r, s_vals.shape[0])
            L = (U[:, :r_cur] * s_vals[:r_cur].unsqueeze(0)) @ Vh[:r_cur, :]
            
            # Paso Sparse: Hard-thresholding sobre (W_weighted - L)
            R_S = W_weighted - L
            if per_row:
                base_k = k // m
                rem = k % m
                abs_R = torch.abs(R_S)
                mask = torch.zeros_like(R_S, dtype=torch.bool)
                for row_idx in range(m):
                    k_this = base_k + (1 if row_idx < rem else 0)
                    if k_this > 0:
                        row_abs = abs_R[row_idx]
                        k_this_clamped = min(k_this, row_abs.numel())
                        th = torch.topk(row_abs, k_this_clamped, largest=True)[0][-1]
                        mask[row_idx] = row_abs >= th
            else:
                abs_R = torch.abs(R_S).flatten()
                threshold = torch.topk(abs_R, k, largest=True)[0][-1]
                mask = torch.abs(R_S) >= threshold
            S = torch.zeros_like(R_S)
            S[mask] = R_S[mask]

        # Deshacer ponderación y extraer factores exactos U y V
        r_cur = min(r, s_vals.shape[0])
        sqrt_s = torch.sqrt(s_vals[:r_cur])
        U_factor = U[:, :r_cur] * sqrt_s.unsqueeze(0)
        if use_cholesky:
            L_unweighted = L @ L_H_inv
            S_unweighted = S @ L_H_inv
            Vh_unw = Vh[:r_cur, :] @ L_H_inv
            V_factor = Vh_unw.t() * sqrt_s.unsqueeze(0)
        else:
            d_inv = 1.0 / d
            L_unweighted = L * d_inv.unsqueeze(0)
            S_unweighted = S * d_inv.unsqueeze(0)
            Vh_unw = Vh[:r_cur, :] * d_inv.unsqueeze(0)
            V_factor = Vh_unw.t() * sqrt_s.unsqueeze(0)

        W_approx = L_unweighted + S_unweighted
        err = self.evaluate_functional_error(W_approx)

        # Contabilidad unificada: paper nominal vs hardware real (bitmap / CSR)
        eff_bits_paper = 16.0 * (1.0 - rho)
        bits_csr, bits_bitmap = compute_effective_bits_bipartite(
            m=m, n=n, r=r_cur, k=k,
            u_bits=16, v_bits=16, s_bits=16,
            s_quant_mode="none"
        )

        return {
            "mode": "OATS_FP16" + ("_Cholesky" if use_cholesky else "_Wanda"),
            "r": r_cur,
            "k": k,
            "rho": rho,
            "kappa": kappa,
            "effective_bits": bits_bitmap,
            "effective_bits_paper": eff_bits_paper,
            "effective_bits_bitmap": bits_bitmap,
            "effective_bits_csr": bits_csr,
            "functional_error": err,
            "W_approx": W_approx,
            "U_factor": U_factor,
            "V_factor": V_factor,
            "S_unweighted": S_unweighted,
            "L_unweighted": L_unweighted
        }

    # -------------------------------------------------------------
    # 1b. 3BASiL (NeurIPS 2025): 3-Block ADMM sobre W D = S + L en FP16
    # -------------------------------------------------------------
    def run_3basil_fp16(
        self,
        rho_budget: float = 0.50,
        kappa: float = 0.30,
        iterations: int = 50,
        rho_admm_init: float = 0.1,
        c_l: float = 10.0
    ) -> Dict[str, Any]:
        """
        Implementación del algoritmo oficial 3BASiL (NeurIPS 2025):
        Resuelve min_{S, L} 0.5 * ||X W - X(S + L)||_F^2 + 0.5 * lambda * ||W - (S + L)||_F^2
        mediante ADMM de 3 bloques sobre (S, L, D) con multiplicador dual V,
        con monitoreo adaptativo de soporte de D y re-fit final sin sesgo dual.
        """
        m, n = self.rows, self.columns
        d_geom = (m * n) / (m + n)
        r = max(1, int(kappa * (1.0 - rho_budget) * d_geom))
        k = max(1, int((1.0 - kappa) * (1.0 - rho_budget) * m * n))

        # Normalización de coordenadas por norma de activación (ALPS / 3BASiL)
        cpu_dev = torch.device("cpu")
        H = self.H.to(cpu_dev)
        X_norm = torch.diag(H).sqrt().clamp(min=1e-8)
        H_norm = (H / X_norm).t() / X_norm
        
        # Damping ridge
        damp = 0.01
        diag_idx = torch.arange(n, device=cpu_dev)
        H_norm[diag_idx, diag_idx] += damp

        # Descomposición espectral de H_norm
        eigvals, Q = torch.linalg.eigh(H_norm.double())
        eigvals = eigvals.clamp(min=1e-8).float()
        Q = Q.float()

        # Matrices raíz normalizadas
        H_sqrt = (Q * torch.sqrt(eigvals).unsqueeze(0)) @ Q.t()
        H_inv_sqrt = (Q * (1.0 / torch.sqrt(eigvals)).unsqueeze(0)) @ Q.t()

        # Inicialización ADMM en coordenadas normalizadas
        W = (self.W_orig.to(cpu_dev) * X_norm.unsqueeze(0))
        L = torch.zeros_like(W)
        S = torch.zeros_like(W)
        D = torch.zeros_like(W)
        V = torch.zeros_like(W)
        rho_t = float(rho_admm_init)

        # Historial de soporte de D para monitoreo y schedule adaptativo
        D_supp_prev = (D != 0)

        for it in range(iterations):
            S_prev = S.clone()
            L_prev = L.clone()

            # 1. Bloque S: S^{(t+1)} = [(W - L) H_norm - V + rho_t D] (H_norm + rho_t I)^{-1}
            rhs = (W - L) @ H_norm - V + (rho_t * D)
            inv_diag = 1.0 / (eigvals + rho_t)
            S = (rhs @ Q * inv_diag.unsqueeze(0)) @ Q.t()

            # 2. Bloque L: L* = P_r((W - S) H_norm^{1/2}) H_norm^{-1/2}
            M = (W - S) @ H_sqrt
            U, s_vals, Vh = torch.linalg.svd(M, full_matrices=False)
            r_cur = min(r, s_vals.shape[0])
            P_r_M = (U[:, :r_cur] * s_vals[:r_cur].unsqueeze(0)) @ Vh[:r_cur, :]
            L_star = P_r_M @ H_inv_sqrt

            # Regla de aceptación para L (gating)
            diff_L = torch.norm(L_star - L_prev, p="fro")
            diff_S = torch.norm(S - S_prev, p="fro")
            if it == 0 or diff_L <= c_l * diff_S:
                L = L_star
            else:
                L = L_prev

            # 3. Bloque D: D^{(t+1)} = P_S(S + V / rho_t)
            target_D = S + (V / rho_t)
            abs_target = torch.abs(target_D).flatten()
            thresh = torch.topk(abs_target, k, largest=True)[0][-1]
            mask_D = torch.abs(target_D) >= thresh
            D = torch.zeros_like(target_D)
            D[mask_D] = target_D[mask_D]

            # 4. Actualización Dual V
            V = V + rho_t * (S - D)

            # 5. Monitoreo del soporte de D y schedule de penalidad rho_t
            D_supp_cur = (D != 0)
            supp_diff = torch.sum(D_supp_cur != D_supp_prev).item() / max(1, k)
            D_supp_prev = D_supp_cur
            if (it + 1) % 5 == 0 or supp_diff < 0.005:
                rho_t = min(rho_t * 1.25, 1e4)

        # Re-fit final alternando 3 veces D <-> L sobre el soporte fijo mask_D en la métrica de H
        # (Elimina sesgo de variable dual y proyecta en el espacio funcional de Fisher)
        D_final = torch.zeros_like(W)
        L_final = L.clone()
        for _ in range(3):
            D_final = torch.zeros_like(W)
            D_final[mask_D] = (W - L_final)[mask_D]
            M_final = (W - D_final) @ H_sqrt
            U_f, s_f, Vh_f = torch.linalg.svd(M_final, full_matrices=False)
            r_f = min(r, s_f.shape[0])
            L_final = ((U_f[:, :r_f] * s_f[:r_f].unsqueeze(0)) @ Vh_f[:r_f, :]) @ H_inv_sqrt

        # Deshacer normalización para recuperar pesos en espacio original
        W_approx = ((D_final + L_final) / X_norm.unsqueeze(0)).to(self.dev)
        err = self.evaluate_functional_error(W_approx)

        eff_bits_paper = 16.0 * (1.0 - rho_budget)
        bits_csr, bits_bitmap = compute_effective_bits_bipartite(
            m=m, n=n, r=r_f, k=k,
            u_bits=16, v_bits=16, s_bits=16,
            s_quant_mode="none"
        )

        return {
            "mode": "3BASiL_FP16",
            "r": r_f,
            "k": k,
            "rho": rho_budget,
            "kappa": kappa,
            "effective_bits": bits_bitmap,
            "effective_bits_paper": eff_bits_paper,
            "effective_bits_bitmap": bits_bitmap,
            "effective_bits_csr": bits_csr,
            "functional_error": err,
            "W_approx": W_approx
        }


    # -------------------------------------------------------------
    # 2. OATS Cuantizado Secuencial (Paradigma QSLR, IEEE Access 2025)
    # -------------------------------------------------------------
    def run_oats_quantized_sequential(
        self,
        rho: float = 0.50,
        kappa: float = 0.30,
        u_bits: int = 4,
        v_bits: int = 4,
        s_bits: int = 8,
        groupsize: int = -1,
        iterations: int = 80,
        grid_steps: int = 15,
        refinement_steps: int = 0,
        per_column_uv: bool = False
    ) -> Dict[str, Any]:
        """
        Cuantización Secuencial Abierta basada en el paradigma QSLR (Kwon, IEEE Access 2025):
        1. Descomposición OATS: W = W_S + U V^T (rho, kappa).
        2. Búsqueda en rejilla Hessian-weighted para escala y zero-point (clipping simétrico):
           - Para U: Minimiza Tr(Delta U H_U Delta U^T) con Hessiana proyectada H_U = V^T H_S V.
           - Para V: Re-target analítico de mínimos cuadrados: V_tgt = V (U^T U_q) G^-1 con G = U_q^T U_q,
                     seguido de cuantización ponderada por G y H_S.
           - Para W_S: Minimiza Tr(Delta W_S H_S Delta W_S^T) sobre las posiciones no-cero.
        3. Soporte para escalas por fila o por columna en U y V (per_column_uv).
        (Nota: AdaRound fue removido debido a inestabilidad numérica en la celosía discreta;
        groupsize y refinement_steps se preservan por compatibilidad de interfaz).
        """
        # Paso 1: OATS continuo y extracción analítica de factores exactos
        res_oats = self.run_oats_fp16(rho=rho, kappa=kappa, iterations=iterations)
        r = res_oats["r"]
        k = res_oats["k"]
        U_factor = res_oats["U_factor"]  # m x r
        V_factor = res_oats["V_factor"]  # n x r
        W_S = res_oats["S_unweighted"]    # m x n (sparse)
        
        H_S = self.H
        H_U = V_factor.t() @ H_S @ V_factor  # r x r

        alphas_grid = torch.linspace(0.70, 1.05, grid_steps, device=self.dev)

        # --- Componente U (m x r, u_bits) ---
        if per_column_uv:
            best_U_q, best_loss_u = grid_search_colwise(U_factor, H_U, u_bits, alphas_grid)
        else:
            best_U_q, best_loss_u = grid_search_rowwise(U_factor, H_U, u_bits, alphas_grid)

        # --- Componente V (n x r, v_bits) con Re-target ---
        G = best_U_q.t() @ best_U_q
        G_reg = G + 1e-6 * torch.eye(r, device=self.dev)
        G_inv = torch.linalg.inv(G_reg)
        V_tgt = V_factor @ (U_factor.t() @ best_U_q) @ G_inv

        if per_column_uv:
            best_V_q, _ = grid_search_colwise(V_tgt, G, v_bits, alphas_grid, H_rows=H_S)
        else:
            best_V_q, _ = grid_search_rowwise(V_tgt, G, v_bits, alphas_grid)

        dV_grid = V_tgt - best_V_q
        best_loss_v = (((H_S @ dV_grid) @ G) * dV_grid).sum().item()

        # --- Componente W_S (m x n, s_bits) ---
        mask_s = (W_S != 0)
        if mask_s.any():
            best_S_q, best_loss_s = grid_search_rowwise(
                W_S, H_S, s_bits, alphas_grid, mask=mask_s
            )
        else:
            best_S_q = W_S.clone()
            best_loss_s = 0.0

        # Reconstrucción
        L_q = best_U_q @ best_V_q.t()
        W_approx = L_q + best_S_q
        err = self.evaluate_functional_error(W_approx)

        scale_dim_str = "col" if per_column_uv else "row"
        eff_bits_csr, eff_bits_bitmap = compute_effective_bits_bipartite(
            self.rows, self.columns, r, k,
            u_bits=u_bits, v_bits=v_bits, s_bits=s_bits,
            groupsize=-1, scale_dim_uv=scale_dim_str, s_quant_mode="rowwise"
        )
        eff_bits = eff_bits_csr

        # Métrica teórica del paper QSLR (sin overhead de índices ni escalas)
        bits_paper = (1.0 - rho) * ((1.0 - kappa) * s_bits + kappa * u_bits)

        # Diagnósticos
        W_orig = self.W_orig
        L_cont = res_oats["L_unweighted"]
        S_cont = res_oats["S_unweighted"]

        err_decomp = W_orig - L_cont - S_cont
        dL = L_cont - L_q
        dS = S_cont - best_S_q

        ip = lambda A, B: ((A @ H_S) * B).sum().item()

        loss_decomp    = ip(err_decomp, err_decomp)
        loss_quant_L   = ip(dL, dL)
        loss_quant_S   = ip(dS, dS)
        cross_LS       = 2.0 * ip(dS, dL)
        cross_decomp_L = 2.0 * ip(err_decomp, dL)
        cross_decomp_S = 2.0 * ip(err_decomp, dS)

        err_total  = W_orig - W_approx
        loss_total = ip(err_total, err_total)
        loss_sum = (loss_decomp + loss_quant_L + loss_quant_S
                    + cross_LS + cross_decomp_L + cross_decomp_S)
        rel_err = abs(loss_total - loss_sum) / (abs(loss_total) + 1e-12)

        diag_info = {
            "grid_loss_U":          best_loss_u,
            "grid_loss_V":          best_loss_v,
            "grid_loss_S":          best_loss_s,
            "loss_decomp_oats":     loss_decomp,
            "loss_quant_L":         loss_quant_L,
            "loss_quant_S":         loss_quant_S,
            "cross_LS":             cross_LS,
            "cross_decomp_L":       cross_decomp_L,
            "cross_decomp_S":       cross_decomp_S,
            "loss_total":           loss_total,
            "closure_rel_err":      rel_err,
            "effective_bits_csr":   eff_bits_csr,
            "effective_bits_bitmap": eff_bits_bitmap,
            "bits_paper":           bits_paper,
            "per_column_uv":        per_column_uv,
            "adaround_active":      False,
        }

        return {
            "mode": "OATS_Quantized_QSLR",
            "r": r,
            "k": k,
            "rho": rho,
            "kappa": kappa,
            "u_bits": u_bits,
            "v_bits": v_bits,
            "s_bits": s_bits,
            "bits_paper": bits_paper,
            "effective_bits": eff_bits,
            "effective_bits_csr": eff_bits_csr,
            "effective_bits_bitmap": eff_bits_bitmap,
            "diagnostics": diag_info,
            "functional_error": err,
            "W_approx": W_approx,
        }

    # -------------------------------------------------------------
    # 3. BCD Bipartito en Bucle Cerrado Cuantizado (Propuesta del Director)
    # -------------------------------------------------------------
    def run_bipartite_bcd_quantized(
        self,
        r: int = 64,
        sparsity: float = 0.05,
        rho: Optional[float] = None,
        kappa: Optional[float] = None,
        u_bits: int = 4,
        v_bits: int = 4,
        s_bits: int = 8,
        groupsize: int = -1,
        cycles: int = 5,
        use_qslr_quantizer: bool = True,
        per_column_uv: bool = False
    ) -> Dict[str, Any]:
        """
        Alterna iterativamente dentro de un bucle cerrado (Closed-Loop BCD):
        Paso S: R_S = W - L_q -> Hard-thresholding -> Cuantizar S_q
        Paso L: R_L = W - S_q -> SVD rank r -> Cuantizar U_q y V_q -> L_q = U_q V_q^T

        Si rho y kappa se especifican, deriva r y k con la misma fórmula de OATS/QSLR
        para garantizar paridad experimental estricta (bucle cerrado vs secuencial abierto).
        """
        m, n = self.rows, self.columns
        if rho is not None and kappa is not None:
            d_geom = (m * n) / (m + n)
            r = max(1, int(kappa * (1.0 - rho) * d_geom))
            k = max(1, int((1.0 - kappa) * (1.0 - rho) * m * n))
            sparsity = k / (m * n)
        else:
            k = max(1, int(sparsity * m * n))
        L_H, L_H_inv = self.get_cholesky()

        # Inicialización de L y S cuantizadas
        L_q = torch.zeros_like(self.W_orig)
        S_q = torch.zeros_like(self.W_orig)

        alphas_grid = torch.linspace(0.70, 1.05, 15, device=self.dev)

        for cycle in range(cycles):
            # --- Actualización de S Cuantizada ---
            R_S = self.W_orig - L_q
            salience = torch.abs(R_S) * torch.sqrt(torch.diag(self.H)).unsqueeze(0)
            threshold = torch.topk(salience.flatten(), k, largest=True)[0][-1]
            mask_s = salience >= threshold

            S_target = torch.zeros_like(R_S)
            S_target[mask_s] = R_S[mask_s]

            if use_qslr_quantizer:
                if mask_s.any():
                    S_q, _ = grid_search_rowwise(S_target, self.H, s_bits, alphas_grid, mask=mask_s)
                else:
                    S_q = S_target.clone()
            else:
                S_q_temp, _ = quantize_matrix_uniform(S_target, n_bits=s_bits, groupsize=groupsize, symmetric=True)
                S_q = torch.zeros_like(self.W_orig)
                S_q[mask_s] = S_q_temp[mask_s]

            # --- Actualización de L Cuantizada ---
            R_L = self.W_orig - S_q
            M_L = R_L @ L_H
            U, s_vals, Vh = torch.linalg.svd(M_L, full_matrices=False)
            r_cur = min(r, s_vals.shape[0])
            sqrt_s = torch.sqrt(s_vals[:r_cur])
            U_cont = U[:, :r_cur] * sqrt_s.unsqueeze(0)
            V_cont = (Vh[:r_cur, :] @ L_H_inv).t() * sqrt_s.unsqueeze(0)

            if use_qslr_quantizer:
                H_U = V_cont.t() @ self.H @ V_cont
                if per_column_uv:
                    U_q, _ = grid_search_colwise(U_cont, H_U, u_bits, alphas_grid)
                else:
                    U_q, _ = grid_search_rowwise(U_cont, H_U, u_bits, alphas_grid)

                # Re-target analítico de V
                G = U_q.t() @ U_q
                G_reg = G + 1e-6 * torch.eye(r_cur, device=self.dev)
                G_inv = torch.linalg.inv(G_reg)
                V_tgt = V_cont @ (U_cont.t() @ U_q) @ G_inv

                if per_column_uv:
                    V_q, _ = grid_search_colwise(V_tgt, G, v_bits, alphas_grid, H_rows=self.H)
                else:
                    V_q, _ = grid_search_rowwise(V_tgt, G, v_bits, alphas_grid)
            else:
                U_q, _ = quantize_matrix_uniform(U_cont, n_bits=u_bits, groupsize=groupsize, symmetric=True)
                V_q, _ = quantize_matrix_uniform(V_cont, n_bits=v_bits, groupsize=groupsize, symmetric=True)

            L_q = U_q @ V_q.t()

        W_approx = L_q + S_q
        err = self.evaluate_functional_error(W_approx)

        s_mode = "rowwise" if use_qslr_quantizer else ("grouped" if groupsize > 0 else "rowwise")
        scale_dim_str = "col" if per_column_uv else "row"
        eff_bits_csr, eff_bits_bitmap = compute_effective_bits_bipartite(
            m, n, r, k,
            u_bits=u_bits, v_bits=v_bits, s_bits=s_bits,
            groupsize=groupsize,
            sym_u=not use_qslr_quantizer,
            sym_v=not use_qslr_quantizer,
            sym_s=not use_qslr_quantizer,
            scale_dim_uv=scale_dim_str,
            s_quant_mode=s_mode
        )

        return {
            "mode": "Bipartite_BCD_Quantized",
            "r": r,
            "k": k,
            "rho": rho,
            "kappa": kappa,
            "sparsity": sparsity,
            "effective_bits": eff_bits_csr,
            "effective_bits_csr": eff_bits_csr,
            "effective_bits_bitmap": eff_bits_bitmap,
            "functional_error": err,
            "W_approx": W_approx
        }


    # -------------------------------------------------------------
    # 4. GPTQ Clásico (Frantar et al., ICLR 2023)
    # -------------------------------------------------------------
    def run_gptq(
        self,
        wbits: int = 3,
        groupsize: int = -1,
        percdamp: float = 0.01,
        blocksize: int = 128,
        sym: bool = False
    ) -> Dict[str, Any]:
        """
        Ejecuta GPTQ clásico sobre la capa lineal utilizando la matriz H acumulada.
        """
        W = self.W_orig.clone().float()
        H = self.H.clone()
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        quantizer = Quantizer()
        quantizer.configure(wbits, perchannel=True, sym=sym, mse=False)
        quantizer.find_params(W, weight=True)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        Q = torch.zeros_like(W)

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1 and (i1 + i) % groupsize == 0:
                    quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)

                q = quantize(
                    w.unsqueeze(1), quantizer.scale, quantizer.zero, quantizer.maxq
                ).flatten()
                Q1[:, i] = q

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        err = self.evaluate_functional_error(Q)
        overhead = (16.0 / groupsize if sym else 32.0 / groupsize) if groupsize > 0 else 0.0
        eff_bits = float(wbits + overhead)

        return {
            "mode": f"GPTQ_{wbits}bit",
            "wbits": wbits,
            "groupsize": groupsize,
            "effective_bits": eff_bits,
            "functional_error": err,
            "W_approx": Q
        }

    # -------------------------------------------------------------
    # 4b. Búsqueda de Escala Óptima Hessian-Aware (Exp 25/26) y Base Q Sola
    # -------------------------------------------------------------
    def _find_optimal_group_params(
        self,
        target_group: torch.Tensor,
        wbits: int,
        H_group: torch.Tensor,
        grid_search_steps: int = 16,
        alpha_min: float = 0.70,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Búsqueda de escala óptima Hessian-Aware (Exp 25/26):
        Busca el factor alpha in [alpha_min, 1.05] que minimiza el error cuadrático
        ponderado por la sub-Hessiana local del grupo diag(H_group).
        """
        dev = target_group.device
        maxq = torch.tensor(2 ** wbits - 1, device=dev)

        tmp = torch.zeros(target_group.shape[0], device=dev)
        xmin = torch.minimum(target_group.min(1)[0], tmp)
        xmax = torch.maximum(target_group.max(1)[0], tmp)

        tmp_mask = (xmin == 0) & (xmax == 0)
        xmin[tmp_mask] = -1.0
        xmax[tmp_mask] = +1.0

        base_scale = (xmax - xmin) / maxq
        base_zero = torch.round(-xmin / base_scale)

        best_scale = base_scale.clone()
        best_zero = base_zero.clone()

        diag_h = torch.diag(H_group).unsqueeze(0)  # (1, G)
        best_loss = torch.full((target_group.shape[0],), float("inf"), device=dev)

        alpha_candidates = torch.linspace(alpha_min, 1.05, grid_search_steps, device=dev)

        for alpha in alpha_candidates:
            cur_xmin = alpha * xmin
            cur_xmax = alpha * xmax
            cur_scale = (cur_xmax - cur_xmin) / maxq
            cur_zero = torch.round(-cur_xmin / cur_scale)

            q_candidate = quantize(target_group, cur_scale.unsqueeze(1), cur_zero.unsqueeze(1), maxq)
            err = (target_group - q_candidate) ** 2
            loss = torch.sum(err * diag_h, dim=1)

            improved = loss < best_loss
            if torch.any(improved):
                best_loss[improved] = loss[improved]
                best_scale[improved] = cur_scale[improved]
                best_zero[improved] = cur_zero[improved]

        return best_scale, best_zero, maxq

    def run_admmq(self, wbits: int = 3, groupsize: int = 256, percdamp: float = 0.01,
                  scale: str = "admm", alpha_min: float = 0.70) -> Dict[str, Any]:
        """Q sola con ADMM-Q (Lucas et al., COLM 2026), código oficial en external/admm-q.
        scale="hessian" reemplaza su escala inicial (MSE sobre W) por la búsqueda ponderada
        por diag(H) de run_triad_quant_only, sin tocar el archivo original."""
        # ADMM-Q exige que g divida las columnas; con g > n queda un grupo por fila, como en GPTQ.
        groupsize = min(groupsize, self.columns)
        n_groups = -(-self.columns // groupsize)
        admm_dir = str(Path(__file__).resolve().parent / "external" / "admm-q")
        if admm_dir not in sys.path:
            sys.path.insert(0, admm_dir)
        import solvers  # type: ignore
        original = solvers.compute_scale_zero
        if scale == "hessian":
            def hessian_scale(W, maxq, sym=False, mse=True, groupsize=-1, **kw):
                sc, zr = torch.empty_like(W), torch.empty_like(W)
                for g0 in range(0, W.shape[1], groupsize):
                    g1 = g0 + groupsize
                    s_g, z_g, _ = self._find_optimal_group_params(
                        W[:, g0:g1], wbits, self.H[g0:g1, g0:g1], alpha_min=alpha_min,
                        grid_search_steps=round(16 * (1.05 - alpha_min) / 0.35))
                    sc[:, g0:g1], zr[:, g0:g1] = s_g.unsqueeze(1), z_g.unsqueeze(1)
                return sc, zr
            solvers.compute_scale_zero = hessian_scale
        # Guarda la última grilla (escala, cero) con la forma de W para recuperar los índices enteros.
        original_fq, grid = solvers.fake_quant, {}
        def recording_fq(x, sc, zr, maxq, sym):
            if sc.shape == self.W_orig.shape:
                grid["scale"], grid["zero"] = sc, zr
            return original_fq(x, sc, zr, maxq, sym)
        solvers.fake_quant = recording_fq
        # hipsolver falla en eigh de 8192x8192 (fc2 de OPT-1.3B); MAGMA no. Las chicas quedan como estaban.
        prev_lib = torch.backends.cuda.preferred_linalg_library()
        if self.columns > 4096:
            torch.backends.cuda.preferred_linalg_library("magma")
        try:
            admm_fn = solvers.ADMM_quantize
            if self.columns > 4096:
                # La reconstrucción de XTX_inv en fp64 domina el tiempo (esta GPU es ~3x más lenta en fp64).
                # Se reemplaza por fp32 al ejecutar, sin tocar el archivo de terceros. Validado en OPT-1.3B
                # (fc2 de los bloques 0 y 12): mismo error funcional, ~3x más rápido.
                import inspect
                # La eigh en fp64 de 8192x8192 se hace en CPU: en GPU dio OOM tras horas de corrida
                # (fragmentación); en CPU tarda ~20 s y no ocupa VRAM. Solo se guarda Q en fp32.
                old_eig = "L, Q = torch.linalg.eigh(XtX.double())"
                new_eig = ("L, Q = torch.linalg.eigh(XtX.double().cpu()); "
                           "L = L.to(XtX.device); Q = Q.to(device=XtX.device, dtype=torch.float32)")
                old_inv = "XTX_inv = (Q @ ((1 / (L + rho)) * Q).T).float()"
                new_inv = "XTX_inv = (Q @ ((1 / (L + rho)).float() * Q).T)"
                src = inspect.getsource(solvers.ADMM_quantize)
                assert src.count(old_inv) == 2 and src.count(old_eig) == 1, "cambió el código de ADMM-Q"
                ns = dict(vars(solvers))
                src = src.replace(old_eig, new_eig).replace(old_inv, new_inv)
                exec(compile(src, "admm_q_fp32_inverse", "exec"), ns)
                torch.cuda.empty_cache()
                admm_fn = ns["ADMM_quantize"]
            Q_res, _ = admm_fn(self.W_orig.clone().float(), self.H, maxq=2 ** wbits - 1,
                                             sym=False, percdamp=percdamp, groupsize=groupsize,
                                             verbose=False)
        finally:
            solvers.compute_scale_zero = original
            solvers.fake_quant = original_fq
            torch.backends.cuda.preferred_linalg_library(prev_lib)
        Q_idx = Q_zero = None
        if grid:
            sc, zr = grid["scale"].to(Q_res), grid["zero"].to(Q_res)
            Q_idx = torch.round(Q_res / sc)
            if (sc * Q_idx - Q_res).abs().max() <= 1e-3 * sc.abs().max():
                Q_zero = zr
            else:
                Q_idx = None
        return {
            "mode": f"ADMMQ_{wbits}bit_g{groupsize}",
            "wbits": wbits,
            "groupsize": groupsize,
            "effective_bits": float(wbits + 32.0 * n_groups / self.columns),
            "functional_error": self.evaluate_functional_error(Q_res),
            "W_approx": Q_res,
            "Q_idx": Q_idx,
            "Q_zero": Q_zero,
        }

    def run_triad_quant_only(
        self,
        wbits: int = 3,
        groupsize: int = 64,
        percdamp: float = 0.01,
        optimize_scale: bool = True,
        blocksize: int = 128,
        alpha_min: float = 0.70
    ) -> Dict[str, Any]:
        """
        Ejecuta exclusivamente la componente de cuantización de la Tríada (Base Q sola),
        sin componentes de bajo rango (r=0) ni esparsidad (s=0), con groupsize=64
        y búsqueda de escala óptima Hessian-aware (Exp 25/26).
        """
        W = self.W_orig.clone().float()
        n = self.columns
        L_H, _ = self.get_cholesky(percdamp=percdamp)
        H_inv_full = torch.cholesky_inverse(L_H)
        H_inv_upper = torch.linalg.cholesky(H_inv_full, upper=True)

        Q_res = torch.zeros_like(W)
        Idx_res = torch.zeros_like(W)  # índice relativo al punto cero
        Zero_res = torch.zeros_like(W)
        quantizer = Quantizer()
        quantizer.configure(wbits, perchannel=True, sym=False, mse=False)

        current_scale = None
        current_zero = None
        current_maxq = torch.tensor(2 ** wbits - 1, device=self.dev)

        for i1 in range(0, n, blocksize):
            i2 = min(i1 + blocksize, n)
            count = i2 - i1
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = H_inv_upper[i1:i2, i1:i2]
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                col_idx = i1 + i
                if groupsize != -1 and col_idx % groupsize == 0:
                    grp_end = min(col_idx + groupsize, n)
                    grp_W = W[:, col_idx:grp_end]
                    grp_H = self.H[col_idx:grp_end, col_idx:grp_end]
                    if optimize_scale:
                        current_scale, current_zero, current_maxq = self._find_optimal_group_params(
                            grp_W, wbits, grp_H, alpha_min=alpha_min,
                            grid_search_steps=round(16 * (1.05 - alpha_min) / 0.35)
                        )
                    else:
                        quantizer.find_params(grp_W, weight=True)
                        current_scale = quantizer.scale
                        current_zero = quantizer.zero
                        current_maxq = quantizer.maxq

                q = quantize(
                    w.unsqueeze(1),
                    current_scale.unsqueeze(1) if current_scale.ndim == 1 else current_scale,
                    current_zero.unsqueeze(1) if current_zero.ndim == 1 else current_zero,
                    current_maxq
                ).flatten()
                Q1[:, i] = q
                Idx_res[:, col_idx] = torch.round(q / current_scale.flatten())
                Zero_res[:, col_idx] = current_zero.flatten()
                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1
            Q_res[:, i1:i2] = Q1
            W[:, i2:] -= Err1.matmul(H_inv_upper[i1:i2, i2:])

        err = self.evaluate_functional_error(Q_res)
        overhead = 32.0 / groupsize if groupsize > 0 else 0.0
        eff_bits = float(wbits + overhead)

        return {
            "mode": f"Triad_QuantOnly_{wbits}bit_g{groupsize}",
            "wbits": wbits,
            "groupsize": groupsize,
            "effective_bits": eff_bits,
            "functional_error": err,
            "W_approx": Q_res,
            "Q_idx": Idx_res,
            "Q_zero": Zero_res
        }

    # -------------------------------------------------------------
    # 5. Tríada Conjunta Q + L + S (Tesis: Closed-Loop BCD)
    # -------------------------------------------------------------
    def run_qls_triad(
        self,
        wbits: int = 3,
        rank: int = 16,
        sparsity: float = 0.01,
        groupsize: int = 64,
        cycles: int = 2,
        percdamp: float = 0.01,
        sym: bool = False,
        optimize_scale: bool = True,
        l_bits: int = 16,
        scale_dim_l: str = "col",
        l_quant: str = "minmax",
        s_struct: str = "unstructured",
        s_val_bits: int = 16,
        s_index: str = "csr",
        s_cols: int = 0
    ) -> Dict[str, Any]:
        """
        Ejecuta la Tríada Conjunta Q + L + S en bucle cerrado (Closed-Loop BCD)
        con búsqueda de escala óptima Hessian-Aware (Exp 25/26).

        Soporta cuantización de factores L = U V^T a l_bits (e.g. 4 u 8 bits).
          scale_dim_l: "col" | "row" | "mixed" (U por columna, V por fila).
          l_quant:     "minmax" (comportamiento histórico) | "hessian"
                       (clipping Hessian-aware + re-target de V, ver
                       quantize_lowrank_factors).

        Componente S:
          s_struct:   "unstructured" (top-k por |w|*sqrt(H_jj), densidad `sparsity`)
                      | "column" (s_cols canales de entrada completos, elegidos por
                      H_jj * ||residuo_:j||^2, al estilo OWQ).
          s_val_bits: bits de los valores de S (16 = FP16 histórico, 8 = int8).
          s_index:    "csr" (histórico) | "rowcount" (conteo por fila en vez de punteros).
          Los valores por defecto reproducen exactamente la Tríada anterior.
        """
        if s_struct not in ("unstructured", "column"):
            raise ValueError(f"s_struct desconocido: {s_struct}")
        has_S = (s_cols > 0) if s_struct == "column" else (sparsity > 0.0)
        last_lq = {}  # diagnóstico: error funcional de cuantizar L en el último paso
        W_orig = self.W_orig.clone().float()
        m, n = self.rows, self.columns
        L_H, L_H_inv = self.get_cholesky(percdamp=percdamp)
        H_inv_full = torch.cholesky_inverse(L_H)
        H_inv_upper = torch.linalg.cholesky(H_inv_full, upper=True)
        diag_H_sqrt = torch.sqrt(torch.diag(self.H)).unsqueeze(0)

        # Helper para paso de rango bajo (con cuantización opcional a l_bits)
        def step_lowrank(target, r):
            if r <= 0:
                return torch.zeros_like(target)
            M = target @ L_H
            U, s, Vh = torch.linalg.svd(M, full_matrices=False)
            r_cur = min(r, s.shape[0])
            sqrt_s = torch.sqrt(s[:r_cur])
            U_cont = U[:, :r_cur] * sqrt_s.unsqueeze(0)
            V_cont = (Vh[:r_cur, :] @ L_H_inv).t() * sqrt_s.unsqueeze(0)

            if l_bits >= 16:
                return U_cont @ V_cont.t()
            else:
                U_q, V_q = quantize_lowrank_factors(
                    U_cont, V_cont, self.H, l_bits,
                    scale_dim=scale_dim_l, method=l_quant,
                )
                L_c = U_cont @ V_cont.t()
                L_q = U_q @ V_q.t()
                dL = L_c - L_q
                last_lq["loss_quant_L"] = torch.trace(dL @ self.H @ dL.t()).item() / m
                last_lq["energy_L"] = torch.trace(L_c @ self.H @ L_c.t()).item() / m
                return L_q

        # Helper para paso de outliers
        def step_sparse(target, sp):
            if not has_S:
                return torch.zeros_like(target)
            S = torch.zeros_like(target)
            if s_struct == "column":
                score = torch.diag(self.H) * (target * target).sum(dim=0)  # (n,)
                cols = torch.topk(score, min(s_cols, n), largest=True)[1]
                S[:, cols] = quantize_sparse_columns(target[:, cols], s_val_bits)
                return S
            k = int(m * n * sp)
            if k <= 0:
                return S
            sal = torch.abs(target) * diag_H_sqrt
            thresh = torch.topk(sal.flatten(), k, largest=True)[0][-1]
            mask = sal >= thresh
            S[mask] = target[mask]
            if s_val_bits < 16:
                S = quantize_sparse_unstructured(S, mask, self.H, s_val_bits)
            return S

        # Helper para paso de cuantización GPTQ sobre residuo
        def step_quantize(target):
            if wbits >= 16:
                return target.clone()
            quantizer = Quantizer()
            quantizer.configure(wbits, perchannel=True, sym=sym, mse=False)
            W_t = target.clone().float()
            Q_res = torch.zeros_like(W_t)
            blocksize = 128
            cur_scale = None
            cur_zero = None
            cur_maxq = torch.tensor(2 ** wbits - 1, device=self.dev)

            for i1 in range(0, n, blocksize):
                i2 = min(i1 + blocksize, n)
                count = i2 - i1
                W1 = W_t[:, i1:i2].clone()
                Q1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1)
                Hinv1 = H_inv_upper[i1:i2, i1:i2]
                for i in range(count):
                    w = W1[:, i]
                    d = Hinv1[i, i]
                    col_idx = i1 + i
                    if groupsize != -1 and col_idx % groupsize == 0:
                        grp_end = min(col_idx + groupsize, n)
                        grp_W = W_t[:, col_idx:grp_end]
                        grp_H = self.H[col_idx:grp_end, col_idx:grp_end]
                        if optimize_scale:
                            cur_scale, cur_zero, cur_maxq = self._find_optimal_group_params(
                                grp_W, wbits, grp_H, grid_search_steps=16
                            )
                        else:
                            quantizer.find_params(grp_W, weight=True)
                            cur_scale = quantizer.scale
                            cur_zero = quantizer.zero
                            cur_maxq = quantizer.maxq

                    q = quantize(
                        w.unsqueeze(1),
                        cur_scale.unsqueeze(1) if cur_scale.ndim == 1 else cur_scale,
                        cur_zero.unsqueeze(1) if cur_zero.ndim == 1 else cur_zero,
                        cur_maxq
                    ).flatten()
                    Q1[:, i] = q
                    err1 = (w - q) / d
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                    Err1[:, i] = err1
                Q_res[:, i1:i2] = Q1
                W_t[:, i2:] -= Err1.matmul(H_inv_upper[i1:i2, i2:])
            return Q_res

        # 1. Outlier inicial
        S_mat = step_sparse(W_orig, sparsity)
        # 2. Rango bajo inicial
        L_mat = step_lowrank(W_orig - S_mat, rank)
        # 3. Cuantización residual inicial
        Q_mat = step_quantize(W_orig - S_mat - L_mat)

        # 4. Ciclos BCD cerrados
        for _ in range(cycles):
            L_mat = step_lowrank(W_orig - S_mat - Q_mat, rank)
            if has_S:
                S_mat = step_sparse(W_orig - L_mat - Q_mat, sparsity)
                L_mat = step_lowrank(W_orig - S_mat - Q_mat, rank)
            Q_mat = step_quantize(W_orig - S_mat - L_mat)

        if l_bits >= 16:
            # Re-ajuste continuo de L al final si no agrega ruido de cuantización
            L_mat = step_lowrank(W_orig - S_mat - Q_mat, rank)

        W_reconstructed = S_mat + L_mat + Q_mat
        err = self.evaluate_functional_error(W_reconstructed)

        overhead_q = (16.0 / groupsize if sym else 32.0 / groupsize) if groupsize > 0 else 0.0

        # Bits de L ("col" y "row" dan exactamente lo mismo que antes; se agrega "mixed")
        bits_l = lowrank_bits(m, n, rank, l_bits, scale_dim_l) / (m * n)

        # Bits de S. Con los defaults (unstructured, 16 bits, csr) es idéntico al cálculo previo.
        sb = sparse_bits(m, n, S_mat, s_struct, s_val_bits, s_index, s_cols)
        k_sparse = sb["k"]
        bits_s_csr = sb["total"] / (m * n)
        if s_struct == "unstructured" and k_sparse > 0:
            bits_s_bmp = (m * n * 1.0 + sb["values"]) / (m * n)
        else:
            bits_s_bmp = bits_s_csr

        eff_bits_csr = float(wbits + overhead_q + bits_l + bits_s_csr)
        eff_bits_bitmap = float(wbits + overhead_q + bits_l + bits_s_bmp)

        return {
            "mode": f"QLS_w{wbits}_r{rank}_s{int(sparsity*1000)}_l{l_bits}",
            "wbits": wbits,
            "rank": rank,
            "sparsity": sparsity,
            "l_bits": l_bits,
            "scale_dim_l": scale_dim_l,
            "l_quant": l_quant,
            "s_struct": s_struct,
            "s_val_bits": s_val_bits,
            "s_index": s_index,
            "s_cols": s_cols,
            "effective_bits": eff_bits_csr,
            "effective_bits_csr": eff_bits_csr,
            "effective_bits_bitmap": eff_bits_bitmap,
            "functional_error": err,
            "diagnostics": {
                "triad_loss_quant_L": last_lq.get("loss_quant_L"),
                "triad_energy_L": last_lq.get("energy_L"),
                "triad_bits_L": bits_l,
                "triad_bits_S": bits_s_csr,
                "triad_bits_S_values": sb["values"] / (m * n),
                "triad_bits_S_index": sb["index"] / (m * n),
                "triad_bits_S_entropy_bound": sb["entropy_bound"] / (m * n),
                "triad_k_sparse": k_sparse,
            },
            "W_approx": W_reconstructed
        }


    # -------------------------------------------------------------
    # 6. Asignador por matriz v0: elige entre Q sola y variantes de la Tríada
    # -------------------------------------------------------------
    def run_triad_alloc(
        self,
        wbits: int = 3,
        groupsize: int = 64,
        candidates: str = "Q+S",
        **triad_kwargs,
    ) -> Dict[str, Any]:
        """
        Para esta matriz:
          1. Corre Q sola a wbits y wbits+1 (triad_quant_only): línea de referencia.
          2. Corre cada candidata de `candidates` ("Q+S", "Q+L", "Q+L+S", separadas
             por coma) con run_qls_triad y los triad_kwargs dados.
          3. Calcula Δbits = b_eq(err) - bits, donde b_eq interpola log-linealmente
             el error de Q sola entre wbits y wbits+1 (igual que test_components.py).
          4. Devuelve la candidata con mayor Δbits, o Q sola si ninguna es positiva.
        """
        q_over = 32.0 / groupsize if groupsize > 0 else 0.0
        res_q = self.run_triad_quant_only(wbits=wbits, groupsize=groupsize)
        err_next = self.run_triad_quant_only(wbits=wbits + 1, groupsize=groupsize)["functional_error"]
        lw = math.log(max(res_q["functional_error"], 1e-30))
        slope = lw - math.log(max(err_next, 1e-30))

        def delta(err, bits):
            if slope <= 0:
                return float("-inf")
            return (wbits + q_over) + (lw - math.log(max(err, 1e-30))) / slope - bits

        best_name, best_res, best_delta = "Q", res_q, 0.0
        tried = {}
        for name in [c.strip() for c in candidates.split(",") if c.strip()]:
            kw = dict(triad_kwargs)
            if "L" not in name:
                kw["rank"] = 0
            if "S" not in name:
                kw["sparsity"] = 0.0
                kw["s_cols"] = 0
            res = self.run_qls_triad(wbits=wbits, groupsize=groupsize, **kw)
            d = delta(res["functional_error"], res["effective_bits_csr"])
            tried[name] = d
            if d > best_delta:
                best_name, best_res, best_delta = name, res, d

        if best_name == "Q":
            bits = float(wbits + q_over)
            out = {
                "effective_bits": bits,
                "effective_bits_csr": bits,
                "effective_bits_bitmap": bits,
                "functional_error": res_q["functional_error"],
                "W_approx": res_q["W_approx"],
            }
        else:
            out = {
                "effective_bits": best_res["effective_bits_csr"],
                "effective_bits_csr": best_res["effective_bits_csr"],
                "effective_bits_bitmap": best_res["effective_bits_bitmap"],
                "functional_error": best_res["functional_error"],
                "W_approx": best_res["W_approx"],
            }
        out["mode"] = f"TriadAlloc_w{wbits}_{best_name}"
        out["diagnostics"] = {
            "alloc_choice": best_name,
            "alloc_delta_bits": best_delta,
            **{f"alloc_delta_{k}": v for k, v in tried.items()},
        }
        return out