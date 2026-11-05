"""
SRR (arXiv 2602.02001) reimplementado en el protocolo propio: Q + L con L elegida en dos pasos.

Sigue el código oficial (github.com/cyoonjun/srr, commit 46b245d, config iter=1), que difiere
del Algoritmo 1 del paper:
  1. P = SVD_k*(W S) S^-1, con S = Cholesky de H (su modo "cholesky", QERA exacto).
  2. Q = cuantizar(W - P) con el mismo cuantizador final que el tubo (ADMM-Q o GPTQ).
  3. L = SVD_r((W - Q) S) S^-1 con rango r completo. P se descarta: solo se guardan Q y L.

k* = argmin_k rho_k(W S) * rho_{r-k}(E S) (find_optimal_k), con rho_k(M) = ||M - SVD_k(M)||^2/||M||^2
y E uniforme de n x n (n = columnas de entrada), una sola sonda.
L se guarda en 16 bits como en su código (l_bits=16) y se cobra entera: 16 r (m+n) / (m n).
Cuantizarla a 3 bits como en el tubo la arruina: L va después de Q y nada absorbe su error
(k_proj del bloque 3 de OPT-125M, 3 bits, r=16: error funcional 18.3 con L en 16 bits, 181 en 4 y 608 en
3 bits, 29.0 con Q sola; results/srr_diagnostics.json). A igual costo que L+S del tubo: r=4 en 16 bits.
"""
from __future__ import annotations
from typing import Any, Dict

import torch

from engine import lowrank_bits, quantize_lowrank_factors


def _tail_ratio(sv: torch.Tensor) -> torch.Tensor:
    """rho_k para k = 0..len(sv): energía que queda fuera de las k primeras componentes."""
    e = sv.double() ** 2
    tail = e.sum() - torch.cat([e.new_zeros(1), e.cumsum(0)])
    return (tail / e.sum()).float()


def _lowrank_q(M_S: torch.Tensor, L_H_inv: torch.Tensor, k: int, H, l_bits: int, scale_dim: str):
    if k <= 0:
        return torch.zeros(M_S.shape[0], L_H_inv.shape[0], device=M_S.device)
    u, s, vh = torch.linalg.svd(M_S, full_matrices=False)
    U = u[:, :k] * s[:k].sqrt()
    V = (s[:k].sqrt().unsqueeze(1) * vh[:k] @ L_H_inv).t()
    if l_bits >= 16:
        return U @ V.t()
    U_q, V_q = quantize_lowrank_factors(U, V, H, l_bits, scale_dim=scale_dim, method="hessian")
    return U_q @ V_q.t()


def srr_compress(eng, wbits: int = 3, rank: int = 16, groupsize: int = 256, l_bits: int = 16,
                 scale_dim_l: str = "col", alpha_min: float = 0.70, quantizer: str = "admm_h",
                 seed: int = 0) -> Dict[str, Any]:
    W = eng.W_orig.clone().float()
    m, n = eng.rows, eng.columns
    L_H, L_H_inv = eng.get_cholesky()

    WS = W @ L_H
    gen = torch.Generator(device=W.device).manual_seed(seed)
    E = torch.rand((n, n), generator=gen, device=W.device) - 0.5
    rw = _tail_ratio(torch.linalg.svdvals(WS))[: rank + 1]
    re = _tail_ratio(torch.linalg.svdvals(E @ L_H))[: rank + 1]
    score = torch.stack([rw[k] * re[rank - k] for k in range(rank + 1)])
    k = int(score.argmin())

    P = _lowrank_q(WS, L_H_inv, k, eng.H, 16, scale_dim_l)
    eng.W_orig = W - P
    try:
        if quantizer.startswith("admm"):
            res = eng.run_admmq(wbits=wbits, groupsize=groupsize, alpha_min=alpha_min,
                                scale="hessian" if quantizer == "admm_h" else "admm")
        else:
            res = eng.run_triad_quant_only(wbits=wbits, groupsize=groupsize, alpha_min=alpha_min)
    finally:
        eng.W_orig = W
    Q = res["W_approx"].float()
    L = _lowrank_q((W - Q) @ L_H, L_H_inv, rank, eng.H, l_bits, scale_dim_l)

    W_approx = Q + L
    g_eff = min(groupsize, n)
    bits_l = lowrank_bits(m, n, rank, l_bits, scale_dim_l) / (m * n)
    bits = wbits + 32.0 * (-(-n // g_eff)) / n + bits_l
    return {
        "functional_error": eng.evaluate_functional_error(W_approx),
        "bits": float(bits), "effective_bits": float(bits), "effective_bits_csr": float(bits),
        "bits_L": bits_l, "srr_k": k, "W_approx": W_approx,
        "Q_idx": res.get("Q_idx"), "Q_zero": res.get("Q_zero"), "groupsize": groupsize, "wbits": wbits,
    }
