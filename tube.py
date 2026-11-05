#!/usr/bin/env python3
"""
Descomposición Q + L + S por tubo: L y S no aproximan W, preparan el residuo para Q.

Con T = W - L - S, intervalo [lo, hi] por fila y grupo, y pesos p_j = r_jj^2 (pivotes
de GPTQ: H^{-1} = U^T U, r_jj = 1 / U_jj), el error de cuantizar T se modela como

    D = sum_{i,g} [ A_g (hi - lo)^2 / (12 (2^b - 1)^2)
                    + sum_{j in g} p_j ((T_ij - hi)_+^2 + (lo - T_ij)_+^2) ]

con A_g = sum_{j in g} p_j. Las entradas dentro del tubo no pagan nada extra.

    min_{U, V, S, lo, hi}  D + lambda ||S||_0

S se elige por exceso ponderado p_j * exceso^2 y U, V, lo, hi por descenso de gradiente.
El presupuesto de S se fija con `sparsity`, que equivale a fijar lambda. Al final se
cuantizan L y S, y GPTQ corre último sobre W - L - S: su error absorbe el de los otros dos.
"""
from __future__ import annotations
import math
from typing import Any, Dict, List, Tuple

import torch

from engine import lowrank_bits, quantize_lowrank_factors, quantize_sparse_unstructured, sparse_bits


def group_extremes(T: torch.Tensor, g: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mínimo y máximo por fila y grupo, incluyendo siempre el cero."""
    lo, hi = [], []
    for start in range(0, T.shape[1], g):
        part = T[:, start:start + g]
        lo.append(part.min(1).values.clamp(max=0))
        hi.append(part.max(1).values.clamp(min=0))
    return torch.stack(lo, 1), torch.stack(hi, 1)


def excess(T: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, grp: torch.Tensor) -> torch.Tensor:
    """Distancia de cada entrada al tubo (cero adentro)."""
    return (T - hi[:, grp]).clamp(min=0) + (lo[:, grp] - T).clamp(min=0)


def select_sparse(T, lo, hi, grp, p, k: int) -> torch.Tensor:
    """Máscara con las k entradas de mayor exceso ponderado. Puede quedar con menos."""
    if k <= 0:
        return torch.zeros_like(T, dtype=torch.bool)
    score = p.unsqueeze(0) * excess(T, lo, hi, grp).pow(2)
    thresh = torch.topk(score.flatten(), k).values[-1]
    return (score >= thresh) & (score > 0)


def tube_loss(T, lo, hi, grp, p, A, width_coef, mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Devuelve (costo del ancho, costo de lo que se sale), ambos por fila."""
    m = T.shape[0]
    width = (width_coef * A.unsqueeze(0) * (hi - lo).pow(2)).sum() / m
    out = p.unsqueeze(0) * excess(T, lo, hi, grp).pow(2)
    if mask is not None:
        out = out.masked_fill(mask, 0.0)
    return width, out.sum() / m


def tube_compress(
    eng,
    wbits: int = 3,
    rank: int = 16,
    sparsity: float = 0.005,
    groupsize: int = 64,
    l_bits: int = 3,
    s_val_bits: int = 8,
    s_index: str = "rowcount",
    scale_dim_l: str = "col",
    rounds: int = 3,
    steps: int = 150,
    lr: float = 3e-3,
    alpha_min: float = 0.40,
    init: str = "hessian",
    quantizer: str = "gptq",
) -> Dict[str, Any]:
    W = eng.W_orig.clone().float()
    m, n = eng.rows, eng.columns
    dev = W.device

    L_H, L_H_inv = eng.get_cholesky()
    U_h = torch.linalg.cholesky(torch.cholesky_inverse(L_H), upper=True)
    p = torch.diag(U_h).pow(-2)
    p = p / p.mean()
    grp = torch.arange(n, device=dev) // groupsize
    A = torch.zeros(int(grp.max()) + 1, device=dev).index_add(0, grp, p)
    width_coef = 1.0 / (12 * (2 ** wbits - 1) ** 2)

    sigma = W.std().clamp(min=1e-12)
    Wn = W / sigma

    U, V = None, None
    if rank > 0:
        if init == "hessian":
            u, s, vh = torch.linalg.svd(Wn @ L_H, full_matrices=False)
            V = vh[:rank].t().mT @ L_H_inv
            V = V.t()
        elif init == "pivot":
            r_diag = p.sqrt()
            u, s, vh = torch.linalg.svd(Wn * r_diag.unsqueeze(0), full_matrices=False)
            V = (vh[:rank] / r_diag.unsqueeze(0)).t()
        else:
            raise ValueError(f"init desconocido: {init}")
        U = u[:, :rank] * s[:rank].unsqueeze(0)
        # Equilibrar factores para que un único lr sirva a ambos
        gamma = (V.norm(dim=0) / U.norm(dim=0).clamp(min=1e-12)).sqrt()
        U, V = U * gamma.unsqueeze(0), V / gamma.unsqueeze(0)

    def residual(u_mat, v_mat):
        return Wn if u_mat is None else Wn - u_mat @ v_mat.t()

    T = residual(U, V)
    lo, hi = group_extremes(T, groupsize)
    lo, hi = 0.85 * lo, 0.85 * hi
    k = int(sparsity * m * n)
    mask = select_sparse(T, lo, hi, grp, p, k)

    width0, out0 = tube_loss(T, lo, hi, grp, p, A, width_coef)
    history: List[float] = [(width0 + out0).item()]

    params = [lo.requires_grad_(), hi.requires_grad_()]
    if U is not None:
        params += [U.requires_grad_(), V.requires_grad_()]

    with torch.enable_grad():
        for _ in range(rounds):
            opt = torch.optim.Adam(params, lr=lr)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
            for _ in range(steps):
                opt.zero_grad()
                T = residual(U, V)
                width, out = tube_loss(T, lo, hi, grp, p, A, width_coef, mask)
                (width + out).backward()
                opt.step()
                sched.step()
                with torch.no_grad():
                    lo.clamp_(max=0)
                    hi.clamp_(min=0)
            with torch.no_grad():
                T = residual(U, V)
                mask = select_sparse(T, lo, hi, grp, p, k)
                width, out = tube_loss(T, lo, hi, grp, p, A, width_coef, mask)
                history.append((width + out).item())

    with torch.no_grad():
        lo, hi = lo.detach(), hi.detach()
        L = torch.zeros_like(W)
        if U is not None:
            root = sigma.sqrt()
            U_c, V_c = U.detach() * root, V.detach() * root
            if l_bits >= 16:
                L = U_c @ V_c.t()
            else:
                U_q, V_q = quantize_lowrank_factors(U_c, V_c, eng.H, l_bits, scale_dim=scale_dim_l,
                                                    method="hessian")
                L = U_q @ V_q.t()

        # S se decide después de cuantizar L, para que vea el residuo real
        S = torch.zeros_like(W)
        if k > 0:
            T = (W - L) / sigma
            mask = select_sparse(T, lo, hi, grp, p, k)
            S[mask] = (W - L)[mask]
            if s_val_bits < 16:
                S = quantize_sparse_unstructured(S, mask, eng.H, s_val_bits)

        target = W - L - S
        rng0 = group_extremes(W, groupsize)
        rng1 = group_extremes(target, groupsize)
        range_ratio = ((rng1[1] - rng1[0]).mean() / (rng0[1] - rng0[0]).mean()).item()

        eng.W_orig = target
        try:
            if quantizer.startswith("admm"):
                res = eng.run_admmq(wbits=wbits, groupsize=groupsize, alpha_min=alpha_min,
                                    scale="hessian" if quantizer == "admm_h" else "admm")
            else:
                res = eng.run_triad_quant_only(wbits=wbits, groupsize=groupsize, alpha_min=alpha_min)
        finally:
            eng.W_orig = W
        W_approx = res["W_approx"] + L + S
        err = eng.evaluate_functional_error(W_approx)

        bits_l = lowrank_bits(m, n, rank, l_bits, scale_dim_l) / (m * n)
        sb = sparse_bits(m, n, S, "unstructured", s_val_bits, s_index, 0)
        bits = wbits + 32.0 / groupsize + bits_l + sb["total"] / (m * n)

    return {
        "functional_error": err,
        "bits": float(bits),
        "effective_bits": float(bits),
        "effective_bits_csr": float(bits),
        "bits_L": bits_l,
        "bits_S": sb["total"] / (m * n),
        "k_sparse": sb["k"],
        "range_ratio": range_ratio,
        "model_history": history,
        "W_approx": W_approx,
        "Q_idx": res.get("Q_idx"),
        "Q_zero": res.get("Q_zero"),
        "groupsize": groupsize,
        "wbits": wbits,
    }
