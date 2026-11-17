"""Multi-head scaled dot-product retrieval over the memory levels.

Eq. (25) per-level attention alpha_k^l, Eq. (26) level-weighted retrieval
r = sum_l omega_l sum_k alpha_k^l m_k^l, Eq. (46) softmax(Q K^T / sqrt(d_k)) V.
8 heads, d_k = d / heads = 8; omega_l = softmax(W_omega q).

The same function is used at rollout (B = 1) and in replay reconstruction
(B = batch), so the two agree to float32 rounding.

Offset-free keys (memory.key_centering): a level may carry explicit attention keys K (the
centred, detached slot keys L_key of M_L / F_key of the A3 flat store); the logits are then
Q W_K K^T and the values stay the (projected) contents M. Without K (key_centering off, or
M_S) keys = contents.

Arrival keys (memory.key_source = "arrivals"): the query is the slot's
arrival key q_t = k_t (R^Z, `qkey`) and every level carries Z-dimensional slot keys; the
logits are cos(q_t, K_j) / key_temperature, ONE head on the key side broadcast to the H
heads, so that the value path (W_V), the head pooling and the level weights omega stay
exactly as under "candidate". W_Q and W_K are not used on that path.
"""
from __future__ import annotations

import math

import torch

from . import EPS


def safe_norm(x: torch.Tensor, dim=-1, keepdim: bool = False) -> torch.Tensor:
    """||x||_2 with the sum of squares clamped at EPS^2 before the sqrt, so that a zero
    vector (empty / padded slot) has a finite gradient (vector_norm has NaN there)."""
    return torch.sqrt(torch.clamp((x * x).sum(dim=dim, keepdim=keepdim), min=EPS * EPS))


def cos_rows(rows: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """cos(z, rows_k) for rows (..., n, d) and z (..., d); the same expression is used for
    the merge test, the "arrivals" retrieval logits, at rollout and in reconstruction, so
    that all of them agree to float32 rounding."""
    num = torch.einsum("...kd,...d->...k", rows, z)
    den = safe_norm(rows) * safe_norm(z)[..., None]
    return num / den


def attend(params, qin: torch.Tensor, levels: list, mean_pool: bool = False,
           qkey: torch.Tensor | None = None, key_tau: float | None = None):
    """Eqs. (25), (26), (46).

    qin    : (B, q_dim) float32 query input [S_sys; tau_bar].
    levels : list of length 2 ([M_S, M_L]) or 1 (A3 flat store). Each item is
             None (level absent: A1 / A2), (M, mask) or (M, mask, K): M (B, K, d) visible
             memory after the Eq. 42 projection, mask (B, K) bool of filled rows, K
             (B, K, d) attention keys or None (keys = M, no explicit keys). Values are
             always M.
    mean_pool : A4 -- uniform attention weights over the filled rows.
    qkey, key_tau : "arrivals" key source: qkey (B, Z) = the slot's arrival key; every level
             must then carry keys K (B, K, Z) and logits_j = cos(qkey, K_j) / key_tau,
             the same for every head. None (default) = the "candidate" path (Q W_K K^T).

    Returns r (B, d), overall (list of (B, K) tensors or None): the overall weight
    omega_l * mean_heads(alpha) of every entry (used for the top-1 entry), and
    present (B, L) bool.

    Level weights: with two levels, omega = softmax(W_omega q) over the levels that
    have at least one filled row (a level that is empty or absent is dropped and the
    remaining level is used alone). With one level (A3) omega = 1. If no level has a
    filled row, r = 0.
    """
    B = qin.shape[0]
    H, dk, d = params.n_heads, params.d_k, params.d
    Q = params.query(qin).view(B, H, dk)
    scale = 1.0 / math.sqrt(dk)
    # Algebraically identical, cheaper form (W_K, W_V have no bias):
    #   Q_h . K_{h,k} = (W_K,h^T Q_h) . m_k ;  sum_k a_hk V_{h,k} = W_V,h (sum_k a_hk m_k)
    WK = params.W_K.weight.view(H, dk, d)
    WV = params.W_V.weight.view(H, dk, d)
    qk = torch.einsum("bhe,hed->bhd", Q, WK) * scale                # (B, H, d)
    r_lv, a_lv, present = [], [], []
    for lv in levels:
        if lv is None:
            r_lv.append(None)
            a_lv.append(None)
            present.append(torch.zeros(B, dtype=torch.bool, device=qin.device))
            continue
        if len(lv) == 2:
            (M, mask), Kmat = lv, None
        else:
            M, mask, Kmat = lv
        K_n = M.shape[1]
        has = mask.any(dim=1)                                        # (B,)
        if mean_pool:
            cnt = mask.sum(dim=1, keepdim=True).clamp(min=1).to(M.dtype)
            a = (mask.to(M.dtype) / cnt)[:, None, :].expand(B, H, K_n)
        elif qkey is not None:                                       # "arrivals" key source
            if Kmat is None:
                raise ValueError("attend: key_source='arrivals' needs keys for every level")
            logits = (cos_rows(Kmat, qkey) / float(key_tau))[:, None, :].expand(B, H, K_n)
            logits = logits.masked_fill(~mask[:, None, :], float("-inf"))
        else:
            logits = torch.einsum("bhd,bkd->bhk", qk, M if Kmat is None else Kmat)
            logits = logits.masked_fill(~mask[:, None, :], float("-inf"))
        if not mean_pool:
            logits = torch.where(has[:, None, None], logits, torch.zeros_like(logits))
            a = torch.softmax(logits, dim=-1) * has[:, None, None].to(M.dtype)
        pooled = torch.einsum("bhk,bkd->bhd", a, M)                   # (B, H, d)
        r = torch.einsum("hed,bhd->bhe", WV, pooled).reshape(B, d)
        r_lv.append(r)
        a_lv.append(a.mean(dim=1))                                    # (B, K) head-averaged
        present.append(has)
    present_t = torch.stack(present, dim=1)                           # (B, L)
    if len(levels) == 1:
        omega = present_t.to(Q.dtype)
    else:
        logit_w = params.W_omega(qin)                                 # (B, 2)
        logit_w = logit_w.masked_fill(~present_t, float("-inf"))
        anyp = present_t.any(dim=1, keepdim=True)
        logit_w = torch.where(anyp, logit_w, torch.zeros_like(logit_w))
        omega = torch.softmax(logit_w, dim=-1) * anyp.to(Q.dtype)
    r_out = torch.zeros(B, d, dtype=Q.dtype, device=qin.device)
    overall = []
    for li, (r, a) in enumerate(zip(r_lv, a_lv)):
        if r is None:
            overall.append(None)
            continue
        r_out = r_out + omega[:, li:li + 1] * r
        overall.append(omega[:, li:li + 1] * a)
    return r_out, overall, present_t
