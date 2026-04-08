"""CKA loss functions for knowledge distillation.

CKALoss          — simple Frobenius-norm CKA on raw hidden states.
LinearCKALoss    — numerically stable linear CKA; avoids explicit N×N Gram matrix.
MultiHeadCKALoss — aggregates per-head Gram matrices before CKA; used for SAMD
                   attention alignment.
linear_cka_loss  — functional version of LinearCKALoss for use inside loops.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CKALoss(nn.Module):
    """Simple Frobenius-norm CKA loss between two hidden-state matrices.

    Computes 1 - CKA(SH, TH) using linear kernels. Both inputs are
    mean-centred column-wise. Returns a scalar in approximately [0, 1].
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, SH: torch.Tensor, TH: torch.Tensor) -> torch.Tensor:
        dS, dT = SH.size(-1), TH.size(-1)
        SH = SH.reshape(-1, dS).to(dtype=torch.float64)
        TH = TH.reshape(-1, dT).to(dtype=torch.float64, device=SH.device)

        SH = SH - SH.mean(0, keepdim=True)
        TH = TH - TH.mean(0, keepdim=True)

        num  = torch.norm(SH.t().matmul(TH), "fro")
        den1 = torch.norm(SH.t().matmul(SH), "fro") + self.eps
        den2 = torch.norm(TH.t().matmul(TH), "fro") + self.eps
        return 1.0 - num / torch.sqrt(den1 * den2)


class LinearCKALoss(nn.Module):
    """Numerically stable linear CKA loss.

    Uses ||X^T Y||_F^2 / (||X^T X||_F * ||Y^T Y||_F) which avoids computing
    an explicit N×N Gram matrix. Input shape: X, Y: [N, D]. Returns 1 - CKA.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        X = X.float() - X.float().mean(dim=0, keepdim=True)
        Y = Y.float() - Y.float().mean(dim=0, keepdim=True)
        num = (X.t().matmul(Y) ** 2).sum()
        den = torch.norm(X.t().matmul(X), p="fro") * torch.norm(Y.t().matmul(Y), p="fro")
        return 1.0 - num / (den + self.eps)


def linear_cka_loss(
    X: torch.Tensor,
    Y: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Functional version of LinearCKALoss, usable without a module instance."""
    X = X.float() - X.float().mean(dim=0, keepdim=True)
    Y = Y.float() - Y.float().mean(dim=0, keepdim=True)
    num = (X.t().matmul(Y) ** 2).sum()
    den = torch.norm(X.t().matmul(X), "fro") * torch.norm(Y.t().matmul(Y), "fro")
    return 1.0 - num / (den + eps)


def _center_gram(K: torch.Tensor, unbiased: bool = False) -> torch.Tensor:
    """Double-centre a Gram matrix (standard or unbiased Kornblith 2019 variant)."""
    if not unbiased:
        return K - K.mean(dim=0, keepdim=True) - K.mean(dim=1, keepdim=True) + K.mean()

    N = K.size(0)
    if N <= 2:
        return K - K.mean()
    K = K.clone()
    K.fill_diagonal_(0)
    means = K.sum(dim=0) / (N - 2)
    means = means - means.sum() / (2 * (N - 1))
    K = K - means.unsqueeze(0) - means.unsqueeze(1)
    K.fill_diagonal_(0)
    return K


def _remove_diag_and_renorm(att: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Zero the self-attention diagonal of [H, N, N] and re-normalise rows."""
    H, N, D = att.shape
    if D != N:
        return att
    eye = torch.eye(N, device=att.device, dtype=att.dtype).unsqueeze(0)
    att = att * (1.0 - eye)
    return att / att.sum(dim=-1, keepdim=True).clamp_min(eps)


def _attn_transform(
    att: torch.Tensor,
    transform: str = "clr",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Transform attention probabilities before Gram matrix construction.

    transform: "none" | "sqrt" | "log" | "clr" (centered log-ratio).
    "clr" is the recommended default for attention weights.
    """
    att = att.clamp_min(eps)
    if transform == "none":
        return att
    if transform == "sqrt":
        return torch.sqrt(att)
    if transform == "log":
        return torch.log(att)
    if transform == "clr":
        logp = torch.log(att)
        return logp - logp.mean(dim=-1, keepdim=True)
    raise ValueError(f"Unknown transform: '{transform}'. Choose from none/sqrt/log/clr.")


def _row_entropy(att: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Shannon entropy per row of an attention distribution [N, D]. Returns [N]."""
    p = att.clamp_min(eps)
    return -(p * torch.log(p)).sum(dim=-1)


def _select_queries_by_focus(
    att_mean: torch.Tensor,
    max_q: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Select up to max_q query indices by focus score (1 - normalised entropy).

    Tokens with sharper attention distributions are selected first.
    Returns a sorted 1-D index tensor.
    """
    N = att_mean.size(0)
    if max_q is None or max_q >= N:
        return torch.arange(N, device=att_mean.device)

    A = att_mean.clamp_min(eps)
    A = A / A.sum(dim=-1, keepdim=True).clamp_min(eps)
    focus = 1.0 - _row_entropy(A, eps=eps) / math.log(float(N) + 1e-12)
    return torch.sort(torch.topk(focus, k=max_q, largest=True).indices).values


class MultiHeadCKALoss(nn.Module):
    """CKA loss computed via multi-head aggregated Gram matrices.

    Aggregates per-head attention into a single Gram matrix before CKA:
        K = (1/H) * sum_h (X_h X_h^T)
        loss = 1 - CKA(K_centered, L_centered)

    Args:
        eps: Numerical stability constant.
        unbiased_center: Use unbiased Gram centering (Kornblith 2019).
        transform: Probability-space transform applied before Gram construction.
                   "clr" (centered log-ratio) is recommended.
        remove_diag: Zero out self-attention diagonal before Gram.
        use_fp64: Compute Gram matrices in float64 for numerical stability.
    """

    def __init__(
        self,
        eps: float = 1e-8,
        unbiased_center: bool = False,
        transform: str = "clr",
        remove_diag: bool = True,
        use_fp64: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.unbiased_center = unbiased_center
        self.transform = transform
        self.remove_diag = remove_diag
        self.use_fp64 = use_fp64

    def _multihead_linear_gram(self, att: torch.Tensor) -> torch.Tensor:
        """Build a head-averaged linear Gram matrix from attention [H, N, D]."""
        if self.remove_diag:
            att = _remove_diag_and_renorm(att, eps=self.eps)

        att = att.clamp_min(self.eps)
        att = att / att.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        att = _attn_transform(att, transform=self.transform, eps=self.eps)

        if self.use_fp64:
            att = att.double()

        # K[n, m] = (1/H) * sum_h sum_k att[h,n,k] * att[h,m,k]
        return torch.einsum("hnk,hmk->nm", att, att) / att.size(0)

    def forward(self, att_s: torch.Tensor, att_t: torch.Tensor) -> torch.Tensor:
        """Compute multi-head CKA loss between student and teacher attention.

        att_s: [H_s, N, N], att_t: [H_t, N, N] (teacher already in student space).
        Returns scalar 1 - CKA.
        """
        K = _center_gram(self._multihead_linear_gram(att_s), unbiased=self.unbiased_center)
        L = _center_gram(self._multihead_linear_gram(att_t), unbiased=self.unbiased_center)
        num = (K * L).sum()
        den = K.norm(p="fro") * L.norm(p="fro")
        return 1.0 - num / (den + self.eps)
