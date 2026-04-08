"""Matryoshka Representation Distillation loss functions.

Contains the contrastive losses (info_nce, Matry_infonce), the KD losses
(matryoshka_prefix_cosine_loss), the main SAMD attention alignment loss
(compute_span_cka_att_loss), and pooling utilities used across all notebooks.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from samd.cka import linear_cka_loss
from samd.span_alignment import build_span_overlap_matrix


def info_nce(
    q: torch.Tensor,
    k: torch.Tensor,
    temperature: float = 0.07,
    neg_valid_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """In-batch InfoNCE (SimCSE-style) contrastive loss.

    Treats all other samples in the batch as negatives. The i-th query is
    expected to match only the i-th key. neg_valid_mask is reserved for future
    hard-negative masking and is currently unused.

    Returns (loss, logits) where logits is the raw [B, B] similarity matrix.
    """
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    logits = torch.matmul(q, k.T) / temperature
    labels = torch.arange(q.size(0), device=q.device)
    return F.cross_entropy(logits, labels), logits


def Matry_infonce(
    a: torch.Tensor,
    b: torch.Tensor,
    temperature: float = 0.07,
    nested_dims: List[int] = (64, 128, 256, 512, 1024),
) -> Tuple[torch.Tensor, dict]:
    """Apply InfoNCE at each nested prefix dimension and sum the losses.

    Dimensions larger than the actual embedding size are skipped.
    Returns (total_loss, logits_per_dim) where logits_per_dim maps
    "dim_{d}" -> raw similarity matrix [B, B].
    """
    assert a.dim() == 2, f"Expected 2-D input [batch, dim], got {a.shape}"
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"

    full_dim   = a.size(1)
    total_loss = torch.zeros(1, device=a.device).squeeze()
    all_logits: dict = {}

    for d in nested_dims:
        if d > full_dim:
            continue
        q = F.normalize(a[:, :d], dim=-1)
        k = F.normalize(b[:, :d], dim=-1)
        logits = torch.matmul(q, k.T) / temperature
        labels = torch.arange(q.size(0), device=q.device)
        total_loss = total_loss + F.cross_entropy(logits, labels)
        all_logits[f"dim_{d}"] = logits

    return total_loss, all_logits


def matryoshka_prefix_cosine_loss(
    student_emb: torch.Tensor,
    teacher_emb: torch.Tensor,
    dims: Tuple[int, ...] = (128, 256, 512),
    eps: float = 1e-8,
) -> torch.Tensor:
    """Cosine KD loss between teacher and student at each prefix dimension.

    For each d in dims (plus full D), computes mean(1 - cosine(student[:d], teacher[:d]))
    and returns the average across all prefix sizes.
    """
    assert student_emb.dim() == 2 and teacher_emb.dim() == 2
    B, D = student_emb.shape
    assert teacher_emb.shape == (B, D)

    dims_list = sorted({int(d) for d in dims if 0 < int(d) <= D} | {D})
    total = 0.0
    for d in dims_list:
        s = student_emb[:, :d].float()
        t = teacher_emb[:, :d].float()
        s = s / (s.norm(dim=-1, keepdim=True) + eps)
        t = t / (t.norm(dim=-1, keepdim=True) + eps)
        total = total + (1.0 - (s * t).sum(dim=-1)).mean()
    return total / len(dims_list)


def token_importance_from_attention(
    att_mean: torch.Tensor,
    mask: torch.Tensor,
    special_token_ids: Optional[List[int]] = None,
) -> torch.Tensor:
    """Per-token importance as column-sum of average attention, softmaxed.

    att_mean: [L, L] mean attention for one sample.
    mask: [L] padding mask (1 = real token).
    Returns importance probabilities [L].
    """
    scores = att_mean.sum(dim=0) * mask.float().to(att_mean.device)
    return torch.softmax(scores, dim=-1)


def select_top_tokens(
    importance: torch.Tensor,
    mask: torch.Tensor,
    top_frac: float = 0.25,
    min_tokens: int = 2,
) -> torch.Tensor:
    """Return sorted indices of the top-frac most important non-padding tokens."""
    real_count = int(mask.sum().item())
    k = min(max(min_tokens, int(real_count * top_frac)), real_count)
    masked_imp = importance * mask.float().to(importance.device)
    return torch.sort(torch.topk(masked_imp, k=k, largest=True).indices).values


def coverage_from_A(A: torch.Tensor) -> Tuple[float, float]:
    """Return (coverage_s, coverage_t) from a span-overlap matrix [L_s, L_t].

    coverage_s: fraction of student tokens with any alignment weight.
    coverage_t: fraction of teacher tokens covered by at least one student token.
    """
    return (
        float((A.sum(dim=1) > 0).float().mean().item()),
        float((A.sum(dim=0) > 0).float().mean().item()),
    )


def compute_span_cka_att_loss(
    att_s: torch.Tensor,        # [B, H, Ls, Ls]  student attention
    att_t: torch.Tensor,        # [B, H, Lt, Lt]  teacher attention
    offsets_s: torch.Tensor,    # [B, Ls, 2]  student character offsets
    offsets_t: torch.Tensor,    # [B, Lt, 2]  teacher character offsets
    mask_s: torch.Tensor,       # [B, Ls]  student padding mask
    mask_t: torch.Tensor,       # [B, Lt]  teacher padding mask
    min_coverage: float = 0.3,
    top_frac: float = 0.25,
    min_tokens: int = 2,
) -> torch.Tensor:
    """Span-aware attention alignment loss using CKA (main SAMD loss).

    For each sample:
    1. Build span-overlap alignment matrix A from character offsets.
    2. Skip if coverage is below min_coverage (degenerate offsets).
    3. Select top-frac teacher tokens by attention importance.
    4. Project teacher attention into student space: A_sel @ Att_t[sel,sel] @ A_sel^T.
    5. Select top student tokens and compute linear CKA between student attention
       and the projected teacher attention.
    6. Weight each sample's CKA loss by alignment coverage confidence.

    Returns mean confidence-weighted (1 - CKA) over valid samples, or 0.0 if none pass.
    """
    B = att_s.size(0)

    # Average over attention heads to reduce variance
    att_s_mean = att_s.mean(dim=1)  # [B, Ls, Ls]
    att_t_mean = att_t.mean(dim=1)  # [B, Lt, Lt]

    losses: List[torch.Tensor] = []

    for b in range(B):
        A = build_span_overlap_matrix(
            offsets_s[b].to(att_s.device),
            offsets_t[b].to(att_s.device),
        )

        cov_s, cov_t = coverage_from_A(A)
        conf = min(cov_s, cov_t)
        if conf < min_coverage:
            continue

        t_mask = mask_t[b].to(att_s.device)
        t_imp  = token_importance_from_attention(att_t_mean[b], t_mask)
        t_sel  = select_top_tokens(t_imp, t_mask, top_frac=top_frac, min_tokens=min_tokens)
        if t_sel.numel() < min_tokens:
            continue

        A_sel  = A[:, t_sel]                                    # [Ls, M]
        s_has  = (A_sel.sum(dim=1) > 1e-12).float()
        s_mask = mask_s[b].to(att_s.device) * s_has
        if s_mask.sum() < min_tokens:
            continue

        att_t_sub  = att_t_mean[b][t_sel][:, t_sel]            # [M, M]
        att_t_proj = A_sel @ att_t_sub @ A_sel.transpose(0, 1) # [Ls, Ls]

        s_imp = token_importance_from_attention(att_s_mean[b], s_mask)
        s_sel = select_top_tokens(s_imp, s_mask, top_frac=top_frac, min_tokens=min_tokens)
        if s_sel.numel() < min_tokens:
            continue

        K = att_s_mean[b][s_sel][:, s_sel]
        L = att_t_proj[s_sel][:, s_sel]
        losses.append(linear_cka_loss(K, L) * conf)

    if not losses:
        return att_s.new_tensor(0.0)
    return torch.stack(losses).mean()


def mean_pooling(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Masked mean pooling over a transformer's last hidden state [B, L, D]."""
    mask   = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def get_student_sentence_emb(
    last_hidden: torch.Tensor,
    att_mask: torch.Tensor,
    mode: str = "cls",
) -> torch.Tensor:
    """Return student sentence embedding using "mean" or "cls" pooling."""
    if mode == "mean":
        return mean_pooling(last_hidden, att_mask)
    return last_hidden[:, 0, :]


def extract_teacher_sentence_embedding(
    T_last: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tok_teacher,
    embed_token: Optional[str] = None,
) -> torch.Tensor:
    """Extract teacher sentence embedding from the last hidden state.

    If embed_token is provided and present in the sequence (e.g. BGE's
    "<|embed|>" or LLM2Vec's embed token), uses that token's hidden state.
    Otherwise falls back to masked mean pooling.
    """
    if embed_token is not None:
        embed_id = None
        try:
            _id = tok_teacher.convert_tokens_to_ids(embed_token)
            if embed_token in tok_teacher.get_vocab():
                embed_id = _id
        except Exception:
            pass

        if embed_id is not None:
            m = (input_ids == embed_id)
            if m.any():
                idx     = m.float().argmax(dim=1)
                has_any = m.any(dim=1)
                out     = T_last[:, 0, :].clone()
                b_idx   = torch.arange(T_last.size(0), device=T_last.device)
                out[has_any] = T_last[b_idx[has_any], idx[has_any], :]
                return out

    return mean_pooling(T_last, attention_mask)
