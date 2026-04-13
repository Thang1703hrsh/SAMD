"""Character-span-based token alignment between teacher and student tokenizers.

The main function is build_span_overlap_matrix, which uses character-offset
overlap (Jaccard IoU) from fast HuggingFace tokenizers to build a soft
alignment matrix without requiring any shared vocabulary.

The edit-distance functions (align_tokens, build_reciprocal_mapping_from_token_lists)
are a fallback for slow tokenizers that don't return offset_mapping.

The DTW-path functions (align_strict_one_to_one, align_by_path_pool_many)
are used by the MinED and CDM baselines.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    import Levenshtein as _Lev
    def _edit_dist(a: str, b: str) -> int:
        return _Lev.distance(a, b)
except ImportError:
    def _edit_dist(a: str, b: str) -> int:  # type: ignore[misc]
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)
        if len(a) < len(b):
            a, b = b, a
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, start=1):
            cur = [i]
            for j, cb in enumerate(b, start=1):
                cur.append(min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]


def _clean_tok(t: Optional[str]) -> str:
    """Strip SentencePiece / GPT / WordPiece markers, whitespace, and lowercase."""
    if t is None:
        return ""
    t = t.replace("▁", "").replace("Ġ", "")
    if t.startswith("##"):
        t = t[2:]
    return t.strip().lower()


def _normalize_token(t: str, marker: Optional[str] = None) -> str:
    """Strip a specific subword marker plus the common ones, then lowercase."""
    markers: List[str] = []
    if marker:
        markers.append(marker)
    markers += ["▁", "Ġ", "##"]
    for m in markers:
        t = t.replace(m, "")
    return t.lower()


def build_span_overlap_matrix(
    offsets_s: torch.Tensor,  # [L_s, 2]  character (start, end) per student token
    offsets_t: torch.Tensor,  # [L_t, 2]  character (start, end) per teacher token
    eps: float = 1e-12,
) -> torch.Tensor:
    """Build a row-normalised soft alignment matrix from character-span IoU.

    Entry A[i, j] is the Jaccard overlap of student token i and teacher token j.
    Rows are L1-normalised so each student token's weight sums to 1.
    Requires fast HuggingFace tokenizers (offset_mapping must be available).

    Returns A of shape [L_s, L_t].
    """
    device = offsets_s.device
    offsets_t = offsets_t.to(device)

    # Broadcast: [L_s, 1] vs [1, L_t]
    s_start = offsets_s[:, 0].unsqueeze(1)
    s_end   = offsets_s[:, 1].unsqueeze(1)
    t_start = offsets_t[:, 0].unsqueeze(0)
    t_end   = offsets_t[:, 1].unsqueeze(0)

    inter = (torch.minimum(s_end, t_end) - torch.maximum(s_start, t_start)).clamp(min=0)
    len_s = (s_end - s_start).clamp(min=0)
    len_t = (t_end - t_start).clamp(min=0)
    union = (len_s + len_t - inter).clamp(min=eps)

    A = inter / union  # raw Jaccard overlap [L_s, L_t]
    return A / A.sum(dim=1, keepdim=True).clamp(min=eps)


def _build_span_overlap_matrix_fast(
    offsets_tea,
    offsets_stu,
    L_t: int,
    L_s: int,
    device,
    eps: float = 1e-12,
) -> Tuple[Optional[torch.Tensor], float, float]:
    """Two-pointer O(L_s + L_t) variant of build_span_overlap_matrix.

    Also returns per-axis coverage fractions used for confidence gating.
    Returns (A, coverage_s, coverage_t), or (None, 0, 0) if offsets are missing.
    """
    if offsets_tea is None or offsets_stu is None:
        return None, 0.0, 0.0

    ot  = offsets_tea[:L_t].detach().cpu().tolist() if torch.is_tensor(offsets_tea) else list(offsets_tea)[:L_t]
    os_ = offsets_stu[:L_s].detach().cpu().tolist() if torch.is_tensor(offsets_stu) else list(offsets_stu)[:L_s]

    A = torch.zeros((L_s, L_t), device=device, dtype=torch.float32)

    i = 0  # teacher pointer
    j = 0  # student pointer
    while i < L_t and j < L_s:
        a, b = ot[i]
        c, d = os_[j]

        if b <= a:   # zero-length teacher span (special token)
            i += 1; continue
        if d <= c:   # zero-length student span (special token)
            j += 1; continue
        if b <= c:   # teacher span ends before student begins
            i += 1; continue
        if d <= a:   # student span ends before teacher begins
            j += 1; continue

        overlap = min(b, d) - max(a, c)
        if overlap > 0:
            A[j, i] = float(overlap)

        if b < d:
            i += 1
        elif d < b:
            j += 1
        else:
            i += 1; j += 1

    row_sum = A.sum(dim=1, keepdim=True)
    A = A / row_sum.clamp_min(eps)

    coverage_s = float((row_sum.squeeze(1) > 0).float().mean())
    coverage_t = float((A.sum(dim=0) > 0).float().mean())
    return A, coverage_s, coverage_t


def align_tokens(
    teacher_tokens: List[str],
    student_tokens: List[str],
    teacher_special: str = "<s>",
    student_special: str = "[CLS]",
) -> Dict[str, str]:
    """Map each teacher token string to the closest student token string.

    Fallback for slow tokenizers. Matches greedily by normalised edit distance.
    Returns a dict mapping teacher token -> best-matching student token.
    """
    out: Dict[str, str] = {}
    if not teacher_tokens or not student_tokens:
        return out

    if teacher_special in teacher_tokens and student_special in student_tokens:
        out[teacher_special] = student_special

    student_set = set(student_tokens)
    stu_clean = [(s, _clean_tok(s)) for s in student_tokens if s != student_special]

    for t in teacher_tokens:
        if t == teacher_special:
            continue

        tmp_t = t.replace(teacher_special, student_special)
        if tmp_t in student_set and tmp_t != student_special:
            out[t] = tmp_t
            continue

        t_clean = _clean_tok(tmp_t)
        if not t_clean:
            continue

        best_s, best_d = None, 10 ** 9
        for s, s_clean in stu_clean:
            if t_clean == s_clean:
                best_s, best_d = s, 0
                break
            d = _edit_dist(t_clean, s_clean)
            if d < best_d:
                best_s, best_d = s, d
        if best_s is not None:
            out[t] = best_s

    return out


def build_reciprocal_mapping_from_token_lists(
    teacher_tokens: List[str],
    student_tokens: List[str],
    teacher_special: str = "<s>",
    student_special: str = "[CLS]",
) -> Dict[str, str]:
    """Keep only mutually consistent (reciprocal) teacher->student token pairs.

    A pair is kept only when teacher->student and student->teacher both agree.
    This filter reduces spurious many-to-one alignments.
    """
    t2s = align_tokens(teacher_tokens, student_tokens, teacher_special, student_special)
    s2t = align_tokens(student_tokens, teacher_tokens, student_special, teacher_special)
    return {t: s for t, s in t2s.items() if s in s2t and s2t[s] == t}


def align_strict_one_to_one(
    base_vals: torch.Tensor,
    blend_vals: torch.Tensor,
    path: Sequence[Tuple[int, int]],
    base_tokens: List[str],
    blend_tokens: List[str],
    base_marker: str,
    blend_marker: str,
    specTok_mapper: Optional[Dict[str, str]] = None,
    *,
    debug: bool = False,
    max_print: int = 20,
    dtw_matrix: Optional[np.ndarray] = None,
    dtw_crop: int = 12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract strictly 1-to-1 aligned token vectors from a DTW path.

    Keeps a pair (i, j) only when:
    - base index i appears exactly once in the path, AND
    - blend index j appears exactly once in the path, AND
    - normalised surface forms agree, or the pair is in specTok_mapper.

    Returns (A_base, A_blend) stacked tensors of shape [K, D].
    """
    if specTok_mapper is None:
        specTok_mapper = {}

    base_counts: Dict[int, int] = {}
    blend_counts: Dict[int, int] = {}
    for i, j in path:
        base_counts[i] = base_counts.get(i, 0) + 1
        blend_counts[j] = blend_counts.get(j, 0) + 1

    base_norm  = [_normalize_token(t, base_marker)  for t in base_tokens]
    blend_norm = [_normalize_token(t, blend_marker) for t in blend_tokens]
    specTok_mapper_rev = {v: k for k, v in specTok_mapper.items()} if specTok_mapper else {}

    def _is_special_pair_ok(b_tok: str, s_tok: str) -> bool:
        return (b_tok in specTok_mapper and specTok_mapper[b_tok] == s_tok) or (
            s_tok in specTok_mapper_rev and specTok_mapper_rev[s_tok] == b_tok
        )

    one_to_one = [(i, j) for i, j in path if base_counts.get(i, 0) == 1 and blend_counts.get(j, 0) == 1]

    kept_pairs: List[Tuple[int, int]] = []
    name_mismatch: List[tuple] = []
    multi_align: List[tuple] = []

    for i, j in path:
        if base_counts.get(i, 0) != 1 or blend_counts.get(j, 0) != 1:
            if len(multi_align) < max_print:
                multi_align.append((i, j, base_tokens[i], blend_tokens[j],
                                    base_counts.get(i, 0), blend_counts.get(j, 0)))
            continue

        bi_raw, sj_raw = base_tokens[i], blend_tokens[j]
        if _is_special_pair_ok(bi_raw, sj_raw) or (base_norm[i] == blend_norm[j]):
            kept_pairs.append((i, j))
        elif len(name_mismatch) < max_print:
            name_mismatch.append((i, j, bi_raw, sj_raw, base_norm[i], blend_norm[j]))

    if len(kept_pairs) == 0:
        A_base  = base_vals.new_empty((0, base_vals.size(-1)))
        A_blend = blend_vals.new_empty((0, blend_vals.size(-1)))
    else:
        A_base  = torch.stack([base_vals[i]  for i, _ in kept_pairs])
        A_blend = torch.stack([blend_vals[j] for _, j in kept_pairs])

    if debug:
        print("\n================= [ALIGN DEBUG] =================")
        print(f"L_base={base_vals.size(0)}, L_blend={blend_vals.size(0)}, |path|={len(path)}")
        print(f"1-1 candidates: {len(one_to_one)} / {len(path)}")
        print(f"Kept (strict name match + special map): {len(kept_pairs)}")
        if multi_align:
            print(f"\n[Dropped: multi-align] (up to {max_print})")
            for i, j, braw, sraw, bc, sc in multi_align[:max_print]:
                print(f"  ({i},{j}) teacher='{braw}' student='{sraw}' counts=({bc},{sc})")
        if name_mismatch:
            print(f"\n[Dropped: name mismatch] (up to {max_print})")
            for i, j, braw, sraw, bn, sn in name_mismatch[:max_print]:
                print(f"  ({i},{j}) teacher='{braw}'->'{bn}'  student='{sraw}'->'{sn}'")
        if kept_pairs:
            print(f"\n[First kept pairs] (up to {max_print})")
            for i, j in kept_pairs[:max_print]:
                print(f"  ({i},{j}) '{base_tokens[i]}' <-> '{blend_tokens[j]}'  "
                      f"norm='{base_norm[i]}' <-> '{blend_norm[j]}'")
        if dtw_matrix is not None:
            H, W = dtw_matrix.shape
            h, w = min(dtw_crop, H), min(dtw_crop, W)
            print(f"\n[DTW matrix] shape={dtw_matrix.shape} (showing {h}x{w} top-left)")
            print(np.array2string(dtw_matrix[:h, :w], precision=2, suppress_small=True))
        print("=================================================\n")

    return A_base, A_blend


def align_by_path_pool_many(
    base_vals: torch.Tensor,
    blend_vals: torch.Tensor,
    path: Sequence[Tuple[int, int]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pool many-to-one / one-to-many DTW groups into aligned vector pairs.

    Unlike align_strict_one_to_one, multi-aligned tokens are averaged rather
    than dropped. Returns (A_base, A_blend) of shape [K, D].
    """
    A_base_list:  List[torch.Tensor] = []
    A_blend_list: List[torch.Tensor] = []

    k, P = 0, len(path)
    while k < P:
        i0, j0 = path[k]
        if k == P - 1:
            A_base_list.append(base_vals[i0])
            A_blend_list.append(blend_vals[j0])
            break

        i1, j1 = path[k + 1]
        di, dj = i1 - i0, j1 - j0

        # Many base tokens -> one blend token: average the base side
        if dj == 0 and di == 1:
            i_run = [i0]
            j_fix = j0
            kk = k + 1
            while kk < P and path[kk][1] == j_fix and path[kk][0] == i_run[-1] + 1:
                i_run.append(path[kk][0])
                kk += 1
            A_base_list.append(base_vals[i_run].mean(dim=0))
            A_blend_list.append(blend_vals[j_fix])
            k = kk
            continue

        # One base token -> many blend tokens: average the blend side
        if di == 0 and dj == 1:
            j_run = [j0]
            i_fix = i0
            kk = k + 1
            while kk < P and path[kk][0] == i_fix and path[kk][1] == j_run[-1] + 1:
                j_run.append(path[kk][1])
                kk += 1
            A_base_list.append(base_vals[i_fix])
            A_blend_list.append(blend_vals[j_run].mean(dim=0))
            k = kk
            continue

        # Diagonal step: 1-to-1
        A_base_list.append(base_vals[i0])
        A_blend_list.append(blend_vals[j0])
        k += 1

    return torch.stack(A_base_list), torch.stack(A_blend_list)
