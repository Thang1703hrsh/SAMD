"""samd — Span-Aware Matryoshka Distillation package.

Re-exports the public API from span_alignment, cka, mrd_loss, data, and eval
so notebooks and scripts can import directly from ``samd``.
"""

from samd.span_alignment import (
    build_span_overlap_matrix,
    align_tokens,
    build_reciprocal_mapping_from_token_lists,
    align_strict_one_to_one,
    align_by_path_pool_many,
)

from samd.cka import (
    CKALoss,
    LinearCKALoss,
    MultiHeadCKALoss,
    linear_cka_loss,
)

from samd.mrd_loss import (
    info_nce,
    Matry_infonce,
    matryoshka_prefix_cosine_loss,
    compute_span_cka_att_loss,
    mean_pooling,
    get_student_sentence_emb,
    extract_teacher_sentence_embedding,
)

from samd.data import (
    TextPairRaw,
    DualTokenizerCollate,
    STSDataset,
    ClasssifyDataset,
    PairDataset,
)

from samd.eval import (
    eval_sts_task,
    eval_classification_task,
    eval_pair_task,
    get_metric_pair_classification,
)

__all__ = [
    "build_span_overlap_matrix",
    "align_tokens",
    "build_reciprocal_mapping_from_token_lists",
    "align_strict_one_to_one",
    "align_by_path_pool_many",
    "CKALoss",
    "LinearCKALoss",
    "MultiHeadCKALoss",
    "linear_cka_loss",
    "info_nce",
    "Matry_infonce",
    "matryoshka_prefix_cosine_loss",
    "compute_span_cka_att_loss",
    "mean_pooling",
    "get_student_sentence_emb",
    "extract_teacher_sentence_embedding",
    "TextPairRaw",
    "DualTokenizerCollate",
    "STSDataset",
    "ClasssifyDataset",
    "PairDataset",
    "eval_sts_task",
    "eval_classification_task",
    "eval_pair_task",
    "get_metric_pair_classification",
]
