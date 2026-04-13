"""Dataset classes and dual-tokenizer collation for multitask embedding training."""
from __future__ import annotations

from typing import List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


class TextPairRaw(Dataset):
    """Multitask training dataset loaded from a flat CSV.

    Supports three task formats:
    - "single_cls": columns text, label. Returns (text, None, label).
    - "pair_cls":   columns premise, hypothesis. Returns (s1, s2).
    - "pair_reg":   columns sentence1, sentence2. Returns (s1, s2).
    """

    def __init__(self, df: pd.DataFrame, task: str):
        self.task = task
        if task == "single_cls":
            self.samples = [
                (str(t), None, int(y))
                for t, y in zip(df["text"].astype(str), df["label"].astype(int))
            ]
        elif task == "pair_cls":
            self.samples = [
                (str(a), str(b))
                for a, b in zip(df["premise"].astype(str), df["hypothesis"].astype(str))
            ]
        else:  # pair_reg
            self.samples = [
                (str(a), str(b))
                for a, b in zip(df["sentence1"].astype(str), df["sentence2"].astype(str))
            ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


class DualTokenizerCollate:
    """Collate function that tokenizes each batch with both student and teacher tokenizers.

    For fast tokenizers (tokenizer.is_fast == True), also returns offset_mapping
    so that build_span_overlap_matrix can use character-level offsets for alignment.
    Slow tokenizers simply omit offsets; the alignment falls back to edit-distance matching.
    """

    def __init__(self, tok_student, tok_teacher, task: str, max_len: int):
        self.ts = tok_student
        self.tt = tok_teacher
        self.task = task
        self.max_len = max_len
        self._stu_fast = bool(getattr(tok_student, "is_fast", False))
        self._tea_fast = bool(getattr(tok_teacher, "is_fast", False))

    def _encode(self, tok, texts: List[str], want_offsets: bool) -> dict:
        kwargs = dict(
            max_length=self.max_len,
            truncation=True,
            padding=True,                # pad each batch only to the length of its longest sequence
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        if want_offsets:
            kwargs["return_offsets_mapping"] = True  # only supported by fast tokenizers
        return tok(list(texts), **kwargs)

    def __call__(self, batch) -> dict:
        if self.task == "single_cls":
            texts, labels = [], []
            for item in batch:
                # tolerate both (text, y) and (text, _, y)
                if len(item) == 2:
                    t, y = item
                else:
                    t, _, y = item
                texts.append(t)
                labels.append(int(y))

            s_enc = self._encode(self.ts, texts, self._stu_fast)
            t_enc = self._encode(self.tt, texts, self._tea_fast)

            out = {
                "input_ids_stu":           s_enc["input_ids"],
                "attention_mask_stu":      s_enc["attention_mask"],
                "special_tokens_mask_stu": s_enc["special_tokens_mask"],
                "input_ids_tea":           t_enc["input_ids"],
                "attention_mask_tea":      t_enc["attention_mask"],
                "special_tokens_mask_tea": t_enc["special_tokens_mask"],
                "labels": torch.tensor(labels, dtype=torch.long),
            }
            if "token_type_ids" in s_enc:
                out["token_type_ids_stu"] = s_enc["token_type_ids"]
            if "token_type_ids" in t_enc:
                out["token_type_ids_tea"] = t_enc["token_type_ids"]
            if "offset_mapping" in s_enc:
                out["offset_mapping_stu"] = s_enc["offset_mapping"]
            if "offset_mapping" in t_enc:
                out["offset_mapping_tea"] = t_enc["offset_mapping"]
            return out

        # pair (bi-encoder)
        s1s, s2s = zip(*batch)

        s1_enc = self._encode(self.ts, s1s, self._stu_fast)
        s2_enc = self._encode(self.ts, s2s, self._stu_fast)
        t1_enc = self._encode(self.tt, s1s, self._tea_fast)
        t2_enc = self._encode(self.tt, s2s, self._tea_fast)

        out = {
            "input_ids1_stu":           s1_enc["input_ids"],
            "attention_mask1_stu":      s1_enc["attention_mask"],
            "special_tokens_mask1_stu": s1_enc["special_tokens_mask"],
            "input_ids2_stu":           s2_enc["input_ids"],
            "attention_mask2_stu":      s2_enc["attention_mask"],
            "special_tokens_mask2_stu": s2_enc["special_tokens_mask"],
            "input_ids1_tea":           t1_enc["input_ids"],
            "attention_mask1_tea":      t1_enc["attention_mask"],
            "special_tokens_mask1_tea": t1_enc["special_tokens_mask"],
            "input_ids2_tea":           t2_enc["input_ids"],
            "attention_mask2_tea":      t2_enc["attention_mask"],
            "special_tokens_mask2_tea": t2_enc["special_tokens_mask"],
        }

        for prefix, enc in [("token_type_ids1_stu", s1_enc), ("token_type_ids2_stu", s2_enc),
                             ("token_type_ids1_tea", t1_enc), ("token_type_ids2_tea", t2_enc)]:
            if "token_type_ids" in enc:
                out[prefix] = enc["token_type_ids"]

        for prefix, enc in [("offset_mapping1_stu", s1_enc), ("offset_mapping2_stu", s2_enc),
                             ("offset_mapping1_tea", t1_enc), ("offset_mapping2_tea", t2_enc)]:
            if "offset_mapping" in enc:
                out[prefix] = enc["offset_mapping"]

        return out


class STSDataset(Dataset):
    """CSV dataset for STS tasks. Expects columns: sentence1, sentence2, score."""

    def __init__(self, file_path: str):
        self.dataset = pd.read_csv(file_path)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        row = self.dataset.iloc[idx]
        return {
            "sentence1": str(row["sentence1"]),
            "sentence2": str(row["sentence2"]),
            "label": torch.tensor(float(row["score"]), dtype=torch.float),
        }


class ClasssifyDataset(Dataset):
    """CSV dataset for single-sentence classification. Expects columns: text, label.

    The double-s in the class name is kept for compatibility with existing notebooks.
    """

    def __init__(self, file_path: str):
        self.dataset = pd.read_csv(file_path)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        row = self.dataset.iloc[idx]
        return {
            "text":  str(row["text"]),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
        }


class PairDataset(Dataset):
    """CSV dataset for sentence-pair tasks (MRPC, WiC, etc.).
    Expects columns: sentence1, sentence2, label (binary 0/1).
    """

    def __init__(self, file_path: str):
        self.dataset = pd.read_csv(file_path)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        row = self.dataset.iloc[idx]
        return {
            "sentence1": str(row["sentence1"]),
            "sentence2": str(row["sentence2"]),
            "label": torch.tensor(float(row["label"]), dtype=torch.float),
        }
