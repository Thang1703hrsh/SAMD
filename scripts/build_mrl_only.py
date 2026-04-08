"""Build samd-mrl-only.ipynb — MRL-only (no SAM, no teacher).
Task loss: Matryoshka InfoNCE (sum of InfoNCE across nested prefix dims).
Same hyperparameters as samd-mrl.ipynb; SAM / teacher code fully removed.
"""
import json, textwrap

def code(src): return {"cell_type":"code","metadata":{},"source":textwrap.dedent(src).lstrip("\n"),"outputs":[],"execution_count":None}
def md(src):   return {"cell_type":"markdown","metadata":{},"source":textwrap.dedent(src).lstrip("\n")}

cells = []

# ── 0  Title ──────────────────────────────────────────────────────────────────
cells.append(md(r"""
# SAMD — MRL-only (no SAM, no teacher)

Training objective:

$$\mathcal{L} = \sum_{d \in \texttt{nested\_dims}} \text{InfoNCE}\!\left(S_1[:d],\; S_2[:d]\right)$$

- **Matryoshka InfoNCE**: same sentence passed twice with different dropout masks (SimCSE);
  contrastive loss computed at every nested prefix dimension.
- **SAM** (Span-Aware attention Matching) is **removed** entirely — no teacher model loaded.
"""))

# ── 1  Imports ─────────────────────────────────────────────────────────────────
cells.append(code("""
# ============================================================
# 0) Setup
# ============================================================
import os
import math
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel, get_cosine_with_hard_restarts_schedule_with_warmup
"""))

# ── 2  set_seed ────────────────────────────────────────────────────────────────
cells.append(code("""
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)
print("torch:", torch.__version__)
"""))

# ── 3  Config (same values as samd-mrl.ipynb) ─────────────────────────────────
cells.append(code("""
@dataclass
class Config:
    # Models
    student_name: str = "huawei-noah/TinyBERT_General_6L_768D"

    # Data
    train_csv: str = "/kaggle/input/multitask-data/merged_9_data_3k_each_ver2.csv"
    text_col:  str = "text"
    max_length: int = 128

    # Training
    batch_size: int = 16
    epochs:     int = 10
    lr:         float = 3e-5
    weight_decay: float = 0.01
    temperature:  float = 0.05
    grad_clip:    float = 1.0

    # Pooling
    student_pool: str = "mean"

    # Matryoshka dims — set automatically from model hidden size if None
    nested_dims: Optional[List[int]] = None

    # Scheduler
    warmup_ratio: float = 0.06
    num_restarts: int   = 1

cfg = Config()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)
"""))

# ── 4  Pooling helpers ─────────────────────────────────────────────────────────
cells.append(code("""
# ============================================================
# 2) Pooling + sentence embedding helpers
# ============================================================
def mean_pooling(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask   = attention_mask.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

def get_sentence_emb(last_hidden: torch.Tensor,
                     attention_mask: torch.Tensor,
                     mode: str) -> torch.Tensor:
    if mode == "mean":
        return mean_pooling(last_hidden, attention_mask)
    return last_hidden[:, 0, :]   # CLS
"""))

# ── 5  Matryoshka InfoNCE ─────────────────────────────────────────────────────
cells.append(code("""
# ============================================================
# 3) Matryoshka InfoNCE loss
# ============================================================
def matryoshka_infonce(
    a: torch.Tensor,
    b: torch.Tensor,
    temperature: float,
    nested_dims: List[int],
    dim_weight: str = "uniform",   # "uniform" or "inverse_sqrt"
) -> Tuple[torch.Tensor, Dict[str, float]]:
    \"\"\"
    Sum of InfoNCE losses at every nested prefix dimension.
    a, b: [B, D]  (two dropout views of the same batch)
    \"\"\"
    full_dim = a.size(1)
    dims     = [d for d in nested_dims if d <= full_dim]
    assert len(dims) > 0, f"No nested_dims <= {full_dim}"

    if dim_weight == "uniform":
        weights = [1.0] * len(dims)
    else:   # inverse_sqrt
        raw = [1.0 / math.sqrt(d) for d in dims]
        s   = sum(raw)
        weights = [w / s * len(dims) for w in raw]

    total      = a.new_tensor(0.0)
    loss_per_d: Dict[str, float] = {}

    for d, w in zip(dims, weights):
        q      = F.normalize(a[:, :d], dim=-1)
        k      = F.normalize(b[:, :d], dim=-1)
        logits = (q @ k.T) / temperature
        labels = torch.arange(q.size(0), device=q.device)
        loss_d = F.cross_entropy(logits, labels)
        total  = total + w * loss_d
        loss_per_d[f"d{d}"] = loss_d.item()

    return total, loss_per_d
"""))

# ── 6  Dataset + Collator (student only — no teacher) ─────────────────────────
cells.append(code("""
# ============================================================
# 4) Dataset + collator  (student-only, no teacher)
# ============================================================
class TextOnlyDataset(Dataset):
    def __init__(self, df: pd.DataFrame, text_col: str):
        self.texts = df[text_col].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {"text": self.texts[idx]}


class SimCSECollate:
    \"\"\"Tokenise each sentence twice (two dropout views). No teacher.\"\"\"
    def __init__(self, tokenizer, max_length: int):
        self.tok        = tokenizer
        self.max_length = max_length

    def _tok(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        return self.tok(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [x["text"] for x in batch]
        s1    = self._tok(texts)
        s2    = self._tok(texts)
        out: Dict[str, torch.Tensor] = {}
        for k, v in s1.items(): out[f"{k}1"] = v
        for k, v in s2.items(): out[f"{k}2"] = v
        return out


def load_train_dataframe(path: str, text_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if text_col not in df.columns:
        raise ValueError(f"text_col='{text_col}' not in {list(df.columns)[:20]}")
    return df.dropna(subset=[text_col]).reset_index(drop=True)
"""))

# ── 7  Model loading ──────────────────────────────────────────────────────────
cells.append(code("""
tok_s     = AutoTokenizer.from_pretrained(cfg.student_name, use_fast=True)
model_s   = AutoModel.from_pretrained(cfg.student_name).to(device)

d_s = model_s.config.hidden_size
print("d_s:", d_s)

# Populate nested_dims
if cfg.nested_dims is None:
    base = [16, 32, 64, 128, 256, 512, 768, 1024]
    cfg.nested_dims = [d for d in base if d <= d_s]
    if cfg.nested_dims[-1] != d_s:
        cfg.nested_dims.append(d_s)

print("nested_dims:", cfg.nested_dims)
"""))

# ── 8  Optimizer + DataLoader ─────────────────────────────────────────────────
cells.append(code("""
# ============================================================
# 6) Optimizer + scheduler + loader
# ============================================================
optimizer = torch.optim.AdamW(
    model_s.parameters(),
    lr=cfg.lr,
    weight_decay=cfg.weight_decay,
)

df_train     = load_train_dataframe(cfg.train_csv, cfg.text_col)
train_ds     = TextOnlyDataset(df_train, cfg.text_col)
collate_fn   = SimCSECollate(tok_s, max_length=cfg.max_length)
train_loader = DataLoader(
    train_ds,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=2,
    collate_fn=collate_fn,
    drop_last=True,
)

total_steps  = cfg.epochs * max(1, len(train_loader))
warmup_steps = int(cfg.warmup_ratio * total_steps)

scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
    num_cycles=cfg.num_restarts,
)

scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
print(f"total_steps: {total_steps} | warmup_steps: {warmup_steps}")
print(f"train samples: {len(df_train)} | steps/epoch: {len(train_loader)}")
"""))

# ── 9  Training loop ──────────────────────────────────────────────────────────
cells.append(code("""
from tqdm.auto import tqdm

global_step = 0
model_s.train()

for epoch in range(cfg.epochs):
    pbar    = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
    running = 0.0

    for batch in pbar:
        optimizer.zero_grad(set_to_none=True)

        ids1  = batch["input_ids1"].to(device)
        mask1 = batch["attention_mask1"].to(device)
        ids2  = batch["input_ids2"].to(device)
        mask2 = batch["attention_mask2"].to(device)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            # Two dropout views of the same batch
            out1 = model_s(input_ids=ids1, attention_mask=mask1, return_dict=True)
            out2 = model_s(input_ids=ids2, attention_mask=mask2, return_dict=True)

            S1 = get_sentence_emb(out1.last_hidden_state, mask1, cfg.student_pool)
            S2 = get_sentence_emb(out2.last_hidden_state, mask2, cfg.student_pool)

            # Matryoshka InfoNCE — loss at every nested prefix dim
            loss, loss_per_d = matryoshka_infonce(
                S1, S2,
                temperature=cfg.temperature,
                nested_dims=cfg.nested_dims,
            )

        scaler.scale(loss).backward()

        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model_s.parameters(), cfg.grad_clip)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running = (0.95 * running + 0.05 * loss.item()
                   if global_step > 0 else loss.item())

        # Show loss at last dim (full dimension)
        last_d = f"d{cfg.nested_dims[-1]}"
        pbar.set_postfix({
            "loss":   f"{running:.4f}",
            last_d:   f"{loss_per_d.get(last_d, 0):.4f}",
            "d16":    f"{loss_per_d.get('d16', 0):.4f}",
        })
        global_step += 1

print("Done. global_step =", global_step)
"""))

# ── 10 encode_texts ───────────────────────────────────────────────────────────
cells.append(code("""
# ============================================================
# ENCODE (for evaluation)
# ============================================================
from tqdm.auto import tqdm

@torch.no_grad()
def encode_texts(
    texts,
    batch_size: int = 256,
    max_length: Optional[int] = None,
    normalize: bool = False,
) -> torch.Tensor:
    model_s.eval()
    if max_length is None:
        max_length = cfg.max_length
    texts   = [str(x) for x in texts]
    all_emb = []
    for i in tqdm(range(0, len(texts), batch_size), desc="encode", leave=False):
        enc = tok_s(
            texts[i:i+batch_size],
            padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model_s(**enc, return_dict=True)
        emb = get_sentence_emb(out.last_hidden_state, enc["attention_mask"], cfg.student_pool)
        if normalize:
            emb = F.normalize(emb, dim=-1)
        all_emb.append(emb.cpu())
    return torch.cat(all_emb, 0)
"""))

# ── 11 Eval helpers ───────────────────────────────────────────────────────────
cells.append(code("""
# ============================================================
# Evaluation helpers (per nested-dim slice)
# ============================================================
import numpy as np

try:
    from scipy.stats import spearmanr
except ImportError:
    spearmanr = None
    print("[WARN] scipy not available -> STS Spearman skipped.")

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
except ImportError:
    LogisticRegression = accuracy_score = f1_score = None
    print("[WARN] scikit-learn not available -> CLS eval skipped.")

try:
    from IPython.display import display
except ImportError:
    display = print


def _spearman(a, b) -> float:
    if spearmanr is None: return 0.0
    r = spearmanr(a, b).correlation
    return float(r) if r == r else 0.0


def eval_cls(train_csv, test_csv, text_col="text", label_col="label",
             dims=None, bs=256):
    tr, te = pd.read_csv(train_csv), pd.read_csv(test_csv)
    Xtr = encode_texts(tr[text_col].tolist(), batch_size=bs)
    Xte = encode_texts(te[text_col].tolist(), batch_size=bs)
    ytr = tr[label_col].astype(int).values
    yte = te[label_col].astype(int).values
    rows = []
    for d in (dims or cfg.nested_dims):
        clf = LogisticRegression(max_iter=2000)
        clf.fit(Xtr[:, :d].numpy(), ytr)
        pred = clf.predict(Xte[:, :d].numpy())
        rows.append({"dim": d,
                     "acc":      float(accuracy_score(yte, pred)),
                     "f1_macro": float(f1_score(yte, pred, average="macro"))})
    return pd.DataFrame(rows).set_index("dim")


def eval_sts(test_csv, s1="sentence1", s2="sentence2",
             score_col="score", dims=None, bs=256):
    te = pd.read_csv(test_csv)
    A  = encode_texts(te[s1].tolist(), batch_size=bs)
    B  = encode_texts(te[s2].tolist(), batch_size=bs)
    y  = te[score_col].astype(float).values
    rows = []
    for d in (dims or cfg.nested_dims):
        sim = (F.normalize(A[:, :d], dim=-1) * F.normalize(B[:, :d], dim=-1)).sum(-1).numpy()
        rows.append({"dim": d, "spearman": _spearman(sim, y)})
    return pd.DataFrame(rows).set_index("dim")


def eval_pair(train_csv, test_csv, s1="sentence1", s2="sentence2",
              label_col="label", dims=None, bs=256):
    tr, te = pd.read_csv(train_csv), pd.read_csv(test_csv)
    Atr = encode_texts(tr[s1].tolist(), batch_size=bs)
    Btr = encode_texts(tr[s2].tolist(), batch_size=bs)
    Ate = encode_texts(te[s1].tolist(), batch_size=bs)
    Bte = encode_texts(te[s2].tolist(), batch_size=bs)
    ytr = tr[label_col].astype(int).values
    yte = te[label_col].astype(int).values
    rows = []
    for d in (dims or cfg.nested_dims):
        sim_tr = (F.normalize(Atr[:, :d], -1) * F.normalize(Btr[:, :d], -1)).sum(-1).numpy()
        sim_te = (F.normalize(Ate[:, :d], -1) * F.normalize(Bte[:, :d], -1)).sum(-1).numpy()
        best_thr = max(np.linspace(-1, 1, 401),
                       key=lambda t: ((sim_tr >= t).astype(int) == ytr).mean())
        pred = (sim_te >= best_thr).astype(int)
        rows.append({"dim": d,
                     "acc": float((pred == yte).mean()),
                     "thr": float(best_thr)})
    return pd.DataFrame(rows).set_index("dim")
"""))

# ── 12 Run eval ───────────────────────────────────────────────────────────────
cells.append(code("""
# ============================================================
# RUN EVAL
# ============================================================
EVAL_ROOT      = "/kaggle/input/multitask-data/multi-data"
STS_EXTRA_ROOT = os.environ.get("STS_EXTRA_ROOT", "")
NESTED_DIMS    = cfg.nested_dims
print("NESTED_DIMS:", NESTED_DIMS)

results = {"cls": {}, "pair": {}, "sts": {}}

if EVAL_ROOT and os.path.exists(EVAL_ROOT):
    for name, trf, tef in [
        ("Banking77", "banking_train.csv",  "banking77_test.csv"),
        ("Emotion",   "emotion_train.csv",  "emotion_test.csv"),
        ("TweetEval", "tweet_train.csv",    "tweet_test.csv"),
    ]:
        tr = os.path.join(EVAL_ROOT, trf)
        te = os.path.join(EVAL_ROOT, tef)
        if os.path.exists(tr) and os.path.exists(te):
            results["cls"][name] = eval_cls(tr, te, dims=NESTED_DIMS)
            print(f"[CLS] {name}")
            display(results["cls"][name])

    for name, trf, tef in [
        ("MRPC",    "mrpc_validation.csv",    "mrpc_test.csv"),
        ("SciTail", "scitail_validation.csv", "scitail_test.csv"),
        ("WiC",     "wic_validation.csv",     "wic_test.csv"),
    ]:
        tr = os.path.join(EVAL_ROOT, trf)
        te = os.path.join(EVAL_ROOT, tef)
        if os.path.exists(tr) and os.path.exists(te):
            results["pair"][name] = eval_pair(tr, te, dims=NESTED_DIMS)
            print(f"[PAIR] {name}")
            display(results["pair"][name])

    for name, tef in [
        ("SICK",  "sick_test.csv"),
        ("STS12", "sts12_test.csv"),
        ("STS-B", "stsb_test.csv"),
    ]:
        te = os.path.join(EVAL_ROOT, tef)
        if os.path.exists(te):
            results["sts"][name] = eval_sts(te, dims=NESTED_DIMS)
            print(f"[STS] {name}")
            display(results["sts"][name])
else:
    print("EVAL_ROOT not found; skipping eval.")

results
"""))

# ── Write notebook ─────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": cells,
}

out = r"C:\Users\Thang Tran\SAMD\samd-mrl-only.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Written: {out}")
print(f"Total cells: {len(cells)}")
for i, c in enumerate(cells):
    src = "".join(c["source"])[:70].replace("\n", " ")
    print(f"  {i:2d} [{c['cell_type']:8s}] {src}")
