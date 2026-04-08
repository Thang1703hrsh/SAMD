"""Build samd-sam-only-v2.ipynb — SAM-only rewrite of samd-mrl.ipynb.
Same parameters; MRD and proj_t2s fully removed; Cell 10 optimizer bug fixed.
"""
import json, textwrap

def code(src): return {"cell_type":"code","metadata":{},"source":textwrap.dedent(src).lstrip("\n"),"outputs":[],"execution_count":None}
def md(src):   return {"cell_type":"markdown","metadata":{},"source":textwrap.dedent(src).lstrip("\n")}

cells = []

# ── 0  Title ──────────────────────────────────────────────────────────────────
cells.append(md(r"""
# SAMD — SAM-only (no MRD)

Training objective:

$$\mathcal{L} = \mathcal{L}_{\text{SimCSE}} + \lambda_{\text{SAM}} \cdot \mathcal{L}_{\text{SAM}}$$

- **SimCSE**: plain InfoNCE (two dropout views of the same sentence).
- **SAM** (Span-Aware attention Matching): character-span overlap matrix $A$ transports
  teacher attention $\tilde{T} = A\,T\,A^\top$; Linear CKA aligns student and transported
  teacher attention patterns.
- **MRD** (Multi-Resolution Distillation) is **removed** entirely.
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
    student_name: str = "bert-base-uncased"
    teacher_name: str = "bert-base-uncased"

    # Data
    train_csv: str = "/kaggle/input/multitask-data/merged_9_data_3k_each_ver2.csv"
    text_col:  str = "text"
    max_length: int = 128          # 256 -> 128: fits larger batch

    # Training
    batch_size: int = 16           # 8 -> 16: more in-batch negatives
    epochs:     int = 10
    lr:         float = 3e-5
    weight_decay: float = 0.01
    temperature:  float = 0.05     # 0.07 -> 0.05: sharper contrast
    grad_clip:    float = 1.0

    # Pooling
    student_pool: str = "mean"     # CLS -> mean: more robust sentence embeddings

    # SAM hyperparameters
    alpha_attn_max:   float = 0.5  # sole distillation signal
    alpha_attn_start: int   = 0    # start from step 0
    alpha_attn_ramp:  int   = 500  # ramp over 500 steps
    att_every:  int   = 2          # compute SAM every N steps
    att_layer:  str   = "last"     # "last" or "mid"
    min_coverage: float = 0.30
    top_frac:   float = 0.7        # fraction of most-attended tokens for CKA
    min_tokens: int   = 1

    # Scheduler
    warmup_ratio: float = 0.06
    num_restarts: int   = 1

    # Kept for eval slicing only
    nested_dims: Optional[List[int]] = None

cfg = Config()

device_s = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_t = torch.device("cuda:1" if torch.cuda.device_count() > 1 else device_s)
print("device_s:", device_s, "| device_t:", device_t, "| n_gpu:", torch.cuda.device_count())
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

# ── 5  Losses ─────────────────────────────────────────────────────────────────
cells.append(code("""
# ============================================================
# 3) Losses: InfoNCE + Linear CKA
# ============================================================
def info_nce(q: torch.Tensor, k: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    logits = (q @ k.T) / temperature
    labels = torch.arange(q.size(0), device=q.device)
    return F.cross_entropy(logits, labels)

def _center_gram(K: torch.Tensor) -> torch.Tensor:
    n   = K.size(0)
    one = torch.ones((n, n), device=K.device, dtype=K.dtype) / n
    return K - one @ K - K @ one + one @ K @ one

def linear_cka_loss(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    \"\"\"1 - Linear CKA(X, Y).  X, Y: [N, D] or [N, N] gram matrices.\"\"\"
    if X.dim() == 2 and X.size(0) != X.size(1):
        K = X @ X.T
        L = Y @ Y.T
    else:
        K, L = X, Y
    K = _center_gram(K)
    L = _center_gram(L)
    hsic_kl = (K * L).sum()
    hsic_kk  = (K * K).sum().sqrt().clamp(min=eps)
    hsic_ll  = (L * L).sum().sqrt().clamp(min=eps)
    return 1.0 - hsic_kl / (hsic_kk * hsic_ll)
"""))

# ── 6  Span overlap matrix ────────────────────────────────────────────────────
cells.append(code("""
# ============================================================
# 4) Span overlap alignment matrix A (student tokens × teacher tokens)
# ============================================================
def build_span_overlap_matrix(
    offsets_s: torch.Tensor,   # [L_s, 2]  character-level offsets
    offsets_t: torch.Tensor,   # [L_t, 2]
    eps: float = 1e-12,
) -> torch.Tensor:
    device   = offsets_s.device
    offsets_t = offsets_t.to(device)

    s_start = offsets_s[:, 0].unsqueeze(1)
    s_end   = offsets_s[:, 1].unsqueeze(1)
    t_start = offsets_t[:, 0].unsqueeze(0)
    t_end   = offsets_t[:, 1].unsqueeze(0)

    inter_start = torch.maximum(s_start, t_start)
    inter_end   = torch.minimum(s_end,   t_end)
    inter       = (inter_end - inter_start).clamp(min=0)

    len_s = (s_end - s_start).clamp(min=0)
    len_t = (t_end - t_start).clamp(min=0)
    union = (len_s + len_t - inter).clamp(min=eps)

    A      = inter / union          # IoU  [Ls, Lt]
    row_sum = A.sum(dim=1, keepdim=True).clamp(min=eps)
    return A / row_sum              # row-normalised

def coverage_from_A(A: torch.Tensor, eps: float = 1e-12) -> Tuple[float, float]:
    row_has = (A.sum(dim=1) > eps).float().mean().item()
    col_has = (A.sum(dim=0) > eps).float().mean().item()
    return row_has, col_has
"""))

# ── 7  Attention CKA loss ─────────────────────────────────────────────────────
cells.append(code("""
# ============================================================
# 5) Attention CKA loss (Span-aware)
# ============================================================
def token_importance_from_attention(att: torch.Tensor,
                                    mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    imp = att.abs().sum(dim=0) + att.abs().sum(dim=1)
    if mask is not None:
        imp = imp * mask.float()
    return imp

def select_top_tokens(importance: torch.Tensor, mask: torch.Tensor,
                      top_frac: float, min_tokens: int) -> torch.Tensor:
    valid_idx   = torch.where(mask > 0)[0]
    if valid_idx.numel() == 0:
        return valid_idx
    imp_valid = importance[valid_idx]
    k = max(min_tokens, int(math.ceil(top_frac * valid_idx.numel())))
    k = min(k, valid_idx.numel())
    topk = torch.topk(imp_valid, k=k, largest=True).indices
    return valid_idx[topk]

def compute_span_cka_att_loss(
    att_s:    torch.Tensor,     # [B, H, Ls, Ls]
    att_t:    torch.Tensor,     # [B, H, Lt, Lt]
    offsets_s: torch.Tensor,    # [B, Ls, 2]
    offsets_t: torch.Tensor,    # [B, Lt, 2]
    mask_s:   torch.Tensor,     # [B, Ls]
    mask_t:   torch.Tensor,     # [B, Lt]
    min_coverage: float,
    top_frac:     float,
    min_tokens:   int,
) -> torch.Tensor:
    B = att_s.size(0)
    att_s_mean = att_s.mean(dim=1)   # [B, Ls, Ls]
    att_t_mean = att_t.mean(dim=1)   # [B, Lt, Lt]

    losses = []
    for b in range(B):
        A = build_span_overlap_matrix(offsets_s[b].to(att_s.device),
                                      offsets_t[b].to(att_s.device))
        cov_s, cov_t = coverage_from_A(A)
        conf = min(cov_s, cov_t)
        if conf < min_coverage:
            continue

        t_mask = mask_t[b].to(att_s.device)
        t_imp  = token_importance_from_attention(att_t_mean[b], t_mask)
        t_sel  = select_top_tokens(t_imp, t_mask, top_frac=top_frac, min_tokens=min_tokens)
        if t_sel.numel() < min_tokens:
            continue

        A_sel  = A[:, t_sel]                          # [Ls, M]
        s_has  = (A_sel.sum(dim=1) > 1e-12).float()
        s_mask = mask_s[b].to(att_s.device) * s_has
        if s_mask.sum() < min_tokens:
            continue

        att_t_sub  = att_t_mean[b][t_sel][:, t_sel]             # [M, M]
        att_t_proj = A_sel @ att_t_sub @ A_sel.transpose(0, 1)  # [Ls, Ls]

        s_imp = token_importance_from_attention(att_s_mean[b], s_mask)
        s_sel = select_top_tokens(s_imp, s_mask, top_frac=top_frac, min_tokens=min_tokens)
        if s_sel.numel() < min_tokens:
            continue

        K = att_s_mean[b][s_sel][:, s_sel]
        L = att_t_proj[s_sel][:, s_sel]
        losses.append(linear_cka_loss(K, L) * conf)

    if len(losses) == 0:
        return att_s.new_tensor(0.0)
    return torch.stack(losses).mean()
"""))

# ── 8  Dataset + Collator ─────────────────────────────────────────────────────
cells.append(code("""
# ============================================================
# 6) Dataset + Dual-tokenizer collator (student + teacher + offsets)
# ============================================================
class TextOnlyDataset(Dataset):
    def __init__(self, df: pd.DataFrame, text_col: str):
        self.texts = df[text_col].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {"text": self.texts[idx]}

class DualTokenizerCollate:
    def __init__(self, tok_s, tok_t, max_length: int):
        self.tok_s      = tok_s
        self.tok_t      = tok_t
        self.max_length = max_length

    def _tokenize(self, tokenizer, texts: List[str]) -> Dict[str, torch.Tensor]:
        return tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_offsets_mapping=True,   # requires fast tokenizer
        )

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [x["text"] for x in batch]

        # Two student views (different dropout masks = SimCSE)
        s1 = self._tokenize(self.tok_s, texts)
        s2 = self._tokenize(self.tok_s, texts)
        # One teacher view for SAM
        t1 = self._tokenize(self.tok_t, texts)

        out: Dict[str, Any] = {"texts": texts}
        for k, v in s1.items(): out[f"{k}1_stu"] = v
        for k, v in s2.items(): out[f"{k}2_stu"] = v
        for k, v in t1.items(): out[f"{k}1_tea"] = v
        return out

def load_train_dataframe(path: str, text_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if text_col not in df.columns:
        raise ValueError(f"text_col='{text_col}' not in columns: {list(df.columns)[:30]}")
    return df.dropna(subset=[text_col]).reset_index(drop=True)
"""))

# ── 9  Model loading ──────────────────────────────────────────────────────────
cells.append(code("""
tok_student  = AutoTokenizer.from_pretrained(cfg.student_name, use_fast=True)
tok_teacher  = AutoTokenizer.from_pretrained(cfg.teacher_name, use_fast=True)

model_student = AutoModel.from_pretrained(cfg.student_name, output_hidden_states=True).to(device_s)
model_teacher = AutoModel.from_pretrained(cfg.teacher_name, output_hidden_states=True).to(device_t)

model_teacher.eval()
for p in model_teacher.parameters():
    p.requires_grad_(False)

d_s = model_student.config.hidden_size
d_t = model_teacher.config.hidden_size
print("d_s:", d_s, "| d_t:", d_t)

# Populate nested_dims for eval slicing
if cfg.nested_dims is None:
    base = [16, 32, 64, 128, 256, 512, 1024]
    cfg.nested_dims = [d for d in base if d <= d_s]
    if cfg.nested_dims[-1] != d_s:
        cfg.nested_dims.append(d_s)

print("nested_dims (eval):", cfg.nested_dims)
"""))

# ── 10 Optimizer + DataLoader ─────────────────────────────────────────────────
cells.append(code("""
# ============================================================
# 8) Optimizer + scheduler + loader
# ============================================================
optimizer = torch.optim.AdamW(
    model_student.parameters(),
    lr=cfg.lr,
    weight_decay=cfg.weight_decay,
)

df_train    = load_train_dataframe(cfg.train_csv, cfg.text_col)
train_ds    = TextOnlyDataset(df_train, cfg.text_col)
collate_fn  = DualTokenizerCollate(tok_student, tok_teacher, max_length=cfg.max_length)
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
print("total_steps:", total_steps, "| warmup_steps:", warmup_steps)
"""))

# ── 11 LR schedule helpers ─────────────────────────────────────────────────────
cells.append(code("""
def linear_ramp(step: int, start: int, ramp: int) -> float:
    if ramp <= 0:
        return 1.0 if step >= start else 0.0
    if step < start:
        return 0.0
    return min(1.0, (step - start) / float(ramp))

def get_alpha_attn(step: int) -> float:
    return cfg.alpha_attn_max * linear_ramp(step, cfg.alpha_attn_start, cfg.alpha_attn_ramp)
"""))

# ── 12 Training loop ──────────────────────────────────────────────────────────
cells.append(code("""
from tqdm.auto import tqdm

global_step = 0
model_student.train()

for epoch in range(cfg.epochs):
    pbar    = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
    running = 0.0

    for batch in pbar:
        optimizer.zero_grad(set_to_none=True)

        batch_s = {k: v.to(device_s, non_blocking=True)
                   for k, v in batch.items()
                   if torch.is_tensor(v) and k.endswith("_stu")}
        batch_t = {k: v.to(device_t, non_blocking=True)
                   for k, v in batch.items()
                   if torch.is_tensor(v) and k.endswith("_tea")}

        need_att = (global_step % cfg.att_every == 0)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            # ── Student: two dropout views (SimCSE) ──────────────────────────
            s_out1 = model_student(
                input_ids=batch_s["input_ids1_stu"],
                attention_mask=batch_s["attention_mask1_stu"],
                output_attentions=need_att,
                return_dict=True,
            )
            s_out2 = model_student(
                input_ids=batch_s["input_ids2_stu"],
                attention_mask=batch_s["attention_mask2_stu"],
                output_attentions=False,
                return_dict=True,
            )

            S1 = get_sentence_emb(s_out1.last_hidden_state,
                                  batch_s["attention_mask1_stu"], cfg.student_pool)
            S2 = get_sentence_emb(s_out2.last_hidden_state,
                                  batch_s["attention_mask2_stu"], cfg.student_pool)

            # ── Task loss: plain SimCSE InfoNCE ──────────────────────────────
            task_loss = info_nce(S1, S2, temperature=cfg.temperature)

            # ── SAM loss ─────────────────────────────────────────────────────
            att_loss = S1.new_tensor(0.0)
            if need_att:
                with torch.inference_mode():
                    t_out = model_teacher(
                        input_ids=batch_t["input_ids1_tea"],
                        attention_mask=batch_t["attention_mask1_tea"],
                        output_attentions=True,
                        return_dict=True,
                    )

                if cfg.att_layer == "mid":
                    idx_s = len(s_out1.attentions) // 2
                    idx_t = len(t_out.attentions) // 2
                else:
                    idx_s = -1
                    idx_t = -1

                att_s = s_out1.attentions[idx_s]
                att_t = t_out.attentions[idx_t].to(device_s)

                offsets_s = batch_s["offset_mapping1_stu"]
                offsets_t = batch_t["offset_mapping1_tea"].to(device_s)
                mask_s    = batch_s["attention_mask1_stu"]
                mask_t    = batch_t["attention_mask1_tea"].to(device_s)

                att_loss = compute_span_cka_att_loss(
                    att_s=att_s, att_t=att_t,
                    offsets_s=offsets_s, offsets_t=offsets_t,
                    mask_s=mask_s, mask_t=mask_t,
                    min_coverage=cfg.min_coverage,
                    top_frac=cfg.top_frac,
                    min_tokens=cfg.min_tokens,
                )

            alpha_attn = get_alpha_attn(global_step)
            total_loss = task_loss + alpha_attn * att_loss

        scaler.scale(total_loss).backward()

        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model_student.parameters(), cfg.grad_clip)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running = (0.95 * running + 0.05 * total_loss.item()
                   if global_step > 0 else total_loss.item())
        pbar.set_postfix({
            "loss":  f"{running:.4f}",
            "task":  f"{task_loss.item():.4f}",
            "att":   f"{att_loss.item():.4f}" if need_att else "skip",
            "alpha": f"{alpha_attn:.3f}",
        })

        global_step += 1

print("Done. global_step =", global_step)
"""))

# ── 13 encode_texts ───────────────────────────────────────────────────────────
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
    pool: Optional[str] = None,
    normalize: bool = False,
) -> torch.Tensor:
    model_student.eval()
    if max_length is None:
        max_length = cfg.max_length
    if pool is None:
        pool = cfg.student_pool

    texts   = [str(x) for x in texts]
    all_emb = []

    for i in tqdm(range(0, len(texts), batch_size), desc="encode", leave=False):
        chunk = texts[i:i+batch_size]
        enc   = tok_student(
            chunk, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        enc = {k: v.to(device_s, non_blocking=True) for k, v in enc.items()}
        out = model_student(**enc, return_dict=True)
        emb = get_sentence_emb(out.last_hidden_state, enc["attention_mask"], pool)
        if normalize:
            emb = F.normalize(emb, dim=-1)
        all_emb.append(emb.detach().cpu())

    return torch.cat(all_emb, dim=0)
"""))

# ── 14 Eval helpers ───────────────────────────────────────────────────────────
cells.append(code("""
# ============================================================
# Evaluation helpers (per slice)
# ============================================================
import numpy as np
import pandas as pd

try:
    from scipy.stats import spearmanr
except Exception as e:
    spearmanr = None
    print("[WARN] scipy not available -> STS Spearman will be skipped.", e)

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
except Exception as e:
    LogisticRegression = accuracy_score = f1_score = None
    print("[WARN] scikit-learn not available -> CLS eval will be skipped.", e)

try:
    from IPython.display import display
except Exception:
    display = print

def _safe_spearman(a, b) -> float:
    if spearmanr is None:
        return 0.0
    r = spearmanr(a, b).correlation
    return float(r) if r == r else 0.0

def eval_cls_task(train_csv, test_csv, text_col="text", label_col="label",
                  dims=None, batch_size=256):
    if LogisticRegression is None:
        raise RuntimeError("scikit-learn not available.")
    tr = pd.read_csv(train_csv)
    te = pd.read_csv(test_csv)
    X_tr = encode_texts(tr[text_col].astype(str).tolist(), batch_size=batch_size)
    X_te = encode_texts(te[text_col].astype(str).tolist(), batch_size=batch_size)
    y_tr = tr[label_col].astype(int).values
    y_te = te[label_col].astype(int).values
    rows = []
    for d in dims:
        clf = LogisticRegression(max_iter=2000, n_jobs=1)
        clf.fit(X_tr[:, :d].numpy(), y_tr)
        pred = clf.predict(X_te[:, :d].numpy())
        rows.append({"dim": d, "acc": float(accuracy_score(y_te, pred)),
                     "f1_macro": float(f1_score(y_te, pred, average="macro"))})
    return pd.DataFrame(rows).set_index("dim")

def eval_pair_task(train_csv, test_csv, s1="sentence1", s2="sentence2",
                   label_col="label", dims=None, batch_size=256):
    tr = pd.read_csv(train_csv)
    te = pd.read_csv(test_csv)
    A_tr = encode_texts(tr[s1].astype(str).tolist(), batch_size=batch_size)
    B_tr = encode_texts(tr[s2].astype(str).tolist(), batch_size=batch_size)
    A_te = encode_texts(te[s1].astype(str).tolist(), batch_size=batch_size)
    B_te = encode_texts(te[s2].astype(str).tolist(), batch_size=batch_size)
    y_tr = tr[label_col].astype(int).values
    y_te = te[label_col].astype(int).values
    rows = []
    for d in dims:
        sim_tr = (F.normalize(A_tr[:, :d], dim=-1) * F.normalize(B_tr[:, :d], dim=-1)).sum(dim=-1).numpy()
        sim_te = (F.normalize(A_te[:, :d], dim=-1) * F.normalize(B_te[:, :d], dim=-1)).sum(dim=-1).numpy()
        best_thr, best_acc = 0.0, -1.0
        for thr in np.linspace(-1, 1, 401):
            acc = ((sim_tr >= thr).astype(int) == y_tr).mean()
            if acc > best_acc:
                best_acc, best_thr = acc, thr
        pred = (sim_te >= best_thr).astype(int)
        rows.append({"dim": d, "acc": float((pred == y_te).mean()), "thr": float(best_thr)})
    return pd.DataFrame(rows).set_index("dim")

def eval_sts_task(test_csv, s1="sentence1", s2="sentence2",
                  score_col="score", dims=None, batch_size=256):
    te  = pd.read_csv(test_csv)
    A   = encode_texts(te[s1].astype(str).tolist(), batch_size=batch_size)
    B   = encode_texts(te[s2].astype(str).tolist(), batch_size=batch_size)
    y   = te[score_col].astype(float).values
    rows = []
    for d in dims:
        sim = (F.normalize(A[:, :d], dim=-1) * F.normalize(B[:, :d], dim=-1)).sum(dim=-1).numpy()
        rows.append({"dim": d, "spearman": _safe_spearman(sim, y)})
    return pd.DataFrame(rows).set_index("dim")
"""))

# ── 15 Run eval ───────────────────────────────────────────────────────────────
cells.append(code("""
# ============================================================
# RUN EVAL
# ============================================================
EVAL_ROOT     = "/kaggle/input/multitask-data/multi-data"
STS_EXTRA_ROOT = os.environ.get("STS_EXTRA_ROOT", "")
NESTED_DIMS   = cfg.nested_dims
print("NESTED_DIMS:", NESTED_DIMS)

results = {"cls": {}, "pair": {}, "sts": {}, "sts_extra": {}}

if EVAL_ROOT and os.path.exists(EVAL_ROOT):
    cls_tasks = [
        ("Banking77", "banking_train.csv",  "banking77_test.csv"),
        ("Emotion",   "emotion_train.csv",  "emotion_test.csv"),
        ("TweetEval", "tweet_train.csv",    "tweet_test.csv"),
    ]
    for name, trf, tef in cls_tasks:
        tr = os.path.join(EVAL_ROOT, trf)
        te = os.path.join(EVAL_ROOT, tef)
        if os.path.exists(tr) and os.path.exists(te):
            results["cls"][name] = eval_cls_task(tr, te, text_col="text",
                                                 label_col="label", dims=NESTED_DIMS)
            print(f"[CLS] {name}")
            display(results["cls"][name])

    pair_tasks = [
        ("MRPC",    "mrpc_validation.csv",    "mrpc_test.csv"),
        ("SciTail", "scitail_validation.csv", "scitail_test.csv"),
        ("WiC",     "wic_validation.csv",     "wic_test.csv"),
    ]
    for name, trf, tef in pair_tasks:
        tr = os.path.join(EVAL_ROOT, trf)
        te = os.path.join(EVAL_ROOT, tef)
        if os.path.exists(tr) and os.path.exists(te):
            results["pair"][name] = eval_pair_task(tr, te, s1="sentence1",
                                                   s2="sentence2", label_col="label",
                                                   dims=NESTED_DIMS)
            print(f"[PAIR] {name}")
            display(results["pair"][name])

    sts_tasks = [
        ("SICK",  "sick_test.csv"),
        ("STS12", "sts12_test.csv"),
        ("STS-B", "stsb_test.csv"),
    ]
    for name, tef in sts_tasks:
        te = os.path.join(EVAL_ROOT, tef)
        if os.path.exists(te):
            results["sts"][name] = eval_sts_task(te, s1="sentence1", s2="sentence2",
                                                  score_col="score", dims=NESTED_DIMS)
            print(f"[STS] {name}")
            display(results["sts"][name])
else:
    print("EVAL_ROOT not found; skipping eval.")

if STS_EXTRA_ROOT and os.path.exists(STS_EXTRA_ROOT):
    candidates = [("STS13","sts13.csv"),("STS14","sts14.csv"),
                  ("STS15","sts15.csv"),("STS16","sts16.csv"),("STS17","sts17.csv")]
    for name, fn in candidates:
        p = os.path.join(STS_EXTRA_ROOT, fn)
        if os.path.exists(p):
            results["sts_extra"][name] = eval_sts_task(p, s1="sentence1", s2="sentence2",
                                                        score_col="score", dims=NESTED_DIMS)
            print(f"[STS_EXTRA] {name}")
            display(results["sts_extra"][name])

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

out = r"C:\Users\Thang Tran\SAMD\samd-sam-only.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Written: {out}")
print(f"Total cells: {len(cells)}")
for i, c in enumerate(cells):
    src = "".join(c["source"])[:70].replace("\n", " ")
    print(f"  {i:2d} [{c['cell_type']:8s}] {src}")
