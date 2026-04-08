"""Generate samd-sam-only.ipynb — SAM-only SAMD (no MRD)."""
import json, pathlib

NB_PATH = pathlib.Path(r"C:\Users\Thang Tran\SAMD\samd-sam-only.ipynb")

def code(src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src,
    }

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


# ── CELLS ────────────────────────────────────────────────────────────────────────

CELL_MD_TITLE = md(
    "# SAMD — SAM-only (no MRD)\n\n"
    "Training objective:\n\n"
    "$$\\mathcal{L} = \\mathcal{L}_{\\text{SimCSE}} + \\lambda_{\\text{SAM}} \\cdot \\mathcal{L}_{\\text{SAM}}$$\n\n"
    "- **$\\mathcal{L}_{\\text{SimCSE}}$** — unsupervised SimCSE InfoNCE on the student.\n"
    "- **$\\mathcal{L}_{\\text{SAM}}$** — Span-Aware attention Matching: "
    "project frozen-teacher attention into student token space via character-span overlap matrix "
    "$\\mathbf{A}$, then minimize CKA distance.\n\n"
    "MRD (Multi-Resolution Distillation / prefix cosine KD) is **disabled**."
)

CELL_SETUP = code(
    "import os\n"
    "import math\n"
    "import random\n"
    "from dataclasses import dataclass\n"
    "from typing import List, Dict, Any, Optional, Tuple\n"
    "\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "\n"
    "import torch\n"
    "import torch.nn as nn\n"
    "import torch.nn.functional as F\n"
    "from torch.utils.data import Dataset, DataLoader\n"
    "\n"
    "from transformers import (\n"
    "    AutoTokenizer, AutoModel,\n"
    "    get_cosine_with_hard_restarts_schedule_with_warmup,\n"
    ")"
)

CELL_SEED = code(
    "def set_seed(seed: int = 42) -> None:\n"
    "    random.seed(seed)\n"
    "    np.random.seed(seed)\n"
    "    torch.manual_seed(seed)\n"
    "    torch.cuda.manual_seed_all(seed)\n"
    "\n"
    "set_seed(42)\n"
    'print("torch:", torch.__version__)'
)

CELL_CONFIG = code(
    "@dataclass\n"
    "class Config:\n"
    "    # ── Models ──────────────────────────────────────────────────────\n"
    '    student_name: str = "bert-base-uncased"\n'
    '    teacher_name: str = "bert-base-uncased"   # frozen; swap for cross-tokenizer\n'
    "\n"
    "    # ── Data ────────────────────────────────────────────────────────\n"
    '    train_csv: str = "/kaggle/input/multitask-data/merged_9_data_3k_each_ver2.csv"\n'
    '    text_col: str = "text"\n'
    "    max_length: int = 96           # shorter seqs -> larger batch fits in VRAM\n"
    "\n"
    "    # ── Training ────────────────────────────────────────────────────\n"
    "    batch_size: int = 32           # more in-batch negatives for InfoNCE\n"
    "    epochs: int = 10\n"
    "    lr: float = 3e-5\n"
    "    weight_decay: float = 0.01\n"
    "    temperature: float = 0.05      # SimCSE standard temperature\n"
    "    grad_clip: float = 1.0\n"
    "\n"
    "    # ── Pooling ─────────────────────────────────────────────────────\n"
    '    student_pool: str = "mean"     # mean pooling > CLS for sentence embeddings\n'
    "\n"
    "    # ── SAM hyperparameters ─────────────────────────────────────────\n"
    "    alpha_attn_max: float = 0.5    # SAM loss weight (sole distillation signal)\n"
    "    alpha_attn_start: int = 0      # start SAM from step 0\n"
    "    alpha_attn_ramp: int = 500     # linearly ramp up over 500 steps\n"
    "    att_every: int = 2             # compute SAM every N steps (cost/benefit)\n"
    '    att_layer: str = "last"        # "last" or "mid"\n'
    "    min_coverage: float = 0.30     # skip batch items with poor span coverage\n"
    "    top_frac: float = 0.7          # fraction of most-attended tokens used in CKA\n"
    "    min_tokens: int = 4            # minimum tokens required after selection\n"
    "\n"
    "    # ── Scheduler ───────────────────────────────────────────────────\n"
    "    warmup_ratio: float = 0.06\n"
    "    num_restarts: int = 1\n"
    "\n"
    "    # nested_dims kept only for eval slicing\n"
    "    nested_dims: Optional[List[int]] = None\n"
    "\n"
    "cfg = Config()\n"
    "\n"
    'device_s = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")\n'
    'device_t = torch.device("cuda:1" if torch.cuda.device_count() > 1 else device_s)\n'
    'print("device_s:", device_s, "| device_t:", device_t)'
)

CELL_POOL = code(
    "def mean_pooling(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:\n"
    "    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)\n"
    "    return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)\n"
    "\n"
    "def get_sentence_emb(\n"
    "    last_hidden: torch.Tensor,\n"
    "    attention_mask: torch.Tensor,\n"
    "    mode: str,\n"
    ") -> torch.Tensor:\n"
    '    if mode == "mean":\n'
    "        return mean_pooling(last_hidden, attention_mask)\n"
    "    return last_hidden[:, 0, :]  # CLS"
)

CELL_LOSSES = code(
    "def info_nce(q: torch.Tensor, k: torch.Tensor, temperature: float) -> torch.Tensor:\n"
    "    q = F.normalize(q, dim=-1)\n"
    "    k = F.normalize(k, dim=-1)\n"
    "    logits = (q @ k.T) / temperature\n"
    "    labels = torch.arange(q.size(0), device=q.device)\n"
    "    return F.cross_entropy(logits, labels)\n"
    "\n"
    "\n"
    "def _center_gram(K: torch.Tensor) -> torch.Tensor:\n"
    "    n = K.size(0)\n"
    "    one = torch.ones(n, n, device=K.device, dtype=K.dtype) / n\n"
    "    return K - one @ K - K @ one + one @ K @ one\n"
    "\n"
    "\n"
    "def linear_cka_loss(K: torch.Tensor, L: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:\n"
    "    Kc, Lc = _center_gram(K), _center_gram(L)\n"
    "    hsic   = (Kc * Lc).sum()\n"
    "    norm_k = (Kc * Kc).sum().clamp_min(eps).sqrt()\n"
    "    norm_l = (Lc * Lc).sum().clamp_min(eps).sqrt()\n"
    "    return 1.0 - (hsic / (norm_k * norm_l)).clamp(0.0, 1.0)"
)

CELL_SPAN = code(
    "def build_span_overlap_matrix(\n"
    "    offsets_s: torch.Tensor,   # [L_s, 2]  character offsets for student tokens\n"
    "    offsets_t: torch.Tensor,   # [L_t, 2]  character offsets for teacher tokens\n"
    "    eps: float = 1e-12,\n"
    ") -> torch.Tensor:             # [L_s, L_t]  row-normalised alignment A\n"
    "    # Eq. 5-7 of the SAMD paper:\n"
    "    #   A_raw[j,i] = |[c_j,d_j) cap [a_i,b_i)|  (raw character overlap length)\n"
    "    #   A[j,i]     = A_raw[j,i] / sum_i A_raw[j,i]  (row-normalised)\n"
    "    offsets_t = offsets_t.to(offsets_s.device)\n"
    "\n"
    "    s_start = offsets_s[:, 0].unsqueeze(1)   # [Ls, 1]\n"
    "    s_end   = offsets_s[:, 1].unsqueeze(1)\n"
    "    t_start = offsets_t[:, 0].unsqueeze(0)   # [1, Lt]\n"
    "    t_end   = offsets_t[:, 1].unsqueeze(0)\n"
    "\n"
    "    overlap = (torch.minimum(s_end, t_end) - torch.maximum(s_start, t_start)).clamp(min=0)\n"
    "    row_sum = overlap.sum(dim=1, keepdim=True).clamp(min=eps)\n"
    "    return overlap / row_sum   # [Ls, Lt]\n"
    "\n"
    "\n"
    "def coverage_conf(A_raw: torch.Tensor, eps: float = 1e-12) -> float:\n"
    "    cov_s = (A_raw.sum(1) > eps).float().mean().item()\n"
    "    cov_t = (A_raw.sum(0) > eps).float().mean().item()\n"
    "    return min(cov_s, cov_t)"
)

CELL_SAM = code(
    "def token_importance(att_mean: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:\n"
    "    imp = att_mean.abs().sum(0) + att_mean.abs().sum(1)\n"
    "    return imp * mask.float()\n"
    "\n"
    "\n"
    "def top_tokens(importance: torch.Tensor, mask: torch.Tensor,\n"
    "               frac: float, min_k: int) -> torch.Tensor:\n"
    "    valid = torch.where(mask > 0)[0]\n"
    "    if valid.numel() == 0:\n"
    "        return valid\n"
    "    k = min(valid.numel(), max(min_k, math.ceil(frac * valid.numel())))\n"
    "    idx = torch.topk(importance[valid], k=k).indices\n"
    "    return valid[idx]\n"
    "\n"
    "\n"
    "def compute_sam_loss(\n"
    "    att_s: torch.Tensor,       # [B, H, Ls, Ls] student attentions\n"
    "    att_t: torch.Tensor,       # [B, H, Lt, Lt] teacher attentions\n"
    "    offsets_s: torch.Tensor,   # [B, Ls, 2]\n"
    "    offsets_t: torch.Tensor,   # [B, Lt, 2]\n"
    "    mask_s: torch.Tensor,      # [B, Ls]\n"
    "    mask_t: torch.Tensor,      # [B, Lt]\n"
    "    min_coverage: float,\n"
    "    top_frac: float,\n"
    "    min_tokens: int,\n"
    ") -> torch.Tensor:\n"
    "    B = att_s.size(0)\n"
    "    att_s_mean = att_s.mean(1)   # [B, Ls, Ls]  head-averaged\n"
    "    att_t_mean = att_t.mean(1)   # [B, Lt, Lt]\n"
    "\n"
    "    losses = []\n"
    "    for b in range(B):\n"
    "        # build raw overlap (unnormalised) to compute coverage\n"
    "        os_b = offsets_s[b].to(att_s.device)\n"
    "        ot_b = offsets_t[b].to(att_s.device)\n"
    "        s_start = os_b[:, 0].unsqueeze(1)\n"
    "        s_end   = os_b[:, 1].unsqueeze(1)\n"
    "        t_start = ot_b[:, 0].unsqueeze(0)\n"
    "        t_end   = ot_b[:, 1].unsqueeze(0)\n"
    "        A_raw = (torch.minimum(s_end, t_end) - torch.maximum(s_start, t_start)).clamp(min=0)\n"
    "\n"
    "        conf = coverage_conf(A_raw)\n"
    "        if conf < min_coverage:\n"
    "            continue\n"
    "\n"
    "        # row-normalised alignment matrix\n"
    "        A = A_raw / A_raw.sum(1, keepdim=True).clamp(min=1e-12)  # [Ls, Lt]\n"
    "\n"
    "        # select top-attended teacher tokens\n"
    "        t_mask = mask_t[b].to(att_s.device)\n"
    "        t_imp  = token_importance(att_t_mean[b], t_mask)\n"
    "        t_sel  = top_tokens(t_imp, t_mask, top_frac, min_tokens)\n"
    "        if t_sel.numel() < min_tokens:\n"
    "            continue\n"
    "\n"
    "        # restrict A to selected teacher columns\n"
    "        A_sel = A[:, t_sel]                                      # [Ls, M]\n"
    "        s_has = (A_sel.sum(1) > 1e-12).float()\n"
    "        s_mask = mask_s[b].to(att_s.device) * s_has\n"
    "        if s_mask.sum() < min_tokens:\n"
    "            continue\n"
    "\n"
    "        # project teacher attention: T_tilde = A_hat @ T[sel,sel] @ A_hat^T\n"
    "        att_t_sub  = att_t_mean[b][t_sel][:, t_sel]             # [M, M]\n"
    "        att_t_proj = A_sel @ att_t_sub @ A_sel.t()              # [Ls, Ls]\n"
    "\n"
    "        # select top-attended student tokens\n"
    "        s_imp = token_importance(att_s_mean[b], s_mask)\n"
    "        s_sel = top_tokens(s_imp, s_mask, top_frac, min_tokens)\n"
    "        if s_sel.numel() < min_tokens:\n"
    "            continue\n"
    "\n"
    "        K = att_s_mean[b][s_sel][:, s_sel]     # student sub-attention\n"
    "        L = att_t_proj[s_sel][:, s_sel]         # projected teacher sub-attention\n"
    "        losses.append(linear_cka_loss(K, L) * conf)\n"
    "\n"
    "    if not losses:\n"
    "        return att_s.new_tensor(0.0)\n"
    "    return torch.stack(losses).mean()"
)

CELL_DATASET = code(
    "class TextOnlyDataset(Dataset):\n"
    "    def __init__(self, df: pd.DataFrame, text_col: str):\n"
    "        self.texts = df[text_col].astype(str).tolist()\n"
    "\n"
    "    def __len__(self) -> int:\n"
    "        return len(self.texts)\n"
    "\n"
    "    def __getitem__(self, idx: int) -> Dict[str, Any]:\n"
    '        return {"text": self.texts[idx]}\n'
    "\n"
    "\n"
    "class DualTokenizerCollate:\n"
    "    def __init__(self, tok_s, tok_t, max_length: int):\n"
    "        self.tok_s = tok_s\n"
    "        self.tok_t = tok_t\n"
    "        self.max_length = max_length\n"
    "\n"
    "    def _tok(self, tokenizer, texts):\n"
    "        return tokenizer(\n"
    "            texts, padding=True, truncation=True,\n"
    "            max_length=self.max_length, return_tensors='pt',\n"
    "            return_offsets_mapping=True,\n"
    "        )\n"
    "\n"
    "    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:\n"
    '        texts = [x["text"] for x in batch]\n'
    "        s1 = self._tok(self.tok_s, texts)   # student view 1 (dropout A)\n"
    "        s2 = self._tok(self.tok_s, texts)   # student view 2 (dropout B)\n"
    "        t1 = self._tok(self.tok_t, texts)   # teacher view  (for SAM offsets)\n"
    "\n"
    '        out: Dict[str, Any] = {"texts": texts}\n'
    "        for k, v in s1.items(): out[f\"{k}_s1\"] = v\n"
    "        for k, v in s2.items(): out[f\"{k}_s2\"] = v\n"
    "        for k, v in t1.items(): out[f\"{k}_t\"]  = v\n"
    "        return out"
)

CELL_MODELS = code(
    "tok_s = AutoTokenizer.from_pretrained(cfg.student_name, use_fast=True)\n"
    "tok_t = AutoTokenizer.from_pretrained(cfg.teacher_name, use_fast=True)\n"
    "\n"
    "model_s = AutoModel.from_pretrained(cfg.student_name).to(device_s)\n"
    "model_t = AutoModel.from_pretrained(cfg.teacher_name).to(device_t)\n"
    "\n"
    "model_t.eval()\n"
    "for p in model_t.parameters():\n"
    "    p.requires_grad_(False)\n"
    "\n"
    "d_s = model_s.config.hidden_size\n"
    "d_t = model_t.config.hidden_size\n"
    'print("d_s:", d_s, "| d_t:", d_t)\n'
    "\n"
    "if cfg.nested_dims is None:\n"
    "    base = [32, 64, 128, 256, 512, 768, 1024]\n"
    "    cfg.nested_dims = [d for d in base if d <= d_s]\n"
    "    if not cfg.nested_dims or cfg.nested_dims[-1] != d_s:\n"
    "        cfg.nested_dims.append(d_s)\n"
    "\n"
    'print("nested_dims (eval):", cfg.nested_dims)'
)

CELL_OPTIM = code(
    "df_train = pd.read_csv(cfg.train_csv).dropna(subset=[cfg.text_col]).reset_index(drop=True)\n"
    "train_ds = TextOnlyDataset(df_train, cfg.text_col)\n"
    "collate  = DualTokenizerCollate(tok_s, tok_t, cfg.max_length)\n"
    "train_loader = DataLoader(\n"
    "    train_ds, batch_size=cfg.batch_size, shuffle=True,\n"
    "    num_workers=2, collate_fn=collate, drop_last=True,\n"
    ")\n"
    "\n"
    "optimizer = torch.optim.AdamW(\n"
    "    model_s.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay\n"
    ")\n"
    "\n"
    "total_steps  = cfg.epochs * len(train_loader)\n"
    "warmup_steps = int(cfg.warmup_ratio * total_steps)\n"
    "scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(\n"
    "    optimizer, warmup_steps, total_steps, num_cycles=cfg.num_restarts\n"
    ")\n"
    "\n"
    "scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())\n"
    'print("total_steps:", total_steps, "| warmup_steps:", warmup_steps)'
)

CELL_SCHEDULE = code(
    "def linear_ramp(step: int, start: int, ramp: int) -> float:\n"
    "    if step < start:\n"
    "        return 0.0\n"
    "    if ramp <= 0:\n"
    "        return 1.0\n"
    "    return min(1.0, (step - start) / ramp)\n"
    "\n"
    "def alpha_attn(step: int) -> float:\n"
    "    return cfg.alpha_attn_max * linear_ramp(step, cfg.alpha_attn_start, cfg.alpha_attn_ramp)"
)

CELL_TRAIN = code(
    "from tqdm.auto import tqdm\n"
    "\n"
    "global_step = 0\n"
    "model_s.train()\n"
    "\n"
    "for epoch in range(cfg.epochs):\n"
    '    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")\n'
    "    ema_loss = 0.0\n"
    "\n"
    "    for batch in pbar:\n"
    "        optimizer.zero_grad(set_to_none=True)\n"
    "\n"
    "        # move student and teacher tensors to the right devices\n"
    "        s1 = {k.removesuffix('_s1'): v.to(device_s) for k, v in batch.items()\n"
    "              if torch.is_tensor(v) and k.endswith('_s1')}\n"
    "        s2 = {k.removesuffix('_s2'): v.to(device_s) for k, v in batch.items()\n"
    "              if torch.is_tensor(v) and k.endswith('_s2')}\n"
    "        t1 = {k.removesuffix('_t'):  v.to(device_t) for k, v in batch.items()\n"
    "              if torch.is_tensor(v) and k.endswith('_t')}\n"
    "\n"
    "        need_sam = (global_step % cfg.att_every == 0)\n"
    "\n"
    "        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):\n"
    "            # ── student: two SimCSE dropout views ───────────────────────────\n"
    "            out_s1 = model_s(\n"
    "                input_ids=s1['input_ids'],\n"
    "                attention_mask=s1['attention_mask'],\n"
    "                output_attentions=need_sam,\n"
    "                return_dict=True,\n"
    "            )\n"
    "            out_s2 = model_s(\n"
    "                input_ids=s2['input_ids'],\n"
    "                attention_mask=s2['attention_mask'],\n"
    "                output_attentions=False,\n"
    "                return_dict=True,\n"
    "            )\n"
    "\n"
    "            S1 = get_sentence_emb(out_s1.last_hidden_state, s1['attention_mask'], cfg.student_pool)\n"
    "            S2 = get_sentence_emb(out_s2.last_hidden_state, s2['attention_mask'], cfg.student_pool)\n"
    "\n"
    "            # ── SimCSE task loss ────────────────────────────────────────────\n"
    "            task_loss = info_nce(S1, S2, cfg.temperature)\n"
    "\n"
    "            # ── SAM loss ────────────────────────────────────────────────────\n"
    "            sam_loss = S1.new_tensor(0.0)\n"
    "            if need_sam:\n"
    "                with torch.inference_mode():\n"
    "                    out_t = model_t(\n"
    "                        input_ids=t1['input_ids'],\n"
    "                        attention_mask=t1['attention_mask'],\n"
    "                        output_attentions=True,\n"
    "                        return_dict=True,\n"
    "                    )\n"
    "\n"
    '                idx_s = -1 if cfg.att_layer == "last" else len(out_s1.attentions) // 2\n'
    '                idx_t = -1 if cfg.att_layer == "last" else len(out_t.attentions) // 2\n'
    "\n"
    "                sam_loss = compute_sam_loss(\n"
    "                    att_s    = out_s1.attentions[idx_s],\n"
    "                    att_t    = out_t.attentions[idx_t].to(device_s),\n"
    "                    offsets_s= s1['offset_mapping'],\n"
    "                    offsets_t= t1['offset_mapping'].to(device_s),\n"
    "                    mask_s   = s1['attention_mask'],\n"
    "                    mask_t   = t1['attention_mask'].to(device_s),\n"
    "                    min_coverage=cfg.min_coverage,\n"
    "                    top_frac =cfg.top_frac,\n"
    "                    min_tokens=cfg.min_tokens,\n"
    "                )\n"
    "\n"
    "            lam = alpha_attn(global_step)\n"
    "            total_loss = task_loss + lam * sam_loss\n"
    "\n"
    "        scaler.scale(total_loss).backward()\n"
    "        if cfg.grad_clip > 0:\n"
    "            scaler.unscale_(optimizer)\n"
    "            torch.nn.utils.clip_grad_norm_(model_s.parameters(), cfg.grad_clip)\n"
    "        scaler.step(optimizer)\n"
    "        scaler.update()\n"
    "        scheduler.step()\n"
    "\n"
    "        ema_loss = 0.95 * ema_loss + 0.05 * total_loss.item() if global_step else total_loss.item()\n"
    "        pbar.set_postfix({\n"
    '            "loss": f"{ema_loss:.4f}",\n'
    '            "task": f"{task_loss.item():.4f}",\n'
    '            "sam":  f"{sam_loss.item():.4f}" if need_sam else "skip",\n'
    '            "lam":  f"{lam:.3f}",\n'
    "        })\n"
    "        global_step += 1\n"
    "\n"
    'print("Done. global_step =", global_step)'
)

CELL_ENCODE = code(
    "from tqdm.auto import tqdm\n"
    "\n"
    "@torch.no_grad()\n"
    "def encode_texts(\n"
    "    texts,\n"
    "    batch_size: int = 256,\n"
    "    max_length: Optional[int] = None,\n"
    "    normalize: bool = False,\n"
    ") -> torch.Tensor:\n"
    "    model_s.eval()\n"
    "    if max_length is None:\n"
    "        max_length = cfg.max_length\n"
    "    texts = [str(x) for x in texts]\n"
    "    embs = []\n"
    "    for i in tqdm(range(0, len(texts), batch_size), desc='encode', leave=False):\n"
    "        enc = tok_s(\n"
    "            texts[i:i+batch_size], padding=True, truncation=True,\n"
    "            max_length=max_length, return_tensors='pt',\n"
    "        )\n"
    "        enc = {k: v.to(device_s) for k, v in enc.items()}\n"
    "        out = model_s(**enc, return_dict=True)\n"
    "        emb = get_sentence_emb(out.last_hidden_state, enc['attention_mask'], cfg.student_pool)\n"
    "        if normalize:\n"
    "            emb = F.normalize(emb, dim=-1)\n"
    "        embs.append(emb.cpu())\n"
    "    return torch.cat(embs, 0)"
)

CELL_EVAL_HELPERS = code(
    "import numpy as np\n"
    "\n"
    "try:\n"
    "    from scipy.stats import spearmanr\n"
    "except ImportError:\n"
    "    spearmanr = None\n"
    "\n"
    "try:\n"
    "    from sklearn.linear_model import LogisticRegression\n"
    "    from sklearn.metrics import accuracy_score, f1_score\n"
    "except ImportError:\n"
    "    LogisticRegression = accuracy_score = f1_score = None\n"
    "\n"
    "try:\n"
    "    from IPython.display import display\n"
    "except ImportError:\n"
    "    display = print\n"
    "\n"
    "\n"
    "def _spearman(a, b):\n"
    "    if spearmanr is None: return 0.0\n"
    "    r = spearmanr(a, b).correlation\n"
    "    return float(r) if r == r else 0.0\n"
    "\n"
    "\n"
    "def eval_cls(train_csv, test_csv, text_col='text', label_col='label', dims=None, bs=256):\n"
    "    tr, te = pd.read_csv(train_csv), pd.read_csv(test_csv)\n"
    "    Xtr = encode_texts(tr[text_col].tolist(), batch_size=bs)\n"
    "    Xte = encode_texts(te[text_col].tolist(), batch_size=bs)\n"
    "    ytr, yte = tr[label_col].astype(int).values, te[label_col].astype(int).values\n"
    "    rows = []\n"
    "    for d in (dims or cfg.nested_dims):\n"
    "        clf = LogisticRegression(max_iter=2000)\n"
    "        clf.fit(Xtr[:, :d].numpy(), ytr)\n"
    "        pred = clf.predict(Xte[:, :d].numpy())\n"
    "        rows.append({'dim': d, 'acc': accuracy_score(yte, pred),\n"
    "                     'f1': f1_score(yte, pred, average='macro')})\n"
    "    return pd.DataFrame(rows).set_index('dim')\n"
    "\n"
    "\n"
    "def eval_sts(test_csv, s1='sentence1', s2='sentence2', score_col='score', dims=None, bs=256):\n"
    "    te = pd.read_csv(test_csv)\n"
    "    A = encode_texts(te[s1].tolist(), batch_size=bs)\n"
    "    B = encode_texts(te[s2].tolist(), batch_size=bs)\n"
    "    y = te[score_col].astype(float).values\n"
    "    rows = []\n"
    "    for d in (dims or cfg.nested_dims):\n"
    "        sim = (F.normalize(A[:, :d], dim=-1) * F.normalize(B[:, :d], dim=-1)).sum(-1).numpy()\n"
    "        rows.append({'dim': d, 'spearman': _spearman(sim, y)})\n"
    "    return pd.DataFrame(rows).set_index('dim')\n"
    "\n"
    "\n"
    "def eval_pair(train_csv, test_csv, s1='sentence1', s2='sentence2',\n"
    "              label_col='label', dims=None, bs=256):\n"
    "    tr, te = pd.read_csv(train_csv), pd.read_csv(test_csv)\n"
    "    Atr = encode_texts(tr[s1].tolist(), batch_size=bs)\n"
    "    Btr = encode_texts(tr[s2].tolist(), batch_size=bs)\n"
    "    Ate = encode_texts(te[s1].tolist(), batch_size=bs)\n"
    "    Bte = encode_texts(te[s2].tolist(), batch_size=bs)\n"
    "    ytr = tr[label_col].astype(int).values\n"
    "    yte = te[label_col].astype(int).values\n"
    "    rows = []\n"
    "    for d in (dims or cfg.nested_dims):\n"
    "        sim_tr = (F.normalize(Atr[:, :d], -1) * F.normalize(Btr[:, :d], -1)).sum(-1).numpy()\n"
    "        sim_te = (F.normalize(Ate[:, :d], -1) * F.normalize(Bte[:, :d], -1)).sum(-1).numpy()\n"
    "        best_thr = max(np.linspace(-1, 1, 401),\n"
    "                       key=lambda t: ((sim_tr >= t).astype(int) == ytr).mean())\n"
    "        pred = (sim_te >= best_thr).astype(int)\n"
    "        rows.append({'dim': d, 'acc': (pred == yte).mean(), 'thr': best_thr})\n"
    "    return pd.DataFrame(rows).set_index('dim')\n"
)

CELL_RUN_EVAL = code(
    "EVAL_ROOT = '/kaggle/input/multitask-data/multi-data'\n"
    "\n"
    "if os.path.exists(EVAL_ROOT):\n"
    "    for name, trf, tef in [\n"
    '        ("Banking77",  "banking_train.csv",  "banking77_test.csv"),\n'
    '        ("Emotion",    "emotion_train.csv",  "emotion_test.csv"),\n'
    '        ("TweetEval",  "tweet_train.csv",    "tweet_test.csv"),\n'
    "    ]:\n"
    "        tr = os.path.join(EVAL_ROOT, trf)\n"
    "        te = os.path.join(EVAL_ROOT, tef)\n"
    "        if os.path.exists(tr) and os.path.exists(te):\n"
    '            print(f"[CLS] {name}")\n'
    "            display(eval_cls(tr, te))\n"
    "\n"
    "    for name, trf, tef in [\n"
    '        ("MRPC",    "mrpc_validation.csv",    "mrpc_test.csv"),\n'
    '        ("SciTail", "scitail_validation.csv", "scitail_test.csv"),\n'
    '        ("WiC",     "wic_validation.csv",     "wic_test.csv"),\n'
    "    ]:\n"
    "        tr = os.path.join(EVAL_ROOT, trf)\n"
    "        te = os.path.join(EVAL_ROOT, tef)\n"
    "        if os.path.exists(tr) and os.path.exists(te):\n"
    '            print(f"[PAIR] {name}")\n'
    "            display(eval_pair(tr, te))\n"
    "\n"
    "    for name, tef in [\n"
    '        ("SICK",  "sick_test.csv"),\n'
    '        ("STS12", "sts12_test.csv"),\n'
    '        ("STS-B", "stsb_test.csv"),\n'
    "    ]:\n"
    "        te = os.path.join(EVAL_ROOT, tef)\n"
    "        if os.path.exists(te):\n"
    '            print(f"[STS] {name}")\n'
    "            display(eval_sts(te))\n"
    "else:\n"
    '    print("EVAL_ROOT not found; skipping eval.")'
)

# ── ASSEMBLE NOTEBOOK ────────────────────────────────────────────────────────────

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": [
        CELL_MD_TITLE,
        CELL_SETUP,
        CELL_SEED,
        CELL_CONFIG,
        CELL_POOL,
        CELL_LOSSES,
        CELL_SPAN,
        CELL_SAM,
        CELL_DATASET,
        CELL_MODELS,
        CELL_OPTIM,
        CELL_SCHEDULE,
        CELL_TRAIN,
        CELL_ENCODE,
        CELL_EVAL_HELPERS,
        CELL_RUN_EVAL,
    ],
}

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Written: {NB_PATH}")
print(f"Total cells: {len(nb['cells'])}")
