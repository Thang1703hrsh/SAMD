#!/usr/bin/env python
"""Train SAMD (Span-Aware Matryoshka Distillation) from the command line.

Reproduces the CTKD/SAMD.ipynb notebook using only the ``samd`` package.

Usage examples:

    # BGE-M3 -> TinyBERT-4L
    python scripts/train_samd.py \
        --teacher BAAI/bge-m3 \
        --student huawei-noah/TinyBERT_General_4L_312D \
        --train_csv data/merged_9_data_3k_each_ver2.csv \
        --eval_dir  data/multi-data

    # LLM2Vec-Mistral -> TinyBERT-6L (requires 2 GPUs and llm2vec)
    python scripts/train_samd.py \
        --teacher McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp \
        --student huawei-noah/TinyBERT_General_6L_768D \
        --teacher_special "<|embed|>" \
        --train_csv data/merged_9_data_3k_each_ver2.csv

    # Qwen3-Embedding -> BERT-base
    python scripts/train_samd.py \
        --teacher Qwen/Qwen3-Embedding-0.6B \
        --student bert-base-uncased \
        --train_csv data/merged_9_data_3k_each_ver2.csv
"""
from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_scheduler,
)

# ── samd package imports ────────────────────────────────────────────────
from samd import (
    DualTokenizerCollate,
    TextPairRaw,
    compute_span_cka_att_loss,
    eval_classification_task,
    eval_pair_task,
    eval_sts_task,
    extract_teacher_sentence_embedding,
    get_student_sentence_emb,
    info_nce,
    matryoshka_prefix_cosine_loss,
)


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_devices():
    if torch.cuda.device_count() >= 2:
        return torch.device("cuda:0"), torch.device("cuda:1")
    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
        return dev, dev
    return torch.device("cpu"), torch.device("cpu")


def ramp(step: int, start: int, ramp_steps: int) -> float:
    if step < start:
        return 0.0
    if ramp_steps <= 0:
        return 1.0
    return min(1.0, (step - start) / float(ramp_steps))


# ═══════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════

def load_teacher(name: str, device: torch.device, hf_token: str | None = None):
    """Load and freeze a teacher model.  Handles LLM2Vec (PeftModel) automatically."""
    tok = AutoTokenizer.from_pretrained(name, token=hf_token)

    is_llm2vec = "llm2vec" in name.lower()
    if is_llm2vec:
        from peft import PeftModel

        config = AutoConfig.from_pretrained(name, trust_remote_code=True, token=hf_token)
        model = AutoModel.from_pretrained(
            name,
            trust_remote_code=True,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=hf_token,
        )
        # merge MNTP LoRA
        model = PeftModel.from_pretrained(model, name, token=hf_token)
        model = model.merge_and_unload()
        # merge supervised SimCSE LoRA
        sup_name = name.replace("-mntp", "-mntp-supervised")
        model = PeftModel.from_pretrained(model, sup_name, token=hf_token)
        model = model.merge_and_unload()
    else:
        model = AutoModel.from_pretrained(
            name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            token=hf_token,
        )

    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model, tok


def load_student(name: str, device: torch.device, hf_token: str | None = None):
    tok = AutoTokenizer.from_pretrained(name, token=hf_token)
    model = AutoModel.from_pretrained(name, output_hidden_states=True, token=hf_token)
    model.to(device)
    return model, tok


# ═══════════════════════════════════════════════════════════════════════
# Eval paths builder
# ═══════════════════════════════════════════════════════════════════════

def build_eval_paths(eval_dir: str):
    d = eval_dir
    cls_tasks = [
        (f"{d}/banking_train.csv",  f"{d}/banking77_test.csv"),
        (f"{d}/emotion_train.csv",  f"{d}/emotion_test.csv"),
        (f"{d}/tweet_train.csv",    f"{d}/tweet_test.csv"),
    ]
    sts_tasks = [
        f"{d}/sick_test.csv",
        f"{d}/sts12_test.csv",
        f"{d}/stsb_test.csv",
    ]
    pair_tasks = [
        f"{d}/mrpc_test.csv",
        f"{d}/scitail_test.csv",
        f"{d}/wic_test.csv",
    ]
    # filter to only existing files
    cls_tasks  = [(a, b) for a, b in cls_tasks if os.path.exists(a) and os.path.exists(b)]
    sts_tasks  = [p for p in sts_tasks if os.path.exists(p)]
    pair_tasks = [p for p in pair_tasks if os.path.exists(p)]
    return cls_tasks, sts_tasks, pair_tasks


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Train SAMD (CLI)")

    # models
    p.add_argument("--teacher", type=str, required=True, help="HuggingFace teacher model name")
    p.add_argument("--student", type=str, default="huawei-noah/TinyBERT_General_6L_768D")
    p.add_argument("--teacher_special", type=str, default="<|embed|>",
                   help="Special embed token for the teacher (e.g. <|embed|>)")
    p.add_argument("--student_special", type=str, default="[CLS]")
    p.add_argument("--hf_token", type=str, default=None,
                   help="HuggingFace token (or set HF_TOKEN env var)")

    # data
    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--eval_dir", type=str, default="data/multi-data")
    p.add_argument("--task_type", type=str, default="pair_cls",
                   choices=["pair_cls", "pair_reg", "single_cls"])

    # training
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", type=str, default="ckpt")

    # loss weights
    p.add_argument("--w_task", type=float, default=0.5, help="SimCSE InfoNCE weight")
    p.add_argument("--alpha_kd", type=float, default=0.5, help="Total KD weight")
    p.add_argument("--student_pool", type=str, default="mean", choices=["mean", "cls"])

    # MRD schedule
    p.add_argument("--beta_mrl_max", type=float, default=1.0)
    p.add_argument("--mrl_start", type=int, default=0)
    p.add_argument("--mrl_ramp", type=int, default=1000)
    p.add_argument("--matryoshka_dims", type=int, nargs="+", default=[128, 256, 384, 512])

    # SAM schedule
    p.add_argument("--alpha_attn_max", type=float, default=0.02)
    p.add_argument("--att_start", type=int, default=1500)
    p.add_argument("--att_ramp", type=int, default=2500)
    p.add_argument("--att_every", type=int, default=4)
    p.add_argument("--top_frac", type=float, default=0.25)
    p.add_argument("--min_tokens", type=int, default=2)
    p.add_argument("--min_coverage", type=float, default=0.30)
    p.add_argument("--k_layers", type=int, default=1, help="Last-k layers for SAM")

    return p.parse_args()


def main():
    args = parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    set_seed(args.seed)

    device_s, device_t = pick_devices()
    print(f"Devices — student: {device_s}, teacher: {device_t}")

    # ── Load models ─────────────────────────────────────────────────
    print(f"Loading student: {args.student}")
    model_s, tok_s = load_student(args.student, device_s, hf_token)

    print(f"Loading teacher: {args.teacher}")
    model_t, tok_t = load_teacher(args.teacher, device_t, hf_token)

    d_s = model_s.config.hidden_size
    d_t = model_t.config.hidden_size
    print(f"Hidden sizes — student: {d_s}, teacher: {d_t}")

    # matryoshka dims (filtered by student hidden size)
    matryoshka_dims = sorted({d for d in args.matryoshka_dims if d <= d_s} | {d_s})
    print(f"Matryoshka dims: {matryoshka_dims}")

    # projection: teacher -> student embedding space
    proj_t2s = nn.Linear(d_t, d_s, bias=False).to(device_s)

    # ── Data ────────────────────────────────────────────────────────
    print(f"Loading training data: {args.train_csv}")
    df = pd.read_csv(args.train_csv).dropna().reset_index(drop=True)

    if args.task_type == "pair_cls" and "premise" not in df.columns:
        df["premise"] = df["text"]
        df["hypothesis"] = df["text"]
        df = df[["premise", "hypothesis"]].copy()

    train_ds = TextPairRaw(df, args.task_type)
    collate = DualTokenizerCollate(tok_s, tok_t, args.task_type, args.max_length)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate, pin_memory=True, num_workers=2,
        persistent_workers=True, drop_last=True,
    )
    print(f"Training samples: {len(train_ds)}, batches/epoch: {len(train_loader)}")

    # ── Eval paths ──────────────────────────────────────────────────
    cls_tasks, sts_tasks, pair_tasks = build_eval_paths(args.eval_dir)

    # ── Optimizer / scheduler ───────────────────────────────────────
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)

    optimizer = torch.optim.AdamW(
        list(model_s.parameters()) + list(proj_t2s.parameters()),
        lr=args.lr,
    )
    scheduler = get_scheduler(
        name="cosine_with_min_lr",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        scheduler_specific_kwargs={"min_lr": 2e-6},
    )
    scaler = GradScaler(enabled=torch.cuda.is_available())

    print(f"Total steps: {total_steps}, warmup: {warmup_steps}")

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    # ── Training ────────────────────────────────────────────────────
    global_step = 0

    for epoch in range(args.epochs):
        model_s.train()
        total_loss, n_items = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch in pbar:
            # split student / teacher tensors
            batch_s = {k: v.to(device_s, non_blocking=True)
                       for k, v in batch.items()
                       if torch.is_tensor(v) and (k.endswith("_stu") or k == "labels")}
            batch_t = {k: v.to(device_t, non_blocking=True)
                       for k, v in batch.items()
                       if torch.is_tensor(v) and k.endswith("_tea")}

            # scheduled weights
            beta_mrl = args.beta_mrl_max * ramp(global_step, args.mrl_start, args.mrl_ramp)
            alpha_attn = args.alpha_attn_max * ramp(global_step, args.att_start, args.att_ramp)
            need_att = (alpha_attn > 0) and (args.att_every > 0) and (global_step % args.att_every == 0)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=torch.cuda.is_available()):
                # ── Teacher forward (frozen) ────────────────────────
                with torch.inference_mode():
                    t_out = model_t(
                        input_ids=batch_t["input_ids1_tea"],
                        attention_mask=batch_t["attention_mask1_tea"],
                        output_attentions=need_att,
                        return_dict=True,
                    )
                    T_sent = extract_teacher_sentence_embedding(
                        T_last=t_out.last_hidden_state,
                        input_ids=batch_t["input_ids1_tea"],
                        attention_mask=batch_t["attention_mask1_tea"],
                        tok_teacher=tok_t,
                        embed_token=args.teacher_special,
                    )
                    T_sent_s = T_sent.to(device_s, non_blocking=True)
                    T_emb_s = proj_t2s(T_sent_s)  # [B, d_s]

                    if need_att:
                        T_atts = [a.to(device_s, non_blocking=True) for a in t_out.attentions]
                    else:
                        T_atts = None

                # ── Student forward (two SimCSE views) ──────────────
                s_out1 = model_s(
                    input_ids=batch_s["input_ids1_stu"],
                    attention_mask=batch_s["attention_mask1_stu"],
                    output_attentions=need_att,
                    return_dict=True,
                )
                s_out2 = model_s(
                    input_ids=batch_s["input_ids2_stu"],
                    attention_mask=batch_s["attention_mask2_stu"],
                    return_dict=True,
                )

                S_emb1 = get_student_sentence_emb(
                    s_out1.last_hidden_state, batch_s["attention_mask1_stu"], args.student_pool,
                )
                S_emb2 = get_student_sentence_emb(
                    s_out2.last_hidden_state, batch_s["attention_mask2_stu"], args.student_pool,
                )

                # (A) SimCSE InfoNCE
                loss_task, _ = info_nce(S_emb1, S_emb2)

                # (B) MRD: Matryoshka prefix cosine KD
                mrl_1 = matryoshka_prefix_cosine_loss(S_emb1, T_emb_s, dims=matryoshka_dims)
                mrl_2 = matryoshka_prefix_cosine_loss(S_emb2, T_emb_s, dims=matryoshka_dims)
                mrl_loss = 0.5 * (mrl_1 + mrl_2)

                # (C) SAM: span-aware attention alignment
                att_loss = torch.tensor(0.0, device=device_s)
                if need_att:
                    # select last-k layers from each model
                    n_t = len(T_atts)
                    n_s = len(s_out1.attentions)
                    layers_per_block = max(1, n_t // n_s)
                    # map teacher layers to student layers, then take last-k
                    mapped_t = [T_atts[i * layers_per_block + (layers_per_block - 1)]
                                for i in range(n_s)]

                    k = min(args.k_layers, n_s)
                    for t_att, s_att in zip(mapped_t[-k:], list(s_out1.attentions)[-k:]):
                        # use span-overlap alignment from samd package
                        offsets_s = batch.get("offset_mapping1_stu")
                        offsets_t = batch.get("offset_mapping1_tea")

                        if offsets_s is not None and offsets_t is not None:
                            att_loss = att_loss + compute_span_cka_att_loss(
                                att_s=s_att,
                                att_t=t_att,
                                offsets_s=offsets_s.to(device_s),
                                offsets_t=offsets_t.to(device_s),
                                mask_s=batch_s["attention_mask1_stu"],
                                mask_t=batch_t["attention_mask1_tea"].to(device_s),
                                min_coverage=args.min_coverage,
                                top_frac=args.top_frac,
                                min_tokens=args.min_tokens,
                            )
                    att_loss = att_loss / max(k, 1)

                # ── Total loss ──────────────────────────────────────
                kd_sum = alpha_attn * att_loss + beta_mrl * mrl_loss
                loss = (args.w_task * loss_task) + (args.alpha_kd * kd_sum)
                loss = loss.float()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            bs = batch_s["input_ids1_stu"].size(0)
            total_loss += loss.item() * bs
            n_items += bs
            avg_loss = total_loss / max(1, n_items)

            pbar.set_postfix({
                "avg": f"{avg_loss:.4f}",
                "task": f"{loss_task.detach().float().item():.3f}",
                "mrl": f"{mrl_loss.detach().float().item():.3f}",
                "att": f"{att_loss.detach().float().item():.3f}",
                "b_mrl": f"{beta_mrl:.2f}",
                "a_att": f"{alpha_attn:.3f}",
            })

            global_step += 1

            del s_out1, s_out2, S_emb1, S_emb2, T_sent, T_sent_s, T_emb_s, t_out
            if need_att and T_atts is not None:
                del T_atts
            torch.cuda.empty_cache()

        # ── End-of-epoch evaluation ─────────────────────────────────
        print(f"\n--- Epoch {epoch + 1} evaluation ---")
        if cls_tasks:
            eval_classification_task(model_s, cls_tasks, tok_s)
        if pair_tasks:
            eval_pair_task(model_s, pair_tasks, tok_s)
        if sts_tasks:
            eval_sts_task(model_s, sts_tasks, tok_s)

        # save checkpoint
        if args.save_dir:
            ckpt_path = os.path.join(args.save_dir, f"epoch_{epoch + 1}")
            os.makedirs(ckpt_path, exist_ok=True)
            model_s.save_pretrained(ckpt_path)
            tok_s.save_pretrained(ckpt_path)
            torch.save(proj_t2s.state_dict(), os.path.join(ckpt_path, "proj_t2s.pt"))
            print(f"Checkpoint saved: {ckpt_path}")

    print(f"\nDone. global_step = {global_step}")


if __name__ == "__main__":
    main()
