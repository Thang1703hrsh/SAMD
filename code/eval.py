"""Evaluation functions for classification, sentence-pair, and STS tasks.

All three task families are evaluated using frozen student embeddings:
- eval_classification_task: fits LogisticRegression on CLS embeddings, reports accuracy + F1.
- eval_pair_task: sweeps a cosine threshold, reports accuracy / F1 / AP.
- eval_sts_task: computes cosine similarity and reports Spearman correlation.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from code.data import ClasssifyDataset, PairDataset, STSDataset


def collate_fn(batch: list, tokenizer, max_len: int = 128) -> dict:
    """Collate sentence-pair samples into padded tensors."""
    s1_list = [item["sentence1"] for item in batch]
    s2_list = [item["sentence2"] for item in batch]
    labels  = torch.stack([item["label"] for item in batch])

    enc1 = tokenizer(s1_list, truncation=True, padding=True,
                     max_length=max_len, return_tensors="pt")
    enc2 = tokenizer(s2_list, truncation=True, padding=True,
                     max_length=max_len, return_tensors="pt")

    return {
        "input_ids1":      enc1["input_ids"],
        "attention_mask1": enc1["attention_mask"],
        "input_ids2":      enc2["input_ids"],
        "attention_mask2": enc2["attention_mask"],
        "labels":          labels,
    }


def clf_collate_fn(batch: list, tokenizer, max_len: int = 512) -> dict:
    """Collate single-sentence classification samples into padded tensors."""
    s1_list = [item["text"] for item in batch]
    labels  = torch.stack([item["label"] for item in batch])
    enc1 = tokenizer(s1_list, truncation=True, padding=True,
                     max_length=max_len, return_tensors="pt")
    return {
        "input_ids1":      enc1["input_ids"],
        "attention_mask1": enc1["attention_mask"],
        "labels":          labels,
    }


def _extract_embeddings(model, loader) -> Tuple[List, List]:
    """Run model over loader and collect CLS embeddings and labels."""
    preds, labels = [], []
    device = model.device
    with torch.cuda.amp.autocast(dtype=torch.float16):
        with torch.no_grad():
            for batch in tqdm(loader, leave=False):
                out = model(
                    input_ids=batch["input_ids1"].to(device),
                    attention_mask=batch["attention_mask1"].to(device),
                )
                preds.extend(out.last_hidden_state[:, 0, :].cpu().numpy())
                labels.extend(batch["labels"].numpy())
    return preds, labels


def eval_sts(model, eval_loader) -> float:
    """Compute Spearman correlation on one STS DataLoader."""
    preds, labels = [], []
    device = model.device
    with torch.cuda.amp.autocast(dtype=torch.float16):
        with torch.no_grad():
            for batch in tqdm(eval_loader, leave=False):
                out1 = model(input_ids=batch["input_ids1"].to(device),
                             attention_mask=batch["attention_mask1"].to(device))
                out2 = model(input_ids=batch["input_ids2"].to(device),
                             attention_mask=batch["attention_mask2"].to(device))

                emb1 = out1.last_hidden_state[:, 0, :]
                emb2 = out2.last_hidden_state[:, 0, :]
                # scale [-1, 1] -> [0, 5] to match STS-B scoring
                score = (F.cosine_similarity(emb1, emb2) + 1) * 2.5

                preds.extend(score.cpu().numpy())
                labels.extend(batch["labels"].numpy())

    rho, _ = spearmanr(preds, labels)
    print(f"  Spearman: {rho:.4f}")
    return float(rho)


def eval_sts_task(model, path_list: List[str], tokenizer) -> None:
    """Evaluate on a list of STS CSV files and print Spearman correlation."""
    model.eval()
    print("  [STS evaluation]")
    for path in path_list:
        print(f"    {path}")
        loader = DataLoader(
            STSDataset(path), batch_size=64, shuffle=False,
            collate_fn=lambda x: collate_fn(x, tokenizer),
        )
        eval_sts(model, loader)
    model.train()


def eval_classification_task(
    model,
    path_list: List[Tuple[str, str]],
    tokenizer,
) -> None:
    """Evaluate on classification tasks using frozen CLS embeddings + LogisticRegression.

    For each (train_csv, dev_csv) pair: extracts embeddings, fits a classifier,
    and prints accuracy + macro-F1.
    """
    model.eval()
    print("  [Classification evaluation]")

    for train_path, dev_path in path_list:
        print(f"    {dev_path}")

        train_loader = DataLoader(
            ClasssifyDataset(train_path), batch_size=64, shuffle=False,
            collate_fn=lambda x: clf_collate_fn(x, tokenizer),
        )
        dev_loader = DataLoader(
            ClasssifyDataset(dev_path), batch_size=64, shuffle=False,
            collate_fn=lambda x: clf_collate_fn(x, tokenizer),
        )

        X_train, y_train = _extract_embeddings(model, train_loader)
        X_test,  y_test  = _extract_embeddings(model, dev_loader)

        clf = LogisticRegression(random_state=42, n_jobs=1, max_iter=200, verbose=0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        print({
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
        })

    model.train()


def get_metric_pair_classification(
    scores: List[float],
    labels: List[float],
) -> Dict[str, float]:
    """Sweep 200 cosine thresholds to find the one maximising accuracy, then report metrics."""
    best_acc, best_thr = 0.0, 0.5
    scores_arr = np.array(scores)
    labels_arr = np.array(labels)

    for thr in np.linspace(0, 1, 200):
        preds = (scores_arr >= thr).astype(int)
        acc   = accuracy_score(labels_arr, preds)
        if acc > best_acc:
            best_acc, best_thr = acc, thr

    preds = (scores_arr >= best_thr).astype(int)
    return {
        "best_threshold":    float(best_thr),
        "accuracy":          float(accuracy_score(labels_arr, preds)),
        "f1":                float(f1_score(labels_arr, preds, average="macro")),
        "precision":         float(precision_score(labels_arr, preds, average="macro")),
        "recall":            float(recall_score(labels_arr, preds, average="macro")),
        "average_precision": float(average_precision_score(labels_arr, scores_arr)),
    }


def eval_pair(model, eval_loader) -> dict:
    """Compute pair-classification metrics on one DataLoader."""
    preds, labels = [], []
    device = model.device
    with torch.cuda.amp.autocast(dtype=torch.float16):
        with torch.no_grad():
            for batch in tqdm(eval_loader, leave=False):
                out1 = model(input_ids=batch["input_ids1"].to(device),
                             attention_mask=batch["attention_mask1"].to(device))
                out2 = model(input_ids=batch["input_ids2"].to(device),
                             attention_mask=batch["attention_mask2"].to(device))

                emb1 = out1.last_hidden_state[:, 0, :]
                emb2 = out2.last_hidden_state[:, 0, :]
                # scale [-1, 1] -> [0, 1]
                sim = (F.cosine_similarity(emb1, emb2) + 1) / 2

                preds.extend(sim.cpu().numpy())
                labels.extend(batch["labels"].numpy())

    metric = get_metric_pair_classification(preds, labels)
    print(f"    {metric}")
    return metric


def eval_pair_task(model, path_list: List[str], tokenizer) -> None:
    """Evaluate on a list of sentence-pair CSV files."""
    model.eval()
    print("  [Pair-task evaluation]")
    for path in path_list:
        print(f"    {path}")
        loader = DataLoader(
            PairDataset(path), batch_size=64, shuffle=False,
            collate_fn=lambda x: collate_fn(x, tokenizer),
        )
        eval_pair(model, loader)
    model.train()
