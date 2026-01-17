# SAMD: Span-Aware Matryoshka Distillation for Cross-Tokenizer Embedding Models

> **Notebook reference implementation** for cross-tokenizer knowledge distillation (CTKD) with **span-aware alignment** and **Matryoshka (nested) embeddings**.

<!-- Badges (edit as needed) -->
![python](https://img.shields.io/badge/Python-3.10%2B-blue)
![pytorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![transformers](https://img.shields.io/badge/Transformers-4.40%2B-yellow)
![license](https://img.shields.io/badge/License-MIT-green)

## Overview

**SAMD (Span-Aware Matryoshka Distillation)** targets two practical bottlenecks in embedding-model compression:

1. **Tokenizer mismatch (cross-tokenizer KD):** teacher and student use different vocabularies/segmentations, so token indices do not align.
2. **Deployment rigidity:** fixed-dimensional embeddings are costly to store and cannot be easily truncated for different latency/storage budgets.

At a high level, SAMD combines:

- **Span-aware alignment** using character-offset overlap to map teacher token relations into the student token space.
- **Matryoshka (nested) supervision** that enforces prefix-consistent teacher–student agreement at multiple embedding dimensions.
- **Optional attention alignment (Span-CKA/IRA)** as structural supervision when tokenization differs.

## Evaluation protocol

We evaluate sentence representation quality on **three task families**, reporting both **in-domain** performance and **robustness on held-out OOD test sets**:

- **Text classification:** TweetEval + Banking77 (in-domain), Emotion (OOD)
- **Sentence-pair tasks:** MRPC + WiC (in-domain), SciTail (OOD)
- **STS:** STS-B + SICK-R (in-domain), STS12 (OOD)

**Metrics.** We follow standard metrics for each task family (configured in notebooks):

- Classification / pair classification: accuracy or macro-F1
- STS: Spearman correlation

**Procedure.** For classification-style tasks, we train a lightweight classifier on **frozen** sentence embeddings. For STS, we compute **cosine similarity** between embeddings.

## Baselines

We compare SAMD against two groups of baselines.

### Cross-tokenizer KD (CTKD)

- **MinED:** minimum-edit-distance token correspondence enabling direct token-level supervision.
- **DSKD:** projects teacher/student outputs into a shared latent space (no strict 1:1 token mapping).
- **CDM:** context-dependent, dynamic token correspondences inferred from contextual representations.
- **EMO:** MinED-based intra-relational distillation + Optimal Transport alignment.

### Matryoshka / elastic embeddings

- **MRL:** Matryoshka Representation Learning (nested prefix embeddings; truncation at inference time).
- **ESE:** extends Matryoshka-style learning across embedding dimensionality and model depth.

## Repository structure

```
.
├── CTKD/
│   ├── MINED.ipynb
│   ├── DSDK.ipynb
│   ├── CDM.ipynb
│   ├── EMO.ipynb
│   └── SAMD.ipynb
├── MRL/
│   ├── MRL.ipynb
│   ├── ESE.ipynb
│   └── SAMD-MRL.ipynb
├── data/
│   ├── README.md
│   ├── merged_9_data_3k_each_ver2.csv
│   └── multi-data/
├── scripts/
│   ├── prepare_demo_multitask_data.py
│   └── validate_notebooks.py
├── requirements.txt
├── environment.yml
├── LICENSE
└── README.md
```

> **Data note:** Many users keep datasets out of Git. This repo includes sane `.gitignore` defaults; if you *do* want to version data, consider Git LFS.

## Requirements

- Python **3.10+**
- PyTorch **2.x**
- Transformers **4.40+**

Install everything via pip or conda.

### Option A: pip + venv

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Option B: conda

```bash
conda env create -f environment.yml
conda activate samd
```

## Data

Your data layout is documented in **`data/README.md`**.

In your setup, the key paths are:

- **Training CSV:** `data/merged_9_data_3k_each_ver2.csv`
- **Evaluation directory:** `data/multi-data/`

### Environment overrides

Most notebooks support environment variables so paths work across Kaggle / local / GitHub:

```bash
export SAMD_TRAIN_CSV="data/merged_9_data_3k_each_ver2.csv"
export SAMD_EVAL_DIR="data/multi-data"
```

### (Optional) generate tiny synthetic CSVs

```bash
python scripts/prepare_demo_multitask_data.py
```

> The synthetic data is intentionally tiny and not meant for reporting results.

## Running experiments

Launch Jupyter:

```bash
jupyter lab
```

### CTKD notebooks

- `CTKD/SAMD.ipynb` — SAMD
- `CTKD/MINED.ipynb` — MinED baseline
- `CTKD/DSDK.ipynb` — DSKD baseline
- `CTKD/CDM.ipynb` — CDM baseline
- `CTKD/EMO.ipynb` — EMO baseline

### Matryoshka notebooks

- `MRL/MRL.ipynb` — MRL baseline
- `MRL/ESE.ipynb` — ESE baseline
- `MRL/SAMD-MRL.ipynb` — SAMD + Matryoshka training/evaluation

## Models

### Teachers (examples)

- `BAAI/bge-m3`
- `Qwen/Qwen3-Embedding-0.6B`
- `McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp` (and supervised variant)

### Students

- `huawei-noah/TinyBERT_General_4L_312D`
- `huawei-noah/TinyBERT_General_6L_768D`
- `bert-base-uncased`

Most notebooks include a config cell where you can change `teacher_name` / `student_name`.

## Custom training schedule (SAMD knobs)

The SAMD notebooks expose practical knobs for balancing:

- **Task loss** (SimCSE / Matryoshka InfoNCE)
- **Matryoshka prefix KD** (teacher → student)
- **Span-aware attention alignment** (optional; computed every *N* steps)

Weights are **ramped over training steps** to improve stability (e.g., delay attention alignment until the student geometry becomes reasonable).

The core objective is:

\[
\mathcal{L} = w_{task}\,\mathcal{L}_{task} + \alpha_{kd}\,\big(\beta_{mrl}\,\mathcal{L}_{mrl} + \alpha_{attn}\,\mathcal{L}_{att}\big).
\]

In `CTKD/SAMD.ipynb`, look for the **"TUNING KNOBS (loss schedule)"** cell and adjust:

- `w_task`, `alpha_kd`
- `beta_mrl_max`, `mrl_start`, `mrl_ramp`, `mrl_weight_mode`
- `alpha_attn_max`, `att_start`, `att_ramp`, `att_every`
- token-selection controls: `top_frac`, `min_tokens`, `k_ira`

Recommended ranges that are typically stable:

- `alpha_attn_max`: **0.01–0.03** (keep attention loss small)
- `top_frac`: **0.20–0.33**
- `att_every`: **2–8** (trade-off compute vs. signal)

## Hugging Face authentication (IMPORTANT)

Some models may require authentication (private/gated repos). **Never hard-code tokens in notebooks.**

Set an environment variable:

```bash
export HF_TOKEN="<your_hf_token>"
```

Or create a local `.env` (ignored by git) based on `.env.example`.

## Training configuration

The default training configuration used across methods is summarized below:

| Setting | DSKD | CDM | MinED | EMO | **SAMD** | MRL | ESE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Epochs | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| Learning rate | 2e-5 | 2e-5 | 2e-5 | 2e-5 | 2e-5 | 2e-5 | 2e-5 |
| Batch size | 32 | 32 | 32 | 32 | 32 | 32 | 32 |
| LR scheduler | Cosine | Cosine | Cosine | Cosine | Cosine | Cosine | Cosine |

Method-specific knobs (e.g., projection details, token selection, Matryoshka slices) are configured inside notebooks.

## Reproducibility tips

- Fix random seeds (Python / NumPy / PyTorch) inside each notebook.
- Log exact versions of `torch`, `transformers`, and `peft` when producing paper tables.
- For large teachers, prefer a GPU runtime and set `torch_dtype` / `device_map` consistently.

## Troubleshooting

- **Blocked push (secret scanning):** if GitHub blocks your push due to a token in history, remove it from notebooks and rewrite history (do not bypass).
- **CUDA OOM (large teachers like LLM2Vec/Mistral):** reduce batch size, use gradient accumulation, or choose a smaller teacher.
- **bf16 issues:** if your GPU does not support bf16, switch to fp16 or fp32.
- **Padding side:** some decoder-style teachers may require `padding_side="left"`.

## Citation

```bibtex
@misc{samd2026,
  title        = {SAMD: Span-Aware Matryoshka Distillation for Cross-Tokenizer Embedding Models},
  year         = {2026},
  howpublished = {GitHub repository},
}
```

## License

Released under the **MIT License**. See `LICENSE` for details.

## Acknowledgements

This repository compares multiple CTKD and Matryoshka-style methods. If you reuse or adapt code from other repositories/papers, please follow their licenses and cite the original works.
