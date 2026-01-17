# SAMD: Span-Aware Matryoshka Distillation for Cross-Tokenizer Embedding Models

> Notebook reference implementation for cross-tokenizer knowledge distillation (CTKD) with **span-aware alignment** and **Matryoshka (nested) embeddings**.

![python](https://img.shields.io/badge/Python-3.10%2B-blue)
![pytorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![transformers](https://img.shields.io/badge/Transformers-4.40%2B-yellow)
![license](https://img.shields.io/badge/License-MIT-green)

## Overview

**SAMD (Span-Aware Matryoshka Distillation)** targets two practical bottlenecks in embedding-model compression:

1. **Tokenizer mismatch (cross-tokenizer KD)**: teacher and student tokenizers induce different vocabularies, segmentations, and sequence lengths—so token indices do not align.
2. **Deployment rigidity**: fixed-dimensional embeddings are costly to store and cannot be easily truncated for different latency / storage budgets.

At a high level, SAMD combines:

- **Span-aware alignment** using character-offset overlap to project teacher token relations into the student token space.
- **Matryoshka (nested) supervision** that enforces prefix-consistent teacher–student agreement at multiple embedding dimensions.
- **Optional attention alignment (Span-CKA/IRA)** for structural supervision when teacher/student tokenization differs.

## Evaluation protocol

We evaluate representation quality on **three task families**, reporting both **in-domain** performance and **robustness on held-out OOD test sets**:

- **Text classification**: TweetEval + Banking77 (in-domain), Emotion (OOD)
- **Sentence-pair tasks**: MRPC + WiC (in-domain), SciTail (OOD)
- **STS**: STS-B + SICK-R (in-domain), STS12 (OOD)

**Metrics.** We follow standard metrics for each task family:

- Classification / pair classification: accuracy or macro-F1 (as configured in the notebooks)
- STS: Spearman correlation

**Evaluation procedure.** For classification-style tasks, we train a lightweight classifier on **frozen** sentence embeddings. For STS tasks, we compute **cosine similarity** between embeddings.

## Baselines

We compare SAMD against two groups of baselines.

### Cross-tokenizer KD (CTKD)

Representative methods that handle tokenizer/vocabulary mismatch via explicit alignment or shared-space learning:

- **MinED**: minimum-edit-distance token correspondence enabling direct token-level supervision.
- **DSKD**: projects teacher/student outputs into a shared latent space (no strict 1:1 token mapping).
- **CDM**: context-dependent, dynamic token correspondences inferred from contextual representations.
- **EMO**: MinED-based intra-relational distillation + Optimal Transport alignment.

### Matryoshka / elastic embeddings

Strong references for low-dimensional and truncatable embeddings:

- **MRL**: Matryoshka Representation Learning (nested prefix embeddings; truncation at inference time).
- **ESE**: extends Matryoshka-style learning across embedding dimensionality and model depth.

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
│   └── multi-data/                 # evaluation CSVs (kept local by default)
├── scripts/
│   ├── prepare_demo_multitask_data.py
│   └── validate_notebooks.py
├── requirements.txt
├── environment.yml
├── LICENSE
└── README.md
```

> **Data note:** `.gitignore` is configured to avoid committing real datasets by default. Keep your CSVs locally in `data/`.

## Installation

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

Your local data layout is documented in `data/README.md`. In your setup, the key files are:

- Training CSV: `data/merged_9_data_3k_each_ver2.csv`
- Evaluation directory: `data/multi-data/`

### Expected CSV columns

The demo generator (`scripts/prepare_demo_multitask_data.py`) uses the following conventions (also a good reference for your real CSVs):

- **Classification**: `text`, `label`
- **Pair classification**: `sentence1`, `sentence2`, `label`
- **STS**: `sentence1`, `sentence2`, `score`

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

### CTKD experiments

Run any notebook under `CTKD/`:

- `CTKD/SAMD.ipynb` — SAMD
- `CTKD/MINED.ipynb` — MinED baseline
- `CTKD/DSDK.ipynb` — DSKD baseline
- `CTKD/CDM.ipynb` — CDM baseline
- `CTKD/EMO.ipynb` — EMO baseline

### Matryoshka experiments

Run any notebook under `MRL/`:

- `MRL/MRL.ipynb` — MRL baseline
- `MRL/ESE.ipynb` — ESE baseline
- `MRL/SAMD-MRL.ipynb` — SAMD with Matryoshka training/evaluation

## Models

### Teachers (examples)

The notebooks are set up to work with teachers such as:

- `BAAI/bge-m3`
- `Qwen/Qwen3-Embedding-0.6B`
- `McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp` (and supervised variant)

### Students

- `huawei-noah/TinyBERT_General_4L_312D`
- `huawei-noah/TinyBERT_General_6L_768D`
- `bert-base-uncased`

Most notebooks include a config cell where you can change `teacher_name` / `student_name`.

### Hugging Face authentication (IMPORTANT)

Some models may require authentication (private/gated repos). **Do not hard-code tokens in notebooks.**

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

### SAMD objective weighting (notebook knobs)

The SAMD notebooks expose practical knobs for balancing:

- **Task loss** (SimCSE / Matryoshka InfoNCE)
- **Matryoshka prefix KD** (teacher → student)
- **Span-aware attention alignment** (optional; computed every N steps)

Weights are ramped over training steps to improve stability (e.g., delay attention alignment until the student geometry becomes reasonable).

## Reproducibility tips

- Fix random seeds (Python / NumPy / PyTorch) inside each notebook.
- Log exact versions of `torch`, `transformers`, and `peft` when producing paper tables.
- For large teachers, prefer a GPU runtime and set `torch_dtype` / `device_map` consistently.

## Troubleshooting

- **Blocked push (secret scanning)**: if GitHub blocks your push due to a token in history, remove it from notebooks and rewrite history (do not bypass).
- **CUDA OOM (large teachers like LLM2Vec/Mistral)**: reduce batch size, use gradient accumulation, or choose a smaller teacher.
- **bf16 issues**: if your GPU does not support bf16, switch to fp16 or fp32.
- **Padding side**: some decoder-style teachers may require `padding_side="left"`.

## Citation

If you use this repository in academic work, please cite:

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
