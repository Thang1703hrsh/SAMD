<<<<<<< HEAD
# SAMD
=======
# SAMD: Span-Aware Matryoshka Distillation for Cross-Tokenizer Embedding Models

> **Notebook-first reference implementation** for cross-tokenizer knowledge distillation (CTKD) with **span-aware alignment** and **Matryoshka (nested) embeddings**.

<!-- Badges (edit as needed) -->
![python](https://img.shields.io/badge/Python-3.10%2B-blue)
![pytorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![transformers](https://img.shields.io/badge/Transformers-4.40%2B-yellow)
![license](https://img.shields.io/badge/License-MIT-green)

## Overview

**SAMD** (Span-Aware Matryoshka Distillation) targets two practical bottlenecks in embedding-model compression:

1. **Tokenizer mismatch** (cross-tokenizer KD): teacher and student use different vocabularies/segmentations, so token indices do not align.
2. **Deployment rigidity**: fixed-dimensional embeddings are expensive to store and cannot be easily truncated for different latency/storage budgets.

This project evaluates sentence representation quality on **three task families** and reports both **in-domain performance** and **robustness on held-out OOD test sets**:

- **Text classification**: TweetEval + Banking77 (in-domain), Emotion (OOD)
- **Sentence-pair tasks**: MRPC + WiC (in-domain), SciTail (OOD)
- **STS**: STS-B + SICK-R (in-domain), STS12 (OOD)

For classification-style tasks, we train a **lightweight classifier on frozen sentence embeddings**. For STS, we compute **cosine similarity** between embeddings.

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

> **Note on data:** `.gitignore` is configured to **avoid committing real datasets** by default. Keep your CSVs locally in `data/`.

## Requirements

- Python **3.10+**
- PyTorch **2.x**
- Transformers **4.40+**

Install everything via `requirements.txt` (pip) or `environment.yml` (conda).

## Quickstart

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

### Run notebooks

```bash
jupyter lab
```

Then open one of the notebooks under `CTKD/` or `MRL/`.

## Data

Your local data should follow the layout documented in **`data/README.md`**. In your setup, the key files are:

- Training CSV: `data/merged_9_data_3k_each_ver2.csv`
- Evaluation directory: `data/multi-data/`

### Expected CSV columns

The demo generator (`scripts/prepare_demo_multitask_data.py`) uses the following column conventions, which are also a good reference for your real CSVs:

- **Classification**: `text`, `label`
- **Pair classification**: `sentence1`, `sentence2`, `label`
- **STS**: `sentence1`, `sentence2`, `score`

### Environment overrides

If your notebooks support environment overrides, you can set:

```bash
export SAMD_TRAIN_CSV="data/merged_9_data_3k_each_ver2.csv"
export SAMD_EVAL_DIR="data/multi-data"
```

### (Optional) Generate tiny synthetic CSVs

```bash
python scripts/prepare_demo_multitask_data.py
```

> The synthetic data is intentionally tiny and not meant for reporting results. If you want the demo files to exactly match your naming (e.g., `banking77_train.csv`), rename the generated file or adjust the script.

## Running experiments

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

Your notebooks are set up to work with teachers such as:

- `BAAI/bge-m3`
- `Qwen/Qwen3-Embedding-0.6B`
- `McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp` (and supervised variant)

### Students

- `huawei-noah/TinyBERT_General_4L_312D`
- `huawei-noah/TinyBERT_General_6L_768D`
- `bert-base-uncased`

Most notebooks include a config cell where you can change `teacher_name` / `student_name`.

### Hugging Face authentication

Some models may require authentication (private/gated repos). If needed:

```bash
export HF_TOKEN="<your_hf_token>"
```

If a notebook uses `trust_remote_code=True`, ensure you understand and trust the source model code before execution.

## Training configuration

The default training configuration used across methods is summarized below:

| Setting | DSKD | CDM | MinED | EMO | **SAMD** | MRL | ESE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Epochs | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| Learning rate | 2e-5 | 2e-5 | 2e-5 | 2e-5 | 2e-5 | 2e-5 | 2e-5 |
| Batch size | 32 | 32 | 32 | 32 | 32 | 32 | 32 |
| LR scheduler | Cosine | Cosine | Cosine | Cosine | Cosine | Cosine | Cosine |

Method-specific knobs (e.g., projection sizes, top-m token selection, Matryoshka slice set) are configured inside the notebooks.

## Utilities

- `scripts/validate_notebooks.py` validates notebook JSON. If you keep notebooks under `CTKD/` and `MRL/` (instead of `notebooks/`), update the `nb_dir` path in that script accordingly.

## Reproducibility tips

- Fix a random seed (Python / NumPy / PyTorch) inside each notebook.
- Log the exact versions of `torch`, `transformers`, and `peft` when producing paper tables.
- When running large teachers, prefer a GPU runtime and set `torch_dtype` / `device_map` consistently.

## Troubleshooting

- **CUDA OOM (large teachers like LLM2Vec/Mistral)**: reduce batch size, use gradient accumulation, or choose a smaller teacher.
- **bfloat16 issues**: if your GPU does not support bf16 well, switch to fp16 or fp32.
- **Padding side**: some decoder-style teachers may require `padding_side="left"`.

## Citation

If you use this repository in academic work, please cite SAMD:

```bibtex
@misc{samd2026,
  title        = {SAMD: Span-Aware Matryoshka Distillation for Cross-Tokenizer Embedding Models},
  author       = {<Your Name(s)>},
  year         = {2026},
  howpublished = {GitHub repository},
}
```

## License

This project is released under the **MIT License**. See `LICENSE` for details.

## Acknowledgements

This project compares multiple CTKD and Matryoshka-style methods. If you reuse or adapt code from other repositories/papers, please ensure you follow their licenses and cite the original works.
>>>>>>> c965565 (First commit: Upload source code)
