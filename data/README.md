# Data directory

This project expects the following **local** data structure (matching the layout in your screenshots):

```
data/
  merged_9_data_3k_each_ver2.csv
  multi-data/
    banking77_train.csv
    banking77_validation.csv
    banking77_test.csv

    emotion_train.csv
    emotion_validation.csv
    emotion_test.csv

    tweet_train.csv
    tweet_validation.csv
    tweet_test.csv

    mrpc_validation.csv
    mrpc_test.csv

    qnli_validation.csv
    qnli_test.csv

    rte_validation.csv
    rte_test.csv

    scitail_validation.csv
    scitail_test.csv

    sick_validation.csv
    sick_test.csv

    sts12_validation.csv
    sts12_test.csv

    stsb_validation.csv
    stsb_test.csv

    wic_validation.csv
    wic_test.csv
```

## Training data

- Default training CSV: `data/merged_9_data_3k_each_ver2.csv`

If your notebook/scripts support environment overrides, you can set:

```bash
export SAMD_TRAIN_CSV="data/merged_9_data_3k_each_ver2.csv"
```

## Evaluation data (multi-task)

Evaluation CSVs live under `data/multi-data/` and follow the naming pattern:

- `{task}_{split}.csv`, where `split ∈ {train, validation, test}`

Not every task includes all splits (e.g., some tasks provide only `validation`/`test`). The evaluation code should load whichever files exist.

You can override the evaluation directory with:

```bash
export SAMD_EVAL_DIR="data/multi-data"
```

## Notes

- Large datasets are often **not committed** to Git (to keep the repo lightweight). If you want to version data files, consider **Git LFS**.
- If you need a tiny runnable demo dataset (for quick CI / smoke tests), you can generate synthetic files:

```bash
python scripts/prepare_demo_multitask_data.py
```
