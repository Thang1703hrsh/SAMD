"""
check_data_leakage.py
─────────────────────
Checks for data leakage between the training CSV and all evaluation test splits.
Produces:
  1. A detailed overlap report (printed + saved to leakage_report.txt).
  2. A clean training CSV with leaking sentences removed
     (data/merged_9_data_3k_each_ver2_noleak.csv).

Usage (local):
    python scripts/check_data_leakage.py

Usage on Kaggle — add a cell:
    !python /kaggle/input/.../check_data_leakage.py \
        --train  /kaggle/input/multitask-data/merged_9_data_3k_each_ver2.csv \
        --eval   /kaggle/input/multitask-data/multi-data \
        --out    /kaggle/working/merged_9_data_3k_each_noleak.csv
"""
import argparse
import os
import unicodedata
from pathlib import Path

import pandas as pd


# ── normalisation ────────────────────────────────────────────────────────────

def normalise(text: str) -> str:
    """Unicode-NFC + collapse whitespace + lowercase for robust matching."""
    text = unicodedata.normalize("NFC", str(text))
    return " ".join(text.lower().split())


# ── collect test sentences ────────────────────────────────────────────────────

# Each entry: (csv_path, list_of_text_columns_to_extract)
# For classification tasks the text lives in 'text'.
# For pair tasks both 'sentence1' and 'sentence2' are evaluation inputs.
TEST_FILE_COLS = {
    # classification
    "banking77_test.csv":   ["text"],
    "emotion_test.csv":     ["text"],
    "tweet_test.csv":       ["text"],
    # pair / NLI
    "mrpc_test.csv":        ["sentence1", "sentence2"],
    "scitail_test.csv":     ["sentence1", "sentence2"],
    "wic_test.csv":         ["sentence1", "sentence2"],
    "rte_test.csv":         ["sentence1", "sentence2"],
    "qnli_test.csv":        ["sentence1", "sentence2"],
    # STS
    "stsb_test.csv":        ["sentence1", "sentence2"],
    "sick_test.csv":        ["sentence1", "sentence2"],
    "sts12_test.csv":       ["sentence1", "sentence2"],
    # validation splits — include these too to be conservative
    "banking77_validation.csv": ["text"],
    "emotion_validation.csv":   ["text"],
    "tweet_validation.csv":     ["text"],
    "mrpc_validation.csv":      ["sentence1", "sentence2"],
    "scitail_validation.csv":   ["sentence1", "sentence2"],
    "wic_validation.csv":       ["sentence1", "sentence2"],
    "rte_validaion.csv":        ["sentence1", "sentence2"],
    "qnli_validation.csv":      ["sentence1", "sentence2"],
    "stsb_validation.csv":      ["sentence1", "sentence2"],
    "sick_validation.csv":      ["sentence1", "sentence2"],
    "sts12_validation.csv":     ["sentence1", "sentence2"],
}


def build_test_sentence_set(eval_root: str) -> dict:
    """
    Returns:
        {filename: set_of_normalised_sentences}
    """
    result = {}
    for fname, cols in TEST_FILE_COLS.items():
        fpath = os.path.join(eval_root, fname)
        if not os.path.exists(fpath):
            continue
        df = pd.read_csv(fpath)
        sentences = set()
        for col in cols:
            if col in df.columns:
                sentences.update(normalise(s) for s in df[col].dropna().astype(str))
        result[fname] = sentences
    return result


# ── main ──────────────────────────────────────────────────────────────────────

def main(train_csv: str, eval_root: str, out_csv: str, report_path: str) -> None:
    # 1. Load training data
    df_train = pd.read_csv(train_csv)
    assert "text" in df_train.columns, f"'text' column not found in {train_csv}"

    n_before = len(df_train)
    train_norm = df_train["text"].astype(str).map(normalise)

    # 2. Build union of all test sentences
    per_file = build_test_sentence_set(eval_root)
    all_test_sentences: set = set()
    for s in per_file.values():
        all_test_sentences.update(s)

    print(f"Training sentences   : {n_before:,}")
    print(f"Test/val sentences   : {len(all_test_sentences):,}  (across {len(per_file)} files)")
    print()

    # 3. Per-file overlap
    lines = []
    lines.append(f"Training CSV : {train_csv}")
    lines.append(f"Eval root    : {eval_root}")
    lines.append(f"Training N   : {n_before:,}")
    lines.append(f"Test/val uniq: {len(all_test_sentences):,}")
    lines.append("")
    lines.append(f"{'File':<40}  {'Test N':>7}  {'Overlap':>7}  {'Rate':>7}")
    lines.append("-" * 70)

    total_overlap_idx = set()
    for fname, test_set in sorted(per_file.items()):
        mask = train_norm.isin(test_set)
        n_overlap = mask.sum()
        overlap_idx = set(df_train.index[mask].tolist())
        total_overlap_idx.update(overlap_idx)
        rate = n_overlap / n_before * 100
        line = f"{fname:<40}  {len(test_set):>7,}  {n_overlap:>7,}  {rate:>6.2f}%"
        print(line)
        lines.append(line)

    lines.append("-" * 70)
    total_line = (f"{'TOTAL OVERLAP':<40}  {'':>7}  "
                  f"{len(total_overlap_idx):>7,}  "
                  f"{len(total_overlap_idx)/n_before*100:>6.2f}%")
    print()
    print(total_line)
    lines.append(total_line)

    # 4. Print examples of overlapping sentences
    if total_overlap_idx:
        print("\n-- Sample overlapping sentences (up to 10) --")
        lines.append("\nSample overlapping sentences (up to 10):")
        sample = df_train.loc[list(total_overlap_idx)[:100], "text"].tolist()
        for s in sample:
            print(f"  {s[:120]}")
            lines.append(f"  {s[:120]}")

    # 5. Write clean CSV
    clean_mask = ~train_norm.isin(all_test_sentences)
    df_clean = df_train[clean_mask].reset_index(drop=True)
    n_after = len(df_clean)
    removed = n_before - n_after

    summary = (f"\nRemoved {removed:,} sentences ({removed/n_before*100:.2f}%). "
               f"Clean training set: {n_after:,} sentences.")
    print(summary)
    lines.append(summary)

    df_clean.to_csv(out_csv, index=False)
    clean_line = f"Clean CSV written : {out_csv}"
    print(clean_line)
    lines.append(clean_line)

    # 6. Save report
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report written    : {report_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description="Check and remove data leakage.")
    parser.add_argument(
        "--train",
        default=str(repo_root / "data" / "merged_9_data_3k_each_ver2.csv"),
        help="Path to training CSV (must have a 'text' column).",
    )
    parser.add_argument(
        "--eval",
        default=str(repo_root / "data" / "multi-data"),
        help="Directory containing evaluation CSV files.",
    )
    parser.add_argument(
        "--out",
        default=str(repo_root / "data" / "merged_9_data_3k_each_ver2_noleak.csv"),
        help="Output path for the clean (deduplicated) training CSV.",
    )
    parser.add_argument(
        "--report",
        default=str(repo_root / "data" / "leakage_report.txt"),
        help="Output path for the text report.",
    )
    args = parser.parse_args()

    main(
        train_csv=args.train,
        eval_root=args.eval,
        out_csv=args.out,
        report_path=args.report,
    )
