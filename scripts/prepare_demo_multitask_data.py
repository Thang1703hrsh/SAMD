from __future__ import annotations

import random
from pathlib import Path
import pandas as pd

SENTENCES = [
    "A quick brown fox jumps over the lazy dog.",
    "Transformers learn contextual representations of text.",
    "Contrastive learning pulls positives closer and pushes negatives apart.",
    "Sentence embeddings are useful for retrieval and clustering.",
    "The model is trained with an unsupervised SimCSE objective.",
    "Token alignment helps transfer structure across tokenizers.",
    "Matryoshka learning enables truncation at inference time.",
    "This dataset is intentionally tiny and synthetic.",
]

def _rand_sentence(rng: random.Random) -> str:
    s = rng.choice(SENTENCES)
    # tiny variation
    if rng.random() < 0.3:
        s += f" (v{rng.randint(1,5)})"
    return s


def write_cls(path: Path, n: int, n_labels: int = 3, seed: int = 42) -> None:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        rows.append({"text": _rand_sentence(rng), "label": rng.randrange(n_labels)})
    pd.DataFrame(rows).to_csv(path, index=False)


def write_pair(path: Path, n: int, seed: int = 43) -> None:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        s1 = _rand_sentence(rng)
        # make some positives by reusing the same sentence
        if rng.random() < 0.5:
            s2 = s1
            label = 1.0
        else:
            s2 = _rand_sentence(rng)
            label = 0.0
        rows.append({"sentence1": s1, "sentence2": s2, "label": label})
    pd.DataFrame(rows).to_csv(path, index=False)


def write_sts(path: Path, n: int, seed: int = 44) -> None:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        s1 = _rand_sentence(rng)
        if rng.random() < 0.5:
            s2 = s1
            score = 5.0
        else:
            s2 = _rand_sentence(rng)
            score = float(rng.choice([0.0, 1.0, 2.5, 3.0]))
        rows.append({"sentence1": s1, "sentence2": s2, "score": score})
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    out_dir = Path("data/multi-data")
    out_dir.mkdir(parents=True, exist_ok=True)

    # classification
    write_cls(out_dir / "banking_train.csv", n=200, seed=1)
    write_cls(out_dir / "banking77_validation.csv", n=80, seed=2)
    write_cls(out_dir / "banking77_test.csv", n=80, seed=3)

    write_cls(out_dir / "tweet_train.csv", n=200, seed=4)
    write_cls(out_dir / "tweet_validation.csv", n=80, seed=5)
    write_cls(out_dir / "tweet_test.csv", n=80, seed=6)

    write_cls(out_dir / "emotion_train.csv", n=200, seed=7)
    write_cls(out_dir / "emotion_validation.csv", n=80, seed=8)
    write_cls(out_dir / "emotion_test.csv", n=80, seed=9)

    # pair classification
    write_pair(out_dir / "mrpc_validation.csv", n=120, seed=10)
    write_pair(out_dir / "mrpc_test.csv", n=120, seed=11)

    write_pair(out_dir / "scitail_validation.csv", n=120, seed=12)
    write_pair(out_dir / "scitail_test.csv", n=120, seed=13)

    write_pair(out_dir / "wic_validation.csv", n=120, seed=14)
    write_pair(out_dir / "wic_test.csv", n=120, seed=15)

    # sts
    write_sts(out_dir / "sick_validation.csv", n=120, seed=16)
    write_sts(out_dir / "sick_test.csv", n=120, seed=17)

    write_sts(out_dir / "sts12_validation.csv", n=120, seed=18)
    write_sts(out_dir / "sts12_test.csv", n=120, seed=19)

    write_sts(out_dir / "stsb_validation.csv", n=120, seed=20)
    write_sts(out_dir / "stsb_test.csv", n=120, seed=21)

    print(f"Wrote demo eval CSVs to: {out_dir}")


if __name__ == "__main__":
    main()
