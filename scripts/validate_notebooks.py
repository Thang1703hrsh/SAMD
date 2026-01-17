#!/usr/bin/env python
"""Validate that all notebooks under ./notebooks are valid Jupyter notebooks."""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    nb_dir = root / "notebooks"
    notebooks = sorted(nb_dir.rglob("*.ipynb"))
    if not notebooks:
        print("No notebooks found.")
        return 0

    ok = True
    for p in notebooks:
        try:
            nb = nbformat.read(p, as_version=4)
            nbformat.validate(nb)
            print(f"OK: {p.relative_to(root)}")
        except Exception as e:
            ok = False
            print(f"FAIL: {p.relative_to(root)}\n  {e}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
