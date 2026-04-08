#!/usr/bin/env python
"""Validate that all notebooks in the CTKD/ and MRL/ directories are well-formed.

Purpose:
    Parses every .ipynb file found under CTKD/ and MRL/ with nbformat and runs
    the official schema validator. Exits with code 0 if all notebooks are valid,
    or code 1 if any notebook fails validation.

Usage:
    python scripts/validate_notebooks.py

Typical failure modes caught:
    - Truncated or corrupted JSON (e.g. from a failed git merge).
    - Missing required nbformat fields (cell_type, source, metadata).
    - Notebook format version mismatches.
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    # Notebooks live in CTKD/ and MRL/, not a top-level notebooks/ directory.
    notebooks = sorted(p for d in ("CTKD", "MRL") for p in (root / d).rglob("*.ipynb"))
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
