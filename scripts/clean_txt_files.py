"""Remove AI-generation artifacts from MINED_code.txt and SAMD_code.txt."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def clean(text: str) -> str:
    # === -> Cell N markers ===
    text = re.sub(r"# ={5,} CELL (\d+) ={5,}", r"# Cell \1", text)

    # inline section dividers with === or --- decoration
    text = re.sub(r"\s*# ={5,} TEACHER \(device_t,? no grad\) ={5,}", "        # teacher forward pass (no grad)", text)
    text = re.sub(r"\s*# ={5,} TEACHER \(bi-encoder\) on device_t ={5,}", "            # teacher forward pass (no grad)", text)
    text = re.sub(r"\s*# ={5,} STUDENT \(device_s\) ={5,}", "        # student forward pass", text)
    text = re.sub(r"\s*# ={5,} STUDENT \(bi-encoder\) on device_s ={5,}", "            # student forward pass", text)

    # lettered loss blocks with === decoration
    text = re.sub(r"\s*# ={5,} \(A\)[^=]+=+", "            # (A) task loss", text)
    text = re.sub(r"\s*# ={5,} \(B\)[^=]+=+", "            # (B) KD_DTW: token-level loss", text)
    text = re.sub(r"\s*# ={5,} \(C\)[^=]+=+", "            # (C) pairwise cosine distillation", text)
    text = re.sub(r"\s*# ={5,} .*LOSS.*={5,}", "            # total loss", text)

    # sentence sub-dividers
    text = re.sub(r"[ \t]*# ={3,} sentence \d+ ={3,}", "                # sentence 1", text)

    # logging divider
    text = re.sub(r"[ \t]*# -{3,} logging -{3,}", "        # logging", text)

    # pair (bi-encoder) divider
    text = re.sub(r"[ \t]*# -{3,} pair \(bi-encoder\) -{3,}\n", "", text)

    # bare "# student" / "# teacher" section labels inside dict literals
    # (lines that are just "            # student" or "            # teacher" with no other text)
    text = re.sub(r"^[ \t]*# student[ \t]*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[ \t]*# teacher[ \t]*$", "", text, flags=re.MULTILINE)

    # collapse runs of 3+ blank lines down to 2
    text = re.sub(r"\n{4,}", "\n\n\n", text)

    return text


for name in ("MINED_code.txt", "SAMD_code.txt"):
    path = ROOT / name
    original = path.read_text(encoding="utf-8")
    cleaned  = clean(original)
    path.write_text(cleaned, encoding="utf-8")
    changed = sum(1 for a, b in zip(original.splitlines(), cleaned.splitlines()) if a != b)
    print(f"{name}: {changed} lines changed")
