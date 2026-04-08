"""Insert module-level markdown docstring cells at the top of all SAMD notebooks."""
import json
import uuid
from pathlib import Path

ROOT = Path(__file__).parent.parent

entries = {
    "CTKD/SAMD.ipynb": (
        "# SAMD \u2014 Span-Aware Matryoshka Distillation (CTKD)\n\n"
        "**Purpose:** Main training notebook for the SAMD method under the "
        "cross-tokenizer knowledge distillation (CTKD) setting.\n\n"
        "**What this notebook does:**\n"
        "1. Loads a multitask training CSV covering classification, sentence-pair, and STS tasks.\n"
        "2. Tokenizes each sample with *both* the student and teacher tokenizer, returning "
        "character-offset maps for span alignment.\n"
        "3. Trains the student encoder (e.g. TinyBERT) against a frozen teacher "
        "(e.g. BAAI/bge-m3) using a three-term loss:\n"
        "   - **Task loss** \u2014 in-batch InfoNCE (SimCSE-style).\n"
        "   - **Matryoshka prefix KD** \u2014 teacher-to-student agreement at multiple "
        "embedding dimensions.\n"
        "   - **Span-CKA attention alignment** \u2014 projects teacher attention into the "
        "student token space via character-span overlap, then minimises CKA distance "
        "(computed sparsely every N steps).\n"
        "4. Evaluates after every epoch on three task families (classification, pair tasks, "
        "STS) with in-domain and OOD test sets.\n\n"
        "**Key configuration cell:** search for `TUNING KNOBS` to adjust loss weights, "
        "ramp schedules, and token-selection parameters."
    ),
    "CTKD/MINED.ipynb": (
        "# MinED \u2014 Minimum-Edit-Distance Token Alignment Baseline (CTKD)\n\n"
        "**Purpose:** Implements the MinED cross-tokenizer KD baseline, which aligns "
        "teacher and student tokens using dynamic-time-warping (DTW) on character-normalised "
        "edit distance.\n\n"
        "**What this notebook does:**\n"
        "1. Loads the same multitask training CSV used by SAMD.\n"
        "2. For each sample, computes a DTW-based soft alignment matrix between teacher and "
        "student tokens by measuring normalised character overlap (no fast-tokenizer requirement).\n"
        "3. Trains the student encoder with a token-level distillation loss guided by the DTW "
        "alignment, plus an in-batch InfoNCE task loss.\n"
        "4. Evaluates on the same three task families as SAMD for fair comparison.\n\n"
        "**Key algorithmic component:** `build_dtw_matrix` \u2014 constructs the pairwise cost "
        "matrix and recovers the monotone alignment path."
    ),
    "CTKD/DSDK.ipynb": (
        "# DSKD \u2014 Dual-Space Knowledge Distillation Baseline (CTKD)\n\n"
        "**Purpose:** Implements the DSKD cross-tokenizer KD baseline, which avoids explicit "
        "1-to-1 token alignment by projecting teacher and student representations into a shared "
        "latent space.\n\n"
        "**What this notebook does:**\n"
        "1. Loads the multitask training CSV.\n"
        "2. Trains lightweight projection heads that map teacher and student hidden states into "
        "a common dimension, removing the need for tokenizer-level token correspondence.\n"
        "3. Minimises a KD loss in the shared space (L2 or cosine) alongside the in-batch "
        "InfoNCE task loss.\n"
        "4. Evaluates on classification, pair, and STS tasks.\n\n"
        "**Key design choice:** projection heads are trained jointly with the student; "
        "the teacher is frozen throughout."
    ),
    "CTKD/CDM.ipynb": (
        "# CDM \u2014 Context-Dependent Dynamic Mapping Baseline (CTKD)\n\n"
        "**Purpose:** Implements the CDM cross-tokenizer KD baseline, which infers "
        "context-sensitive token correspondences from the models' own contextual representations "
        "rather than from surface string similarity.\n\n"
        "**What this notebook does:**\n"
        "1. Loads the multitask training CSV.\n"
        "2. For each batch, computes soft token-alignment weights by attending teacher hidden "
        "states against student hidden states (cross-attention style), producing a dynamic "
        "mapping that changes with context.\n"
        "3. Uses the alignment to distil token-level teacher knowledge into the student "
        "alongside the InfoNCE task loss.\n"
        "4. Evaluates on the three standard task families.\n\n"
        "**Key algorithmic component:** the context-dependent alignment matrix \u2014 search "
        "for the cross-attention weight computation block."
    ),
    "CTKD/EMO.ipynb": (
        "# EMO \u2014 Edit-distance + Optimal-Transport Distillation Baseline (CTKD)\n\n"
        "**Purpose:** Implements the EMO cross-tokenizer KD baseline, which combines "
        "MinED-based intra-relational distillation with an Optimal Transport (Sinkhorn) "
        "alignment step.\n\n"
        "**What this notebook does:**\n"
        "1. Loads the multitask training CSV.\n"
        "2. Computes reciprocal MinED token mappings between teacher and student to identify "
        "a high-confidence aligned subset.\n"
        "3. Applies Sinkhorn OT on the token-level cost matrix to obtain a smooth transport "
        "plan used as distillation alignment weights.\n"
        "4. Optionally adds a CKA-based attention alignment loss on the aligned token subsets.\n"
        "5. Evaluates on classification, pair, and STS tasks.\n\n"
        "**Key algorithmic component:** `sinkhorn()` \u2014 entropy-regularised optimal "
        "transport; search for `sinkhorn` to locate it."
    ),
    "MRL/MRL.ipynb": (
        "# MRL \u2014 Matryoshka Representation Learning Baseline\n\n"
        "**Purpose:** Implements the MRL baseline, which trains a single encoder whose "
        "embedding prefixes are independently useful at multiple dimensionalities.\n\n"
        "**What this notebook does:**\n"
        "1. Loads the multitask training CSV (single encoder; no cross-tokenizer teacher "
        "required).\n"
        "2. Applies `Matry_infonce`: runs the in-batch InfoNCE contrastive loss at each of "
        "several nested prefix dimensions (e.g. 64, 128, 256, 512, 768), summing the losses.\n"
        "3. Optionally adds a token-level CKA self-distillation loss via "
        "`MatryoshkaHiddenStateProcessor`, where smaller prefix dimensions learn from the "
        "largest.\n"
        "4. Evaluates truncated embeddings at each nested dimension on the three task families.\n\n"
        "**Key configuration:** `nested_dims` list in the config cell controls which prefix "
        "sizes are trained."
    ),
    "MRL/ESE.ipynb": (
        "# ESE \u2014 Extended Matryoshka across Dimensionality and Depth Baseline\n\n"
        "**Purpose:** Implements the ESE baseline (inspired by ESPRESSO), which extends "
        "Matryoshka-style learning across both embedding *dimensionality* and model *layer "
        "depth*.\n\n"
        "**What this notebook does:**\n"
        "1. Loads the multitask training CSV.\n"
        "2. Extracts hidden states from multiple encoder layers (not only the last).\n"
        "3. Applies a contrastive loss at each (layer, dim) combination, encouraging all "
        "intermediate representations to be directly usable.\n"
        "4. Evaluates by probing each layer-dim combination on the three task families.\n\n"
        "**Key configuration:** `layer_indices` and `nested_dims` in the config cell select "
        "which layers and dimensions are supervised."
    ),
    "MRL/SAMD-MRL.ipynb": (
        "# SAMD-MRL \u2014 Span-Aware Matryoshka Distillation with Nested Embeddings\n\n"
        "**Purpose:** Full integration of SAMD's span-aware cross-tokenizer alignment with "
        "Matryoshka Representation Learning. This is the most complete SAMD variant: it "
        "combines teacher-to-student distillation across tokenizers *and* nested prefix "
        "training for flexible-dimension deployment.\n\n"
        "**What this notebook does:**\n"
        "1. Loads the multitask training CSV and dual-tokenizes with both student and teacher, "
        "returning character-offset maps.\n"
        "2. Trains the student with a three-term loss:\n"
        "   - **Matryoshka InfoNCE** \u2014 contrastive task loss applied at each nested "
        "prefix dimension.\n"
        "   - **Matryoshka prefix KD** \u2014 teacher-to-student cosine matching at each "
        "prefix size.\n"
        "   - **Span-CKA attention alignment** \u2014 character-span-overlap-based attention "
        "transfer (sparse, every N steps).\n"
        "3. Evaluates truncated student embeddings at each nested dimension on classification, "
        "pair, and STS tasks.\n\n"
        "**Key algorithmic components:**\n"
        "- `build_span_overlap_matrix` \u2014 constructs the IoU-based soft alignment from "
        "tokenizer offset maps.\n"
        "- `compute_span_cka_att_loss` \u2014 projects teacher attention into student space "
        "and minimises CKA distance.\n"
        "- `Matry_infonce` \u2014 applies InfoNCE at multiple prefix dimensions."
    ),
}

for rel_path, docstring in entries.items():
    path = ROOT / rel_path
    nb = json.loads(path.read_text(encoding="utf-8"))
    new_cell = {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": docstring,
    }
    nb["cells"].insert(0, new_cell)
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Done: {rel_path}")
