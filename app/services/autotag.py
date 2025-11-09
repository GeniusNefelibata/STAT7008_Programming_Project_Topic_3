from __future__ import annotations
"""
Auto-tagging utilities built on top of your existing EMB (CLIP) service.

Key points in this version:
- Canonical (singular) category set + 'others'
- Plural & synonym folding (cats->cat, landscapes->landscape, receipts->receipt, screenshot(s)->others, etc.)
- predict_labels() defaults to prefer_plural=False (we don't pluralize by default)
- Storage should ALWAYS use canonicalize() on the chosen label
"""

from dataclasses import dataclass
from typing import Iterable, List, Dict, Tuple, Optional
import numpy as np

from . import embeddings as EMB  # SentenceTransformer("clip-ViT-B-32")

# ---------------------------------------------------------------------
# Canonical category set (singular + 'others')
# ---------------------------------------------------------------------
CATEGORY_CANON: Tuple[str, ...] = (
    "cat", "dog", "elephant", "flower", "forest",
    "fruit", "human", "landscape", "receipt", "sea", "street",
    "others",
)

# Plurals / synonyms -> canonical (None means ignore)
ALIASES: Dict[str, Optional[str]] = {
    # explicit singular
    "cat": "cat", "dog": "dog", "elephant": "elephant", "flower": "flower",
    "forest": "forest", "fruit": "fruit", "human": "human", "landscape": "landscape",
    "receipt": "receipt", "sea": "sea", "street": "street", "others": "others",

    # plurals -> singular
    "cats": "cat", "dogs": "dog", "elephants": "elephant", "flowers": "flower",
    "forests": "forest", "fruits": "fruit", "humans": "human", "landscapes": "landscape",
    "receipts": "receipt", "seas": "sea", "streets": "street",

    # screenshot -> others
    "screenshot": "others", "screenshots": "others",

    # a few common synonyms (optional)
    "people": "human", "person": "human", "documents": "receipt", "doc": "receipt",
}

def canonicalize(label: str) -> Optional[str]:
    """Return canonical singular/others, or None if cannot map."""
    s = (label or "").strip().lower().replace("_", " ")
    if not s:
        return None
    if s in CATEGORY_CANON:
        return s
    return ALIASES.get(s)

def _canonize_list(labels: Iterable[str]) -> List[str]:
    """Map list to canonical unique labels, preserving order."""
    out: List[str] = []
    seen = set()
    for lb in labels:
        c = canonicalize(lb)
        if not c or c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out

# Display-only pluralization map (NOT used for storage)
PLURAL_FOR: Dict[str, str] = {
    "cat": "cats",
    "dog": "dogs",
    "elephant": "elephants",
    "flower": "flowers",
    "forest": "forests",
    "fruit": "fruits",
    "human": "humans",
    "landscape": "landscapes",
    "receipt": "receipts",
    "sea": "seas",
    "street": "streets",
    "others": "others",
}

def _to_output_name(canon: str, prefer_plural: bool) -> str:
    """Optionally pluralize for display while keeping internal canon stable."""
    if not prefer_plural:
        return canon
    return PLURAL_FOR.get(canon, canon)

# Default label set for zero-shot
DEFAULT_LABELS: Tuple[str, ...] = CATEGORY_CANON

# ---------------------------------------------------------------------
# Prompt engineering
# ---------------------------------------------------------------------
def prompts_for(label: str) -> List[str]:
    """Build a small set of discriminative prompts; we average their embeddings."""
    l = canonicalize(label) or label
    l = str(l).lower().strip().replace("_", " ")

    if l == "others":
        return [
            "a generic photo with unknown category",
            "a miscellaneous scene",
            "an image that does not fit known categories",
            "random picture, undefined subject",
            "various objects, unclear category",
        ]

    return [
        f"a photo of {l}",
        f"{l}, high quality photo",
        f"natural {l}",
        f"{l}, close-up",
        f"{l}, detailed",
    ]

def _average_text_embedding(prompts: Iterable[str]) -> np.ndarray:
    """Encode multiple prompts and average to one normalized vector."""
    vecs = [EMB.encode_text(p) for p in prompts]  # already normalized
    if not vecs:
        raise ValueError("No prompts to encode.")
    M = np.stack(vecs, axis=0)  # (m, dim)
    v = M.mean(axis=0)
    n = np.linalg.norm(v)
    return (v if n == 0 else v / n).astype("float32")

# ---------------------------------------------------------------------
# Scoring / prediction
# ---------------------------------------------------------------------
@dataclass
class TaggingResult:
    primary: str                  # output name (may be plural if prefer_plural=True)
    labels: List[str]             # output names (aligned with prefer_plural)
    scores: Dict[str, float]      # scores keyed by canonical names

def score_image(image_path: str, labels: Optional[Iterable[str]] = None) -> Dict[str, float]:
    """
    Compute cosine/IP scores for an image against each label prototype.
    Returns: {canonical_label: score in [-1,1]} (higher is better).
    """
    label_list_in = list(labels) if labels is not None else list(DEFAULT_LABELS)
    label_list = _canonize_list(label_list_in)
    if not label_list:
        raise ValueError("Empty label set for auto-tagging after canonicalization.")

    ivec = EMB.encode_image(image_path)  # normalized (dim,)

    protos: List[np.ndarray] = []
    for lbl in label_list:
        pv = _average_text_embedding(prompts_for(lbl))
        protos.append(pv)

    P = np.stack(protos, axis=0)  # (L, dim)
    scores = P @ ivec             # cosine/IP (both normalized)
    return {label_list[i]: float(scores[i]) for i in range(len(label_list))}

def predict_labels(
    image_path: str,
    labels: Optional[Iterable[str]] = None,
    top_k: int = 3,
    threshold: float = 0.30,
    *,
    prefer_plural: bool = False,  # <- default: DO NOT pluralize
) -> TaggingResult:
    """
    Pick a primary label (argmax) and optional set of labels.
    Returns output names transformed by prefer_plural for display,
    but scores keys always stay canonical/singular for storage.
    """
    scores = score_image(image_path, labels=labels)
    if not scores:
        raise RuntimeError("No scores computed — empty label set?")

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    primary_canon = ordered[0][0]

    keep = [(lbl, sc) for lbl, sc in ordered if sc >= threshold]
    if not keep:
        keep = [ordered[0]]
    keep = keep[:max(1, top_k)]

    label_list_canon = [lbl for lbl, _ in keep]
    if primary_canon not in label_list_canon:
        label_list_canon.insert(0, primary_canon)

    # Only for display we optionally pluralize:
    primary_out = _to_output_name(primary_canon, prefer_plural)
    labels_out = [_to_output_name(lbl, prefer_plural) for lbl in label_list_canon]

    return TaggingResult(primary=primary_out, labels=labels_out, scores=scores)
