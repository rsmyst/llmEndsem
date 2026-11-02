"""Token counting with an optional exact tokenizer.

``tiktoken`` is not a core dependency: when it is installed and the
``cl100k_base`` encoding can be loaded, token counts are exact for GPT-family
tokenizers.  Otherwise we fall back to a documented word/punctuation heuristic
that is closer to real BPE tokenisation than a raw ``chars / 4`` estimate.
Every benchmark result records which tokenizer was used.
"""

from __future__ import annotations

import os
import re
from typing import Optional

_WORD = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")


def _load_encoder():
    if os.environ.get("MATP_TOKENIZER", "auto").lower() in {"heuristic", "off", "none"}:
        return None
    try:
        import tiktoken  # type: ignore
    except ImportError:
        return None
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:  # pragma: no cover - offline model download failure
        return None


_ENCODER = None
_ENCODER_CHECKED = False


def _encoder():
    global _ENCODER, _ENCODER_CHECKED
    if not _ENCODER_CHECKED:
        _ENCODER = _load_encoder()
        _ENCODER_CHECKED = True
    return _ENCODER


def tokenizer_name() -> str:
    return "tiktoken/cl100k_base" if _encoder() is not None else "heuristic(words+punct)"


def count_tokens(text: str) -> int:
    """Return a deterministic token count for ``text`` (never zero)."""
    text = str(text or "")
    if not text:
        return 1
    encoder = _encoder()
    if encoder is not None:
        try:
            return max(1, len(encoder.encode(text, disallowed_special=())))
        except Exception:  # pragma: no cover - defensive
            pass
    # Heuristic: each word or punctuation mark is roughly one BPE token, with a
    # floor based on characters to approximate long identifiers/subwords.
    pieces = len(_WORD.findall(text))
    return max(1, pieces, len(text) // 6)


def truncate_to_tokens(text: str, budget: int) -> str:
    """Truncate ``text`` so that ``count_tokens`` is at most ``budget``."""
    text = str(text or "")
    if budget <= 0:
        return ""
    if count_tokens(text) <= budget:
        return text
    encoder = _encoder()
    if encoder is not None:
        try:
            return encoder.decode(encoder.encode(text, disallowed_special=())[:budget])
        except Exception:  # pragma: no cover - defensive
            pass
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count_tokens(text[:mid]) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo]
