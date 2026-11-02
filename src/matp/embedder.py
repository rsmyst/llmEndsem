"""
Enhanced Local Deterministic Vectorizer for Zero-Dependency Semantic Vector Search.
Combines subword n-gram hashing with term frequency saturation (TF-IDF style) and L2 normalization
to achieve high-fidelity cosine matching between query intents and distilled trajectory records.
"""

import math
import hashlib
from typing import List, Dict
from collections import Counter

class LocalNGramEmbedder:
    """
    Deterministic Subword & N-Gram Feature Vectorizer with Term Saturation.
    Generates a normalized float vector of dimension D without external APIs or GPUs.
    """
    def __init__(self, dim: int = 256, n_gram_range: tuple = (1, 3)):
        self.dim = dim
        self.min_n, self.max_n = n_gram_range

    def _tokenize(self, text: str) -> List[str]:
        clean = "".join(c.lower() if c.isalnum() or c.isspace() or c in ('_', '-', '.') else " " for c in text)
        words = clean.split()
        tokens = list(words)
        
        # Add character n-grams for subword and prefix/suffix matching
        for word in words:
            if len(word) >= self.min_n:
                for n in range(self.min_n, min(self.max_n + 1, len(word) + 1)):
                    for i in range(len(word) - n + 1):
                        tokens.append(word[i:i+n])
        return tokens

    def embed(self, text: str) -> List[float]:
        tokens = self._tokenize(text)
        vec = [0.0] * self.dim
        if not tokens:
            return vec

        counts = Counter(tokens)
        for token, count in counts.items():
            # Term saturation: sqrt(count) or log(1 + count) to prevent raw boilerplate explosion
            weight = math.log(1.0 + count)
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            bucket = h % self.dim
            sign = 1.0 if (h >> 16) % 2 == 0 else -1.0
            vec[bucket] += sign * weight

        # L2 Normalize
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 1e-9:
            vec = [x / norm for x in vec]
        return vec

    @staticmethod
    def cosine_similarity(v1: List[float], v2: List[float]) -> float:
        if len(v1) != len(v2) or not v1:
            return 0.0
        dot = sum(a * b for a, b in zip(v1, v2))
        return float(max(-1.0, min(1.0, dot)))
