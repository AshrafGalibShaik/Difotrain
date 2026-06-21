"""Language conditioning.

A real VLA uses a pretrained text encoder; for a dependency-light, fully
testable default we ship a deterministic bag-of-words encoder. It implements the
same ``encode(text) -> np.ndarray`` contract, so swapping in a transformer
encoder later is a drop-in replacement.
"""
from __future__ import annotations

import re
from typing import Iterable, List

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class LanguageEncoder:
    """Fixed-vocabulary bag-of-words encoder.

    The vocabulary is either supplied explicitly or fit from a corpus. Encoding
    is deterministic and order-independent, which is all the toy reaching task
    needs while still making the instruction genuinely drive the action.
    """

    def __init__(self, vocab: Iterable[str] | None = None):
        self.vocab: List[str] = sorted(set(vocab)) if vocab is not None else []
        self._index = {w: i for i, w in enumerate(self.vocab)}

    @classmethod
    def fit(cls, instructions: Iterable[str]) -> "LanguageEncoder":
        words = set()
        for text in instructions:
            words.update(tokenize(text))
        return cls(sorted(words))

    @property
    def dim(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            idx = self._index.get(tok)
            if idx is not None:
                vec[idx] = 1.0
        return vec

    def encode_batch(self, texts: Iterable[str]) -> np.ndarray:
        return np.stack([self.encode(t) for t in texts]) if self.vocab else np.zeros((0, 0))

    def to_dict(self) -> dict:
        return {"vocab": self.vocab}

    @classmethod
    def from_dict(cls, d: dict) -> "LanguageEncoder":
        return cls(d.get("vocab", []))
