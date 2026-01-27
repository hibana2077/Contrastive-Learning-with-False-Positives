from __future__ import annotations

import numpy as np


class TokenSampler:
    """Samples token indices for a given time t."""

    def __init__(self, N: int):
        if N <= 0:
            raise ValueError("N must be positive")
        self.N = N

    def sample(self, rng: np.random.Generator, k: int) -> np.ndarray:
        if k <= 0:
            raise ValueError("k must be positive")
        if k <= self.N:
            return rng.choice(self.N, size=k, replace=False)
        return rng.choice(self.N, size=k, replace=True)
