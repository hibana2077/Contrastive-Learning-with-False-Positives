from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .config import SyntheticDataConfig


@dataclass(frozen=True)
class EventInterval:
    start: int  # inclusive
    end: int  # exclusive
    beta: float

    def contains(self, t: int) -> bool:
        return self.start <= t < self.end


@dataclass
class SyntheticDataset:
    config: SyntheticDataConfig
    b: np.ndarray  # (T,)
    beta: np.ndarray  # (T,)
    a: np.ndarray  # (T,)
    y: np.ndarray  # (T, N)
    events: list[EventInterval]

    @property
    def s_star(self) -> float:
        return float(np.mean(self.a))

    @property
    def true_boundaries(self) -> list[int]:
        # Boundaries are entry/exit points between t-1 and t, represented by t.
        boundaries: list[int] = []
        for e in self.events:
            boundaries.append(e.start)
            boundaries.append(e.end)
        boundaries = sorted(set([b for b in boundaries if 0 < b < self.config.T]))
        return boundaries


def _sample_event_starts(
    *,
    rng: np.random.Generator,
    T: int,
    num_segments: int,
    segment_len: int,
    min_gap: int,
) -> list[int]:
    """Sample non-overlapping event starts with minimum gaps."""

    starts: list[int] = []
    max_start = T - segment_len
    if max_start <= 1:
        raise ValueError("T too small for the given segment_len")

    attempts = 0
    while len(starts) < num_segments:
        attempts += 1
        if attempts > 50_000:
            raise RuntimeError("Failed to place events; relax min_gap/num_segments")

        s = int(rng.integers(1, max_start))
        ok = True
        for s0 in starts:
            # intervals [s, s+len) and [s0, s0+len) must be separated by min_gap
            if not (s + segment_len + min_gap <= s0 or s0 + segment_len + min_gap <= s):
                ok = False
                break
        if ok:
            starts.append(s)

    return sorted(starts)


def generate_synthetic_dataset(
    config: SyntheticDataConfig,
    *,
    seed: int,
) -> SyntheticDataset:
    rng = np.random.default_rng(seed)

    # Background drift b_t with increments in {+gamma, -gamma}
    b = np.zeros(config.T, dtype=np.float64)
    steps = rng.choice([-config.gamma, config.gamma], size=config.T - 1)
    b[1:] = np.cumsum(steps)

    # Events: constant shift +/- delta over each segment
    beta = np.zeros(config.T, dtype=np.float64)
    starts = _sample_event_starts(
        rng=rng,
        T=config.T,
        num_segments=config.num_segments,
        segment_len=config.segment_len,
        min_gap=config.min_gap,
    )

    events: list[EventInterval] = []
    for s in starts:
        sign = float(rng.choice([-1.0, 1.0]))
        shift = sign * float(config.delta)
        e = s + config.segment_len
        beta[s:e] = shift
        events.append(EventInterval(start=s, end=e, beta=shift))

    a = b + beta

    # y_{t,i} = a_t + eps
    y = a[:, None] + rng.normal(0.0, config.sigma, size=(config.T, config.N))

    return SyntheticDataset(config=config, b=b, beta=beta, a=a, y=y, events=events)


def iter_background_anchors(T: int, L: int) -> Iterable[int]:
    if L <= 0:
        raise ValueError("L must be positive")
    for t in range(0, T, L):
        yield t
    if (T - 1) % L != 0:
        yield T - 1
