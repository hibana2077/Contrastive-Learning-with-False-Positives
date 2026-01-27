from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SyntheticDataConfig:
    """Configuration for generating the 1D synthetic score-direction data."""

    T: int = 2000
    N: int = 256

    num_segments: int = 5
    segment_len: int = 20
    min_gap: int = 100

    sigma: float = 1.0
    gamma: float = 0.05
    delta: float = 0.25


@dataclass(frozen=True)
class TwoStageConfig:
    """Configuration for the proposed two-stage estimator."""

    # Stage I
    r: int = 32

    # Stage II refinement sampling per anchor
    m_event: int = 128
    m_bg: int = 128

    # Background anchoring frequency (every L steps)
    L: int = 20

    # Threshold strategy
    threshold: str = "oracle"  # "oracle" or "mad"
    tau: float | None = None  # if set, overrides threshold strategy

    # MAD parameters (used when threshold == "mad")
    mad_k: float = 6.0


@dataclass(frozen=True)
class UniformConfig:
    """Uniform baseline sampling config."""

    budget: int


@dataclass(frozen=True)
class ExperimentIOConfig:
    out_dir: str = "exp/synth/out"
    seeds: int = 200
    boundary_tolerance: int = 1
