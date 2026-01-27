from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .config import TwoStageConfig, UniformConfig
from .data import SyntheticDataset, iter_background_anchors
from .metrics import BoundaryMetrics, boundary_metrics
from .samplers import TokenSampler


@dataclass(frozen=True)
class EstimateResult:
    s_hat: float
    score_error: float
    pred_boundaries: list[int]
    boundary: BoundaryMetrics
    budget_used: int


class Estimator(Protocol):
    name: str

    def estimate(self, dataset: SyntheticDataset, *, rng: np.random.Generator) -> EstimateResult: ...


def _threshold_oracle(*, delta: float) -> float:
    return float(delta) / 2.0


def _threshold_mad(g: np.ndarray, *, k: float) -> float:
    med = float(np.median(g))
    mad = float(np.median(np.abs(g - med)))
    # 1.4826 rescales MAD to std under Gaussian, but we just need a robust scale.
    return med + k * 1.4826 * mad


def _screen_boundaries(
    dataset: SyntheticDataset,
    *,
    r: int,
    tau: float,
    rng: np.random.Generator,
) -> tuple[list[int], int]:
    """Stage I: sample r tokens per t, compute g_t and threshold."""

    sampler = TokenSampler(dataset.config.N)
    T = dataset.config.T

    hat_a = np.zeros(T, dtype=np.float64)
    budget = 0

    for t in range(T):
        idx = sampler.sample(rng, r)
        hat_a[t] = float(np.mean(dataset.y[t, idx]))
        budget += len(idx)

    g = np.abs(hat_a[1:] - hat_a[:-1])
    boundaries = [int(t) for t in (np.where(g > tau)[0] + 1)]
    return boundaries, budget


def _pair_boundaries_to_events(
    boundaries: list[int],
    *,
    max_event_len: int,
    slack: int = 2,
) -> list[tuple[int, int]]:
    """Heuristic pairing: treat close consecutive boundaries as (start,end) of an event."""

    b = sorted(boundaries)
    events: list[tuple[int, int]] = []
    i = 0
    while i + 1 < len(b):
        s = b[i]
        e = b[i + 1]
        if 1 <= (e - s) <= (max_event_len + slack):
            events.append((s, e))
            i += 2
        else:
            # likely a spurious boundary; drop it
            i += 1
    return events


class TwoStageAnchoredEstimator:
    name = "two_stage"

    def __init__(self, cfg: TwoStageConfig, *, delta_for_oracle: float):
        self.cfg = cfg
        self._delta_for_oracle = float(delta_for_oracle)

    def estimate(self, dataset: SyntheticDataset, *, rng: np.random.Generator) -> EstimateResult:
        # Stage I
        if self.cfg.tau is not None:
            tau = float(self.cfg.tau)
            boundaries, b1 = _screen_boundaries(dataset, r=self.cfg.r, tau=tau, rng=rng)
        else:
            # compute boundaries with a two-pass if MAD
            if self.cfg.threshold == "oracle":
                tau = _threshold_oracle(delta=self._delta_for_oracle)
                boundaries, b1 = _screen_boundaries(dataset, r=self.cfg.r, tau=tau, rng=rng)
            elif self.cfg.threshold == "mad":
                # First run with tau=+inf to obtain g, then set tau via MAD and rerun cheaply.
                # To avoid 2x cost, compute hat_a once and build g here.
                sampler = TokenSampler(dataset.config.N)
                T = dataset.config.T
                hat_a = np.zeros(T, dtype=np.float64)
                b1 = 0
                for t in range(T):
                    idx = sampler.sample(rng, self.cfg.r)
                    hat_a[t] = float(np.mean(dataset.y[t, idx]))
                    b1 += len(idx)
                g = np.abs(hat_a[1:] - hat_a[:-1])
                tau = _threshold_mad(g, k=self.cfg.mad_k)
                boundaries = [int(t) for t in (np.where(g > tau)[0] + 1)]
            else:
                raise ValueError(f"Unknown threshold strategy: {self.cfg.threshold}")

        pred_events = _pair_boundaries_to_events(boundaries, max_event_len=dataset.config.segment_len)

        # Stage II: anchors
        sampler = TokenSampler(dataset.config.N)
        T = dataset.config.T

        # background anchors every L (piecewise constant fill)
        bg_anchor_times = list(iter_background_anchors(T, self.cfg.L))
        bg_anchor_values: dict[int, float] = {}
        budget = b1
        for t in bg_anchor_times:
            idx = sampler.sample(rng, self.cfg.m_bg)
            bg_anchor_values[t] = float(np.mean(dataset.y[t, idx]))
            budget += len(idx)

        # fill background estimate for each t by nearest previous anchor (piecewise constant)
        bg_est = np.zeros(T, dtype=np.float64)
        last_anchor_t = bg_anchor_times[0]
        last_anchor_val = bg_anchor_values[last_anchor_t]
        anchor_set = set(bg_anchor_times)
        for t in range(T):
            if t in anchor_set:
                last_anchor_t = t
                last_anchor_val = bg_anchor_values[t]
            bg_est[t] = last_anchor_val

        # event anchors: midpoint of each predicted event interval
        a_est = bg_est.copy()
        for s, e in pred_events:
            mid = (s + e) // 2
            idx = sampler.sample(rng, self.cfg.m_event)
            val = float(np.mean(dataset.y[mid, idx]))
            budget += len(idx)
            a_est[s:e] = val

        s_hat = float(np.mean(a_est))

        bm = boundary_metrics(dataset.true_boundaries, boundaries, tolerance=1)
        return EstimateResult(
            s_hat=s_hat,
            score_error=abs(s_hat - dataset.s_star),
            pred_boundaries=boundaries,
            boundary=bm,
            budget_used=budget,
        )


class OracleSegmentationEstimator:
    name = "oracle_seg"

    def __init__(self, cfg: TwoStageConfig):
        self.cfg = cfg

    def estimate(self, dataset: SyntheticDataset, *, rng: np.random.Generator) -> EstimateResult:
        sampler = TokenSampler(dataset.config.N)
        T = dataset.config.T

        # Stage II only (events are known)
        bg_anchor_times = list(iter_background_anchors(T, self.cfg.L))
        bg_anchor_values: dict[int, float] = {}
        budget = 0
        for t in bg_anchor_times:
            idx = sampler.sample(rng, self.cfg.m_bg)
            bg_anchor_values[t] = float(np.mean(dataset.y[t, idx]))
            budget += len(idx)

        bg_est = np.zeros(T, dtype=np.float64)
        last_anchor_val = bg_anchor_values[bg_anchor_times[0]]
        anchor_set = set(bg_anchor_times)
        for t in range(T):
            if t in anchor_set:
                last_anchor_val = bg_anchor_values[t]
            bg_est[t] = last_anchor_val

        a_est = bg_est.copy()
        for ev in dataset.events:
            mid = (ev.start + ev.end) // 2
            idx = sampler.sample(rng, self.cfg.m_event)
            val = float(np.mean(dataset.y[mid, idx]))
            budget += len(idx)
            a_est[ev.start : ev.end] = val

        s_hat = float(np.mean(a_est))
        bm = boundary_metrics(dataset.true_boundaries, dataset.true_boundaries, tolerance=1)
        return EstimateResult(
            s_hat=s_hat,
            score_error=abs(s_hat - dataset.s_star),
            pred_boundaries=dataset.true_boundaries,
            boundary=bm,
            budget_used=budget,
        )


class UniformEstimator:
    name = "uniform"

    def __init__(self, cfg: UniformConfig):
        self.cfg = cfg

    def estimate(self, dataset: SyntheticDataset, *, rng: np.random.Generator) -> EstimateResult:
        sampler = TokenSampler(dataset.config.N)
        T = dataset.config.T
        q = max(1, self.cfg.budget // T)

        a_bar = np.zeros(T, dtype=np.float64)
        budget = 0
        for t in range(T):
            idx = sampler.sample(rng, q)
            a_bar[t] = float(np.mean(dataset.y[t, idx]))
            budget += len(idx)

        s_hat = float(np.mean(a_bar))
        bm = boundary_metrics(dataset.true_boundaries, [], tolerance=1)
        return EstimateResult(
            s_hat=s_hat,
            score_error=abs(s_hat - dataset.s_star),
            pred_boundaries=[],
            boundary=bm,
            budget_used=budget,
        )
