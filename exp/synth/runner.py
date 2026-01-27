from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .data import SyntheticDataset, generate_synthetic_dataset
from .methods import Estimator, EstimateResult


@dataclass(frozen=True)
class SeedResult:
    seed: int
    result: EstimateResult


def run_seeds(
    *,
    data_cfg,
    estimator: Estimator,
    seeds: Iterable[int],
) -> list[SeedResult]:
    out: list[SeedResult] = []
    for seed in seeds:
        ds: SyntheticDataset = generate_synthetic_dataset(data_cfg, seed=int(seed))
        rng = np.random.default_rng(int(seed) + 12345)
        res = estimator.estimate(ds, rng=rng)
        out.append(SeedResult(seed=int(seed), result=res))
    return out


def summarize_seed_results(seed_results: list[SeedResult]) -> dict[str, float]:
    score_err = np.array([sr.result.score_error for sr in seed_results], dtype=np.float64)
    prec = np.array([sr.result.boundary.precision for sr in seed_results], dtype=np.float64)
    rec = np.array([sr.result.boundary.recall for sr in seed_results], dtype=np.float64)
    f1 = np.array([sr.result.boundary.f1 for sr in seed_results], dtype=np.float64)
    budget = np.array([sr.result.budget_used for sr in seed_results], dtype=np.float64)

    def ms(x: np.ndarray) -> tuple[float, float]:
        return float(np.mean(x)), float(np.std(x))

    m_score, s_score = ms(score_err)
    m_prec, s_prec = ms(prec)
    m_rec, s_rec = ms(rec)
    m_f1, s_f1 = ms(f1)
    m_budget, s_budget = ms(budget)

    return {
        "score_error_mean": m_score,
        "score_error_std": s_score,
        "boundary_precision_mean": m_prec,
        "boundary_precision_std": s_prec,
        "boundary_recall_mean": m_rec,
        "boundary_recall_std": s_rec,
        "boundary_f1_mean": m_f1,
        "boundary_f1_std": s_f1,
        "budget_used_mean": m_budget,
        "budget_used_std": s_budget,
        "n": float(len(seed_results)),
    }


def write_csv(path: str | os.PathLike, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write")
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)
