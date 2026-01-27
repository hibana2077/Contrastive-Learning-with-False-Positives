from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundaryMetrics:
    precision: float
    recall: float
    f1: float


def boundary_metrics(
    true_boundaries: list[int],
    pred_boundaries: list[int],
    *,
    tolerance: int = 1,
) -> BoundaryMetrics:
    """Match boundaries with a +/- tolerance window, one-to-one."""

    true_sorted = sorted(true_boundaries)
    pred_sorted = sorted(pred_boundaries)

    used_pred = [False] * len(pred_sorted)
    matched = 0

    for tb in true_sorted:
        best_j = None
        best_dist = None
        for j, pb in enumerate(pred_sorted):
            if used_pred[j]:
                continue
            dist = abs(pb - tb)
            if dist <= tolerance:
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_j = j
        if best_j is not None:
            used_pred[best_j] = True
            matched += 1

    precision = matched / len(pred_sorted) if pred_sorted else 0.0
    recall = matched / len(true_sorted) if true_sorted else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return BoundaryMetrics(precision=precision, recall=recall, f1=f1)
