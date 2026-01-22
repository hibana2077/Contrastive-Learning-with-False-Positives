# Constrastive Learning with False Positives

## Abstract

This paper research the statistical limits of representation learning via the InfoNCE objective when augmentations introduce false positives through semantic label flipping. We establish two-sided phase transition boundaries on the flip rate $p$ by reducing the InfoNCE objective to a spiked random matrix model in a high-dimensional regime. In the success regime ($p < p_-$), we prove that any empirical near-minimizer aligns with the ground-truth semantic subspace, supporting downstream linear probing with low sample complexity. Conversely, in the failure regime ($p > p_+$), we prove that the population optimal representations are non-identifiable and contain no label information. Under specific scaling of the number of negatives $N$ and temperature $\tau$, we show these boundaries converge to an asymptotically sharp phase transition. Our results provide a rigorous theoretical foundation for understanding how label noise and hyperparameter scaling jointly determine the success of contrastive learning.

## Introduction

## Main Results

## Proof Overview

## Experiments