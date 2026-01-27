# Synthetic experiment scripts

This folder contains OOP-style, reusable code to reproduce the two synthetic figures described in `exp/guide.md`.

## Install

```bash
pip install numpy matplotlib
```

## Figure 1: screening phase transition (κ = Δ - 2γ)

```bash
python -m exp.synth.run_phase_transition --seeds 200
```

Outputs:
- `exp/synth/out/phase_transition/phase_transition.csv`
- `exp/synth/out/phase_transition/phase_transition_recall.png` (or `_f1.png`)

## Figure 2: anchoring trade-off (drift bias ~ O(γL))

Two-stage (includes screening):

```bash
python -m exp.synth.run_anchoring_tradeoff --seeds 200
```

Oracle segmentation (bypasses screening; isolates anchoring bias more cleanly):

```bash
python -m exp.synth.run_anchoring_tradeoff --use_oracle_seg --seeds 200
```
