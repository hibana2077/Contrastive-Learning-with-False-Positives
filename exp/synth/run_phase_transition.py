from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt

from .config import ExperimentIOConfig, SyntheticDataConfig, TwoStageConfig
from .methods import TwoStageAnchoredEstimator
from .runner import run_seeds, summarize_seed_results, write_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Synthetic Figure 1: kappa phase transition in screening")

    p.add_argument("--T", type=int, default=2000)
    p.add_argument("--N", type=int, default=256)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.05)
    p.add_argument("--deltas", type=float, nargs="+", default=[0.12, 0.16, 0.20, 0.28])
    p.add_argument("--r_list", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])

    p.add_argument("--seeds", type=int, default=200)
    p.add_argument("--out_dir", type=str, default="exp/synth/out/phase_transition")
    p.add_argument("--metric", type=str, choices=["recall", "f1"], default="recall")

    p.add_argument("--threshold", type=str, choices=["oracle", "mad"], default="oracle")
    p.add_argument("--mad_k", type=float, default=6.0)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    io = ExperimentIOConfig(out_dir=args.out_dir, seeds=args.seeds)
    out_dir = Path(io.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for delta in args.deltas:
        data_cfg = SyntheticDataConfig(T=args.T, N=args.N, sigma=args.sigma, gamma=args.gamma, delta=float(delta))
        kappa = float(delta) - 2.0 * float(args.gamma)

        xs = []
        ys = []
        yerr = []

        for r in args.r_list:
            method_cfg = TwoStageConfig(
                r=int(r),
                m_event=1,
                m_bg=1,
                L=20,
                threshold=args.threshold,
                mad_k=args.mad_k,
            )
            est = TwoStageAnchoredEstimator(method_cfg, delta_for_oracle=float(delta))
            seed_results = run_seeds(data_cfg=data_cfg, estimator=est, seeds=range(args.seeds))
            summary = summarize_seed_results(seed_results)

            metric_mean = summary[f"boundary_{args.metric}_mean"]
            metric_std = summary[f"boundary_{args.metric}_std"]

            row = {
                "T": args.T,
                "N": args.N,
                "sigma": args.sigma,
                "gamma": args.gamma,
                "delta": float(delta),
                "kappa": kappa,
                "r": int(r),
                "threshold": args.threshold,
                "mad_k": args.mad_k,
                **summary,
                "metric": args.metric,
            }
            rows.append(row)

            xs.append(int(r))
            ys.append(metric_mean)
            yerr.append(metric_std)

        plt.errorbar(xs, ys, yerr=yerr, marker="o", linewidth=2, label=f"Δ={delta:.2f} (κ={kappa:.2f})")

    plt.xscale("log", base=2)
    plt.ylim(-0.02, 1.02)
    plt.xlabel("r (tokens per time step; Stage I)")
    plt.ylabel(f"Boundary {args.metric}")
    plt.title("Screening phase transition vs κ = Δ - 2γ")
    plt.legend()

    csv_path = out_dir / "phase_transition.csv"
    fig_path = out_dir / f"phase_transition_{args.metric}.png"

    write_csv(str(csv_path), rows)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {fig_path}")


if __name__ == "__main__":
    main()
