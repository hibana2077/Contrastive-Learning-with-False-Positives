from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from .config import ExperimentIOConfig, SyntheticDataConfig, TwoStageConfig
from .methods import TwoStageAnchoredEstimator, OracleSegmentationEstimator
from .runner import run_seeds, summarize_seed_results, write_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Synthetic Figure 2: drift bias O(gamma L) vs anchoring frequency")

    p.add_argument("--T", type=int, default=2000)
    p.add_argument("--N", type=int, default=256)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.05)
    p.add_argument("--delta", type=float, default=0.25)

    p.add_argument("--r", type=int, default=32)
    p.add_argument("--m_event", type=int, default=512)
    p.add_argument("--m_bg", type=int, default=512)

    p.add_argument("--L_list", type=int, nargs="+", default=[5, 10, 20, 40, 80, 160])

    p.add_argument("--seeds", type=int, default=200)
    p.add_argument("--out_dir", type=str, default="exp/synth/out/anchoring_tradeoff")

    p.add_argument("--use_oracle_seg", action="store_true", help="Use oracle segmentation (bypass screening)")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    io = ExperimentIOConfig(out_dir=args.out_dir, seeds=args.seeds)
    out_dir = Path(io.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = SyntheticDataConfig(T=args.T, N=args.N, sigma=args.sigma, gamma=args.gamma, delta=float(args.delta))

    rows = []
    xs = []
    ys = []
    yerr = []

    for L in args.L_list:
        method_cfg = TwoStageConfig(
            r=int(args.r),
            m_event=int(args.m_event),
            m_bg=int(args.m_bg),
            L=int(L),
            threshold="oracle",
        )

        if args.use_oracle_seg:
            est = OracleSegmentationEstimator(method_cfg)
        else:
            est = TwoStageAnchoredEstimator(method_cfg, delta_for_oracle=float(args.delta))

        seed_results = run_seeds(data_cfg=data_cfg, estimator=est, seeds=range(args.seeds))
        summary = summarize_seed_results(seed_results)

        row = {
            "T": args.T,
            "N": args.N,
            "sigma": args.sigma,
            "gamma": args.gamma,
            "delta": float(args.delta),
            "r": int(args.r),
            "m_event": int(args.m_event),
            "m_bg": int(args.m_bg),
            "L": int(L),
            "estimator": est.name,
            **summary,
        }
        rows.append(row)

        xs.append(int(L))
        ys.append(summary["score_error_mean"])
        yerr.append(summary["score_error_std"])

    plt.errorbar(xs, ys, yerr=yerr, marker="o", linewidth=2)
    plt.xscale("log", base=2)
    plt.xlabel("L (background anchor interval)")
    plt.ylabel("|ŝ - s*| (score error)")
    plt.title("Anchoring trade-off: drift bias ~ O(γL)")

    csv_path = out_dir / f"anchoring_tradeoff_{'oracle' if args.use_oracle_seg else 'two_stage'}.csv"
    fig_path = out_dir / f"anchoring_tradeoff_{'oracle' if args.use_oracle_seg else 'two_stage'}.png"

    write_csv(str(csv_path), rows)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {fig_path}")


if __name__ == "__main__":
    main()
