#!/usr/bin/env python
"""Generate and commit the synthetic data (baseline + ground truth + example).

Standalone entry point; also invoked by scripts/run_demo.py.
Usage: python scripts/generate_data.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import pipeline


def main():
    cfg = pipeline.load_config()
    print(f"Generating synthetic data (seed={cfg['seed']}) ...")
    out = pipeline.generate_and_save_data(cfg)
    baseline = out["baseline"]
    ground = out["ground"]
    print(f"  baseline_hourly.csv: {len(baseline):,} hourly rows "
          f"({cfg['history_start']} .. {cfg['history_end']})")
    print(f"  true annual baseline revenue: "
          f"${ground['period_baselines']['annual']/1e6:,.2f}M "
          f"({cfg['target_year']})")
    print(f"  true annual impact at {cfg['true_relative_lift']:.1%} lift: "
          f"${ground['true_impact']['year']/1e6:,.2f}M")
    print("  wrote data/: baseline_hourly.csv, ground_truth.json, "
          "example_experiment.csv, seasonal_factors_true.csv")


if __name__ == "__main__":
    main()
