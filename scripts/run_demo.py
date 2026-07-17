#!/usr/bin/env python
"""Single entry point: reproduce the entire demonstration from a fixed seed.

Steps:
  1. generate and commit the synthetic data,
  2. learn seasonality and run the full analysis (extrapolation, calibration,
     minimum run-length, explorer grid),
  3. write outputs/results.json and outputs/report.md,
  4. render docs/images/*.png and docs/explorer_data.json,
  5. print a concise console summary.

Usage: python scripts/run_demo.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import pipeline, plotting, report


def main():
    t0 = time.time()
    cfg = pipeline.load_config()
    print("=" * 70)
    print("Seasonal extrapolation demo  (portfolio demonstration, synthetic data)")
    print("=" * 70)

    print("\n[1/5] Generating synthetic data ...")
    data = pipeline.generate_and_save_data(cfg)
    baseline = data["baseline"]
    print(f"      baseline rows: {len(baseline):,}")

    print("\n[2/5] Running analysis (fit, extrapolate, calibrate, run-length) ...")
    results = pipeline.run_analysis(cfg, baseline)

    print("\n[3/5] Writing outputs/ ...")
    pipeline.save_outputs(results)
    report.save_report(results, pipeline.OUTPUTS)

    print("\n[4/5] Rendering figures ...")
    paths = plotting.generate_all(results, baseline, pipeline.IMAGES)
    for p in paths:
        print(f"      {os.path.relpath(p, pipeline.REPO)}")

    print("\n[5/5] Summary")
    _summary(results)
    print(f"\nDone in {time.time() - t0:.1f}s. "
          "See outputs/report.md and docs/index.html.")


def _summary(results):
    de = results["default_experiment"]
    ti = results["ground_truth"]["true_impact"]
    aw = de["aware"]["year"]
    nv = de["naive"]["year"]
    cal = results["calibration"]
    rec = results["runlength"]["recommendation"]

    def m(x):
        return f"${x/1e6:,.2f}M"

    print(f"  True annual impact          : {m(ti['year'])}")
    print(f"  Aware annual estimate       : {m(aw['point'])} "
          f"[{m(aw['ci_low'])}, {m(aw['ci_high'])}]")
    print(f"  Naive annual estimate       : {m(nv['point'])} "
          f"[{m(nv['ci_low'])}, {m(nv['ci_high'])}]  (bias vs truth "
          f"{100*(nv['point']-ti['year'])/ti['year']:+.0f}%)")
    print("  Annual-impact coverage (year horizon), by window:")
    for w in cal["windows"]:
        print(f"    {w['label']:<34s} naive {w['naive']['year']['coverage']:.0%}"
              f"   aware {w['aware']['year']['coverage']:.0%}")
    if rec["recommended_hours"] is not None:
        print(f"  Recommended minimum run length: "
              f"{rec['recommended_hours']} h "
              f"({rec['recommended_days']} days)")


if __name__ == "__main__":
    main()
