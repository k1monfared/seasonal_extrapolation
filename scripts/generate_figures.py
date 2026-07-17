#!/usr/bin/env python
"""Regenerate all figures from committed outputs/results.json and data.

Standalone entry point; also invoked by scripts/run_demo.py.
Usage: python scripts/generate_figures.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import pipeline, plotting


def main():
    results_path = os.path.join(pipeline.OUTPUTS, "results.json")
    if not os.path.exists(results_path):
        print("outputs/results.json not found; run scripts/run_demo.py first.")
        sys.exit(1)
    with open(results_path) as f:
        results = json.load(f)
    baseline = pipeline.load_baseline()
    print("Generating figures ...")
    paths = plotting.generate_all(results, baseline, pipeline.IMAGES)
    for p in paths:
        print(f"  {os.path.relpath(p, pipeline.REPO)}")


if __name__ == "__main__":
    main()
