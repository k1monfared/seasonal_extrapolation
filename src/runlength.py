"""Minimum run-length analysis.

As a function of how many hours the experiment runs and where in the cycle it
starts, we measure the precision and calibration of the annualized estimate.
Two ideas drive the recommendation:

* Sample-size: longer windows collect more sessions, shrinking the experiment
  sampling variance and the annual interval.
* Seasonality floor: a window shorter than one full weekly cycle cannot see all
  days of the week, so its composition is unrepresentative. The naive estimate
  is then biased and even the aware estimate is noisier. Covering at least one
  full week removes the weekly composition bias regardless of start phase.
"""

from __future__ import annotations

import numpy as np

from . import datagen
from .calibration import MCContext, _truth
from .extrapolate import naive_extrapolation, aware_extrapolation


def _mc_point(ctx, truth, cfg, start, hours, n_reps, rng):
    horizon = "year"
    t = truth[horizon]
    rec = {m: {"cov": 0, "pts": [], "half": []} for m in ("naive", "aware")}
    for _ in range(n_reps):
        exp_result = datagen.simulate_experiment(start, hours, cfg, rng)
        fit = ctx.sample_fit(rng)
        by = ctx.b_year(fit)
        naive = naive_extrapolation(exp_result, cfg)
        aware = aware_extrapolation(exp_result, fit, cfg, b_year=by,
                                    n_hours_year=ctx.n_hours_year)
        for m, res in (("naive", naive), ("aware", aware)):
            d = res[horizon]
            if d["ci_low"] <= t <= d["ci_high"]:
                rec[m]["cov"] += 1
            rec[m]["pts"].append(d["point"])
            rec[m]["half"].append((d["ci_high"] - d["ci_low"]) / 2.0)
    out = {"start": start, "hours": hours}
    for m in ("naive", "aware"):
        pts = np.array(rec[m]["pts"])
        half = np.array(rec[m]["half"])
        out[m] = {
            "coverage": rec[m]["cov"] / n_reps,
            "bias_pct": float(100.0 * (pts.mean() - t) / t),
            "rel_halfwidth_pct": float(100.0 * half.mean() / t),
        }
    return out


def run_runlength(baseline_df, cfg):
    ctx = MCContext(baseline_df, cfg)
    truth = _truth(cfg)
    rng = np.random.default_rng(cfg["seed"] + 2000)
    n_reps = cfg["runlength"]["n_reps"]

    grid = []
    for anchor in cfg["runlength"]["start_anchors"]:
        for hours in cfg["runlength"]["hours_grid"]:
            res = _mc_point(ctx, truth, cfg, anchor["start"], hours, n_reps, rng)
            res["anchor"] = anchor["label"]
            grid.append(res)

    recommendation = _recommend(grid, cfg)
    return {"grid": grid, "recommendation": recommendation,
            "nominal": cfg["calibration"]["nominal"],
            "precision_target_pct": 25.0}


def _recommend(grid, cfg):
    """Recommend the smallest run length that is calibrated across all start
    phases and meets a relative-precision target."""
    nominal = cfg["calibration"]["nominal"]
    cov_tol = 0.03
    precision_target = 25.0  # annual interval half-width within +/-25% of point

    hours_set = sorted({g["hours"] for g in grid})
    per_hours = {}
    for h in hours_set:
        cells = [g for g in grid if g["hours"] == h]
        min_cov = min(c["aware"]["coverage"] for c in cells)
        max_abs_bias = max(abs(c["aware"]["bias_pct"]) for c in cells)
        max_half = max(c["aware"]["rel_halfwidth_pct"] for c in cells)
        naive_min_cov = min(c["naive"]["coverage"] for c in cells)
        naive_max_bias = max(abs(c["naive"]["bias_pct"]) for c in cells)
        per_hours[h] = {
            "hours": h,
            "aware_min_coverage": min_cov,
            "aware_max_abs_bias_pct": max_abs_bias,
            "aware_max_rel_halfwidth_pct": max_half,
            "naive_min_coverage": naive_min_cov,
            "naive_max_abs_bias_pct": naive_max_bias,
            "covers_full_week": h >= 168,
        }

    chosen = None
    for h in hours_set:
        s = per_hours[h]
        if (s["covers_full_week"]
                and s["aware_min_coverage"] >= nominal - cov_tol
                and s["aware_max_rel_halfwidth_pct"] <= precision_target):
            chosen = h
            break

    return {
        "recommended_hours": chosen,
        "recommended_days": None if chosen is None else round(chosen / 24.0, 2),
        "criteria": {
            "cover_full_weekly_cycle": True,
            "aware_coverage_at_least": nominal - cov_tol,
            "aware_rel_halfwidth_at_most_pct": precision_target,
        },
        "per_hours": per_hours,
    }
