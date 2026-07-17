"""Monte Carlo calibration: does each interval cover the truth at nominal rate?

For a given experiment window we repeatedly (a) resimulate the short experiment
and (b) resimulate the always-on baseline and refit the seasonal model, form
the naive and seasonality-aware annual/weekly/monthly intervals, and record
whether each interval covers the KNOWN true impact. Aggregating over
replications gives empirical coverage, bias, and interval width. This is the
core integrity check: the seasonality-aware interval should cover at the
nominal rate while the naive interval is biased and over-confident.
"""

from __future__ import annotations

import numpy as np

from . import seasonality as S
from . import datagen
from .seasonal_model import FastSeasonalRefitter, build_design
from .extrapolate import naive_extrapolation, aware_extrapolation, HORIZON_HOURS


class MCContext:
    """Precomputed, reusable state for fast Monte Carlo replication."""

    def __init__(self, baseline_df, cfg):
        self.cfg = cfg
        self.refitter = FastSeasonalRefitter(baseline_df, cfg)
        sigma = baseline_df["noise_sigma"].values
        self.base_log_mean = np.log(baseline_df["revenue_mean"].values) - 0.5 * sigma ** 2
        self.base_sigma = sigma
        year_idx = S.year_index(cfg["target_year"])
        self.X_year, _ = build_design(year_idx, cfg)
        self.n_hours_year = float(len(year_idx))

    def sample_fit(self, rng):
        z = rng.standard_normal(len(self.base_sigma))
        log_rev = self.base_log_mean + self.base_sigma * z
        return self.refitter.fit_from_log_revenue(log_rev)

    def b_year(self, fit):
        pred = np.exp(self.X_year @ fit.beta) * fit.smear
        B = float(pred.sum())
        grad = self.X_year.T @ pred
        var_B = float(grad @ fit.cov_beta @ grad)
        return B, var_B


def _truth(cfg):
    tb = S.true_period_baselines(cfg)
    r = cfg["true_relative_lift"]
    return {"week": r * tb["week"], "month": r * tb["month"], "year": r * tb["annual"]}


def run_window_calibration(ctx, window, truth, cfg, rng, single_fit=None):
    """Run the replication loop for one window; return naive/aware summaries."""
    n_reps = cfg["calibration"]["n_reps"]
    refit = cfg["calibration"]["refit_baseline"]
    horizons = list(HORIZON_HOURS.keys())

    rec = {m: {h: {"cov": 0, "pts": [], "widths": []} for h in horizons}
           for m in ("naive", "aware")}

    for _ in range(n_reps):
        exp_result = datagen.simulate_experiment(
            window["start"], window["hours"], cfg, rng)
        if refit:
            fit = ctx.sample_fit(rng)
        else:
            fit = single_fit
        by = ctx.b_year(fit)

        naive = naive_extrapolation(exp_result, cfg)
        aware = aware_extrapolation(exp_result, fit, cfg, b_year=by,
                                    n_hours_year=ctx.n_hours_year)
        for method, res in (("naive", naive), ("aware", aware)):
            for h in horizons:
                d = res[h]
                t = truth[h]
                if d["ci_low"] <= t <= d["ci_high"]:
                    rec[method][h]["cov"] += 1
                rec[method][h]["pts"].append(d["point"])
                rec[method][h]["widths"].append(d["ci_high"] - d["ci_low"])

    summary = {"label": window["label"], "start": window["start"],
               "hours": window["hours"], "n_reps": n_reps}
    for method in ("naive", "aware"):
        summary[method] = {}
        for h in horizons:
            pts = np.array(rec[method][h]["pts"])
            widths = np.array(rec[method][h]["widths"])
            t = truth[h]
            summary[method][h] = {
                "coverage": rec[method][h]["cov"] / n_reps,
                "mean_point": float(pts.mean()),
                "true": float(t),
                "bias": float(pts.mean() - t),
                "bias_pct": float(100.0 * (pts.mean() - t) / t),
                "rmse": float(np.sqrt(np.mean((pts - t) ** 2))),
                "mean_ci_width": float(widths.mean()),
            }
    return summary


def run_calibration(baseline_df, cfg, seed_offset=0):
    """Run calibration across all configured windows."""
    ctx = MCContext(baseline_df, cfg)
    truth = _truth(cfg)
    rng = np.random.default_rng(cfg["seed"] + 1000 + seed_offset)

    single_fit = None
    if not cfg["calibration"]["refit_baseline"]:
        from .seasonal_model import fit_seasonal
        single_fit = fit_seasonal(baseline_df, cfg)

    results = []
    for window in cfg["calibration"]["windows"]:
        results.append(run_window_calibration(ctx, window, truth, cfg, rng, single_fit))
    return {"nominal": cfg["calibration"]["nominal"], "truth": truth,
            "windows": results}
