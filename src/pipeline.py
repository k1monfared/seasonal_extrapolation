"""End-to-end pipeline: data generation and the full analysis run.

Kept separate from the thin ``scripts/`` entry points so the same functions
back both the single-command demo and the standalone data generator.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from . import seasonality as S
from . import datagen
from .seasonal_model import fit_seasonal
from .extrapolate import naive_extrapolation, aware_extrapolation, HORIZON_HOURS
from .transportability import homogeneity_test
from . import calibration as calib
from . import runlength as rl

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data")
OUTPUTS = os.path.join(REPO, "outputs")
DOCS = os.path.join(REPO, "docs")
IMAGES = os.path.join(DOCS, "images")


def load_config(path=None) -> dict:
    if path is None:
        path = os.path.join(REPO, "configs", "default.json")
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_and_save_data(cfg: dict) -> dict:
    """Generate the baseline, ground truth, and an example experiment; commit."""
    os.makedirs(DATA, exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])

    baseline = datagen.generate_baseline(cfg, rng)
    baseline.to_csv(os.path.join(DATA, "baseline_hourly.csv"), index=False)

    truth = S.true_period_baselines(cfg)
    truth_factors = S.true_factor_table()
    ground = {"period_baselines": truth, "true_factors": truth_factors,
              "true_relative_lift": cfg["true_relative_lift"],
              "true_impact": {
                  "week": cfg["true_relative_lift"] * truth["week"],
                  "month": cfg["true_relative_lift"] * truth["month"],
                  "year": cfg["true_relative_lift"] * truth["annual"],
              }}
    with open(os.path.join(DATA, "ground_truth.json"), "w") as f:
        json.dump(ground, f, indent=2)

    exp = cfg["default_experiment"]
    exp_rng = np.random.default_rng(cfg["seed"] + 7)
    ex = datagen.simulate_experiment(exp["start"], exp["hours"], cfg, exp_rng,
                                     return_hourly=True)
    ex["hourly"].to_csv(os.path.join(DATA, "example_experiment.csv"), index=True)

    # A compact seasonal-factor CSV for quick inspection.
    rows = []
    for dim in ("hour", "dow", "month"):
        for k, v in truth_factors[dim].items():
            rows.append({"dimension": dim, "level": k, "true_factor": v})
    pd.DataFrame(rows).to_csv(os.path.join(DATA, "seasonal_factors_true.csv"),
                              index=False)
    return {"baseline": baseline, "ground": ground}


def load_baseline() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(DATA, "baseline_hourly.csv"),
                     parse_dates=["timestamp"])
    return df


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _factor_recovery_report(fit, cfg) -> dict:
    rec = fit.recover_factors()
    truth = S.true_factor_table()
    report = {}
    for dim in ("hour", "dow", "month"):
        levels = rec[dim]["levels"]
        tvals = [truth[dim][int(l)] for l in levels]
        rvals = rec[dim]["factor"]
        err = [abs(rv - tv) for rv, tv in zip(rvals, tvals)]
        cover = [lo <= tv <= hi for lo, hi, tv in
                 zip(rec[dim]["ci_low"], rec[dim]["ci_high"], tvals)]
        report[dim] = {
            "levels": levels,
            "true": tvals,
            "recovered": rvals,
            "ci_low": rec[dim]["ci_low"],
            "ci_high": rec[dim]["ci_high"],
            "max_abs_error": float(max(err)),
            "mean_abs_error": float(np.mean(err)),
            "ci_coverage": float(np.mean(cover)),
        }
    report["trend_annual_growth_recovered"] = rec["trend_annual_growth"]
    report["trend_annual_growth_true"] = cfg["annual_growth"]
    return report


def _local_effect_and_composition(start, hours, cfg, b_year_true):
    """Analytic local effect multiplier g_local and composition factor c_local.

    g_local = volume-weighted mean of the true effect modulation over the window
              (annual mean is 1 by construction), so g_local - 1 is the residual
              relative-effect bias the aware method cannot remove.
    c_local = window baseline rate / annual baseline rate, so c_local - 1 is the
              baseline-composition bias the aware method DOES remove.
    """
    idx = pd.DatetimeIndex(pd.date_range(pd.Timestamp(start), periods=hours, freq="h"))
    comp = S.deterministic_components(idx, cfg)
    rev = comp["revenue"].values
    g = S.effect_multiplier(idx, cfg)
    g_local = float(np.sum(rev * g) / np.sum(rev))
    window_rate = float(np.sum(rev) / hours)
    annual_rate = b_year_true / 8760.0
    c_local = window_rate / annual_rate
    return g_local, c_local


def _effect_heterogeneity_analysis(baseline_df, cfg) -> dict:
    """Quantify the residual bias left after the baseline-composition correction.

    The seasonality-aware method fixes WHERE the volume is across the cycle, but
    it still assumes ONE relative lift for every hour and day. Because the true
    lift is stronger in the evening and on weekends, a window placed in a
    low-effect or high-effect period measures a local lift that is not the annual
    average, and the annualized estimate stays biased by (g_local - 1) even after
    the composition correction. That residual is systematic: it does not shrink
    with more samples in the same window, because it comes from the unobserved
    variation of the effect, not from sampling noise.
    """
    fit = fit_seasonal(baseline_df, cfg)
    tb = S.true_period_baselines(cfg)
    b_year_true = tb["annual"]
    r_true = cfg["true_relative_lift"]
    truth_year = r_true * b_year_true
    conf = cfg["effect_heterogeneity"]
    n_reps = conf["n_reps"]

    windows = []
    for w in conf["windows"]:
        g_local, c_local = _local_effect_and_composition(
            w["start"], w["hours"], cfg, b_year_true)
        rng = np.random.default_rng(cfg["seed"] + 5000)
        aws, nvs, rs, ses = [], [], [], []
        for _ in range(n_reps):
            ex = datagen.simulate_experiment(w["start"], w["hours"], cfg, rng)
            rs.append(ex["r_hat"]); ses.append(ex["se_r_hat"])
            aws.append(aware_extrapolation(ex, fit, cfg)["year"]["point"])
            nvs.append(naive_extrapolation(ex, cfg)["year"]["point"])
        aws, nvs = np.array(aws), np.array(nvs)
        windows.append({
            "label": w["label"], "start": w["start"], "hours": w["hours"],
            "g_local": g_local, "c_local": c_local,
            "mean_local_lift": float(np.mean(rs)),
            "mean_se_r": float(np.mean(ses)),
            "aware_residual_bias_pct": float(100 * (aws.mean() - truth_year) / truth_year),
            "naive_total_bias_pct": float(100 * (nvs.mean() - truth_year) / truth_year),
            "predicted_aware_bias_pct": float(100 * (g_local - 1.0)),
            "predicted_naive_bias_pct": float(100 * (g_local * c_local - 1.0)),
            "composition_bias_pct": float(100 * (c_local - 1.0)),
        })

    # Sample-size invariance: same window, more traffic, so the sampling error
    # shrinks but the residual bias does not.
    iw = conf["invariance_window"]
    invariance = []
    for scale in conf["sample_scales"]:
        cfg_s = dict(cfg)
        cfg_s["base_sessions_per_hour"] = cfg["base_sessions_per_hour"] * scale
        rng = np.random.default_rng(cfg["seed"] + 5500)
        aws, ses = [], []
        for _ in range(n_reps):
            ex = datagen.simulate_experiment(iw["start"], iw["hours"], cfg_s, rng)
            ses.append(ex["se_r_hat"])
            # Annual baseline reconstruction uses the ORIGINAL fitted model.
            aws.append(aware_extrapolation(ex, fit, cfg)["year"]["point"])
        aws = np.array(aws)
        invariance.append({
            "sessions_scale": scale,
            "mean_se_r": float(np.mean(ses)),
            "aware_residual_bias_pct": float(100 * (aws.mean() - truth_year) / truth_year),
        })

    return {"window": iw["label"], "truth_year": truth_year,
            "windows": windows, "invariance": invariance}


def _explorer_grid(baseline_df, cfg) -> dict:
    """Precompute extrapolations over a grid of windows for the HTML explorer."""
    ctx = calib.MCContext(baseline_df, cfg)
    truth = calib._truth(cfg)
    rng = np.random.default_rng(cfg["seed"] + 3000)
    n_reps = cfg["explorer"]["n_reps"]

    cells = []
    for anchor in cfg["explorer"]["anchors"]:
        for hours in cfg["explorer"]["hours_grid"]:
            agg = {m: {h: {"pts": [], "los": [], "his": [], "cov": 0}
                       for h in HORIZON_HOURS} for m in ("naive", "aware")}
            for _ in range(n_reps):
                exp_result = datagen.simulate_experiment(anchor["start"], hours, cfg, rng)
                fit = ctx.sample_fit(rng)
                by = ctx.b_year(fit)
                naive = naive_extrapolation(exp_result, cfg)
                aware = aware_extrapolation(exp_result, fit, cfg, b_year=by,
                                            n_hours_year=ctx.n_hours_year)
                for m, res in (("naive", naive), ("aware", aware)):
                    for h in HORIZON_HOURS:
                        d = res[h]
                        agg[m][h]["pts"].append(d["point"])
                        agg[m][h]["los"].append(d["ci_low"])
                        agg[m][h]["his"].append(d["ci_high"])
                        if d["ci_low"] <= truth[h] <= d["ci_high"]:
                            agg[m][h]["cov"] += 1
            cell = {"anchor": anchor["label"], "start": anchor["start"], "hours": hours}
            for m in ("naive", "aware"):
                cell[m] = {}
                for h in HORIZON_HOURS:
                    a = agg[m][h]
                    cell[m][h] = {
                        "point": float(np.mean(a["pts"])),
                        "ci_low": float(np.mean(a["los"])),
                        "ci_high": float(np.mean(a["his"])),
                        "coverage": a["cov"] / n_reps,
                    }
            cells.append(cell)
    return {"truth": truth, "cells": cells,
            "horizon_hours": HORIZON_HOURS}


def _variance_profile(baseline_df, cfg) -> list:
    """Share of the annual-interval variance from the experiment vs the seasonal
    reconstruction, as the length of available baseline history varies.

    With a multi-year always-on baseline the seasonal factors are estimated so
    precisely that the annual interval is dominated by the short experiment; the
    seasonal term is correctly propagated but small. It grows when history is
    short and the forward projection of next year's baseline is less certain.
    """
    df = baseline_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    end = df["timestamp"].max()
    exp = cfg["default_experiment"]
    rows = []
    for months in [12, 18, 24, 36]:
        sub = df[df["timestamp"] > end - pd.Timedelta(days=int(months * 30.4))]
        fit = fit_seasonal(sub, cfg)
        rng = np.random.default_rng(cfg["seed"] + 4000 + months)
        ve, vs = [], []
        for _ in range(80):
            ex = datagen.simulate_experiment(exp["start"], exp["hours"], cfg, rng)
            a = aware_extrapolation(ex, fit, cfg)["year"]
            if np.isfinite(a["var_experiment"]) and np.isfinite(a["var_seasonal"]):
                ve.append(a["var_experiment"]); vs.append(a["var_seasonal"])
        ve_m, vs_m = float(np.mean(ve)), float(np.mean(vs))
        tot = ve_m + vs_m
        rows.append({"history_months": months,
                     "experiment_share": 100 * ve_m / tot,
                     "seasonal_share": 100 * vs_m / tot})
    return rows


def run_analysis(cfg: dict, baseline_df=None) -> dict:
    if baseline_df is None:
        baseline_df = load_baseline()

    truth = S.true_period_baselines(cfg)
    true_impact = {"week": cfg["true_relative_lift"] * truth["week"],
                   "month": cfg["true_relative_lift"] * truth["month"],
                   "year": cfg["true_relative_lift"] * truth["annual"]}

    # 1) Learn seasonality.
    fit = fit_seasonal(baseline_df, cfg)
    factor_report = _factor_recovery_report(fit, cfg)

    # 2) Default experiment + extrapolation + transportability.
    exp = cfg["default_experiment"]
    exp_rng = np.random.default_rng(cfg["seed"] + 7)
    ex = datagen.simulate_experiment(exp["start"], exp["hours"], cfg, exp_rng,
                                     return_hourly=True)
    naive = naive_extrapolation(ex, cfg)
    aware = aware_extrapolation(ex, fit, cfg)
    transport = homogeneity_test(ex["hourly"], cfg)
    default_exp = {
        "label": exp.get("label", ""),
        "start": exp["start"], "hours": exp["hours"],
        "r_hat": ex["r_hat"], "se_r_hat": ex["se_r_hat"],
        "true_relative_lift": cfg["true_relative_lift"],
        "ctrl_rev_per_hour": ex["ctrl_rev_per_hour"],
        "naive": naive, "aware": aware, "transportability": transport,
    }

    # 3) Calibration study.
    calibration = calib.run_calibration(baseline_df, cfg)

    # 4) Minimum run-length.
    runlength = rl.run_runlength(baseline_df, cfg)

    # 5) Variance profile vs baseline history length.
    variance_profile = _variance_profile(baseline_df, cfg)

    # 6) Effect-heterogeneity residual bias (what the aware method cannot fix).
    effect_heterogeneity = _effect_heterogeneity_analysis(baseline_df, cfg)

    # 7) Explorer grid.
    explorer = _explorer_grid(baseline_df, cfg)

    return {
        "config": cfg,
        "ground_truth": {"period_baselines": truth, "true_impact": true_impact},
        "factor_recovery": factor_report,
        "default_experiment": default_exp,
        "calibration": calibration,
        "runlength": runlength,
        "variance_profile": variance_profile,
        "effect_heterogeneity": effect_heterogeneity,
        "explorer": explorer,
    }


def save_outputs(results: dict):
    os.makedirs(OUTPUTS, exist_ok=True)
    slim = dict(results)
    # explorer grid written both as JSON and as a JS assignment so the static
    # docs/index.html works when opened directly from disk (no server needed).
    with open(os.path.join(DOCS, "explorer_data.json"), "w") as f:
        json.dump(results["explorer"], f, default=float)
    with open(os.path.join(DOCS, "explorer_data.js"), "w") as f:
        f.write("window.EXPLORER_DATA = ")
        json.dump(results["explorer"], f, default=float)
        f.write(";\n")
    with open(os.path.join(OUTPUTS, "results.json"), "w") as f:
        json.dump(slim, f, indent=2, default=float)
