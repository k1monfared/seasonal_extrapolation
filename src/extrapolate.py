"""Extrapolate a short-window effect to weekly / monthly / yearly impact.

Two estimators are implemented so we can contrast them:

NAIVE (time-scaling): take the absolute impact per hour observed in the window
and multiply by the number of hours in the target period. This implicitly
assumes the window is representative of every hour of the year. It is biased
whenever the window's traffic composition differs from the annual average
(the composition trap) and its interval ignores both the seasonal
reconstruction uncertainty and any period never observed, so it is too narrow.

SEASONALITY-AWARE (composition reweighting): the treatment is a relative lift,
so annual impact = r_hat * B_year, where B_year is the annual baseline total
reconstructed from the always-on data through the learned seasonal model. This
reweights the window's relative effect by the true seasonal composition. Its
interval combines three sources of uncertainty:
  (a) experiment sampling variance of r_hat,
  (b) seasonal-factor and trend estimation variance in B_year_hat,
  (c) a transportability term for the assumption that the relative effect is
      stable across parts of the cycle the experiment never touched.
Uncertainty is propagated with the delta method (product of two independent
estimates); a Monte Carlo cross-check is available in calibration.py.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from . import seasonality as S


# Canonical horizon lengths in hours.
HORIZON_HOURS = {"week": 168.0, "month": 730.0, "year": 8760.0}


def weekly_cycle_coverage(exp_result: dict) -> float:
    """Fraction of the 168 (dow, hour) cells of the weekly cycle the window hit.

    This is the testable footprint of the experiment on the weekly seasonality.
    A window shorter than a full week cannot cover the whole cycle, which is the
    "seasonality floor" behind the minimum run-length recommendation.
    """
    dows = set(exp_result["covered_dows"])
    hours = set(exp_result["covered_hours"])
    if exp_result["n_hours"] >= 168:
        return 1.0
    # Approximate the covered cells: count distinct (dow, hour) reached.
    covered_cells = min(exp_result["n_hours"], len(dows) * len(hours))
    return float(covered_cells) / 168.0


def naive_extrapolation(exp_result: dict, cfg: dict) -> dict:
    """Time-scaling extrapolation with a sampling-only interval."""
    z = stats.norm.ppf(0.5 + cfg["confidence"] / 2.0)
    r = exp_result["r_hat"]
    var_r = exp_result["var_r_hat"]
    rate = exp_result["ctrl_rev_per_hour"]  # baseline revenue per hour in window

    out = {}
    for name, hours in HORIZON_HOURS.items():
        point = r * rate * hours
        var = (rate * hours) ** 2 * var_r
        se = np.sqrt(var)
        out[name] = {
            "point": float(point),
            "se": float(se),
            "ci_low": float(point - z * se),
            "ci_high": float(point + z * se),
        }
    return out


def aware_extrapolation(exp_result: dict, fit, cfg: dict, b_year=None,
                        n_hours_year: float = 8760.0) -> dict:
    """Composition-reweighted extrapolation with a fully propagated interval.

    ``b_year`` may be a precomputed ``(B_year, var_B_year)`` tuple to avoid
    rebuilding the annual design matrix inside a Monte Carlo loop; otherwise it
    is reconstructed from ``fit``.
    """
    z = stats.norm.ppf(0.5 + cfg["confidence"] / 2.0)
    r = exp_result["r_hat"]
    var_r = exp_result["var_r_hat"]

    # Reconstruct the annual baseline total (and variance) from the model.
    if b_year is None:
        year_idx = S.year_index(cfg["target_year"])
        B_year, var_B_year = fit.predict_period_total(year_idx)
        n_hours_year = float(len(year_idx))
    else:
        B_year, var_B_year = b_year

    # Transportability: extra relative uncertainty on the effect for the part
    # of the weekly cycle the experiment never observed.
    coverage = weekly_cycle_coverage(exp_result)
    tau = cfg.get("transportability_sd", 0.0) * (1.0 - coverage)
    var_r_eff = var_r + tau ** 2

    out = {"B_year_hat": float(B_year), "se_B_year": float(np.sqrt(var_B_year)),
           "weekly_cycle_coverage": coverage}
    for name, hours in HORIZON_HOURS.items():
        # Scale the annual baseline to the canonical period length.
        B = B_year * (hours / n_hours_year)
        var_B = var_B_year * (hours / n_hours_year) ** 2
        point = r * B
        # Var of product of independent estimates.
        var = B ** 2 * var_r_eff + r ** 2 * var_B + var_r_eff * var_B
        se = np.sqrt(var)
        out[name] = {
            "point": float(point),
            "se": float(se),
            "ci_low": float(point - z * se),
            "ci_high": float(point + z * se),
            "var_experiment": float(B ** 2 * var_r),
            "var_seasonal": float(r ** 2 * var_B),
            "var_transport": float(B ** 2 * tau ** 2),
        }
    return out
