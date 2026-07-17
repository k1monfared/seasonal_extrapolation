"""Synthetic data generation: always-on baseline and short experiment windows.

Design choices (documented so the synthetic world is auditable):

1. Metric = hourly revenue = sessions * value_per_session.
2. Seasonality (see seasonality.py) lives mostly in traffic, with a mild
   per-session value seasonality. Traffic seasonality is what creates the
   composition trap the framework is built to solve.
3. Observation noise is MULTIPLICATIVE and lognormal, with a standard
   deviation that shrinks with traffic (sigma ~ k / sqrt(sessions)). Busy
   hours are measured more precisely than quiet hours, which is realistic for
   an aggregate of many independent sessions, and it makes the WLS weights in
   the seasonal model meaningful.
4. The treatment effect is modeled as a RELATIVE (multiplicative) lift on the
   per-session value. Relative lift is the primary model because it tends to
   be transportable across the cycle: a "+5% conversion value" behaves the
   same whether traffic is high or low, whereas an absolute per-hour lift
   mechanically scales with traffic and does not transport. An additive
   variant is provided for comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import seasonality as S


# ---------------------------------------------------------------------------
# Always-on baseline
# ---------------------------------------------------------------------------

def hourly_noise_sigma(sessions: np.ndarray, cfg: dict) -> np.ndarray:
    """Lognormal sigma for the hourly metric, shrinking with traffic."""
    return cfg["hourly_noise_k"] / np.sqrt(sessions) + cfg["hourly_noise_floor"]


def generate_baseline(cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Generate the noisy always-on control baseline over the history window.

    Returns an hourly DataFrame with the realized (observed) revenue plus the
    deterministic mean columns kept for transparency and validation.
    """
    idx = pd.date_range(
        pd.Timestamp(cfg["history_start"]),
        pd.Timestamp(cfg["history_end"]) + pd.Timedelta(hours=23),
        freq="h",
    )
    comp = S.deterministic_components(idx, cfg)
    mean_rev = comp["revenue"].values
    sigma = hourly_noise_sigma(comp["sessions"].values, cfg)

    # Lognormal multiplicative noise with mean 1: exp(sigma z - sigma^2 / 2).
    z = rng.standard_normal(len(idx))
    mult = np.exp(sigma * z - 0.5 * sigma ** 2)
    revenue = mean_rev * mult

    out = pd.DataFrame(
        {
            "timestamp": idx,
            "hour": idx.hour,
            "dow": idx.dayofweek,
            "month": idx.month,
            "sessions_mean": comp["sessions"].values,
            "value_mean": comp["value_per_session"].values,
            "revenue_mean": mean_rev,
            "noise_sigma": sigma,
            "revenue": revenue,
        }
    )
    return out


# ---------------------------------------------------------------------------
# Short experiment on a window (A/B test)
# ---------------------------------------------------------------------------

def _window_index(start: str, hours: int) -> pd.DatetimeIndex:
    start_ts = pd.Timestamp(start)
    return pd.date_range(start_ts, periods=hours, freq="h")


def window_baseline_frame(start: str, hours: int, cfg: dict) -> pd.DataFrame:
    """Deterministic baseline components for the hours of an experiment window."""
    idx = _window_index(start, hours)
    comp = S.deterministic_components(idx, cfg)
    comp = comp.copy()
    comp["hour"] = idx.hour
    comp["dow"] = idx.dayofweek
    comp["month"] = idx.month
    return comp


def simulate_experiment(
    start: str,
    hours: int,
    cfg: dict,
    rng: np.random.Generator,
    return_hourly: bool = False,
):
    """Simulate an A/B experiment on the window and estimate the relative lift.

    Each hour's sessions are split into control and treatment arms. Control
    per-session value has mean ``value_mean`` and session-level sd
    ``session_value_sd``. Treatment applies the true effect. We aggregate to a
    session-count-weighted mean per arm over the whole window and estimate the
    relative lift with its sampling variance via the delta method.

    Returns a dict with r_hat, var_r_hat (variance of the relative-lift
    estimate), the control revenue rate per hour in the window, and window
    metadata. This is the ONLY thing the extrapolator learns about the effect.
    """
    comp = window_baseline_frame(start, hours, cfg)
    sessions = comp["sessions"].values
    value_mean = comp["value_per_session"].values
    share = cfg["ab_treatment_share"]
    sd = cfg["session_value_sd"]

    n_treat = np.maximum(np.round(sessions * share), 2).astype(int)
    n_ctrl = np.maximum(np.round(sessions * (1.0 - share)), 2).astype(int)

    # True effect at each hour = annual-average lift * within-week modulation.
    g = S.effect_multiplier(pd.DatetimeIndex(_window_index(start, hours)), cfg)
    if cfg["effect_type"] == "relative":
        treat_mean_true = value_mean * (1.0 + cfg["true_relative_lift"] * g)
    else:
        treat_mean_true = value_mean + cfg["additive_lift"] * g

    # Simulate arm sample means. Sum of sessions is large, so the sample mean
    # is Gaussian by CLT with variance sd^2 / n.
    ctrl_means = value_mean + rng.standard_normal(len(comp)) * sd / np.sqrt(n_ctrl)
    treat_means = treat_mean_true + rng.standard_normal(len(comp)) * sd / np.sqrt(n_treat)

    # Session-count-weighted pooled means over the window.
    w_c = n_ctrl / n_ctrl.sum()
    w_t = n_treat / n_treat.sum()
    ctrl_bar = float(np.sum(w_c * ctrl_means))
    treat_bar = float(np.sum(w_t * treat_means))

    # Variance of each pooled mean (independent hours).
    var_ctrl_bar = float(np.sum(w_c ** 2 * sd ** 2 / n_ctrl))
    var_treat_bar = float(np.sum(w_t ** 2 * sd ** 2 / n_treat))

    # Relative lift r = treat_bar / ctrl_bar - 1. Delta method for variance.
    r_hat = treat_bar / ctrl_bar - 1.0
    var_r_hat = (
        var_treat_bar / ctrl_bar ** 2
        + var_ctrl_bar * treat_bar ** 2 / ctrl_bar ** 4
    )

    # Control revenue rate per hour observed in the window (naive analyst uses
    # this as if it represented every hour of the year).
    ctrl_rev_per_hour = float(np.mean(sessions * ctrl_means))

    result = {
        "start": start,
        "hours": int(hours),
        "n_hours": int(len(comp)),
        "total_sessions": float(sessions.sum()),
        "r_hat": r_hat,
        "var_r_hat": var_r_hat,
        "se_r_hat": float(np.sqrt(var_r_hat)),
        "ctrl_rev_per_hour": ctrl_rev_per_hour,
        "covered_dows": sorted(set(int(d) for d in comp["dow"].values)),
        "covered_hours": sorted(set(int(h) for h in comp["hour"].values)),
        "covered_months": sorted(set(int(m) for m in comp["month"].values)),
    }
    if return_hourly:
        hourly = comp.copy()
        hourly["ctrl_mean"] = ctrl_means
        hourly["treat_mean"] = treat_means
        result["hourly"] = hourly
    return result
