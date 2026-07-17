"""Make the transportability assumption explicit and testable.

The seasonality-aware extrapolation multiplies one measured relative lift by the
annual baseline. That is only valid if the lift measured on the window equals
the annual-average lift. When the window spans more than one kind of period we
can TEST for effect heterogeneity by estimating the lift within each sub-period
and checking whether they agree; when the window touches only one kind of
period the assumption cannot be tested and is flagged.

We test two partitions of the covered hours, daypart (overnight / daytime /
evening) and weekend vs weekday, because those are the dimensions along which
the effect can plausibly move. Each partition gets a weighted chi-square test
of equal effects. A single short window often lacks the power to confirm
heterogeneity even when it exists, which is itself a reason to cover a full
cycle rather than trust a slice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _daypart(hour: int) -> str:
    if hour < 6:
        return "overnight"
    if hour < 18:
        return "daytime"
    return "evening"


def _group_effect(sub: pd.DataFrame, cfg: dict):
    """Session-weighted relative lift and variance for a sub-frame of hours."""
    sessions = sub["sessions"].values
    share = cfg["ab_treatment_share"]
    sd = cfg["session_value_sd"]
    n_t = np.maximum(np.round(sessions * share), 2)
    n_c = np.maximum(np.round(sessions * (1 - share)), 2)
    w_c = n_c / n_c.sum()
    w_t = n_t / n_t.sum()
    ctrl_bar = float(np.sum(w_c * sub["ctrl_mean"].values))
    treat_bar = float(np.sum(w_t * sub["treat_mean"].values))
    var_c = float(np.sum(w_c ** 2 * sd ** 2 / n_c))
    var_t = float(np.sum(w_t ** 2 * sd ** 2 / n_t))
    r = treat_bar / ctrl_bar - 1.0
    var_r = var_t / ctrl_bar ** 2 + var_c * treat_bar ** 2 / ctrl_bar ** 4
    return r, var_r


def _partition_test(df: pd.DataFrame, key: str, cfg: dict):
    groups = {}
    for name, sub in df.groupby(key):
        if len(sub) < 2:
            continue
        r, var_r = _group_effect(sub, cfg)
        if var_r <= 0 or not np.isfinite(var_r):
            continue
        groups[str(name)] = {"r_hat": float(r), "se": float(np.sqrt(var_r)),
                             "n_hours": int(len(sub))}
    if len(groups) < 2:
        return {"testable": False, "groups": groups}
    rs = np.array([g["r_hat"] for g in groups.values()])
    ws = np.array([1.0 / g["se"] ** 2 for g in groups.values()])
    r_bar = float(np.sum(ws * rs) / np.sum(ws))
    stat = float(np.sum(ws * (rs - r_bar) ** 2))
    dof = len(groups) - 1
    pval = float(stats.chi2.sf(stat, dof))
    return {"testable": True, "groups": groups, "pooled_r": r_bar,
            "chi2": stat, "dof": dof, "p_value": pval,
            "homogeneous": bool(pval > 0.05)}


def homogeneity_test(exp_hourly: pd.DataFrame, cfg: dict) -> dict:
    """Test effect homogeneity across daypart and weekend partitions."""
    df = exp_hourly.copy()
    df["daypart"] = [_daypart(int(h)) for h in df["hour"].values]
    df["weekend"] = np.where(df["dow"].values >= 5, "weekend", "weekday")

    daypart = _partition_test(df, "daypart", cfg)
    weekend = _partition_test(df, "weekend", cfg)

    testable = daypart["testable"] or weekend["testable"]
    result = {"daypart": daypart, "weekend": weekend, "testable": testable}

    if not testable:
        result["flag"] = (
            "Transportability NOT testable: the window covers a single period "
            "type. Stability of the relative effect across the cycle is assumed, "
            "not verified. Cover a full weekly cycle before trusting the annual "
            "number.")
        return result

    msgs = []
    any_hetero = False
    for name, res in (("daypart", daypart), ("weekend", weekend)):
        if not res["testable"]:
            continue
        if res["p_value"] <= 0.05:
            any_hetero = True
            msgs.append(f"{name.capitalize()} partition shows heterogeneity "
                        f"(chi2={res['chi2']:.2f}, dof={res['dof']}, "
                        f"p={res['p_value']:.3f})")
        else:
            msgs.append(f"{name.capitalize()} partition is consistent with a "
                        f"single lift (p={res['p_value']:.3f})")
    result["any_heterogeneity"] = any_hetero
    prefix = ("Effect heterogeneity flagged. " if any_hetero
              else "No significant heterogeneity on the covered cycle. ")
    result["flag"] = (
        prefix + ". ".join(msgs)
        + ". Cross-month effect stability is assumed and cannot be tested from a "
          "within-month window.")
    return result
