"""Ground-truth seasonal structure of the data-generating process (DGP).

This module defines the TRUE seasonality of the always-on control baseline.
Everything here is the "physics" of the synthetic world. The analysis
framework never imports these true values, it only sees noisy realized data,
so recovering these numbers is a genuine test.

The metric is hourly revenue for a large consumer web platform, decomposed as

    revenue_hour(t) = sessions(t) * value_per_session(t)

Seasonality lives mostly in the traffic (sessions) because that is what drives
the composition trap: a short high-traffic window is not representative of the
whole year. All seasonal factors are multiplicative and centered so that their
geometric mean is 1, which makes "factor" interpretable as "relative to a
typical hour / day / month".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# True multiplicative seasonal factors (geometric mean normalized to 1).
# ---------------------------------------------------------------------------

# Intra-day shape: quiet overnight, morning ramp, evening peak around 20:00.
_HOUR_RAW = np.array([
    0.45, 0.38, 0.34, 0.33, 0.36, 0.45,   # 00-05
    0.62, 0.85, 1.05, 1.12, 1.10, 1.08,   # 06-11
    1.12, 1.15, 1.10, 1.05, 1.08, 1.20,   # 12-17
    1.40, 1.55, 1.60, 1.45, 1.05, 0.70,   # 18-23
])

# Day-of-week (0=Mon ... 6=Sun): weekends clearly lower.
_DOW_RAW = np.array([1.06, 1.08, 1.09, 1.07, 1.02, 0.80, 0.76])

# Month (index 0=Jan ... 11=Dec): summer dip, strong Q4 holiday build.
_MONTH_RAW = np.array([
    0.95, 0.92, 0.97, 1.00, 1.03, 0.98,   # Jan-Jun
    0.92, 0.90, 1.00, 1.06, 1.16, 1.28,   # Jul-Dec
])

# Mild per-session value seasonality (people spend a bit more per session in Q4).
_VALUE_MONTH_RAW = np.array([
    0.97, 0.96, 0.98, 1.00, 1.01, 1.00,
    0.98, 0.97, 1.00, 1.03, 1.08, 1.12,
])


def _geo_normalize(x: np.ndarray) -> np.ndarray:
    """Scale a positive vector so its geometric mean equals 1."""
    x = np.asarray(x, dtype=float)
    return x / np.exp(np.mean(np.log(x)))


HOUR_FACTOR = _geo_normalize(_HOUR_RAW)
DOW_FACTOR = _geo_normalize(_DOW_RAW)
MONTH_FACTOR = _geo_normalize(_MONTH_RAW)
VALUE_MONTH_FACTOR = _geo_normalize(_VALUE_MONTH_RAW)


# The seasonal model is fit on REVENUE = sessions * value_per_session. Hour and
# day-of-week seasonality live only in traffic, so the true revenue factors for
# those dimensions equal the traffic factors. The monthly revenue factor,
# however, combines the traffic month factor and the per-session value month
# factor, so the true monthly revenue factor is their (renormalized) product.
REVENUE_MONTH_FACTOR = _geo_normalize(MONTH_FACTOR * VALUE_MONTH_FACTOR)


def true_factor_table() -> dict:
    """Return the true multiplicative REVENUE factors by dimension and level.

    These are what the seasonal model (fit on revenue) should recover.
    """
    return {
        "hour": {int(h): float(HOUR_FACTOR[h]) for h in range(24)},
        "dow": {int(d): float(DOW_FACTOR[d]) for d in range(7)},
        "month": {int(m + 1): float(REVENUE_MONTH_FACTOR[m]) for m in range(12)},
    }


# ---------------------------------------------------------------------------
# Deterministic mean of the metric (no noise). This is the ground truth used
# to compute true weekly / monthly / yearly baseline totals and impacts.
# ---------------------------------------------------------------------------

def _years_since_start(index: pd.DatetimeIndex, start: pd.Timestamp) -> np.ndarray:
    return (index - start).total_seconds().values / (365.25 * 24 * 3600.0)


def deterministic_components(index: pd.DatetimeIndex, cfg: dict) -> pd.DataFrame:
    """Compute the deterministic (noise-free) mean of sessions, value, revenue.

    Parameters
    ----------
    index : hourly DatetimeIndex.
    cfg   : configuration dict (uses base levels, growth).
    """
    start = pd.Timestamp(cfg["history_start"])
    tau = _years_since_start(index, start)
    growth = (1.0 + cfg["annual_growth"]) ** tau

    hour = index.hour.values
    dow = index.dayofweek.values
    month = index.month.values - 1

    sessions = (
        cfg["base_sessions_per_hour"]
        * growth
        * HOUR_FACTOR[hour]
        * DOW_FACTOR[dow]
        * MONTH_FACTOR[month]
    )
    value = cfg["base_value_per_session"] * VALUE_MONTH_FACTOR[month]
    revenue = sessions * value

    return pd.DataFrame(
        {"sessions": sessions, "value_per_session": value, "revenue": revenue},
        index=index,
    )


# ---------------------------------------------------------------------------
# True treatment-effect modulation across the weekly cycle.
#
# The relative lift is NOT perfectly constant: it is modestly stronger in the
# evening and on weekends and weaker overnight. This within-week heterogeneity
# is what makes the "cover a full cycle" rule bite for the trustworthy method:
# a window that touches only part of the week measures a volume-weighted effect
# for that part, which is not the annual average, so extrapolating it is biased.
# The modulation is normalized so its ANNUAL volume-weighted mean is 1, which
# means the configured lift equals the true annual-average relative effect and
# the true period impacts stay exactly (annual-average lift) * (period baseline).
# The effect pattern does NOT vary by month, so one full week is sufficient to
# capture the average; cross-month effect stability is an assumption we flag.
# ---------------------------------------------------------------------------

_EFFECT_DAYPART = {"overnight": 0.80, "daytime": 0.95, "evening": 1.20}
_EFFECT_WEEKEND = 1.12

_effect_norm_cache: dict = {}


def _daypart_factor(hour: np.ndarray) -> np.ndarray:
    f = np.full(len(hour), _EFFECT_DAYPART["daytime"])
    f[hour < 6] = _EFFECT_DAYPART["overnight"]
    f[hour >= 18] = _EFFECT_DAYPART["evening"]
    return f


def _raw_effect_multiplier(index: pd.DatetimeIndex) -> np.ndarray:
    dp = _daypart_factor(index.hour.values)
    wk = np.where(index.dayofweek.values >= 5, _EFFECT_WEEKEND, 1.0)
    return dp * wk


def _effect_norm(cfg: dict) -> float:
    key = (cfg["target_year"], cfg["base_sessions_per_hour"], cfg["annual_growth"])
    if key not in _effect_norm_cache:
        idx = year_index(cfg["target_year"])
        rev = deterministic_components(idx, cfg)["revenue"].values
        graw = _raw_effect_multiplier(idx)
        _effect_norm_cache[key] = float(np.sum(rev * graw) / np.sum(rev))
    return _effect_norm_cache[key]


def effect_multiplier(index: pd.DatetimeIndex, cfg: dict) -> np.ndarray:
    """Normalized effect modulation g(t); annual volume-weighted mean is 1."""
    return _raw_effect_multiplier(index) / _effect_norm(cfg)


def year_index(year: int) -> pd.DatetimeIndex:
    """Hourly index covering a full calendar year."""
    return pd.date_range(
        f"{year}-01-01 00:00:00", f"{year}-12-31 23:00:00", freq="h"
    )


def true_period_baselines(cfg: dict) -> dict:
    """True baseline revenue totals for the target year at several horizons.

    Returns a dict with the annual total, a representative (mean) week, a
    representative (mean) calendar month, plus per-month and per-dow detail.
    All values are noise-free ground truth from the deterministic mean.
    """
    year = cfg["target_year"]
    idx = year_index(year)
    comp = deterministic_components(idx, cfg)
    rev = comp["revenue"]

    annual = float(rev.sum())
    n_hours = len(idx)
    # A representative week / month expressed as the annual total rescaled to
    # the canonical period length (168 h, 730 h). This is what "weekly impact"
    # means to an executive: the steady-state contribution of one such period.
    week = annual * (168.0 / n_hours)
    month = annual * (730.0 / n_hours)

    per_month = {
        int(m): float(rev[idx.month == m].sum()) for m in range(1, 13)
    }
    per_dow = {
        int(d): float(rev[idx.dayofweek == d].sum()) for d in range(7)
    }

    return {
        "target_year": year,
        "n_hours": n_hours,
        "annual": annual,
        "week": week,
        "month": month,
        "per_month": per_month,
        "per_dow": per_dow,
    }
