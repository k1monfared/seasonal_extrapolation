"""Learn seasonality from the always-on baseline, with uncertainty.

We fit a multiplicative seasonal-trend model to the hourly revenue by weighted
least squares on the log scale:

    log revenue(t) = intercept
                     + trend * years_since_start(t)
                     + hour_effect[hour(t)]
                     + dow_effect[dow(t)]
                     + month_effect[month(t)]
                     + noise

Effect (sum-to-zero) coding is used for the three seasonal dimensions so that
each coefficient is the log deviation from a typical hour / day / month, i.e.
exactly the centered log seasonal factor we want to recover. Weights are the
observed revenue, which is proportional to traffic, because the hourly noise
shrinks with traffic (see datagen.hourly_noise_sigma). Weighting therefore
matches the true heteroskedasticity.

Two things are exported to the extrapolator:

* ``recover_factors`` : recovered vs interpretable seasonal factors with CIs.
* ``predict_period_total`` : the reconstructed baseline revenue total for any
  set of hours (a week, a month, the target year), with a variance that
  propagates the trend and seasonal-factor estimation uncertainty via the
  delta method. Reconstructing a FUTURE year makes the trend extrapolation
  uncertainty real, which is the seasonal component of the annual-impact
  interval.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Design matrix (shared by fit and predict so columns always line up)
# ---------------------------------------------------------------------------

_N_HOUR = 24
_N_DOW = 7
_N_MONTH = 12


def _effect_block(levels: np.ndarray, n_levels: int) -> np.ndarray:
    """Sum-to-zero effect coding: n_levels-1 columns, last level = -1 row."""
    n = len(levels)
    block = np.zeros((n, n_levels - 1))
    for k in range(n_levels - 1):
        block[levels == k, k] = 1.0
    block[levels == (n_levels - 1), :] = -1.0
    return block


def build_design(index: pd.DatetimeIndex, cfg: dict):
    """Return (X, meta) for an hourly index. meta maps column groups to slices."""
    start = pd.Timestamp(cfg["history_start"])
    tau = (index - start).total_seconds().values / (365.25 * 24 * 3600.0)

    hour = index.hour.values
    dow = index.dayofweek.values
    month = index.month.values - 1

    intercept = np.ones((len(index), 1))
    trend = tau.reshape(-1, 1)
    hb = _effect_block(hour, _N_HOUR)
    db = _effect_block(dow, _N_DOW)
    mb = _effect_block(month, _N_MONTH)

    X = np.hstack([intercept, trend, hb, db, mb])

    i = 0
    meta = {}
    meta["intercept"] = slice(i, i + 1); i += 1
    meta["trend"] = slice(i, i + 1); i += 1
    meta["hour"] = slice(i, i + (_N_HOUR - 1)); i += _N_HOUR - 1
    meta["dow"] = slice(i, i + (_N_DOW - 1)); i += _N_DOW - 1
    meta["month"] = slice(i, i + (_N_MONTH - 1)); i += _N_MONTH - 1
    meta["p"] = i
    return X, meta


def _centered_logfactors(coef_block: np.ndarray):
    """Given effect-coded coefficients (n_levels-1), return full centered vector.

    The dropped last level equals minus the sum of the others. Returns the
    full vector and the linear map C (n_levels x (n_levels-1)) such that
    full = C @ coef_block, used to propagate covariance.
    """
    k = len(coef_block)
    C = np.vstack([np.eye(k), -np.ones((1, k))])
    return C @ coef_block, C


# ---------------------------------------------------------------------------
# Fitted model
# ---------------------------------------------------------------------------

@dataclass
class SeasonalFit:
    beta: np.ndarray
    cov_beta: np.ndarray
    smear: float
    sigma2: float
    meta: dict
    cfg: dict
    n_obs: int

    # -- reconstruction -----------------------------------------------------
    def predict_log_mean(self, index: pd.DatetimeIndex) -> np.ndarray:
        X, _ = build_design(index, self.cfg)
        return X @ self.beta

    def predict_period_total(self, index: pd.DatetimeIndex):
        """Reconstructed baseline revenue total over ``index`` with variance.

        B = sum_t exp(x_t . beta) * smear. Delta method:
            dB/dbeta = sum_t pred_t * x_t
            Var(B)   = (dB/dbeta)^T Cov(beta) (dB/dbeta)
        The smearing factor is treated as fixed (its extra variance is second
        order); this is stated as a limitation in the report.
        """
        X, _ = build_design(index, self.cfg)
        pred = np.exp(X @ self.beta) * self.smear
        B = float(pred.sum())
        grad = X.T @ pred
        var_B = float(grad @ self.cov_beta @ grad)
        return B, var_B

    # -- interpretation -----------------------------------------------------
    def recover_factors(self) -> dict:
        """Recovered multiplicative seasonal factors with 95% CIs."""
        out = {}
        for dim, n_levels in [("hour", _N_HOUR), ("dow", _N_DOW), ("month", _N_MONTH)]:
            sl = self.meta[dim]
            coef = self.beta[sl]
            cov = self.cov_beta[sl, sl]
            full_log, C = _centered_logfactors(coef)
            full_cov = C @ cov @ C.T
            se = np.sqrt(np.clip(np.diag(full_cov), 0, None))
            factor = np.exp(full_log)
            lo = np.exp(full_log - 1.96 * se)
            hi = np.exp(full_log + 1.96 * se)
            out[dim] = {
                "levels": list(range(n_levels)) if dim != "month"
                else list(range(1, 13)),
                "factor": factor.tolist(),
                "ci_low": lo.tolist(),
                "ci_high": hi.tolist(),
                "log_se": se.tolist(),
            }
        out["trend_annual_growth"] = float(np.exp(self.beta[self.meta["trend"]][0]) - 1.0)
        return out


# ---------------------------------------------------------------------------
# Fitters
# ---------------------------------------------------------------------------

def _wls_normal_equations(X: np.ndarray, y: np.ndarray, w: np.ndarray):
    """WLS via normal equations. Returns beta, cov_beta, resid, sigma2, XtWX_inv."""
    Xw = X * w[:, None]
    XtWX = X.T @ Xw
    XtWX_inv = np.linalg.inv(XtWX)
    beta = XtWX_inv @ (Xw.T @ y)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    sigma2 = float(np.sum(w * resid ** 2) / dof)
    cov_beta = sigma2 * XtWX_inv
    return beta, cov_beta, resid, sigma2, XtWX_inv


def fit_seasonal(df: pd.DataFrame, cfg: dict) -> SeasonalFit:
    """Fit the seasonal model to a baseline DataFrame (needs timestamp, revenue)."""
    index = pd.DatetimeIndex(df["timestamp"].values)
    X, meta = build_design(index, cfg)
    y = np.log(df["revenue"].values)
    w = df["revenue"].values.copy()
    w = w / w.mean()

    beta, cov_beta, resid, sigma2, _ = _wls_normal_equations(X, y, w)
    smear = float(np.sum(w * np.exp(resid)) / np.sum(w))

    return SeasonalFit(
        beta=beta, cov_beta=cov_beta, smear=smear, sigma2=sigma2,
        meta=meta, cfg=cfg, n_obs=len(y),
    )


class FastSeasonalRefitter:
    """Pre-factorized WLS refit for Monte Carlo (design matrix fixed).

    The always-on baseline timestamps are fixed across simulations; only the
    realized revenue changes. We precompute the hat operator so each refit is a
    couple of matrix multiplies, which keeps the calibration study fast.
    """

    def __init__(self, df: pd.DataFrame, cfg: dict):
        self.index = pd.DatetimeIndex(df["timestamp"].values)
        self.X, self.meta = build_design(self.index, cfg)
        self.cfg = cfg
        w = df["revenue_mean"].values.copy()  # weights from mean (fixed design)
        w = w / w.mean()
        self.w = w
        Xw = self.X * w[:, None]
        self.XtWX_inv = np.linalg.inv(self.X.T @ Xw)
        self.H = self.XtWX_inv @ Xw.T  # beta = H @ y
        self.dof = len(self.index) - self.X.shape[1]

    def fit_from_log_revenue(self, log_rev: np.ndarray) -> SeasonalFit:
        beta = self.H @ log_rev
        resid = log_rev - self.X @ beta
        sigma2 = float(np.sum(self.w * resid ** 2) / self.dof)
        cov_beta = sigma2 * self.XtWX_inv
        smear = float(np.sum(self.w * np.exp(resid)) / np.sum(self.w))
        return SeasonalFit(
            beta=beta, cov_beta=cov_beta, smear=smear, sigma2=sigma2,
            meta=self.meta, cfg=self.cfg, n_obs=len(log_rev),
        )
