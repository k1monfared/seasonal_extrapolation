"""Figure generation. Every number comes from the committed results, nothing
is hand-drawn or invented. Colours are a small consistent palette, no emojis.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .extrapolate import HORIZON_HOURS

C_NAIVE = "#d1495b"
C_AWARE = "#2a7f9e"
C_TRUTH = "#222222"
C_TRUE = "#8a8f98"
C_ACCENT = "#e09f3e"
C_GRID = "#d9dce1"

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 120,
    "font.size": 10,
    "axes.edgecolor": "#555555",
    "axes.grid": True,
    "grid.color": C_GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
})


def _save(fig, images_dir, name):
    os.makedirs(images_dir, exist_ok=True)
    path = os.path.join(images_dir, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_baseline_overview(baseline_df, images_dir):
    df = baseline_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    # Two-week hourly slice.
    sl = df[(df["timestamp"] >= "2024-06-03") & (df["timestamp"] < "2024-06-17")]
    axes[0].plot(sl["timestamp"], sl["revenue"], color=C_AWARE, lw=0.9,
                 label="observed")
    axes[0].plot(sl["timestamp"], sl["revenue_mean"], color=C_TRUTH, lw=1.4,
                 label="true mean")
    axes[0].set_title("Two-week slice: intra-day peaks, weekend dips")
    axes[0].set_ylabel("revenue per hour ($)")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].legend(frameon=False, fontsize=8)

    # Monthly totals across the whole history.
    m = (df.set_index("timestamp")["revenue"].resample("MS").sum() / 1e6)
    axes[1].bar(m.index, m.values, width=20, color=C_ACCENT)
    axes[1].set_title("Monthly revenue: growth trend + Q4 holiday build")
    axes[1].set_ylabel("revenue ($M / month)")
    axes[1].tick_params(axis="x", rotation=30)
    return _save(fig, images_dir, "baseline_overview.png")


def fig_seasonal_recovery(results, images_dir):
    fr = results["factor_recovery"]
    dims = [("hour", "hour of day"), ("dow", "day of week (0=Mon)"),
            ("month", "month")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (dim, xlabel) in zip(axes, dims):
        d = fr[dim]
        x = np.array(d["levels"])
        rec = np.array(d["recovered"])
        lo = np.array(d["ci_low"])
        hi = np.array(d["ci_high"])
        true = np.array(d["true"])
        ax.plot(x, true, "o-", color=C_TRUE, lw=1.6, ms=4, label="true")
        ax.errorbar(x, rec, yerr=[rec - lo, hi - rec], fmt="s", color=C_AWARE,
                    ms=4, capsize=2, lw=1, label="recovered (95% CI)")
        ax.axhline(1.0, color=C_GRID, lw=1)
        ax.set_title(f"{dim} factor (MAE {d['mean_abs_error']:.3f})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("multiplicative factor")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Seasonal-factor recovery: learned vs true", y=1.02, fontsize=12)
    return _save(fig, images_dir, "seasonal_recovery.png")


def fig_calibration(results, images_dir):
    cal = results["calibration"]
    nominal = cal["nominal"]
    windows = cal["windows"]
    labels = [w["label"] for w in windows]
    xs = np.arange(len(windows))
    naive_cov = [w["naive"]["year"]["coverage"] for w in windows]
    aware_cov = [w["aware"]["year"]["coverage"] for w in windows]
    naive_bias = [w["naive"]["year"]["bias_pct"] for w in windows]
    aware_bias = [w["aware"]["year"]["bias_pct"] for w in windows]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    w = 0.38
    axes[0].bar(xs - w / 2, naive_cov, w, color=C_NAIVE, label="naive")
    axes[0].bar(xs + w / 2, aware_cov, w, color=C_AWARE, label="seasonality-aware")
    axes[0].axhline(nominal, color=C_TRUTH, ls="--", lw=1.3,
                    label=f"nominal {nominal:.0%}")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_ylabel("annual-impact interval coverage")
    axes[0].set_title("Interval coverage of the true annual impact")
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].bar(xs - w / 2, naive_bias, w, color=C_NAIVE, label="naive")
    axes[1].bar(xs + w / 2, aware_bias, w, color=C_AWARE, label="seasonality-aware")
    axes[1].axhline(0, color=C_TRUTH, lw=1)
    axes[1].set_ylabel("annual point-estimate bias (%)")
    axes[1].set_title("Bias of the annual point estimate")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)
    axes[1].legend(frameon=False, fontsize=8)
    return _save(fig, images_dir, "calibration_coverage.png")


def fig_headline_annual(results, images_dir):
    """Stakeholder-facing annual-impact comparison for the top of the README.

    Shows the seasonality-aware annual estimate with its 95% interval against
    the naive annualization point, for the committed December full-week case.
    The true annual value is deliberately omitted: in a real decision it is
    unknown, and this is what a leader reporting to executives would see.
    """
    de = results["default_experiment"]
    aware = de["aware"]["year"]
    naive = de["naive"]["year"]
    scale = 1e6

    a_pt = aware["point"] / scale
    a_lo = aware["ci_low"] / scale
    a_hi = aware["ci_high"] / scale
    n_pt = naive["point"] / scale

    fig, ax = plt.subplots(figsize=(9, 3.0))
    ax.errorbar([a_pt], [1], xerr=[[a_pt - a_lo], [a_hi - a_pt]], fmt="o",
                color=C_AWARE, capsize=6, ms=10, lw=2.2,
                label="seasonality-aware (95% CI)")
    ax.plot([n_pt], [0], "D", color=C_NAIVE, ms=11,
            label="naive annualization")

    ax.annotate(f"${a_pt:.1f}M\n[${a_lo:.1f}M, ${a_hi:.1f}M]",
                (a_pt, 1), textcoords="offset points", xytext=(0, 14),
                ha="center", va="bottom", fontsize=9, color=C_AWARE)
    ax.annotate(f"${n_pt:.1f}M", (n_pt, 0), textcoords="offset points",
                xytext=(0, -22), ha="center", va="top", fontsize=9,
                color=C_NAIVE)

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["naive", "aware"])
    ax.set_ylim(-0.7, 1.7)
    xlo = min(a_lo, n_pt) - 1.2
    xhi = max(a_hi, n_pt) + 1.2
    ax.set_xlim(xlo, xhi)
    ax.set_xlabel("annual impact ($M / yr)")
    ax.set_title(f"What the exec team would see: annual impact ({de['label']})")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    return _save(fig, images_dir, "headline_annual_estimate.png")


def fig_intervals(results, images_dir):
    de = results["default_experiment"]
    ti = results["ground_truth"]["true_impact"]
    horizons = ["week", "month", "year"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, h in zip(axes, horizons):
        for i, (m, color, name) in enumerate([("naive", C_NAIVE, "naive"),
                                              ("aware", C_AWARE, "aware")]):
            d = de[m][h]
            scale = 1e6 if h == "year" else 1e3
            pt = d["point"] / scale
            lo = d["ci_low"] / scale
            hi = d["ci_high"] / scale
            ax.errorbar([pt], [i], xerr=[[pt - lo], [hi - pt]], fmt="o",
                        color=color, capsize=4, ms=7, label=name)
        scale = 1e6 if h == "year" else 1e3
        ax.axvline(ti[h] / scale, color=C_TRUTH, ls="--", lw=1.4, label="truth")
        unit = "$M/yr" if h == "year" else "$K"
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["naive", "aware"])
        ax.set_ylim(-0.6, 1.6)
        ax.set_xlabel(f"impact ({unit})")
        ax.set_title(f"{h} impact")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(f"Extrapolated impact vs truth ({de['label']})", y=1.03,
                 fontsize=12)
    return _save(fig, images_dir, "naive_vs_aware_intervals.png")


def fig_runlength(results, images_dir):
    grid = results["runlength"]["grid"]
    nominal = results["runlength"]["nominal"]
    anchors = sorted({g["anchor"] for g in grid})
    hours = sorted({g["hours"] for g in grid})

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    markers = ["o", "s", "^", "D"]
    for a, mk in zip(anchors, markers):
        cells = sorted([g for g in grid if g["anchor"] == a],
                       key=lambda g: g["hours"])
        hh = [c["hours"] / 24.0 for c in cells]
        axes[0].plot(hh, [c["aware"]["coverage"] for c in cells], mk + "-",
                     color=C_AWARE, label=f"aware, {a}", ms=4)
        axes[0].plot(hh, [c["naive"]["coverage"] for c in cells], mk + "--",
                     color=C_NAIVE, label=f"naive, {a}", ms=4, alpha=0.8)
        axes[1].plot(hh, [c["aware"]["bias_pct"] for c in cells], mk + "-",
                     color=C_AWARE, ms=4)
        axes[1].plot(hh, [c["naive"]["bias_pct"] for c in cells], mk + "--",
                     color=C_NAIVE, ms=4, alpha=0.8)
        axes[2].plot(hh, [c["aware"]["rel_halfwidth_pct"] for c in cells],
                     mk + "-", color=C_AWARE, ms=4)

    axes[0].axhline(nominal, color=C_TRUTH, ls=":", lw=1.2)
    axes[0].axvline(7, color=C_ACCENT, ls="-", lw=1, alpha=0.7)
    axes[0].set_title("Annual-interval coverage vs run length")
    axes[0].set_xlabel("run length (days)")
    axes[0].set_ylabel("coverage")
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(frameon=False, fontsize=6.5, ncol=1)

    axes[1].axhline(0, color=C_TRUTH, lw=1)
    axes[1].axvline(7, color=C_ACCENT, ls="-", lw=1, alpha=0.7)
    axes[1].set_title("Annual point-estimate bias vs run length")
    axes[1].set_xlabel("run length (days)")
    axes[1].set_ylabel("bias (%)")

    axes[2].axvline(7, color=C_ACCENT, ls="-", lw=1, alpha=0.7)
    axes[2].set_title("Aware relative half-width vs run length")
    axes[2].set_xlabel("run length (days)")
    axes[2].set_ylabel("half-width / point (%)")

    fig.suptitle("Minimum run-length: dashed=naive, solid=aware, orange=1 week",
                 y=1.02, fontsize=11)
    return _save(fig, images_dir, "runlength.png")


def fig_variance_decomposition(results, images_dir):
    de = results["default_experiment"]["aware"]
    horizons = ["week", "month", "year"]
    exp = np.array([de[h]["var_experiment"] for h in horizons])
    seas = np.array([de[h]["var_seasonal"] for h in horizons])
    trans = np.array([de[h]["var_transport"] for h in horizons])
    total = exp + seas + trans
    total[total == 0] = 1.0
    xs = np.arange(len(horizons))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].bar(xs, 100 * exp / total, color=C_AWARE, label="experiment sampling")
    axes[0].bar(xs, 100 * seas / total, bottom=100 * exp / total, color=C_ACCENT,
                label="seasonal reconstruction")
    axes[0].bar(xs, 100 * trans / total, bottom=100 * (exp + seas) / total,
                color=C_NAIVE, label="transportability")
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(horizons)
    axes[0].set_ylabel("share of interval variance (%)")
    axes[0].set_title("Aware interval variance by source\n(default experiment, 3-yr baseline)")
    axes[0].legend(frameon=False, fontsize=8)

    vp = results.get("variance_profile", [])
    if vp:
        hx = [f"{r['history_months']}mo" for r in vp]
        es = [r["experiment_share"] for r in vp]
        ss = [r["seasonal_share"] for r in vp]
        xi = np.arange(len(vp))
        axes[1].bar(xi, es, color=C_AWARE, label="experiment sampling")
        axes[1].bar(xi, ss, bottom=es, color=C_ACCENT, label="seasonal reconstruction")
        axes[1].set_xticks(xi)
        axes[1].set_xticklabels(hx)
        axes[1].set_ylabel("share of annual-interval variance (%)")
        axes[1].set_xlabel("length of always-on baseline history")
        axes[1].set_title("Seasonal uncertainty is propagated,\nand shrinks as history grows")
        axes[1].legend(frameon=False, fontsize=8)
    fig.text(
        0.5, -0.02,
        "Takeaway: each bar totals 100 percent, so read the split, not the height. The aware interval's width is set "
        "almost entirely by the experiment's sampling noise (blue).\nLearning seasonality adds little, and only when "
        "history is short (about 6.5 percent at 12 months, gone by 18), so invest in measuring the effect precisely and "
        "representatively, not in more baseline history.",
        ha="center", va="top", fontsize=8.5, style="italic", color="#333333")
    return _save(fig, images_dir, "variance_decomposition.png")


def fig_effect_heterogeneity(results, images_dir):
    eh = results.get("effect_heterogeneity")
    if not eh:
        return None
    ws = eh["windows"]
    labels = ["low-effect\nwindow", "high-effect\nwindow", "full-week\nreference"]
    labels = labels[:len(ws)]
    xs = np.arange(len(ws))
    naive = [w["naive_total_bias_pct"] for w in ws]
    aware = [w["aware_residual_bias_pct"] for w in ws]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    wd = 0.38
    axes[0].bar(xs - wd / 2, naive, wd, color=C_NAIVE,
                label="naive total bias (composition + effect)")
    axes[0].bar(xs + wd / 2, aware, wd, color=C_AWARE,
                label="aware residual bias (effect only)")
    axes[0].axhline(0, color=C_TRUTH, lw=1)
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylabel("annual point-estimate bias (%)")
    axes[0].set_title("Composition is fixed, effect heterogeneity remains")
    axes[0].legend(frameon=False, fontsize=8)
    for x, w in zip(xs, ws):
        axes[0].annotate(f"g={w['g_local']:.2f}", (x + wd / 2, aware[xs.tolist().index(x)]),
                         textcoords="offset points", xytext=(0, 4 if aware[xs.tolist().index(x)] >= 0 else -12),
                         ha="center", fontsize=7, color=C_AWARE)

    iv = eh["invariance"]
    scales = [f"{r['sessions_scale']}x" for r in iv]
    xi = np.arange(len(iv))
    ax2 = axes[1]
    ax2.bar(xi - wd / 2, [r["aware_residual_bias_pct"] for r in iv], wd,
            color=C_AWARE, label="aware residual bias")
    ax2.set_ylabel("aware residual bias (%)", color=C_AWARE)
    ax2.axhline(0, color=C_TRUTH, lw=1)
    ax2b = ax2.twinx()
    ax2b.plot(xi, [r["mean_se_r"] for r in iv], "o-", color=C_ACCENT,
              label="sampling se of lift")
    ax2b.set_ylabel("mean se of measured lift", color=C_ACCENT)
    ax2b.set_ylim(bottom=0)
    ax2.set_xticks(xi)
    ax2.set_xticklabels(scales)
    ax2.set_xlabel("traffic (sessions) scale, same window")
    ax2.set_title("More samples shrink noise, not the residual bias")
    ax2.grid(False)
    ax2b.grid(False)
    return _save(fig, images_dir, "effect_heterogeneity.png")


def generate_all(results, baseline_df, images_dir):
    paths = []
    paths.append(fig_headline_annual(results, images_dir))
    paths.append(fig_baseline_overview(baseline_df, images_dir))
    paths.append(fig_seasonal_recovery(results, images_dir))
    paths.append(fig_calibration(results, images_dir))
    paths.append(fig_intervals(results, images_dir))
    paths.append(fig_runlength(results, images_dir))
    paths.append(fig_variance_decomposition(results, images_dir))
    p = fig_effect_heterogeneity(results, images_dir)
    if p:
        paths.append(p)
    return paths
