"""Generate outputs/report.md, a numbers-first digest of a single demo run.

Everything here is formatted from the results dict produced by
pipeline.run_analysis, so the report always matches the committed run.
"""

from __future__ import annotations

import os


def _money(x, scale="auto"):
    ax = abs(x)
    if scale == "M" or (scale == "auto" and ax >= 1e6):
        return f"${x / 1e6:,.2f}M"
    if scale == "K" or (scale == "auto" and ax >= 1e3):
        return f"${x / 1e3:,.1f}K"
    return f"${x:,.0f}"


def _pct(x):
    return f"{x:+.1f}%"


def build_report(results: dict) -> str:
    cfg = results["config"]
    de = results["default_experiment"]
    ti = results["ground_truth"]["true_impact"]
    fr = results["factor_recovery"]
    cal = results["calibration"]
    rls = results["runlength"]

    L = []
    L.append("# Seasonal extrapolation demo report")
    L.append("")
    L.append("> **This is a portfolio demonstration built on synthetic data.**")
    L.append("")
    L.append(f"Fixed seed: `{cfg['seed']}`. Target horizon year: "
             f"`{cfg['target_year']}`. True relative lift: "
             f"`{cfg['true_relative_lift']:.1%}`.")
    L.append("")

    # --- headline ----------------------------------------------------------
    aw = de["aware"]["year"]
    nv = de["naive"]["year"]
    L.append("## Headline (default experiment)")
    L.append("")
    L.append(f"Experiment window: **{de['label']}** "
             f"({de['hours']} h starting {de['start']}).")
    L.append("")
    L.append(f"- Measured relative lift: `{de['r_hat']:.4f}` "
             f"(true `{de['true_relative_lift']:.4f}`, "
             f"se `{de['se_r_hat']:.4f}`).")
    L.append(f"- **Seasonality-aware annual impact: {_money(aw['point'])}** "
             f"[{_money(aw['ci_low'])}, {_money(aw['ci_high'])}].")
    L.append(f"- Naive annual impact: {_money(nv['point'])} "
             f"[{_money(nv['ci_low'])}, {_money(nv['ci_high'])}].")
    L.append(f"- True annual impact: **{_money(ti['year'])}**.")
    L.append("")

    # --- extrapolation table ----------------------------------------------
    L.append("## Extrapolation vs truth")
    L.append("")
    L.append("| horizon | truth | naive point | naive 95% CI | aware point | aware 95% CI |")
    L.append("|---|--:|--:|:--:|--:|:--:|")
    for h in ("week", "month", "year"):
        n = de["naive"][h]
        a = de["aware"][h]
        L.append(f"| {h} | {_money(ti[h])} | {_money(n['point'])} | "
                 f"[{_money(n['ci_low'])}, {_money(n['ci_high'])}] | "
                 f"{_money(a['point'])} | "
                 f"[{_money(a['ci_low'])}, {_money(a['ci_high'])}] |")
    L.append("")

    # --- transportability --------------------------------------------------
    tr = de["transportability"]
    L.append("## Transportability check")
    L.append("")
    L.append(tr["flag"])
    L.append("")
    for part in ("daypart", "weekend"):
        p = tr.get(part, {})
        if not p.get("testable"):
            continue
        L.append(f"Partition by {part} "
                 f"(chi2={p['chi2']:.2f}, dof={p['dof']}, p={p['p_value']:.3f}):")
        L.append("")
        L.append("| sub-period | r_hat | se | hours |")
        L.append("|---|--:|--:|--:|")
        for k, g in p["groups"].items():
            L.append(f"| {k} | {g['r_hat']:.4f} | {g['se']:.4f} | {g['n_hours']} |")
        L.append("")

    # --- factor recovery ---------------------------------------------------
    L.append("## Seasonal-factor recovery")
    L.append("")
    L.append(f"Recovered annual growth: "
             f"`{fr['trend_annual_growth_recovered']:.4f}` "
             f"(true `{fr['trend_annual_growth_true']:.4f}`).")
    L.append("")
    L.append("| dimension | mean abs error | max abs error | 95% CI coverage of true |")
    L.append("|---|--:|--:|--:|")
    for dim in ("hour", "dow", "month"):
        d = fr[dim]
        L.append(f"| {dim} | {d['mean_abs_error']:.4f} | "
                 f"{d['max_abs_error']:.4f} | {d['ci_coverage']:.0%} |")
    L.append("")

    # --- calibration -------------------------------------------------------
    L.append("## Calibration (Monte Carlo)")
    L.append("")
    L.append(f"Nominal level: **{cal['nominal']:.0%}**. Replications per window: "
             f"`{cfg['calibration']['n_reps']}`. Coverage is of the TRUE annual "
             "impact across replications.")
    L.append("")
    L.append("| window | naive cov | naive bias | aware cov | aware bias | naive width | aware width |")
    L.append("|---|--:|--:|--:|--:|--:|--:|")
    for w in cal["windows"]:
        n = w["naive"]["year"]
        a = w["aware"]["year"]
        L.append(f"| {w['label']} | {n['coverage']:.0%} | {_pct(n['bias_pct'])} | "
                 f"{a['coverage']:.0%} | {_pct(a['bias_pct'])} | "
                 f"{_money(n['mean_ci_width'])} | {_money(a['mean_ci_width'])} |")
    L.append("")

    # --- run length --------------------------------------------------------
    rec = rls["recommendation"]
    L.append("## Minimum run-length recommendation")
    L.append("")
    if rec["recommended_hours"] is not None:
        L.append(f"**Recommended minimum run length: "
                 f"{rec['recommended_hours']} hours "
                 f"({rec['recommended_days']} days).**")
    else:
        L.append("No tested run length met all criteria; extend the grid.")
    L.append("")
    L.append("Criteria: cover at least one full weekly cycle, aware coverage "
             f">= {rec['criteria']['aware_coverage_at_least']:.0%}, aware "
             f"relative half-width <= "
             f"{rec['criteria']['aware_rel_halfwidth_at_most_pct']:.0f}% of the "
             "point estimate, across every start phase tested.")
    L.append("")
    L.append("| run length (days) | covers full week | aware min cov | aware max |bias| | aware max half-width | naive min cov | naive max |bias| |")
    L.append("|--:|:--:|--:|--:|--:|--:|--:|")
    for h in sorted(rec["per_hours"]):
        s = rec["per_hours"][h]
        L.append(f"| {h/24:.1f} | {'yes' if s['covers_full_week'] else 'no'} | "
                 f"{s['aware_min_coverage']:.0%} | "
                 f"{s['aware_max_abs_bias_pct']:.1f}% | "
                 f"{s['aware_max_rel_halfwidth_pct']:.0f}% | "
                 f"{s['naive_min_coverage']:.0%} | "
                 f"{s['naive_max_abs_bias_pct']:.1f}% |")
    L.append("")

    # --- effect heterogeneity ---------------------------------------------
    eh = results.get("effect_heterogeneity")
    if eh:
        L.append("## Residual bias: effect heterogeneity across the cycle")
        L.append("")
        L.append("The aware method corrects the baseline composition (where the "
                 "volume is) but still assumes one relative lift for the whole "
                 "cycle. When a window sits in a low- or high-effect period, the "
                 "annual estimate stays biased by about (local effect / annual "
                 "effect - 1), even after the composition correction.")
        L.append("")
        L.append("| window | g_local | c_local | aware residual bias | naive total bias |")
        L.append("|---|--:|--:|--:|--:|")
        for w in eh["windows"]:
            L.append(f"| {w['label']} | {w['g_local']:.3f} | {w['c_local']:.3f} | "
                     f"{_pct(w['aware_residual_bias_pct'])} | "
                     f"{_pct(w['naive_total_bias_pct'])} |")
        L.append("")
        L.append("g_local is the true local effect relative to the annual mean "
                 "(residual effect-heterogeneity error the aware method keeps). "
                 "c_local is the window baseline rate relative to the annual rate "
                 "(composition error the aware method removes). They are two "
                 "different errors.")
        L.append("")
        L.append("Same window, more traffic (residual bias does not shrink with "
                 "samples, only the sampling error does):")
        L.append("")
        L.append("| sessions scale | mean se of lift | aware residual bias |")
        L.append("|--:|--:|--:|")
        for iv in eh["invariance"]:
            L.append(f"| {iv['sessions_scale']}x | {iv['mean_se_r']:.4f} | "
                     f"{_pct(iv['aware_residual_bias_pct'])} |")
        L.append("")

    # --- business framing --------------------------------------------------
    L.append("## Business impact")
    L.append("")
    days = rec["recommended_days"]
    if days is not None:
        saved = 365 - days
        L.append(f"A trustworthy annual number needs about **{days:.1f} days** "
                 f"of experiment, not a full year: roughly **{saved:.0f} days "
                 f"({saved/365:.0%}) of calendar time saved** per decision, "
                 "while still delivering a C-level-ready annual impact with a "
                 "calibrated interval.")
    L.append("")
    return "\n".join(L)


def save_report(results: dict, outputs_dir: str):
    os.makedirs(outputs_dir, exist_ok=True)
    text = build_report(results)
    with open(os.path.join(outputs_dir, "report.md"), "w") as f:
        f.write(text)
    return text
