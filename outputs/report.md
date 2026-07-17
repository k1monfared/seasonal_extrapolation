# Seasonal extrapolation demo report

> **This is a portfolio demonstration built on synthetic data.**

Fixed seed: `42`. Target horizon year: `2026`. True relative lift: `5.0%`.

## Headline (default experiment)

Experiment window: **Full week in December (peak season)** (168 h starting 2026-12-07T00:00:00).

- Measured relative lift: `0.0494` (true `0.0500`, se `0.0035`).
- **Seasonality-aware annual impact: $8.14M** [$7.02M, $9.25M].
- Naive annual impact: $11.69M [$10.09M, $13.29M].
- True annual impact: **$8.20M**.

## Extrapolation vs truth

| horizon | truth | naive point | naive 95% CI | aware point | aware 95% CI |
|---|--:|--:|:--:|--:|:--:|
| week | $157.3K | $224.2K | [$193.5K, $255.0K] | $156.0K | [$134.6K, $177.4K] |
| month | $683.5K | $974.2K | [$840.6K, $1.11M] | $678.0K | [$585.0K, $771.0K] |
| year | $8.20M | $11.69M | [$10.09M, $13.29M] | $8.14M | [$7.02M, $9.25M] |

## Transportability check

No significant heterogeneity on the covered cycle. Daypart partition is consistent with a single lift (p=0.952). Weekend partition is consistent with a single lift (p=0.205). Cross-month effect stability is assumed and cannot be tested from a within-month window.

Partition by daypart (chi2=0.10, dof=2, p=0.952):

| sub-period | r_hat | se | hours |
|---|--:|--:|--:|
| daytime | 0.0495 | 0.0047 | 84 |
| evening | 0.0502 | 0.0059 | 42 |
| overnight | 0.0464 | 0.0107 | 42 |

Partition by weekend (chi2=1.60, dof=1, p=0.205):

| sub-period | r_hat | se | hours |
|---|--:|--:|--:|
| weekday | 0.0470 | 0.0039 | 120 |
| weekend | 0.0576 | 0.0073 | 48 |

## Seasonal-factor recovery

Recovered annual growth: `0.0808` (true `0.0800`).

| dimension | mean abs error | max abs error | 95% CI coverage of true |
|---|--:|--:|--:|
| hour | 0.0012 | 0.0040 | 88% |
| dow | 0.0004 | 0.0006 | 100% |
| month | 0.0008 | 0.0016 | 100% |

## Calibration (Monte Carlo)

Nominal level: **95%**. Replications per window: `800`. Coverage is of the TRUE annual impact across replications.

| window | naive cov | naive bias | aware cov | aware bias | naive width | aware width |
|---|--:|--:|--:|--:|--:|--:|
| 15h Tue evening (spec example) | 95% | -8.6% | 97% | +3.8% | $9.00M | $10.20M |
| 12h Tue daytime peak | 90% | +26.3% | 96% | -0.5% | $12.07M | $9.49M |
| 1 day (Sat, low traffic) | 83% | -20.0% | 95% | +9.6% | $6.49M | $8.87M |
| 3.5 days (partial week) | 88% | -9.6% | 93% | -4.2% | $3.93M | $4.16M |
| 7 days (full week, March) | 77% | -10.3% | 95% | -0.1% | $2.71M | $3.02M |
| 7 days (full week, December peak) | 1% | +43.7% | 95% | +0.0% | $3.21M | $2.23M |
| 14 days (two weeks, March) | 60% | -10.3% | 95% | -0.2% | $1.92M | $2.13M |

## Minimum run-length recommendation

**Recommended minimum run length: 168 hours (7.0 days).**

Criteria: cover at least one full weekly cycle, aware coverage >= 92%, aware relative half-width <= 25% of the point estimate, across every start phase tested.

| run length (days) | covers full week | aware min cov | aware max |bias| | aware max half-width | naive min cov | naive max |bias| |
|--:|:--:|--:|--:|--:|--:|--:|
| 0.5 | no | 95% | 12.4% | 90% | 42% | 51.3% |
| 1.0 | no | 93% | 9.2% | 54% | 81% | 20.2% |
| 2.0 | no | 92% | 9.8% | 39% | 67% | 21.7% |
| 3.5 | no | 94% | 3.2% | 28% | 57% | 20.0% |
| 5.0 | no | 94% | 3.5% | 22% | 79% | 11.1% |
| 7.0 | yes | 94% | 0.8% | 18% | 76% | 10.7% |
| 10.0 | yes | 94% | 1.7% | 16% | 63% | 11.6% |
| 14.0 | yes | 94% | 0.7% | 13% | 56% | 10.6% |

## Residual bias: effect heterogeneity across the cycle

The aware method corrects the baseline composition (where the volume is) but still assumes one relative lift for the whole cycle. When a window sits in a low- or high-effect period, the annual estimate stays biased by about (local effect / annual effect - 1), even after the composition correction.

| window | g_local | c_local | aware residual bias | naive total bias |
|---|--:|--:|--:|--:|
| low-effect window (weekday overnight to morning) | 0.866 | 0.718 | -12.2% | -37.5% |
| high-effect window (Saturday evening) | 1.182 | 0.695 | +20.0% | -17.2% |
| full-week reference (Mon to Sun) | 1.000 | 0.899 | +0.4% | -10.1% |

g_local is the true local effect relative to the annual mean (residual effect-heterogeneity error the aware method keeps). c_local is the window baseline rate relative to the annual rate (composition error the aware method removes). They are two different errors.

Same window, more traffic (residual bias does not shrink with samples, only the sampling error does):

| sessions scale | mean se of lift | aware residual bias |
|--:|--:|--:|
| 1x | 0.0201 | +18.4% |
| 4x | 0.0100 | +18.4% |

## Business impact

A trustworthy annual number needs about **7.0 days** of experiment, not a full year: roughly **358 days (98%) of calendar time saved** per decision, while still delivering a C-level-ready annual impact with a calibrated interval.
