"""Figures of fit results."""

from __future__ import annotations

from typing import TYPE_CHECKING

from forster.diffusivity import label as law_label

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes

    from forster.model import FitResult
    from forster.rescale import TimeDistribution
    from forster.sampling import ICPosterior

# Okabe-Ito blue for the model, black for data, grey for context.
MODEL_COLOR = "#0072B2"
DATA_COLOR = "0.1"
CONTEXT_COLOR = "0.55"
SIGMA = "\N{GREEK SMALL LETTER SIGMA}"
# One law a colour (Okabe-Ito, the model blue first) and a line style, so the curves
# stay apart in greyscale; both repeat past their length.
LAW_COLORS = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9")
LAW_STYLES = ("-", (0, (6, 2)), "-.", (0, (1.5, 1.5)))


def _new_axes(figsize: tuple[float, float]) -> Axes:
    import matplotlib.pyplot as plt

    _, ax = plt.subplots(figsize=figsize, layout="constrained")
    return ax


def _common_name(names: Sequence[str]) -> str:
    """What the names start with (``AZ18`` of ``AZ18_dipra`` and ``AZ18_refit``),
    trimmed of trailing separators; the first name if they share nothing."""
    prefix = names[0]
    for name in names[1:]:
        while not name.startswith(prefix):
            prefix = prefix[:-1]
    return prefix.rstrip("_ -") or names[0]


def _plot_measured(ax: Axes, result: FitResult, scale: float) -> None:
    """The measured points: open black circles, grey outside the model domain."""
    x = result.model_x_um
    inside = (result.data_x_um >= x[0]) & (result.data_x_um <= x[-1])
    for keep, color, label in (
        (inside, DATA_COLOR, "Measured"),
        (~inside, CONTEXT_COLOR, "Outside the model domain"),
    ):
        if keep.any():
            ax.plot(
                result.data_x_um[keep],
                result.data_fo[keep] * scale,
                linestyle="none",
                marker="o",
                markersize=4,
                markerfacecolor="white",
                markeredgecolor=color,
                label=label,
            )


def plot_fit(result: FitResult, ax: Axes | None = None) -> Axes:
    """Measured profile, initial condition, best fit and the 1-sigma range.

    Fo is shown in mol%; an element fit (`forster.elements.fit_element`) in the units
    of its data.

    The kriged profile is drawn only for the legacy uncertainty method, the one
    that fits to it. Without ``ax`` a new pyplot figure is made; pass an ``ax``
    from ``matplotlib.figure.Figure`` to draw without pyplot (scripts, threads).
    """
    if ax is None:
        ax = _new_axes((7, 4))
    _, low, high = result.indices
    days = result.time_days
    low_days, high_days = days + result.time_minus_days, days + result.time_plus_days
    x = result.model_x_um
    is_fo = result.element == "Fe-Mg"
    scale = 100 if is_fo else 1
    if result.uncertainty == "legacy":
        kriged = result.interp_fo * 100
        sigma = result.interp_fo_sigma * 100
        ax.fill_between(
            result.interp_x_um,
            kriged - sigma,
            kriged + sigma,
            color=CONTEXT_COLOR,
            alpha=0.2,
            linewidth=0,
            label=f"Kriged ± {SIGMA} (fitted)",
        )
    ax.plot(x, result.initial * scale, color=CONTEXT_COLOR, linestyle="--", lw=1.2, label="Initial")
    if result.indices2 is not None:
        low2_days, high2_days = result.time_range_days(sigmas=2)
        ax.fill_between(
            x,
            result.curve(result.indices2[0]) * scale,
            result.curve(result.indices2[1]) * scale,
            color=MODEL_COLOR,
            alpha=0.12,
            linewidth=0,
            label=f"2{SIGMA} range, {low2_days:.1f}-{high2_days:.1f} d",
        )
    ax.fill_between(
        x,
        result.curve(low) * scale,
        result.curve(high) * scale,
        color=MODEL_COLOR,
        alpha=0.25,
        linewidth=0,
        label=f"1{SIGMA} range, {low_days:.1f}-{high_days:.1f} d",
    )
    ax.plot(x, result.best_fit * scale, color=MODEL_COLOR, lw=2.2, label=f"Best fit, {days:.1f} d")
    _plot_measured(ax, result, scale)
    ax.set_xlabel("Distance from the crystal edge (µm)")
    ax.set_ylabel("Fo (mol%)" if is_fo else result.element)
    title = result.name if is_fo else f"{result.name} {result.element}"
    ax.set_title(
        f"{title}\n{days:.1f} days (1{SIGMA} {low_days:.1f}-{high_days:.1f}), "
        f"{result.uncertainty} uncertainty",
        fontsize="medium",
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize="small")
    return ax


def plot_law_comparison(results: Sequence[FitResult], ax: Axes | None = None) -> Axes:
    """One profile fitted with several diffusivity laws: the measured points and the
    initial condition (from the first result) and each law's best fit.

    Each legend entry is the law's label (`forster.diffusivity.label`), its best fit and
    its 1-sigma range from the profile alone (`FitResult.time_range_days`); the title
    is the results' common name. Units, scaling and ``ax`` as in `plot_fit`.
    """
    if not results:
        raise ValueError("plot_law_comparison needs at least one fit")
    if ax is None:
        ax = _new_axes((7, 5))
    first = results[0]
    is_fo = first.element == "Fe-Mg"
    scale = 100 if is_fo else 1
    ax.plot(
        first.model_x_um,
        first.initial * scale,
        color=CONTEXT_COLOR,
        linestyle="--",
        lw=1.2,
        label="Initial",
    )
    for i, result in enumerate(results):
        low, high = result.time_range_days()
        ax.plot(
            result.model_x_um,
            result.best_fit * scale,
            color=LAW_COLORS[i % len(LAW_COLORS)],
            linestyle=LAW_STYLES[i % len(LAW_STYLES)],
            lw=1.8,
            label=f"{law_label(result.inputs['diffusivity'])}: {result.time_days:.1f} d "
            f"({low:.1f}-{high:.1f})",
        )
    _plot_measured(ax, first, scale)
    ax.set_xlabel("Distance from the crystal edge (µm)")
    ax.set_ylabel("Fo (mol%)" if is_fo else first.element)
    days = [r.time_days for r in results]
    name = _common_name([r.name for r in results])
    ax.set_title(
        f"{name}\nbest fits {min(days):.1f}-{max(days):.1f} days across {len(results)} "
        f"laws; 1{SIGMA} ranges from the profile alone",
        fontsize="medium",
    )
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.14), frameon=False, fontsize="small")
    return ax


def plot_law_times(results: Sequence[FitResult], ax: Axes | None = None) -> Axes:
    """The best-fit time under each law, one row a law (the first at the top) on a log
    time axis: the point, its 1-sigma range from the profile alone (thick bar) and, for
    a law with a covariance, the wider range with its calibration uncertainty
    (``FitResult.time_range_days("law")``, thin bar). Colours as `plot_law_comparison`,
    so the two read together; ``ax`` as in `plot_fit`.
    """
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    if not results:
        raise ValueError("plot_law_times needs at least one fit")
    if ax is None:
        ax = _new_axes((7, 2.6))
    n = len(results)
    for i, result in enumerate(results):
        y = n - 1 - i
        color = LAW_COLORS[i % len(LAW_COLORS)]
        if result.law_sigma_log10 is not None:
            ax.plot(result.time_range_days("law"), [y, y], color=color, lw=1.2, gid="law_range")
        else:
            ax.annotate(  # under the point, so it stays inside the axes
                "no law covariance",
                (result.time_days, y),
                xytext=(0, -7),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize="small",
                color=CONTEXT_COLOR,
            )
        ax.plot(result.time_range_days(), [y, y], color=color, lw=5, alpha=0.45, gid="fit_range")
        ax.plot(
            result.time_days,
            y,
            marker="o",
            markersize=7,
            markerfacecolor=color,
            markeredgecolor="white",
            linestyle="none",
            gid="best_fit",
        )
    ax.set_yticks(range(n), [law_label(r.inputs["diffusivity"]) for r in reversed(results)])
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xscale("log")
    ax.margins(x=0.1)  # room at the ends for the bars and the note under a point
    # Label 1, 2, 3 and 5 of each decade: a few days rarely spans a power of ten.
    ax.xaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 3.0, 5.0)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis="x", which="major", color="0.9", lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Time (days, log scale)")
    ax.set_title(f"Best-fit time under each law, with 1{SIGMA} ranges", fontsize="medium")
    ax.legend(
        handles=[
            Line2D([], [], color=CONTEXT_COLOR, lw=5, alpha=0.45, label="Profile alone"),
            Line2D([], [], color=CONTEXT_COLOR, lw=1.2, label="With the law's calibration"),
        ],
        loc="upper left",
        bbox_to_anchor=(0, -0.3),
        ncols=2,
        frameon=False,
        fontsize="small",
    )
    return ax


def _per_log10_day(days, density):
    """Density per log10 day, scaled to a peak of 1 (time steps from day 0 are dropped)."""
    import numpy as np

    keep = days > 0
    d = np.asarray(density)[keep] * days[keep]
    return days[keep], d / d.max()


def plot_time_distributions(
    result: FitResult,
    posterior: ICPosterior | None = None,
    mc: TimeDistribution | None = None,
    ax: Axes | None = None,
) -> Axes:
    """The distributions of time a run has, on a log time axis, each scaled to a peak of
    1: the fit at the nominal initial condition (its own posterior), with the
    initial-condition uncertainty (`forster.sampling`), and with the conditions and the
    law as well (`forster.rescale`). Each legend entry gives the median and central 68%."""
    import numpy as np

    from forster.rescale import fit_time_density

    if ax is None:
        ax = _new_axes((6.5, 3.6))

    def stats(days, density):
        cdf = np.cumsum(density)
        cdf = cdf / cdf[-1]
        lo, mid, hi = np.interp([0.16, 0.5, 0.84], cdf, days)
        return f"{mid:.3g} d ({lo:.3g}-{hi:.3g})"

    spans = []  # where each distribution is above 0.1% of its peak, for the x limits

    def span(x, y):
        x = x[y > 1e-3]
        if len(x):
            spans.append((x.min(), x.max()))

    days, density = fit_time_density(result)
    x, y = _per_log10_day(days, density)
    span(x, y)
    ax.plot(x, y, color=CONTEXT_COLOR, lw=1.6, label=f"Fit alone: {stats(days, density)}")
    if posterior is not None:
        x, y = _per_log10_day(posterior.days, posterior.density)
        span(x, y)
        ax.plot(
            x,
            y,
            color=MODEL_COLOR,
            lw=1.8,
            label=f"With the initial condition: {stats(posterior.days, posterior.density)}",
        )
    if mc is not None and len(mc.days):
        log_days = np.log10(mc.days)
        counts, edges = np.histogram(log_days, bins=60)
        centres = 10 ** (0.5 * (edges[1:] + edges[:-1]))
        lo, mid, hi = (mc.percentiles_days[p] for p in (16, 50, 84))
        spans.append((float(np.percentile(mc.days, 0.1)), float(np.percentile(mc.days, 99.9))))
        ax.fill_between(
            centres,
            counts / counts.max(),
            step="mid",
            color=LAW_COLORS[1],
            alpha=0.35,
            lw=0,
            label=f"With conditions and law: {mid:.3g} d ({lo:.3g}-{hi:.3g})",
        )
    ax.set_xscale("log")
    ax.set_xlim(min(a for a, _ in spans) / 1.2, max(b for _, b in spans) * 1.2)
    ax.set_ylim(0, 1.5)  # headroom: the legend sits above the peaks
    ax.set_xlabel("Time (days, log scale)")
    ax.set_ylabel("Relative probability")
    ax.set_yticks([0, 0.5, 1])
    ax.set_title(f"{result.name}: time with median and central 68%")
    ax.legend(loc="upper left", fontsize="small", frameon=False)
    return ax
