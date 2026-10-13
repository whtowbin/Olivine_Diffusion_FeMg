"""Figures of fit results."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from forster.model import FitResult

# Okabe-Ito blue for the model, black for data, grey for context.
MODEL_COLOR = "#0072B2"
DATA_COLOR = "0.1"
CONTEXT_COLOR = "0.55"
SIGMA = "\N{GREEK SMALL LETTER SIGMA}"


def plot_fit(result: FitResult, ax: Axes | None = None) -> Axes:
    """Measured profile, initial condition, best fit and the 1-sigma range.

    Fo is shown in mol%; an element fit (`forster.elements.fit_element`) in the units
    of its data.

    The kriged profile is drawn only for the legacy uncertainty method, the one
    that fits to it. Without ``ax`` a new pyplot figure is made; pass an ``ax``
    from ``matplotlib.figure.Figure`` to draw without pyplot (scripts, threads).
    """
    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots(figsize=(7, 4), layout="constrained")
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
    ax.fill_between(
        x,
        result.curves[low] * scale,
        result.curves[high] * scale,
        color=MODEL_COLOR,
        alpha=0.25,
        linewidth=0,
        label=f"1{SIGMA} range, {low_days:.1f}-{high_days:.1f} d",
    )
    ax.plot(x, result.best_fit * scale, color=MODEL_COLOR, lw=2.2, label=f"Best fit, {days:.1f} d")
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
