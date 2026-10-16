"""Plot colours that follow the system's light or dark theme (no Qt here: the window
says which one is active).

Light uses FORSTER's figure colours (`forster.plotting`: Okabe-Ito blue for the model,
near-black data); dark swaps in lighter tints of the same hues so the model, the data
and the drag handles keep their contrast on a dark background.
"""

from __future__ import annotations

from dataclasses import dataclass

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from forster.plotting import MODEL_COLOR


@dataclass(frozen=True)
class Palette:
    background: str
    foreground: str
    muted: str
    grid: str
    model: str
    data: str
    handle: str  # the draggable starting profile (Okabe-Ito vermilion / orange)
    warning: str


LIGHT = Palette(
    background="#ffffff",
    foreground="#1a1a1a",
    muted="#8c8c8c",
    grid="#e6e6e6",
    model=MODEL_COLOR,
    data="#1a1a1a",
    handle="#D55E00",
    warning="#8a5a00",
)
DARK = Palette(
    background="#1f1f1f",
    foreground="#e8e8e8",
    muted="#9a9a9a",
    grid="#3a3a3a",
    model="#56B4E9",
    data="#e8e8e8",
    handle="#E69F00",
    warning="#f0c060",
)


def palette(dark: bool) -> Palette:
    return DARK if dark else LIGHT


def style_axes(figure: Figure, axes: list[Axes], p: Palette) -> None:
    """Background, spines, ticks, labels and legend text in the palette's colours."""
    figure.set_facecolor(p.background)
    for ax in axes:
        ax.set_facecolor(p.background)
        for spine in ax.spines.values():
            spine.set_color(p.muted)
        ax.tick_params(colors=p.foreground, which="both")
        ax.xaxis.label.set_color(p.foreground)
        ax.yaxis.label.set_color(p.foreground)
        ax.title.set_color(p.foreground)
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_color(p.foreground)
