"""Plot colours that follow the system's light or dark theme (no Qt here: the window
says which one is active).

Light uses FORSTER's figure colours (`forster.plotting`: Okabe-Ito blue for the model,
near-black data); dark swaps in lighter tints of the same hues so the model, the data
and the drag handles keep their contrast on a dark background. The views drawn by
`forster.plotting` (time distribution, law comparison) are recoloured after drawing
(`restyle_figure`).
"""

from __future__ import annotations

from dataclasses import dataclass

from matplotlib.axes import Axes
from matplotlib.collections import Collection
from matplotlib.colors import to_hex, to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text

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


def _swap(color, p: Palette, *, line: bool):
    """A figure colour in the palette: near black becomes the foreground, near white the
    background (a line, such as a light grid, becomes the grid colour), the model blue
    the palette's; other colours and "none" stay."""
    if isinstance(color, str) and color.lower() in ("none", "auto"):
        return color
    rgba = to_rgba(color)
    if rgba[3] == 0:
        return color
    lightness = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    if lightness < 0.3:
        return to_rgba(p.foreground, rgba[3])
    if lightness > 0.85:
        return to_rgba(p.grid if line else p.background, rgba[3])
    if to_hex(rgba) == to_hex(LIGHT.model):
        return to_rgba(p.model, rgba[3])
    return color


def restyle_figure(figure: Figure, p: Palette) -> None:
    """A figure drawn by `forster.plotting` (the command line's colours, for a white
    page) recoloured for the palette: lines, markers, text and filled areas (legends
    included), then the backgrounds of the figure, its subfigures and axes. Light leaves
    the colours as drawn."""
    figure.set_facecolor(p.background)
    if p == LIGHT:
        return
    for line in figure.findobj(Line2D):
        line.set_color(_swap(line.get_color(), p, line=True))
        line.set_markerfacecolor(_swap(line.get_markerfacecolor(), p, line=False))
        line.set_markeredgecolor(_swap(line.get_markeredgecolor(), p, line=False))
    for text in figure.findobj(Text):
        text.set_color(_swap(text.get_color(), p, line=True))
    for patch in figure.findobj(Patch):  # legend handles of filled areas, among others
        patch.set_facecolor(_swap(patch.get_facecolor(), p, line=False))
        patch.set_edgecolor(_swap(patch.get_edgecolor(), p, line=True))
    for collection in figure.findobj(Collection):
        collection.set_facecolor([_swap(c, p, line=False) for c in collection.get_facecolor()])
        collection.set_edgecolor([_swap(c, p, line=True) for c in collection.get_edgecolor()])
    # Backgrounds last: the swaps above would turn a dark background into the foreground.
    for sub in figure.subfigs:
        sub.set_facecolor(p.background)
    style_axes(figure, list(figure.findobj(Axes)), p)
