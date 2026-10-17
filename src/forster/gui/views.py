"""The centre's extra views: the distributions of time (`plot_time_distributions`) and
the law comparison (`plot_law_comparison` above `plot_law_times`), drawn with FORSTER's
own figure functions, so they match what ``forster fit`` writes, recoloured for the
dark theme like the profile plot (`forster.gui.style.restyle_figure`).

The window computes what they show (the law fits run in a worker thread) and hands it
to `TimePage.draw` and `LawPage.draw`.
"""

from __future__ import annotations

from collections.abc import Callable

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from forster.model import FitResult
from forster.plotting import (
    SIGMA,
    plot_law_comparison,
    plot_law_times,
    plot_time_distributions,
)

from .state import TIME_VIEWS, TimeView
from .style import LIGHT, Palette, restyle_figure


class FigurePage(QWidget):
    """A matplotlib figure with its toolbar, and a short status line under it."""

    def __init__(self, figsize: tuple[float, float]) -> None:
        super().__init__()
        self.figure = Figure(figsize=figsize, layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.layout_ = QVBoxLayout(self)
        self.top = QHBoxLayout()
        self.top.addWidget(NavigationToolbar2QT(self.canvas, self), 1)
        self.layout_.addLayout(self.top)
        self.layout_.addWidget(self.canvas, 1)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.layout_.addWidget(self.status)
        self.drawn = False  # False while it shows a message, not a figure
        self.theme: Palette = LIGHT
        self.redraw: Callable[[], None] | None = None  # draws the figure shown again

    def set_palette(self, palette: Palette) -> None:
        """Follow the theme: what is shown is drawn again in the palette's colours."""
        if palette == self.theme:
            return
        self.theme = palette
        if self.redraw is not None:
            self.redraw()

    def _finish(self) -> None:
        restyle_figure(self.figure, self.theme)
        self.canvas.draw_idle()

    def show_message(self, text: str) -> None:
        self.figure.clear()
        self.figure.text(0.5, 0.5, text, ha="center", va="center", wrap=True)
        self.drawn = False
        self.redraw = lambda: self.show_message(text)
        self._finish()


class TimePage(FigurePage):
    """The run's distributions of time: the fit alone, with the starting profile's
    uncertainty, and with the conditions and the law (`forster.gui.state.TIME_VIEWS`)."""

    mode_changed = Signal(str)

    def __init__(self) -> None:
        super().__init__((6.5, 3.8))
        self.mode = QComboBox()
        for key, title in TIME_VIEWS.items():
            self.mode.addItem(title, key)
        self.mode.setToolTip(
            "Which distributions to draw: each adds to the one before. Each curve is scaled "
            "to a peak of 1; the legend gives the median and the central 68%."
        )
        self.mode.currentIndexChanged.connect(lambda _: self.mode_changed.emit(self.current))
        self.top.insertWidget(0, QLabel("Show"))
        self.top.insertWidget(1, self.mode)

    @property
    def current(self) -> str:
        return str(self.mode.currentData())

    def set_mode(self, mode: str) -> None:
        index = self.mode.findData(mode)
        if index >= 0:
            blocked = self.mode.blockSignals(True)
            self.mode.setCurrentIndex(index)
            self.mode.blockSignals(blocked)

    def draw(self, view: TimeView | str) -> None:
        if isinstance(view, str):
            self.show_message(view)
            return
        result, posterior, mc = view
        self.figure.clear()
        plot_time_distributions(result, posterior, mc, self.figure.subplots())
        self.drawn = True
        self.redraw = lambda: self.draw(view)
        self._finish()


class LawPage(FigurePage):
    """The run fitted under each offered Fe-Mg equation (its own first): the profile with
    each best fit, above each equation's time and ranges."""

    def __init__(self) -> None:
        super().__init__((7.0, 8.0))
        self.status.setText(
            "The run refitted under each offered Fe-Mg equation; its own equation first. "
            f"Thick bars: 1{SIGMA} from the profile alone; thin bars: with the equation's "
            "calibration (equations with a covariance)."
        )
        self.key: tuple[str, ...] | None = None  # the fits drawn (their cache keys)

    def draw(self, results: list[FitResult], key: tuple[str, ...]) -> None:
        self.figure.clear()
        # Subfigures, so the times panel's long labels do not narrow the profile panel.
        top, bottom = self.figure.subfigures(2, 1, height_ratios=(2.2, 1))
        plot_law_comparison(results, top.subplots())
        plot_law_times(results, bottom.subplots())
        self.key, self.drawn = key, True
        self.redraw = lambda: self.draw(results, key)
        self._finish()
