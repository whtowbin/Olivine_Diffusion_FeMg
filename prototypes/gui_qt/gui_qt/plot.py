"""The profile plot: measured points, the fit (`forster.plotting.plot_fit`) and the
starting profile as three draggable lines (the step position, the edge Fo and the
core Fo)."""

from __future__ import annotations

import time

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from PySide6.QtCore import Signal

from forster.model import FitResult, InitialCondition, Profile
from forster.plotting import DATA_COLOR, plot_fit

HANDLE_COLOR = "#D55E00"  # Okabe-Ito vermilion: apart from the model blue and data black
HANDLES = ("step", "edge", "core")


class ProfileCanvas(FigureCanvasQTAgg):
    """Emits ``moved(handle, value)`` while a line is dragged and ``released(handle,
    value, t)`` when it is let go (``t``: `time.perf_counter` at the release). Values
    are in the plot's units: µm for the step, mol% for Fo."""

    moved = Signal(str, float)
    released = Signal(str, float, float)

    def __init__(self) -> None:
        self.figure = Figure(figsize=(7.5, 4.5), layout="constrained")
        super().__init__(self.figure)
        self.ax = self.figure.add_subplot()
        self.lines: dict[str, Line2D] = {}
        self.domain = (0.0, 1.0)
        self.values = {"step": 0.0, "edge": 0.0, "core": 0.0}
        self._dragging: str | None = None
        self.toolbar = None  # set by the window, so pan and zoom modes suppress dragging
        self.mpl_connect("button_press_event", self._press)
        self.mpl_connect("motion_notify_event", self._motion)
        self.mpl_connect("button_release_event", self._release)
        self.show_empty("Open a project folder, or start one from the example.")

    # -- drawing --------------------------------------------------------------------

    def show_empty(self, message: str) -> None:
        self.ax.clear()
        self.lines = {}
        self.ax.set_axis_off()
        self.ax.text(0.5, 0.5, message, ha="center", va="center", transform=self.ax.transAxes)
        self.draw()

    def show_run(
        self,
        name: str,
        profile: Profile,
        initial: InitialCondition | None,
        result: FitResult | None,
        note: str = "",
    ) -> None:
        """Redraw everything: the fit if there is one, else the measured points; then
        the starting profile's handles (``initial`` None: a run this prototype cannot
        drag, such as one with two steps)."""
        ax = self.ax
        ax.clear()
        ax.set_axis_on()
        ax.set_autoscale_on(True)
        if result is not None:
            plot_fit(result, ax)
        else:
            ax.plot(
                profile.distance_um,
                np.asarray(profile.fo) * 100,
                linestyle="none",
                marker="o",
                markersize=4,
                markerfacecolor="white",
                markeredgecolor=DATA_COLOR,
            )
            ax.set_xlabel("Distance from the crystal edge (µm)")
            ax.set_ylabel("Fo (mol%)")
            ax.set_title(f"{name}\n{note or 'not fitted yet'}", fontsize="medium")
        self.lines = {}
        if initial is not None:
            self.domain = initial.domain_um
            self.values = {
                "step": float(initial.steps_um[0]),
                "edge": initial.edge_fo * 100,
                "core": initial.core_fo * 100,
            }
            style = {"color": HANDLE_COLOR, "lw": 2.5, "pickradius": 8, "zorder": 5}
            self.lines["step"] = ax.axvline(self.values["step"], ls=(0, (4, 2)), **style)
            self.lines["edge"] = ax.add_line(Line2D([], [], **style))
            self.lines["core"] = ax.add_line(Line2D([], [], **style))
            self._place()
            low, high = ax.get_ylim()
            pad = 0.15 * (high - low)  # room to drag Fo past the data
            ax.set_ylim(low - pad, high + pad)
        ax.set_autoscale_on(False)  # the axes stay put while a line is dragged
        self.draw()

    def set_values(self, **values: float) -> None:
        """Move the handles (to mirror the number fields, or to undo a refused drag)."""
        self.values.update(values)
        if self.lines:
            self._place()
            self.draw_idle()

    def _place(self) -> None:
        step, (left, right) = self.values["step"], self.domain
        self.lines["step"].set_xdata([step, step])
        self.lines["edge"].set_data([left, step], [self.values["edge"]] * 2)
        self.lines["core"].set_data([step, right], [self.values["core"]] * 2)

    # -- dragging -------------------------------------------------------------------

    def _press(self, event) -> None:
        if event.inaxes is not self.ax or event.button != 1 or not self.lines:
            return
        if self.toolbar is not None and self.toolbar.mode:  # panning or zooming
            return
        for handle in HANDLES:
            hit, _ = self.lines[handle].contains(event)
            if hit:
                self._dragging = handle
                return

    def _motion(self, event) -> None:
        if self._dragging is None or event.xdata is None or event.ydata is None:
            return
        value = event.xdata if self._dragging == "step" else event.ydata
        self.values[self._dragging] = self._snap(self._dragging, value)
        self._place()
        self.draw_idle()
        self.moved.emit(self._dragging, self.values[self._dragging])

    def _release(self, event) -> None:
        if self._dragging is None:
            return
        released_at = time.perf_counter()
        handle, self._dragging = self._dragging, None
        self.released.emit(handle, self.values[handle], released_at)

    @staticmethod
    def _snap(handle: str, value: float) -> float:
        # 0.1 µm and 0.01 mol%: finer than the data, and keeps parameters.csv readable.
        return round(value, 1) if handle == "step" else round(value, 2)
