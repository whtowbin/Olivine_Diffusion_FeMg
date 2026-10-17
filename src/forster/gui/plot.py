"""The centre plot: measured points, the best fit with its 1-sigma and 2-sigma bands, the
starting profile as draggable lines, the segment labels above the axes, and the
residuals below.

Handles, as ``(kind, index)``: ``("step", i)`` the i-th step position, ``("fo", i)``
the i-th segment's Fo, ``("start", 0)`` and ``("end", 0)`` the model domain's ends.
The canvas emits ``moved(kind, index, value)`` while a handle is dragged and
``released(kind, index, value)`` when it is let go: um for positions, mol% for Fo.
A shift-click emits ``mask_clicked(x_um, fo_pct, px_per_um, px_per_pct)`` (the window
masks or unmasks the nearest point); masked points are drawn as grey crosses.

A zoom or pan the user makes is kept while the same run is redrawn (after every refit);
`user_limits` reads it for the run's saved view and `set_user_limits` restores it.
"""

from __future__ import annotations

import numpy as np
from matplotlib.backend_bases import MouseEvent
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.transforms import blended_transform_factory
from PySide6.QtCore import Signal

from forster.io import fit_tables
from forster.model import FitResult, Profile
from forster.plotting import SIGMA

from .state import SegmentRow
from .style import LIGHT, Palette, style_axes

Handle = tuple[str, int]


def title_of(name: str, result: FitResult | None, note: str = "") -> str:
    """The plot's title: the run, its time and both ranges."""
    if result is None:
        return f"{name}\n{note or 'not fitted yet'}"
    low, high = result.time_range_days()
    text = f"{result.time_days:.3g} days (1{SIGMA} {low:.3g}-{high:.3g}"
    if result.indices2 is not None:
        low2, high2 = result.time_range_days(sigmas=2)
        text += f", 2{SIGMA} {low2:.3g}-{high2:.3g}"
    return f"{name}\n{text})"


class ProfileCanvas(FigureCanvasQTAgg):
    moved = Signal(str, int, float)
    released = Signal(str, int, float)
    mask_clicked = Signal(float, float, float, float)

    def __init__(self) -> None:
        self.figure = Figure(figsize=(7.5, 5.0), layout="constrained")
        super().__init__(self.figure)
        grid = self.figure.add_gridspec(2, 1, height_ratios=(3, 1))
        self.ax = self.figure.add_subplot(grid[0])
        self.rax = self.figure.add_subplot(grid[1], sharex=self.ax)
        self.palette: Palette = LIGHT
        self.rows: list[SegmentRow] = []
        self.handles: dict[Handle, Line2D] = {}
        self.labels: list[Text] = []
        self.dragging: Handle | None = None
        self.last_value: float | None = None
        self.toolbar = None  # set by the window: pan and zoom modes suppress dragging
        self.run_name: str | None = None
        self.auto_limits: tuple[tuple[float, float], tuple[float, float]] | None = None
        self.mpl_connect("button_press_event", self._press)
        self.mpl_connect("motion_notify_event", self._motion)
        self.mpl_connect("button_release_event", self._release)
        self.show_message("Open a project folder, or start one from the example.")

    # -- drawing --------------------------------------------------------------------

    def show_message(self, message: str) -> None:
        for ax in (self.ax, self.rax):
            ax.clear()
            ax.set_axis_off()
        self.handles, self.labels, self.rows = {}, [], []
        self.run_name = self.auto_limits = None
        self.ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            transform=self.ax.transAxes,
            color=self.palette.foreground,
        )
        self.figure.set_facecolor(self.palette.background)
        self.draw_idle()

    def show_run(
        self,
        name: str,
        profile: Profile,
        rows: list[SegmentRow],
        result: FitResult | None,
        note: str = "",
        masked: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        """Redraw everything for one run (``masked``: the profile's masked points,
        distances and Fo as a fraction), keeping the user's zoom on the same run."""
        p = self.palette
        ax, rax = self.ax, self.rax
        zoom = self.user_limits() if name == self.run_name else None
        for a in (ax, rax):
            a.clear()
            a.set_axis_on()
        x, fo = np.asarray(profile.distance_um), np.asarray(profile.fo) * 100
        if result is not None:
            self._draw_fit(result)
        inside = (x >= rows[0].start_um) & (x <= rows[-1].end_um) if rows else x == x
        for keep, color, text in (
            (inside, p.data, "Measured"),
            (~inside, p.muted, "Outside the model domain"),
        ):
            if keep.any():
                ax.plot(
                    x[keep],
                    fo[keep],
                    linestyle="none",
                    marker="o",
                    markersize=4.5,
                    markerfacecolor=p.background,
                    markeredgecolor=color,
                    label=text,
                    zorder=4,
                )
        if masked is not None and len(masked[0]):
            ax.plot(
                masked[0],
                np.asarray(masked[1]) * 100,
                linestyle="none",
                marker="x",
                markersize=5,
                color=p.muted,
                label="Masked (left out)",
                zorder=4,
            )
        self.rows = list(rows)
        self._draw_handles()
        ax.set_ylabel("Fo (mol%)")
        ax.set_title(title_of(name, result, note), fontsize="medium", pad=18)
        ax.tick_params(labelbottom=False)
        ax.legend(loc="best", fontsize="small", frameon=False)
        self._draw_residuals(result)
        self._set_limits(x, fo)
        self.run_name = name
        self.auto_limits = (ax.get_xlim(), ax.get_ylim())
        if zoom is not None:
            self.set_user_limits(*zoom)
        style_axes(self.figure, [ax, rax], p)
        self.draw_idle()

    def user_limits(self) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """The axes limits if the user zoomed or panned away from the drawn ones."""
        if self.auto_limits is None:
            return None
        (x0, x1), (y0, y1) = self.ax.get_xlim(), self.ax.get_ylim()
        now = ((float(x0), float(x1)), (float(y0), float(y1)))
        return None if np.allclose(now, self.auto_limits) else now

    def set_user_limits(self, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.draw_idle()

    def _draw_fit(self, result: FitResult) -> None:
        p, ax = self.palette, self.ax
        x = result.model_x_um
        _, low, high = result.indices
        if result.indices2 is not None:
            ax.fill_between(
                x,
                result.curve(result.indices2[0]) * 100,
                result.curve(result.indices2[1]) * 100,
                color=p.model,
                alpha=0.13,
                linewidth=0,
                label=f"2{SIGMA} range",
            )
        ax.fill_between(
            x,
            result.curve(low) * 100,
            result.curve(high) * 100,
            color=p.model,
            alpha=0.28,
            linewidth=0,
            label=f"1{SIGMA} range",
        )
        ax.plot(x, result.best_fit * 100, color=p.model, lw=2.2, label="Best fit", zorder=3)

    def _draw_handles(self) -> None:
        p, ax = self.palette, self.ax
        self.handles = {}
        rows = self.rows
        for i, row in enumerate(rows):
            self.handles[("fo", i)] = ax.add_line(
                Line2D(
                    [row.start_um, row.end_um],
                    [row.fo * 100] * 2,
                    color=p.handle,
                    lw=2.6,
                    pickradius=7,
                    zorder=6,
                )
            )
        for i, row in enumerate(rows[:-1]):
            self.handles[("step", i)] = ax.axvline(
                row.end_um, color=p.handle, lw=1.8, ls=(0, (4, 2)), pickradius=7, zorder=6
            )
        if rows:
            self.handles[("start", 0)] = ax.axvline(
                rows[0].start_um, color=p.muted, lw=1.2, pickradius=7, zorder=6
            )
            self.handles[("end", 0)] = ax.axvline(
                rows[-1].end_um, color=p.muted, lw=1.2, pickradius=7, zorder=6
            )
            self.handles[("fo", 0)].set_label("Starting profile (drag)")
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        self.labels = [
            ax.text(
                (row.start_um + row.end_um) / 2,
                1.01,
                row.name,
                transform=trans,
                ha="center",
                va="bottom",
                fontsize="small",
                fontweight="bold",
                color=p.handle,
                clip_on=False,
            )
            for row in rows
        ]

    def _draw_residuals(self, result: FitResult | None) -> None:
        p, rax = self.palette, self.rax
        rax.axhline(0, color=p.muted, lw=0.8)
        if result is not None:
            points = fit_tables(result)["points"]
            rax.plot(
                points.distance_um,
                points.residual_fo * 100,
                linestyle="none",
                marker="o",
                markersize=3.5,
                markerfacecolor=p.background,
                markeredgecolor=p.data,
            )
        rax.set_ylabel("Residual (mol%)")
        rax.set_xlabel("Distance from edge 1 (µm)")

    def _set_limits(self, x: np.ndarray, fo: np.ndarray) -> None:
        """Fixed limits (they stay put while a handle is dragged), with room to drag
        the Fo lines past the data."""
        xs = [*x, *(r.start_um for r in self.rows), *(r.end_um for r in self.rows)]
        ys = [*fo[np.isfinite(fo)], *(r.fo * 100 for r in self.rows)]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        pad_x, pad_y = 0.04 * (x1 - x0 or 1), 0.18 * (y1 - y0 or 1)
        self.ax.set_xlim(x0 - pad_x, x1 + pad_x)
        self.ax.set_ylim(y0 - pad_y, y1 + pad_y)
        self.ax.set_autoscale_on(False)

    def set_rows(self, rows: list[SegmentRow]) -> None:
        """Move the handles and labels to ``rows`` (during a drag, or to undo one)."""
        self.rows = list(rows)
        for (kind, i), line in self.handles.items():
            if kind == "fo":
                line.set_data([rows[i].start_um, rows[i].end_um], [rows[i].fo * 100] * 2)
                continue
            x = {"step": rows[i].end_um, "start": rows[0].start_um, "end": rows[-1].end_um}[kind]
            line.set_xdata([x, x])
        for text, row in zip(self.labels, rows, strict=False):
            text.set_x((row.start_um + row.end_um) / 2)
        self.draw_idle()

    def set_palette(self, palette: Palette) -> None:
        self.palette = palette

    # -- dragging -------------------------------------------------------------------

    def handle_at(self, event: MouseEvent) -> Handle | None:
        """The handle under the mouse: positions before Fo lines."""
        order = sorted(self.handles, key=lambda h: h[0] == "fo")
        for handle in order:
            hit, _ = self.handles[handle].contains(event)
            if hit:
                return handle
        return None

    def _press(self, event) -> None:
        if event.inaxes is not self.ax or event.button != 1 or not self.handles:
            return
        if self.toolbar is not None and self.toolbar.mode:  # panning or zooming
            return
        if "shift" in (getattr(event, "modifiers", None) or ()):
            self._mask_click(event)
            return
        self.dragging = self.handle_at(event)

    def _mask_click(self, event) -> None:
        """Shift-click: the place and the plot's scale, for the window to find the
        nearest point."""
        (x0, y0), (x1, y1) = self.ax.transData.transform([(0, 0), (1, 1)])
        self.mask_clicked.emit(float(event.xdata), float(event.ydata), abs(x1 - x0), abs(y1 - y0))

    def _motion(self, event) -> None:
        if self.dragging is None or event.xdata is None or event.ydata is None:
            return
        kind, index = self.dragging
        self.last_value = float(event.ydata if kind == "fo" else event.xdata)
        self.moved.emit(kind, index, self.last_value)

    def _release(self, event) -> None:
        """Let go: the last position inside the axes (a press without a move is no edit)."""
        if self.dragging is None:
            return
        kind, index = self.dragging
        self.dragging = None
        if self.last_value is not None:
            self.released.emit(kind, index, self.last_value)
        self.last_value = None
