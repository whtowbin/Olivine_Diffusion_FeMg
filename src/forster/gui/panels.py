"""The right-hand tabs: Segments (Henry's side table), Conditions, Orientation,
Diffusion equation and Uncertainties. Each shows the selected run's `RunEditor` and
forwards edits to it; ``edited`` is emitted after an accepted edit (the window refits),
and a refusal is shown next to its field from ``editor.errors``."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from forster.model import FO2_BUFFERS

from .state import (
    DIFFUSIVITY_NOTES,
    SCAN_METHODS,
    SEGMENT_COLUMNS,
    SEGMENT_TITLES,
    RunEditor,
    SegmentRow,
    pct,
    um,
)

ERROR_COLOR = "#d03030"  # readable on light and dark backgrounds


class ErrorLabel(QLabel):
    """A refusal shown under its field; hidden while there is none."""

    def __init__(self) -> None:
        super().__init__()
        self.setWordWrap(True)
        self.setStyleSheet(f"color: {ERROR_COLOR};")
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.setVisible(False)

    def show_errors(self, messages: list[str]) -> None:
        self.setText("\n".join(messages))
        self.setVisible(bool(messages))


def _note(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setStyleSheet("color: palette(placeholder-text);")
    return label


class Tab(QWidget):
    """A tab bound to the selected run's editor; ``edited`` after an accepted edit."""

    edited = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.editor: RunEditor | None = None
        self.loading = False

    def load(self, editor: RunEditor | None) -> None:
        self.editor = editor
        self.setEnabled(editor is not None)
        if editor is not None:
            self.loading = True
            try:
                self.refresh()
            finally:
                self.loading = False

    def refresh(self) -> None:  # fill the fields from self.editor
        raise NotImplementedError

    def apply(self, edit: Callable[[RunEditor], bool]) -> None:
        """Run an edit; refresh (showing a refusal) and signal an accepted one."""
        if self.editor is None or self.loading:
            return
        accepted = edit(self.editor)
        self.load(self.editor)
        if accepted:
            self.edited.emit()

    def errors(self, *prefixes: str) -> list[str]:
        if self.editor is None:
            return []
        return [m for k, m in self.editor.errors.items() if k.startswith(prefixes)]


class SegmentsTab(Tab):
    """One row per segment: name, start, end, Fo and its uncertainty."""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.addWidget(
            _note(
                "Segments run from edge 1 inward; each ends where the next starts (a step). "
                "Drag the orange lines on the plot or type here. Fo in mol%."
            )
        )
        self.table = QTableWidget(0, len(SEGMENT_COLUMNS))
        self.table.setHorizontalHeaderLabels([SEGMENT_TITLES[c] for c in SEGMENT_COLUMNS])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.itemChanged.connect(self._cell_changed)
        layout.addWidget(self.table, 1)
        buttons = QHBoxLayout()
        self.split_button = QPushButton("Split segment")
        self.split_button.setToolTip("Split the selected segment in two at its middle")
        self.split_button.clicked.connect(lambda: self.apply(lambda e: e.add_segment(self._row())))
        self.remove_button = QPushButton("Remove segment")
        self.remove_button.setToolTip("The next segment takes the selected one's place")
        self.remove_button.clicked.connect(
            lambda: self.apply(lambda e: e.remove_segment(self._row()))
        )
        buttons.addWidget(self.split_button)
        buttons.addWidget(self.remove_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        form = QFormLayout()
        self.edge1 = QLineEdit()
        self.edge1.setPlaceholderText("segment a's Fo")
        self.edge1.setToolTip(
            "Fo held at edge 1 during diffusion, when it differs from the first "
            "segment's (table column edge1_fo). Blank: the first segment's Fo."
        )
        self.edge1.editingFinished.connect(
            lambda: self.apply(lambda e: e.set_edge1(self.edge1.text()))
        )
        self.centre = QLineEdit()
        self.centre.setPlaceholderText("half profile: far end held")
        self.centre.setToolTip(
            "Distance of the crystal's centre (centre_um). Given, the profile is mirrored "
            "there, so no Fo crosses the centre and the core is free to change (a whole "
            "crystal; implicit scheme, no growth stages). Blank: the far end is held at "
            "the last segment's Fo."
        )
        self.centre.editingFinished.connect(
            lambda: self.apply(lambda e: e.set_centre(self.centre.text()))
        )
        form.addRow("Edge 1 Fo (mol%)", self.edge1)
        form.addRow("Crystal centre (µm)", self.centre)
        layout.addLayout(form)
        self.error = ErrorLabel()
        layout.addWidget(self.error)

    def _row(self) -> int:
        row = self.table.currentRow()
        return row if row >= 0 else self.table.rowCount() - 1

    def refresh(self) -> None:
        assert self.editor is not None
        self.show_rows(self.editor.rows)
        self.edge1.setText(self.editor.edge1_text)
        self.centre.setText(self.editor.centre_text)
        self.error.show_errors(self.errors("segment", "edge1", "centre", "drag"))

    def show_rows(self, rows: list[SegmentRow]) -> None:
        """Fill the table (also while a handle is dragged), marking refused cells."""
        errors = self.editor.errors if self.editor is not None else {}
        blocked = self.table.blockSignals(True)
        self.table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            texts = (r.name, um(r.start_um), um(r.end_um), pct(r.fo), pct(r.spread), r.kind)
            for j, (column, text) in enumerate(zip(SEGMENT_COLUMNS, texts, strict=True)):
                item = self.table.item(i, j)
                if item is None:
                    item = QTableWidgetItem()
                    self.table.setItem(i, j, item)
                item.setText(text)
                message = errors.get(f"segment:{i}:{column}")
                item.setToolTip(message or "")
                item.setForeground(QBrush(QColor(ERROR_COLOR)) if message else QBrush())
        self.table.blockSignals(blocked)

    def _cell_changed(self, item: QTableWidgetItem) -> None:
        row, column = item.row(), SEGMENT_COLUMNS[item.column()]
        text = item.text()
        self.apply(lambda e: e.set_cell(row, column, text))


class ConditionsTab(Tab):
    """T, P and fO2 (relative to a buffer, or in Pa)."""

    def __init__(self) -> None:
        super().__init__()
        form = QFormLayout(self)
        self.fields: dict[str, QLineEdit] = {}
        for key, title, tip in (
            ("T_C", "T (°C)", "Temperature during diffusion"),
            ("P_MPa", "P (MPa)", "Pressure during diffusion (the table stores Pa)"),
        ):
            form.addRow(title, self._field(key, tip))
        self.buffer_mode = QRadioButton("Relative to a buffer")
        self.pa_mode = QRadioButton("In Pa")
        group = QButtonGroup(self)
        for button in (self.buffer_mode, self.pa_mode):
            group.addButton(button)
            button.toggled.connect(lambda on: on and self._edit("fO2_mode"))
        row = QHBoxLayout()
        row.addWidget(self.buffer_mode)
        row.addWidget(self.pa_mode)
        form.addRow("fO2", row)
        self.buffer = QComboBox()
        self.buffer.addItems([b for b in FO2_BUFFERS if b != "FMQ"])
        self.buffer.currentTextChanged.connect(lambda _: self._edit("fO2_buffer"))
        form.addRow("Buffer", self.buffer)
        form.addRow("Offset (log units)", self._field("fO2_delta_log", "log10 fO2 - buffer"))
        form.addRow("fO2 (Pa)", self._field("fO2_Pa", "Oxygen fugacity in Pa, e.g. 1e-6"))
        self.error = ErrorLabel()
        form.addRow(self.error)
        form.addRow(
            _note("A buffer-relative fO2 follows T and P: changing T recomputes fO2 in Pa.")
        )

    def _field(self, key: str, tip: str) -> QLineEdit:
        edit = QLineEdit()
        edit.setToolTip(tip)
        edit.editingFinished.connect(lambda: self._edit(key))
        self.fields[key] = edit
        return edit

    def values(self) -> dict[str, str]:
        out = {k: e.text() for k, e in self.fields.items()}
        out["fO2_mode"] = "buffer" if self.buffer_mode.isChecked() else "Pa"
        out["fO2_buffer"] = self.buffer.currentText()
        return out

    def _edit(self, key: str) -> None:
        if self.loading:
            return
        values = self.values()
        self.apply(lambda e: e.set_conditions(values, key))

    def refresh(self) -> None:
        assert self.editor is not None
        values = self.editor.conditions_text()
        for key, edit in self.fields.items():
            edit.setText(values[key])
        buffer_mode = values["fO2_mode"] == "buffer"
        (self.buffer_mode if buffer_mode else self.pa_mode).setChecked(True)
        self.buffer.setCurrentText("QFM" if values["fO2_buffer"] == "FMQ" else values["fO2_buffer"])
        for key in ("fO2_delta_log",):
            self.fields[key].setEnabled(buffer_mode)
        self.buffer.setEnabled(buffer_mode)
        self.fields["fO2_Pa"].setEnabled(not buffer_mode)
        self.error.show_errors(self.errors("T_C", "P_MPa", "fO2"))


class OrientationTab(Tab):
    """Angles to the crystal axes, or EBSD Euler angles with the profile's direction."""

    ANGLES = (
        ("alpha_deg", "\N{GREEK SMALL LETTER ALPHA} to [100] (°)"),
        ("beta_deg", "\N{GREEK SMALL LETTER BETA} to [010] (°)"),
        ("gamma_deg", "\N{GREEK SMALL LETTER GAMMA} to [001] (°)"),
    )
    EULER = (
        ("phi1_deg", "φ1 (°)"),
        ("Phi_deg", "Φ (°)"),
        ("phi2_deg", "φ2 (°)"),
        ("profile_angle_deg", "Profile angle (°)"),
    )

    def __init__(self) -> None:
        super().__init__()
        form = QFormLayout(self)
        self.angles_mode = QRadioButton("Angles to the axes")
        self.euler_mode = QRadioButton("EBSD Euler angles")
        group = QButtonGroup(self)
        row = QHBoxLayout()
        for button in (self.angles_mode, self.euler_mode):
            group.addButton(button)
            row.addWidget(button)
            button.toggled.connect(lambda on: on and self._mode_changed())
        form.addRow(row)
        self.fields: dict[str, QLineEdit] = {}
        for key, title in (*self.ANGLES, *self.EULER):
            edit = QLineEdit()
            edit.editingFinished.connect(lambda k=key: self._edit(k))
            self.fields[key] = edit
            form.addRow(title, edit)
        form.addRow(
            _note(
                "Euler angles: Bunge convention; the profile angle is measured from the "
                "EBSD map's positive x axis (east)."
            )
        )
        self.error = ErrorLabel()
        form.addRow(self.error)

    def _mode(self) -> str:
        return "angles" if self.angles_mode.isChecked() else "euler"

    def _mode_changed(self) -> None:
        self._enable()
        if not self.loading and self._mode() == "angles":
            self._edit("alpha_deg")  # the angles shown are the run's: switch at once

    def _enable(self) -> None:
        angles = self._mode() == "angles"
        for key, _ in self.ANGLES:
            self.fields[key].setEnabled(angles)
        for key, _ in self.EULER:
            self.fields[key].setEnabled(not angles)

    def _edit(self, key: str) -> None:
        if self.loading:
            return
        values = {k: e.text() for k, e in self.fields.items()} | {"mode": self._mode()}
        self.apply(lambda e: e.set_orientation(values, key))

    def refresh(self) -> None:
        assert self.editor is not None
        values = self.editor.orientation_text()
        for key, edit in self.fields.items():
            edit.setText(values[key])
        (self.angles_mode if values["mode"] == "angles" else self.euler_mode).setChecked(True)
        self._enable()
        keys = [k for k, _ in (*self.ANGLES, *self.EULER)]
        self.error.show_errors(self.errors(*keys))


class DiffusionTab(Tab):
    """The four named Fe-Mg equations (and the run's own, if it uses another)."""

    def __init__(self) -> None:
        super().__init__()
        self.layout_ = QVBoxLayout(self)
        self.layout_.addWidget(
            _note("The Fe-Mg diffusivity equation: it sets D, so it scales every time.")
        )
        self.group = QButtonGroup(self)
        self.buttons: dict[str, QRadioButton] = {}
        self.box = QVBoxLayout()
        self.layout_.addLayout(self.box)
        self.error = ErrorLabel()
        self.layout_.addWidget(self.error)
        self.layout_.addStretch(1)

    def refresh(self) -> None:
        assert self.editor is not None
        choices = self.editor.diffusivity_choices()
        if [k for k, _ in choices] != list(self.buttons):
            self._build(choices)
        self.buttons[self.editor.diffusivity_key].setChecked(True)
        self.error.show_errors(self.errors("diffusivity"))

    def _build(self, choices: list[tuple[str, str]]) -> None:
        while self.box.count():
            item = self.box.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.deleteLater()
        for button in self.buttons.values():
            self.group.removeButton(button)
        self.buttons = {}
        for key, text in choices:
            button = QRadioButton(text)
            button.toggled.connect(lambda on, k=key: on and self._chosen(k))
            self.group.addButton(button)
            self.buttons[key] = button
            self.box.addWidget(button)
            self.box.addWidget(_note(DIFFUSIVITY_NOTES.get(key, "From this project.")))

    def _chosen(self, key: str) -> None:
        self.apply(lambda e: e.set_diffusivity(key))


class UncertaintiesTab(Tab):
    """Step-position uncertainties, the scan method and the starting-profile fit. (The
    segments' Fo uncertainties are in the Segments tab.)"""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.addWidget(
            _note(
                "Give a ± for the step positions here and for the segments' Fo in the "
                "Segments tab; 'Fit starting profile' then fits them with the time and "
                "samples their posterior. Blank: held fixed."
            )
        )
        self.steps = QTableWidget(0, 2)
        self.steps.setHorizontalHeaderLabels(["Step at (µm)", "± (µm)"])
        self.steps.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.steps.verticalHeader().setVisible(False)
        self.steps.itemChanged.connect(self._step_changed)
        layout.addWidget(self.steps, 1)
        form = QFormLayout()
        self.kind = QComboBox()
        self.kind.addItems(["sd", "range"])
        self.kind.setToolTip("sd: Gaussian sigma; range: uniform half-width (all steps)")
        self.kind.currentTextChanged.connect(
            lambda word: self.apply(lambda e: e.set_steps_kind(word))
        )
        form.addRow("Steps' ± is a", self.kind)
        self.method = QComboBox()
        self.method.addItems(list(SCAN_METHODS))
        self.method.setToolTip(
            "auto: importance sampling from a Laplace proposal, nested sampling if that "
            "proposal is poor; laplace; nested; grid: every node of a grid"
        )
        self.method.currentTextChanged.connect(lambda m: self.apply(lambda e: e.set_scan_method(m)))
        form.addRow("Method", self.method)
        layout.addLayout(form)
        buttons = QHBoxLayout()
        self.fit_button = QPushButton("Fit starting profile")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setVisible(False)
        buttons.addWidget(self.fit_button)
        buttons.addWidget(self.cancel_button)
        layout.addLayout(buttons)
        self.progress = QProgressBar()
        self.progress.setRange(0, 3)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        self.result = QLabel()
        self.result.setWordWrap(True)
        self.result.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.result)
        self.apply_button = QPushButton("Use the best starting profile")
        self.apply_button.setEnabled(False)
        layout.addWidget(self.apply_button)
        self.error = ErrorLabel()
        layout.addWidget(self.error)

    def refresh(self) -> None:
        assert self.editor is not None
        rows = self.editor.rows
        blocked = self.steps.blockSignals(True)
        self.steps.setRowCount(len(rows) - 1)
        for i, (row, spread) in enumerate(zip(rows[:-1], self.editor.step_spreads, strict=True)):
            where = QTableWidgetItem(f"{um(row.end_um)}  ({row.name} | {rows[i + 1].name})")
            where.setFlags(where.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.steps.setItem(i, 0, where)
            self.steps.setItem(i, 1, QTableWidgetItem(um(spread)))
        self.steps.blockSignals(blocked)
        self.kind.setCurrentText(self.editor.steps_kind)
        self.method.setCurrentText(self.editor.run.scan_method)
        self.error.show_errors(self.errors("step", "scan_method", "initial"))

    def _step_changed(self, item: QTableWidgetItem) -> None:
        if item.column() != 1:
            return
        i, text = item.row(), item.text()
        self.apply(lambda e: e.set_step_spread(i, text))


class SidePanel(QTabWidget):
    """The five tabs; ``edited`` after any accepted edit."""

    edited = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.segments = SegmentsTab()
        self.conditions = ConditionsTab()
        self.orientation = OrientationTab()
        self.diffusion = DiffusionTab()
        self.uncertainties = UncertaintiesTab()
        for tab, title in (
            (self.segments, "Segments"),
            (self.conditions, "Conditions"),
            (self.orientation, "Orientation"),
            (self.diffusion, "Diffusion equation"),
            (self.uncertainties, "Uncertainties"),
        ):
            self.addTab(tab, title)
            tab.edited.connect(self._edited)
        self.load(None)

    def tabs(self) -> list[Tab]:
        return [
            self.segments,
            self.conditions,
            self.orientation,
            self.diffusion,
            self.uncertainties,
        ]

    def load(self, editor: RunEditor | None) -> None:
        for tab in self.tabs():
            tab.load(editor)

    def _edited(self) -> None:
        editor = self.segments.editor
        for tab in self.tabs():  # an edit in one tab can change what another shows
            tab.load(editor)
        self.edited.emit()
