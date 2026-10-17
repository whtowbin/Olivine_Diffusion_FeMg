"""The right-hand tabs: Segments (Henry's side table), Data (the profile's points, masking,
moving the distances, import), Conditions, Orientation, Diffusion equation,
Uncertainties (the starting-profile fit and the Monte Carlo) and Advanced (scheme, grid,
growth stages). Each shows the selected run's `RunEditor` and forwards edits to it;
``edited`` is emitted after an accepted edit (the window refits), and a refusal is shown
next to its field from ``editor.errors``. Every field's tooltip gives its unit and the
parameter-table column it is saved in."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from forster.model import FO2_BUFFERS, SCHEMES

from .state import (
    DIFFUSIVITY_NOTES,
    SCAN_METHODS,
    SEGMENT_COLUMNS,
    SEGMENT_TITLES,
    STAGE_FIELDS,
    STAGE_TITLES,
    RunEditor,
    SegmentRow,
    pct,
    stage_rows,
    stage_text,
    um,
)

ERROR_COLOR = "#d03030"  # readable on light and dark backgrounds
SEGMENT_TIPS = {
    "name": "The segment's label (segment_names): a, b, c ... from edge 1 inward",
    "start": "Where the segment starts, µm from edge 1 (domain_start_um or a step)",
    "end": "Where it ends, µm (a step in steps_um, or domain_end_um for the last)",
    "fo": "The segment's starting Fo, mol% (segment_fo, stored as a fraction)",
    "spread": "Uncertainty of that Fo for the starting-profile fit, mol% "
    "(segment_fo_sd or segment_fo_range); blank: held fixed",
    "kind": "sd: a Gaussian sigma; range: a uniform half-width",
}


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
        for j, column in enumerate(SEGMENT_COLUMNS):
            header = self.table.horizontalHeaderItem(j)
            if header is not None:
                header.setToolTip(SEGMENT_TIPS[column])
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
            ("T_C", "T (°C)", "Temperature during diffusion, °C (T_C)"),
            ("P_MPa", "P (MPa)", "Pressure during diffusion, MPa (the table stores Pa: P_Pa)"),
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
        self.buffer.setToolTip("The fO2 buffer (fO2_buffer): QFM or NNO")
        form.addRow(
            "Offset (log units)",
            self._field("fO2_delta_log", "log10 fO2 minus the buffer's, log units (fO2_delta_log)"),
        )
        form.addRow("fO2 (Pa)", self._field("fO2_Pa", "Oxygen fugacity in Pa, e.g. 1e-6 (fO2_Pa)"))
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


ORIENTATION_TIPS = {
    "alpha_deg": "Angle between the profile and [100], degrees (alpha_deg)",
    "beta_deg": "Angle between the profile and [010], degrees (beta_deg)",
    "gamma_deg": "Angle between the profile and [001], degrees (gamma_deg)",
    "phi1_deg": "EBSD Euler angle phi1, degrees, Bunge convention (phi1_deg)",
    "Phi_deg": "EBSD Euler angle Phi, degrees (Phi_deg)",
    "phi2_deg": "EBSD Euler angle phi2, degrees (phi2_deg)",
    "profile_angle_deg": "The profile's direction on the EBSD map, degrees from its "
    "positive x axis (profile_angle_deg)",
}


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
            edit.setToolTip(ORIENTATION_TIPS[key])
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
        for key, button in self.buttons.items():
            button.setToolTip(f"Source: {self.editor.diffusivity_source(key)}")
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
    """Step-position uncertainties, the scan method and the starting-profile fit (the
    segments' Fo uncertainties are in the Segments tab); then the Monte Carlo over the
    conditions and the equation, with its own run button."""

    MC_FIELDS = (
        ("T_sd_C", "T sd (°C)", "1-sigma uncertainty of T, °C (mc_T_sd_C)"),
        (
            "P_sd_MPa",
            "P sd (MPa)",
            "1-sigma uncertainty of P, MPa (the table stores Pa: mc_P_sd_Pa)",
        ),
        ("fO2_sd_log", "fO2 sd (log units)", "1-sigma uncertainty of log10 fO2 (mc_fO2_sd_log)"),
        (
            "samples",
            "Draws",
            "How many draws (mc_samples); blank: 4000, or 200 when every "
            "draw is refitted (growth stages)",
        ),
        ("seed", "Seed", "The random seed (mc_seed): the same seed gives the same draws"),
    )

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.addWidget(self._build_scan())
        layout.addWidget(self._build_mc())
        layout.addStretch(1)

    def _build_scan(self) -> QGroupBox:
        box = QGroupBox("Starting profile")
        layout = QVBoxLayout(box)
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
        self.steps.setToolTip(
            "Each step's uncertainty, µm (steps_sd_um or steps_range_um); blank: fixed"
        )
        self.steps.setMinimumHeight(70)
        self.steps.setMaximumHeight(130)
        self.steps.itemChanged.connect(self._step_changed)
        layout.addWidget(self.steps)
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
            "scan_method. auto: importance sampling from a Laplace proposal, nested "
            "sampling if that proposal is poor; laplace; nested; grid: every node of a grid"
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
        return box

    def _build_mc(self) -> QGroupBox:
        box = QGroupBox("Monte Carlo: conditions and equation")
        layout = QVBoxLayout(box)
        layout.addWidget(
            _note(
                "Draws T, P, fO2 and the equation's parameters and turns each draw into a "
                "time (around the fit's own spread, or the starting-profile fit's when there "
                "is one). Results go to the Results table and the Time distribution view."
            )
        )
        self.mc_on = QCheckBox("Run a Monte Carlo for this run")
        self.mc_on.setToolTip("Saved as the mc_ columns of the parameter table")
        self.mc_on.toggled.connect(lambda _: self._mc_edit("on"))
        layout.addWidget(self.mc_on)
        form = QFormLayout()
        self.mc_fields: dict[str, QLineEdit] = {}
        for key, title, tip in self.MC_FIELDS:
            edit = QLineEdit()
            edit.setToolTip(tip)
            edit.editingFinished.connect(lambda k=key: self._mc_edit(k))
            self.mc_fields[key] = edit
            form.addRow(title, edit)
        self.mc_buffer = QComboBox()
        self.mc_buffer.setToolTip(
            "Hold fO2 to a buffer (mc_fO2_buffer): each draw's fO2 follows its T and P at "
            "the run's offset from the buffer, as a rock-buffered fO2 would. Not held: "
            "log fO2 is drawn on its own. A run with fO2 given relative to a buffer is "
            "always held."
        )
        self.mc_buffer.currentIndexChanged.connect(lambda _: self._mc_edit("fO2_buffer"))
        form.addRow("fO2 held to", self.mc_buffer)
        self.mc_law = QCheckBox("Draw the equation's parameters (its covariance)")
        self.mc_law.setToolTip("mc_law; an equation without a covariance adds nothing")
        self.mc_law.toggled.connect(lambda _: self._mc_edit("law"))
        form.addRow(self.mc_law)
        self.mc_between = QCheckBox("Add the spread between laboratories")
        self.mc_between.setToolTip(
            "mc_between_study: for an absolute time (the FORSTER refit carries it)"
        )
        self.mc_between.toggled.connect(lambda _: self._mc_edit("between_study"))
        form.addRow(self.mc_between)
        layout.addLayout(form)
        buttons = QHBoxLayout()
        self.mc_button = QPushButton("Run Monte Carlo")
        self.mc_cancel = QPushButton("Cancel")
        self.mc_cancel.setVisible(False)
        buttons.addWidget(self.mc_button)
        buttons.addWidget(self.mc_cancel)
        layout.addLayout(buttons)
        self.mc_progress = QProgressBar()
        self.mc_progress.setVisible(False)
        layout.addWidget(self.mc_progress)
        self.mc_result = QLabel()
        self.mc_result.setWordWrap(True)
        self.mc_result.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.mc_result)
        self.mc_error = ErrorLabel()
        layout.addWidget(self.mc_error)
        return box

    def mc_values(self) -> dict[str, str]:
        out = {k: e.text() for k, e in self.mc_fields.items()}
        out["on"] = "yes" if self.mc_on.isChecked() else "no"
        out["fO2_buffer"] = str(self.mc_buffer.currentData() or "")
        out["law"] = "yes" if self.mc_law.isChecked() else "no"
        out["between_study"] = "yes" if self.mc_between.isChecked() else "no"
        return out

    def _mc_edit(self, key: str) -> None:
        if self.loading:
            return
        values = self.mc_values()
        self.apply(lambda e: e.set_mc(values, key))

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
        self._refresh_mc()

    def _refresh_mc(self) -> None:
        assert self.editor is not None
        values = self.editor.mc_text()
        on = values["on"] == "yes"
        self.mc_on.setChecked(on)
        for key, edit in self.mc_fields.items():
            edit.setText(values[key])
            edit.setEnabled(on)
        self.mc_buffer.clear()
        for buffer in self.editor.mc_buffers():
            self.mc_buffer.addItem(buffer or "not held", buffer)
        self.mc_buffer.setCurrentIndex(max(0, self.mc_buffer.findData(values["fO2_buffer"])))
        self.mc_law.setChecked(values["law"] == "yes")
        self.mc_between.setChecked(values["between_study"] == "yes")
        for widget in (self.mc_buffer, self.mc_law, self.mc_between, self.mc_button):
            widget.setEnabled(on)
        keys = ("on", "fO2_buffer", "law", "between_study", *self.mc_fields)
        self.mc_error.show_errors(self.errors(*keys))

    def _step_changed(self, item: QTableWidgetItem) -> None:
        if item.column() != 1:
            return
        i, text = item.row(), item.text()
        self.apply(lambda e: e.set_step_spread(i, text))


class ProfileDataTab(Tab):
    """The run's profile: its points with a check box for each (in the fit or masked),
    moving the distances so the rim is at 0, and importing more profiles."""

    import_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        self.title = QLabel()
        self.title.setWordWrap(True)
        top.addWidget(self.title, 1)
        self.import_button = QPushButton("Import profiles…")
        self.import_button.setToolTip(
            "Add profiles from a CSV or Excel table, choosing which columns hold the "
            "profile name, the distance and Fo"
        )
        self.import_button.clicked.connect(self.import_requested.emit)
        top.addWidget(self.import_button)
        layout.addLayout(top)
        layout.addWidget(
            _note(
                "Untick a point (or shift-click it on the plot) to mask it: it stays in the "
                "project, drawn as a grey cross, and is left out of every fit of this profile "
                "(profiles.csv marks it Marked_bad, so forster fit leaves it out too)."
            )
        )
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Distance (µm)", "Fo (mol%)", "In the fit"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setMinimumHeight(180)
        self.table.itemChanged.connect(self._mask_changed)
        layout.addWidget(self.table, 1)
        box = QGroupBox("Move the distances so the rim is at 0")
        form = QFormLayout(box)
        self.flip = QCheckBox("The rim is at the far end (flip)")
        self.flip.setToolTip(
            "Distances become offset - x: the far end becomes the rim. Runs on this "
            "profile get a new starting profile."
        )
        form.addRow(self.flip)
        row = QHBoxLayout()
        self.offset = QLineEdit()
        self.offset.setToolTip(
            "µm subtracted from every distance (or, flipped, the distance that becomes 0)"
        )
        self.rim_button = QPushButton("Rim at 0")
        self.rim_button.setToolTip("The offset that puts the first point (flipped: the last) at 0")
        self.rim_button.clicked.connect(self._rim_at_zero)
        row.addWidget(self.offset, 1)
        row.addWidget(self.rim_button)
        form.addRow("Offset (µm)", row)
        self.move_button = QPushButton("Move")
        self.move_button.clicked.connect(
            lambda: self.apply(lambda e: e.move_profile(self.flip.isChecked(), self.offset.text()))
        )
        form.addRow(self.move_button)
        form.addRow(_note("A shift moves the runs on this profile with it (domain, steps)."))
        layout.addWidget(box)
        self.error = ErrorLabel()
        layout.addWidget(self.error)

    def _rim_at_zero(self) -> None:
        if self.editor is not None:
            self.offset.setText(self.editor.rim_offset_text(self.flip.isChecked()))

    def refresh(self) -> None:
        assert self.editor is not None
        points = self.editor.points()
        masked = int(points.masked.sum())
        name = self.editor.run.profile_name
        users = sum(r.profile_name == name for r in self.editor.project.runs)
        self.title.setText(
            f"<b>{name}</b>: {len(points)} points, {masked} masked; used by {users} run(s)"
        )
        blocked = self.table.blockSignals(True)
        self.table.setRowCount(len(points))
        for i, (x, fo, out) in enumerate(points.itertuples(index=False)):
            self.table.setItem(i, 0, QTableWidgetItem(um(x)))
            self.table.setItem(i, 1, QTableWidgetItem(pct(fo)))
            check = QTableWidgetItem()
            check.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            check.setCheckState(Qt.CheckState.Unchecked if out else Qt.CheckState.Checked)
            self.table.setItem(i, 2, check)
        self.table.blockSignals(blocked)
        self.error.show_errors(self.errors("mask", "move"))

    def _mask_changed(self, item: QTableWidgetItem) -> None:
        if item.column() != 2:
            return
        i, masked = item.row(), item.checkState() != Qt.CheckState.Checked
        self.apply(lambda e: e.set_point_masked(i, masked))


class AdvancedTab(Tab):
    """The finite-difference scheme, the grid (blank parts chosen at fit time) with the
    fit's grid diagnostics, and the growth stages."""

    GRID = (
        ("dx_um", "dx (µm)", "Node spacing, µm (dx_um); blank: chosen for 1% accuracy"),
        ("dt_s", "dt (s)", "Time step, s (dt_s); blank: chosen to resolve the fitted time"),
        (
            "max_time_days",
            "Longest time (days)",
            "How long the model runs, days (max_time_days); blank: 3 x a quick estimate "
            "of the time",
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        box = QGroupBox("Scheme and grid")
        form = QFormLayout(box)
        self.scheme = QComboBox()
        self.scheme.addItems(list(SCHEMES))
        self.scheme.setToolTip(
            "scheme. implicit (the default): second order, stable at any dt; central and "
            "dipra: the explicit schemes of the paper-era code"
        )
        self.scheme.currentTextChanged.connect(lambda s: self.apply(lambda e: e.set_scheme(s)))
        form.addRow("Scheme", self.scheme)
        self.grid_fields: dict[str, QLineEdit] = {}
        for key, title, tip in self.GRID:
            edit = QLineEdit()
            edit.setPlaceholderText("chosen")
            edit.setToolTip(tip)
            edit.editingFinished.connect(lambda k=key: self._grid_edit(k))
            self.grid_fields[key] = edit
            form.addRow(title, edit)
        self.diagnostics = QLabel()
        self.diagnostics.setWordWrap(True)
        self.diagnostics.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow(self.diagnostics)
        self.grid_error = ErrorLabel()
        form.addRow(self.grid_error)
        layout.addWidget(box)
        stages = QGroupBox("Growth stages")
        inner = QVBoxLayout(stages)
        inner.addWidget(
            _note(
                "After the starting profile, each stage moves the rim out to a new position "
                "(filling the new zone with its Fo) and holds a rim Fo from then on. Growth "
                "stages need dx and dt given."
            )
        )
        self.stages = QTableWidget(0, len(STAGE_FIELDS))
        self.stages.setHorizontalHeaderLabels([STAGE_TITLES[c] for c in STAGE_FIELDS])
        self.stages.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.stages.verticalHeader().setVisible(False)
        self.stages.setToolTip(
            "stage_days, stage_rim_um, stage_zone_fo and stage_rim_fo; Fo in mol%"
        )
        self.stages.setMinimumHeight(110)
        self.stages.itemChanged.connect(self._stage_changed)
        inner.addWidget(self.stages)
        buttons = QHBoxLayout()
        self.add_stage = QPushButton("Add stage")
        self.add_stage.clicked.connect(lambda: self.apply(lambda e: e.add_stage()))
        self.remove_stage = QPushButton("Remove stage")
        self.remove_stage.clicked.connect(
            lambda: self.apply(lambda e: e.remove_stage(self._stage_row()))
        )
        buttons.addWidget(self.add_stage)
        buttons.addWidget(self.remove_stage)
        buttons.addStretch(1)
        inner.addLayout(buttons)
        self.stage_error = ErrorLabel()
        inner.addWidget(self.stage_error)
        layout.addWidget(stages)
        layout.addStretch(1)

    def _stage_row(self) -> int:
        row = self.stages.currentRow()
        return row if row >= 0 else self.stages.rowCount() - 1

    def _grid_edit(self, key: str) -> None:
        if self.loading:
            return
        values = {k: e.text() for k, e in self.grid_fields.items()}
        self.apply(lambda e: e.set_grid(values, key))

    def refresh(self) -> None:
        assert self.editor is not None
        values = self.editor.grid_text()
        self.scheme.setCurrentText(values["scheme"])
        for key, edit in self.grid_fields.items():
            edit.setText(values[key])
        self.show_diagnostics()
        self.grid_error.show_errors(self.errors("scheme", *self.grid_fields))
        blocked = self.stages.blockSignals(True)
        rows = stage_rows(self.editor.run)
        self.stages.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, text in enumerate(stage_text(row)):
                self.stages.setItem(i, j, QTableWidgetItem(text))
        self.stages.blockSignals(blocked)
        self.stage_error.show_errors(self.errors("stage"))

    def show_diagnostics(self) -> None:
        if self.editor is not None:
            self.diagnostics.setText("\n".join(self.editor.grid_lines()))

    def _stage_changed(self, item: QTableWidgetItem) -> None:
        i, column, text = item.row(), STAGE_FIELDS[item.column()], item.text()
        self.apply(lambda e: e.set_stage_cell(i, column, text))


class SidePanel(QTabWidget):
    """The tabs, each in a scroll area; ``edited`` after any accepted edit."""

    edited = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.segments = SegmentsTab()
        self.profile = ProfileDataTab()
        self.conditions = ConditionsTab()
        self.orientation = OrientationTab()
        self.diffusion = DiffusionTab()
        self.uncertainties = UncertaintiesTab()
        self.advanced = AdvancedTab()
        self.titles = {
            "Segments": self.segments,
            "Data": self.profile,
            "Conditions": self.conditions,
            "Orientation": self.orientation,
            "Equation": self.diffusion,
            "Uncertainties": self.uncertainties,
            "Advanced": self.advanced,
        }
        for title, tab in self.titles.items():
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QScrollArea.Shape.NoFrame)
            scroll.setWidget(tab)
            self.addTab(scroll, title)
            tab.edited.connect(self._edited)
        self.setTabToolTip(1, "The profile's points: masking, moving, importing")
        self.setTabToolTip(4, "The Fe-Mg diffusion equation")
        self.load(None)

    def tabs(self) -> list[Tab]:
        return list(self.titles.values())

    def show_tab(self, tab: Tab) -> None:
        self.setCurrentIndex(self.tabs().index(tab))

    @property
    def current_title(self) -> str:
        return self.tabText(self.currentIndex())

    def show_title(self, title: str) -> None:
        if title in self.titles:
            self.setCurrentIndex(list(self.titles).index(title))

    def load(self, editor: RunEditor | None) -> None:
        for tab in self.tabs():
            tab.load(editor)

    def _edited(self) -> None:
        editor = self.segments.editor
        for tab in self.tabs():  # an edit in one tab can change what another shows
            tab.load(editor)
        self.edited.emit()
