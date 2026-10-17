"""Dialogs: importing profiles with a column mapping, and copying settings to other runs.

Both only collect choices; the window applies them through `forster.gui.importing` and
`forster.gui.state.copy_settings`, which the tests drive without the dialogs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from .importing import (
    ROLE_TITLES,
    ROLES,
    TO_UM,
    UNITS,
    Column,
    guess_mapping,
    profiles_from_table,
    rim_offset,
    role_columns,
)
from .panels import ErrorLabel, _note
from .state import COPY_GROUPS, number

PREVIEW_ROWS = 8


class ImportDialog(QDialog):
    """Choose which column holds the profile name, the distance and Fo (with their
    units), whether to flip and shift the distances so the rim is at 0, and whether to
    add a run for each new profile. The first guess comes from the headers; the line at
    the bottom says what would be imported."""

    def __init__(self, path: str | Path, table: pd.DataFrame, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Import profiles from {Path(path).name}")
        self.resize(760, 640)
        self.table = table
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"<b>{Path(path).name}</b>: {len(table)} rows"))
        layout.addWidget(self._preview())
        layout.addWidget(self._mapping_box(guess_mapping(table)))
        layout.addWidget(self._options_box())
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.error = ErrorLabel()
        layout.addWidget(self.error)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.update_summary()

    def _preview(self) -> QTableWidget:
        head = self.table.head(PREVIEW_ROWS)
        view = QTableWidget(len(head), len(head.columns))
        view.setHorizontalHeaderLabels([str(c) for c in head.columns])
        view.verticalHeader().setVisible(False)
        view.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        for i, row in enumerate(head.itertuples(index=False)):
            for j, value in enumerate(row):
                view.setItem(i, j, QTableWidgetItem("" if pd.isna(value) else str(value)))
        view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        view.setMaximumHeight(200)
        return view

    def _mapping_box(self, guess: dict[str, Column]) -> QGroupBox:
        box = QGroupBox("Columns")
        grid = QGridLayout(box)
        for j, title in enumerate(("Column", "Holds", "Unit")):
            grid.addWidget(QLabel(f"<b>{title}</b>"), 0, j)
        self.roles: dict[str, QComboBox] = {}
        self.units: dict[str, QComboBox] = {}
        for i, (column, spec) in enumerate(guess.items(), start=1):
            role = QComboBox()
            for key in ROLES:
                role.addItem(ROLE_TITLES[key], key)
            role.setCurrentIndex(ROLES.index(spec.role))
            unit = QComboBox()
            self.roles[column], self.units[column] = role, unit
            self._fill_units(column, spec.unit)
            role.currentIndexChanged.connect(lambda _, c=column: self._role_changed(c))
            unit.currentIndexChanged.connect(lambda _: self.update_summary())
            grid.addWidget(QLabel(column), i, 0)
            grid.addWidget(role, i, 1)
            grid.addWidget(unit, i, 2)
        return box

    def _fill_units(self, column: str, unit: str | None) -> None:
        combo, role = self.units[column], str(self.roles[column].currentData())
        blocked = combo.blockSignals(True)
        combo.clear()
        combo.addItems(list(UNITS.get(role, ())))
        if unit is not None:
            combo.setCurrentText(unit)
        combo.setEnabled(role in UNITS)
        combo.blockSignals(blocked)

    def _role_changed(self, column: str) -> None:
        role = str(self.roles[column].currentData())
        guess = guess_mapping(self.table[[column]]).get(column, Column())
        self._fill_units(column, guess.unit if guess.role == role else None)
        self.update_summary()

    def _options_box(self) -> QGroupBox:
        box = QGroupBox("Options")
        form = QFormLayout(box)
        self.name = QLineEdit("profile")
        self.name.setToolTip("The profile's name when no column holds one")
        form.addRow("Name (no name column)", self.name)
        self.flip = QCheckBox("The rim is at the far end (flip the distances)")
        self.flip.setToolTip("Distances become offset - x, so the far end is the rim")
        self.flip.toggled.connect(lambda _: self.update_summary())
        form.addRow(self.flip)
        row = QHBoxLayout()
        self.offset = QLineEdit("0")
        self.offset.setToolTip(
            "µm subtracted from every distance (flipped: the one that becomes 0)"
        )
        self.offset.editingFinished.connect(self.update_summary)
        rim = QPushButton("Rim at 0")
        rim.setToolTip("The offset that puts the first point (flipped: the last) at 0")
        rim.clicked.connect(self._rim_at_zero)
        row.addWidget(self.offset, 1)
        row.addWidget(rim)
        form.addRow("Offset (µm)", row)
        self.add_runs = QCheckBox("Add a run for each new profile (the selected run's settings)")
        self.add_runs.setChecked(True)
        form.addRow(self.add_runs)
        form.addRow(
            _note(
                "Rows marked Marked_bad = bad or Ignore = yes come in as masked points: kept, "
                "but left out of the fits."
            )
        )
        return box

    def mapping(self) -> dict[str, Column]:
        out = {}
        for column, role in self.roles.items():
            key = str(role.currentData())
            unit = self.units[column].currentText() if key in UNITS else None
            out[column] = Column(key, unit or None)
        return out

    def options(self) -> tuple[str, bool, float]:
        """(name for a table without a name column, flip, offset in um)."""
        offset = number(self.offset.text(), "Offset (µm)", optional=True) or 0.0
        return self.name.text().strip() or "profile", self.flip.isChecked(), offset

    def _rim_at_zero(self) -> None:
        try:
            column = role_columns(self.mapping())["distance"]
        except ValueError as error:
            self.error.show_errors([str(error)])
            return
        unit = self.mapping()[column].unit or "um"
        x = pd.to_numeric(self.table[column], errors="coerce").to_numpy() * TO_UM[unit]
        self.offset.setText(f"{rim_offset(x, self.flip.isChecked()):.12g}")
        self.update_summary()

    def update_summary(self) -> None:
        """What would be imported, or why not (the OK button follows)."""
        ok = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        try:
            name, flip, offset = self.options()
            imported = profiles_from_table(
                self.table, self.mapping(), name=name, flip=flip, offset_um=offset
            )
        except ValueError as error:
            self.summary.setText("")
            self.error.show_errors([str(error)])
            ok.setEnabled(False)
            return
        parts = [
            f"{name} ({len(p.fo)} points, {p.distance_um[0]:.4g}-{p.distance_um[-1]:.4g} µm)"
            for name, p in list(imported.profiles.items())[:6]
        ]
        more = len(imported.profiles) - len(parts)
        text = f"Would add {len(imported.profiles)} profile(s): {'; '.join(parts)}"
        text += f"; and {more} more" if more > 0 else ""
        self.summary.setText(text)
        self.error.show_errors(imported.notes)
        ok.setEnabled(True)


class CopySettingsDialog(QDialog):
    """Choose which settings to copy from the selected run and to which runs."""

    def __init__(self, source: str, runs: list[str], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Copy settings from {source}")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"Copy from <b>{source}</b>:"))
        self.groups: dict[str, QCheckBox] = {}
        for key, (title, fields) in COPY_GROUPS.items():
            box = QCheckBox(title)
            box.setToolTip("Copies " + ", ".join(fields))
            self.groups[key] = box
            layout.addWidget(box)
        layout.addWidget(QLabel("to the runs:"))
        self.runs = QListWidget()
        for name in runs:
            if name == source:
                continue
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.runs.addItem(item)
        layout.addWidget(self.runs)
        layout.addWidget(
            _note(
                "Each run is checked as the table reader would; if one cannot take the "
                "settings, none is changed."
            )
        )
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def choice(self) -> tuple[list[str], list[str]]:
        """(groups, target runs)."""
        groups = [k for k, box in self.groups.items() if box.isChecked()]
        targets = []
        for i in range(self.runs.count()):
            item = self.runs.item(i)
            if item is not None and item.checkState() == Qt.CheckState.Checked:
                targets.append(item.text())
        return groups, targets
