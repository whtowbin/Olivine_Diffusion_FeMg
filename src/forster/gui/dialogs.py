"""Dialogs: importing profiles with a column mapping, and copying settings to other runs.

Both only collect choices; the window applies them through `forster.gui.importing` (on
top of `forster.imports`) and `forster.gui.state.copy_settings`, which the tests drive
without the dialogs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
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
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from forster.imports import (
    CONCENTRATION_UNITS,
    ELEMENT_MASS,
    FO_UNITS,
    OXIDE_MASS,
    POSITION_UNITS,
    ROLES,
    ColumnMap,
    ImportTemplate,
    TemplateMismatch,
    check_columns,
    guess,
)

from .importing import Imported, import_table, position_um, rim_offset
from .panels import ErrorLabel, _note
from .state import COPY_GROUPS, number

PREVIEW_ROWS = 8
FLAGGED_SHOWN = 12  # flagged points listed one by one in the dialog
ROLE_TITLES = {
    "ignore": "(ignore)",
    "profile": "Profile name",
    "position": "Distance",
    "fo": "Fo",
    "oxide": "Oxide",
    "element": "Element",
    "total": "Total (wt%)",
    "stage_x": "Stage X",
    "stage_y": "Stage Y",
    "exclude": "Exclude marked rows",
}
ROLE_ORDER = ("ignore", *[r for r in ROLES if r != "ignore"])
UNITS = {
    "position": tuple(POSITION_UNITS),
    "fo": FO_UNITS,
    "oxide": tuple(CONCENTRATION_UNITS),
    "element": tuple(CONCENTRATION_UNITS),
}
QUANTITIES = {"oxide": tuple(OXIDE_MASS), "element": tuple(ELEMENT_MASS)}
DEFAULT_UNIT = {"position": "um", "oxide": "wt%"}  # when a role is chosen by hand


class ColumnRow:
    """One column's role, quantity (which oxide or element) and unit, as combo boxes;
    the guess's note is their tooltip."""

    def __init__(self, cmap: ColumnMap, changed: Callable[[], None]) -> None:
        self.column, self.note = cmap.column, cmap.note
        self.label = QLabel(cmap.column)
        self.role, self.quantity, self.unit = QComboBox(), QComboBox(), QComboBox()
        for key in ROLE_ORDER:
            self.role.addItem(ROLE_TITLES[key], key)
        self.set(cmap)
        self.role.currentIndexChanged.connect(lambda _: self._role_changed(changed))
        self.quantity.currentIndexChanged.connect(lambda _: changed())
        self.unit.currentIndexChanged.connect(lambda _: changed())

    def widgets(self) -> tuple[QWidget, ...]:
        return self.label, self.role, self.quantity, self.unit

    def set(self, cmap: ColumnMap) -> None:
        """Show ``cmap`` (signals blocked: the caller updates the summary once)."""
        self.note = cmap.note
        tip = cmap.note or "chosen by hand or from an import template"
        for widget in self.widgets():
            widget.setToolTip(f"{cmap.column}: {tip}")
        blocked = self.role.blockSignals(True)
        self.role.setCurrentIndex(max(0, self.role.findData(cmap.role)))
        self.role.blockSignals(blocked)
        _fill(self.quantity, QUANTITIES.get(cmap.role, ()), cmap.quantity)
        _fill(self.unit, UNITS.get(cmap.role, ()), cmap.unit)

    def _role_changed(self, changed: Callable[[], None]) -> None:
        role = str(self.role.currentData())
        _fill(self.quantity, QUANTITIES.get(role, ()), self.quantity.currentData())
        _fill(self.unit, UNITS.get(role, ()), self.unit.currentData() or DEFAULT_UNIT.get(role))
        changed()

    def column_map(self) -> ColumnMap:
        role = str(self.role.currentData())
        quantity = self.quantity.currentData() if role in QUANTITIES else None
        unit = self.unit.currentData() if role in UNITS else None
        return ColumnMap(self.column, role, quantity, unit, self.note)


def _fill(combo: QComboBox, items: tuple[str, ...], current: str | None) -> None:
    """The choices for a role ("(choose)" first, so a missing one is a problem the
    dialog names), ``current`` selected; disabled when the role has none."""
    blocked = combo.blockSignals(True)
    combo.clear()
    if items:
        combo.addItem("(choose)", None)
        for item in items:
            combo.addItem(item, item)
        combo.setCurrentIndex(max(0, combo.findData(current)))
    combo.setEnabled(bool(items))
    combo.blockSignals(blocked)


class ImportDialog(QDialog):
    """Say what each column holds (`forster.imports`: its role, which oxide or element,
    the unit; the first guess from `forster.imports.guess`, each column's reason in its
    tooltip), the window of analytical totals outside which points are flagged (kept),
    whether to flip and shift the distances so the rim is at 0, and whether to add a run
    for each new profile. The mapping can be saved as an import template and loaded for
    the next file. OK stays disabled until the mapping has no problems; the lines at the
    bottom say what would be imported."""

    def __init__(self, path: str | Path, table: pd.DataFrame, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Import profiles from {Path(path).name}")
        self.resize(840, 860)
        self.path, self.table = Path(path), table
        self.guessed = guess(table)
        self.template_name = ""  # the saved or loaded import template's name
        self.status = ""  # the last save or load, or why a template did not load
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"<b>{self.path.name}</b>: {len(table)} rows"))
        layout.addWidget(self._preview())
        layout.addWidget(self._mapping_box(), 1)
        layout.addWidget(self._options_box())
        layout.addLayout(self._template_buttons())
        self.notes = _note("")
        layout.addWidget(self.notes)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.flagged = QLabel()
        self.flagged.setWordWrap(True)
        self.flagged.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.flagged)
        self.error = ErrorLabel()
        layout.addWidget(self.error)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.update_summary()

    # -- layout -----------------------------------------------------------------------

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
        view.setMaximumHeight(140)
        return view

    def _mapping_box(self) -> QGroupBox:
        box = QGroupBox("Columns (hover over one to see why it was guessed)")
        inner = QWidget()
        grid = QGridLayout(inner)
        for j, title in enumerate(("Column", "Holds", "Oxide or element", "Unit")):
            grid.addWidget(QLabel(f"<b>{title}</b>"), 0, j)
        self.rows: dict[str, ColumnRow] = {}
        for i, cmap in enumerate(self.guessed.columns, start=1):
            row = ColumnRow(cmap, self.update_summary)
            self.rows[cmap.column] = row
            for j, widget in enumerate(row.widgets()):
                grid.addWidget(widget, i, j)
        grid.setRowStretch(len(self.rows) + 1, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        scroll.setMinimumHeight(200)
        QVBoxLayout(box).addWidget(scroll)
        return box

    def _options_box(self) -> QGroupBox:
        box = QGroupBox("Options")
        form = QFormLayout(box)
        window = QHBoxLayout()
        lo, hi = self.guessed.total_window
        self.total_low, self.total_high = QLineEdit(f"{lo:g}"), QLineEdit(f"{hi:g}")
        for edit in (self.total_low, self.total_high):
            edit.setToolTip(
                "Points whose analytical total (wt%) lies outside this window are flagged: "
                "kept in the profile and listed below, not dropped"
            )
            edit.editingFinished.connect(self.update_summary)
        window.addWidget(self.total_low)
        window.addWidget(QLabel("to"))
        window.addWidget(self.total_high)
        form.addRow("Total window (wt%)", window)
        self.name = QLineEdit(self.path.stem)
        self.name.setToolTip("The profile's name when no column holds one")
        self.name.editingFinished.connect(self.update_summary)
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
                "Fo comes from an Fo column, or in moles from MgO and FeO (all iron as "
                "FeO) or from Mg and Fe. Rows marked in an exclusion column (Marked_bad = "
                "bad, Ignore = yes) come in as masked points: kept, but left out of the fits."
            )
        )
        return box

    def _template_buttons(self) -> QHBoxLayout:
        row = QHBoxLayout()
        save = QPushButton("Save mapping as import template…")
        save.setToolTip("A JSON file that maps the columns of files laid out like this one")
        save.clicked.connect(self._choose_save)
        load = QPushButton("Load import template…")
        load.setToolTip("Use a saved mapping; a file with other columns is refused")
        load.clicked.connect(self._choose_load)
        row.addWidget(save)
        row.addWidget(load)
        row.addStretch(1)
        return row

    # -- the mapping ------------------------------------------------------------------

    def template(self) -> ImportTemplate:
        """The mapping shown, as an import template (a ValueError for a total window
        that is not a number; `ImportTemplate.problems` lists the rest)."""
        lo = number(self.total_low.text(), "Total window, low")
        hi = number(self.total_high.text(), "Total window, high")
        assert lo is not None and hi is not None
        columns = tuple(row.column_map() for row in self.rows.values())
        return ImportTemplate(columns, (lo, hi), self.template_name, self.guessed.notes)

    def set_template(self, template: ImportTemplate) -> None:
        by_column = {c.column: c for c in template.columns}
        for column, row in self.rows.items():
            row.set(by_column.get(column, ColumnMap(column, "ignore")))
        lo, hi = template.total_window
        self.total_low.setText(f"{lo:g}")
        self.total_high.setText(f"{hi:g}")
        self.template_name = template.name
        self.update_summary()

    def options(self) -> tuple[str, bool, float]:
        """(name for a table without a name column, flip, offset in um)."""
        offset = number(self.offset.text(), "Offset (µm)", optional=True) or 0.0
        return self.name.text().strip() or self.path.stem, self.flip.isChecked(), offset

    def save_template(self, path: str | Path) -> bool:
        """Save the mapping (complete, or refused with its problems) as an import
        template."""
        try:
            template = self.template()
            template.check()
            saved = replace(template, name=Path(path).name).save(path)
        except (OSError, ValueError) as error:
            self.status = f"Not saved: {error}"
            self.update_summary()
            return False
        self.template_name = saved.name
        self.status = f"Saved the mapping as the import template {saved.name}."
        self.update_summary()
        return True

    def load_template(self, path: str | Path) -> bool:
        """Show a saved import template's mapping if it matches this file's columns;
        otherwise say why not (`forster.imports.TemplateMismatch`) and leave the
        mapping shown as it was."""
        try:
            template = ImportTemplate.load(path)
            check_columns(self.table.columns, template)
        except TemplateMismatch as error:
            self.status = f"Not loaded: {error}"
            self.update_summary()
            return False
        except (OSError, ValueError) as error:
            self.status = f"Cannot load {Path(path).name}: {error}"
            self.update_summary()
            return False
        self.status = f"Loaded the import template {template.name}."
        self.set_template(template)
        return True

    def _choose_save(self) -> None:
        start = self.path.with_suffix(".import.json")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save import template", str(start), "Import templates (*.json)"
        )
        if path:
            self.save_template(path)

    def _choose_load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load import template", str(self.path.parent), "Import templates (*.json)"
        )
        if path:
            self.load_template(path)

    def _rim_at_zero(self) -> None:
        try:
            x = position_um(self.table, self.template())
        except ValueError as error:
            self.error.show_errors([str(error)])
            return
        self.offset.setText(f"{rim_offset(x, self.flip.isChecked()):.12g}")
        self.update_summary()

    # -- what would be imported -------------------------------------------------------

    def preview(self) -> Imported:
        """What OK would import (a ValueError says why not)."""
        template = self.template()
        template.check()
        name, flip, offset = self.options()
        return import_table(self.table, template, name=name, flip=flip, offset_um=offset)

    def update_summary(self) -> None:
        """The notes, what would be imported and the flagged points, or why not (the OK
        button follows)."""
        ok = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        notes = [self.status] if self.status else []
        notes += [] if self.template_name else list(self.guessed.notes)
        try:
            imported = self.preview()
        except ValueError as error:
            self._show(notes, "", "", [str(error)])
            ok.setEnabled(False)
            return
        parts = [
            f"{key} ({len(p.fo)} points, {p.distance_um[0]:.4g}-{p.distance_um[-1]:.4g} µm)"
            for key, p in list(imported.profiles.items())[:6]
        ]
        more = len(imported.profiles) - len(parts)
        text = f"Would add {len(imported.profiles)} profile(s): {'; '.join(parts)}"
        text += f"; and {more} more" if more > 0 else ""
        # The first note is `forster.imports.apply`'s count, which ``text`` gives.
        self._show(notes + imported.notes[1:], text, flagged_text(imported), [])
        ok.setEnabled(True)

    def _show(self, notes: list[str], summary: str, flagged: str, errors: list[str]) -> None:
        self.notes.setText("\n".join(notes))
        self.notes.setVisible(bool(notes))
        self.summary.setText(summary)
        self.flagged.setText(flagged)
        self.flagged.setVisible(bool(flagged))
        self.error.show_errors(errors)


def flagged_text(imported: Imported) -> str:
    """The flagged points (total outside the window), one line each, for the dialog."""
    rows = imported.flagged
    if rows.empty:
        return ""
    lo, hi = imported.result.template.total_window
    lines = [
        f"{profile}: {x:.4g} µm, Fo {100 * fo:.2f} mol%, total {total:.2f} wt%"
        for profile, x, fo, total in zip(
            rows["profile"][:FLAGGED_SHOWN],
            rows["distance_um"][:FLAGGED_SHOWN],
            rows["fo"][:FLAGGED_SHOWN],
            rows["total_wt%"][:FLAGGED_SHOWN],
            strict=True,
        )
    ]
    more = len(rows) - len(lines)
    head = f"Flagged (total outside {lo:g}-{hi:g} wt%), kept in the profiles:"
    return "\n".join([head, *lines] + ([f"and {more} more"] if more > 0 else []))


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
