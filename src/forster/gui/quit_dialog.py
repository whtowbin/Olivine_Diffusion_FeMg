"""The quit confirmation: what is still running and what is unsaved, in one dialog.
Imported and built only when the window is closed with work at stake."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QPushButton, QVBoxLayout, QWidget

from .quit_plan import QuitPlan

REFRESH_MS = 500
ROLES = {
    "save": QDialogButtonBox.ButtonRole.AcceptRole,
    "quit": QDialogButtonBox.ButtonRole.DestructiveRole,
    "cancel": QDialogButtonBox.ButtonRole.RejectRole,
}


class QuitDialog(QDialog):
    """``plan()`` is asked again every half second, so a job that finishes while the
    dialog is open leaves the list, and a result that makes the project dirty adds the
    unsaved section and its buttons. ``choice`` is "save", "quit" or "cancel" (also
    when the dialog is closed with Escape)."""

    def __init__(self, parent: QWidget, plan: Callable[[], QuitPlan]) -> None:
        super().__init__(parent)
        self.setWindowTitle("Quit FORSTER?")
        self.setModal(True)
        self.setMinimumWidth(420)
        self.plan, self.choice = plan, "cancel"
        self.shown: tuple[tuple[str, str], ...] = ()
        layout = QVBoxLayout(self)
        self.text = QLabel()
        self.text.setWordWrap(True)
        self.text.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self.text)
        self.box = QDialogButtonBox()
        layout.addWidget(self.box)
        self.refresh()
        self.timer = QTimer(self, interval=REFRESH_MS)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()

    def refresh(self) -> None:
        plan = self.plan()
        self.text.setText(self.html(plan))
        if plan.buttons != self.shown:
            self.shown = plan.buttons
            self.box.clear()
            for action, label in plan.buttons:
                button = self.box.addButton(label, ROLES[action])
                button.clicked.connect(lambda _=False, a=action: self.finish(a))
                button.setObjectName(f"quit-{action}")
                if action == "cancel":
                    button.setDefault(True)
                    button.setAutoDefault(True)
                    button.setFocus()
                elif isinstance(button, QPushButton):
                    button.setAutoDefault(False)

    @staticmethod
    def html(plan: QuitPlan) -> str:
        from html import escape

        parts: list[str] = []
        if plan.running:
            items = "".join(f"<li>{escape(line)}</li>" for line in plan.running)
            tail = "Quitting stops it and its result is lost."
            if plan.waits:
                tail += " A fit cannot be interrupted, so quitting waits for the current one."
            parts.append(f"<p><b>Still running</b></p><ul>{items}</ul><p>{tail}</p>")
        if plan.unsaved is not None:
            parts.append(
                f"<p><b>Unsaved changes</b></p><p>The project {escape(plan.unsaved)} "
                "has changes that are not saved.</p>"
            )
        if not parts:
            parts.append("<p>Nothing is running and nothing is unsaved.</p>")
        return "".join(parts)

    def finish(self, action: str) -> None:
        self.choice = action
        self.timer.stop()
        if action == "cancel":
            self.reject()
        else:
            self.accept()
