"""What to tell the user when the window is closed: the running work and the unsaved
changes, as plain lines and the buttons that apply. No Qt here, so it is tested without a
display; `quit_dialog.QuitDialog` shows it."""

from __future__ import annotations

from dataclasses import dataclass

# Jobs whose model runs cannot be interrupted: quitting waits for the current one.
UNINTERRUPTIBLE = ("fit", "laws")


@dataclass(frozen=True)
class RunningJob:
    kind: str  # "fit", "scan", "mc" or "laws"
    run: str
    progress: str = ""  # the latest text the job reported ("120 model runs", "40/200")


@dataclass(frozen=True)
class QuitPlan:
    running: tuple[str, ...]  # one plain line per running job
    waits: bool  # a running job cannot be interrupted: quitting waits for it
    unsaved: str | None  # the project folder's name when it has unsaved changes
    buttons: tuple[tuple[str, str], ...]  # (action, label); action is save, quit or cancel

    @property
    def needs_dialog(self) -> bool:
        return bool(self.running) or self.unsaved is not None


def job_line(job: RunningJob) -> str:
    extra = f" ({job.progress})" if job.progress else ""
    if job.kind == "fit":
        return f"Fitting {job.run}"
    if job.kind == "scan":
        # The scan also reports its stage names; only the model-run count reads as progress.
        extra = extra if job.progress.endswith("model runs") else ""
        return f"Starting-profile scan of {job.run}{extra}"
    if job.kind == "mc":
        return f"Monte Carlo of {job.run}{extra}"
    if job.kind == "laws":
        return f"Comparing diffusion laws for {job.run}"
    raise ValueError(f"unknown job kind {job.kind!r}")


def quit_plan(jobs: list[RunningJob], dirty_folder: str | None) -> QuitPlan:
    """``dirty_folder`` is the project folder's name when it has unsaved changes."""
    if dirty_folder is not None:
        buttons = (
            ("save", "Save and quit"),
            ("quit", "Quit without saving"),
            ("cancel", "Cancel"),
        )
    elif jobs:
        buttons = (("quit", "Stop and quit"), ("cancel", "Cancel"))
    else:
        buttons = (("quit", "Quit"), ("cancel", "Cancel"))
    return QuitPlan(
        running=tuple(job_line(j) for j in jobs),
        waits=any(j.kind in UNINTERRUPTIBLE for j in jobs),
        unsaved=dirty_folder,
        buttons=buttons,
    )
