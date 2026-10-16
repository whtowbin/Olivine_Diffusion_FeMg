"""A FORSTER project: a folder of tables a GUI (or a script) opens, edits and saves.

The folder holds the command line's own tables, so ``forster fit profiles.csv
parameters.csv`` runs a saved project unchanged:

- ``profiles.csv``: every profile (template layout: profile, distance_um, fo);
- ``parameters.csv``: one row per run (`forster.io.run_row`);
- ``diffusivities.csv``: the project's own diffusion equations, if any;
- ``project.json``: what only an interactive session needs: the format version, the
  selected run, per-run view settings, and which cached result belongs to which run.

Results are cached in ``.forster/cache/<key>/`` under a key computed from everything
that determines the fit (the run's table row, the profile's numbers, the equations and
the FORSTER version), so a reopened project shows its results at once, and a run whose
settings changed has no result rather than a stale one (`Project.result`). The fit's
warnings are cached with it (`Project.fit_warnings`), and the starting-profile fit
(`forster.io.scan_run`) is cached the same way under its own key (`Project.scan_task`,
`Project.store_scan`, `Project.scan`).

Fits for a worker thread are captured as tasks (`FitTask`, `ScanTask`): they collect
FORSTER's warnings as data (`forster.notes`), never through ``warnings.catch_warnings``,
which is not thread-safe.
``autosave`` writes the same files into ``.forster/autosave/`` without touching the
saved tables.

The class is plain Python with no GUI imports: every front end (and the tests) uses the
same validation, through the table reader.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, fields, replace
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from forster import notes
from forster.diffusivity import ArrheniusDiffusivity, diffusivity_table
from forster.io import (
    METADATA_COLUMNS,
    UNCERTAINTY_COLUMNS,
    Run,
    fit_run,
    read_diffusivities,
    read_profiles,
    read_runs,
    read_table,
    run_row,
    scan_run,
    scan_summary,
    summary_row,
)
from forster.model import Conditions, FitResult, Grid, InitialCondition, Orientation, Profile
from forster.sampling import ICPosterior
from forster.scan import ScanResult, is_valid
from forster.template import write_template
from forster.uncertainty import NoiseModel

FORMAT_VERSION = 1
PROFILES, PARAMETERS, DIFFUSIVITIES, STATE = (
    "profiles.csv",
    "parameters.csv",
    "diffusivities.csv",
    "project.json",
)
WORK = ".forster"  # cache and autosave, inside the project folder
_KEY = re.compile(r"[0-9a-f]{24}")  # a cache key (`Project.result_key`)
# Table columns that do not change the nominal fit (the run's name, notes, and the
# initial-condition scan and Monte Carlo settings), left out of the cache key.
_NOT_IN_FIT = {"run", *METADATA_COLUMNS} | {
    c for _, sd, half in UNCERTAINTY_COLUMNS for c in (sd, half)
}


def _fit_row(run: Run) -> dict[str, Any]:
    """The table row of what determines the fit, with fO2 in Pa and the orientation as
    angles whatever form they were entered in (the same physics, the same key)."""
    plain = replace(run, fO2_form=None, euler_form=None)
    return {
        k: v
        for k, v in run_row(plain).items()
        if k not in _NOT_IN_FIT and not k.startswith(("scan_", "mc_"))
    }


def _scan_row(run: Run) -> dict[str, Any]:
    """The table row of what determines the starting-profile fit: the fit's row plus the
    uncertainty and scan columns (not the Monte Carlo's)."""
    plain = replace(run, fO2_form=None, euler_form=None)
    return {
        k: v
        for k, v in run_row(plain).items()
        if k not in {"run", *METADATA_COLUMNS} and not k.startswith("mc_")
    }


def _text_tuple(value: Any) -> tuple[str, ...]:
    """A list of strings read from a cache file (anything else: empty)."""
    return tuple(str(v) for v in value) if isinstance(value, list) else ()


@dataclass(frozen=True)
class FitTask:
    """One run's fit, captured with its settings so it can run in a worker thread while
    the project is edited: call it, then hand the result to `Project.store`, which files
    it under the settings it was made with (``key``)."""

    name: str
    key: str
    run: Run
    profiles: Mapping[str, Profile]

    def __call__(self) -> FitResult:
        return fit_run(self.run, self.profiles)

    def run_collecting(self) -> FitOutcome:
        """The fit with FORSTER's warnings collected as data (`forster.notes`): safe in a
        worker thread, and the warnings are not issued."""
        with notes.collect() as caught:
            result = fit_run(self.run, self.profiles)
        return FitOutcome(result, notes.messages(caught))


@dataclass(frozen=True)
class FitOutcome:
    """A fit and its warnings (`FitTask.run_collecting`)."""

    result: FitResult
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScanRecord:
    """What a starting-profile fit (`forster.io.scan_run`) leaves for display and reuse:
    its summary.csv columns (`forster.io.scan_summary`), the best starting profile, the
    marginal posterior of time (sampled methods only: ``days``, ``density`` per day) and
    its warnings."""

    summary: dict[str, Any]
    best_initial: InitialCondition
    warnings: tuple[str, ...] = ()
    days: np.ndarray | None = None
    density: np.ndarray | None = None

    @classmethod
    def of(cls, scan: ScanResult | ICPosterior, warnings: tuple[str, ...] = ()) -> ScanRecord:
        days = density = None
        if isinstance(scan, ICPosterior):
            days, density = np.asarray(scan.days), np.asarray(scan.density)
        return cls(scan_summary(scan), scan.best_initial, warnings, days, density)


@dataclass(frozen=True)
class ScanTask:
    """One run's starting-profile fit, captured like `FitTask`: call it in a worker
    thread (``stage`` and ``cancel`` as `forster.io.scan_run`), then hand the record to
    `Project.store_scan`."""

    name: str
    key: str
    run: Run
    profiles: Mapping[str, Profile]
    time_days: float | None

    def __call__(
        self,
        *,
        stage: Callable[[str], None] | None = None,
        cancel: Callable[[], bool] | None = None,
    ) -> ScanRecord:
        with notes.collect() as caught:
            scan = scan_run(
                self.run, self.profiles, time_days=self.time_days, stage=stage, cancel=cancel
            )
        return ScanRecord.of(scan, notes.messages(caught))


def profile_table(profiles: Mapping[str, Profile]) -> pd.DataFrame:
    """Every profile as the template profile table (Fo as a fraction)."""
    frames = [
        pd.DataFrame({"profile": name, "distance_um": p.distance_um, "fo": p.fo})
        for name, p in profiles.items()
    ]
    if not frames:
        return pd.DataFrame({"profile": [], "distance_um": [], "fo": []})
    return pd.concat(frames, ignore_index=True)


def _no_links(*paths: Path) -> None:
    """Refuse to read or write a project file through a symbolic link: a project folder
    someone sent could hold a link that points a read (whose data a later export would
    package) or a write outside the folder."""
    for path in paths:
        if path.is_symlink():
            raise ValueError(
                f"{path} is a symbolic link; FORSTER does not read or write project files "
                "through links"
            )


def _write_text(text: str, path: Path) -> None:
    """Write through a temporary file in the same folder and rename it over ``path``, so
    an interrupted save leaves the old file whole (and a link at ``path`` is refused)."""
    _no_links(path.parent, path)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, suffix=".tmp", delete=False) as tmp:
        tmp.write(text)
    Path(tmp.name).replace(path)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    _write_text(df.to_csv(index=False), path)


@dataclass
class Project:
    """Profiles, runs and session state of one project folder."""

    folder: Path
    profiles: dict[str, Profile] = field(default_factory=dict)
    runs: list[Run] = field(default_factory=list)
    diffusivities: dict[str, ArrheniusDiffusivity] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
    dirty: bool = False  # edited since the last save

    # -- opening and saving ---------------------------------------------------------

    @classmethod
    def open(cls, folder: str | Path) -> Project:
        """Read a project folder (any folder holding the template tables)."""
        folder = Path(folder)
        _no_links(*(folder / f for f in (PROFILES, PARAMETERS, DIFFUSIVITIES, STATE, WORK)))
        if not (folder / PARAMETERS).exists():
            raise FileNotFoundError(f"{folder}: no {PARAMETERS}; not a FORSTER project")
        custom = {}
        if (folder / DIFFUSIVITIES).exists():
            custom = {
                name: d
                for name, d in read_diffusivities(folder / DIFFUSIVITIES).items()
                if isinstance(d, ArrheniusDiffusivity)
            }
        profiles = read_profiles(folder / PROFILES) if (folder / PROFILES).exists() else {}
        runs = read_runs(folder / PARAMETERS, diffusivities=custom)
        state = {}
        if (folder / STATE).exists():
            state = json.loads((folder / STATE).read_text())
            if not isinstance(state, dict):
                raise ValueError(f"{folder / STATE}: not a FORSTER project file")
            # project.json comes from whoever sent the folder: keep only well-formed cache
            # keys, which are used to build paths (and to delete stale entries).
            for kind in ("results", "scans"):
                records = state.get(kind, {})
                state[kind] = {
                    str(k): v
                    for k, v in (records.items() if isinstance(records, dict) else ())
                    if isinstance(v, str) and _KEY.fullmatch(v)
                }
            if state.get("format", FORMAT_VERSION) > FORMAT_VERSION:
                raise ValueError(
                    f"{folder / STATE} was written by a newer FORSTER (format "
                    f"{state['format']}); update FORSTER to open it"
                )
        return cls(folder, profiles, runs, custom, state)

    @classmethod
    def new(cls, folder: str | Path, *, example: bool = True) -> Project:
        """A new project in an empty or new folder; ``example`` starts from the packaged
        worked example (`forster.write_template`), otherwise empty."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        if any(folder.iterdir()):
            raise FileExistsError(f"{folder} is not empty")
        if example:
            write_template(folder)
            (folder / "COLUMNS.md").unlink(missing_ok=True)
            return cls.open(folder)
        project = cls(folder)
        project.save()
        return project

    def _write(self, folder: Path) -> None:
        _no_links(folder)
        folder.mkdir(parents=True, exist_ok=True)
        _write_csv(profile_table(self.profiles), folder / PROFILES)
        _write_csv(pd.DataFrame([run_row(r) for r in self.runs]), folder / PARAMETERS)
        if self.diffusivities:
            _write_csv(diffusivity_table(self.diffusivities.values()), folder / DIFFUSIVITIES)
        state = {**self.state, "format": FORMAT_VERSION, "forster_version": version("forster")}
        _write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", folder / STATE)

    def save(self, folder: str | Path | None = None) -> None:
        """Write the tables and project.json (to ``folder``: save as), and clear the
        autosave."""
        if folder is not None:
            self.folder = Path(folder)
        self._write(self.folder)
        shutil.rmtree(self.folder / WORK / "autosave", ignore_errors=True)
        self.dirty = False

    def autosave(self) -> None:
        """Write everything to ``.forster/autosave/``, leaving the saved tables alone."""
        self._write(self.folder / WORK / "autosave")

    @property
    def has_autosave(self) -> bool:
        """An autosave newer than the saved tables (offer to restore it)."""
        auto = self.folder / WORK / "autosave" / PARAMETERS
        saved = self.folder / PARAMETERS
        return auto.exists() and (
            not saved.exists() or auto.stat().st_mtime > saved.stat().st_mtime
        )

    def restore_autosave(self) -> Project:
        """The project as last autosaved (to be saved over the tables by `save`)."""
        _no_links(self.folder / WORK, self.folder / WORK / "autosave")
        restored = Project.open(self.folder / WORK / "autosave")
        restored.folder, restored.dirty = self.folder, True
        return restored

    def export_zip(self, path: str | Path) -> Path:
        """The project's tables and project.json as one zip file, for sharing. They are
        written afresh from memory, not copied from the folder, so nothing else in it
        (the cache, other files, or a link pointing elsewhere) ends up in the zip."""
        path = Path(path)
        with tempfile.TemporaryDirectory() as tmp:
            self._write(Path(tmp) / self.folder.name)
            archive = shutil.make_archive(str(path.with_suffix("")), "zip", tmp)
        return Path(archive)

    # -- runs -----------------------------------------------------------------------

    @property
    def run_names(self) -> list[str]:
        return [r.name for r in self.runs]

    def run(self, name: str) -> Run:
        for r in self.runs:
            if r.name == name:
                return r
        raise KeyError(f"no run {name!r}; runs are {', '.join(self.run_names)}")

    def _index(self, name: str) -> int:
        return self.run_names.index(self.run(name).name)

    def _check_new_name(self, name: str) -> None:
        if not name.strip():
            raise ValueError("a run needs a name")
        if name in self.run_names:
            raise ValueError(f"there is already a run called {name!r}")

    def add_run(self, run: Run) -> Run:
        self._check_new_name(run.name)
        if run.profile_name not in self.profiles:
            raise ValueError(f"run {run.name!r}: no profile {run.profile_name!r} in the project")
        self.runs.append(run)
        self.dirty = True
        return run

    def duplicate_run(self, name: str, new_name: str | None = None) -> Run:
        """A copy of the run right after it (for another equation or starting profile)."""
        new_name = new_name or self._free_name(f"{name} copy")
        self._check_new_name(new_name)
        copy = replace(self.run(name), name=new_name)
        self.runs.insert(self._index(name) + 1, copy)
        self.dirty = True
        return copy

    def _free_name(self, base: str) -> str:
        name, i = base, 2
        while name in self.run_names:
            name, i = f"{base} {i}", i + 1
        return name

    def remove_run(self, name: str) -> None:
        del self.runs[self._index(name)]
        for key in ("views", "results", "scans"):
            self.state.get(key, {}).pop(name, None)
        self.dirty = True

    def rename_run(self, name: str, new_name: str) -> Run:
        self._check_new_name(new_name)
        i = self._index(name)
        self.runs[i] = replace(self.runs[i], name=new_name)
        for key in ("views", "results", "scans"):
            if name in self.state.get(key, {}):
                self.state[key][new_name] = self.state[key].pop(name)
        if self.state.get("selected_run") == name:
            self.state["selected_run"] = new_name
        self.dirty = True
        return self.runs[i]

    def update_run(self, name: str, **changes: Any) -> Run:
        """The run with ``changes`` (Run fields; build new `Conditions`,
        `InitialCondition` ... for nested edits). The edited run is checked by writing it
        as a table row and reading it back, so the GUI accepts exactly what the
        command line does; a ValueError names the problem and nothing changes."""
        i = self._index(name)
        candidate = _reconcile(self.runs[i], changes)
        if "name" in changes and changes["name"] != name:
            self._check_new_name(changes["name"])
        self.validate(candidate)
        self.runs[i] = candidate
        self.dirty = True
        return candidate

    def copy_settings(self, source: str, targets: Iterable[str], fields: Iterable[str]) -> None:
        """Copy the named Run fields (conditions, orientation, diffusivity, ...) from one
        run to others; each target is validated, and none changes if one fails."""
        names = set(fields)
        # The form fO2 and the orientation were entered in travels with them.
        names |= {"fO2_form"} if "conditions" in names else set()
        names |= {"euler_form"} if "orientation" in names else set()
        values = {f: getattr(self.run(source), f) for f in names}
        updated = {t: _reconcile(self.run(t), values) for t in targets}
        for run in updated.values():
            self.validate(run)
        for name, run in updated.items():
            self.runs[self._index(name)] = run
        self.dirty = True

    def validate(self, run: Run) -> None:
        """Raise ValueError if ``run`` is not a valid run: steps inside the model domain
        and in order, and the run reads back from a table as itself."""
        if not is_valid(run.initial, run.initial.domain_um):
            lo, hi = run.initial.domain_um
            raise ValueError(
                f"run {run.name!r}: steps {list(run.initial.steps_um)} must lie inside the "
                f"model domain ({lo:g}-{hi:g} um), in order"
            )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / PARAMETERS
            pd.DataFrame([run_row(run)]).to_csv(path, index=False)
            with notes.collect():  # dropped; thread-safe, unlike catch_warnings
                (back,) = read_runs(path, diffusivities=self.diffusivities)
        if back != run:
            raise ValueError(f"run {run.name!r} does not survive a save and reopen")

    # -- profiles -------------------------------------------------------------------

    def import_profiles(
        self,
        path: str | Path,
        *,
        sheet: str | int = 0,
        name_column: str | None = None,
        distance_column: str | None = None,
        fo_column: str | None = None,
        fo_in_percent: bool | None = None,
    ) -> list[str]:
        """Add the profiles of a CSV or Excel table (one row per analysis point; several
        profiles split by the name column). Columns default to the template's or the lab
        layout's (`forster.io.read_profiles`). With ``fo_in_percent`` None and explicit
        columns, Fo above 1.5 anywhere means percent. A name already in the project is
        an error."""

        def read(percent: bool | None) -> dict[str, Profile]:
            return read_profiles(
                path,
                sheet=sheet,
                name_column=name_column,
                distance_column=distance_column,
                fo_column=fo_column,
                fo_in_percent=percent,
            )

        if fo_in_percent is None and fo_column is not None:
            raw = pd.to_numeric(read_table(path, sheet)[fo_column], errors="coerce")
            fo_in_percent = bool(np.nanmax(raw) > 1.5)
        new = read(fo_in_percent)
        clash = sorted(set(new) & set(self.profiles))
        if clash:
            raise ValueError(f"profiles already in the project: {clash}; rename them first")
        self.profiles.update(new)
        self.dirty = True
        return list(new)

    def remove_profile(self, name: str) -> None:
        users = [r.name for r in self.runs if r.profile_name == name]
        if users:
            raise ValueError(f"profile {name!r} is used by runs {users}; remove them first")
        del self.profiles[name]
        self.dirty = True

    # -- results --------------------------------------------------------------------

    def result_key(self, name: str) -> str:
        """Hash of everything that determines the run's nominal fit (not its name, notes,
        scan or Monte Carlo settings)."""
        return self._key(self.run(name), _fit_row(self.run(name)))

    def scan_key(self, name: str, time_days: float | None = None) -> str:
        """Hash of everything that determines the run's starting-profile fit: the fit's
        settings, the uncertainties and scan settings, and ``time_days`` (the nominal
        fit's time, which sizes the scan grid)."""
        run = self.run(name)
        row = {**_scan_row(run), "_scan_time_days": time_days}
        return self._key(run, row)

    def _key(self, run: Run, row: Mapping[str, Any]) -> str:
        profile = self.profiles[run.profile_name]
        h = hashlib.sha256()
        h.update(json.dumps(row, sort_keys=True, default=str).encode())
        h.update(np.asarray(profile.distance_um, dtype=float).tobytes())
        h.update(np.asarray(profile.fo, dtype=float).tobytes())
        if isinstance(run.diffusivity, ArrheniusDiffusivity):
            h.update(diffusivity_table([run.diffusivity]).to_csv(index=False).encode())
        h.update(version("forster").encode())
        return h.hexdigest()[:24]

    def _cache(self, key: str) -> Path:
        if not _KEY.fullmatch(key):
            raise ValueError(f"not a cache key: {key!r}")
        return self.folder / WORK / "cache" / key

    def fit_task(self, name: str) -> FitTask:
        """The run's fit as a task for a worker thread (`FitTask`)."""
        return FitTask(name, self.result_key(name), self.run(name), dict(self.profiles))

    def store(self, task: FitTask, result: FitResult, warnings: Iterable[str] = ()) -> bool:
        """Cache a task's result (and its warnings, `FitOutcome`) under the settings it
        was made with. Returns whether they are still the run's settings (if not, the
        result is kept for them but the run shows no result). The run's previous cache
        entry is removed unless another run uses it, so the cache holds one entry per
        run."""
        _store_fit(result, self._cache(task.key), tuple(warnings))
        current = task.name in self.run_names and self.result_key(task.name) == task.key
        if current:
            self._record("results", task.name, task.key)
        return current

    def _record(self, kind: str, name: str, key: str) -> None:
        """Note ``key`` as the run's cache entry of ``kind`` ("results" or "scans") and
        remove its previous entry unless another run uses it."""
        records = self.state.setdefault(kind, {})
        old = records.get(name)
        records[name] = key
        in_use = {*self.state.get("results", {}).values(), *self.state.get("scans", {}).values()}
        if isinstance(old, str) and _KEY.fullmatch(old) and old not in in_use:
            stale = self._cache(old)
            if (
                not stale.is_symlink()
                and stale.resolve().parent == (self.folder / WORK / "cache").resolve()
            ):
                shutil.rmtree(stale, ignore_errors=True)

    def fit_warnings(self, name: str) -> tuple[str, ...] | None:
        """The warnings of the cached fit for the run's current settings (None: no
        cached fit)."""
        path = self._cache(self.result_key(name)) / "warnings.json"
        if not (path.parent / "fit.json").exists():
            return None
        if not path.exists():
            return ()
        _no_links(path.parents[2], path.parents[1], path.parent, path)
        return _text_tuple(json.loads(path.read_text()))

    # -- the starting-profile fit ---------------------------------------------------

    def scan_task(self, name: str) -> ScanTask:
        """The run's starting-profile fit as a task for a worker thread (`ScanTask`),
        sized by the cached nominal fit's time when there is one."""
        cached = self.summary(name)
        time_days = float(cached["time_days"]) if cached else None
        return ScanTask(
            name,
            self.scan_key(name, time_days),
            self.run(name),
            dict(self.profiles),
            time_days,
        )

    def store_scan(self, task: ScanTask, record: ScanRecord) -> bool:
        """Cache a starting-profile fit under the settings it was made with; whether they
        are still the run's (as `store`)."""
        _store_scan(record, self._cache(task.key))
        current = task.name in self.run_names and self.scan_task(task.name).key == task.key
        if current:
            self._record("scans", task.name, task.key)
        return current

    def scan(self, name: str) -> ScanRecord | None:
        """The cached starting-profile fit for the run's current settings, or None."""
        folder = self._cache(self.scan_task(name).key)
        if not (folder / "scan.json").exists():
            return None
        _no_links(folder.parents[1], folder.parent, folder, folder / "scan.json")
        return _load_scan(folder)

    def fit(self, name: str) -> FitResult:
        """Fit the run (`forster.io.fit_run`), or return the cached fit of the same
        settings."""
        cached = self.result(name)
        if cached is not None:
            return cached
        task = self.fit_task(name)
        outcome = task.run_collecting()
        for message in outcome.warnings:  # issued here, as fit_run issues them
            notes.warn(message, stacklevel=2)
        self.store(task, outcome.result, outcome.warnings)
        return outcome.result

    @property
    def selected_run(self) -> str | None:
        name = self.state.get("selected_run")
        return name if name in self.run_names else None

    @selected_run.setter
    def selected_run(self, name: str | None) -> None:
        if name is not None:
            self.run(name)  # raises for an unknown run
        self.state["selected_run"] = name

    def view(self, name: str) -> dict[str, Any]:
        """The run's view settings in project.json (plot limits, open tabs, ...), to read
        and update in place."""
        self.run(name)
        return self.state.setdefault("views", {}).setdefault(name, {})

    def result(self, name: str) -> FitResult | None:
        """The cached fit for the run's current settings, or None (never fitted, or its
        settings changed since)."""
        folder = self._cache(self.result_key(name))
        if not (folder / "fit.json").exists():
            return None
        _no_links(folder.parents[1], folder.parent, folder, folder / "fit.json", folder / "fit.npz")
        return _load_fit(folder, self.run(name))

    def summary(self, name: str) -> dict[str, Any] | None:
        """The cached fit's summary row, or None (cheap: no unpickling)."""
        path = self._cache(self.result_key(name)) / "summary.json"
        if not path.exists():
            return None
        _no_links(path.parents[2], path.parents[1], path.parent, path)
        return json.loads(path.read_text())

    def is_stale(self, name: str) -> bool:
        """The run was fitted before (project.json says so) but its settings have changed."""
        recorded = self.state.get("results", {}).get(name)
        return recorded is not None and recorded != self.result_key(name)


def _reconcile(run: Run, changes: Mapping[str, Any]) -> Run:
    """``run`` with ``changes``, keeping fO2 and the orientation consistent with the form
    they are entered in. A new ``conditions`` whose fO2 differs, without a new
    ``fO2_form``, switches the run to fO2 in Pa; otherwise a run entered relative to a
    buffer gets its fO2 from the buffer at its (new) T and P. Likewise a new
    ``orientation`` without a new ``euler_form`` drops the Euler form, and a run entered
    as Euler angles gets its orientation from them."""
    changes = dict(changes)
    new_c: Conditions | None = changes.get("conditions")
    if new_c is not None and "fO2_form" not in changes and new_c.fO2_Pa != run.conditions.fO2_Pa:
        changes["fO2_form"] = None
    if "orientation" in changes and "euler_form" not in changes:
        changes["euler_form"] = None
    out = replace(run, **changes)
    if out.fO2_form is not None:
        c, (buffer, delta) = out.conditions, out.fO2_form
        out = replace(out, conditions=Conditions.from_buffer(c.T_C, c.P_Pa, delta, buffer))
    if out.euler_form is not None:
        out = replace(out, orientation=Orientation.from_euler(*out.euler_form))
    return out


# The fit cache holds no pickles (a shared project folder must not run code when
# opened): arrays in fit.npz, read with allow_pickle=False; the rest in fit.json; and
# FitResult.inputs rebuilt from the run with the grid the fit used.
_INPUT_FIELDS = ("inputs",)


def _store_fit(result: FitResult, folder: Path, warnings: tuple[str, ...] = ()) -> None:
    _no_links(folder.parents[1], folder.parent, folder)  # .forster, cache, the entry
    folder.mkdir(parents=True, exist_ok=True)
    arrays, scalars = {}, {}
    for f in fields(result):
        if f.name in _INPUT_FIELDS:
            continue
        value = getattr(result, f.name)
        if isinstance(value, np.ndarray):
            arrays[f.name] = value
        elif isinstance(value, NoiseModel):
            scalars[f.name] = {
                "noise": [value.white_sigma, value.correlated_sigma, value.correlation_length_um]
            }
        else:
            scalars[f.name] = list(value) if isinstance(value, tuple) else value
    grid: Grid = result.inputs["grid"]
    scalars["grid"] = [grid.dx_um, grid.dt_s, grid.max_time_days]
    scalars["stop_early"] = result.inputs["stop_early"]
    _no_links(folder / "fit.npz")
    with tempfile.NamedTemporaryFile(dir=folder, suffix=".npz", delete=False) as tmp:
        np.savez(tmp, **arrays)  # uncompressed: compressing took most of a drag's refit
    Path(tmp.name).replace(folder / "fit.npz")
    _write_text(json.dumps(summary_row(result), default=float), folder / "summary.json")
    _write_text(json.dumps(list(warnings)), folder / "warnings.json")
    _write_text(json.dumps(scalars, default=float), folder / "fit.json")  # last: marks complete


def _load_fit(folder: Path, run: Run) -> FitResult:
    scalars = json.loads((folder / "fit.json").read_text())
    with np.load(folder / "fit.npz", allow_pickle=False) as npz:
        values: dict[str, Any] = {k: npz[k] for k in npz.files}
    grid = Grid(*scalars.pop("grid"))
    stop_early = scalars.pop("stop_early")
    for name, value in scalars.items():
        if isinstance(value, dict) and "noise" in value:
            value = NoiseModel(*value["noise"])
        elif name in ("indices", "indices2", "curve_rows") and value is not None:
            value = tuple(value)
        values[name] = value
    values["inputs"] = {
        "conditions": run.conditions,
        "orientation": run.orientation,
        "initial": run.initial,
        "grid": grid,
        "kriging": run.kriging,
        "diffusivity": run.diffusivity,
        "scheme": run.scheme,
        "uncertainty": run.uncertainty,
        "stop_early": stop_early,
        "stages": tuple(run.stages),
        "rim_fo": run.rim_fo,
    }
    return FitResult(**values)


def _initial_json(initial: InitialCondition) -> dict[str, Any]:
    return {
        "domain_um": list(initial.domain_um),
        "steps_um": list(initial.steps_um),
        "segment_fo": list(initial.segment_fo),
        "segment_names": list(initial.segment_names),
        "centre_um": initial.centre_um,
    }


def _initial_from_json(value: Mapping[str, Any]) -> InitialCondition:
    low, high = (float(v) for v in value["domain_um"])
    return InitialCondition(
        (low, high),
        steps_um=[float(v) for v in value["steps_um"]],
        segment_fo=[float(v) for v in value["segment_fo"]],
        segment_names=[str(v) for v in value["segment_names"]],
        centre_um=None if value.get("centre_um") is None else float(value["centre_um"]),
    )


def _store_scan(record: ScanRecord, folder: Path) -> None:
    """A starting-profile fit as scan.json (and scan.npz for the posterior of time), no
    pickles, like `_store_fit`."""
    _no_links(folder.parents[1], folder.parent, folder)
    folder.mkdir(parents=True, exist_ok=True)
    if record.days is not None and record.density is not None:
        _no_links(folder / "scan.npz")
        with tempfile.NamedTemporaryFile(dir=folder, suffix=".npz", delete=False) as tmp:
            np.savez(tmp, days=record.days, density=record.density)
        Path(tmp.name).replace(folder / "scan.npz")
    data = {
        "summary": record.summary,
        "best_initial": _initial_json(record.best_initial),
        "warnings": list(record.warnings),
        "posterior": record.days is not None,
    }
    _write_text(json.dumps(data, default=float), folder / "scan.json")  # last: marks complete


def _load_scan(folder: Path) -> ScanRecord:
    data = json.loads((folder / "scan.json").read_text())
    days = density = None
    if data.get("posterior"):
        _no_links(folder / "scan.npz")
        with np.load(folder / "scan.npz", allow_pickle=False) as npz:
            days, density = npz["days"], npz["density"]
    return ScanRecord(
        dict(data["summary"]),
        _initial_from_json(data["best_initial"]),
        _text_tuple(data.get("warnings")),
        days,
        density,
    )
