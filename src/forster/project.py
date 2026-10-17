"""A FORSTER project: a folder of tables a GUI (or a script) opens, edits and saves.

The folder holds the command line's own tables, so ``forster fit profiles.csv
parameters.csv`` runs a saved project unchanged:

- ``profiles.csv``: every profile (template layout: profile, distance_um, fo), with a
  ``Marked_bad`` column ("bad") on the points masked out of the fits when there are any:
  the table reader drops those rows, so ``forster fit`` leaves them out too;
- ``parameters.csv``: one row per run (`forster.io.run_row`);
- ``diffusivities.csv``: the project's own diffusion equations, if any;
- ``project.json``: what only an interactive session needs: the format version, the
  selected run, per-run view settings, and which cached result belongs to which run.

Results are cached in ``.forster/cache/<key>/`` under a key computed from everything
that determines the fit (the run's table row, the profile's numbers, the equations and
the FORSTER version), so a reopened project shows its results at once, and a run whose
settings changed has no result rather than a stale one (`Project.result`). The fit's
warnings are cached with it (`Project.fit_warnings`), and the starting-profile fit
(`forster.io.scan_run`) and the Monte Carlo (`forster.rescale.monte_carlo`) are cached
the same way under their own keys (`Project.scan_task`, `Project.store_scan`,
`Project.scan`; `Project.mc_task`, `Project.store_mc`, `Project.mc`).

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
    LAB_PROFILE_LAYOUT,
    METADATA_COLUMNS,
    MIN_PROFILE_POINTS,
    TEMPLATE_PROFILE_LAYOUT,
    UNCERTAINTY_COLUMNS,
    Run,
    _usable_rows,
    fit_run,
    mc_summary,
    read_diffusivities,
    read_profiles,
    read_runs,
    read_table,
    run_row,
    scan_run,
    scan_summary,
    summary_row,
    uncertainty_columns,
)
from forster.model import Conditions, FitResult, Grid, InitialCondition, Orientation, Profile
from forster.rescale import PosteriorOfTime, TimeDistribution, monte_carlo
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
RECORDS = ("results", "scans", "mcs")  # project.json's cache indexes, run -> key
MASK_COLUMN = "Marked_bad"  # profiles.csv: "bad" on a masked point (`read_profiles`)
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


@dataclass(frozen=True)
class McRecord:
    """A Monte Carlo over the conditions and the law (`forster.rescale.monte_carlo`) as
    cached: its summary.csv columns (`forster.io.mc_summary`), the drawn times and what
    `TimeDistribution` holds besides, and its warnings."""

    summary: dict[str, Any]
    days: np.ndarray
    method: str
    percentiles_days: dict[int, float]
    sd_log10_by_source: dict[str, float]
    seed: int
    warnings: tuple[str, ...] = ()

    @classmethod
    def of(cls, dist: TimeDistribution, warnings: tuple[str, ...] = ()) -> McRecord:
        return cls(
            mc_summary(dist),
            np.asarray(dist.days, dtype=float),
            dist.method,
            dict(dist.percentiles_days),
            dict(dist.sd_log10_by_source),
            dist.seed,
            warnings,
        )

    @property
    def distribution(self) -> TimeDistribution:
        """As `forster.rescale.TimeDistribution` (for `plot_time_distributions`)."""
        return TimeDistribution(
            self.method, self.days, self.percentiles_days, self.sd_log10_by_source, self.seed
        )


@dataclass(frozen=True)
class McTask:
    """One run's Monte Carlo, captured like `FitTask` with the fit it draws around and,
    when the run has a current starting-profile fit, that fit's posterior of time (as
    ``forster fit`` uses it): call it in a worker thread, then hand the record to
    `Project.store_mc`."""

    name: str
    key: str
    run: Run
    result: FitResult
    posterior: PosteriorOfTime | None

    def __call__(
        self,
        *,
        progress: Callable[[int, int], None] | None = None,
        cancel: Callable[[], bool] | None = None,
    ) -> McRecord:
        if self.run.mc is None:
            raise ValueError(f"run {self.name!r} has no Monte Carlo settings")
        with notes.collect() as caught:
            dist = monte_carlo(
                self.result,
                self.run.mc,
                posterior=self.posterior,
                n=self.run.mc_samples,
                seed=self.run.mc_seed,
                progress=progress,
                cancel=cancel,
            )
        return McRecord.of(dist, notes.messages(caught))


def profile_table(
    profiles: Mapping[str, Profile],
    masked: Mapping[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> pd.DataFrame:
    """Every profile as the template profile table (Fo as a fraction). With masked points
    (per profile: their distances and Fo), they are added in distance order with
    ``Marked_bad`` "bad", which the table reader drops."""
    masked = {k: v for k, v in (masked or {}).items() if k in profiles and len(v[0])}
    frames = []
    for name, p in profiles.items():
        frame = pd.DataFrame({"profile": name, "distance_um": p.distance_um, "fo": p.fo})
        if masked:
            x, fo = masked.get(name, (np.empty(0), np.empty(0)))
            frame[MASK_COLUMN] = ""
            bad = pd.DataFrame({"profile": name, "distance_um": x, "fo": fo, MASK_COLUMN: "bad"})
            frame = pd.concat([frame, bad], ignore_index=True)
            frame = frame.sort_values("distance_um", kind="stable")
        frames.append(frame)
    if not frames:
        return pd.DataFrame({"profile": [], "distance_um": [], "fo": []})
    return pd.concat(frames, ignore_index=True)


def _masked_points(path: Path, names: Iterable[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The points of a profiles table left out of the fits (``Marked_bad`` "bad" or
    ``Ignore`` "yes", as `read_profiles` drops them), per profile in ``names``, Fo as a
    fraction; in the table's layout (template or lab)."""
    df = read_table(path)
    layout = (
        TEMPLATE_PROFILE_LAYOUT
        if set(TEMPLATE_PROFILE_LAYOUT[:3]) <= set(df.columns)
        else LAB_PROFILE_LAYOUT
    )
    flagged = df[~_usable_rows(df)]
    out = {}
    for name in names:
        rows = flagged[flagged[layout.name_column].astype(str) == name]
        rows = rows.dropna(subset=[layout.distance_column, layout.fo_column])
        if rows.empty:
            continue
        rows = rows.sort_values(layout.distance_column, kind="stable")
        fo = rows[layout.fo_column].to_numpy(dtype=float)
        out[name] = (
            rows[layout.distance_column].to_numpy(dtype=float),
            fo / 100 if layout.fo_in_percent else fo,
        )
    return out


def moved_x(x: np.ndarray | float, flip: bool, offset_um: float) -> np.ndarray:
    """Distances flipped (x -> offset - x) or shifted (x -> x - offset), so the point at
    ``offset_um`` goes to 0."""
    x = np.asarray(x, dtype=float)
    return offset_um - x if flip else x - offset_um


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
    # Points masked out of the fits, per profile (distances, Fo as a fraction): kept in
    # the project and in profiles.csv, never in ``profiles``, which is what is fitted.
    masked: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

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
        profiles, masked = {}, {}
        if (folder / PROFILES).exists():
            profiles = read_profiles(folder / PROFILES)
            masked = _masked_points(folder / PROFILES, profiles)
        runs = read_runs(folder / PARAMETERS, diffusivities=custom)
        state = {}
        if (folder / STATE).exists():
            state = json.loads((folder / STATE).read_text())
            if not isinstance(state, dict):
                raise ValueError(f"{folder / STATE}: not a FORSTER project file")
            # project.json comes from whoever sent the folder: keep only well-formed cache
            # keys, which are used to build paths (and to delete stale entries).
            for kind in RECORDS:
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
        return cls(folder, profiles, runs, custom, state, masked=masked)

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
        _write_csv(profile_table(self.profiles, self.masked), folder / PROFILES)
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
        for key in ("views", *RECORDS):
            self.state.get(key, {}).pop(name, None)
        self.dirty = True

    def rename_run(self, name: str, new_name: str) -> Run:
        self._check_new_name(new_name)
        i = self._index(name)
        self.runs[i] = replace(self.runs[i], name=new_name)
        for key in ("views", *RECORDS):
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
        return self.add_profiles(read(fo_in_percent))

    def add_profiles(self, new: Mapping[str, Profile]) -> list[str]:
        """Add profiles (a name already in the project is an error, and then none is
        added); their names."""
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
        self.masked.pop(name, None)
        self.dirty = True

    def points(self, name: str) -> pd.DataFrame:
        """Every point of a profile, masked ones too, in distance order: ``distance_um``,
        ``fo`` (a fraction) and ``masked``. `set_masked` takes a mask in this order."""
        p = self.profiles[name]
        x, fo = self.masked.get(name, (np.empty(0), np.empty(0)))
        table = pd.DataFrame(
            {
                "distance_um": np.concatenate([p.distance_um, x]),
                "fo": np.concatenate([p.fo, fo]),
                "masked": np.r_[np.zeros(len(p.fo), bool), np.ones(len(fo), bool)],
            }
        )
        return table.sort_values("distance_um", kind="stable").reset_index(drop=True)

    def set_masked(self, name: str, masked: Iterable[bool]) -> None:
        """Mask points out of the fits (True) or back in, one flag per row of
        `points`. Masked points stay in the project (and in profiles.csv, flagged); at
        least `forster.io.MIN_PROFILE_POINTS` points must stay in."""
        table = self.points(name)
        flags = np.asarray(list(masked), dtype=bool)
        if flags.shape != (len(table),):
            raise ValueError(f"profile {name!r} has {len(table)} points; got {len(flags)} flags")
        kept, out = table[~flags], table[flags]
        if len(kept) < MIN_PROFILE_POINTS:
            raise ValueError(
                f"profile {name!r}: keep at least {MIN_PROFILE_POINTS} points in the fit"
            )
        self.profiles[name] = Profile(kept.distance_um.to_numpy(), kept.fo.to_numpy(), name=name)
        if len(out):
            self.masked[name] = (out.distance_um.to_numpy(), out.fo.to_numpy())
        else:
            self.masked.pop(name, None)
        self.dirty = True

    def move_profile(self, name: str, *, flip: bool = False, offset_um: float = 0.0) -> None:
        """Flip a profile's distances (x -> offset - x) or shift them (x -> x - offset),
        so the rim is at 0 (`moved_x`); masked points move too. The runs on the profile
        move with a shift (domain, steps, crystal centre, growth-stage rims); a flip
        would turn their starting profiles around, so it is refused while runs use the
        profile."""
        users = [r for r in self.runs if r.profile_name == name]
        if flip and users:
            raise ValueError(
                f"profile {name!r} is used by runs {[r.name for r in users]}: flip it before "
                "adding runs, or remove them first"
            )
        moved = {r.name: _shifted(r, offset_um) for r in users}
        for run in moved.values():
            self.validate(run)
        p = self.profiles[name]
        x = moved_x(p.distance_um, flip, offset_um)
        order = np.argsort(x, kind="stable")
        self.profiles[name] = Profile(x[order], p.fo[order], name=name)
        if name in self.masked:
            mx, mfo = self.masked[name]
            mx = moved_x(mx, flip, offset_um)
            order = np.argsort(mx, kind="stable")
            self.masked[name] = (mx[order], mfo[order])
        for run_name, run in moved.items():
            self.runs[self._index(run_name)] = run
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

    def variant_task(self, name: str, **changes: Any) -> FitTask:
        """The fit of the run with ``changes`` (Run fields; another diffusion equation,
        say) as a task keyed by those settings; the run itself is not changed. A front
        end keeps such results itself (the cache holds one entry per run): `store` files
        one under the run only if the changes made no difference."""
        variant = _reconcile(self.run(name), changes)
        return FitTask(name, self._key(variant, _fit_row(variant)), variant, dict(self.profiles))

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
        in_use = {k for kind in RECORDS for k in self.state.get(kind, {}).values()}
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

    # -- the Monte Carlo ----------------------------------------------------------------

    def _posterior(self, name: str) -> tuple[PosteriorOfTime | None, str | None]:
        """The run's current starting-profile fit's posterior of time and that fit's key
        (None, None without one, or for a grid scan, which has none)."""
        if not self.run(name).uncertainties:
            return None, None
        record = self.scan(name)
        if record is None or record.days is None or record.density is None:
            return None, None
        return PosteriorOfTime(record.days, record.density), self.scan_task(name).key

    def mc_key(self, name: str) -> str:
        """Hash of everything that determines the run's Monte Carlo: the fit's settings,
        the mc_ columns and the starting-profile fit whose posterior it draws from."""
        run = self.run(name)
        mc = {k: v for k, v in uncertainty_columns(run).items() if k.startswith("mc_")}
        return self._key(run, {**_fit_row(run), **mc, "_scan": self._posterior(name)[1]})

    def mc_task(self, name: str) -> McTask:
        """The run's Monte Carlo as a task for a worker thread (`McTask`). It needs the
        run's Monte Carlo settings (``Run.mc``) and its cached fit."""
        run = self.run(name)
        if run.mc is None:
            raise ValueError(f"run {name!r}: give the Monte Carlo's uncertainties first")
        result = self.result(name)
        if result is None:
            raise ValueError(f"run {name!r}: fit the run first")
        return McTask(name, self.mc_key(name), run, result, self._posterior(name)[0])

    def store_mc(self, task: McTask, record: McRecord) -> bool:
        """Cache a Monte Carlo under the settings it was made with; whether they are
        still the run's (as `store`)."""
        _store_mc(record, self._cache(task.key))
        current = task.name in self.run_names and self.mc_key(task.name) == task.key
        if current:
            self._record("mcs", task.name, task.key)
        return current

    def mc(self, name: str) -> McRecord | None:
        """The cached Monte Carlo for the run's current settings, or None."""
        if self.run(name).mc is None:
            return None
        folder = self._cache(self.mc_key(name))
        if not (folder / "mc.json").exists():
            return None
        _no_links(folder.parents[1], folder.parent, folder, folder / "mc.json", folder / "mc.npz")
        return _load_mc(folder)

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
    if out.mc is not None and out.mc.fO2_buffer is None and out.fO2_form is not None:
        # The table reader holds a buffer-relative fO2 to its own buffer when
        # mc_fO2_buffer is blank: say so here, so the run reads back as itself.
        out = replace(out, mc=replace(out.mc, fO2_buffer=out.fO2_form[0]))
    return out


def _shifted(run: Run, offset_um: float) -> Run:
    """The run's geometry moved by -offset (`Project.move_profile`)."""
    if not offset_um:
        return run
    ic = run.initial
    low, high = ic.domain_um
    initial = replace(
        ic,
        domain_um=(low - offset_um, high - offset_um),
        steps_um=tuple(s - offset_um for s in ic.steps_um),
        centre_um=None if ic.centre_um is None else ic.centre_um - offset_um,
    )
    stages = tuple(replace(s, rim_um=s.rim_um - offset_um) for s in run.stages)
    return replace(run, initial=initial, stages=stages)


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


def _store_mc(record: McRecord, folder: Path) -> None:
    """A Monte Carlo as mc.json and mc.npz (the drawn times), no pickles."""
    _no_links(folder.parents[1], folder.parent, folder, folder / "mc.npz")
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=folder, suffix=".npz", delete=False) as tmp:
        np.savez(tmp, days=record.days)
    Path(tmp.name).replace(folder / "mc.npz")
    data = {
        "summary": record.summary,
        "method": record.method,
        "percentiles_days": {str(k): v for k, v in record.percentiles_days.items()},
        "sd_log10_by_source": record.sd_log10_by_source,
        "seed": record.seed,
        "warnings": list(record.warnings),
    }
    _write_text(json.dumps(data, default=float), folder / "mc.json")  # last: marks complete


def _load_mc(folder: Path) -> McRecord:
    data = json.loads((folder / "mc.json").read_text())
    with np.load(folder / "mc.npz", allow_pickle=False) as npz:
        days = npz["days"]
    return McRecord(
        dict(data["summary"]),
        days,
        str(data["method"]),
        {int(k): float(v) for k, v in data["percentiles_days"].items()},
        {str(k): float(v) for k, v in data["sd_log10_by_source"].items()},
        int(data["seed"]),
        _text_tuple(data.get("warnings")),
    )
