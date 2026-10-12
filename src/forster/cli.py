"""Command line: ``forster template DIR`` and ``forster fit PROFILES PARAMETERS``."""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from forster.io import Run, fit_run, fit_tables, read_profiles, read_runs, summary_row
from forster.model import FitResult
from forster.plotting import plot_fit
from forster.template import write_template

DPI = 200


def _sheet(value: str | None) -> str | int:
    """Excel sheet name; the first sheet when not given (ignored for CSV)."""
    return 0 if value is None else value


def file_stem(run_name: str) -> str:
    """A run name made safe for file names: anything but letters, digits, ``-_.`` becomes ``_``."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", run_name)


def _select(runs: list[Run], names: Sequence[str] | None) -> list[Run]:
    if not names:
        return runs
    unknown = sorted(set(names) - {r.name for r in runs})
    if unknown:
        raise ValueError(f"--run {unknown} not in the parameter table")
    return [r for r in runs if r.name in names]


def _check_stems(runs: list[Run]) -> None:
    stems: dict[str, str] = {}
    for run in runs:
        stem = file_stem(run.name)
        if stem in stems:
            raise ValueError(
                f"runs {stems[stem]!r} and {run.name!r} would write the same files ({stem}_*)"
            )
        stems[stem] = run.name


def write_fit(result: FitResult, out_dir: Path, stem: str) -> list[Path]:
    """The fit figure and its tables (see `forster.io.fit_tables`)."""
    paths = []
    for kind, table in fit_tables(result).items():
        path = out_dir / f"{stem}_{kind}.csv"
        table.to_csv(path, index=False)
        paths.append(path)
    fig = Figure(figsize=(7, 4), layout="constrained")
    plot_fit(result, fig.subplots())
    path = out_dir / f"{stem}_fit.png"
    fig.savefig(path, dpi=DPI)
    return [path, *paths]


def _user_warnings(caught: list[warnings.WarningMessage]) -> list[str]:
    return sorted({str(w.message) for w in caught if issubclass(w.category, UserWarning)})


def _fit_one(run: Run, profiles: dict, out_dir: Path) -> dict:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fit_run(run, profiles)
    notes = _user_warnings(caught)
    write_fit(result, out_dir, file_stem(run.name))
    row = summary_row(result)
    row = {"run": row.pop("run"), "profile": run.profile_name, **row}
    row["warnings"] = "; ".join(notes)
    row["forster_version"] = version("forster")
    for note in notes:
        print(f"  warning: {note}", file=sys.stderr)
    return row


def fit_command(args: argparse.Namespace) -> int:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        profiles = read_profiles(args.profiles, sheet=_sheet(args.profiles_sheet))
        runs = read_runs(args.parameters, sheet=_sheet(args.parameters_sheet))
    for note in _user_warnings(caught):
        print(f"warning: {note}", file=sys.stderr)
    runs = _select(runs, args.run)
    _check_stems(runs)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, failed = [], []
    for i, run in enumerate(runs, 1):
        print(f"[{i}/{len(runs)}] {run.name}", flush=True)
        try:
            row = _fit_one(run, profiles, out_dir)
        except (KeyError, ValueError) as error:
            # One bad row should not lose the other fits.
            print(f"  failed: {error}", file=sys.stderr)
            failed.append(run.name)
            continue
        rows.append(row)
        print(f"  {row['time_days']:.2f} days ({row['low_days']:.2f}-{row['high_days']:.2f})")
    summary = out_dir / "summary.csv"
    pd.DataFrame(rows).to_csv(summary, index=False)
    print(f"{len(rows)} of {len(runs)} runs fitted; results in {out_dir}, summary in {summary}")
    if failed:
        print(f"failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


def template_command(args: argparse.Namespace) -> int:
    for path in write_template(args.dir, overwrite=args.force):
        print(path)
    print(
        f"Fit the example as it is: forster fit {args.dir}/profiles.csv {args.dir}/parameters.csv"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forster",
        description="FORSTER olivine diffusion: fit Fe-Mg diffusion times to Fo profiles.",
    )
    parser.add_argument("--version", action="version", version=f"forster {version('forster')}")
    commands = parser.add_subparsers(dest="command", required=True)

    template = commands.add_parser(
        "template",
        help="write profiles.csv and parameters.csv to fill in (with a worked example)",
    )
    template.add_argument("dir", help="folder to write into (created if needed)")
    template.add_argument("--force", action="store_true", help="overwrite existing files")
    template.set_defaults(handler=template_command)

    fit = commands.add_parser("fit", help="fit every run in a parameter table")
    fit.add_argument("profiles", help="profile table (.csv or .xlsx)")
    fit.add_argument("parameters", help="parameter table (.csv or .xlsx)")
    fit.add_argument("--out", default="results", help="output folder (default: results)")
    fit.add_argument("--run", nargs="+", metavar="NAME", help="fit only these runs")
    fit.add_argument("--profiles-sheet", help="Excel sheet of the profile table")
    fit.add_argument("--parameters-sheet", help="Excel sheet of the parameter table")
    fit.set_defaults(handler=fit_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        print(f"forster: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
