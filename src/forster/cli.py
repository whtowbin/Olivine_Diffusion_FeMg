"""Command line: ``forster template DIR`` and ``forster fit PROFILES PARAMETERS``."""

from __future__ import annotations

import argparse
import re
import sys
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from forster.diffusivity import (
    FE_MG_CHOICES,
    ArrheniusDiffusivity,
    Diffusivity,
    diffusivity_table,
    resolve,
)
from forster.io import (
    Run,
    fit_run,
    fit_tables,
    read_diffusivities,
    read_profiles,
    read_runs,
    scan_run,
    scan_size,
    scan_summary,
    scan_table,
    summary_row,
    uncertainty_columns,
)
from forster.model import FitResult
from forster.plotting import plot_fit
from forster.scan import MAX_CANDIDATES
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


def _with_laws(
    runs: list[Run],
    diffusivity: str | None,
    compare: bool,
    custom: Mapping[str, Diffusivity] | None,
) -> list[Run]:
    """``--diffusivity`` sets every run's equation; ``--compare-laws`` repeats each run
    once per offered Fe-Mg choice, named ``<run>_<alias>``."""
    if diffusivity is not None and compare:
        raise ValueError("give --diffusivity or --compare-laws, not both")
    if diffusivity is not None:
        equation = resolve(diffusivity, custom)
        return [replace(run, diffusivity=equation) for run in runs]
    if compare:
        return [
            replace(run, name=f"{run.name}_{alias}", diffusivity=resolve(alias))
            for run in runs
            for alias in FE_MG_CHOICES
        ]
    return runs


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


def _describe(error: Exception) -> str:
    """The message of an input error; other errors also name their type."""
    if isinstance(error, KeyError) and error.args:
        return str(error.args[0])  # str(KeyError) adds quotes around the message
    if isinstance(error, ValueError):
        return str(error)
    return f"{type(error).__name__}: {error}"


def _fit_one(run: Run, profiles: dict, out_dir: Path, max_candidates: int) -> dict:
    n_scan = scan_size(run, max_candidates=max_candidates)  # raises early if too large
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fit_run(run, profiles)
    seconds = time.perf_counter() - t0
    write_fit(result, out_dir, file_stem(run.name))
    row = summary_row(result)
    row = {"run": row.pop("run"), "profile": run.profile_name, **row, **uncertainty_columns(run)}
    if run.uncertainties:
        if n_scan:
            print(
                f"  scanning {n_scan} initial conditions (roughly {n_scan * seconds / 3:.0f} s "
                "or less)",
                flush=True,
            )
        else:
            print(f"  fitting the initial condition ({run.scan_method})", flush=True)
        with warnings.catch_warnings(record=True) as scan_caught:
            warnings.simplefilter("always")
            scan = scan_run(
                run, profiles, time_days=result.time_days, max_candidates=max_candidates
            )
        caught += scan_caught
        scan_table(scan).to_csv(out_dir / f"{file_stem(run.name)}_scan.csv", index=False)
        row.update(scan_summary(scan))
    notes = _user_warnings(caught)
    row["warnings"] = "; ".join(notes)
    row["forster_version"] = version("forster")
    for note in notes:
        print(f"  warning: {note}", file=sys.stderr)
    return row


def fit_command(args: argparse.Namespace) -> int:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        profiles = read_profiles(args.profiles, sheet=_sheet(args.profiles_sheet))
        custom = read_diffusivities(args.diffusivities) if args.diffusivities else None
        runs = read_runs(args.parameters, sheet=_sheet(args.parameters_sheet), diffusivities=custom)
    for note in _user_warnings(caught):
        print(f"warning: {note}", file=sys.stderr)
    runs = _select(runs, args.run)
    runs = _with_laws(runs, args.diffusivity, args.compare_laws, custom)
    _check_stems(runs)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, failed = [], []
    for i, run in enumerate(runs, 1):
        print(f"[{i}/{len(runs)}] {run.name}", flush=True)
        try:
            row = _fit_one(run, profiles, out_dir, args.max_candidates)
        except Exception as error:  # one bad run, whatever the error, must not lose the others
            print(f"  failed: {_describe(error)}", file=sys.stderr)
            failed.append(run.name)
            continue
        rows.append(row)
        print(f"  {row['time_days']:.2f} days ({row['low_days']:.2f}-{row['high_days']:.2f})")
        if "scan_best_days" in row:
            print(
                f"  with initial-condition uncertainty: {row['scan_best_days']:.2f} days "
                f"({row['scan_low_days']:.2f}-{row['scan_high_days']:.2f}); best {row['scan_best']}"
            )
            if row.get("scan_sampler") in ("laplace", "nested"):
                print(
                    f"  {row['scan_ic_share']:.0%} of the variance of time from the initial "
                    f"condition ({row['scan_sampler']}, {row['scan_candidates']} model runs)"
                )
    summary = out_dir / "summary.csv"
    pd.DataFrame(rows).to_csv(summary, index=False)
    # The equations behind summary.csv, so it can be refitted with --diffusivities.
    used = {r.diffusivity.name: r.diffusivity for r in runs}
    tabled = [d for d in used.values() if isinstance(d, ArrheniusDiffusivity)]
    diffusivity_table(tabled).to_csv(out_dir / "diffusivities.csv", index=False)
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
        help="write profiles.csv, parameters.csv and diffusivities.csv to fill in "
        "(with a worked example)",
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
    fit.add_argument(
        "--max-candidates",
        type=int,
        default=MAX_CANDIDATES,
        help=f"largest initial-condition scan to run (default {MAX_CANDIDATES} model runs)",
    )
    fit.add_argument(
        "--diffusivities",
        help="table of diffusivity equations named in the parameter table's diffusivity column",
    )
    laws = fit.add_mutually_exclusive_group()
    laws.add_argument(
        "--diffusivity",
        metavar="NAME",
        help="fit every run with this Fe-Mg equation, overriding the diffusivity column: "
        + ", ".join(FE_MG_CHOICES)
        + ", a preset name, or a name from --diffusivities",
    )
    laws.add_argument(
        "--compare-laws",
        action="store_true",
        help="fit every run once with each of "
        + ", ".join(FE_MG_CHOICES)
        + " (runs named <run>_<choice>)",
    )
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
