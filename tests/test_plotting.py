"""`plot_law_comparison` and the ``--compare-laws`` figure, on the packaged example."""

from dataclasses import replace

import numpy as np
import pytest
from matplotlib.figure import Figure

from forster import cli as cli_module
from forster import (
    fit_run,
    plot_law_comparison,
    plot_law_times,
    read_profiles,
    read_runs,
    write_template,
)
from forster.diffusivity import FE_MG_CHOICES, resolve

STEM = "AZ18_WHT01_Ol_63_prof_1"


@pytest.fixture(scope="module")
def example(tmp_path_factory):
    """The example's tables and its fit under each offered law (alias -> FitResult)."""
    folder = tmp_path_factory.mktemp("template")
    write_template(folder)
    run = read_runs(folder / "parameters.csv")[0]
    profiles = read_profiles(folder / "profiles.csv")
    fits = {
        alias: fit_run(replace(run, diffusivity=resolve(alias)), profiles)
        for alias in FE_MG_CHOICES
    }
    return folder, fits


def test_one_line_a_law_labelled_with_its_time(example):
    _, fits = example
    ax = plot_law_comparison(list(fits.values()), Figure().subplots())
    legend = ax.get_legend()
    assert legend is not None
    labels = [text.get_text() for text in legend.get_texts()]
    law_lines = [
        line
        for line in ax.get_lines()
        if any(str(line.get_label()).startswith(lab) for _, lab in FE_MG_CHOICES.values())
    ]
    assert len(law_lines) == 4
    for (_, display), fit in zip(FE_MG_CHOICES.values(), fits.values(), strict=True):
        low, high = fit.time_range_days()
        expected = f"{display}: {fit.time_days:.1f} d ({low:.1f}-{high:.1f})"
        assert expected in labels
    # Measured points and the initial condition once each, not once a law.
    assert labels.count("Measured") == 1
    assert labels.count("Initial") == 1
    assert ax.get_title().startswith("AZ18 WHT01_Ol_63_prof_1\n")


def test_times_one_row_a_law_on_a_log_axis(example):
    """Rows top to bottom in the order given; the law's calibration bar only where the
    law has a covariance (not Classic DIPRA), and it spans that wider range."""
    _, fits = example
    results = list(fits.values())
    ax = plot_law_times(results, Figure().subplots())
    assert ax.get_xscale() == "log"
    rows = {
        int(y): text.get_text()
        for y, text in zip(ax.get_yticks(), ax.get_yticklabels(), strict=True)
    }
    assert [rows[y] for y in (3, 2, 1, 0)] == [lab for _, lab in FE_MG_CHOICES.values()]

    def bars(gid):
        lines = [line for line in ax.get_lines() if line.get_gid() == gid]
        return {int(np.asarray(line.get_ydata())[0]): line for line in lines}

    fit_bars, law_bars = bars("fit_range"), bars("law_range")
    assert sorted(fit_bars) == [0, 1, 2, 3]
    with_covariance = {3 - i for i, fit in enumerate(results) if fit.law_sigma_log10 is not None}
    assert with_covariance == {0, 1, 2}  # Classic DIPRA, the top row, has none
    assert set(law_bars) == with_covariance
    for i, fit in enumerate(results):
        assert list(fit_bars[3 - i].get_xdata()) == list(fit.time_range_days())
        if 3 - i in law_bars:
            assert list(law_bars[3 - i].get_xdata()) == list(fit.time_range_days("law"))
    assert "no law covariance" in [text.get_text() for text in ax.texts]


@pytest.mark.parametrize("plot", [plot_law_comparison, plot_law_times])
def test_no_fits_is_an_error(plot):
    with pytest.raises(ValueError, match="at least one"):
        plot([], Figure().subplots())


def test_a_failed_law_still_leaves_the_others_figure(example, tmp_path, monkeypatch, capsys):
    """The last law fails (the one whose turn writes the figure): the other three are
    still drawn. The fits are the module's, looked up by law, so no fit is repeated."""
    folder, fits = example
    by_name = {fit.inputs["diffusivity"].name: fit for fit in fits.values()}
    failing = resolve("dfens_global").name

    def fake_fit_run(run, profiles, **_):
        if run.diffusivity.name == failing:
            raise ValueError("this law failed")
        return replace(by_name[run.diffusivity.name], name=run.name)

    monkeypatch.setattr(cli_module, "fit_run", fake_fit_run)
    code = cli_module.main(
        [
            "fit",
            str(folder / "profiles.csv"),
            str(folder / "parameters.csv"),
            "--compare-laws",
            "--out",
            str(tmp_path),
        ]
    )
    assert code == 1
    out = capsys.readouterr()
    assert "this law failed" in out.err
    assert f"law comparison of 3 fits: {STEM}_laws.png" in out.out
    assert (tmp_path / f"{STEM}_laws.png").stat().st_size > 0
    assert not (tmp_path / f"{STEM}_dfens_global_fit.png").exists()
    assert (tmp_path / f"{STEM}_dipra_fit.png").stat().st_size > 0
