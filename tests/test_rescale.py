"""`forster.rescale`: Monte Carlo over the conditions and the diffusivity law.

The rescaling t = t_fit D_fit / D_draw is checked against real refits at shifted
conditions; the law-only spread against the linear law sigma of the fit; buffer-held
fO2 against the buffer; and the refit path on a fit with a growth stage.
"""

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from forster import Conditions, Grid, fit_profile, fit_run, read_profiles, read_runs, write_template
from forster.boundary import Stage
from forster.diffusivity import resolve
from forster.io import mc_run, mc_summary
from forster.rescale import ConditionUncertainty, _condition_draws, _log10_d, monte_carlo


@pytest.fixture(scope="module")
def example(tmp_path_factory):
    out = tmp_path_factory.mktemp("template")
    write_template(out)
    (run,) = read_runs(out / "parameters.csv")
    profiles = read_profiles(out / "profiles.csv")
    run = replace(run, diffusivity=resolve("refit"))
    return run, profiles, fit_run(run, profiles), out


def test_rescaling_matches_a_refit_at_shifted_conditions(example):
    """A fixed-rim fit at T + 30 C and fO2 x 3 gives the time the D ratio predicts."""
    run, profiles, result, _ = example
    profile = profiles[run.profile_name]
    shifted = Conditions(run.conditions.T_C + 30, run.conditions.P_Pa, run.conditions.fO2_Pa * 3)
    grid = Grid(1.0, result.dt_s / 4, result.inputs["grid"].max_time_days)
    args = (profile, run.orientation, run.initial)
    base = fit_profile(args[0], run.conditions, *args[1:], grid, None, run.diffusivity)
    moved = fit_profile(args[0], shifted, *args[1:], grid, None, run.diffusivity)
    x_fo = run.initial.edge_fo + (run.initial.core_fo - run.initial.edge_fo) / 3
    ratio = 10 ** (
        _log10_d(run.diffusivity, run.conditions, run.orientation, x_fo)
        - _log10_d(run.diffusivity, shifted, run.orientation, x_fo)
    )
    assert moved.time_days / base.time_days == pytest.approx(ratio, rel=0.01)


def test_law_only_spread_is_the_law_sigma(example):
    """With only the law drawn, the spread added to log10 t over the fit's own is the
    fit's linear law sigma (forster.model._law_uncertainty)."""
    _, _, result, _ = example
    fit_only = monte_carlo(result, ConditionUncertainty(law=False), n=20000, seed=2)
    with_law = monte_carlo(result, ConditionUncertainty(law=True), n=20000, seed=2)
    added = math.sqrt(np.var(np.log10(with_law.days)) - np.var(np.log10(fit_only.days)))
    assert added == pytest.approx(result.law_sigma_log10, rel=0.05)
    assert with_law.sd_log10_by_source["law"] == pytest.approx(result.law_sigma_log10, rel=0.05)
    assert fit_only.sd_log10_by_source["law"] == 0.0


def test_buffered_fo2_follows_temperature(example):
    run, _, _, _ = example
    c = run.conditions
    drawn = _condition_draws(
        c, ConditionUncertainty(T_sd_C=40, fO2_buffer="QFM"), 50, np.random.default_rng(0)
    )
    offset = math.log10(c.fO2_Pa / Conditions.from_buffer(c.T_C, c.P_Pa, 0.0, "QFM").fO2_Pa)
    for d in drawn:
        at_buffer = Conditions.from_buffer(d.T_C, d.P_Pa, offset, "QFM").fO2_Pa
        assert d.fO2_Pa == pytest.approx(at_buffer, rel=1e-12)
    assert np.std([d.T_C for d in drawn]) > 20
    free = _condition_draws(c, ConditionUncertainty(T_sd_C=40), 50, np.random.default_rng(0))
    assert all(d.fO2_Pa == c.fO2_Pa for d in free)


def test_the_same_seed_gives_the_same_draws(example):
    _, _, result, _ = example
    u = ConditionUncertainty(T_sd_C=10, fO2_sd_log=0.3)
    a = monte_carlo(result, u, n=500, seed=5)
    b = monte_carlo(result, u, n=500, seed=5)
    assert np.array_equal(a.days, b.days)
    assert not np.array_equal(a.days, monte_carlo(result, u, n=500, seed=6).days)
    with pytest.raises(ValueError, match="zero or more"):
        ConditionUncertainty(T_sd_C=-1)


def test_a_fit_with_a_growth_stage_is_refitted(example):
    run, profiles, result, _ = example
    stage = Stage(0.5 * 86400, rim_um=1.0, zone_value=0.915)
    grid = Grid(0.5, result.dt_s, result.inputs["grid"].max_time_days)
    staged = fit_profile(
        profiles[run.profile_name],
        run.conditions,
        run.orientation,
        run.initial,
        grid,
        None,
        run.diffusivity,
        stages=[stage],
    )
    dist = monte_carlo(staged, ConditionUncertainty(T_sd_C=10, law=False), n=6, seed=0)
    assert dist.method == "refit"
    assert len(dist.days) == 6 and np.all(np.isfinite(dist.days)) and np.all(dist.days > 0)
    assert dist.sd_log10_by_source == {}


def test_table_columns_run_the_monte_carlo_and_the_summary_refits(tmp_path, example):
    _, _, _, out = example
    params = pd.read_csv(out / "parameters.csv").assign(
        diffusivity="refit", mc_T_sd_C=15.0, mc_fO2_sd_log=0.3, mc_samples=800, mc_seed=4
    )
    path = tmp_path / "parameters.csv"
    params.to_csv(path, index=False)
    (table_run,) = read_runs(path)
    # The example gives fO2 relative to QFM, so the Monte Carlo keeps it there.
    assert table_run.mc == ConditionUncertainty(T_sd_C=15.0, fO2_sd_log=0.3, fO2_buffer="QFM")
    profiles = read_profiles(out / "profiles.csv")
    result = fit_run(table_run, profiles)
    row = mc_summary(mc_run(table_run, result))
    assert row["mc_method"] == "rescale" and row["mc_draws"] == 800
    assert row["mc_p05_days"] < row["mc_low_days"] < row["mc_median_days"] < row["mc_high_days"]
    assert "T=" in row["mc_sd_log10"]
    params.assign(mc_law="maybe").to_csv(path, index=False)
    with pytest.raises(ValueError, match="mc_law 'maybe'"):
        read_runs(path)


def test_time_distributions_figure(example):
    from matplotlib.figure import Figure

    from forster import plot_time_distributions

    _, _, result, _ = example
    mc = monte_carlo(result, ConditionUncertainty(T_sd_C=10), n=400, seed=0)
    ax = Figure().subplots()
    plot_time_distributions(result, None, mc, ax)
    legend = ax.get_legend()
    assert legend is not None
    labels = [t.get_text() for t in legend.get_texts()]
    assert labels[0].startswith("Fit alone") and labels[1].startswith("With conditions and law")
    assert ax.get_xscale() == "log"
    lo, hi = ax.get_xlim()
    assert lo < mc.percentiles_days[5] and hi > mc.percentiles_days[95]
