"""`forster.sampling`: best-fit initial condition and the posterior of time over it.

The samplers are checked against a dense grid on the same likelihood (the exact
answer up to the grid's quadrature), against a profile made from a known initial
condition, and for reproducibility. Their accuracy on real profiles against a
nested-sampling reference is measured in analyses/ic_sampling/.
"""

import itertools
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from forster import (
    Grid,
    InitialCondition,
    Profile,
    Uncertainty,
    fit_initial_condition,
    fit_run,
    read_profiles,
    read_runs,
    scan_run,
    scan_summary,
    write_template,
)
from forster import sampling as S
from forster.model import _diffusion_model, scheme_grid

TWO = {"step1_um": Uncertainty(2.0), "edge_fo": Uncertainty(0.002)}


@pytest.fixture(scope="module")
def example(tmp_path_factory):
    out = tmp_path_factory.mktemp("template")
    write_template(out)
    (run,) = read_runs(out / "parameters.csv")
    profiles = read_profiles(out / "profiles.csv")
    profile = profiles[run.profile_name]
    nominal = fit_run(run, profiles)
    # A scan-like grid: implicit, 80 steps to the fit, 6x the fitted time.
    grid = Grid(1.0, nominal.time_days * 86400 / 80, 6 * nominal.time_days)
    return run, profile, grid


def _args(run, profile, grid, uncertainties):
    return (
        profile,
        run.conditions,
        run.orientation,
        run.initial,
        uncertainties,
        grid,
        run.diffusivity,
        "implicit",
    )


@pytest.fixture(scope="module")
def laplace(example):
    run, profile, grid = example
    return fit_initial_condition(*_args(run, profile, grid, TWO), method="laplace", seed=3)


def test_laplace_and_nested_match_a_dense_grid_on_the_same_likelihood(example, laplace):
    run, profile, grid = example
    like = S.ICLikelihood(*_args(run, profile, grid, TWO))
    like.use_noise(laplace.noise)  # the likelihood the sampler used
    axes = [np.linspace(lo, hi, 31) for lo, hi in zip(like.lo, like.hi, strict=True)]
    thetas, log_w, rows = [], [], []
    for theta in itertools.product(*axes):
        r = like.log_like_rows(np.array(theta))
        if r is not None:
            thetas.append(theta)
            log_w.append(like.log_prior(np.array(theta)))
            rows.append(r)
    exact = S._summarise(like, S._Samples(np.array(thetas), np.array(log_w), np.array(rows)))
    nested = fit_initial_condition(
        *_args(run, profile, grid, TWO), method="nested", seed=3, nlive=100
    )
    assert laplace.method == "laplace" and nested.method == "nested"
    reference = (exact["low_days"], exact["median_days"], exact["high_days"])
    for post, tol in ((laplace, 0.01), (nested, 0.03)):
        got = (post.low_days, post.median_days, post.high_days)
        assert got == pytest.approx(reference, rel=tol)
        assert post.ic_share == pytest.approx(exact["ic_share"], abs=0.05)
    assert laplace.log_evidence == pytest.approx(nested.log_evidence, abs=0.3)


def test_best_fit_recovers_a_known_initial_condition(example):
    """A profile made from a known step and edge at a known time (plus noise): starting
    3 um and 0.004 Fo off, the best fit finds them and the range covers the time."""
    run, _, grid = example
    truth = replace(run.initial, steps_um=(27.0,), edge_fo=0.912)
    x_model, dx, (low, high) = scheme_grid(
        "implicit", np.linspace(2.5, 72, 30), grid, truth.domain_um
    )
    model_x = x_model[low : high + 1]
    start, advance = _diffusion_model(
        "implicit", run.conditions, run.orientation, truth, grid, run.diffusivity, model_x, dx
    )
    rows = advance(start, 200)
    true_row = 60
    x = np.linspace(truth.domain_um[0] + 0.5, truth.domain_um[1], 40)
    rng = np.random.default_rng(0)
    fo = np.interp(x, model_x, rows[true_row]) + rng.normal(0, 0.0007, len(x))
    profile = Profile(x, fo, name="synthetic")
    nominal = replace(truth, steps_um=(30.0,), edge_fo=0.916)
    unc = {"step1_um": Uncertainty(4.0), "edge_fo": Uncertainty(0.006)}
    post = fit_initial_condition(
        profile, run.conditions, run.orientation, nominal, unc, grid, run.diffusivity, "implicit"
    )
    true_days = true_row * grid.dt_s / 86400
    assert post.best_values["step1_um"] == pytest.approx(27.0, abs=1.5)
    assert post.best_values["edge_fo"] == pytest.approx(0.912, abs=0.002)
    assert post.low_days <= true_days <= post.high_days
    assert post.best_initial.steps_um[0] == post.best_values["step1_um"]


def test_the_same_seed_gives_the_same_result(example, laplace):
    run, profile, grid = example
    again = fit_initial_condition(*_args(run, profile, grid, TWO), method="laplace", seed=3)
    assert (again.low_days, again.median_days, again.high_days) == (
        laplace.low_days,
        laplace.median_days,
        laplace.high_days,
    )
    other = fit_initial_condition(*_args(run, profile, grid, TWO), method="laplace", seed=4)
    assert other.median_days == pytest.approx(laplace.median_days, rel=0.01)


def test_variance_split_adds_up(example, laplace):
    """E[Var(t | IC)] + Var(E[t | IC]) is the variance of the marginal posterior of time."""
    days, density = laplace.days, laplace.density
    p = density / density.sum()
    var = p @ days**2 - (p @ days) ** 2
    assert 0 < laplace.ic_share < 1
    assert var > 0
    run, profile, grid = example
    like = S.ICLikelihood(*_args(run, profile, grid, TWO))
    like.use_noise(laplace.noise)
    samples, _ = S._importance(like, np.zeros(2), 256, 0)
    summary = S._summarise(like, samples)
    q = summary["density"] / summary["density"].sum()
    total = q @ summary["days"] ** 2 - (q @ summary["days"]) ** 2
    t = summary["days"]
    cond = np.exp(samples.rows - samples.rows.max(axis=1, keepdims=True))
    cond /= cond.sum(axis=1, keepdims=True)
    e_t, var_t = cond @ t, cond @ t**2 - (cond @ t) ** 2
    marginal = samples.log_w + np.log(np.exp(samples.rows - samples.rows.max()).sum(axis=1))
    omega = np.exp(marginal - marginal.max())
    omega /= omega.sum()
    between = omega @ (e_t - omega @ e_t) ** 2
    assert omega @ var_t + between == pytest.approx(total, rel=1e-9)
    assert summary["ic_share"] == pytest.approx(between / total, rel=1e-9)


def test_auto_falls_back_to_nested_when_the_proposal_is_poor(example):
    run, profile, grid = example
    post = fit_initial_condition(*_args(run, profile, grid, TWO), ess_fallback=1.01, nlive=60)
    assert post.method == "nested"
    with pytest.raises(ValueError, match="unknown method"):
        fit_initial_condition(*_args(run, profile, grid, TWO), method="mcmc")


def test_unknown_parameters_are_errors(example):
    run, profile, grid = example
    with pytest.raises(ValueError, match="no parameter"):
        S.ICLikelihood(*_args(run, profile, grid, {"step2_um": Uncertainty(1.0)}))
    with pytest.raises(ValueError, match="no uncertain parameters"):
        S.ICLikelihood(*_args(run, profile, grid, {}))


def test_table_runs_default_to_auto_and_the_summary_refits(tmp_path):
    write_template(tmp_path / "t")
    params = pd.read_csv(tmp_path / "t" / "parameters.csv").assign(
        steps_sd_um=2.0, edge_fo_sd=0.002
    )
    path = tmp_path / "parameters.csv"
    params.to_csv(path, index=False)
    (table_run,) = read_runs(path)
    assert (table_run.scan_method, table_run.scan_seed) == ("auto", 0)
    profiles = read_profiles(tmp_path / "t" / "profiles.csv")
    nominal = fit_run(table_run, profiles)
    post = scan_run(table_run, profiles, time_days=nominal.time_days)
    row = scan_summary(post)
    assert row["scan_sampler"] in ("laplace", "nested")
    assert row["scan_low_days"] <= row["scan_median_days"] <= row["scan_high_days"]
    assert 0 <= row["scan_ic_share"] <= 1
    again = scan_run(table_run, profiles, time_days=nominal.time_days)
    assert scan_summary(again)["scan_median_days"] == row["scan_median_days"]
    # scan_points without scan_method still means the grid, as tables written before.
    params.assign(scan_points=3).to_csv(path, index=False)
    assert read_runs(path)[0].scan_method == "grid"
    params.assign(scan_method="mcmc").to_csv(path, index=False)
    with pytest.raises(ValueError, match="scan_method 'mcmc'"):
        read_runs(path)
    # InitialCondition is reused unchanged by the sampler.
    assert isinstance(post.best_initial, InitialCondition)
