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


@pytest.mark.needs_numba
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


def _synthetic(example):
    """A profile made from a known step (27 um) and edge (0.912) at a known time, with
    white noise; returns the profile, the truth and the true time in days."""
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
    x = np.linspace(truth.domain_um[0] + 0.5, truth.domain_um[1], 40)
    rng = np.random.default_rng(0)
    fo = np.interp(x, model_x, rows[60]) + rng.normal(0, 0.0007, len(x))
    return Profile(x, fo, name="synthetic"), truth, 60 * grid.dt_s / 86400


@pytest.mark.needs_numba
def test_a_prior_centred_off_the_data_does_not_inflate_the_noise_model(example):
    """A tight prior centred 2 sd from the truth: the first noise model is fitted at a
    data-driven best fit and refitted until stable, so the correlated term stays as with
    a prior centred on the truth and the range still covers the time. (Fitted at the
    prior's centre, it grew to 0.12 mol% over 35 um and the range missed the time.)"""
    run, _, grid = example
    profile, truth, true_days = _synthetic(example)
    args = (profile, run.conditions, run.orientation)
    centred = fit_initial_condition(
        *args, truth, {"step1_um": Uncertainty(1.0), "edge_fo": Uncertainty(0.002)}, grid,
        run.diffusivity, "implicit", method="laplace",
    )  # fmt: skip
    off = replace(truth, steps_um=(29.0,), edge_fo=0.916)
    post = fit_initial_condition(
        *args, off, {"step1_um": Uncertainty(1.0), "edge_fo": Uncertainty(0.002)}, grid,
        run.diffusivity, "implicit", method="laplace",
    )  # fmt: skip
    assert post.noise.correlated_sigma < 1.5 * centred.noise.correlated_sigma + 1e-4
    assert post.low_days <= true_days <= post.high_days


@pytest.mark.needs_numba
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
    assert post.best_values["a_fo"] == pytest.approx(0.912, abs=0.002)
    # One seeded draw: its 95% range must cover the time (a 1-sigma range misses about one
    # draw in three; this one's did only after the noise-start fix narrowed it honestly).
    cdf = np.cumsum(post.density)
    lo95, hi95 = np.interp([0.025, 0.975], cdf / cdf[-1], post.days)
    assert lo95 <= true_days <= hi95
    assert post.best_initial.steps_um[0] == post.best_values["step1_um"]


@pytest.mark.needs_numba
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


@pytest.mark.needs_numba
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


@pytest.mark.needs_numba
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


@pytest.mark.needs_numba
def test_table_runs_default_to_auto_and_the_summary_refits(tmp_path):
    write_template(tmp_path / "t")
    params = pd.read_csv(tmp_path / "t" / "parameters.csv").assign(
        steps_sd_um=2.0, segment_fo_sd="0.002, "
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
    # the near-step check at the best-fit initial condition (no near-step misfit here)
    assert -6 < row["scan_near_step_daic"] <= 2.0 + 1e-9
    again = scan_run(table_run, profiles, time_days=nominal.time_days)
    assert scan_summary(again)["scan_median_days"] == row["scan_median_days"]
    # scan_points without scan_method still means the grid, as tables written before.
    params.assign(scan_points=3).to_csv(path, index=False)
    (grid_run,) = read_runs(path)
    assert grid_run.scan_method == "grid"
    grid_row = scan_summary(scan_run(grid_run, profiles, time_days=nominal.time_days))
    assert -6 < grid_row["scan_near_step_daic"] <= 2.0 + 1e-9
    params.assign(scan_method="mcmc").to_csv(path, index=False)
    with pytest.raises(ValueError, match="scan_method 'mcmc'"):
        read_runs(path)
    # InitialCondition is reused unchanged by the sampler.
    assert isinstance(post.best_initial, InitialCondition)


def test_proposal_density_matches_scipy():
    """`_log_t_density` (from the Cholesky factor) equals scipy's multivariate t."""
    from scipy import stats

    rng = np.random.default_rng(0)
    a = rng.normal(size=(4, 4))
    cov = a @ a.T + 0.1 * np.eye(4)
    loc, z = rng.normal(size=4), rng.normal(size=(50, 4))
    expected = stats.multivariate_t(loc=loc, shape=cov, df=5).logpdf(z)
    got = S._log_t_density(z, loc, np.linalg.cholesky(cov), 5)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)


@pytest.mark.needs_numba
def test_a_cancel_hook_that_never_fires_leaves_the_result_unchanged(example, laplace):
    run, profile, grid = example
    calls = []

    def cancel() -> bool:
        calls.append(1)
        return False

    again = fit_initial_condition(
        *_args(run, profile, grid, TWO), method="laplace", seed=3, cancel=cancel
    )
    assert len(calls) == again.runs  # checked before every model run
    assert again.best_values == laplace.best_values
    np.testing.assert_array_equal(again.density, laplace.density)
    assert (again.low_days, again.median_days, again.high_days) == (
        laplace.low_days,
        laplace.median_days,
        laplace.high_days,
    )


def test_cancel_stops_the_sampler_and_the_grid_scan(example):
    run, profile, grid = example
    calls = []

    def cancel_after_five() -> bool:
        calls.append(1)
        return len(calls) > 5

    with pytest.raises(S.Cancelled):
        fit_initial_condition(
            *_args(run, profile, grid, TWO), method="nested", cancel=cancel_after_five
        )
    assert len(calls) == 6  # stopped at once, in the optimiser
    gridded = replace(run, uncertainties={"step1_um": Uncertainty(2.0)}, scan_method="grid")
    profiles = {run.profile_name: profile}
    with pytest.raises(S.Cancelled):
        scan_run(gridded, profiles, time_days=5.0, cancel=lambda: True)
    stages = []
    with pytest.raises(S.Cancelled):
        scan_run(
            replace(gridded, scan_method="laplace"),
            profiles,
            time_days=5.0,
            stage=stages.append,
            cancel=lambda: bool(stages),
        )
    assert stages == ["best fit"]


@pytest.mark.needs_numba
def test_a_conflicting_prior_is_reported_in_prior_sd_and_warned(example):
    """The data put the edge at 0.912; a Gaussian prior 0.9165 +- 0.002 then holds the
    best value about 2.2 sd from its nominal (inside the 2.5 sd edge): a warning says
    the prior and the data disagree, and the offset is reported."""
    run, _, grid = example
    profile, truth, _ = _synthetic(example)
    nominal = replace(truth, edge_fo=0.9165)
    with pytest.warns(UserWarning, match="the prior and the data disagree"):
        post = fit_initial_condition(
            profile, run.conditions, run.orientation, nominal,
            {"edge_fo": Uncertainty(0.002)}, grid, run.diffusivity, "implicit",
            method="laplace",
        )  # fmt: skip
    assert -2.5 < post.prior_offsets["a_fo"] <= -2.0
    assert scan_summary(post)["scan_prior_offsets"].startswith("a_fo=-2.")
