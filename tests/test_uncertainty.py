"""The correlated-noise uncertainty method (forster.uncertainty).

The coverage tests are the ones that matter: with the true time known, the 1-sigma
range should contain it about 68% of the time, both when the model is exact and
when the profile carries a smooth misfit the model cannot follow. The old
kriged-grid rule covered 99.7% and 62% in the same two cases, raw-point
delta chi2 = 1 with independent errors 69% and 23%
(docs/references/fit-uncertainty.md).
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from conftest import ROUND_ROBIN_DATA, ROUND_ROBIN_PARAMS, require_round_robin, requires_round_robin
from test_differential import BATCH_DEFAULTS

from forster import (
    Conditions,
    Grid,
    InitialCondition,
    NoiseModel,
    Orientation,
    Profile,
    RunDefaults,
    fit_profile,
    fit_run,
    read_profiles,
    read_runs,
)
from forster.uncertainty import (
    chi2_per_step,
    delta_chi2_interval,
    fit_correlated,
    fit_noise_model,
    interpolation_matrix,
)

RNG_SEED = 20261012


def test_interpolation_matrix_matches_np_interp():
    rng = np.random.default_rng(RNG_SEED)
    grid = np.linspace(0, 300, 301)
    x = np.sort(rng.uniform(0, 300, 40))
    curve = np.cumsum(rng.normal(size=grid.size))
    np.testing.assert_allclose(interpolation_matrix(grid, x) @ curve, np.interp(x, grid, curve))


def test_quadratic_chi2_equals_explicitly_whitened_residuals():
    rng = np.random.default_rng(RNG_SEED)
    grid = np.linspace(0, 100, 101)
    t = np.arange(1, 501)[:, None]
    curves = 0.82 + 0.06 * 0.5 * (1 + np.tanh((grid - 40) / np.sqrt(t)))  # (steps, grid)
    x = np.sort(rng.uniform(0, 100, 60))
    W = interpolation_matrix(grid, x)
    y = W @ curves[200] + rng.normal(0, 0.002, x.size)
    cov = NoiseModel(0.002, 0.003, 8.0).covariance(x)
    inv_chol = np.linalg.inv(np.linalg.cholesky(cov))
    direct = np.sum(((curves @ W.T - y) @ inv_chol.T) ** 2, axis=1)
    np.testing.assert_allclose(chi2_per_step(curves, W, y, cov), direct, rtol=1e-8)
    np.testing.assert_allclose(
        chi2_per_step(curves, W, y), np.sum((curves @ W.T - y) ** 2, axis=1), rtol=1e-8
    )


def test_interval_ends_are_interpolated_crossings():
    k = np.arange(200.0)
    chi2 = ((k - 50.3) / 7.0) ** 2 + 12.0  # exact 1-sigma range 43.3 to 57.3
    span = delta_chi2_interval(chi2)
    assert span.best == 50
    assert span.low == pytest.approx(43.3, abs=0.02)
    assert span.high == pytest.approx(57.3, abs=0.02)
    assert (span.low_row, span.high_row) == (44, 57)


def test_interval_at_the_last_time_step_warns():
    chi2 = np.linspace(10.0, 9.0, 50)  # still falling at the last step
    with pytest.warns(UserWarning, match="last modelled time step"):
        span = delta_chi2_interval(chi2)
    assert span.high == 49


def test_noise_model_recovers_known_parameters():
    rng = np.random.default_rng(RNG_SEED)
    x = np.linspace(0, 300, 150)
    truth = NoiseModel(white_sigma=0.002, correlated_sigma=0.004, correlation_length_um=15.0)
    r = np.linalg.cholesky(truth.covariance(x)) @ rng.standard_normal(x.size)
    fitted = fit_noise_model(x, r)
    assert fitted.white_sigma == pytest.approx(0.002, rel=0.3)
    assert fitted.correlated_sigma == pytest.approx(0.004, rel=0.5)
    assert fitted.correlation_length_um == pytest.approx(15.0, rel=0.6)


def test_noise_model_finds_no_correlation_in_white_noise():
    rng = np.random.default_rng(RNG_SEED)
    x = np.linspace(0, 300, 150)
    fitted = fit_noise_model(x, rng.normal(0, 0.002, x.size))
    assert fitted.white_sigma == pytest.approx(0.002, rel=0.2)
    assert fitted.correlated_sigma < 0.001


@pytest.fixture(scope="module")
def round_robin_fit():
    """KS20-527_2-transect-1 (29 points), fitted with the default method."""
    require_round_robin()
    runs = {r.name: r for r in read_runs(ROUND_ROBIN_PARAMS, defaults=RunDefaults())}
    return fit_run(runs["KS20-527_2-transect-1"], read_profiles(ROUND_ROBIN_DATA), keep_curves=True)


@pytest.mark.parametrize(
    ("correlated_sigma", "low", "high"),
    [
        (0.0, 0.60, 0.78),  # model exact: white noise only
        (0.004, 0.55, 0.78),  # plus a smooth 0.4 Fo misfit over 10 um
    ],
)
def test_coverage_is_close_to_68_percent(round_robin_fit, correlated_sigma, low, high):
    r = round_robin_fit
    k_true = r.indices[0]
    curves = r.curves[: 3 * k_true]  # every plausible best fit, at a third of the memory
    x = r.fit_x_um
    truth = interpolation_matrix(r.model_x_um, x) @ curves[k_true]
    noise = NoiseModel(0.002, correlated_sigma, 10.0)
    chol = np.linalg.cholesky(noise.covariance(x))
    rng = np.random.default_rng(RNG_SEED)
    trials, hits = 200, 0
    for _ in range(trials):
        y = truth + chol @ rng.standard_normal(x.size)
        with warnings.catch_warnings():  # grid-edge warnings are tested elsewhere
            warnings.simplefilter("ignore")
            span = fit_correlated(curves, r.model_x_um, x, y).interval
        hits += span.low <= k_true <= span.high
    assert low <= hits / trials <= high


@requires_round_robin
def test_default_fit_reports_a_range_around_the_best_fit(round_robin_fit):
    r = round_robin_fit
    assert r.uncertainty == "correlated"
    assert r.noise is not None and r.noise.white_sigma > 0
    assert r.time_minus_s < 0 < r.time_plus_s
    assert r.indices[1] <= r.indices[0] <= r.indices[2]
    assert len(r.fit_x_um) == 29  # the measured points, not the 300-point kriged grid
    # The best fit agrees with the legacy rule's to within the new 1-sigma range.
    runs = {run.name: run for run in read_runs(ROUND_ROBIN_PARAMS, defaults=BATCH_DEFAULTS)}
    legacy = fit_run(runs["KS20-527_2-transect-1"], read_profiles(ROUND_ROBIN_DATA))
    assert legacy.uncertainty == "legacy"
    assert r.time_s + r.time_minus_s <= legacy.time_s <= r.time_s + r.time_plus_s


def test_unknown_uncertainty_method_is_an_error():
    profile = Profile(np.arange(0.0, 50.0, 5.0), np.linspace(0.82, 0.88, 10))
    initial = InitialCondition(domain_um=(0, 45), edge_fo=0.82, core_fo=0.88, steps_um=(20,))
    with pytest.raises(ValueError, match="unknown uncertainty method"):
        fit_profile(
            profile,
            Conditions(1200, 4.2e7, 1e-3),
            Orientation(90, 90, 0),
            initial,
            Grid(dx_um=1, dt_s=600, max_time_days=1),
            uncertainty="bootstrap",
        )


@requires_round_robin
def test_uncertainty_column_is_read_per_run(tmp_path):
    table = pd.read_excel(ROUND_ROBIN_PARAMS, engine="openpyxl")
    table["uncertainty"] = None
    table.loc[0, "uncertainty"] = "Legacy"
    path = tmp_path / "params.csv"
    table.to_csv(path, index=False)
    methods = {run.name: run.uncertainty for run in read_runs(path)}
    assert methods[table.loc[0, "Sample"]] == "legacy"
    assert set(methods.values()) == {"correlated", "legacy"}


def test_small_sample_threshold():
    from forster.uncertainty import interval_threshold

    assert interval_threshold(6) == pytest.approx(1.480, abs=1e-3)
    assert interval_threshold(29) == pytest.approx(1.074, abs=1e-3)
    assert interval_threshold(10_000) == pytest.approx(1.0, abs=1e-3)
    assert interval_threshold(29, 2) > interval_threshold(29, 1)
    with pytest.raises(ValueError, match="cannot constrain"):
        interval_threshold(2, 2)


def test_posterior_interval_matches_delta_chi2_for_many_points_and_is_wider_for_few():
    from forster.uncertainty import posterior_interval

    k = np.arange(400.0)
    n = 10_000
    chi2 = n + ((k - 150.3) / 7.0) ** 2  # chi2 near n at its minimum, as for a fitted s
    many = posterior_interval(chi2, n)
    assert many.best == 150
    assert many.low == pytest.approx(143.3, abs=0.1)
    assert many.high == pytest.approx(157.3, abs=0.1)
    # Same delta chi2 curve with 6 points: the Student-t-like posterior is wider.
    few = posterior_interval(6 + ((k - 150.3) / 7.0) ** 2, 6)
    assert few.high - few.low > 1.15 * (many.high - many.low)


def test_sparse_profile_coverage_with_the_small_sample_correction(round_robin_fit):
    """6 points (50 um apart): 57% before the correction, about 64% with it."""
    r = round_robin_fit
    k_true = r.indices[0]
    curves = r.curves[: 3 * k_true]
    x = r.fit_x_um[::5]
    truth = interpolation_matrix(r.model_x_um, x) @ curves[k_true]
    rng = np.random.default_rng(RNG_SEED)
    trials, hits = 200, 0
    for _ in range(trials):
        y = truth + rng.normal(0, 0.002, x.size)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            span = fit_correlated(curves, r.model_x_um, x, y).interval
        hits += span.low <= k_true <= span.high
    assert 0.58 <= hits / trials <= 0.78


def test_grid_edge_warning_only_for_a_profile_wide_misfit():
    rng = np.random.default_rng(RNG_SEED)
    x = np.linspace(0, 300, 29)
    warned = 0
    for _ in range(100):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit_noise_model(x, rng.normal(0, 0.002, x.size))
        warned += any("spans the profile" in str(w.message) for w in caught)
    assert warned <= 5  # about 1%; the old rule fired in 25-35% of white-noise fits
    trend = 0.006 * (x / 300 - 0.5) + rng.normal(0, 0.001, x.size)  # misfit across the profile
    with pytest.warns(UserWarning, match="spans the profile"):
        fit_noise_model(x, trend)


@requires_round_robin
def test_early_stopping_gives_the_same_fit_as_the_full_run(monkeypatch):
    """Two-step profile; also with stop settings so tight that the run has to be extended."""
    import forster.model as model

    runs = {r.name: r for r in read_runs(ROUND_ROBIN_PARAMS)}
    profiles = read_profiles(ROUND_ROBIN_DATA)
    run = runs["KS20-527-8_transect-2"]
    assert run.initial.step_fo is not None
    full = fit_run(run, profiles, stop_early=False)
    early = fit_run(run, profiles)
    assert early.modelled_days < full.modelled_days
    assert early.indices[0] == full.indices[0]
    assert (early.time_minus_s, early.time_plus_s) == (full.time_minus_s, full.time_plus_s)
    assert early.noise == full.noise
    monkeypatch.setattr(model, "STOP_TIME_FACTOR", 1.0)
    monkeypatch.setattr(model, "STOP_RSS_RATIO", 1.0)
    extended = fit_run(run, profiles)
    assert (extended.indices[0], extended.time_minus_s, extended.time_plus_s) == (
        full.indices[0],
        full.time_minus_s,
        full.time_plus_s,
    )


def test_near_step_check_flags_misfit_concentrated_at_the_step():
    """Residuals with a smooth misfit only near the step favour the near-step term by far
    more than 6 AIC; white residuals rarely do (the validation: at most 0.2%)."""
    from forster.uncertainty import NEAR_STEP_AIC, near_step_check

    rng = np.random.default_rng(3)
    x = np.linspace(0.0, 90.0, 60)
    step, ell = 40.0, 6.0
    envelope = np.exp(-0.5 * ((x - step) / ell) ** 2)
    smooth = np.sin(x / 3.0)
    near = 0.002 * rng.standard_normal(x.size) + 0.012 * envelope * smooth
    check = near_step_check(x, near, [step], ell)
    assert check.warn and check.delta_aic < NEAR_STEP_AIC and check.h > 0
    false_alarms = sum(
        near_step_check(x, 0.002 * rng.standard_normal(x.size), [step], ell).warn for _ in range(60)
    )
    assert false_alarms <= 2


def test_fits_warn_on_the_paper_profiles_with_near_step_misfit():
    """The three paper profiles where the validity study found the near-step term clearly
    preferred warn; a well-behaved one does not; and the check can be switched off."""
    from conftest import PAPER_DATA, PAPER_PARAMS

    runs = {r.name: r for r in read_runs(PAPER_PARAMS)}
    profiles = read_profiles(PAPER_DATA, sheet="Olivine")
    for name, expect in (("Xenocryst_ol6_prof_1", True), ("AZ18 WHT01_Ol_63_prof_1", False)):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = fit_run(runs[name], profiles)
        assert any("near the step" in str(w.message) for w in caught) is expect, name
        assert result.near_step_delta_aic is not None
        assert (result.near_step_delta_aic < -6) is expect
    quiet = fit_run(runs["Xenocryst_ol6_prof_1"], profiles, check_near_step=False)
    assert quiet.near_step_delta_aic is None
