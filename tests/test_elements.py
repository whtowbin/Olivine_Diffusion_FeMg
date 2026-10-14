"""Ni, Mn and Ca diffusion alongside Fo (forster.elements), against solutions that share
no code with it, and the element and joint fits.

- Mn uses the same law as Fe-Mg, so a Mn start that is a + b Fo stays a + b Fo.
- Constant D (Ni in unzoned olivine, Ca at any Fo): the error-function solution.
- Ni in zoned olivine: an independent flux-form solver for both species, with D on
  the cell faces.
"""

import numpy as np
import pytest
from scipy.special import erf

from forster import (
    DIFFUSIVITY_PRESETS,
    Conditions,
    ElementInitial,
    ElementProfile,
    Grid,
    InitialCondition,
    Orientation,
    Profile,
    compiled,
    engine,
    fit_element,
    fit_joint,
    fit_profile,
)
from forster.elements import coupled_model, element_diffusivity
from forster.model import _diffusion_model, fe_mg_diffusivity

DX_UM = 1.0
DX_M = DX_UM * 1e-6
X_UM = np.arange(0, 300, DX_UM)
N_X = len(X_UM)
STEP_UM = 150
CONDITIONS = Conditions(T_C=1200, P_Pa=4.2e7, fO2_Pa=0.00138)  # round robin
ALONG_001 = Orientation(90, 90, 0)
FO_INITIAL = InitialCondition((0, 299), edge_fo=0.80, core_fo=0.90, steps_um=(STEP_UM,))
NI_INITIAL = ElementInitial(edge=0.40, core=0.30)  # wt% NiO, say


def d_profile(name, fo, orientation=ALONG_001):
    return DIFFUSIVITY_PRESETS[name].functions(CONDITIONS, orientation)[0](np.asarray(fo))


def stable_grid(*names, days=5.0, ratio=0.2):
    d_max = max(d_profile(n, [0.80, 0.90]).max() for n in names)
    return Grid(dx_um=DX_UM, dt_s=ratio * DX_M**2 / d_max, max_time_days=days)


def run(elements, grid, fo_initial=FO_INITIAL, orientation=ALONG_001, scheme="central"):
    start, advance = coupled_model(
        CONDITIONS, orientation, fo_initial, elements, grid, X_UM, DX_UM, scheme=scheme
    )
    return advance(start, grid.timesteps)


def erf_step(x_um, edge, core, d, t_s):
    return (edge + core) / 2 + (core - edge) / 2 * erf(
        (x_um - (STEP_UM - 0.5)) * 1e-6 / (2 * np.sqrt(d * t_s))
    )


def flux_form_pair(fo0, c0, d_fo, d_c, t_end_s, dx_m, safety=0.05):
    """Fo and an element with D_el(Fo), conservative scheme, D on the faces, fixed ends."""
    fo_range = np.array([fo0.min(), fo0.max()])
    d_max = max(d_fo(fo_range).max(), d_c(fo_range).max())
    n = int(np.ceil(t_end_s / (safety * dx_m**2 / d_max)))
    h = t_end_s / n
    fo, c = fo0.copy(), c0.copy()
    for _ in range(n):
        full_fo = np.concatenate([[fo0[0]], fo, [fo0[-1]]])
        full_c = np.concatenate([[c0[0]], c, [c0[-1]]])
        face_fo = 0.5 * (d_fo(full_fo[:-1]) + d_fo(full_fo[1:]))
        face_c = 0.5 * (d_c(full_fo[:-1]) + d_c(full_fo[1:]))
        fo = fo + h * np.diff(face_fo * np.diff(full_fo) / dx_m) / dx_m
        c = c + h * np.diff(face_c * np.diff(full_c) / dx_m) / dx_m
    return fo, c


@pytest.mark.parametrize("scheme", ["central", "dipra"])
def test_mn_with_the_fe_mg_law_stays_a_linear_function_of_fo(scheme):
    grid = stable_grid("dohmen_chakraborty2007")
    rows = run([("Mn", ElementInitial(0.4, 0.2), None)], grid, scheme=scheme)  # Mn = 2 - 2 Fo
    fo, mn = rows[:, :N_X], rows[:, N_X:]
    assert np.abs(mn - (2 - 2 * fo)).max() < 1e-12  # measured 1.5e-15 (central)
    assert np.abs(fo[-1] - fo[0]).max() > 0.01  # Fo did diffuse


@pytest.mark.parametrize("scheme", ["central", "dipra"])
def test_fo_rows_are_those_of_the_fo_model(scheme):
    grid = stable_grid("dohmen_chakraborty2007", "girona_costa2013_ni", days=1)
    rows = run([("Ni", NI_INITIAL, None)], grid, scheme=scheme)
    start, advance = _diffusion_model(
        scheme,
        CONDITIONS,
        ALONG_001,
        FO_INITIAL,
        grid,
        fe_mg_diffusivity("dohmen_chakraborty2007"),
        X_UM,
        DX_UM,
        fast=False,  # element fits use the NumPy steppers; the compiled ones agree to rounding
    )
    assert np.array_equal(rows[:, :N_X], advance(start, grid.timesteps))


def test_ni_in_unzoned_olivine_matches_the_erf_solution():
    flat = InitialCondition((0, 299), edge_fo=0.85, core_fo=0.85, steps_um=(STEP_UM,))
    grid = stable_grid("girona_costa2013_ni")
    rows = run([("Ni", NI_INITIAL, None)], grid, fo_initial=flat)
    d = d_profile("girona_costa2013_ni", [0.85])[0]
    exact = erf_step(X_UM, 0.40, 0.30, d, grid.timesteps * grid.dt_s)
    assert np.abs(rows[-1, N_X:] - exact).max() < 1e-5


def test_ca_does_not_need_fo_and_matches_the_erf_solution():
    orientation = Orientation(56.99, 76.75, 36.23)  # the AZ18 example's profile
    # Ca along this profile from the engine's own Ca function (not forster.diffusivity).
    d_ca = engine.D_Func_Ca(CONDITIONS.T_K, CONDITIONS.fO2_Pa, 56.99, 76.75, 36.23)
    # A time step unstable for Fe-Mg is fine, since Fo is not advanced for Ca alone.
    d_fo = d_profile("dohmen_chakraborty2007", [0.80], orientation)[0]
    grid = Grid(dx_um=DX_UM, dt_s=0.2 * DX_M**2 / d_ca, max_time_days=10)
    assert d_fo * grid.dt_s / DX_M**2 > 0.5
    rows = run([("Ca", ElementInitial(0.10, 0.20), None)], grid, orientation=orientation)
    assert np.array_equal(rows[-1, :N_X], rows[0, :N_X])  # Fo held
    exact = erf_step(X_UM, 0.10, 0.20, d_ca, grid.timesteps * grid.dt_s)
    assert np.abs(rows[-1, N_X:] - exact).max() < 1e-5


@pytest.mark.parametrize("scheme", ["central", "dipra"])
def test_ni_in_zoned_olivine_matches_an_independent_flux_form_solver(scheme):
    grid = stable_grid("dohmen_chakraborty2007", "girona_costa2013_ni")
    rows = run([("Ni", NI_INITIAL, None)], grid, scheme=scheme)
    fo0 = np.where(X_UM < STEP_UM - 0.5, 0.80, 0.90)
    ni0 = np.where(X_UM < STEP_UM - 0.5, 0.40, 0.30)

    def d_fo(fo):
        return d_profile("dohmen_chakraborty2007", fo)

    def d_ni(fo):
        return d_profile("girona_costa2013_ni", fo)

    t_end = grid.timesteps * grid.dt_s
    _, ni_ref = flux_form_pair(fo0, ni0, d_fo, d_ni, t_end, DX_M)
    # Measured 1.2e-4 (central) and 1.35e-4 (dipra) on a 0.1 step; the reference's own
    # grid error is 3-4e-4 (dx 1 against 0.5 um).
    assert np.abs(rows[-1, N_X:] - ni_ref).max() < 5e-4

    # Control: Ni with D frozen at the mean Fo misses the reference by far more, so the
    # comparison does test the coupling to the evolving Fo.
    flat = InitialCondition((0, 299), edge_fo=0.85, core_fo=0.85, steps_um=(STEP_UM,))
    frozen = run([("Ni", NI_INITIAL, None)], grid, fo_initial=flat, scheme=scheme)
    assert np.abs(frozen[-1, N_X:] - ni_ref).max() > 2 * 5e-4  # measured 1.7e-3


def test_an_unstable_time_step_for_the_element_is_an_error():
    # Paper conditions (1250 C, 0.5 GPa, QFM+0.3), where Ni diffuses 1.24x faster than
    # Fe-Mg at Fo 0.80: a time step stable for Fo is not stable for Ni.
    paper = Conditions(T_C=1250, P_Pa=5e8, fO2_Pa=8.3e-3)
    fo_range = np.array([0.80, 0.90])
    d_fo = DIFFUSIVITY_PRESETS["dohmen_chakraborty2007"].functions(paper, ALONG_001)[0]
    d_ni = DIFFUSIVITY_PRESETS["girona_costa2013_ni"].functions(paper, ALONG_001)[0]
    assert d_ni(fo_range).max() > 1.2 * d_fo(fo_range).max()
    grid = Grid(dx_um=DX_UM, dt_s=0.45 * DX_M**2 / d_fo(fo_range).max(), max_time_days=1)
    start = coupled_model(paper, ALONG_001, FO_INITIAL, [], grid, X_UM, DX_UM)[0]
    assert len(start) == N_X  # stable for Fo alone
    with pytest.raises(ValueError, match="Ni: D dt / dx\\^2"):
        coupled_model(paper, ALONG_001, FO_INITIAL, [("Ni", NI_INITIAL, None)], grid, X_UM, DX_UM)


# --- fits -------------------------------------------------------------------------

TRUE_DAYS = 4.0
POINTS_UM = np.arange(1.0, 299.0, 4.0)


@pytest.fixture(scope="module")
def synthetic():
    """Fo and Ni profiles made by the reference solver at TRUE_DAYS, with noise."""
    fo0 = np.where(X_UM < STEP_UM - 0.5, 0.80, 0.90)
    ni0 = np.where(X_UM < STEP_UM - 0.5, 0.40, 0.30)

    def d_fo(fo):
        return d_profile("dohmen_chakraborty2007", fo)

    def d_ni(fo):
        return d_profile("girona_costa2013_ni", fo)

    fo, ni = flux_form_pair(fo0, ni0, d_fo, d_ni, TRUE_DAYS * 86400, DX_M)
    rng = np.random.default_rng(7)
    fo_pts = np.interp(POINTS_UM, X_UM, fo) + rng.normal(0, 0.001, len(POINTS_UM))
    ni_pts = np.interp(POINTS_UM, X_UM, ni) + rng.normal(0, 0.002, len(POINTS_UM))
    return (
        Profile(POINTS_UM, fo_pts, name="synthetic"),
        ElementProfile(POINTS_UM, ni_pts, "Ni", name="synthetic"),
    )


FIT_GRID = Grid(dx_um=DX_UM, dt_s=60.0, max_time_days=30)


def fit_ni(profile, stop_early=True):
    return fit_element(
        profile, CONDITIONS, ALONG_001, FO_INITIAL, NI_INITIAL, FIT_GRID, stop_early=stop_early
    )


def test_fit_element_recovers_the_time_of_a_reference_profile(synthetic):
    result = fit_ni(synthetic[1])
    assert result.element == "Ni"
    assert result.time_days == pytest.approx(TRUE_DAYS, rel=0.05)
    assert result.time_days + result.time_minus_days < result.time_days
    assert result.time_days + result.time_plus_days > result.time_days


def test_early_stopping_gives_the_same_element_fit(synthetic):
    early, full = fit_ni(synthetic[1]), fit_ni(synthetic[1], stop_early=False)
    assert early.indices == full.indices
    assert (early.time_s, early.time_minus_s, early.time_plus_s) == (
        full.time_s,
        full.time_minus_s,
        full.time_plus_s,
    )
    assert len(early.curves) < len(full.curves)


def test_joint_fit_with_fo_alone_equals_fit_profile(synthetic, monkeypatch):
    monkeypatch.setattr(compiled, "_kernels", None)  # bit for bit: both on the NumPy steppers
    fo_profile = synthetic[0]
    joint = fit_joint(fo_profile, [], CONDITIONS, ALONG_001, FO_INITIAL, {}, FIT_GRID)
    single = fit_profile(
        fo_profile, CONDITIONS, ALONG_001, FO_INITIAL, FIT_GRID, scheme="central"
    )  # element fits use the central scheme
    assert joint.indices == single.indices
    assert (joint.time_s, joint.time_minus_s, joint.time_plus_s) == (
        single.time_s,
        single.time_minus_s,
        single.time_plus_s,
    )
    assert np.array_equal(joint.chi2, single.sum_chi2)


def test_joint_fit_of_fo_and_ni_is_at_least_as_tight_as_either(synthetic):
    fo_profile, ni_profile = synthetic
    fo = fit_profile(fo_profile, CONDITIONS, ALONG_001, FO_INITIAL, FIT_GRID)
    ni = fit_ni(ni_profile)
    joint = fit_joint(
        fo_profile, [ni_profile], CONDITIONS, ALONG_001, FO_INITIAL, {"Ni": NI_INITIAL}, FIT_GRID
    )
    full = fit_joint(
        fo_profile,
        [ni_profile],
        CONDITIONS,
        ALONG_001,
        FO_INITIAL,
        {"Ni": NI_INITIAL},
        FIT_GRID,
        stop_early=False,
    )
    assert joint.indices == full.indices
    assert set(joint.parts) == {"Fo", "Ni"}
    assert joint.time_days == pytest.approx(TRUE_DAYS, rel=0.05)
    width = joint.high_days - joint.low_days
    assert width <= min(
        fo.time_plus_days - fo.time_minus_days, ni.time_plus_days - ni.time_minus_days
    )


def test_input_errors_name_the_problem(synthetic):
    with pytest.raises(ValueError, match="element 'Fe'"):
        ElementProfile(POINTS_UM, synthetic[1].values, "Fe")
    with pytest.raises(ValueError, match="is for Ni, not Mn"):
        element_diffusivity("Mn", "girona_costa2013_ni")
    two_step_fo = InitialCondition((0, 299), 0.80, 0.90, steps_um=(50, 150), step_fo=0.85)
    with pytest.raises(ValueError, match="taken from the Fo initial condition"):
        fit_element(synthetic[1], CONDITIONS, ALONG_001, two_step_fo, NI_INITIAL, FIT_GRID)
    with pytest.raises(ValueError, match="once in a joint fit"):
        fit_joint(
            None,
            [synthetic[1], synthetic[1]],
            CONDITIONS,
            ALONG_001,
            FO_INITIAL,
            {"Ni": NI_INITIAL},
            FIT_GRID,
        )


# --- fitted concentrations and consistency ----------------------------------------


@pytest.mark.parametrize("scheme", ["central", "dipra"])
def test_element_runs_are_linear_in_their_concentrations(scheme):
    """The basis of FittedInitial: for a given Fo history an element's run at any
    concentrations is the constant core value plus unit runs."""
    two_step_fo = InitialCondition((0, 299), 0.80, 0.90, steps_um=(60, 150), step_fo=0.86)
    grid = stable_grid("dohmen_chakraborty2007", "girona_costa2013_ni", days=2)
    given = ElementInitial(0.40, 0.30, step=0.45)
    unit_edge = ElementInitial(1.0, 0.0, step=0.0)
    unit_step = ElementInitial(0.0, 0.0, step=1.0)
    rows = run(
        [("Ni", given, None), ("Ni", unit_edge, None), ("Ni", unit_step, None)],
        grid,
        fo_initial=two_step_fo,
        scheme=scheme,
    )
    ni, phi_edge, phi_step = (rows[:, k * N_X : (k + 1) * N_X] for k in (1, 2, 3))
    combined = 0.30 + (0.40 - 0.30) * phi_edge + (0.45 - 0.30) * phi_step
    assert np.abs(ni - combined).max() < 1e-12


def test_fitted_concentrations_recover_the_starting_profile(synthetic):
    result = fit_element(synthetic[1], CONDITIONS, ALONG_001, FO_INITIAL, None, FIT_GRID)
    fitted = result.inputs["element_initial"]
    assert result.inputs["concentrations_fitted"]
    assert result.time_days == pytest.approx(TRUE_DAYS, rel=0.1)
    assert fitted.edge == pytest.approx(0.40, abs=0.005)
    assert fitted.core == pytest.approx(0.30, abs=0.005)
    # Fitting the concentrations widens the range relative to giving the true ones.
    given = fit_ni(synthetic[1])
    width = result.time_plus_days - result.time_minus_days
    assert width >= given.time_plus_days - given.time_minus_days


def test_early_stopping_gives_the_same_fit_with_fitted_concentrations(synthetic):
    args = (synthetic[1], CONDITIONS, ALONG_001, FO_INITIAL, None, FIT_GRID)
    early, full = fit_element(*args), fit_element(*args, stop_early=False)
    assert early.indices == full.indices
    assert (early.time_s, early.time_minus_s, early.time_plus_s) == (
        full.time_s,
        full.time_minus_s,
        full.time_plus_s,
    )


def test_consistency_accepts_one_time_and_flags_two(synthetic):
    fo_profile, ni_profile = synthetic
    same = fit_joint(fo_profile, [ni_profile], CONDITIONS, ALONG_001, FO_INITIAL, None, FIT_GRID)
    assert same.consistency_p > 0.05
    assert set(same.species_days) == {"Fo", "Ni"}
    # Ni made at four times the Fo time: the joint fit must say they disagree.
    fo0 = np.where(X_UM < STEP_UM - 0.5, 0.80, 0.90)
    ni0 = np.where(X_UM < STEP_UM - 0.5, 0.40, 0.30)

    def d_fo(fo):
        return d_profile("dohmen_chakraborty2007", fo)

    def d_ni(fo):
        return d_profile("girona_costa2013_ni", fo)

    _, ni_late = flux_form_pair(fo0, ni0, d_fo, d_ni, 4 * TRUE_DAYS * 86400, DX_M)
    rng = np.random.default_rng(11)
    late = ElementProfile(
        POINTS_UM, np.interp(POINTS_UM, X_UM, ni_late) + rng.normal(0, 0.002, len(POINTS_UM)), "Ni"
    )
    with pytest.warns(UserWarning, match="do not agree on one time"):
        apart = fit_joint(fo_profile, [late], CONDITIONS, ALONG_001, FO_INITIAL, None, FIT_GRID)
    assert apart.consistency_p < 0.01
    assert apart.species_days["Ni"] > 2 * apart.species_days["Fo"]


def test_a_profile_without_zoning_is_reported_as_unconstrained():
    rng = np.random.default_rng(3)
    flat = ElementProfile(POINTS_UM, 0.35 + rng.normal(0, 0.002, len(POINTS_UM)), "Ni")
    with pytest.warns(UserWarning, match="does not constrain a diffusion time"):
        fit_element(flat, CONDITIONS, ALONG_001, FO_INITIAL, ElementInitial(0.35, 0.35), FIT_GRID)


def test_fo_diffuses_when_it_is_fitted_with_ca_alone(synthetic):
    """Ca does not need Fo, but a joint fit of Fo and Ca must still advance Fo."""
    fo_profile = synthetic[0]
    ca = ElementProfile(
        POINTS_UM, 0.15 + 0.0 * POINTS_UM + np.linspace(0, 0.01, len(POINTS_UM)), "Ca"
    )
    joint = fit_joint(fo_profile, [ca], CONDITIONS, ALONG_001, FO_INITIAL, None, FIT_GRID)
    fo_alone = fit_profile(fo_profile, CONDITIONS, ALONG_001, FO_INITIAL, FIT_GRID)
    assert joint.species_days["Fo"] == pytest.approx(fo_alone.time_days, rel=0.02)


def test_element_runs_accept_first_row_and_reject_other_schemes():
    grid = stable_grid("dohmen_chakraborty2007", "girona_costa2013_ni", days=0.5)
    start, advance = coupled_model(
        CONDITIONS, ALONG_001, FO_INITIAL, [("Ni", NI_INITIAL, None)], grid, X_UM, DX_UM
    )
    assert np.array_equal(advance(start, 5), advance(start, 5, first_row=100))
    with pytest.raises(ValueError, match=r"'implicit'.*element fits use central or dipra"):
        coupled_model(
            CONDITIONS,
            ALONG_001,
            FO_INITIAL,
            [("Ni", NI_INITIAL, None)],
            grid,
            X_UM,
            DX_UM,
            scheme="implicit",
        )
