"""`forster.joint`: one time fitted to the profiles of one crystal.

Checked against the single-profile fits it generalises (a crystal of one profile), and
on a synthetic crystal: two profiles made at one known time from shared segment Fo,
with their own steps and orientations (so their own D), and white noise; then with a
measurement drift on one profile, and with one profile's distances shifted.
"""

import warnings
from dataclasses import replace

import numpy as np
import pytest

from forster import (
    Grid,
    ICPosterior,
    InitialCondition,
    Orientation,
    Profile,
    Stage,
    Uncertainty,
    fit_profile,
    fit_run,
    read_profiles,
    read_runs,
    scan_run,
    write_template,
)
from forster import joint as J
from forster.joint import fit_crystal
from forster.model import _diffusion_model, scheme_grid

TRUE_DAYS = 5.0
TRUE_FO = (0.912, 0.8985)  # segments a (edge) and b (core), shared by both profiles
NOISE = 0.0007
PRIORS = {"step1_um": Uncertainty(3.0), "a_fo": Uncertainty(0.002), "b_fo": Uncertainty(0.002)}


@pytest.fixture(scope="module")
def template(tmp_path_factory):
    out = tmp_path_factory.mktemp("template")
    write_template(out)
    (run,) = read_runs(out / "parameters.csv")
    return run, read_profiles(out / "profiles.csv")


def _measured(run, initial, orientation, x, seed, shift_um=0.0, drift=0.0):
    """Fo at ``x`` after TRUE_DAYS from ``initial`` (a fine implicit run) plus white
    noise; the distances reported shifted by ``shift_um`` and the Fo by ``drift``."""
    grid = Grid(0.25, TRUE_DAYS * 86400 / 400, 2 * TRUE_DAYS)
    nodes, dx, (low, high) = scheme_grid("implicit", x, grid, initial.domain_um)
    model_x = nodes[low : high + 1]
    start, advance = _diffusion_model(
        "implicit", run.conditions, orientation, initial, grid, run.diffusivity, model_x, dx
    )
    fo = np.interp(x, model_x, advance(start, 400)[400])
    fo = fo + np.random.default_rng(seed).normal(0, NOISE, len(x)) + drift
    return x + shift_um, fo


def _crystal(template, shift_um=0.0, drift=0.0, priors=PRIORS, seed=0):
    """Runs A and B of one synthetic crystal and their profiles. A lies along the template
    profile's direction (near [001]), B near [100] (D 3.8x smaller), with its own step;
    the segments are shared. ``shift_um`` and ``drift`` apply to B."""
    run, _ = template
    orient = {"A": run.orientation, "B": Orientation(10.0, 85.0, 81.36)}
    initial = {
        "A": InitialCondition((0.0, 80.0), steps_um=(30.0,), segment_fo=TRUE_FO),
        "B": InitialCondition((0.0, 60.0), steps_um=(22.0,), segment_fo=TRUE_FO),
    }
    xs = {"A": np.linspace(0.5, 80, 40), "B": np.linspace(0.5, 60, 40)}
    profiles, runs = {}, []
    for k, name in enumerate("AB"):
        shift, dfo = (shift_um, drift) if name == "B" else (0.0, 0.0)
        x, fo = _measured(run, initial[name], orient[name], xs[name], seed + k + 1, shift, dfo)
        profiles[name] = Profile(x, fo, name=name)
        runs.append(
            replace(
                run,
                name=name,
                profile_name=name,
                orientation=orient[name],
                initial=initial[name],
                uncertainties=priors,
                metadata={"crystal": "X1"},
            )
        )
    return runs, profiles


def _alone(run, profiles, time_days) -> ICPosterior:
    """The run's own starting-profile fit (`scan_run` with the sampler)."""
    fit = scan_run(run, profiles, time_days=time_days)
    assert isinstance(fit, ICPosterior)
    return fit


def _quiet_fit(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit_crystal(*args, **kwargs)


@pytest.fixture(scope="module")
def crystal(template):
    runs, profiles = _crystal(template)
    fits = {r.name: fit_run(r, profiles) for r in runs}
    return runs, profiles, fits


@pytest.fixture(scope="module")
def plain(crystal):
    """The joint fit with the offsets off (the individual fits' likelihood)."""
    runs, profiles, fits = crystal
    return fit_crystal(runs, profiles, fits=fits, fo_offset_sd=0.0, edge_offset_sd_um=0.0)


@pytest.fixture(scope="module")
def with_offsets(crystal):
    runs, profiles, fits = crystal
    return fit_crystal(runs, profiles, fits=fits)


def test_a_crystal_of_one_profile_is_its_starting_profile_fit(template):
    """Same likelihood, optimiser and seed: the same best values and noise model, and the
    range within sampling error (the joint fit refines the Laplace proposal)."""
    run, profiles = template
    run = replace(run, uncertainties={"step1_um": Uncertainty(2.0), "a_fo": Uncertainty(0.002)})
    nominal = fit_run(run, profiles)
    single = _alone(run, profiles, nominal.time_days)
    joint = fit_crystal(
        [run], profiles, fits={run.name: nominal}, fo_offset_sd=0.0, edge_offset_sd_um=0.0
    )
    assert joint.best_values == {
        f"{run.name}:step1_um": single.best_values["step1_um"],
        "a_fo": single.best_values["a_fo"],
    }
    assert joint.best_days == single.best_days
    assert joint.noise[run.name] == single.noise
    assert joint.grids[run.name] == single.grid
    assert joint.range_days == pytest.approx(single.range_days, rel=0.01)
    assert joint.median_days == pytest.approx(single.median_days, rel=0.01)
    assert joint.method == single.method == "laplace"


def test_a_fixed_starting_profile_gives_fit_profiles_time(template):
    """Nothing free: the joint time is `fit_profile`'s on the same grid, and the range by
    `fit_profile`'s rule (the small-sample threshold) is its range. The posterior's
    central range, the starting-profile fit's convention, sits a few percent later."""
    run, profiles = template
    grid = fit_run(run, profiles).inputs["grid"]
    joint = fit_crystal([run], profiles, grid=grid, fo_offset_sd=0.0, edge_offset_sd_um=0.0)
    fit = fit_profile(
        profiles[run.profile_name],
        run.conditions,
        run.orientation,
        run.initial,
        grid,
        run.kriging,
        run.diffusivity,
        run.scheme,
    )
    assert joint.method == "fixed" and joint.names == ()
    assert joint.best_days == fit.time_days
    assert joint.threshold_range_days == pytest.approx(fit.time_range_days(), rel=1e-9)
    assert joint.threshold_range2_days == pytest.approx(fit.time_range_days(sigmas=2), rel=1e-9)
    print("fixed start: threshold range", joint.threshold_range_days, "central", joint.range_days)


def _width(r):
    return r[1] - r[0]


def test_two_profiles_recover_the_time_and_the_shared_fo(crystal, plain):
    """The joint 2-sigma range covers the true time and the shared segments come back. On
    one noise draw the joint range need not be narrower than the narrower individual one
    (in a toy model it was wider in 2 of 12 draws, when the slow profile alone gave a
    falsely narrow range): the narrowing is tested as a median over draws below; here only
    a loose bound guards against a broken joint range."""
    runs, profiles, fits = crystal
    single = {r.name: _alone(r, profiles, fits[r.name].time_days) for r in runs}
    assert plain.agree and plain.crystal == "X1" and plain.method == "laplace"
    assert plain.low2_days < TRUE_DAYS < plain.high2_days
    narrowest = min(_width(s.range_days) for s in single.values())
    assert _width(plain.range_days) <= 1.5 * narrowest
    for label, truth in zip("ab", TRUE_FO, strict=True):
        assert abs(plain.segment_fo[label] - truth) < 3 * plain.segment_fo_sd[label]
    assert set(plain.steps_um) == {"A", "B"} and plain.fo_offset == {"A": 0.0, "B": 0.0}
    print(
        f"joint {plain.best_days:.3f} d ({plain.low_days:.3f}-{plain.high_days:.3f}); "
        + "; ".join(f"{n} {s.best_days:.3f} d {s.range_days}" for n, s in single.items())
    )


def test_fixed_joint_range_is_narrower_than_either_profiles_fit_over_draws(template):
    """With fixed starting profiles (the individual `fit_run` fits' setting), over 8 noise
    draws the joint range by the same rule is narrower than the narrower individual range
    in the median (a property of the method, not of every draw)."""
    ratios = []
    for seed in range(0, 80, 10):
        runs, profiles = _crystal(template, priors={}, seed=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fits = {r.name: fit_run(r, profiles) for r in runs}
            joint = fit_crystal(runs, profiles, fits=fits, fo_offset_sd=0.0, edge_offset_sd_um=0.0)
        assert joint.method == "fixed"
        narrowest = min(_width(f.time_range_days()) for f in fits.values())
        ratios.append(_width(joint.threshold_range_days) / narrowest)
    print("joint / narrowest individual width:", np.round(ratios, 2))
    assert np.median(ratios) < 1.0


def test_the_offsets_widen_the_shared_fo_but_keep_the_time(plain, with_offsets):
    assert with_offsets.median_days == pytest.approx(plain.median_days, rel=0.02)
    for label in "ab":
        assert with_offsets.segment_fo_sd[label] > plain.segment_fo_sd[label]
    # Defaults: one profile is the drift reference (the one with most points, here the
    # first of two equal ones), and edge offsets are off.
    offsets = {n for n in with_offsets.names if n.endswith("offset") or n.endswith("offset_um")}
    assert offsets == {"B:fo_offset"} and with_offsets.fo_offset["A"] == 0.0


def test_a_drift_is_fitted_by_the_fo_offset(template, crystal):
    """A 0.15 mol% offset on profile B. With the Fo offsets on, the time stays where the
    undrifted crystal puts it and the offsets differ by about the drift (with A as the
    reference, B's offset is the drift); with them off, the time moves. Edge offsets off:
    they do not bear on the drift, and with free steps they push the sampler to its
    slow fallback (nested sampling, minutes)."""
    runs0, profiles0, fits0 = crystal
    base = fit_crystal(runs0, profiles0, fits=fits0, edge_offset_sd_um=0.0).median_days
    runs, profiles = _crystal(template, drift=0.0015)
    fits = {r.name: fit_run(r, profiles) for r in runs}
    on = _quiet_fit(runs, profiles, fits=fits, edge_offset_sd_um=0.0)
    ref = _quiet_fit(runs, profiles, fits=fits, fo_offset_sd={"A": 0.0}, edge_offset_sd_um=0.0)
    off = _quiet_fit(runs, profiles, fits=fits, fo_offset_sd=0.0, edge_offset_sd_um=0.0)
    assert on.method == ref.method == "laplace"
    assert on.median_days == pytest.approx(base, rel=0.02)
    assert on.fo_offset["B"] - on.fo_offset["A"] == pytest.approx(0.0015, abs=0.0003)
    assert abs(ref.fo_offset["B"] - 0.0015) < 2 * ref.sd["B:fo_offset"]
    assert abs(off.median_days - base) > abs(on.median_days - base)
    print(
        f"drift: no drift {base:.3f} d, offsets on {on.median_days:.3f} d (B-A "
        f"{on.fo_offset['B'] - on.fo_offset['A']:.5f}), A as reference: B "
        f"{ref.fo_offset['B']:.5f} +- {ref.sd['B:fo_offset']:.5f}, offsets off "
        f"{off.median_days:.3f} d"
    )


def test_a_shifted_profile_is_fitted_by_the_edge_offset(template):
    """B's distances reported 2 um too large: its fitted edge offset is about 2 um, A's
    about 0. Steps held: a free step absorbs most of a shift (it is nearly the same
    change), leaving the offset to its prior."""
    priors = {"a_fo": Uncertainty(0.002), "b_fo": Uncertainty(0.002)}
    runs, profiles = _crystal(template, shift_um=2.0, priors=priors)
    joint = _quiet_fit(runs, profiles, edge_offset_sd_um=J.DEFAULT_EDGE_OFFSET_SD_UM)
    b, a = joint.edge1_offset_um["B"], joint.edge1_offset_um["A"]
    assert abs(b - 2.0) < 1.5 * joint.sd["B:edge1_offset_um"]
    assert abs(a) < 1.5 * joint.sd["A:edge1_offset_um"]
    assert joint.steps_um == {"A": (30.0,), "B": (22.0,)}
    print(f"edge offsets: B {b:.2f} +- {joint.sd['B:edge1_offset_um']:.2f} um, A {a:.2f}")


def test_seeded_runs_reproduce_exactly(crystal, with_offsets):
    runs, profiles, fits = crystal
    again = fit_crystal(runs, profiles, fits=fits)
    assert again.best_values == with_offsets.best_values
    assert again.mean == with_offsets.mean and again.sd == with_offsets.sd
    assert (again.low_days, again.high_days) == (with_offsets.low_days, with_offsets.high_days)
    assert np.array_equal(again.density, with_offsets.density)
    assert again.model_runs == with_offsets.model_runs


def test_nested_sampling_runs_on_the_joint_likelihood(crystal, plain):
    runs, profiles, fits = crystal
    nested = fit_crystal(
        runs,
        profiles,
        fits=fits,
        method="nested",
        nlive=60,
        fo_offset_sd=0.0,
        edge_offset_sd_um=0.0,
    )
    assert nested.method == "nested"
    assert nested.median_days == pytest.approx(plain.median_days, rel=0.05)


def test_profiles_that_disagree_are_reported(template, crystal):
    """B fitted with A's orientation: the individual times part, which warns."""
    runs, profiles, fits = crystal
    wrong = replace(runs[1], orientation=runs[0].orientation)
    with pytest.warns(UserWarning, match="do not agree on one time"):
        agree = J._check_agreement({"A": fits["A"], "B": fit_run(wrong, profiles)}, "X1")
    assert not agree


def test_runs_that_cannot_be_fitted_jointly_are_errors(crystal):
    runs, profiles, _ = crystal
    a, b = runs
    hot = replace(b, conditions=replace(b.conditions, T_C=b.conditions.T_C + 10))
    with pytest.raises(ValueError, match=r"identical T, P and fO2.*A .*B "):
        fit_crystal([a, hot], profiles)
    loose = replace(b, uncertainties={**PRIORS, "b_fo": Uncertainty(0.004)})
    with pytest.raises(ValueError, match=r"segment 'b' is shared by runs that disagree.*A:.*B:"):
        fit_crystal([a, loose], profiles)
    richer = replace(b, initial=replace(b.initial, segment_fo=(0.912, 0.9)))
    with pytest.raises(ValueError, match="segment 'b' is shared"):
        fit_crystal([a, richer], profiles)
    staged = replace(b, stages=(Stage(86400.0, 1.0, 0.91),))
    with pytest.raises(ValueError, match="run 'B' has growth stages"):
        fit_crystal([a, staged], profiles)
    with pytest.raises(ValueError, match="different crystals"):
        fit_crystal([a, replace(b, metadata={"crystal": "X2"})], profiles)
    with pytest.raises(ValueError, match="no segment"):
        fit_crystal([a, b], profiles, own_segments={"B": ["c"]})
    with pytest.raises(ValueError, match="fo_offset_sd must be 0"):
        fit_crystal([a, b], profiles, fo_offset_sd=-1.0)


def test_own_segments_free_a_profiles_core(crystal):
    """A core B fits on its own may differ from A's; it gets its own parameter."""
    runs, profiles, fits = crystal
    a, b = runs
    richer = replace(b, initial=replace(b.initial, segment_fo=(0.912, 0.9)))
    own = {"B": ("b",)}
    J._check_shared([a, richer], own)  # no error
    grids = J._grids([a, richer], profiles, fits, Grid(2.0, 4000.0, 15.0))
    like = J._likelihood([a, richer], profiles, grids, own, {"A": 0, "B": 0}, {"A": 0, "B": 0})
    assert like.names == ("A:step1_um", "a_fo", "b_fo", "B:step1_um", "B:b_fo")
    assert like.nominal[-1] == 0.9 and like.nominal[2] == 0.8985


def test_a_label_for_different_zones_warns(crystal):
    """Labels are matched by name: a label that is one profile's core and another's
    middle segment is probably two zones."""
    runs, _, _ = crystal
    a, b = runs
    three = replace(
        b,
        initial=InitialCondition(
            (0.0, 60.0), steps_um=(15.0, 22.0), segment_fo=(0.912, 0.8985, 0.89)
        ),
        uncertainties={},
    )
    fixed_a = replace(a, uncertainties={})
    with pytest.warns(UserWarning, match="segment 'b' is shared by profiles that use it for"):
        J._check_shared([fixed_a, three], {})


def test_every_fo_offset_free_without_a_reference(crystal):
    """``fo_reference=None`` keeps both profiles' Fo offsets (only their difference is
    fixed by the data); a named reference picks that run."""
    runs, profiles, fits = crystal
    free = _quiet_fit(runs, profiles, fits=fits, fo_reference=None, method="laplace")
    assert {n for n in free.names if n.endswith("fo_offset")} == {"A:fo_offset", "B:fo_offset"}
    named = _quiet_fit(runs, profiles, fits=fits, fo_reference="B", method="laplace")
    assert {n for n in named.names if n.endswith("fo_offset")} == {"A:fo_offset"}
    with pytest.raises(ValueError, match="fo_reference: no run 'C'"):
        fit_crystal(runs, profiles, fits=fits, fo_reference="C")
