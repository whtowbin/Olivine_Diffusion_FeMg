"""Analyses behind the noise-findings figures (make_figures.py in this folder).

Two questions, each answered on synthetic profiles and on published Towbin and Plank
(2026) profiles (round robin profiles only with ``--private``):

1. Initial condition (IC). Does whitening (scoring a fit against a correlated noise
   model) flatten the likelihood of the IC parameters, and how far does fitting the IC
   (`forster.sampling.fit_initial_condition`) move the time?
   - `sweep`: one IC parameter at a time, scored four ways (time profiled in each):
     (a) white:    -n/2 log RSS (no correlated noise);
     (b) fixedC:   -n/2 log r^T C^-1 r, C fitted once at the nominal IC;
     (c) refitC:   the same with C refitted at the best (b) candidate (FORSTER's second
                   pass, `fit_initial_condition`);
     (d) profiled: the Gaussian log likelihood (log det included) maximised over the
                   noise parameters (s, rho, L) for every candidate and time step.
   - `ic_recovery_trial`: synthetic profiles started from a WRONG nominal IC; how often
     the 1-sigma interval of each scoring covers the true value.
   - `fit_ic`: FORSTER's joint IC fit, and the default fit at the nominal and at the
     fitted IC.
2. Plateau. Does the length of the measured core plateau bias the time or its range?
   - `plateau_trial`: synthetic profiles cut at 3, 5, 10 and 20 diffusion lengths past
     the step;
   - `plateau_cuts`: real profiles with core-side points dropped progressively.

Ported from the experiment scripts of 2026-10-16 (scratchpad ic_misfit/ and plateau/);
the noise model and near-step model are those of `forster.uncertainty` and
docs/plans/2026-10-15-whitening-validity/zone_noise.py.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import itertools  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import warnings  # noqa: E402
from dataclasses import dataclass, replace  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.linalg as sla  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
STUDY = REPO / "docs" / "plans" / "2026-10-15-whitening-validity"
PRIVATE_OUT = REPO / "Diffusion Round Robin" / "outputs" / "noise_findings"
sys.path.insert(0, str(STUDY))
sys.path.insert(0, str(REPO / "tests"))

from synthetic import make_noise  # noqa: E402
from zone_noise import _ranges, bump, fit_zone_noise  # noqa: E402

import forster  # noqa: E402
import forster.sampling as fs  # noqa: E402
from forster.grid import choose_grid, diffusion_length_um, fo_range  # noqa: E402
from forster.model import SECONDS_PER_DAY, Grid, InitialCondition, Profile  # noqa: E402
from forster.scan import NORMAL_SPAN, Uncertainty, parameter_names  # noqa: E402
from forster.uncertainty import (  # noqa: E402
    TWO_SIGMA,
    NoiseModel,
    delta_chi2_interval,
    fit_correlated_points,
    fit_noise_model,
    interpolation_matrix,
    interval_threshold,
    near_step_check,
    points_in_domain,
    white_rss,
)

DAY = SECONDS_PER_DAY
ZONE = 2.0  # "near the step" = within 2 diffusion lengths
NEAR_STEP_WARN = -6.0  # forster.uncertainty.NEAR_STEP_AIC

# ----------------------------------------------------------------------------- loading


def load_paper():
    from conftest import PAPER_DATA, PAPER_PARAMS

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runs = {r.name: r for r in forster.read_runs(PAPER_PARAMS)}
        return runs, forster.read_profiles(PAPER_DATA, sheet="Olivine")


def load_round_robin():
    from conftest import HAS_ROUND_ROBIN, ROUND_ROBIN_DATA, ROUND_ROBIN_PARAMS

    if not HAS_ROUND_ROBIN:
        raise SystemExit("the round robin submodule is not checked out")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runs = {}
        for r in forster.read_runs(ROUND_ROBIN_PARAMS,
                                   defaults=forster.RunDefaults(max_time_days=730)):  # fmt: skip
            runs.setdefault(r.name, r)
        return runs, forster.read_profiles(ROUND_ROBIN_DATA)


def template_run():
    with tempfile.TemporaryDirectory() as tmp:
        forster.write_template(tmp)
        return forster.read_runs(Path(tmp) / "parameters.csv")[0]


# ------------------------------------------------------------------ noise and scoring

# Fine rho grid for the profiled score (d): the eigen trick makes rho nearly free, and a
# fine grid keeps the curves over an IC parameter from jumping between grid nodes.
RHO_FINE = np.concatenate([[0.0], np.geomspace(0.03, 30.0, 64)])
N_LENGTHS_FINE = 24


class NoiseBasis:
    """Eigen-decompositions of K_L for a grid of lengths: for B = I + rho^2 K_L,
    r^T B^-1 r = sum (U^T r)^2 / (1 + rho^2 lam) and log|B| = sum log(1 + rho^2 lam),
    so every rho costs one projection per L. Same likelihood as fit_noise_model."""

    def __init__(self, x, n_lengths=N_LENGTHS_FINE, rho=RHO_FINE):
        n = len(x)
        spacing = float(np.median(np.diff(np.sort(x))))
        span = float(x.max() - x.min())
        self.lengths = np.geomspace(max(spacing, 1e-3), max(span / 2, 2 * spacing), n_lengths)
        self.rho = np.asarray(rho)
        self.n = n
        d2 = (x[:, None] - x[None, :]) ** 2
        self.U, self.inv, self.logdet = [], [], []
        for L in self.lengths:
            lam, U = np.linalg.eigh(np.exp(-0.5 * d2 / L**2))
            lam = np.clip(lam, 0, None)
            f = 1.0 + self.rho[None, :] ** 2 * lam[:, None]  # n x nrho
            self.U.append(U)
            self.inv.append(1.0 / f)
            self.logdet.append(np.log(f).sum(axis=0))

    def loglik(self, R):
        """R: (rows, n) residuals -> (rows, nL, nrho) log likelihood profiled over s
        (constant dropped): -n/2 log(r^T B^-1 r) - 1/2 log|B|."""
        out = np.empty((R.shape[0], len(self.lengths), len(self.rho)))
        for j in range(len(self.lengths)):
            z2 = (R @ self.U[j]) ** 2
            chi2 = np.maximum(z2 @ self.inv[j], 1e-300)
            out[:, j, :] = -0.5 * self.n * np.log(chi2) - 0.5 * self.logdet[j][None, :]
        return out

    def best(self, R):
        """Max over rows and the noise grid: (log likelihood, row)."""
        ll = self.loglik(R)
        k = np.unravel_index(int(np.argmax(ll)), ll.shape)
        return float(ll[k]), int(k[0])


def chol_of(noise: NoiseModel, x):
    return np.linalg.cholesky(noise.covariance(x))


def fixed_rows(M, y, chol):
    """-n/2 log chi2 at every row for a fixed covariance factor."""
    z = sla.solve_triangular(chol, (M - y).T, lower=True)
    return -0.5 * len(y) * np.log(np.sum(z**2, axis=0))


def white_rows(M, y):
    r = M - y
    return -0.5 * len(y) * np.log(np.einsum("kp,kp->k", r, r))


def nominal_noise(x, y, M):
    """Noise model at the white best row of M (as ICLikelihood.fit_noise)."""
    best = int(np.argmin(np.sum((M - y) ** 2, axis=1)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit_noise_model(x, y - M[best])


def width_from_curve(v, ll, drop=0.5):
    """Best value, interval where ll >= max - drop (linear interpolation; 0.5 is the
    1-sigma rule for one parameter), whether it reaches the end of the sweep, and the
    sub-grid best value from a quadratic through the top 5 points."""
    v, ll = np.asarray(v, float), np.asarray(ll, float)
    ok = np.isfinite(ll)
    v, ll = v[ok], ll[ok]
    o = np.argsort(v)
    v, ll = v[o], ll[o]
    k = int(np.argmax(ll))
    level = ll[k] - drop
    lo_open = hi_open = False
    i = k
    while i > 0 and ll[i - 1] >= level:
        i -= 1
    if i == 0:
        lo, lo_open = v[0], True
    else:
        lo = v[i - 1] + (v[i] - v[i - 1]) * (level - ll[i - 1]) / (ll[i] - ll[i - 1])
    j = k
    while j < len(v) - 1 and ll[j + 1] >= level:
        j += 1
    if j == len(v) - 1:
        hi, hi_open = v[-1], True
    else:
        hi = v[j + 1] + (v[j] - v[j + 1]) * (level - ll[j + 1]) / (ll[j] - ll[j + 1])
    a0, a1 = max(0, k - 2), min(len(v), k + 3)
    best_q = v[k]
    if a1 - a0 >= 3:
        c = np.polyfit(v[a0:a1], ll[a0:a1], 2)
        if c[0] < 0:
            bq = -c[1] / (2 * c[0])
            if v[a0] <= bq <= v[a1 - 1]:
                best_q = float(bq)
    return {"best": float(v[k]), "best_q": float(best_q), "lo": float(lo), "hi": float(hi),
            "half_width": float(0.5 * (hi - lo)), "open": bool(lo_open or hi_open)}  # fmt: skip


# ------------------------------------------------------------------- IC: real profiles

# Fitted times (days) at the nominal IC, from the noise-models guide's tables
# (real_examples.csv, rr_examples.csv): they size each profile's scan grid.
NOMINAL_DAYS = {
    "SH63olv066-1": 10.817,
    "Xenocryst_ol6_prof_1": 0.14792,
    "AZ18 WHT06_ol43_prof 1": 1.9229,
    "AZ18 WHT06_ol48_prof 1": 3.0389,
}
# Prior spreads (1 sigma; the prior is a Gaussian cut at +-2.5 sigma) of the joint fits.
SPREADS = {
    "SH63olv066-1": {"step1_um": 2.0, "step2_um": 2.0, "edge_fo": 0.02, "core_fo": 0.01,
                     "step_fo": 0.02},
    "AZ18 WHT06_ol43_prof 1": {"step1_um": 3.0, "step2_um": 2.0, "edge_fo": 0.02,
                               "core_fo": 0.01, "step_fo": 0.01},
    "Xenocryst_ol6_prof_1": {"step1_um": 2.0, "step2_um": 2.0, "edge_fo": 0.02,
                             "core_fo": 0.01, "step_fo": 0.01},
    "AZ18 WHT06_ol48_prof 1": {"step1_um": 2.0, "edge_fo": 0.02, "core_fo": 0.01},
}  # fmt: skip


@dataclass
class Case:
    name: str
    run: object
    profile: object
    like: fs.ICLikelihood
    grid: Grid

    @property
    def x(self):
        return self.like.x

    @property
    def y(self):
        return self.like.y

    @property
    def names(self):
        return self.like.names

    @property
    def nominal(self):
        return self.like.nominal

    def days(self, row):
        return float(row) * self.grid.dt_s / DAY

    def ell(self, row):
        return diffusion_length_um(self.days(max(row, 1)),
                                   [(self.run.diffusivity, fo_range(self.run.initial))],
                                   self.run.conditions, self.run.orientation)  # fmt: skip

    def at_points(self, theta):
        ic = self.like.candidate(np.asarray(theta, float))
        if ic is None:
            return None, None
        return self.like.at_points(ic), ic


def make_case(name, runs, profiles, spreads=None, factor=3.0) -> Case:
    """The profile's run on a scan grid to ``factor`` x its nominal time, with an
    ICLikelihood over every IC parameter (spreads: 2 um for steps, 1 mol% for Fo,
    unless given)."""
    run = runs[name]
    profile = profiles[run.profile_name]
    t = NOMINAL_DAYS[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid = choose_grid(profile, run.conditions, run.orientation, run.initial,
                           run.diffusivity, purpose="scan", time_days=t,
                           max_time_days=factor * t, scheme=run.scheme).grid  # fmt: skip
    names = parameter_names(run.initial)
    spreads = spreads or {}
    unc = {n: Uncertainty(spreads.get(n, 2.0 if n.endswith("_um") else 0.01)) for n in names}
    like = fs.ICLikelihood(profile, run.conditions, run.orientation, run.initial, unc, grid,
                           run.diffusivity, run.scheme)  # fmt: skip
    return Case(name, run, profile, like, grid)


SWEEP_STEP_L = 0.25  # bracket step for step positions, in diffusion lengths
SWEEP_FO_STEP = 0.0025  # bracket step for Fo values (0.25 mol%)
SWEEP_DROP = 4.0  # bracket each side until every scoring has dropped this far
SWEEP_MAX_EXPAND = 24
SWEEP_N = 25
SCORINGS = ("white", "fixedC", "refitC", "profiled")


def sweep(case: Case) -> pd.DataFrame:
    """One-parameter sweeps of every IC parameter (others at nominal), scored (a)-(d)."""
    basis = NoiseBasis(case.x)
    x, y = case.x, case.y
    M0, _ = case.at_points(case.nominal)
    chol_b = chol_of(nominal_noise(x, y, M0), x)
    ell0 = case.ell(int(np.argmin(((M0 - y) ** 2).sum(1))))

    def quick(M):
        return np.array([white_rows(M, y).max(), fixed_rows(M, y, chol_b).max(),
                         basis.best(y - M)[0]])  # fmt: skip

    rows = []
    for k, pname in enumerate(case.names):
        h = SWEEP_STEP_L * ell0 if pname.endswith("_um") else SWEEP_FO_STEP
        # bracket: walk out from the nominal until every scoring has dropped SWEEP_DROP
        s0 = quick(M0)
        ends = []
        for sign in (-1, 1):
            best, v = s0.copy(), case.nominal[k]
            last_ok = v
            for _ in range(SWEEP_MAX_EXPAND):
                v = v + sign * h
                th = case.nominal.copy()
                th[k] = v
                M, _ = case.at_points(th)
                if M is None:
                    break
                last_ok = v
                s = quick(M)
                best = np.maximum(best, s)
                if np.all(s < best - SWEEP_DROP):
                    break
            ends.append(last_ok)
        lo, hi = ends
        cand = {}

        def run(v, k=k, cand=cand):
            th = case.nominal.copy()
            th[k] = v
            M, _ = case.at_points(th)
            if M is not None:
                cand[float(v)] = M

        for v in np.linspace(lo, hi, SWEEP_N):
            run(v)
        # refine: dense points around each scoring's peak from the first pass
        vals = sorted(cand)
        first = np.array([quick(cand[v]) for v in vals])
        for j in range(3):
            wv = width_from_curve(vals, first[:, j])
            hw = max(wv["half_width"], (hi - lo) / (4 * SWEEP_N))
            for v in np.linspace(wv["best"] - 2.5 * hw, wv["best"] + 2.5 * hw, 9):
                if lo - 1e-12 <= v <= hi + 1e-12 and float(v) not in cand:
                    run(float(v))
        vals = sorted(cand)
        # (c): C refitted at the best (b) candidate (the sampler's second pass)
        bvals = [fixed_rows(cand[v], y, chol_b).max() for v in vals]
        chol_c = chol_of(nominal_noise(x, y, cand[vals[int(np.argmax(bvals))]]), x)
        for v in vals:
            M = cand[v]
            rec = {"param": pname, "value": v, "nominal": case.nominal[k]}
            for key, sc in (("white", white_rows(M, y)), ("fixedC", fixed_rows(M, y, chol_b)),
                            ("refitC", fixed_rows(M, y, chol_c))):  # fmt: skip
                rec[key], rec[f"days_{key}"] = float(sc.max()), case.days(int(np.argmax(sc)))
            ll, rd = basis.best(y - M)
            rec["profiled"], rec["days_profiled"] = ll, case.days(rd)
            rows.append(rec)
    return pd.DataFrame(rows)


def sweep_widths(t: pd.DataFrame) -> pd.DataFrame:
    """Per parameter: best value and 1-sigma half-width (log L drop 0.5) per scoring."""
    out = []
    for p, g in t.groupby("param", sort=False):
        rec = {"param": p, "nominal": g.nominal.iloc[0]}
        for k in SCORINGS:
            w = width_from_curve(g.value, g[k])
            rec |= {f"{k}_best": w["best_q"], f"{k}_half_width": w["half_width"],
                    f"{k}_open": w["open"]}  # fmt: skip
        out.append(rec)
    return pd.DataFrame(out)


def _band(curves, low, high):
    block = curves[max(int(low), 0) : min(int(high), len(curves) - 1) + 1]
    return block.min(axis=0), block.max(axis=0)


def default_fit(run, profiles, initial):
    """FORSTER's default fit (fit_run) with the IC replaced; curves and bands for drawing."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fr = forster.fit_run(replace(run, initial=initial), profiles, keep_curves=True)
    best, low, high = fr.indices
    lo1, hi1 = fr.time_range_days(sigmas=1)
    lo2, hi2 = fr.time_range_days(sigmas=2)
    x, y = points_in_domain(fr.model_x_um, fr.fit_x_um, fr.fit_fo)
    o = np.argsort(x)
    x, y = x[o], y[o]
    at = fr.curves[best] @ interpolation_matrix(fr.model_x_um, x).T
    species = [(run.diffusivity, fo_range(initial))]
    ell = diffusion_length_um(fr.time_days, species, run.conditions, run.orientation)
    b_lo, b_hi = _band(fr.curves, low, high)
    return {
        "time_d": fr.time_days, "lo1_d": lo1, "hi1_d": hi1, "lo2_d": lo2, "hi2_d": hi2,
        "near_step_daic": fr.near_step_delta_aic, "s": fr.noise.white_sigma,
        "a": fr.noise.correlated_sigma, "L": fr.noise.correlation_length_um, "ell_um": ell,
        "model_x": fr.model_x_um, "best_curve": fr.curves[best], "band_lo": b_lo,
        "band_hi": b_hi, "start_curve": fr.curves[0], "x": x, "y": y, "resid": y - at,
        "model_end_um": float(fr.model_x_um[-1]), "steps": tuple(initial.steps_um),
    }  # fmt: skip


def fit_ic(name, runs, profiles, seed=0):
    """FORSTER's joint IC fit (method auto) on the scan grid, as in the experiment."""
    spreads = SPREADS[name]
    case = make_case(name, runs, profiles, spreads=spreads)
    run = case.run
    unc = {n: Uncertainty(s) for n, s in spreads.items()}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        post = fs.fit_initial_condition(case.profile, run.conditions, run.orientation,
                                        run.initial, unc, case.grid, run.diffusivity,
                                        run.scheme, method="auto", seed=seed)  # fmt: skip
    like = case.like
    lo_dom, hi_dom = run.initial.domain_um
    params = []
    for i, n in enumerate(like.names):
        s = spreads.get(n)
        if s is None:
            continue
        p_lo, p_hi = like.nominal[i] - NORMAL_SPAN * s, like.nominal[i] + NORMAL_SPAN * s
        if n.endswith("_um"):  # a step must lie inside the model domain
            p_lo, p_hi = max(p_lo, lo_dom), min(p_hi, hi_dom)
        else:
            p_hi = min(p_hi, 1.0)
        best = post.best_values[n]
        z = (best - like.nominal[i]) / s
        params.append({"profile": name, "param": n, "nominal": like.nominal[i],
                       "prior_sd": s, "prior_lo": p_lo, "prior_hi": p_hi, "fitted": best,
                       "posterior_mean": post.mean[n], "posterior_sd": post.sd[n],
                       "shift_in_prior_sd": z,
                       "at_prior_edge": bool(abs(z) >= NORMAL_SPAN - 1e-3)})  # fmt: skip
    joint = {"profile": name, "method": post.method, "best_days": post.best_days,
             "median_days": post.median_days, "low_days": post.low_days,
             "high_days": post.high_days, "ic_share": post.ic_share, "runs": post.runs,
             "warnings": " | ".join(sorted({str(w.message)[:160] for w in caught}))}  # fmt: skip
    return params, joint, post.best_initial


# ------------------------------------------------------------- IC: synthetic recovery

IC_N, IC_DX, IC_ELL, IC_STEP, IC_WHITE, IC_AMP = 30, 2.5, 10.0, 25.0, 0.001, 0.004
IC_EDGE, IC_CORE = 0.86, 0.90
IC_K_TRUE = 400
IC_SEED = 20261016
IC_NOISES = {"white": None, "white + sqexp L=3": 3.0, "white + sqexp L=10": 10.0}
IC_SCENARIOS = {
    # name: (parameter index, true value, wrong nominal, sweep values)
    "step off by l/2": (0, IC_STEP, IC_STEP + 5.0, np.arange(17.0, 38.0001, 0.25)),
    "edge Fo off 0.5 mol%": (1, IC_EDGE, IC_EDGE + 0.005, np.arange(0.850, 0.8750001, 0.0005)),
    "control: nominal correct (step)": (0, IC_STEP, IC_STEP, np.arange(17.0, 38.0001, 0.25)),
}


class ICRecovery:
    """Synthetic geometry of the noise-models guide: 30 points 2.5 um apart, a 4 mol%
    normally zoned step at 25 um, diffusion length 10 um at the true time (row 400),
    packaged-example conditions, implicit scheme. Candidate model curves for each
    scenario's sweep are computed once (they do not depend on the noise)."""

    def __init__(self):
        run = template_run()
        initial = InitialCondition(domain_um=(0.0, 90.0), edge_fo=IC_EDGE, core_fo=IC_CORE,
                                   steps_um=(IC_STEP,))  # fmt: skip
        species = [(run.diffusivity, fo_range(initial))]
        per_day = diffusion_length_um(1.0, species, run.conditions, run.orientation) ** 2
        t_days = IC_ELL**2 / per_day
        dt = t_days * DAY / IC_K_TRUE
        grid = Grid(dx_um=IC_ELL / 8, dt_s=dt, max_time_days=3 * t_days + dt / DAY)
        x = IC_DX * (1 + np.arange(IC_N))
        prof = Profile(x, np.full_like(x, 0.88), name="synthetic")
        unc = {"step1_um": Uncertainty(2.0), "edge_fo": Uncertainty(0.01)}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.like = fs.ICLikelihood(prof, run.conditions, run.orientation, initial, unc,
                                        grid, run.diffusivity, "implicit")  # fmt: skip
        self.x = self.like.x
        self.truth = self.at(np.array([IC_STEP, IC_EDGE]))[IC_K_TRUE]
        self.basis = NoiseBasis(self.x)
        self._cands = {}

    def at(self, theta):
        return self.like.at_points(self.like.candidate(np.asarray(theta, float)))

    def cands(self, scen):
        if scen not in self._cands:
            k, _, wrong, values = IC_SCENARIOS[scen]
            out = []
            for v in values:
                th = np.array([IC_STEP, IC_EDGE])
                th[k] = v
                out.append(self.at(th))
            th = np.array([IC_STEP, IC_EDGE])
            th[k] = wrong
            self._cands[scen] = (out, self.at(th))
        return self._cands[scen]


def _sq_chol(x, L):
    d2 = (x[:, None] - x[None, :]) ** 2
    return np.linalg.cholesky(np.exp(-0.5 * d2 / L**2) + 1e-10 * np.eye(len(x)))


def ic_recovery_trial(g: ICRecovery, scen, noise, trial) -> dict:
    """One noise draw: sweep the scenario's parameter, score (a)-(d), record whether each
    scoring's 1-sigma (and 2-sigma) interval covers the true value."""
    _, true, _, values = IC_SCENARIOS[scen]
    cands, M_nom = g.cands(scen)
    x = g.x
    L = IC_NOISES[noise]
    rng = np.random.default_rng([IC_SEED, trial])
    e = IC_WHITE * rng.standard_normal(IC_N)
    ec = rng.standard_normal(IC_N)
    if L is not None:
        e = e + IC_AMP * _sq_chol(x, L) @ ec
    y = g.truth + e
    nb = nominal_noise(x, y, M_nom)
    chol_b = chol_of(nb, x)
    curves = {kk: np.empty(len(values)) for kk in SCORINGS}
    for i, M in enumerate(cands):
        curves["white"][i] = white_rows(M, y).max()
        curves["fixedC"][i] = fixed_rows(M, y, chol_b).max()
        curves["profiled"][i] = g.basis.best(y - M)[0]
    chol_c = chol_of(nominal_noise(x, y, cands[int(np.argmax(curves["fixedC"]))]), x)
    for i, M in enumerate(cands):
        curves["refitC"][i] = fixed_rows(M, y, chol_c).max()
    rec = {"scenario": scen, "noise": noise, "trial": trial, "true": true,
           "nominal_a": nb.correlated_sigma}  # fmt: skip
    for kk in SCORINGS:
        w1 = width_from_curve(values, curves[kk])
        w2 = width_from_curve(values, curves[kk], drop=2.0)
        rec |= {f"{kk}_best": w1["best_q"], f"{kk}_lo": w1["lo"], f"{kk}_hi": w1["hi"],
                f"{kk}_cov1": w1["lo"] <= true <= w1["hi"],
                f"{kk}_cov2": w2["lo"] <= true <= w2["hi"]}  # fmt: skip
    return rec


def ic_recovery_summary(t: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (scen, noise), g in t.groupby(["scenario", "noise"], sort=False):
        for kk in SCORINGS:
            err = g[f"{kk}_best"] - g["true"]
            hw = 0.5 * (g[f"{kk}_hi"] - g[f"{kk}_lo"])
            out.append({"scenario": scen, "noise": noise, "scoring": kk, "trials": len(g),
                        "coverage_1s": g[f"{kk}_cov1"].mean(),
                        "coverage_2s": g[f"{kk}_cov2"].mean(), "bias": err.mean(),
                        "rmse": np.sqrt((err**2).mean()),
                        "median_half_width": hw.median()})  # fmt: skip
    return pd.DataFrame(out)


def ic_recovery(trials) -> pd.DataFrame:
    g = ICRecovery()
    rows = [ic_recovery_trial(g, s, n, t) for s in IC_SCENARIOS for n in IC_NOISES
            for t in range(trials)]  # fmt: skip
    return pd.DataFrame(rows)


# --------------------------------------------------------------- plateau: synthetic

PL_DX, PL_ELL, PL_STEP, PL_WHITE = 2.5, 10.0, 25.0, 0.001
PL_K_TRUE = 400
PL_DOMAIN_END = 300.0
PL_N_ALL = 90
PL_LENGTHS = {3: 22, 5: 30, 10: 50, 20: 90}  # diffusion lengths past the step -> points
PL_CORES = np.round(np.arange(0.8980, 0.90301, 0.0005), 4)
PL_SLOPE = 0.001 / 100.0  # +0.1 mol% Fo per 100 um past the step
PL_SEED = 20261016
PL_X = PL_DX * (1 + np.arange(PL_N_ALL))
PL_CASES = {
    # name -> (synthetic.CASES noise draw, true core Fo, plateau slope)
    "white": ("white (control)", 0.90, 0.0),
    "smooth L=3": ("sq-exp L=3 (control)", 0.90, 0.0),
    "smooth L=10": ("sq-exp L=10 (control)", 0.90, 0.0),
    "near-step 0.4": ("near-step sq-exp L=3", 0.90, 0.0),
    "near-step 1.2": ("strong near-step L=3", 0.90, 0.0),
    "core +0.1": ("white (control)", 0.901, 0.0),
    "core +0.2": ("white (control)", 0.902, 0.0),
    "plateau slope": ("white (control)", 0.90, PL_SLOPE),
}


def _pl_curves(run, core, x_end, dt, t_days):
    initial = InitialCondition(domain_um=(0.0, PL_DOMAIN_END), edge_fo=0.86, core_fo=core,
                               steps_um=(PL_STEP,))  # fmt: skip
    grid = Grid(dx_um=PL_ELL / 8, dt_s=dt, max_time_days=3 * t_days + dt / DAY)
    xp = PL_DX * (1 + np.arange(round(x_end / PL_DX)))
    prof = Profile(xp, np.full_like(xp, 0.88), name="synthetic")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return forster.fit_profile(prof, run.conditions, run.orientation, initial, grid,
                                   diffusivity=run.diffusivity, scheme="implicit",
                                   stop_early=False, keep_curves=True,
                                   check_near_step=False)  # fmt: skip


def plateau_geometry() -> dict:
    """Points 2.5 um apart from 2.5 um, rim Fo 0.86, a 4 mol% step at 25 um, l = 10 um
    at the true time (row 400), model to 3x. "fixed": the model runs to 300 um for every
    plateau length (one per core Fo on PL_CORES); "last_<m>": the model ends at the last
    measured point, core Fo held there (what FORSTER does with a shorter profile)."""
    run = template_run()
    initial = InitialCondition(domain_um=(0.0, PL_DOMAIN_END), edge_fo=0.86, core_fo=0.90,
                               steps_um=(PL_STEP,))  # fmt: skip
    species = [(run.diffusivity, fo_range(initial))]
    per_day = diffusion_length_um(1.0, species, run.conditions, run.orientation) ** 2
    t_days = PL_ELL**2 / per_day
    dt = t_days * DAY / PL_K_TRUE
    out = {"t_days": t_days, "dt_s": dt}
    fixed = []
    for core in PL_CORES:
        res = _pl_curves(run, float(core), PL_DOMAIN_END, dt, t_days)
        fixed.append(res.curves @ interpolation_matrix(res.model_x_um, PL_X).T)
    out["fixed"] = np.stack(fixed)  # cores x rows x 90
    for m, n in PL_LENGTHS.items():
        x = PL_X[:n]
        res = _pl_curves(run, 0.90, float(x[-1]), dt, t_days)
        out[f"last_{m}"] = res.curves @ interpolation_matrix(res.model_x_um, x).T
    return out


def _span(chi2, n, p=1):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        i1 = delta_chi2_interval(chi2, interval_threshold(n, p), end_is_limit=False)
        i2 = delta_chi2_interval(chi2, interval_threshold(n, p, TWO_SIGMA), end_is_limit=False)
    return i1, i2


def _rec(i1, i2, **extra):
    return {"best": i1.best, "lo1": i1.low, "hi1": i1.high, "lo2": i2.low, "hi2": i2.high} | extra


def _fit_three(at, x, y):
    """White-only, default (whitened) and near-step fits of y; model at points at."""
    n = len(y)
    rss = white_rss(at, y)
    w_row = int(np.argmin(rss))
    out = {"white": _rec(*_span(rss / (rss[w_row] / n), n))}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = fit_correlated_points(at, x, y, end_is_limit=False)
    out["default"] = _rec(d.interval, d.interval2)
    ell = PL_ELL * np.sqrt(max(w_row, 1) / PL_K_TRUE)
    cov, _, daic = fit_zone_noise(x, y - at[w_row], bump(x, [PL_STEP], ell), "corr")
    out["near-step"] = _rec(*_ranges(at, y, cov), daic=daic)
    return out


def _fit_free_core(at_c, x, y):
    """Default noise with core Fo a second fitted parameter (grid PL_CORES)."""
    n = len(y)
    r = at_c - y  # cores x rows x n
    rss = np.einsum("ckp,ckp->ck", r, r)
    c0, k0 = np.unravel_index(int(np.argmin(rss)), rss.shape)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nm = fit_noise_model(x, -r[c0, k0])
    chol = np.linalg.cholesky(nm.covariance(x))
    z = np.linalg.solve(chol, r.reshape(-1, n).T)
    chi2 = np.einsum("pk,pk->k", z, z).reshape(rss.shape)
    prof = chi2.min(axis=0)
    core = float(PL_CORES[np.argmin(chi2[:, int(np.argmin(prof))])])
    return {"default free core": _rec(*_span(prof, n, 2), core=core)}


def plateau_trial(g, case, trial) -> list[dict]:
    """One noise draw on the 90 points, cut to each plateau length (the step region and
    its noise are identical across lengths)."""
    draw, core, slope = PL_CASES[case]
    rng = np.random.default_rng([PL_SEED, trial])
    gnoise = {"x": PL_X, "white": PL_WHITE, "diffusion_length_um": PL_ELL,
              "steps": np.array([PL_STEP])}  # fmt: skip
    e = make_noise(gnoise, draw, rng)
    ci = int(np.argmin(np.abs(PL_CORES - core)))
    y_all = g["fixed"][ci, PL_K_TRUE] + slope * np.clip(PL_X - PL_STEP, 0, None) + e
    i90 = int(np.argmin(np.abs(PL_CORES - 0.90)))
    rows = []
    for m, n in PL_LENGTHS.items():
        x, y = PL_X[:n], y_all[:n]
        variants = {"fixed": _fit_three(g["fixed"][i90, :, :n], x, y),
                    "to last point": _fit_three(g[f"last_{m}"], x, y)}  # fmt: skip
        variants["fixed"] |= _fit_free_core(g["fixed"][:, :, :n], x, y)
        for dom, fits in variants.items():
            for model, rec in fits.items():
                rows.append({"case": case, "trial": trial, "plateau_l": m, "n": n,
                             "domain": dom, "model": model, "k_true": PL_K_TRUE} | rec)  # fmt: skip
    return rows


def _chunk(cases, start, stop):
    g = plateau_geometry()
    return [r for c in cases for t in range(start, stop) for r in plateau_trial(g, c, t)]


def plateau_synthetic(trials, cases=tuple(PL_CASES), workers=2) -> pd.DataFrame:
    from concurrent.futures import ProcessPoolExecutor

    edges = np.linspace(0, trials, workers + 1).astype(int)
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(_chunk, cases, int(a), int(b)) for a, b in itertools.pairwise(edges)]
        rows = [r for f in futs for r in f.result()]
    return pd.DataFrame(rows)


def plateau_summary(t: pd.DataFrame) -> pd.DataFrame:
    """Per case x plateau length x domain x model: median error of the best fit, 1- and
    2-sigma coverage of the true time, median 1-sigma width (percent)."""
    k = t.k_true
    t = t.assign(
        ratio=t.best / k,
        c1=((t.lo1 <= k) & (k <= t.hi1)).astype(float),
        c2=((t.lo2 <= k) & (k <= t.hi2)).astype(float),
        w1=(t.hi1 - t.lo1) / k,
    )
    g = t.groupby(["case", "plateau_l", "n", "domain", "model"])
    out = pd.DataFrame({
        "trials": g.size(),
        "bias_pct": 100 * (g.ratio.median() - 1),
        "cov1": 100 * g.c1.mean(),
        "cov2": 100 * g.c2.mean(),
        "width1_pct": 100 * g.w1.median(),
    })  # fmt: skip
    return out.reset_index()


# ------------------------------------------------------------------- plateau: real

PL_CUTS = [None, 15, 10, 7, 5, 4, 3, 2]  # diffusion lengths kept past the last step


def _real_rec(i1, i2, dt, **extra):
    d = dt / DAY
    return {"time_d": i1.best * d, "lo1_d": i1.low * d, "hi1_d": i1.high * d,
            "lo2_d": i2.low * d, "hi2_d": i2.high * d} | extra  # fmt: skip


def plateau_cuts(run, profiles) -> list[dict]:
    """Drop core-side points past the last step at step + m l (l at the full profile's
    default fit) and refit two ways: "fixed domain" (the full profile's model, only the
    measured points change; white, default and near-step noise) and "as used"
    (fit_run on the cut profile: the model ends at the last measured point, core Fo
    held there)."""
    from forster.model import fit_profile

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = forster.fit_run(run, profiles)
    species = [(run.diffusivity, fo_range(run.initial))]

    def length_days(days):
        return diffusion_length_um(days, species, run.conditions, run.orientation)

    ell = length_days(full.time_days)
    hi2 = full.time_days + full.time_plus2_s / DAY
    long_grid = replace(run.grid, max_time_days=max(3 * full.time_days, 1.5 * hi2))
    prof = profiles[run.profile_name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = fit_profile(prof, run.conditions, run.orientation, run.initial, long_grid,
                          run.kriging, run.diffusivity, run.scheme, run.uncertainty,
                          stop_early=False, keep_curves=True, check_near_step=False)  # fmt: skip
    dt = res.dt_s
    x_all, y_all = points_in_domain(res.model_x_um, res.fit_x_um, res.fit_fo)
    o = np.argsort(x_all)
    x_all, y_all = x_all[o], y_all[o]
    at_all = res.curves @ interpolation_matrix(res.model_x_um, x_all).T
    del res
    step = max(run.initial.steps_um)
    rows, seen = [], set()
    for m in PL_CUTS:
        cut = np.inf if m is None else step + m * ell
        keep = x_all <= cut
        key = int(keep.sum())
        if key in seen or (x_all[keep] > step).sum() < 2:
            continue
        seen.add(key)
        x, y, at = x_all[keep], y_all[keep], at_all[:, keep]
        base = {"profile": run.name, "cut_l": "all" if m is None else m, "n": key,
                "kept_past_step_l": (x.max() - step) / ell, "ell_full_um": ell,
                "last_point_um": float(x.max())}  # fmt: skip
        n = len(y)
        rss = white_rss(at, y)
        w = int(np.argmin(rss))
        cov, _, daic = fit_zone_noise(x, y - at[w], bump(x, run.initial.steps_um,
                                      length_days(max(w, 1) * dt / DAY)), "corr")  # fmt: skip
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = fit_correlated_points(at, x, y, end_is_limit=False)
        fits = {"white": _span(rss / (rss[w] / n), n), "default": (d.interval, d.interval2),
                "near-step": _ranges(at, y, cov)}  # fmt: skip
        for model, (i1, i2) in fits.items():
            rows.append(base | {"domain": "fixed domain", "model": model,
                                "near_step_daic": daic} | _real_rec(i1, i2, dt))  # fmt: skip
        pk = prof.distance_um <= cut
        cut_prof = {run.profile_name: Profile(prof.distance_um[pk], prof.fo[pk], prof.name)}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fr = forster.fit_run(run, cut_prof)
        lo1, hi1 = fr.time_range_days(sigmas=1)
        lo2, hi2c = fr.time_range_days(sigmas=2)
        rows.append(base | {"domain": "as used", "model": "default", "time_d": fr.time_days,
                            "lo1_d": lo1, "hi1_d": hi1, "lo2_d": lo2, "hi2_d": hi2c,
                            "near_step_daic": fr.near_step_delta_aic,
                            "model_end_um": float(fr.model_x_um[-1])})  # fmt: skip
    return rows


def near_step_daic_at(x, y, at_row, steps, ell):
    """forster.uncertainty.near_step_check's dAIC for residuals y - at_row."""
    return near_step_check(x, y - at_row, steps, ell).delta_aic
