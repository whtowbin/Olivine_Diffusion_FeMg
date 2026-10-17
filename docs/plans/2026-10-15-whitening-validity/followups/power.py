"""Follow-up 2: detection power of the near-step warning against the number of points.

    uv run python docs/plans/2026-10-15-whitening-validity/followups/power.py run [draws]
    uv run python docs/plans/2026-10-15-whitening-validity/followups/power.py summarise

Geometry (the noise-models guide's, with the number of points varied): a 4 mol% normally
zoned step (Fo 86 -> 90) at 25 um, the packaged example's conditions, true diffusion
length l = 10 um (row 400 of the run, modelled to 3x), implicit scheme. The data run from
the rim to 75 um, i.e. 5 diffusion lengths past the step (2.5 l of rim side, about 2.7 l of
plateau beyond where the plateau starts), and that span is fixed: N points sit at
75/N, 2 x 75/N, ..., 75 um, so only the density changes (N = 30 is the guide's geometry,
2.5 um apart). The model runs to 300 um so its core end does not move the fit.

Noise: 0.1 mol% white noise everywhere plus a squared-exponential misfit (L = 3 um, and
L = 10 um at the strongest amplitude) multiplied by a Gaussian bump of width l at the step
(peak amplitude 0.4, 0.8 or 1.2 mol%), as synthetic.py's near-step cases. Controls (the
noise model is right): white noise alone, and a 0.4 mol% misfit with L = 10 um of the same
size everywhere. Same random draws for every case at a given (N, draw), so the cases are
paired.

Per draw: the default fit (`fit_correlated_points`: best fit, 1- and 2-sigma ranges), the
near-step check at the white best fit on FORSTER's grid of h and with h continuous
(nearstep.py), and the 1-sigma range under the continuous near-step covariance.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from nearstep import NearStep, chi2_rows  # noqa: E402

from forster.uncertainty import (  # noqa: E402
    TWO_SIGMA,
    delta_chi2_interval,
    fit_correlated_points,
    interpolation_matrix,
    interval_threshold,
    white_rss,
)

CACHE = Path(os.environ.get("FOLLOWUP_CACHE", "/tmp")) / "near_step_followups"
TRIALS_CSV = HERE / "power_trials.csv"
SUMMARY_CSV = HERE / "power_summary.csv"
SEED = 20261017
WORKERS = 2
CHUNK = 50
N_POINTS = (12, 17, 25, 35, 50, 100)
SPAN_UM, STEP_UM, ELL_UM, WHITE = 75.0, 25.0, 10.0, 0.001
EDGE_FO, CORE_FO, DOMAIN_END = 0.86, 0.90, 300.0
K_TRUE, FACTOR = 400, 3
NOISES = {
    # name -> (peak misfit amplitude, misfit length um, concentrated at the step)
    "white": (0.0, 3.0, True),
    "smooth 0.4 mol%, L=10, everywhere": (0.004, 10.0, False),
    "0.4 mol%, L=3": (0.004, 3.0, True),
    "0.8 mol%, L=3": (0.008, 3.0, True),
    "1.2 mol%, L=3": (0.012, 3.0, True),
    "1.2 mol%, L=10": (0.012, 10.0, True),
}


def _curves():
    """Model rows 0..3 x 400 on a 1.25 um grid to 300 um (cached)."""
    path = CACHE / "power_curves.npz"
    if path.exists():
        z = np.load(path)
        return z["model_x"], z["curves"], float(z["dt_s"])
    import forster
    from forster.grid import diffusion_length_um, fo_range
    from forster.model import SECONDS_PER_DAY, Grid, InitialCondition, Profile

    with tempfile.TemporaryDirectory() as tmp:
        forster.write_template(tmp)
        run = forster.read_runs(Path(tmp) / "parameters.csv")[0]
    initial = InitialCondition(
        domain_um=(0.0, DOMAIN_END), edge_fo=EDGE_FO, core_fo=CORE_FO, steps_um=(STEP_UM,)
    )
    species = [(run.diffusivity, fo_range(initial))]
    per_day = diffusion_length_um(1.0, species, run.conditions, run.orientation) ** 2
    t_days = ELL_UM**2 / per_day
    dt = t_days * SECONDS_PER_DAY / K_TRUE
    grid = Grid(dx_um=ELL_UM / 8, dt_s=dt, max_time_days=FACTOR * t_days + dt / SECONDS_PER_DAY)
    xp = 2.5 * (1 + np.arange(round(DOMAIN_END / 2.5)))
    prof = Profile(xp, np.full_like(xp, 0.88), name="synthetic")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = forster.fit_profile(
            prof, run.conditions, run.orientation, initial, grid,
            diffusivity=run.diffusivity, scheme="implicit", stop_early=False,
            keep_curves=True, check_near_step=False,
        )  # fmt: skip
    curves = res.curves[: FACTOR * K_TRUE + 1]
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, model_x=res.model_x_um, curves=curves, dt_s=dt)
    return res.model_x_um, curves, dt


_GEO: dict[int, dict] = {}


def geometry(n: int) -> dict:
    """Model at the n points for every row, in the layout synthetic.py's helpers use."""
    if n not in _GEO:
        model_x, curves, dt = _curves()
        x = (SPAN_UM / n) * (1 + np.arange(n))
        _GEO[n] = {
            "at_points": curves @ interpolation_matrix(model_x, x).T,
            "x": x,
            "k_true": K_TRUE,
            "dt_s": dt,
            "white": WHITE,
            "steps": np.array([STEP_UM]),
            "diffusion_length_um": ELL_UM,
        }
    return _GEO[n]


def noise(g: dict, name: str, rng) -> np.ndarray:
    x = g["x"]
    n = len(x)
    e_white = rng.standard_normal(n)
    e_corr = rng.standard_normal(n)
    amp, L, at_step = NOISES[name]
    e = WHITE * e_white
    if amp > 0:
        K = np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2 / L**2) + 1e-10 * np.eye(n)
        env = np.exp(-0.5 * ((x - STEP_UM) / ELL_UM) ** 2) if at_step else 1.0
        e = e + amp * env * (np.linalg.cholesky(K) @ e_corr)
    return e


def ranges(chi2, n):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        i1 = delta_chi2_interval(chi2, interval_threshold(n), end_is_limit=False)
        i2 = delta_chi2_interval(chi2, interval_threshold(n, level=TWO_SIGMA), end_is_limit=False)
    return i1, i2


def one_trial(n: int, name: str, trial: int) -> dict:
    g = geometry(n)
    at, x, k = g["at_points"], g["x"], K_TRUE
    rng = np.random.default_rng([SEED, n, trial])
    y = at[k] + noise(g, name, rng)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = fit_correlated_points(at, x, y, end_is_limit=False)
    w_row = int(np.argmin(white_rss(at, y)))
    ell = ELL_UM * np.sqrt(max(w_row, 1) / k)
    out = NearStep(x, y - at[w_row], [STEP_UM], ell).fit()
    c1, c2 = ranges(chi2_rows(at, y, out.cov_cont), n)
    return {
        "n": n,
        "noise": name,
        "trial": trial,
        "n_near": out.n_near,
        "grid_daic": out.daic_grid,
        "grid_h": out.h_grid,
        "cont_daic": out.daic_cont,
        "cont_h": out.h_cont,
        "best": d.interval.best,
        "covered": d.interval.low <= k <= d.interval.high,
        "covered2": d.interval2.low <= k <= d.interval2.high,
        "width": (d.interval.high - d.interval.low) / k,
        "cont_best": c1.best,
        "cont_covered": c1.low <= k <= c1.high,
        "cont_covered2": c2.low <= k <= c2.high,
        "cont_width": (c1.high - c1.low) / k,
    }


def run_chunk(n, name, start, stop):
    return [one_trial(n, name, t) for t in range(start, stop)]


def run(draws):
    _curves()
    done = set()
    if TRIALS_CSV.exists():
        old = pd.read_csv(TRIALS_CSV, usecols=["n", "noise", "trial"])
        done = set(zip(old.n, old.noise, old.trial, strict=True))
    tasks = [
        (n, name, s, min(s + CHUNK, draws))
        for n in sorted(N_POINTS, reverse=True)
        for name in NOISES
        for s in range(0, draws, CHUNK)
        if not all((n, name, t) in done for t in range(s, min(s + CHUNK, draws)))
    ]
    print(f"{len(tasks)} chunks", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(WORKERS) as pool:
        futures = [pool.submit(run_chunk, *task) for task in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            pd.DataFrame(fut.result()).to_csv(
                TRIALS_CSV, mode="a", header=not TRIALS_CSV.exists(), index=False,
                float_format="%.6g",
            )  # fmt: skip
            if i % 10 == 0 or i == len(tasks):
                print(f"{time.perf_counter() - t0:7.1f}s [{i}/{len(tasks)}]", flush=True)
    summarise()


def _band(p, n):
    se = np.sqrt(p * (1 - p) / n)
    return 100 * (p - 1.96 * se), 100 * (p + 1.96 * se)


def summarise():
    t = pd.read_csv(TRIALS_CSV).drop_duplicates(["n", "noise", "trial"])
    rows = []
    for (n, name), g in t.groupby(["n", "noise"]):
        rows.append(
            {
                "n": n,
                "noise": name,
                "draws": len(g),
                "n_near": int(g.n_near.median()),
                "grid_warn_pct": 100 * (g.grid_daic < -6).mean(),
                "cont_warn_pct": 100 * (g.cont_daic < -6).mean(),
                "cont_warn_m8_pct": 100 * (g.cont_daic < -8).mean(),
                "grid_daic_median": g.grid_daic.median(),
                "cont_daic_median": g.cont_daic.median(),
                "grid_h100_pct": 100 * (g.grid_h == 100).mean(),
                "cov1_default": 100 * g.covered.mean(),
                "cov2_default": 100 * g.covered2.mean(),
                "cov1_cont": 100 * g.cont_covered.mean(),
                "cov2_cont": 100 * g.cont_covered2.mean(),
                "width_default_pct": 100 * g.width.median(),
                "width_cont_pct": 100 * g.cont_width.median(),
                "cov1_default_if_silent": 100 * g.covered[g.grid_daic >= -6].mean(),
                "cov1_default_if_warned": 100 * g.covered[g.grid_daic < -6].mean()
                if (g.grid_daic < -6).any()
                else np.nan,
            }
        )
    s = pd.DataFrame(rows)
    s["noise"] = pd.Categorical(s.noise, list(NOISES))
    s = s.sort_values(["noise", "n"])
    s.to_csv(SUMMARY_CSV, index=False, float_format="%.4g")
    pd.set_option("display.width", 250)
    print(s.round(1).to_string(index=False))


def measure():
    for n in N_POINTS:
        geometry(n)
        t = time.perf_counter()
        for i in range(3):
            one_trial(n, "1.2 mol%, L=3", i)
        print(f"N={n:3d} {(time.perf_counter() - t) / 3:.3f} s/trial", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "measure"
    if cmd == "run":
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 400)
    elif cmd == "summarise":
        summarise()
    else:
        measure()
