"""Two diagnostics of the importance samples other than the ESS, for `rules.py`.

For every case `calibrate.py` cached and every Laplace run it made (1024, 2048, 4096
samples; seeds 0-9), this redraws the same importance samples (same likelihood, best
fit and seed, so the same draws) and records:

- ``pareto_k``: the Pareto shape k of the largest importance weights (Vehtari et al.
  2024, "Pareto smoothed importance sampling", JMLR 25(72); Zhang and Stephens 2009
  estimator, the largest min(n/5, 3 sqrt(n)) weights; equal to arviz 0.19 psislw to
  1e-14 on 10 test sets). Above 0.7 the weights' tail is
  too heavy for the estimates to be trusted, whatever the ESS says.
- ``boot_se``: the bootstrap standard error of p16, p50 and p84 of time, relative to
  each value (largest of the three; 100 resamples of the weighted samples). Scrambled
  Sobol draws are not independent, so this overstates the error a little.

    uv run python -u docs/plans/2026-10-16-joint-calibration/diagnostics.py --set paper
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import special

import forster
from forster.sampling import _importance, _summarise

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import calibrate as C  # noqa: E402

SEEDS = tuple(range(10))
BOOTSTRAP = 100


def pareto_k(log_w: np.ndarray) -> float:
    """Zhang-Stephens estimate of the generalised Pareto shape of the largest weights."""
    n = len(log_w)
    m = math.ceil(min(n / 5, 3 * math.sqrt(n)))  # as arviz psislw
    w = np.exp(np.sort(log_w) - log_w.max())
    tail, threshold = w[-m:], w[-m - 1]
    x = tail - threshold
    if np.all(x <= 0):
        return -math.inf
    x = np.sort(x)
    prior = 3
    k_grid = 30 + int(math.sqrt(m))
    q = x[int(m / 4 + 0.5) - 1]
    b = 1 / x[-1] + (1 - np.sqrt(k_grid / (np.arange(1, k_grid + 1) - 0.5))) / prior / q
    k = np.mean(np.log1p(-b[:, None] * x[None, :]), axis=1)
    log_l = m * (np.log(-b / k) - k - 1)
    weights = np.exp(log_l - special.logsumexp(log_l))
    b_hat = float(np.sum(b * weights))
    k_hat = float(np.mean(np.log1p(-b_hat * x)))
    return (m * k_hat + 10 * 0.5) / (m + 10)  # weakly regularised toward 0.5, as PSIS


def bootstrap_se(like, samples, rng) -> float:
    """Largest relative bootstrap SE of p16, p50, p84 of time."""
    base = _summarise(like, samples)
    ref = np.array([base["low_days"], base["median_days"], base["high_days"]])
    out = []
    n = len(samples.theta)
    for _ in range(BOOTSTRAP):
        pick = rng.integers(0, n, n)
        s = type(samples)(samples.theta[pick], samples.log_w[pick], samples.rows[pick])
        r = _summarise(like, s)
        out.append([r["low_days"], r["median_days"], r["high_days"]])
    return float(np.max(np.std(out, axis=0) / ref))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", choices=("paper", "stress"), default="paper")
    args = parser.parse_args()
    cache = json.loads((HERE / f"cache_{args.set}.json").read_text())
    out_path = HERE / f"diagnostics_{args.set}.json"
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    crystals = C.PAPER_CRYSTALS if args.set == "paper" else C.STRESS_CRYSTALS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runs = {r.name: r for r in forster.read_runs(C.PARAMS)}
        profiles = forster.read_profiles(C.DATA, sheet="Olivine")
    for case, c in cache.items():
        if case in out:
            continue
        print(case, flush=True)
        step_sd, fo_sd, kwargs = C.SETTINGS[c["setting"]]
        rs = C.crystal_runs(crystals[c["crystal"]], runs, step_sd, fo_sd, c["crystal"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            individual = {r.name: forster.fit_run(r, profiles) for r in rs}
            like, z = C.rebuild(rs, profiles, individual, kwargs)
        rows = []
        for n in C.SAMPLE_COUNTS:
            for seed in SEEDS:
                samples, ess = _importance(like, z, n, seed, whiten=True)
                marginal = samples.log_w + special.logsumexp(samples.rows, axis=1)
                rows.append(
                    {
                        "n": n,
                        "seed": seed,
                        "ess": ess,
                        "pareto_k": pareto_k(marginal),
                        "boot_se": bootstrap_se(like, samples, np.random.default_rng(seed)),
                    }
                )
            print(
                f"  n={n}: k {[round(r['pareto_k'], 2) for r in rows[-len(SEEDS) :]]}", flush=True
            )
        out[case] = rows
        out_path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
