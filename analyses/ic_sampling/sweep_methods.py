"""Compare ways to combine initial-condition uncertainty with the time fit.

Every method uses the same likelihood: one noise model (fitted once at a 9-point grid's
best fit), chi2 at every time step from one forward run per initial condition, flat
prior on time, posterior ~ chi2^(-n/2) * prior. So differences are sampling error only.
Output: 16/50/84th percentiles of the marginal posterior of time (days), forward runs
used, wall seconds.
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, "tests")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import ROUND_ROBIN_DATA, ROUND_ROBIN_PARAMS
from likelihood import Problem
from methods import (
    method_dynesty,
    method_emcee,
    method_grid,
    method_laplace_is,
    timed,
)

from forster import Uncertainty, read_profiles, read_runs

warnings.filterwarnings("ignore")


def main():
    runs = {r.name: r for r in read_runs(ROUND_ROBIN_PARAMS)}
    profiles = read_profiles(ROUND_ROBIN_DATA)
    which = sys.argv[1:] or ["all"]
    cases = []
    for name in ("KS20-527_2-transect-1", "KS20-527_5-transect-2"):
        for label, unc in (
            (
                "gauss",
                {
                    "step1_um": Uncertainty(3.0),
                    "edge_fo": Uncertainty(0.003),
                    "core_fo": Uncertainty(0.003),
                },
            ),
            (
                "wide",
                {
                    "step1_um": Uncertainty(10.0, "uniform"),
                    "edge_fo": Uncertainty(0.01, "uniform"),
                    "core_fo": Uncertainty(0.01, "uniform"),
                },
            ),
        ):
            cases.append((name, label, unc, 3))
    for label, unc in (
        (
            "gauss",
            {
                "step1_um": Uncertainty(3.0),
                "step2_um": Uncertainty(3.0),
                "edge_fo": Uncertainty(0.003),
                "core_fo": Uncertainty(0.003),
                "step_fo": Uncertainty(0.003),
            },
        ),
        (
            "wide",
            {
                "step1_um": Uncertainty(10.0, "uniform"),
                "step2_um": Uncertainty(10.0, "uniform"),
                "edge_fo": Uncertainty(0.01, "uniform"),
                "core_fo": Uncertainty(0.01, "uniform"),
                "step_fo": Uncertainty(0.01, "uniform"),
            },
        ),
    ):
        cases.append(("KS20-527-8_transect-2", label, unc, 5))

    for name, label, unc, d in cases:
        if which != ["all"] and f"{name}:{label}" not in which and name not in which:
            continue
        P = Problem(runs[name], profiles, unc)
        print(
            f"\n== {name} [{label}, {d} params]  grid {P.grid.timesteps} steps, "
            f"dt {P.grid.dt_s:.0f}s",
            flush=True,
        )

        def show(tag, out, n, sec, extra=""):
            q, tail = out[0], out[1]
            print(
                f"  {tag:24s} p16/50/84 = {q[0]:7.2f} {q[1]:7.2f} {q[2]:7.2f} d"
                f"  runs {n:7d}  {sec:6.1f}s  tail {tail:.1e} {extra}",
                flush=True,
            )

        ref_pts = 31 if d == 3 else 11
        grid_pts = (5, 9, 13, 17) if d == 3 else (3, 5, 7)
        out, n, sec = timed(P, method_grid, ref_pts)
        show(f"REF grid {ref_pts}^{d}", out, n, sec)
        for pts in grid_pts:
            out, n, sec = timed(P, method_grid, pts)
            show(f"grid {pts}^{d}", out, n, sec)
        for ns in (256, 1024, 4096):
            (q, tail, ess), n, sec = timed(P, method_laplace_is, ns)
            show(f"laplace+IS N={ns}", (q, tail), n, sec, f"ESS {ess:.0f}")
        for steps in (250, 1000):
            out, n, sec = timed(P, method_emcee, steps)
            show(f"emcee 32x{steps}", out, n, sec)
        for nlive in (100, 400):
            (out, logz), n, sec = timed(P, method_dynesty, nlive)
            show(f"dynesty nlive={nlive}", out, n, sec, f"logZ {logz:.2f}")


if __name__ == "__main__":
    main()
