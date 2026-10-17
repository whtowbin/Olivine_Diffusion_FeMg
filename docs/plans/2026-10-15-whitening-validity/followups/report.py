"""Print the README's tables from the CSVs (public data only).

uv run python docs/plans/2026-10-15-whitening-validity/followups/report.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(out)


def fmt_h(v):
    return "inf" if not np.isfinite(v) else (f"{v:.0f}" if v >= 10 else f"{v:.1f}")


def real():
    t = pd.read_csv(HERE / "real_continuous_h.csv").sort_values("cont_daic")
    t = t[(t.cont_daic < 0) | (t.grid_h == 100)]
    out = pd.DataFrame(
        {
            "profile": t.run,
            "points (near step)": [f"{a} ({b})" for a, b in zip(t.n_points, t.n_near, strict=True)],
            "dAIC grid": t.grid_daic.round(2),
            "dAIC continuous": t.cont_daic.round(2),
            "h grid": t.grid_h.astype(int),
            "h continuous [95% interval]": [
                f"{fmt_h(h)} [{fmt_h(lo)}, {fmt_h(hi)}]"
                for h, lo, hi in zip(t.cont_h, t.h_lo95, t.h_hi95, strict=True)
            ],
            "white sd": t.white_sd_molpct.round(3),
            "misfit sd far": t.misfit_far_molpct.round(3),
            "misfit sd at step": t.misfit_peak_molpct.round(2),
            "sqrt(1+h)": t.sqrt_1_plus_h.round(1),
            "near/far rms implied": t.implied_near_far_rms.round(2),
            "near/far rms measured": t.measured_near_far_rms.round(2),
            "time shift grid %": (100 * (t.time_grid_d / t.time_default_d - 1)).round(1),
            "time shift continuous %": (100 * (t.time_cont_d / t.time_default_d - 1)).round(1),
        }
    )
    print(md(out))


def null():
    s = pd.read_csv(HERE / "null_summary.csv")
    out = pd.DataFrame(
        {
            "geometry": s.geometry.str.replace(r" \(.*\)", "", regex=True),
            "case": s.case,
            "draws": s.draws,
            "flagged grid %": s.grid_lt_m6_pct.round(2),
            "flagged continuous %": s.cont_lt_m6_pct.round(2),
            "1% point of dAIC, grid": s.grid_q99.round(1),
            "1% point, continuous": s.cont_q99.round(1),
            "largest drop in dAIC": s.max_shift.round(1),
        }
    )
    print(md(out))


def power():
    s = pd.read_csv(HERE / "power_summary.csv")
    order = [
        "white",
        "smooth 0.4 mol%, L=10, everywhere",
        "0.4 mol%, L=3",
        "0.8 mol%, L=3",
        "1.2 mol%, L=3",
        "1.2 mol%, L=10",
    ]
    rows = []
    for name in order:
        g = s[s.noise == name].sort_values("n")
        for r in g.itertuples():
            rows.append(
                {
                    "misfit": name,
                    "points": r.n,
                    "within 2 l of step": r.n_near,
                    "warns, grid %": round(r.grid_warn_pct, 1),
                    "warns, continuous %": round(r.cont_warn_pct, 1),
                    "1s cover default %": round(r.cov1_default),
                    "2s cover default %": round(r.cov2_default),
                    "1s cover near-step model %": round(r.cov1_cont),
                    "1s width default %": round(r.width_default_pct),
                    "1s width near-step %": round(r.width_cont_pct),
                }
            )
    print(md(pd.DataFrame(rows)))


if __name__ == "__main__":
    print("## real\n")
    real()
    print("\n## null\n")
    null()
    print("\n## power\n")
    power()
