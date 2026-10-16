"""Print the Markdown tables and headline counts used in README.md, from the CSVs.

uv run python docs/plans/2026-10-15-whitening-validity/report.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CORE = {  # the whiteness tests used for "passes" (var z is ~1 by construction: see README)
    "boot_p_lb_p_lag5": "Ljung-Box 1-5",
    "boot_p_runs_p": "runs",
    "boot_p_shapiro_p": "Shapiro-Wilk",
    "boot_p_max_abs_z": "largest |z|",
}


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def part_a():
    k = pd.read_csv(HERE / "real_kernels.csv")
    r = pd.read_csv(HERE / "real_profiles.csv")
    for where in ("white best fit", "correlated best fit"):
        p = k[k.residuals_at == where].pivot(index="run", columns="kernel", values="aic")
        best = p.min(axis=1)
        fam = np.minimum(p.white, p.sqexp) - best
        kb = p[["sqexp", "exp", "matern32"]].min(axis=1)
        print(f"\n## A2 kernels, residuals at the {where} ({len(p)} profiles)")
        print("best by AIC:", p.idxmin(axis=1).value_counts().to_dict())
        print("current family within 2 of best:", int((fam <= 2).sum()))
        print("sq-exp within 2 of best correlated kernel:", int(((p.sqexp - kb) <= 2).sum()))
        print("sq-exp is the best correlated kernel:", int((p.sqexp == kb).sum()))
        print("correlated term needed (white > best + 2):", int((p.white - best > 2).sum()))
        strong = p[(p.white - best) > 2]
        print(
            "  of those, sq-exp best correlated kernel:",
            int((strong.sqexp == strong[["sqexp", "exp", "matern32"]].min(axis=1)).sum()),
            "of",
            len(strong),
        )
        if where == "white best fit":
            d = p.sub(best, axis=0).round(1)
            d = d.loc[d.white.sort_values(ascending=False).index]
            d.columns = [f"dAIC {c}" for c in d.columns]
            print(md(d[["dAIC white", "dAIC sqexp", "dAIC exp", "dAIC matern32"]].reset_index()))
    print("\n## A3 whiteness")
    nom = {"lb_p_lag5": "LB", "runs_p": "runs", "shapiro_p": "SW"}
    print("nominal p<0.05:", {v: int((r[c] < 0.05).sum()) for c, v in nom.items()})
    allb = [c for c in r.columns if c.startswith("boot_p_")]
    print("bootstrap p<0.05:", {c[7:]: int((r[c] < 0.05).sum()) for c in allb})
    fail_any = (r[list(CORE)] < 0.05).any(axis=1)
    fail_bonf = (r[list(CORE)] < 0.05 / len(CORE)).any(axis=1)
    corr_fail = (r[["boot_p_lb_p_lag5", "boot_p_runs_p"]] < 0.05).any(axis=1)
    dist_fail = (r[["boot_p_shapiro_p", "boot_p_max_abs_z"]] < 0.05).any(axis=1)
    print(f"pass all four core tests at 0.05: {int((~fail_any).sum())} of {len(r)}")
    print(f"pass with Bonferroni (0.05/4): {int((~fail_bonf).sum())} of {len(r)}")
    print(f"fail a correlation test (LB or runs): {int(corr_fail.sum())}")
    print(f"fail a distribution test (SW or max|z|): {int(dist_fail.sum())}")
    print("var z range:", r.var_z.min().round(3), r.var_z.max().round(3))
    print("lag1 raw > 0.3:", int((r.lag1_raw > 0.3).sum()), " |lag1 z| max:", r.lag1_z.abs().max())
    print("max |LOO z| > 4:", int((r.max_abs_loo_z > 4).sum()))
    t = r[
        [
            "run",
            "source",
            "n_points",
            "lag1_raw",
            "lag1_z",
            "var_z",
            "max_abs_z",
            "kurtosis_z",
            *CORE,
            "max_abs_loo_z",
            "max_loo_z_steps_away_l",
        ]
    ].copy()
    t["source"] = t.source.map({"round_robin": "RR", "paper": "TP26", "example": "ex"})
    t = t.rename(columns={**CORE, "max_loo_z_steps_away_l": "max LOO z: steps away (l)"})
    print(md(t.round(3)))
    print("\n## A4 near-step scatter")
    s = r.dropna(subset=["step_var_ratio_z"])
    print(
        f"{len(s)} profiles with >=3 points in each zone; whitened var near/far > 1 in "
        f"{int((s.step_var_ratio_z > 1).sum())}; raw rms ratio > 1 in "
        f"{int((s.step_rms_ratio_raw > 1).sum())}; median raw rms ratio "
        f"{s.step_rms_ratio_raw.median():.2f}, "
        f"median var ratio z {s.step_var_ratio_z.median():.2f}; "
        f"bootstrap p<0.05 in {int((s.boot_p_step_var_ratio_z < 0.05).sum())}; "
        f"Brown-Forsythe p<0.05 in {int((s.step_bf_p_z < 0.05).sum())}"
    )
    print("rim zone usable (>=3 points, not near a step):", int(r.rim_var_ratio_z.notna().sum()))
    t = s[
        [
            "run",
            "diffusion_length_um",
            "step_n_near",
            "step_n_far",
            "step_rms_ratio_raw",
            "step_var_ratio_z",
            "boot_p_step_var_ratio_z",
        ]
    ].round(3)
    print(md(t))


def part_b():
    c = pd.read_csv(HERE / "coverage.csv")
    pct = lambda v: f"{100 * v:.0f}"  # noqa: E731
    out = pd.DataFrame(
        {
            "geometry": c.geometry.str.split(" \\(").str[0],
            "case": c.case,
            "1s default": c.coverage_1s.map(pct),
            "1s indep.": c.white_coverage_1s.map(pct),
            "2s default": c.coverage_2s.map(lambda v: f"{100 * v:.1f}"),
            "2s indep.": c.white_coverage_2s.map(lambda v: f"{100 * v:.1f}"),
            "bias %": c.median_bias_pct.map(lambda v: f"{v:+.1f}"),
            "width %": c.median_width_pct.map(lambda v: f"{v:.0f}"),
            "indep. width %": c.white_median_width_pct.map(lambda v: f"{v:.0f}"),
        }
    )
    print("\n## B coverage")
    print(
        f"trials per case {int(c.trials.min())}-{int(c.trials.max())}; 68% band "
        f"{100 * c.band68_lo.iloc[0]:.1f}-{100 * c.band68_hi.iloc[0]:.1f}; 95.45% band "
        f"{100 * c.band95_lo.iloc[0]:.1f}-{100 * c.band95_hi.iloc[0]:.1f}"
    )
    print(md(out))
    cand = pd.DataFrame(
        {
            "geometry": c.geometry.str.split(" \\(").str[0],
            "case": c.case,
            "1s default": c.coverage_1s.map(pct),
            "1s kernel by AIC": c.aic_coverage_1s.map(pct),
            "1s outlier clip": c.clip_coverage_1s.map(pct),
            "2s default": c.coverage_2s.map(pct),
            "2s kernel by AIC": c.aic_coverage_2s.map(pct),
            "2s outlier clip": c.clip_coverage_2s.map(pct),
            "AIC not sq-exp %": c.share_aic_not_sqexp.map(pct),
            "clipped %": c.share_clipped.map(pct),
            "LB p<.05 %": c.share_lb5_p_lt_05.map(pct),
            "SW p<.05 %": c.share_shapiro_p_lt_05.map(pct),
            "warn %": c.edge_warning_share.map(pct),
        }
    )
    print("\n## B candidates and diagnostics")
    print(md(cand))


def part_zone():
    z = pd.read_csv(HERE / "zone_coverage.csv")
    pct = lambda v: "" if pd.isna(v) else f"{100 * v:.0f}"  # noqa: E731
    out = pd.DataFrame(
        {
            "geometry": z.geometry.str.split(" \\(").str[0],
            "case": z.case,
            "near/far raw rms": z.median_near_far_rms_raw.map(lambda v: f"{v:.2f}"),
            "1s default": z.coverage_1s.map(pct),
            "1s candidate": z.corr_cand_coverage_1s.map(pct),
            "2s default": z.coverage_2s.map(pct),
            "2s candidate": z.corr_cand_coverage_2s.map(pct),
            "width default %": z.median_width_pct.map(lambda v: f"{v:.0f}"),
            "width candidate %": z.corr_cand_median_width_pct.map(lambda v: f"{v:.0f}"),
            "dAIC<-2 %": z.share_corr_daic_lt_m2.map(pct),
            "dAIC<-6 %": z.share_corr_daic_lt_m6.map(pct),
            "1s white-near-step cand.": z.white_cand_coverage_1s.map(pct),
        }
    )
    print("\n## Candidate near-step models (500 trials per case)")
    print(md(out))
    r = pd.read_csv(HERE / "zone_real.csv")
    r["outside default 1s"] = (r.corr_time_days < r.low_days) | (r.corr_time_days > r.high_days)
    t = r[r.corr_delta_aic < -2][
        [
            "run",
            "corr_delta_aic",
            "corr_g",
            "time_days",
            "low_days",
            "high_days",
            "corr_time_days",
            "corr_low_days",
            "corr_high_days",
            "outside default 1s",
        ]
    ].round(3)
    print(
        f"\nreal: candidate preferred by >2 AIC in {len(t)} of {len(r)}, by >6 in "
        f"{int((r.corr_delta_aic < -6).sum())}; h at grid max in {int((r.corr_g == 100).sum())}"
    )
    print(md(t))


if __name__ == "__main__":
    part_a()
    if (HERE / "coverage.csv").exists():
        part_b()
    if (HERE / "zone_coverage.csv").exists():
        part_zone()
