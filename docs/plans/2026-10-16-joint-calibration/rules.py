"""Score nested-sampling fallback rules on the cases `calibrate.py` measured.

A Laplace run is **accurate** when its p16, p50 and p84 of the joint time are each within
a tolerance of the nested-sampling reference (the mean of its runs), and every shared
segment Fo's posterior mean is within a fraction of a reference sd of the reference's:
5% and 0.5 sd (the primary tolerance, Henry 2026-10-17: the diffusion equation alone is
uncertain by about 0.1 log units in D, 25% in time) and 2% and 0.2 sd (secondary).

Each rule sees what `forster.sampling.sample_posterior` sees: the ESS of the importance
samples at 1024 samples and, when the rule doubles, at 2048 and 4096 (same seed, so the
same samples the doubling loop draws). For each case and Laplace seed the rule either
keeps Laplace (at the sample count it stopped at) or falls back to nested sampling. A
rule is scored by:
- kept_accurate: of the runs where Laplace (at the rule's sample count) was accurate,
  how many it kept (higher is better: a fallback costs minutes);
- kept_inaccurate: of the runs where Laplace was inaccurate, how many it kept (the
  costly mistake: a wrong range reported silently).

    uv run python docs/plans/2026-10-16-joint-calibration/rules.py CACHE.json [...] \
        [--csv OUT.csv] [--rules-csv RULES.csv] [--figure OUT.png]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from forster.sampling import ideal_ess_fraction

TOLERANCES = {"5%": (0.05, 0.5), "2%": (0.02, 0.2)}  # (time, shared Fo in reference sd)
TOL_TIME, TOL_FO_SD = TOLERANCES["5%"]
N0, NMAX = 1024, 4096


def rows_frame(caches: list[Path]) -> pd.DataFrame:
    """One row per sampler run, with the case's fields (``extra_<set>.json`` beside a
    ``cache_<set>.json`` adds its rows)."""
    out = []
    for path in caches:
        extra_path = path.with_name(path.name.replace("cache_", "extra_"))
        extra = json.loads(extra_path.read_text()) if extra_path.exists() else {}
        for case, c in json.loads(path.read_text()).items():
            for r in c["rows"] + extra.get(case, []):
                if r["method"].startswith("auto->nested"):
                    r = {**r, "ess_fraction": math.nan}  # n_samples counts nested draws
                out.append(
                    {
                        "case": case,
                        "crystal": c["crystal"],
                        "setting": c["setting"],
                        "step_sd_um": c["step_sd_um"],
                        "fo_sd": c["fo_sd"],
                        "d": c["d"],
                        **r,
                    }
                )
    return pd.DataFrame(out)


def score(df: pd.DataFrame, tol: str = "5%") -> pd.DataFrame:
    """Each run's errors against the case's reference (mean of the nested seeds)."""
    parts = []
    for _, g in df.groupby("case", sort=False):
        nested = g[g.method == "nested"]
        if nested.empty:  # reference not run yet
            continue
        ref = nested.mean(numeric_only=True)
        g = g.copy()
        for p in ("p16", "p50", "p84"):
            g[f"{p}_err"] = g[f"{p}_days"] / ref[f"{p}_days"] - 1
        g["time_err"] = g[[f"{p}_err" for p in ("p16", "p50", "p84")]].abs().max(axis=1)
        labels = [c[:-5] for c in g.columns if c.endswith("_fo_mean")]
        fo_err = pd.DataFrame(
            {
                lab: (g[f"{lab}_mean"] - ref[f"{lab}_mean"]).abs() / ref[f"{lab}_sd"]
                for lab in labels
            }
        )
        g["fo_err_sd"] = fo_err.max(axis=1) if labels else 0.0
        g["fo_err_label"] = fo_err.idxmax(axis=1) if labels else ""
        tol_time, tol_fo = TOLERANCES[tol]
        g["accurate"] = (g.time_err <= tol_time) & (g.fo_err_sd <= tol_fo)
        # the reference's own noise: seed 1 against seed 0
        if len(nested) < 2:
            g["ref_noise_time"] = g["ref_noise_fo_sd"] = math.nan
        else:
            a, b = nested.iloc[0], nested.iloc[-1]
            g["ref_noise_time"] = max(
                abs(b[f"{p}_days"] / a[f"{p}_days"] - 1) for p in ("p16", "p50", "p84")
            )
            g["ref_noise_fo_sd"] = max(
                (abs(b[f"{lab}_mean"] - a[f"{lab}_mean"]) / a[f"{lab}_sd"] for lab in labels),
                default=0.0,
            )
        parts.append(g)
    return pd.concat(parts)


# ----------------------------------------------------------------------------- rules
# Each rule gets the case's number of parameters d and, per sample count n, that run's
# ESS (``ess[n]``) and, from diagnostics.py, its Pareto k and bootstrap SE (``diag[n]``);
# it returns (keep Laplace?, the sample count it stopped at).
def scaled_rule(reference_d: float, base: float = 0.25, double: bool = True):
    """The current rule with ESS_REFERENCE_DIMENSION = reference_d (math.inf: unscaled)."""

    def rule(d, ess, diag):
        frac = (
            base
            if math.isinf(reference_d)
            else base * ideal_ess_fraction(d) / ideal_ess_fraction(int(reference_d))
        )
        n = N0
        while double and frac * n <= ess[n] < base * N0 and n < NMAX:
            n *= 2
        return ess[n] >= frac * n, n

    return rule


def floor_rule(floor: float):
    """Keep Laplace once the ESS reaches ``floor``, doubling up to 4x while below it."""

    def rule(d, ess, diag):
        n = N0
        while ess[n] < floor and n < NMAX:
            n *= 2
        return ess[n] >= floor, n

    return rule


def pareto_rule(k_max: float = 0.7):
    """Keep Laplace at 1024 when the weights' Pareto k is below ``k_max`` (no doubling)."""

    def rule(d, ess, diag):
        return diag[N0]["pareto_k"] < k_max, N0

    return rule


def se_rule(se_max: float, k_max: float = math.inf):
    """Keep Laplace once the bootstrap SE of p16/p50/p84 is at most ``se_max`` (relative),
    doubling up to 4x while above it; and, if given, the Pareto k below ``k_max``."""

    def rule(d, ess, diag):
        n = N0
        while diag[n]["boot_se"] > se_max and n < NMAX:
            n *= 2
        return diag[n]["boot_se"] <= se_max and diag[n]["pareto_k"] < k_max, n

    return rule


def fixed_rule(n_keep: int):
    """Never fall back: Laplace at ``n_keep`` samples."""

    def rule(d, ess, diag):
        return True, n_keep

    return rule


def fixed_floor_rule(n_keep: int, floor: float):
    """Laplace at ``n_keep`` samples; fall back only when its ESS is below ``floor`` (a
    safety net for a posterior the bell shape misses entirely)."""

    def rule(d, ess, diag):
        return ess[n_keep] >= floor, n_keep

    return rule


def jump_rule(near_ideal: float | None = 0.4, floor: float = 20, se_max: float = math.inf):
    """Keep 1024 samples when ESS / n is at least ``near_ideal`` (None: ideal(d), the
    fraction for an exactly bell-shaped posterior); otherwise draw 4096 and keep them
    unless their ESS is below ``floor`` or their bootstrap SE above ``se_max``."""

    def rule(d, ess, diag):
        cut = ideal_ess_fraction(d) if near_ideal is None else near_ideal
        if ess[N0] >= cut * N0:
            return True, N0
        return ess[NMAX] >= floor and diag[NMAX]["boot_se"] <= se_max, NMAX

    return rule


def scaled_double_to(target_fraction: float):
    """The current threshold, but double (up to 4x) while ESS / n is below
    ``target_fraction`` rather than until the ESS reaches 256."""

    def rule(d, ess, diag):
        frac = 0.25 * ideal_ess_fraction(d) / ideal_ess_fraction(4)
        n = N0
        while ess[n] < target_fraction * n and n < NMAX:
            n *= 2
        return ess[n] >= frac * n, n

    return rule


RULES = {
    "current: 0.25 x ideal(d)/ideal(4), doubling": scaled_rule(4),
    # (doubling never triggers unscaled: it needs 0.25 n <= ESS < 0.25 x 1024 at n = 1024)
    "unscaled 0.25 at 1024": scaled_rule(math.inf, double=False),
    "ESS/n >= 0.4 at 1024 (tuned on these runs)": scaled_rule(math.inf, base=0.4, double=False),
    "ESS floor 256, doubling": floor_rule(256),
    "ESS floor 512, doubling": floor_rule(512),
    "current threshold, doubling while ESS/n < 0.4": scaled_double_to(0.4),
    "never fall back: Laplace 1024": fixed_rule(1024),
    "never fall back: Laplace 4096": fixed_rule(4096),
    "Laplace 4096, fall back if ESS < 20": fixed_floor_rule(4096, 20),
    "1024 if ESS/n >= 0.4, else 4096; fall back if ESS < 20": jump_rule(),
    "1024 if ESS/n >= ideal(d), else 4096; fall back if ESS < 20": jump_rule(None),
    "as above; also fall back if bootstrap SE at 4096 > 2.5%": jump_rule(None, se_max=0.025),
    "Pareto k < 0.7 at 1024": pareto_rule(0.7),
    "bootstrap SE <= 2%, doubling": se_rule(0.02),
    "bootstrap SE <= 0.7%, doubling": se_rule(0.007),
    "bootstrap SE <= 0.5%, doubling": se_rule(0.005),
}


def with_diagnostics(scored: pd.DataFrame, caches: list[Path]) -> pd.DataFrame:
    """Laplace rows joined with diagnostics.py's Pareto k and bootstrap SE (NaN where not
    run); its ESS must equal calibrate.py's (same draws)."""
    rows = []
    for path in caches:
        dpath = path.with_name(path.name.replace("cache_", "diagnostics_"))
        if dpath.exists():
            for case, rs in json.loads(dpath.read_text()).items():
                rows += [{"case": case, **r} for r in rs]
    if not rows:
        return scored.assign(pareto_k=math.nan, boot_se=math.nan)
    diag = pd.DataFrame(rows).rename(columns={"ess": "diag_ess"})
    diag["method"] = "laplace"
    out = scored.merge(diag, on=["case", "method", "n", "seed"], how="left")
    both = out.dropna(subset=["diag_ess", "ess"])
    bad = both[(both.diag_ess / both.ess - 1).abs() > 1e-9]
    if len(bad):
        raise RuntimeError(f"diagnostics drew different samples:\n{bad[['case', 'n', 'seed']]}")
    return out.drop(columns="diag_ess")


def evaluate(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per case and seed, each rule's decision and whether Laplace was accurate there;
    and the totals per rule."""
    lap = scored[scored.method == "laplace"]
    # the fallback as fit_crystal runs it (nested, nlive 200): its accuracy rate and time
    fallback = scored[scored.method == "nested200"].groupby("case")
    fb_accurate = fallback.accurate.mean().to_dict()
    fb_seconds = fallback.seconds.mean().to_dict()
    decisions = []
    for (case, seed), g in lap.groupby(["case", "seed"], sort=False):
        by_n = g.set_index("n")
        ess = by_n.ess.to_dict()
        diag = by_n[["pareto_k", "boot_se"]].to_dict("index")
        has_diag = not by_n[["pareto_k", "boot_se"]].isna().any().any()
        d = int(g.d.iloc[0])
        for name, rule in RULES.items():
            if not has_diag and name.split()[0] in ("Pareto", "bootstrap"):
                continue
            keep, n = rule(d, ess, diag)
            # doubling redraws every sample, so each sample count up to n is paid for
            seconds = float(by_n.seconds[by_n.index <= n].sum())
            if not keep:
                seconds += fb_seconds.get(case, math.nan)
            decisions.append(
                {
                    "case": case,
                    "seed": seed,
                    "d": d,
                    "rule": name,
                    "keep": bool(keep),
                    "n": n,
                    "ess_1024": ess[N0],
                    "accurate": bool(by_n.accurate[n]),
                    "time_err": float(by_n.time_err[n]),
                    "fo_err_sd": float(by_n.fo_err_sd[n]),
                    # chance the reported result is accurate: Laplace's, or the fallback's rate
                    "result_accurate": float(by_n.accurate[n])
                    if keep
                    else fb_accurate.get(case, math.nan),
                    "seconds": seconds,
                }
            )
    dec = pd.DataFrame(decisions)
    totals = []
    for name, g in dec.groupby("rule", sort=False):
        acc, bad = g[g.accurate], g[~g.accurate]
        totals.append(
            {
                "rule": name,
                "runs": len(g),
                "laplace_accurate": len(acc),
                "kept_when_accurate": int(acc.keep.sum()),
                "laplace_inaccurate": len(bad),
                "kept_when_inaccurate": int(bad.keep.sum()),
                "result_accurate": float(g.result_accurate.mean()),
                # cases where the seeds disagree on keep or fall back: an unstable rule
                "cases_mixed": int(
                    sum(0 < k.keep.sum() < len(k) for _, k in g.groupby("case", sort=False))
                ),
                "mean_seconds": float(g.seconds.mean()),
                "worst_kept_time_err": float(g[g.keep].time_err.max())
                if g.keep.any()
                else math.nan,
                "worst_kept_fo_err_sd": float(g[g.keep].fo_err_sd.max())
                if g.keep.any()
                else math.nan,
            }
        )
    return dec, pd.DataFrame(totals)


def markdown(df: pd.DataFrame) -> str:
    """A pipe table (pandas' to_markdown needs tabulate)."""
    rows = [list(map(str, df.columns))] + [[str(v) for v in r] for r in df.itertuples(index=False)]
    lines = ["| " + " | ".join(r) + " |" for r in rows]
    lines.insert(1, "|" + "---|" * len(df.columns))
    return "\n".join(lines)


CASE_COLUMNS = {
    "case": "case",
    "d": "d",
    "auto": "fit_crystal auto (error)",
    "ess": "ESS/n at 1024: median (range)",
    "acc": "Laplace accurate within 5%, of 10: 1024 / 2048 / 4096",
    "acc2": "within 2%",
    "err": "largest Laplace error at 1024: % (% of 68% width)",
    "fb": "fallback, nested nlive 200: within 5% / within 2% / runs (largest error %)",
    "ref": "reference p16 / p50 / p84 (days)",
    "refrun": "reference runs (nlive); own noise %",
    "sec": "wall time (s): Laplace 1024 / fallback / reference",
}


def case_table(scored: pd.DataFrame, scored2: pd.DataFrame) -> pd.DataFrame:
    """One row per case, for the README (``scored2``: scored at the 2% tolerance)."""
    out = []
    for case, g in scored.groupby("case", sort=False):
        g2 = scored2[scored2.case == case]
        lap2, fb2 = g2[g2.method == "laplace"], g2[g2.method == "nested200"]
        lap, ref = g[g.method == "laplace"], g[g.method == "nested"]
        fb, auto = g[g.method == "nested200"], g[g.method.str.startswith("auto")].iloc[0]
        r = ref.mean(numeric_only=True)
        width = r.p84_days - r.p16_days
        l1 = lap[lap.n == N0]
        f = l1.ess_fraction
        noise = g.ref_noise_time.iloc[0]
        rel = 100 * (l1.time_err * l1.p50_days).max() / width
        out.append(
            {
                "case": case.replace(" | ", ", "),
                "d": int(g.d.iloc[0]),
                "auto": f"{auto.method[6:]} ({100 * auto.time_err:.1f}%)",
                "ess": f"{f.median():.2f} ({f.min():.2f}-{f.max():.2f})",
                "acc": " / ".join(
                    str(int(lap[lap.n == n].accurate.sum())) for n in (1024, 2048, 4096)
                ),
                "acc2": " / ".join(
                    str(int(lap2[lap2.n == n].accurate.sum())) for n in (1024, 2048, 4096)
                ),
                "err": f"{100 * l1.time_err.max():.1f} ({rel:.0f})",
                "fb": f"{int(fb.accurate.sum())} / {int(fb2.accurate.sum())} / {len(fb)} "
                f"({100 * fb.time_err.max():.1f})"
                if len(fb)
                else "not run",
                "ref": f"{r.p16_days:.3g} / {r.p50_days:.3g} / {r.p84_days:.3g}",
                "refrun": f"{len(ref)} ({int(ref.nlive.iloc[0])}); "
                + ("-" if math.isnan(noise) else f"{100 * noise:.1f}"),
                "sec": f"{l1.seconds.mean():.0f} / {fb.seconds.mean():.0f} / "
                f"{ref.seconds.mean():.0f}",
            }
        )
    return pd.DataFrame(out).rename(columns=CASE_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("caches", nargs="+", type=Path)
    parser.add_argument("--csv", type=Path, help="write every sampler run, scored")
    parser.add_argument("--rules-csv", type=Path, help="write the rule totals")
    parser.add_argument("--decisions-csv", type=Path, help="write each rule decision")
    parser.add_argument("--markdown", action="store_true", help="print the README tables")
    args = parser.parse_args()
    frame = rows_frame(args.caches)
    scored = with_diagnostics(score(frame), args.caches)
    dec, totals = evaluate(scored)
    if args.markdown:
        scored2 = with_diagnostics(score(frame, "2%"), args.caches)
        print(markdown(case_table(scored, scored2)))
        print()
        totals2 = evaluate(scored2)[1].set_index("rule")
        t = totals.copy()

        def pct(v):  # "-" where no fallback was run (its accuracy and time unknown)
            return "-" if math.isnan(v) else f"{100 * v:.0f}"

        def of(a, b):
            return [f"{x} of {y}" for x, y in zip(a, b, strict=True)]

        t2 = totals2.loc[t.rule]
        t["kept when accurate (5%)"] = of(t.kept_when_accurate, t.laplace_accurate)
        t["kept when inaccurate (5%)"] = of(t.kept_when_inaccurate, t.laplace_inaccurate)
        t["result accurate, 5% (%)"] = [pct(v) for v in t.result_accurate]
        t["kept when inaccurate (2%)"] = of(t2.kept_when_inaccurate, t2.laplace_inaccurate)
        t["result accurate, 2% (%)"] = [pct(v) for v in t2.result_accurate.values]
        t["cases where seeds disagree"] = t.cases_mixed
        t["mean wall time (s)"] = [pct(v / 100) for v in t.mean_seconds]
        t["worst kept error (%)"] = [
            "-" if math.isnan(v) else f"{100 * v:.1f}" for v in t.worst_kept_time_err
        ]
        print(markdown(t[["rule", *list(t.columns[-8:])]]))
        return
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 500)
    cols = [
        "case",
        "d",
        "method",
        "n",
        "seed",
        "ess",
        "ess_fraction",
        "p16_days",
        "p50_days",
        "p84_days",
    ]
    cols += ["time_err", "fo_err_sd", "fo_err_label", "accurate", "seconds"]
    print(scored[cols].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print()
    print(
        scored.groupby("case")[["ref_noise_time", "ref_noise_fo_sd"]]
        .first()
        .to_string(float_format=lambda v: f"{v:.4f}")
    )
    print()
    print(totals.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if args.csv:
        scored.drop(columns=[]).to_csv(args.csv, index=False, float_format="%.6g")
    if args.rules_csv:
        totals.to_csv(args.rules_csv, index=False, float_format="%.6g")
    if args.decisions_csv:
        dec.to_csv(args.decisions_csv, index=False, float_format="%.6g")


if __name__ == "__main__":
    main()
