# ruff: noqa: RUF001  # sigma, minus, en dash and times signs are wanted in figure text
"""Figures for the noise findings of 2026-10-16: fitting the initial condition, and how
long a core plateau to measure. Run from the repository root:

    uv run python docs/references/figures/noise_findings/make_figures.py            # public
    uv run python docs/references/figures/noise_findings/make_figures.py --private  # round robin

Public figures (this folder) use only synthetic profiles and published Towbin and Plank
(2026) profiles (analyses/xenolith_archive):

- fig1_ic_fit_examples.png: the default fit from the nominal initial condition (IC: the
  profile before diffusion, a set of sharp steps) and from the IC fitted by
  `forster.sampling.fit_initial_condition`; fig1_ic_parameters.csv, fig1_ic_times.csv;
- fig2_ic_scoring.png: how flat the likelihood of each IC parameter is under four ways
  of scoring a candidate IC, and how often each recovers a known IC on synthetic
  profiles; fig2_sweeps.csv, fig2_sweep_widths.csv, fig2_synthetic_recovery.csv;
- fig3_plateau_synthetic.png: bias and coverage of the time against the length of
  plateau measured, synthetic profiles; fig3_plateau_synthetic.csv;
- fig4_plateau_real.png: real profiles with core-side points dropped; fig4_plateau_real.csv;
- fig5_short_plateau.png: why a short plateau moves the time (FORSTER ends the model at
  the last measured point); fig5_short_plateau.csv.

The analyses are in findings.py beside this file. Their results are kept in the CSVs
above and reused; ``--recompute`` reruns them (the two Monte Carlo studies take about
25 min (IC recovery) and 5 min (plateau, 2 worker processes) at 200 draws).

``--private`` makes fig1 and fig4 for the round robin profile SH63olv066-1 (another
lab's unpublished data). They are written only to
``Diffusion Round Robin/outputs/noise_findings/``, inside the private submodule.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import replace
from pathlib import Path

import findings as F
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forster.model import Profile

HERE = F.HERE
OKABE = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
C_NOM = OKABE[5]  # blue: nominal IC / default noise model
C_FIT = OKABE[3]  # green: fitted IC
C_NEAR = OKABE[6]  # vermillion: near-step noise model
C_CUT = OKABE[7]  # reddish purple: model ends at the last kept point
C_ZONE = "#F6EFA6"  # pale yellow: within 2 diffusion lengths of a step
mpl.rcParams.update(
    {
        "font.size": 7.5,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "0.92",
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "lines.linewidth": 1.1,
        "savefig.dpi": 200,
        "pdf.fonttype": 42,
    }
)

PUBLIC_IC = ["AZ18 WHT06_ol43_prof 1", "Xenocryst_ol6_prof_1", "AZ18 WHT06_ol48_prof 1"]
PRIVATE_IC = ["SH63olv066-1"]
PUBLIC_PLATEAU = ["AZ18 WHT06_ol43_prof 1", "AZ18 WHT06_ol48_prof 1",
                  "AZ18 WHT01_Ol_192_prof_1", "AZ18 WHT06_ol_overgrowth2_prof_1"]  # fmt: skip
PRIVATE_PLATEAU = ["SH63olv066-1"]
SWEEP_PROFILE = "AZ18 WHT06_ol43_prof 1"
SHORT_PROFILE = "AZ18 WHT06_ol43_prof 1"
SHORT_CUT_L = 2  # cut at step + 2 l (last kept point 1.6 l past the step)
# x range of the zoom on the region where the IC moved (um)
ZOOM = {"AZ18 WHT06_ol43_prof 1": (0, 26), "Xenocryst_ol6_prof_1": (0, 17),
        "AZ18 WHT06_ol48_prof 1": (8, 38), "SH63olv066-1": (0, 30)}  # fmt: skip
PARAM_LABEL = {
    "step1_um": "step 1 position",
    "step2_um": "step 2 position",
    "edge_fo": "edge (rim) Fo",
    "core_fo": "core Fo",
    "step_fo": "plateau Fo (between steps)",
}
SHORT_PARAM = {"step_fo": "plateau Fo"}  # fmt: skip


def short(name: str) -> str:
    return name.replace("AZ18 ", "")


def pct(v):
    return 100 * np.asarray(v)


def fmt_signed(v: float, f: str = "+.1f") -> str:
    return format(v, f).replace("-", "−")


def cached(path: Path, recompute: bool, make):
    """Read ``path`` if it exists (and no recompute), else ``make()`` and write it."""
    if path.exists() and not recompute:
        return pd.read_csv(path)
    df = make()
    df.to_csv(path, index=False, float_format="%.6g")
    print(f"wrote {path}")
    return df


def ic_with(initial, values: dict):
    """``initial`` with the IC parameter values in ``values`` (template names)."""
    steps = list(initial.steps_um)
    for i in range(len(steps)):
        steps[i] = values.get(f"step{i + 1}_um", steps[i])
    kw = {k: values[k] for k in ("edge_fo", "core_fo", "step_fo") if k in values}
    return replace(initial, steps_um=tuple(steps), **kw)


def ic_profile(initial, x_end):
    """The IC as a step function from the rim side of the domain to ``x_end``."""
    lo = initial.domain_um[0]
    s = list(initial.steps_um)
    levels = [initial.edge_fo]
    if len(s) == 2:
        levels.append(initial.step_fo)
    levels.append(initial.core_fo)
    xs, ys = [lo], [levels[0]]
    for k, sk in enumerate(s):
        xs += [sk, sk]
        ys += [levels[k], levels[k + 1]]
    xs.append(x_end)
    ys.append(levels[-1])
    return np.array(xs), np.array(ys)


# =============================================================== fig 1: IC fit examples


def ic_fit_data(names, runs, profiles, out_dir: Path, recompute: bool, tag=""):
    def make_params():
        rows, joints = [], []
        for n in names:
            print(f"joint IC fit: {n} ...", flush=True)
            p, j, _ = F.fit_ic(n, runs, profiles)
            rows += p
            joints.append(j)
        pd.DataFrame(joints).to_csv(out_dir / f"fig1_ic_joint{tag}.csv", index=False,
                                    float_format="%.6g")  # fmt: skip
        return pd.DataFrame(rows)

    params = cached(out_dir / f"fig1_ic_parameters{tag}.csv", recompute, make_params)
    joint = pd.read_csv(out_dir / f"fig1_ic_joint{tag}.csv")
    fits, times = {}, []
    for n in names:
        run = runs[n]
        prof = profiles[run.profile_name]
        vals = dict(params[params.profile == n].set_index("param").fitted)
        fitted_ic = ic_with(run.initial, vals)
        x0 = float(np.nanmin(prof.distance_um))
        f0 = float(prof.fo[np.nanargmin(prof.distance_um)])
        for label, ic in (("nominal", run.initial), ("fitted", fitted_ic)):
            d = F.default_fit(run, profiles, ic)
            d["initial"] = ic
            fits[(n, label)] = d
            times.append({"profile": n, "ic": label, "first_point_um": x0,
                          "first_point_fo": f0} | {k: v for k, v in d.items()
                          if isinstance(v, float)})  # fmt: skip
    times = pd.DataFrame(times)
    times.to_csv(out_dir / f"fig1_ic_times{tag}.csv", index=False, float_format="%.6g")
    return params, joint, fits, times


def _param_line(r) -> str:
    if r.param.endswith("_um"):
        nom, fit, lo, hi, unit = r.nominal, r.fitted, r.prior_lo, r.prior_hi, "µm"
        shift = f"{fmt_signed(fit - nom)} µm"
    else:
        nom, fit, lo, hi = pct(r.nominal), pct(r.fitted), pct(r.prior_lo), pct(r.prior_hi)
        unit = "mol%"
        shift = f"{fmt_signed(fit - nom, '+.2f')} mol%"
    edge = "  AT PRIOR EDGE" if r.at_prior_edge else ""
    name = f"{PARAM_LABEL[r.param]} ({unit})"
    prior = f"{lo:.1f} – {hi:.1f}"
    move = f"{shift} ({fmt_signed(r.shift_in_prior_sd)} σ)"
    return f"{name:<35}{nom:7.2f} → {fit:6.2f}   {prior:<15}{move:<22}{edge}"


PARAM_HEADER = f"{'IC parameter':<35}{'nominal → fitted':<19}{'prior (±2.5 σ)':<15}shift (prior σ)"


def _shade_zone(ax, d):
    for s in d["steps"]:
        ax.axvspan(s - F.ZONE * d["ell_um"], s + F.ZONE * d["ell_um"], color=C_ZONE, lw=0,
                   zorder=0)  # fmt: skip


def _draw_fit(ax, d, colour, ls, x_end):
    m = d["model_x"] <= x_end + 1
    mx = d["model_x"][m]
    ax.fill_between(mx, pct(d["band_lo"][m]), pct(d["band_hi"][m]), color=colour, alpha=0.3,
                    lw=0)  # fmt: skip
    ax.plot(mx, pct(d["best_curve"][m]), color=colour, lw=1.1)
    xi, yi = ic_profile(d["initial"], float(d["model_x"][-1]))
    ax.plot(xi, pct(yi), color=colour, lw=1.0, ls=ls)


def fig1_title(params, joint, fits, names) -> str:
    """Title from the numbers: time change of the joint fit and near-step dAIC per profile."""
    parts, daic = [], []
    for n in names:
        j = joint[joint.profile == n].iloc[0]
        d0, d1 = fits[(n, "nominal")], fits[(n, "fitted")]
        chg = 100 * (j.best_days / d0["time_d"] - 1)
        parts.append(f"{short(n).split('_prof')[0]} {fmt_signed(chg, '+.0f')}%")
        daic.append(f"{fmt_signed(d0['near_step_daic'])} → {fmt_signed(d1['near_step_daic'])}")
    return ("Fitting the initial condition (IC, the profile before diffusion) changes the "
            "time:\n" + ", ".join(parts) + ".\nNear-step ΔAIC, nominal → fitted IC: "
            + ", ".join(daic) + " (the warning fires below −6).\n"
            "Are the fitted ICs physically plausible?")  # fmt: skip


def fig1(names, params, joint, fits, path: Path, title: str):
    rows = len(names)
    fig = plt.figure(figsize=(7.2, 3.65 * rows + 0.8), layout="constrained")
    subs = np.atleast_1d(fig.subfigures(rows, 1))
    for sub, n in zip(subs, names, strict=True):
        d0, d1 = fits[(n, "nominal")], fits[(n, "fitted")]
        j = joint[joint.profile == n].iloc[0]
        gs = sub.add_gridspec(3, 2, height_ratios=[1.6, 0.8, 1.2], width_ratios=[1.3, 1.0])
        ax_p = sub.add_subplot(gs[0, 0])
        ax_z = sub.add_subplot(gs[0, 1])
        ax_r0 = sub.add_subplot(gs[1, 0])
        ax_r1 = sub.add_subplot(gs[1, 1], sharey=ax_r0)
        ax_t = sub.add_subplot(gs[2, :])
        ax_t.axis("off")
        x_end = max(float(d0["x"].max()), float(d0["model_end_um"])) + 2
        for ax, xlim in ((ax_p, (min(0, d0["x"].min()), x_end)), (ax_z, ZOOM[n])):
            for d, c, ls in ((d0, C_NOM, (0, (3, 1.5))), (d1, C_FIT, (0, (1.2, 1.0)))):
                _draw_fit(ax, d, c, ls, x_end)
            ax.plot(d0["x"], pct(d0["y"]), "o", ms=2.6, mfc="white", mec="black", mew=0.6,
                    zorder=5)  # fmt: skip
            ax.set_xlim(*xlim)
            ax.set_xlabel("Distance from rim (µm)", labelpad=1)
        # zoom: y range from the data and the IC levels inside it
        zx = (d0["x"] >= ZOOM[n][0]) & (d0["x"] <= ZOOM[n][1])
        levels = []
        for d in (d0, d1):
            xi, yi = ic_profile(d["initial"], x_end)
            levels += list(yi[(xi >= ZOOM[n][0] - 1) & (xi <= ZOOM[n][1])])
        ylo = min(pct(d0["y"][zx]).min(), pct(min(levels))) - 0.4
        yhi = max(pct(d0["y"][zx]).max(), pct(max(levels))) + 0.4
        ax_z.set_ylim(ylo, yhi)
        ax_p.axvspan(*ZOOM[n], color="0.9", lw=0, zorder=0)
        ax_p.set_ylabel("Fo (mol%)")
        ax_p.set_title(short(n), fontsize=7.5)
        ax_z.set_title("Zoom on the region where the IC moved", fontweight="normal",
                       fontsize=7)  # fmt: skip
        for ax, d, c, lab in ((ax_r0, d0, C_NOM, "nominal IC"), (ax_r1, d1, C_FIT, "fitted IC")):
            _shade_zone(ax, d)
            ax.axhline(0, color="0.5", lw=0.5)
            ax.plot(d["x"], pct(d["resid"]), "o", ms=2.2, color=c, zorder=5)
            ax.set_title(f"Residuals, {lab} (near-step ΔAIC {fmt_signed(d['near_step_daic'])})",
                         fontsize=6.5, fontweight="normal", color=c)  # fmt: skip
            ax.set_xlim(min(0, d0["x"].min()), x_end)
            ax.set_xlabel("Distance from rim (µm)")
        ax_r0.set_ylabel("Residual\n(mol% Fo)")
        ax_p.set_ylabel("Fo (mol%)")
        rmax = 1.25 * max(np.abs(pct(d0["resid"])).max(), np.abs(pct(d1["resid"])).max())
        ax_r0.set_ylim(-rmax, rmax)

        def warn(v):
            return " (warns: < −6)" if v < F.NEAR_STEP_WARN else ""

        chg = 100 * (j.best_days / d0["time_d"] - 1)
        x0 = float(np.min(d0["x"]))
        f0 = float(pct(d0["y"][np.argmin(d0["x"])]))
        lines = [
            f"Nominal IC: {d0['time_d']:.3g} d, 1σ {d0['lo1_d']:.3g}–{d0['hi1_d']:.3g}, "
            f"2σ {d0['lo2_d']:.3g}–{d0['hi2_d']:.3g} d; near-step ΔAIC "
            f"{fmt_signed(d0['near_step_daic'])}{warn(d0['near_step_daic'])}",
            f"Fitted IC, held fixed: {d1['time_d']:.3g} d, 1σ {d1['lo1_d']:.3g}–"
            f"{d1['hi1_d']:.3g} d; near-step ΔAIC {fmt_signed(d1['near_step_daic'])}"
            f"{warn(d1['near_step_daic'])}",
            f"Joint fit (IC uncertainty included): best {j.best_days:.3g} d "
            f"({fmt_signed(chg, '+.0f')}%), 1σ {j.low_days:.3g}–{j.high_days:.3g} d; "
            f"first data point at {x0:.1f} µm, Fo {f0:.2f} mol%",
        ]
        p = params[params.profile == n]
        table = [PARAM_HEADER] + [_param_line(r) for r in p.itertuples()]
        ax_t.text(0.0, 1.0, "\n".join([*lines, "", *table]), va="top", ha="left",
                  fontsize=6.0, family="monospace", linespacing=1.25,
                  transform=ax_t.transAxes)  # fmt: skip
    handles = [
        mpl.lines.Line2D([], [], ls="none", marker="o", ms=3, mfc="white", mec="black",
                         label="data"),
        mpl.lines.Line2D([], [], color=C_NOM, ls=(0, (3, 1.5)), label="nominal IC (t = 0)"),
        mpl.lines.Line2D([], [], color=C_FIT, ls=(0, (1.2, 1.0)), label="fitted IC (t = 0)"),
        mpl.lines.Line2D([], [], color=C_NOM, lw=4, alpha=0.5,
                         label="best fit from nominal IC, 1σ band"),
        mpl.lines.Line2D([], [], color=C_FIT, lw=4, alpha=0.5,
                         label="best fit from fitted IC, 1σ band"),
        mpl.patches.Patch(color=C_ZONE, label="within 2 diffusion lengths of a step"),
        mpl.patches.Patch(color="0.9", label="zoomed region"),
    ]  # fmt: skip
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=6.3,
               handlelength=2.2, columnspacing=1.2)  # fmt: skip
    fig.suptitle(title, fontsize=8, fontweight="bold", x=0.01, ha="left")
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


# ================================================================= fig 2: IC scoring

SCORE_STYLE = {  # key: (label, colour, linestyle)
    "white": ("(a) white noise only", "0.55", "-"),
    "fixedC": ("(b) noise fitted once, at the nominal IC", OKABE[1], (0, (4, 1.5))),
    "refitC": ("(c) noise refitted at the best (b) IC: FORSTER", OKABE[5], "-"),
    "profiled": ("(d) noise refitted for every candidate IC", OKABE[6], (0, (1.2, 1.0))),
}
NOISE_LABEL = {"white": "white noise", "white + sqexp L=3": "+ smooth misfit, L 3 µm",
               "white + sqexp L=10": "+ smooth misfit, L 10 µm"}  # fmt: skip
SCEN_LABEL = {"step off by l/2": "Step position, nominal 5 µm off",
              "edge Fo off 0.5 mol%": "Edge Fo, nominal 0.5 mol% off",
              "control: nominal correct (step)": "Control: nominal step correct"}  # fmt: skip


def fig2(sw: pd.DataFrame, widths: pd.DataFrame, rec: pd.DataFrame, fitted: dict, path):
    params = list(dict.fromkeys(sw.param))
    fig = plt.figure(figsize=(7.2, 6.4), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.0, 1.25])
    axs = top.subplots(1, len(params), sharey=True)
    for ax, p in zip(axs, params, strict=True):
        g = sw[sw.param == p].sort_values("value")
        scale = 1.0 if p.endswith("_um") else 100.0
        v = g.value * scale
        for k, (_, col, ls) in SCORE_STYLE.items():
            ax.plot(v, g[k] - g[k].max(), color=col, ls=ls, lw=1.1)
        ax.axhline(-0.5, color="0.4", lw=0.6, ls=":")
        ax.axvline(g.nominal.iloc[0] * scale, color="k", lw=0.7, ls=(0, (1, 1.5)))
        if p in fitted:
            ax.axvline(fitted[p] * scale, color=C_FIT, lw=1.0)
        ax.set_ylim(-6, 0.5)
        unit = "µm" if p.endswith("_um") else "mol%"
        ax.set_xlabel(f"{SHORT_PARAM.get(p, PARAM_LABEL[p])} ({unit})")
        w = widths.set_index("param").loc[p]
        ax.set_title(f"(b) ÷ (a)  {w.fixedC_half_width / w.white_half_width:.2f}\n"
                     f"(b) ÷ (c)  {w.fixedC_half_width / w.refitC_half_width:.2f}\n"
                     f"(c) ÷ (d)  {w.refitC_half_width / w.profiled_half_width:.2f}",
                     fontsize=6.2, fontweight="normal")  # fmt: skip
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    axs[0].set_ylabel("log likelihood − its maximum\n(time fitted at each value)")
    handles = [mpl.lines.Line2D([], [], color=c, ls=ls, label=lab)
               for lab, c, ls in SCORE_STYLE.values()]  # fmt: skip
    handles += [
        mpl.lines.Line2D([], [], color="k", lw=0.7, ls=(0, (1, 1.5)), label="nominal value"),
        mpl.lines.Line2D([], [], color=C_FIT, label="joint fit (fig. 1)"),
        mpl.lines.Line2D([], [], color="0.4", lw=0.6, ls=":", label="−0.5: 1σ interval"),
    ]
    top.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=6.3)
    top.suptitle(f"(A) {short(SWEEP_PROFILE)}: one IC parameter varied at a time, the "
                 "others at nominal; width = 1σ half-width ratio",
                 fontsize=7.5, fontweight="bold", x=0.01, ha="left")  # fmt: skip

    ax_c, ax_w = bottom.subplots(1, 2, sharey=True, width_ratios=[1.0, 1.0])
    rec = rec.copy()
    rec["label"] = rec.scenario.map(SCEN_LABEL) + "\n" + rec.noise.map(NOISE_LABEL)
    labels = list(dict.fromkeys(rec.label))
    y = np.arange(len(labels))[::-1]
    off = {k: (i - 1.5) * 0.17 for i, k in enumerate(SCORE_STYLE)}
    markers = {"white": "o", "fixedC": "s", "refitC": "^", "profiled": "D"}
    for k, (lab, col, _) in SCORE_STYLE.items():
        g = rec[rec.scoring == k].set_index("label").loc[labels]
        ax_c.plot(100 * g.coverage_1s, y + off[k], markers[k], color=col, ms=3.5, label=lab)
        ax_w.plot(g.median_half_width / g.rmse, y + off[k], markers[k], color=col, ms=3.5)
    ax_c.axvline(68.3, color="k", lw=0.7, ls="--")
    ax_c.set_xlim(0, 103)
    ax_c.set_xlabel("1σ interval covers the true value (% of draws)\n(dashed: 68.3%, honest)")
    ax_w.axvline(1, color="k", lw=0.7, ls="--")
    ax_w.set_xscale("log")
    ax_w.set_xticks([0.2, 0.5, 1, 2, 5])
    ax_w.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%g"))
    ax_w.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax_w.set_xlabel("1σ half-width ÷ actual error (RMSE)\n(1: honest; < 1: too narrow)")
    ax_c.set_yticks(y, labels, fontsize=6.2)
    for ax in (ax_c, ax_w):
        for b in (2.5, 5.5):
            ax.axhline(b, color="0.7", lw=0.6)
        ax.grid(axis="y", visible=False)
    n = int(rec.trials.min())
    bottom.suptitle(f"(B) Synthetic profiles: recover a known IC value starting from a wrong "
                    f"nominal ({n} noise draws per row)", fontsize=7.5, fontweight="bold",
                    x=0.01, ha="left")  # fmt: skip
    fig.suptitle("Fitting the noise once at the nominal IC widens the IC likelihood; "
                 "refitting it at the best IC,\nas FORSTER does, gives intervals close to "
                 "refitting it for every candidate (d)",
                 fontsize=8.5, fontweight="bold", x=0.01, ha="left")  # fmt: skip
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


# ============================================================ fig 3: plateau synthetic

FIG3_CASES = {
    "white": "White noise only\n(sd 0.1 mol%)",
    "near-step 0.4": "Misfit near the step\n(peak 0.4 mol%)",
    "near-step 1.2": "Strong misfit near\nthe step (1.2 mol%)",
    "core +0.2": "Core Fo 0.2 mol%\nabove the model's",
    "plateau slope": "Plateau slope (+0.1\nmol% per 100 µm)",
}
FIG3_SERIES = [
    # (domain, model, label, colour, linestyle, marker)
    ("fixed", "white", "white-only noise", OKABE[0], "-", "o"),
    ("fixed", "default", "default noise model (whitened)", C_NOM, "-", "s"),
    ("fixed", "near-step", "near-step noise model", C_NEAR, "-", "^"),
    ("fixed", "default free core", "default, core Fo also fitted", C_FIT, "--", "D"),
    ("to last point", "default", "default, model ends at the last point", C_CUT, ":", "x"),
]


def fig3(s: pd.DataFrame, path: Path):
    metrics = [("bias_pct", "Median error of best\nfit (% of true time)"),
               ("cov1", "1σ range covers the\ntrue time (% of draws)"),
               ("width1_pct", "Median 1σ width\n(% of true time)")]  # fmt: skip
    cases = list(FIG3_CASES)
    fig, axes = plt.subplots(len(metrics), len(cases), figsize=(7.2, 5.6),
                             layout="constrained", sharex=True, sharey="row")  # fmt: skip
    for j, case in enumerate(cases):
        for i, (m, ylab) in enumerate(metrics):
            ax = axes[i, j]
            for dom, model, label, c, ls, mk in FIG3_SERIES:
                d = s[(s["case"] == case) & (s.domain == dom) & (s.model == model)]
                d = d.sort_values("plateau_l")
                ax.plot(d.plateau_l, d[m], color=c, ls=ls, marker=mk, ms=3, lw=1.0, label=label)
            ref = {"cov1": 68.27, "bias_pct": 0.0}.get(m)
            if ref is not None:
                ax.axhline(ref, color="0.5", lw=0.6, ls="--")
            ax.set_xscale("log")
            ax.set_xticks([3, 5, 10, 20], ["3", "5", "10", "20"])
            ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
            if j == 0:
                ax.set_ylabel(ylab)
            if i == 0:
                ax.set_title(FIG3_CASES[case], fontsize=6.8)
            if i == len(metrics) - 1:
                ax.set_xlabel("Plateau measured\npast the step (l)")
    axes[1, 0].set_ylim(0, 100)
    axes[2, 0].set_yscale("log")
    axes[2, 0].set_yticks([10, 20, 50, 100, 200], ["10", "20", "50", "100", "200"])
    axes[2, 0].yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncols=3, fontsize=6.3)
    n = int(s[s["case"].isin(cases)].trials.min())
    cov = {}
    for c in ("near-step 0.4", "near-step 1.2"):
        ns = s[(s["case"] == c) & (s.domain == "fixed") & (s.model == "default")]
        cov[c] = [float(ns[ns.plateau_l == v].cov1.iloc[0]) for v in (3, 20)]
    a, b = cov["near-step 0.4"], cov["near-step 1.2"]
    fig.suptitle("Plateau length leaves the best-fit time alone. With misfit near the step, a "
                 "longer plateau makes\nthe default 1σ range too narrow (coverage "
                 f"{a[0]:.0f}% → {a[1]:.0f}% and {b[0]:.0f}% → {b[1]:.0f}% from 3 to 20 l); the "
                 "near-step model holds.\nA wrong core Fo biases the time at every length unless "
                 "core Fo is fitted (which fails on a sloping plateau).\nl: diffusion length; "
                 f"{n} noise draws per point", fontsize=7.8, fontweight="bold", x=0.01,
                 ha="left")  # fmt: skip
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


# ================================================================ fig 4: plateau real

FIG4_SERIES = [
    # (domain, model, label, colour, marker, offset)
    ("fixed domain", "default", "default noise model, model domain unchanged", C_NOM, "s", -1),
    ("fixed domain", "near-step", "near-step noise model, model domain unchanged", C_NEAR,
     "^", 0),
    ("as used", "default", "default, as FORSTER runs it (model ends at the last point)",
     C_CUT, "D", 1),
]  # fmt: skip
MOVE = 0.2  # mark an "as used" time this far (fraction) from the full profile's


def fig4(df: pd.DataFrame, names, path: Path, title: str):
    ncol = 2
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.2, 2.75 * nrow + 0.7), layout="constrained")
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(names) :]:
        ax.axis("off")
    for ax, name in zip(axes, names, strict=False):
        d = df[df.profile == name]
        fixed = d[(d.domain == "fixed domain") & (d.model == "default")]
        full = fixed[fixed.cut_l.astype(str) == "all"].iloc[0]
        for dom, model, label, c, mk, off in FIG4_SERIES:
            s = d[(d.domain == dom) & (d.model == model)].sort_values("kept_past_step_l")
            x = s.kept_past_step_l.to_numpy() * 10 ** (0.018 * off)
            t = s.time_d.to_numpy()
            ax.errorbar(x, t, yerr=[t - s.lo2_d, s.hi2_d - t], fmt="none", ecolor=c, lw=0.6,
                        alpha=0.7)  # fmt: skip
            ax.errorbar(x, t, yerr=[t - s.lo1_d, s.hi1_d - t], fmt=mk, color=c, ms=3.2,
                        lw=1.8, label=label, capsize=0)  # fmt: skip
        ax.axhline(full.time_d, color="0.6", lw=0.6, ls="--", zorder=0)
        ax.set_xscale("log")
        ticks = [2, 3, 5, 10, 20]
        ax.set_xticks(ticks, [str(v) for v in ticks])
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.set_xlim(1.3, max(5.0, d.kept_past_step_l.max() * 1.5))
        lo = d.lo2_d.min() * 0.85
        hi_all = d.hi2_d.max()
        hi = min(hi_all, 3 * full.hi2_d) * 1.35
        ax.set_yscale("log")
        ax.set_ylim(lo, hi)
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.tick_params(axis="y", which="minor", labelsize=6)
        used = d[(d.domain == "as used") & (d.model == "default")]
        # near-step dAIC (as used) above each cut
        prev, low = -np.inf, False
        for r in used.sort_values("kept_past_step_l").itertuples():
            v = r.near_step_daic
            lx = np.log10(r.kept_past_step_l)
            low = (lx - prev < 0.1) and not low  # stagger labels that would touch
            prev = lx
            ax.annotate(fmt_signed(v), (r.kept_past_step_l, 1.0),
                        xycoords=("data", "axes fraction"),
                        xytext=(0, -9 if low else -1), textcoords="offset points",
                        ha="center", va="top",
                        fontsize=6, color=C_NEAR if v < F.NEAR_STEP_WARN else "0.35",
                        fontweight="bold" if v < F.NEAR_STEP_WARN else "normal")  # fmt: skip
        for r in used.itertuples():
            ch = r.time_d / full.time_d - 1
            if abs(ch) >= MOVE:
                xy = (r.kept_past_step_l * 10**0.018, r.time_d)
                ax.annotate(f"{fmt_signed(100 * ch, '+.0f')}%\nmodel ends {r.model_end_um:.0f} µm",
                            xy, xytext=(30, -34 if ch < 0 else 10), textcoords="offset points",
                            fontsize=6, color=C_CUT, ha="left", va="center",
                            arrowprops=dict(arrowstyle="-", lw=0.5, color=C_CUT))  # fmt: skip
            if r.hi2_d > hi:
                ax.annotate(f"2σ to {r.hi2_d:.0f} d", (r.kept_past_step_l * 10 ** 0.018, hi),
                            xytext=(8, -22), textcoords="offset points", fontsize=6,
                            color=C_CUT, ha="left",
                            arrowprops=dict(arrowstyle="-", lw=0.5, color=C_CUT))  # fmt: skip
        label = short(name)
        if name in F.NOMINAL_DAYS and name.startswith("SH63"):
            label += " (round robin, PRIVATE)"
        dom_end = used[used.cut_l.astype(str) == "all"].model_end_um.iloc[0]
        ax.set_title(f"{label}\n{int(full.n)} points, l = {full.ell_full_um:.1f} µm, "
                     f"model domain ends at {dom_end:.0f} µm", fontsize=6.8, pad=10)  # fmt: skip
        ax.set_xlabel("Plateau kept past the last step (diffusion lengths l)")
        ax.set_ylabel("Fitted time (days)")
    h, lab = axes[0].get_legend_handles_labels()
    extra = [mpl.lines.Line2D([], [], color="0.4", lw=1.8, label="thick bar: 1σ range"),
             mpl.lines.Line2D([], [], color="0.4", lw=0.6, label="thin bar: 2σ range")]  # fmt: skip
    fig.legend(h + extra, lab + [e.get_label() for e in extra], loc="outside lower center",
               ncols=2, fontsize=6.3)  # fmt: skip
    fig.suptitle(title, fontsize=8, fontweight="bold", x=0.01, ha="left")
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


# ================================================================ fig 5: short plateau


def short_plateau_data(runs, profiles):
    run = runs[SHORT_PROFILE]
    prof = profiles[run.profile_name]
    full = F.default_fit(run, profiles, run.initial)
    full["initial"] = run.initial
    cut = max(run.initial.steps_um) + SHORT_CUT_L * full["ell_um"]
    pk = prof.distance_um <= cut
    cut_prof = {run.profile_name: Profile(prof.distance_um[pk], prof.fo[pk], prof.name)}
    part = F.default_fit(run, cut_prof, run.initial)
    part["initial"] = run.initial
    dropped = (prof.distance_um[~pk], prof.fo[~pk])
    return full, part, dropped, cut


def fig5(full, part, dropped, cut, path: Path):
    fig, (ax_p, ax_z) = plt.subplots(1, 2, figsize=(7.2, 3.3), layout="constrained",
                                     width_ratios=[1.35, 1.0])  # fmt: skip
    step = max(full["steps"])
    for ax in (ax_p, ax_z):
        for d, c in ((full, C_NOM), (part, C_CUT)):
            m = d["model_x"]
            ax.fill_between(m, pct(d["band_lo"]), pct(d["band_hi"]), color=c, alpha=0.3, lw=0)
            ax.plot(m, pct(d["best_curve"]), color=c, lw=1.1)
            ax.plot(m[-1], pct(d["best_curve"][-1]), "|", color=c, ms=9, mew=1.6)
            xi, yi = ic_profile(d["initial"], float(m[-1]))
            ax.plot(xi, pct(yi), color=c, lw=0.9, ls=(0, (3, 1.5)))
        both = full["x"] <= cut
        ax.plot(full["x"][both], pct(full["y"][both]), "o", ms=2.8, mfc="white", mec="black",
                mew=0.6, zorder=5)  # fmt: skip
        ax.plot(full["x"][~both], pct(full["y"][~both]), "o", ms=2.8, mfc="0.55", mec="0.3",
                mew=0.6, zorder=5)  # fmt: skip
        out = dropped[0] > full["model_end_um"]
        ax.plot(dropped[0][out], pct(dropped[1][out]), "o", ms=2.8, mfc="white", mec="0.7",
                mew=0.6, zorder=4)  # fmt: skip
        ax.axvline(cut, color="0.5", lw=0.6, ls=":")
        ax.set_xlabel("Distance from rim (µm)")
    ax_p.set_ylabel("Fo (mol%)")
    ax_p.axvspan(25, 60, color="0.93", lw=0, zorder=0)
    ax_z.set_xlim(25, 60)
    zx = (full["x"] >= 25) & (full["x"] <= 60)
    ax_z.set_ylim(pct(full["y"][zx]).min() - 0.6, pct(full["y"][zx]).max() + 0.4)
    ell = full["ell_um"]
    ax_z.annotate(f"model ends at the last kept point\n({part['model_end_um']:.1f} µm, "
                  f"{(part['model_end_um'] - step) / ell:.1f} l past the step);\n"
                  "core Fo is held there", (part["model_end_um"], pct(part["best_curve"][-1])),
                  xytext=(18, -95), textcoords="offset points", fontsize=6.2, color=C_CUT,
                  arrowprops=dict(arrowstyle="-", lw=0.5, color=C_CUT))  # fmt: skip
    ax_p.annotate(f"full profile: the model\ndomain ends at {full['model_end_um']:.0f} µm",
                  (full["model_end_um"], pct(full["best_curve"][-1])), xytext=(6, -28),
                  textcoords="offset points", fontsize=6.2, color=C_NOM,
                  arrowprops=dict(arrowstyle="-", lw=0.5, color=C_NOM))  # fmt: skip
    ax_z.text(step + 0.4, ax_z.get_ylim()[0] + 0.1, "step 2", fontsize=6, color="0.4")
    ax_z.axvline(step, color="0.6", lw=0.5, ls=(0, (1, 1.5)))
    ax_p.set_title(short(SHORT_PROFILE) + ": all points against the profile cut at "
                   f"step + {SHORT_CUT_L} l", fontsize=7)  # fmt: skip
    ax_z.set_title("Zoom on the core side of step 2", fontsize=7, fontweight="normal")
    chg = 100 * (part["time_d"] / full["time_d"] - 1)
    handles = [
        mpl.lines.Line2D([], [], ls="none", marker="o", ms=3, mfc="white", mec="black",
                         label="data used by both fits"),
        mpl.lines.Line2D([], [], ls="none", marker="o", ms=3, mfc="0.55", mec="0.3",
                         label="data dropped by the cut"),
        mpl.lines.Line2D([], [], ls="none", marker="o", ms=3, mfc="white", mec="0.7",
                         label="data past the model domain (used by neither)"),
        mpl.lines.Line2D([], [], color=C_NOM, lw=4, alpha=0.5,
                         label=f"all points: {full['time_d']:.2f} d (1σ {full['lo1_d']:.2f}–"
                               f"{full['hi1_d']:.2f})"),
        mpl.lines.Line2D([], [], color=C_CUT, lw=4, alpha=0.5,
                         label=f"cut: {part['time_d']:.2f} d (1σ {part['lo1_d']:.2f}–"
                               f"{part['hi1_d']:.2f}), {fmt_signed(chg, '+.0f')}%"),
        mpl.lines.Line2D([], [], color="0.4", lw=0.9, ls=(0, (3, 1.5)),
                         label="initial condition (t = 0), to the model's end"),
        mpl.lines.Line2D([], [], ls="none", marker="|", ms=8, mew=1.6, color="0.4",
                         label="end of the model (core Fo held fixed)"),
        mpl.lines.Line2D([], [], color="0.5", lw=0.6, ls=":", label="cut position"),
    ]  # fmt: skip
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=6.3)
    fig.suptitle("A plateau measured less than ~2 diffusion lengths past the step moves the "
                 "time: FORSTER ends the model at\nthe last measured point and holds core Fo "
                 f"there, which steepens the modelled step ({fmt_signed(chg, '+.0f')}% here)",
                 fontsize=8, fontweight="bold", x=0.01, ha="left")  # fmt: skip
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")
    return pd.DataFrame([
        {"fit": lab, "n_points": len(d["x"]), "last_point_um": float(d["x"].max()),
         "model_end_um": d["model_end_um"], "time_d": d["time_d"], "lo1_d": d["lo1_d"],
         "hi1_d": d["hi1_d"], "lo2_d": d["lo2_d"], "hi2_d": d["hi2_d"],
         "near_step_daic": d["near_step_daic"], "ell_um": d["ell_um"]}
        for lab, d in (("all points", full), (f"cut at step + {SHORT_CUT_L} l", part))
    ])  # fmt: skip


# ============================================================================== main


def plateau_real(names, runs, profiles, path, recompute):
    def make():
        rows = []
        for n in names:
            print(f"plateau cuts: {n} ...", flush=True)
            rows += F.plateau_cuts(runs[n], profiles)
        return pd.DataFrame(rows)

    return cached(path, recompute, make)


def public(recompute: set, trials: int):
    runs, profiles = F.load_paper()
    rc = lambda k: "all" in recompute or k in recompute  # noqa: E731

    params, joint, fits, _ = ic_fit_data(PUBLIC_IC, runs, profiles, HERE, rc("ic_fit"))
    fig1(
        PUBLIC_IC,
        params,
        joint,
        fits,
        HERE / "fig1_ic_fit_examples.png",
        fig1_title(params, joint, fits, PUBLIC_IC),
    )

    sw = cached(HERE / "fig2_sweeps.csv", rc("sweeps"),
                lambda: F.sweep(F.make_case(SWEEP_PROFILE, runs, profiles)))  # fmt: skip
    widths = F.sweep_widths(sw)
    widths.to_csv(HERE / "fig2_sweep_widths.csv", index=False, float_format="%.6g")
    recov = cached(HERE / "fig2_synthetic_recovery.csv", rc("ic_recovery"),
                   lambda: F.ic_recovery_summary(F.ic_recovery(trials)))  # fmt: skip
    fitted = dict(params[params.profile == SWEEP_PROFILE].set_index("param").fitted)
    fig2(sw, widths, recov, fitted, HERE / "fig2_ic_scoring.png")

    ps = cached(HERE / "fig3_plateau_synthetic.csv", rc("plateau_synthetic"),
                lambda: F.plateau_summary(F.plateau_synthetic(trials)))  # fmt: skip
    fig3(ps, HERE / "fig3_plateau_synthetic.png")

    real = plateau_real(PUBLIC_PLATEAU, runs, profiles, HERE / "fig4_plateau_real.csv",
                        rc("plateau_real"))  # fmt: skip
    fig4(real, PUBLIC_PLATEAU, HERE / "fig4_plateau_real.png",
         "Dropping core-side plateau points barely moves the time while the model domain is "
         "unchanged;\nFORSTER's own fit of a cut profile moves it once the model ends within "
         "~2 diffusion lengths (l) of the step.\nNumbers on top: near-step ΔAIC of each cut "
         "(bold: the warning fires, below −6)")  # fmt: skip

    full, part, dropped, cut = short_plateau_data(runs, profiles)
    fig5(full, part, dropped, cut, HERE / "fig5_short_plateau.png").to_csv(
        HERE / "fig5_short_plateau.csv", index=False, float_format="%.6g")  # fmt: skip


def private(recompute: set):
    out = F.PRIVATE_OUT
    out.mkdir(parents=True, exist_ok=True)
    rr_runs, rr_profiles = F.load_round_robin()
    runs, profiles = F.load_paper()
    rc = lambda k: "all" in recompute or k in recompute  # noqa: E731
    params, joint, fits, _ = ic_fit_data(PRIVATE_IC, rr_runs, rr_profiles, out,
                                         rc("ic_fit"), tag="_rr")  # fmt: skip
    fig1(PRIVATE_IC, params, joint, fits, out / "fig1_ic_fit_examples_rr.png",
         "Round robin SH63olv066-1 (PRIVATE): fitting the initial condition (IC)")  # fmt: skip
    rr = plateau_real(PRIVATE_PLATEAU, rr_runs, rr_profiles, out / "fig4_plateau_real_rr.csv",
                      rc("plateau_real"))  # fmt: skip
    pub = plateau_real(PUBLIC_PLATEAU, runs, profiles, HERE / "fig4_plateau_real.csv", False)
    fig4(pd.concat([pub, rr]), PUBLIC_PLATEAU + PRIVATE_PLATEAU,
         out / "fig4_plateau_real_rr.png",
         "Plateau truncation, paper profiles and round robin SH63olv066-1 (PRIVATE).\nNumbers "
         "on top: near-step ΔAIC of each cut (bold: warns, below −6)")  # fmt: skip


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--private",
        action="store_true",
        help="round robin figures, written only inside the private submodule",
    )
    ap.add_argument(
        "--recompute",
        nargs="*",
        default=[],
        help="rerun analyses instead of reading their CSVs: all, ic_fit, sweeps, "
        "ic_recovery, plateau_synthetic, plateau_real",
    )
    ap.add_argument("--trials", type=int, default=200, help="Monte Carlo draws (recompute)")
    args = ap.parse_args()  # fmt: skip
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if args.private:
            private(set(args.recompute))
        else:
            public(set(args.recompute), args.trials)
