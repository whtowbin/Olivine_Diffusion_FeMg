"""Refits with Oeser et al. (2026) added, and an additive two-mechanism law.

The compilation and the femg_refit2026 preset are unchanged: this script adds data and
a form beside them and writes its own results. Two questions:

1. What do Oeser et al.'s 54 solid-source profiles (1100-1303 C, 1e-7 to 1e-2.3 Pa) do
   to the power-law fit (the dc07_free form, random study offsets)? Runs 7, 14 and 15
   were measured by laser linescan after the crystals broke and are convolution
   corrected; they are fitted with and without.
2. Does a sum of two mechanisms fit better than one power law in fO2?

       D = D_TaMED + D_ext
       log D_TaMED = a - E/(ln10 R T) + m (log fO2 + 7) + n (0.9 - X_Fo) + k_axis - PV term
       log D_ext   = a_x - E_x/(ln10 R T)              + n (0.9 - X_Fo) + k_axis - PV term

   D_ext has no fO2 dependence, like the PED law of Chakraborty (2010); the two share the
   composition, anisotropy and pressure terms (V fixed at 7 cm3/mol, as dc07_free). Since
   the sum is meant to span both regimes, it is fitted to all fO2, including the 31
   Dohmen et al. points below 1e-10 Pa that the power-law fits exclude.

Both are fitted by nonlinear least squares with a random offset per study (REML over a
grid of tau^2/sigma^2, as fit_models.fit_random). The machinery is checked first: on the
power law and the TaMED data it must reproduce fit_random (the preset's fit).

Run: uv run python analyses/femg_refit/fit_extended.py
Writes results/extended/{fits,predictions,study_effects}.csv and figures/extended_fo2.png.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import fit_models as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fit_ols import load
from scipy import linalg, optimize, stats

HERE = Path(__file__).parent
OUT = HERE / "results" / "extended"
FIG = HERE / "figures" / "extended_fo2.png"
LN10, R, V_DC07 = math.log(10), 8.3145, 7e-6
LINESCAN_RUNS = (7, 14, 15)  # Oeser et al.: crystals broke; linescans, convolution corrected
MELT = ("spandler_oneill2010",)
LAMBDAS = np.concatenate([[0.0], np.logspace(-3, 2, 41)])  # tau^2 / sigma^2
PED_BELOW = -10.0  # log10 fO2 (Pa): Dohmen et al.'s PED points


@dataclass(frozen=True)
class Inputs:
    """Predictors, one value per observation."""

    log_fo2: np.ndarray
    x_fo: np.ndarray
    T: np.ndarray
    P: np.ndarray
    i100: np.ndarray
    i010: np.ndarray
    melt: np.ndarray

    @classmethod
    def of(cls, df: pd.DataFrame) -> Inputs:
        return cls(
            df["log10_fO2_Pa"].to_numpy(float),
            df["X_Fo"].to_numpy(float),
            df["T_K"].to_numpy(float),
            df["P_Pa"].to_numpy(float),
            (df["axis"] == "[100]").to_numpy(float),
            (df["axis"] == "[010]").to_numpy(float),
            df["source"].isin(MELT).to_numpy(float) if "source" in df else np.zeros(len(df)),
        )

    def at(self, **kw) -> Inputs:
        return Inputs(
            **{**self.__dict__, **{k: np.broadcast_to(v, self.T.shape) for k, v in kw.items()}}
        )


def _shared(p: dict, v: Inputs) -> np.ndarray:
    """Composition, anisotropy, pressure (log10 units), shared by both mechanisms."""
    return (
        p["n"] * (0.9 - v.x_fo)
        + p["k100"] * v.i100
        + p["k010"] * v.i010
        - (v.P - 1e5) * V_DC07 / (LN10 * R * v.T)
    )


def _tamed(p: dict, v: Inputs) -> np.ndarray:
    return p["a"] - p["E"] / (LN10 * R * v.T) + p["m"] * (v.log_fo2 + 7) + _shared(p, v)


@dataclass(frozen=True)
class Model:
    name: str
    params: tuple[str, ...]
    start: tuple[float, ...]
    scale: tuple[float, ...]

    def log10_d(self, theta: np.ndarray, v: Inputs) -> np.ndarray:
        p = dict(zip(self.params, theta, strict=True))
        out = _tamed(p, v)
        if "a_x" in p:
            ext = p["a_x"] - p["E_x"] / (LN10 * R * v.T) + _shared(p, v)
            out = np.logaddexp(out * LN10, ext * LN10) / LN10
        return out + p.get("melt", 0.0) * v.melt


POWER = Model(
    "power",
    ("a", "E", "m", "n", "k100", "k010"),
    (-8.9, 2.1e5, 0.22, 3.1, -0.8, -0.7),
    (1.0, 1e4, 0.05, 0.3, 0.1, 0.1),
)
ADDITIVE = Model(
    "additive",
    (*POWER.params, "a_x", "E_x"),
    (*POWER.start, -8.9, 2.2e5),  # D_ext starts at Chakraborty (2010)'s PED law
    (*POWER.scale, 1.0, 1e4),
)


def with_melt(model: Model) -> Model:
    return Model(
        model.name + "+melt", (*model.params, "melt"), (*model.start, 0.5), (*model.scale, 0.1)
    )


def compilation(all_fo2: bool) -> pd.DataFrame:
    """The refit compilation as fit_models.tamed(), optionally without its fO2 cut."""
    if not all_fo2:
        return F.tamed()
    df = load()
    df["run_id"] = df["source"] + ":" + df["run"].astype(str)
    df.loc[df["P_Pa"] < 2e5, "P_Pa"] = 1e5
    df["ln_D"] = df["log10_D_m2s"] * LN10
    return df


def oeser(drop_linescan: bool) -> pd.DataFrame:
    o = pd.read_csv(HERE / "data" / "oeser2026.csv")
    if drop_linescan:
        o = o[~o["run"].isin(LINESCAN_RUNS)]
    return pd.DataFrame(
        {
            "source": "oeser2026",
            "run": o["run"],
            "T_K": o["T_C"] + 273.15,
            "P_Pa": 1e5,
            "log10_fO2_Pa": o["log10_fO2_Pa"],
            "X_Fo": o["X_Fo"],
            "axis": o["axis"],
            "log10_D_m2s": o["log10_D_FeMg"],
            "ln_D": o["log10_D_FeMg"] * LN10,
            "run_id": "oeser2026:" + o["run"].astype(str),
        }
    )


def _fit_at(model: Model, v: Inputs, y, L, theta0):
    """GLS fit at one tau^2/sigma^2 (L: Cholesky factor of I + lam Z Z')."""

    def resid(theta):
        return linalg.solve_triangular(L, y - model.log10_d(theta, v), lower=True)

    sol = optimize.least_squares(resid, theta0, x_scale=np.array(model.scale), method="trf")
    n, p = len(y), len(theta0)
    s2 = float(sol.fun @ sol.fun) / (n - p)
    info = sol.jac.T @ sol.jac
    loglik = -0.5 * (
        (n - p) * math.log(s2) + 2 * np.log(np.diag(L)).sum() + np.linalg.slogdet(info)[1]
    )
    return loglik, sol.x, s2, s2 * np.linalg.pinv(info)


def fit(model: Model, df: pd.DataFrame) -> dict:
    """Random study offsets by REML on a grid, the nonlinear analogue of fit_random."""
    v = Inputs.of(df)
    y = df["log10_D_m2s"].to_numpy(float)
    Z = pd.get_dummies(df["source"]).to_numpy(float)
    theta = np.array(model.start, float)
    fits = []
    for lam in LAMBDAS:
        L = np.linalg.cholesky(np.eye(len(y)) + lam * Z @ Z.T)
        res = _fit_at(model, v, y, L, theta)
        theta = res[1]  # warm start the next lambda
        fits.append(res)
    loglik = np.array([f[0] for f in fits])
    tau = np.sqrt(LAMBDAS * np.array([f[2] for f in fits]))
    k = int(loglik.argmax())
    inside = tau[loglik >= loglik[k] - stats.chi2.ppf(0.95, 1) / 2]
    _, beta, s2, cov = fits[k]
    Vm = np.eye(len(y)) + LAMBDAS[k] * Z @ Z.T
    resid = y - model.log10_d(beta, v)
    effects = LAMBDAS[k] * Z.T @ np.linalg.solve(Vm, resid)
    return {
        "model": model,
        "beta": beta,
        "cov": cov,
        "sd": math.sqrt(s2),
        "tau": float(tau[k]),
        "tau_range": (float(inside.min()), float(inside.max())),
        "effects": dict(zip(pd.get_dummies(df["source"]).columns, effects, strict=True)),
        "resid": resid,
        "n": len(y),
        "lam_at_edge": k in (0, len(LAMBDAS) - 1),
    }


def check_control() -> None:
    """The nonlinear machinery on the power law and TaMED data must equal fit_random."""
    df = F.tamed()
    mine = fit(POWER, df)
    ref = F.fit_random(F.FORMS["dc07_free"], df)
    b = ref["beta"]
    expect = {"a": b[0], "E": b[1], "m": b[2], "n": b[3], "tau": ref["tau"] / LN10}
    got = dict(zip(POWER.params, mine["beta"], strict=True)) | {"tau": mine["tau"]}
    print("control (power law, TaMED) against fit_random:")
    for k, e in expect.items():
        print(f"  {k:4s} {got[k]:12.5g}  fit_random {e:12.5g}")
    # The lambda grids differ (41 here, 400 in fit_random), so allow the grid step.
    assert abs(got["tau"] - expect["tau"]) < 0.02, "tau does not reproduce fit_random"
    assert abs(got["m"] - expect["m"]) < 0.005, "fO2 exponent does not reproduce fit_random"
    assert abs(got["E"] - expect["E"]) < 2000, "E does not reproduce fit_random"


def _grad(model: Model, beta: np.ndarray, v: Inputs) -> np.ndarray:
    h = np.array(model.scale) * 1e-5
    return np.array(
        [
            (
                model.log10_d(beta + h[j] * np.eye(len(beta))[j], v)[0]
                - model.log10_d(beta - h[j] * np.eye(len(beta))[j], v)[0]
            )
            / (2 * h[j])
            for j in range(len(beta))
        ]
    )


def _local(model: Model, beta, v: Inputs) -> tuple[float, float]:
    """d log D / d log fO2 and the effective activation energy (kJ/mol) at v."""
    up, dn = v.at(log_fo2=v.log_fo2 + 0.01), v.at(log_fo2=v.log_fo2 - 0.01)
    m = float((model.log10_d(beta, up) - model.log10_d(beta, dn))[0] / 0.02)
    t1, t2 = v.at(T=v.T * 1.001), v.at(T=v.T / 1.001)
    slope = (model.log10_d(beta, t1) - model.log10_d(beta, t2))[0] / (1 / t1.T[0] - 1 / t2.T[0])
    return m, float(-LN10 * R * slope / 1e3)


def _inputs(log_fo2: float, x_fo: float, t_c: float, p_pa: float) -> Inputs:
    a = lambda x: np.array([x], float)  # noqa: E731
    return Inputs(a(log_fo2), a(x_fo), a(t_c + 273.15), a(p_pa), a(0), a(0), a(0))


def predictions(name: str, f: dict) -> list[dict]:
    model, beta, cov = f["model"], f["beta"], f["cov"]
    rows = []
    conds = {k.split(",")[0]: c for k, c in F._conditions().items()}
    for label, c in conds.items():
        v = _inputs(float(c.log_fo2_pa[0]), float(c.x_fo[0]), float(c.T[0]) - 273.15, float(c.P[0]))
        g = _grad(model, beta, v)
        se = math.sqrt(max(g @ cov @ g, 0.0))
        m, e = _local(model, beta, v)
        rows.append(
            {
                "fit": name,
                "conditions": label,
                "vs_dc07": float(model.log10_d(beta, v)[0]) - F._dc07_preset(c),
                "se_law": se,
                "se_new_lab": math.hypot(se, f["tau"]),
                "local_fO2_exponent": m,
                "local_E_kJ_mol": e,
            }
        )
    # Local fO2 exponents where Oeser et al. measured steps (1150 C, [001], Fo91.5).
    for log_fo2 in (-6.0, -4.35):
        m, _ = _local(model, beta, _inputs(log_fo2, 0.915, 1150.0, 1e5))
        rows.append({"fit": name, "conditions": f"1150 C, 1e{log_fo2} Pa", "local_fO2_exponent": m})
    return rows


def _rms(f: dict, df: pd.DataFrame, mask) -> float:
    r = f["resid"][np.asarray(mask)]
    return float(np.sqrt(np.mean(r**2))) if len(r) else float("nan")


def summary_row(name: str, f: dict, df: pd.DataFrame) -> dict:
    p = dict(zip(f["model"].params, f["beta"], strict=True))
    se = dict(zip(f["model"].params, np.sqrt(np.diag(f["cov"])), strict=True))
    row = {"fit": name, "n_points": f["n"], "tau": f["tau"], "tau_lo95": f["tau_range"][0]}
    row |= {
        "tau_hi95": f["tau_range"][1],
        "sd_within": f["sd"],
        "lambda_at_grid_edge": f["lam_at_edge"],
    }
    for k in f["model"].params:
        row[k] = p[k] / 1e3 if k.startswith("E") else p[k]
        row[f"{k}_se"] = se[k] / 1e3 if k.startswith("E") else se[k]
    fo2 = df["log10_fO2_Pa"].to_numpy()
    row["rms_tamed_points"] = _rms(
        f, df, (fo2 > PED_BELOW) & (df["source"] != "oeser2026").to_numpy()
    )
    row["rms_ped_points"] = _rms(f, df, fo2 <= PED_BELOW)
    row["rms_oeser"] = _rms(f, df, (df["source"] == "oeser2026").to_numpy())
    return row


def figure(fits: dict, data: pd.DataFrame) -> None:
    """log D[001] against fO2 at 1150 and 1250 C: the power law and the sum, with data."""
    panels = (1150.0, 1250.0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained", sharey=True)
    fo2 = np.linspace(-12, -1, 111)
    curves = {
        "power law + Oeser (no linescans)": ("power + oeser_dp", "#E69F00", "-"),
        "sum of two mechanisms + Oeser (no linescans)": ("additive + oeser_dp", "#0072B2", "-"),
    }
    for ax, t_c in zip(axes, panels, strict=True):
        n = len(fo2)
        v = Inputs(
            fo2, np.full(n, 0.915), np.full(n, t_c + 273.15), np.full(n, 1e5), *(np.zeros(n),) * 3
        )
        for label, (key, colour, ls) in curves.items():
            f = fits[key]
            ax.plot(fo2, f["model"].log10_d(f["beta"], v), color=colour, ls=ls, lw=1.8, label=label)
        dc07 = [F.preset_log10_d("dohmen_chakraborty2007", t_c, 1e5, g, 0.915) for g in fo2]
        ax.plot(fo2, dc07, color="0.2", ls=":", lw=1.1, label="DC07 (default)")
        # Points within 15 C of the panel, moved to it, [001], Fo91.5, 1 atm by the sum.
        f = fits["additive + oeser_dp"]
        near = data[(data["T_K"] - 273.15 - t_c).abs() <= 15]
        vn = Inputs.of(near)
        to = vn.at(T=t_c + 273.15, x_fo=0.915, P=1e5, i100=0.0, i010=0.0)
        moved = (
            near["log10_D_m2s"].to_numpy()
            + f["model"].log10_d(f["beta"], to)
            - f["model"].log10_d(f["beta"], vn)
        )
        for src, (colour, marker, label) in {
            **F.STUDY_STYLE,
            "oeser2026": ("#000000", "P", "Oeser et al. 2026"),
        }.items():
            for ls_run, alpha, suffix in (
                (False, 1.0, ""),
                (True, 0.35, " (linescan runs, not fitted)"),
            ):
                m = (near["source"] == src).to_numpy()
                if src == "oeser2026":
                    m &= near["run"].isin(LINESCAN_RUNS).to_numpy() == ls_run
                elif ls_run:
                    continue
                if m.any():
                    ax.plot(
                        near["log10_fO2_Pa"][m],
                        moved[m],
                        marker,
                        color=colour,
                        ms=5,
                        alpha=alpha,
                        mfc=colour if src == "oeser2026" and not ls_run else "none",
                        ls="",
                        label=label + suffix,
                    )
        ax.axvspan(-3.75, -2.1, color="0.9", lw=0, zorder=0)
        ax.set_title(f"{t_c:.0f} C, [001], Fo91.5, 1 atm (points within 15 C, moved)", loc="left")
        ax.set_xlabel("log10 fO2 (Pa)")
    axes[0].set_ylabel("log10 D[001] (m2/s)")
    seen: dict[str, object] = {}  # label -> handle, each label once across both panels
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
            seen.setdefault(label, handle)
    fig.legend(
        list(seen.values()),
        list(seen.keys()),
        loc="outside lower center",
        ncol=3,
        fontsize=7,
        frameon=False,
    )
    # The message: where the two laws differ. Largest gap between them at 1e-7 to 1e-2 Pa.
    a, s = fits["additive + oeser_dp"], fits["power + oeser_dp"]
    mid = np.linspace(-7, -2, 51)
    gap = max(
        float(np.abs(a["model"].log10_d(a["beta"], w) - s["model"].log10_d(s["beta"], w)).max())
        for t_c in panels
        for w in [Inputs(mid, *(np.full(51, x) for x in (0.915, t_c + 273.15, 1e5, 0, 0, 0)))]
    )
    fig.suptitle(
        f"The second mechanism bends the law only below about 1e-10 Pa:\n"
        f"from 1e-7 to 1e-2 Pa the two laws differ by at most {gap:.2f} log "
        "(shaded: round robin and paper fO2)",
        fontsize=10,
    )
    fig.savefig(FIG, dpi=200)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    check_control()
    tamed, every = compilation(False), compilation(True)
    sets = {
        "power + tamed": (POWER, tamed),
        "power + oeser_all": (POWER, pd.concat([tamed, oeser(False)], ignore_index=True)),
        "power + oeser_dp": (POWER, pd.concat([tamed, oeser(True)], ignore_index=True)),
        "power+melt + oeser_dp": (
            with_melt(POWER),
            pd.concat([tamed, oeser(True)], ignore_index=True),
        ),
        "additive + all_fo2": (ADDITIVE, every),
        "additive + oeser_dp": (ADDITIVE, pd.concat([every, oeser(True)], ignore_index=True)),
        "additive+melt + oeser_dp": (
            with_melt(ADDITIVE),
            pd.concat([every, oeser(True)], ignore_index=True),
        ),
    }
    fits, rows, preds, effects = {}, [], [], []
    for name, (model, df) in sets.items():
        f = fit(model, df)
        fits[name] = f
        rows.append(summary_row(name, f, df))
        preds += predictions(name, f)
        effects += [{"fit": name, "study": k, "effect": e} for k, e in f["effects"].items()]
    pd.DataFrame(rows).to_csv(OUT / "fits.csv", index=False)
    pd.DataFrame(preds).to_csv(OUT / "predictions.csv", index=False)
    pd.DataFrame(effects).to_csv(OUT / "study_effects.csv", index=False)
    figure(fits, pd.concat([every, oeser(False)], ignore_index=True))
    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(pd.DataFrame(preds).round(3).to_string(index=False))
    print(
        pd.DataFrame(effects)
        .pivot(index="study", columns="fit", values="effect")
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    main()
