"""Monte Carlo over temperature, pressure, oxygen fugacity and the diffusivity law.

A fitted time is a fitted D t: the profile fixes the product, and the time follows from
the diffusivity at the conditions and with the law assumed. `monte_carlo` draws the
conditions (T, P, log fO2, each Gaussian; fO2 optionally held relative to a buffer, so
it follows T and P as an fO2 buffered by the rock would) and the law's parameters (from
its covariance, `forster.diffusivity.LawUncertainty`, plus optionally the spread between
laboratories), and turns each draw into a time.

- **Rescaling** (fixed rim, no growth stages): t = t_fit D_fit / D_draw, exact for every
  parameter but the composition coefficient, which also changes the shape of D; D is
  taken a third of the way from the edge Fo to the core Fo, where a change in that
  coefficient moved the fitted time as the rescaling predicts to 0.0011 log
  (`forster.model._law_uncertainty`). No model runs.
- **Refitting** (growth stages, or a rim value that changes in time): their times are
  fixed in seconds, so the time no longer scales as 1/D
  (``docs/references/law-uncertainty-limits.md``); every draw is refitted.

Each draw's t_fit comes from the fit's own posterior (correlated method: chi2^(-n/2)
over the time steps), or from an initial-condition posterior (`forster.sampling`) when
one is given, so the result combines the fit, the initial condition, the conditions
and the law. The seed is new: the paper-era script had none, so its Monte Carlo times
moved between runs by up to about 15%.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np

from forster.diffusivity import ArrheniusDiffusivity, Diffusivity
from forster.model import SECONDS_PER_DAY, Conditions, FitResult, Profile, fit_profile
from forster.sampling import ICPosterior

DEFAULT_DRAWS = 4000
DEFAULT_REFITS = 200
PERCENTILES = (5, 16, 50, 84, 95)
SOURCES = ("fit", "T", "P", "fO2", "law", "studies")


@dataclass(frozen=True)
class ConditionUncertainty:
    """1-sigma uncertainties for `monte_carlo` (zero: held at the fit's value).

    ``fO2_buffer`` (QFM or NNO): draw fO2 relative to that buffer, at the fit's offset
    from it plus ``fO2_sd_log``, so it follows each draw's T and P. Without it, log fO2
    is drawn around the fit's value on its own. ``law``: draw the law's parameters from
    its covariance (laws without one are exact); ``between_study``: add the spread
    between laboratories (for an absolute time, `forster.model.FitResult`).
    """

    T_sd_C: float = 0.0
    P_sd_Pa: float = 0.0
    fO2_sd_log: float = 0.0
    fO2_buffer: str | None = None
    law: bool = True
    between_study: bool = False

    def __post_init__(self) -> None:
        for name in ("T_sd_C", "P_sd_Pa", "fO2_sd_log"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value >= 0):
                raise ValueError(f"{name} must be zero or more, got {value}")


@dataclass(frozen=True)
class TimeDistribution:
    """Times drawn by `monte_carlo`, with percentiles and the spread from each source."""

    method: str  # "rescale" or "refit"
    days: np.ndarray
    percentiles_days: dict[int, float]
    # sd of log10 t with only that source drawn (rescaling only; empty for "refit")
    sd_log10_by_source: dict[str, float]
    seed: int

    @property
    def median_days(self) -> float:
        return self.percentiles_days[50]

    @property
    def range_days(self) -> tuple[float, float]:
        """Central 68%."""
        return self.percentiles_days[16], self.percentiles_days[84]


def fit_time_density(result: FitResult) -> tuple[np.ndarray, np.ndarray]:
    """The fit's own posterior of time (correlated method: chi2^(-n/2) over the time
    steps, the noise scale integrated out): days and unnormalised density."""
    if result.uncertainty != "correlated" or result.fit_fo is None:
        raise ValueError("the posterior of time needs a fit made with the correlated method")
    chi2 = np.asarray(result.sum_chi2, dtype=float)
    ok = np.isfinite(chi2) & (chi2 > 0)
    log_post = np.full(len(chi2), -np.inf)
    log_post[ok] = -0.5 * len(result.fit_fo) * np.log(chi2[ok])
    days = np.arange(len(chi2)) * result.dt_s / SECONDS_PER_DAY
    return days, np.exp(log_post - log_post.max())


def _fit_time_draws(
    result: FitResult, posterior: ICPosterior | None, n: int, rng: np.random.Generator
) -> np.ndarray:
    """t_fit in days from the initial-condition posterior, else the fit's own."""
    if posterior is not None:
        days, density = posterior.days, posterior.density
    else:
        days, density = fit_time_density(result)
    cdf = np.cumsum(density)
    cdf /= cdf[-1]
    return np.interp(rng.uniform(size=n), cdf, days)


def _perturbed(law: ArrheniusDiffusivity, delta: Mapping[str, float]) -> ArrheniusDiffusivity:
    """``law`` with its parameters shifted by ``delta`` (names from LAW_PARAMETERS).
    log10_D0_001 and the activation energy move every axis (the anisotropy terms are
    separate parameters); the anisotropy terms move [100] or [010] alone."""
    d0 = np.array(law.log10_D0_m2_s, dtype=float) + delta.get("log10_D0_001", 0.0)
    d0[0] += delta.get("log10_100_vs_001", 0.0)
    d0[1] += delta.get("log10_010_vs_001", 0.0)
    energy = np.array(law.activation_energy_J_mol) + delta.get("activation_energy_J_mol", 0.0)
    scalars = {
        name: getattr(law, name) + delta[name]
        for name in (
            "fO2_exponent",
            "fo_coefficient",
            "activation_volume_m3_mol",
            "pressure_coefficient_per_Pa",
        )
        if name in delta
    }
    # A plain ArrheniusDiffusivity: the engine shortcut of the default preset is exact
    # only at its own values.
    fields = {f: getattr(law, f) for f in law.__dataclass_fields__}
    fields.update(log10_D0_m2_s=tuple(d0), activation_energy_J_mol=tuple(energy), **scalars)
    return ArrheniusDiffusivity(**fields)


def _law_draws(
    law: Diffusivity, uncertainty: ConditionUncertainty, n: int, rng: np.random.Generator
) -> tuple[list[dict[str, float]], np.ndarray]:
    """Parameter shifts per draw (empty dicts for an exact law) and the between-study
    offset in log10 D per draw."""
    u = law.uncertainty if isinstance(law, ArrheniusDiffusivity) else None
    shifts: list[dict[str, float]] = [{} for _ in range(n)]
    studies = np.zeros(n)
    if u is None:
        return shifts, studies
    if uncertainty.law:
        draws = rng.multivariate_normal(np.zeros(len(u.parameters)), np.array(u.covariance), n)
        shifts = [dict(zip(u.parameters, map(float, row), strict=True)) for row in draws]
    if uncertainty.between_study:
        studies = rng.normal(0.0, u.between_study_log10, n)
    return shifts, studies


def _condition_draws(
    conditions: Conditions, uncertainty: ConditionUncertainty, n: int, rng: np.random.Generator
) -> list[Conditions]:
    T = conditions.T_C + uncertainty.T_sd_C * rng.standard_normal(n)
    P = np.abs(conditions.P_Pa + uncertainty.P_sd_Pa * rng.standard_normal(n))
    d_log = uncertainty.fO2_sd_log * rng.standard_normal(n)
    if uncertainty.fO2_buffer is None:
        return [
            Conditions(float(t), float(p), conditions.fO2_Pa * 10 ** float(d))
            for t, p, d in zip(T, P, d_log, strict=True)
        ]
    buffer = uncertainty.fO2_buffer
    nominal = Conditions.from_buffer(conditions.T_C, conditions.P_Pa, 0.0, buffer).fO2_Pa
    offset = math.log10(conditions.fO2_Pa / nominal)
    return [
        Conditions.from_buffer(float(t), float(p), offset + float(d), buffer)
        for t, p, d in zip(T, P, d_log, strict=True)
    ]


def _log10_d(law: Diffusivity, conditions: Conditions, orientation, x_fo: float) -> float:
    D, _ = law.functions(conditions, orientation)
    return math.log10(float(D(np.array([x_fo]))[0]))


def _percentiles(days: np.ndarray) -> dict[int, float]:
    return {p: float(np.percentile(days, p)) for p in PERCENTILES}


def monte_carlo(
    result: FitResult,
    uncertainty: ConditionUncertainty,
    *,
    posterior: ICPosterior | None = None,
    n: int | None = None,
    seed: int = 0,
) -> TimeDistribution:
    """Times over the fit's posterior, the conditions and the law (module docstring).

    ``result`` is a fit made with the correlated method; ``posterior`` an
    initial-condition posterior of the same run (its time draws then replace the fit's).
    ``n``: draws (default `DEFAULT_DRAWS` rescaled, `DEFAULT_REFITS` refitted). The same
    ``seed`` gives the same draws.
    """
    inputs = result.inputs
    stages, rim = inputs["stages"], inputs["rim_fo"]
    refit = bool(stages) or not (rim is None or isinstance(rim, int | float))
    n = n or (DEFAULT_REFITS if refit else DEFAULT_DRAWS)
    rng = np.random.default_rng(seed)
    conditions, orientation, law = (
        inputs["conditions"],
        inputs["orientation"],
        inputs["diffusivity"],
    )
    t_fit = _fit_time_draws(result, posterior, n, rng)
    drawn = _condition_draws(conditions, uncertainty, n, rng)
    shifts, studies = _law_draws(law, uncertainty, n, rng)
    laws = [_perturbed(law, s) if s else law for s in shifts]
    if refit:
        return _refit(result, t_fit, drawn, laws, studies, seed)
    initial = inputs["initial"]
    x_fo = initial.edge_fo + (initial.core_fo - initial.edge_fo) / 3
    ref = _log10_d(law, conditions, orientation, x_fo)
    shift_law = np.array(
        [
            _log10_d(lw, conditions, orientation, x_fo) - ref if s else 0.0
            for lw, s in zip(laws, shifts, strict=True)
        ]
    )
    shift_all = (
        np.array([_log10_d(lw, c, orientation, x_fo) for lw, c in zip(laws, drawn, strict=True)])
        - ref
        + studies
    )
    days = t_fit * 10 ** (-shift_all)
    by_source = {
        "fit": float(np.std(np.log10(t_fit))),
        "law": float(np.std(shift_law)),
        "studies": float(np.std(studies)),
        **_sd_by_condition(conditions, uncertainty, orientation, law, x_fo, ref, n, seed),
    }
    return TimeDistribution("rescale", days, _percentiles(days), by_source, seed)


def _sd_by_condition(conditions, uncertainty, orientation, law, x_fo, ref, n, seed):
    """sd of log10 D with only T, only P or only fO2 drawn (buffer coupling kept)."""
    out = {}
    for source, keep in (("T", "T_sd_C"), ("P", "P_sd_Pa"), ("fO2", "fO2_sd_log")):
        alone = replace(
            uncertainty,
            **{
                k: (getattr(uncertainty, k) if k == keep else 0.0)
                for k in ("T_sd_C", "P_sd_Pa", "fO2_sd_log")
            },
        )
        drawn = _condition_draws(conditions, alone, n, np.random.default_rng(seed + 1))
        out[source] = float(np.std([_log10_d(law, c, orientation, x_fo) - ref for c in drawn]))
    return out


def _refit(result, t_fit, drawn, laws, studies, seed) -> TimeDistribution:
    """Refit every draw (growth stages, changing rim), scaling each by the fit's own
    relative scatter (t_fit / best) and the between-study offset."""
    inputs = result.inputs
    profile = Profile(result.data_x_um, result.data_fo, name=result.name)
    days = []
    for t, c, lw, b in zip(t_fit, drawn, laws, studies, strict=True):
        fit = fit_profile(
            profile,
            c,
            inputs["orientation"],
            inputs["initial"],
            inputs["grid"],
            None,
            lw,
            inputs["scheme"],
            stop_early=inputs["stop_early"],
            stages=inputs["stages"],
            rim_fo=inputs["rim_fo"],
        )
        days.append(fit.time_days * (t / result.time_days) * 10 ** (-b))
    days = np.array(days)
    return TimeDistribution("refit", days, _percentiles(days), {}, seed)
