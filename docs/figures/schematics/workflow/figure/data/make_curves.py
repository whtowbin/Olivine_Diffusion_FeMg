"""Illustrative curves for the FORSTER workflow schematic (not FORSTER output).

Constant-D diffusion of a one-step starting profile on a half line, with the rim (x = 0)
held at the rim segment's Fo: u = Fo - Fo_a, u(0, t) = 0, u(x, 0) = dFo * H(x - x0), so by
the method of images
    Fo(x, t) = Fo_a + dFo/2 * [erfc((x0 - x) / 2s) - erfc((x0 + x) / 2s)],  s = sqrt(D t).
Units are arbitrary (D = 1); the figure shows no numbers. Synthetic "measured" points are
the model at t_true plus white noise and a smooth correlated misfit (squared-exponential
kernel). Misfit is the generalised chi-squared with that covariance, as FORSTER's
correlated noise model does; the 1-sigma range is where it rises by 1 above its minimum.
Run: uv run --with numpy python figure/data/make_curves.py  (from the project folder)
"""

import json
import math
from pathlib import Path

import numpy as np

erfc = np.vectorize(math.erfc)
OUT = Path(__file__).with_name("curves.json")
rng = np.random.default_rng(7)

FO_A, FO_B, X0 = 0.80, 0.88, 20.0  # rim segment, core segment, step position (illustrative)
XMAX = 60.0
T_TRUE = 14.0
WHITE, CORR, CORR_L = 0.004, 0.003, 8.0


def model(x, t, x0=X0, fo_a=FO_A, fo_b=FO_B):
    x = np.asarray(x, float)
    if t <= 0:
        return np.where(x < x0, fo_a, fo_b)
    s = 2.0 * math.sqrt(t)
    return fo_a + (fo_b - fo_a) / 2.0 * (erfc((x0 - x) / s) - erfc((x0 + x) / s))


xs = np.linspace(0, XMAX, 241)
spots = np.arange(1.5, 34.0, 2.5)  # evenly spaced microprobe spots
n = len(spots)
K = CORR**2 * np.exp(-0.5 * ((spots[:, None] - spots[None, :]) / CORR_L) ** 2)
C = K + WHITE**2 * np.eye(n)
Cinv = np.linalg.inv(C)
smooth_true = rng.multivariate_normal(np.zeros(n), K)
data = model(spots, T_TRUE) + smooth_true + rng.normal(0, WHITE, n)


def chi2(t, **ic):
    r = data - model(spots, t, **ic)
    return float(r @ Cinv @ r)


times = np.linspace(0.5, 45.0, 400)
misfit = np.array([chi2(t) for t in times])
i0 = int(np.argmin(misfit))
t_best = float(times[i0])
inside = times[misfit <= misfit[i0] + 1.0]
t_lo, t_hi = float(inside.min()), float(inside.max())

# residuals at the best fit and their smooth (Gaussian-process mean) part
res = data - model(spots, t_best)
smooth_part = K @ Cinv @ res

# three model times for panel b: short, best, long
t_show = [round(0.25 * t_best, 2), round(t_best, 2), round(2.6 * t_best, 2)]
curves = {str(t): model(xs, t).round(5).tolist() for t in [0.0, *t_show]}

# uncertain starting profile: prior on step position and core Fo; posterior of time
# p(t) ∝ sum_k prior_k exp(-chi2_k(t)/2) over a grid of starting profiles
logt = np.linspace(math.log10(2.0), math.log10(80.0), 300)
tt = 10**logt
fit_alone = np.exp(-0.5 * (np.array([chi2(t) for t in tt]) - misfit[i0]))
step_sd, fo_sd = 3.0, 0.006
ic_grid = [
    (X0 + dx, FO_B + df)
    for dx in np.linspace(-2 * step_sd, 2 * step_sd, 9)
    for df in np.linspace(-2 * fo_sd, 2 * fo_sd, 9)
]
post = np.zeros_like(tt)
ref = None
chi_ic = []
for x0, fo_b in ic_grid:
    prior = math.exp(-0.5 * (((x0 - X0) / step_sd) ** 2 + ((fo_b - FO_B) / fo_sd) ** 2))
    c = np.array([chi2(t, x0=x0, fo_b=fo_b) for t in tt])
    chi_ic.append(c.min())
    ref = c.min() if ref is None else min(ref, c.min())
    post += prior * np.exp(-0.5 * (c - misfit[i0]))
# per log10 t (the grid is uniform in log t, the likelihoods are in t): multiply by t
fit_alone_log = fit_alone * tt
post_log = post * tt
# Monte Carlo over T, P, fO2 and the law: log t = log(Dt)_fit - log D, D lognormal
SIG_LOGD = 0.12  # illustrative spread in log10 D
dlog = logt[1] - logt[0]
k = np.arange(-int(4 * SIG_LOGD / dlog), int(4 * SIG_LOGD / dlog) + 1) * dlog
kern = np.exp(-0.5 * (k / SIG_LOGD) ** 2)
mc_log = np.convolve(post_log, kern, mode="same")


def norm(y):
    return (np.asarray(y) / np.max(y)).round(5).tolist()


def pct(y, q):
    c = np.cumsum(y)
    c = c / c[-1]
    return float(np.interp(q, c, logt))


# a few D draws for panel e (log10 D relative to nominal, and the time they give)
draws = rng.normal(0, SIG_LOGD, 9).round(4).tolist()

out = {
    "note": "illustrative; constant-D erfc model, synthetic data; arbitrary units",
    "params": {
        "fo_a": FO_A,
        "fo_b": FO_B,
        "x0": X0,
        "xmax": XMAX,
        "t_true": T_TRUE,
        "white": WHITE,
        "corr": CORR,
        "corr_L": CORR_L,
        "step_sd": step_sd,
        "fo_sd": fo_sd,
        "sig_logD": SIG_LOGD,
    },
    "x": xs.round(3).tolist(),
    "curves": curves,
    "t_show": t_show,
    "spots": spots.tolist(),
    "data": data.round(5).tolist(),
    "times": times.round(3).tolist(),
    "misfit": (misfit - misfit[i0]).round(4).tolist(),
    "t_best": t_best,
    "t_lo": t_lo,
    "t_hi": t_hi,
    "misfit_at_show": [round(chi2(t) - misfit[i0], 3) for t in t_show],
    "residuals": res.round(5).tolist(),
    "smooth": smooth_part.round(5).tolist(),
    "logt": logt.round(4).tolist(),
    "dist_fit": norm(fit_alone_log),
    "dist_ic": norm(post_log),
    "dist_mc": norm(mc_log),
    "pct_mc": [pct(mc_log, q) for q in (0.16, 0.5, 0.84)],
    "pct_fit": [pct(fit_alone_log, q) for q in (0.16, 0.5, 0.84)],
    "pct_ic": [pct(post_log, q) for q in (0.16, 0.5, 0.84)],
    "ic_alternatives": [
        [X0 + dx, FO_B + df]
        for dx, df in [(-3, 0.004), (2.5, -0.003), (-1.5, -0.005), (3.5, 0.002), (1, 0.005)]
    ],
    "draws_logD": draws,
}
OUT.write_text(json.dumps(out))
print(f"t_best {t_best:.2f} (1 sigma {t_lo:.2f}-{t_hi:.2f}); n={n}")
print("pct fit", out["pct_fit"], "ic", out["pct_ic"], "mc", out["pct_mc"])
print("misfit at shown times", out["misfit_at_show"])
