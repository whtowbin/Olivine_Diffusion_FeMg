"""Illustrative profiles and misfit curves for the joint-fit schematic.

Everything here is synthetic and dimensionless-ish ("units" for distance, "mol%" for Fo
relative to the edge value). Nothing is a measured value. The physics:

* half-space 0 < x, Fo held at the edge value E at x = 0 (edge 1, the rim in contact with melt),
  step-shaped starting profile (segments rim zone, mantle, core), exact solution of the 1D
  diffusion equation by the method of images (sums of erf terms);
* one time t for both profiles; D along each profile from
  D = D[100] cos^2(alpha) + D[010] cos^2(beta) + D[001] cos^2(gamma),
  with D[001] = 6 D[100] = 6 D[010]
  (FORSTER_FACTS.md), profile 1 at 15 deg from c, profile 2 at 15 deg from a, both in the a-c plane
  (an illustrative choice);
* per profile: its own step positions, its own D, an Fo drift offset, an edge position offset;
* misfit: -2 ln L_i = n_i ln chi2_i (noise scale integrated out, as in the joint-fit plan),
  joint = sum over the two profiles. Time and edge Fo are fitted; the other parameters are
  held at their true values. 1 sigma range where the misfit is 1 above its minimum.

Run: uv run --with numpy python figure/data/make_profiles.py   (from the project folder)
Writes figure/data/profiles.json.
"""

import json
import math
from pathlib import Path

import numpy as np

erf = np.vectorize(math.erf)

# ---- illustrative choices (not data) -------------------------------------------------------------
D_A = 1.0  # D along a [100] = along b [010] (relative)
D_C = 6.0 * D_A  # c [001] about 6x faster (FORSTER_FACTS.md)
ANG = math.radians(15.0)  # each traverse 15 deg off its nearest axis, in the a-c plane


def d_profile(gamma_from_c):
    alpha = (
        math.pi / 2 - gamma_from_c
    )  # angle to a; beta = 90 deg (b normal to the section)
    return D_A * math.cos(alpha) ** 2 + D_C * math.cos(gamma_from_c) ** 2


D1 = d_profile(ANG)  # profile 1: 15 deg from c
D2 = d_profile(math.pi / 2 - ANG)  # profile 2: 15 deg from a
T_TRUE = 16.0 / (4 * D1)  # so that sqrt(4 D1 t) = 4 units on profile 1

E = 0.0  # shared edge Fo (relative, mol%)
SEG = {
    "rim": 0.9,
    "mantle": 2.0,
    "core": 3.0,
}  # shared segment Fo by label (relative, mol%)
NOISE = 0.10  # point-to-point scatter (mol%)
SPACING = 2.0  # microprobe spot spacing (units), same on both traverses

GEOM = json.loads(
    Path(__file__).with_name("crystal.json").read_text()
)  # from make_crystal.py
PROFILES = {
    # steps (from the true rim) and spot range from the drawn crystal; drift drawn enlarged
    name: {
        "D": dvalue,
        "steps": GEOM[name]["steps_model"],
        "x_end": GEOM[name]["x_end_meas"],
        "cross_meas": GEOM[name]["cross_meas"],
        "drift": drift,
        "edge_off": GEOM[name]["edge_off"],
        "seed": seed,
    }
    for name, dvalue, drift, seed in (("p1", D1, 0.0, 70), ("p2", D2, 0.30, 170))
}


def model(x, D, t, steps, edge, levels):
    """Fo(x, t) on a half-space with Fo = edge at x = 0 and a step starting profile."""
    W = math.sqrt(4 * D * t)
    bounds = [0.0, *steps, math.inf]
    u = np.zeros_like(x, dtype=float)
    for a, b, c in zip(bounds[:-1], bounds[1:], levels, strict=True):
        direct = 0.5 * (
            erf((x - a) / W) - (erf((x - b) / W) if math.isfinite(b) else -1.0)
        )
        image = 0.5 * (
            (erf((x + b) / W) if math.isfinite(b) else 1.0) - erf((x + a) / W)
        )
        u += (c - edge) * (direct - image)
    return edge + u


def starting(x, steps, levels):
    return np.select([x < steps[0], x < steps[1]], levels[:2], levels[2])


levels = [SEG["rim"], SEG["mantle"], SEG["core"]]
out = {
    "meta": {
        "D1_over_DA": D1,
        "D2_over_DA": D2,
        "D1_over_D2": D1 / D2,
        "t_true": T_TRUE,
        "width_ratio": math.sqrt(D1 / D2),
        "edge": E,
        "segments": SEG,
        "noise": NOISE,
        "spacing": SPACING,
        "note": "illustrative, synthetic; no measured values",
    }
}

# Each profile's points sit at measured distances xm = 0, 2, 4 ... from its first spot; the true
# rim lies edge_off beyond the first spot, so the model distance is xm + edge_off.
fits = {}
for name, p in PROFILES.items():
    rng = np.random.default_rng(p["seed"])
    xm = np.arange(0.0, p["x_end"] + 1e-9, SPACING)
    xmod = xm + p["edge_off"]
    truth = model(xmod, p["D"], T_TRUE, p["steps"], E, levels)
    data = truth + p["drift"] + rng.normal(0.0, NOISE, xm.size)
    xs = np.linspace(-p["edge_off"], p["x_end"] + 2, 400)  # curve in measured distance
    curve = (
        model(xs + p["edge_off"], p["D"], T_TRUE, p["steps"], E, levels) + p["drift"]
    )
    start = starting(xs + p["edge_off"], p["steps"], levels) + p["drift"]
    out[name] = {
        **{k: v for k, v in p.items() if k != "seed"},
        "x_meas": xm.tolist(),
        "fo": data.tolist(),
        "curve_x": xs.tolist(),
        "curve_fo": curve.tolist(),
        "start_fo": start.tolist(),
        "steps_meas": [s - p["edge_off"] for s in p["steps"]],
    }
    fits[name] = (xmod, data - p["drift"], p)

# ---- misfit: time, edge Fo and segment Fo fitted; steps, drift and edge offset held at truth ------
# The model is linear in the four Fo values (edge, rim, mantle, core) at a fixed time, so they are
# solved by least squares at each time. For the joint fit the four are shared, and each profile's
# noise scale is integrated out: minimise n1 ln chi2_1 + n2 ln chi2_2 by reweighted least squares.


def design(x, p, t):
    cols = []
    for k in range(4):
        e = 1.0 if k == 0 else 0.0
        lv = [1.0 if k == j + 1 else 0.0 for j in range(3)]
        cols.append(model(x, p["D"], t, p["steps"], e, lv))
    return np.array(cols).T


def m2lnl(profs, t, fix_edge=None, iters=20):
    mats = [design(x, p, t) for x, _, p in profs]
    ds = [d for _, d, _ in profs]
    w = [1.0] * len(profs)
    chis = [1.0] * len(profs)
    for _ in range(iters):
        a = np.vstack([m * math.sqrt(wi) for m, wi in zip(mats, w, strict=True)])
        b = np.concatenate([d * math.sqrt(wi) for d, wi in zip(ds, w, strict=True)])
        if fix_edge is None:
            th = np.linalg.lstsq(a, b, rcond=None)[0]
        else:
            rest = np.linalg.lstsq(a[:, 1:], b - a[:, 0] * fix_edge, rcond=None)[0]
            th = np.r_[fix_edge, rest]
        chis = [float(((d - m @ th) ** 2).sum()) for m, d in zip(mats, ds, strict=True)]
        w = [d.size / c for d, c in zip(ds, chis, strict=True)]
    return sum(d.size * math.log(c) for d, c in zip(ds, chis, strict=True))


def interval(prof, axis):
    prof = prof - prof.min()
    ok = np.where(prof <= 1.0)[0]
    lo, hi = float(axis[ok[0]]), float(axis[ok[-1]])
    return [lo, hi, float(axis[int(prof.argmin())])]


TAU = np.linspace(0.1, 2.6, 251)  # t / t_true
EDGES = np.linspace(-2.5, 1.5, 201)  # edge Fo (relative)
CASES = {"p1": ["p1"], "p2": ["p2"], "joint": ["p1", "p2"]}


def run_case(profs, tau, edges):
    vt = np.array([m2lnl(profs, t * T_TRUE) for t in tau])
    ve = np.array([min(m2lnl(profs, t * T_TRUE, e) for t in tau[::5]) for e in edges])
    return vt, ve


misfit = {"tau": TAU.tolist()}
for k, names in CASES.items():
    vt, ve = run_case([fits[n] for n in names], TAU, EDGES)
    misfit[k] = {
        "vs_time": (vt - vt.min()).tolist(),
        "time_1sigma": interval(vt, TAU),
        "edge_1sigma": interval(ve, EDGES),
        "edge_hits_grid": bool((ve - ve.min() <= 1.0)[[0, -1]].any()),
    }
out["misfit"] = misfit

# ---- how typical is this seed? the same fits over 12 noise seeds (coarser grids) -----------------
tau_c, edges_c = np.linspace(0.3, 2.6, 116), np.linspace(-1.5, 1.5, 121)
ratios = []
for seed in range(12):
    profs = {}
    for name, p in PROFILES.items():
        rng = np.random.default_rng(seed * 7 + (0 if name == "p1" else 100))
        x = np.arange(0.0, p["x_end"] + 1e-9, SPACING) + p["edge_off"]
        d = model(x, p["D"], T_TRUE, p["steps"], E, levels) + rng.normal(
            0, NOISE, x.size
        )
        profs[name] = (x, d, p)
    wid = {}
    for k, names in CASES.items():
        vt, ve = run_case([profs[n] for n in names], tau_c, edges_c)
        it, ie = interval(vt, tau_c), interval(ve, edges_c)
        wid[k] = (it[1] - it[0], ie[1] - ie[0])
    best_t = min(wid["p1"][0], wid["p2"][0])
    best_e = min(wid["p1"][1], wid["p2"][1])
    ratios.append(
        {
            "seed": seed,
            "widths": wid,
            "joint_over_better_time": wid["joint"][0] / best_t,
            "joint_over_better_edge": wid["joint"][1] / best_e,
            "p2_over_p1_edge": wid["p2"][1] / wid["p1"][1],
        }
    )
out["seed_sweep"] = ratios

Path(__file__).with_name("profiles.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out["meta"], indent=1))
for k in CASES:
    m = misfit[k]
    print(
        k,
        "time 1s:",
        [round(v, 3) for v in m["time_1sigma"]],
        "edge 1s:",
        [round(v, 3) for v in m["edge_1sigma"]],
        "hits grid:",
        m["edge_hits_grid"],
    )
for key in ("joint_over_better_time", "joint_over_better_edge", "p2_over_p1_edge"):
    v = [r[key] for r in ratios]
    print(
        key,
        "median",
        round(float(np.median(v)), 2),
        "range",
        round(min(v), 2),
        round(max(v), 2),
    )
